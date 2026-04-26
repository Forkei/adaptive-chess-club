"""Match lifecycle + turn loop.

Async API because the engine step is blocking and we don't want to stall
FastAPI's event loop. All DB reads/writes happen on the calling thread —
the only thing we ship to a worker thread is the engine's `get_move`.
"""

from __future__ import annotations

import asyncio
import logging
import random
from dataclasses import dataclass
from datetime import datetime

import chess
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.agents.soul import SoulInput, run_soul
from app.agents.subconscious import SubconsciousInput, run_subconscious
from app.director import MoodState, choose_engine_config, initial_mood_for_character, smooth_mood
from app.director.director import MatchContext
from app.director.mood import apply_deltas, load_mood, save_mood
from app.engine import EngineUnavailable, available_engines, get_engine
from app.engine.base import EngineConfig, MoveResult
from app.engine.board_abstraction import board_to_english
from app.models.character import Character
from app.models.match import (
    Color,
    Match,
    MatchResult,
    MatchStatus,
    Move,
    OpponentProfile,
    Player,
)
from app.schemas.agents import SoulResponse, SurfacedMemory

logger = logging.getLogger(__name__)

START_FEN = chess.STARTING_FEN


class MatchError(RuntimeError):
    pass


class MatchNotFound(MatchError):
    pass


class IllegalMove(MatchError):
    pass


class NotYourTurn(MatchError):
    pass


class GameAlreadyOver(MatchError):
    pass


# --- Helpers ------------------------------------------------------------


def _phase_from_board(board: chess.Board) -> str:
    # Keep this in sync with board_abstraction._phase; duplicated because
    # the matcher service shouldn't depend on board_abstraction.
    if board.fullmove_number < 10:
        return "opening"
    non_pawn_material = 0
    for piece in board.piece_map().values():
        if piece.piece_type in (chess.PAWN, chess.KING):
            continue
        non_pawn_material += {chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}[
            piece.piece_type
        ]
    return "endgame" if non_pawn_material <= 16 else "middlegame"


def _board_from_match(match: Match) -> chess.Board:
    return chess.Board(match.current_fen)


def board_with_history(match: Match, session: Session | None = None) -> chess.Board:
    """Rebuild a board by replaying every Move from `initial_fen`.

    Patch Pass 2 Item 3: `chess.Board(fen)` drops the move stack, so
    `board.outcome(claim_draw=True)` can never detect 3-fold repetition.
    Call this when you need the termination check to honour the
    repetition rule (every time we ask "did the game just end?").

    Pass `session` to fetch Move rows fresh — `match.moves` from a
    relationship can be stale mid-flush (after session.add but before
    commit, the collection may not reflect the newest row).
    """
    board = chess.Board(match.initial_fen or chess.STARTING_FEN)
    if session is not None:
        rows = list(
            session.execute(
                select(Move).where(Move.match_id == match.id).order_by(Move.move_number.asc())
            ).scalars()
        )
    else:
        rows = sorted(match.moves or [], key=lambda m: m.move_number)
    for mv in rows:
        try:
            board.push(chess.Move.from_uci(mv.uci))
        except Exception:
            return chess.Board(match.current_fen)
    return board


def _char_color(match: Match) -> chess.Color:
    return chess.BLACK if match.player_color == Color.WHITE else chess.WHITE


def _finalize_if_over(match: Match, board: chess.Board) -> None:
    # Patch Pass 2 Item 3: callers now pass a board built from move-history
    # replay, so `board.outcome(claim_draw=True)` and the mandatory-draw
    # checks actually work. 5-fold and 75-move are mandatory and fire even
    # without a claim.
    if board.is_fivefold_repetition() or board.is_seventyfive_moves():
        match.status = MatchStatus.COMPLETED
        match.ended_at = datetime.utcnow()
        match.character_elo_at_end = match.character_elo_at_start
        match.result = MatchResult.DRAW
        return
    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        return
    match.status = MatchStatus.COMPLETED
    match.ended_at = datetime.utcnow()
    match.character_elo_at_end = match.character_elo_at_start
    if outcome.winner is None:
        match.result = MatchResult.DRAW
    elif outcome.winner == chess.WHITE:
        match.result = MatchResult.WHITE_WIN
    else:
        match.result = MatchResult.BLACK_WIN


def _load_or_init_mood(session: Session, match: Match) -> tuple[MoodState, MoodState]:
    """Return (raw, smoothed) mood. Initialized from character if missing."""
    smoothed = load_mood(match.id, smoothed=True)
    raw = load_mood(match.id, smoothed=False)
    if smoothed is None or raw is None:
        character = session.get(Character, match.character_id)
        if character is None:
            raise MatchError(f"Character {match.character_id} missing")
        # Phase 4.3 — pull tone drift from the character's evolution state
        # (if any). Characters coming off a loss streak start the match
        # with a slightly elevated tilt baseline; win streak raises
        # confidence. `tone_bias_for(None)` is a zero-bias no-op, so the
        # no-evolution-state case stays byte-identical to pre-4.3.
        from app.models.evolution import CharacterEvolutionState
        from app.post_match.evolution import tone_bias_for

        state = session.get(CharacterEvolutionState, character.id)
        initial = initial_mood_for_character(character, tone_bias=tone_bias_for(state))
        save_mood(match.id, initial, smoothed=True)
        save_mood(match.id, initial, smoothed=False)
        return initial, initial
    return raw, smoothed


def _apply_mood_step(match_id: str, raw: MoodState, previous_smoothed: MoodState) -> MoodState:
    smoothed = smooth_mood(previous_smoothed, raw)
    save_mood(match_id, smoothed, smoothed=True)
    save_mood(match_id, raw, smoothed=False)
    return smoothed


# --- Creation / lookup --------------------------------------------------


def get_or_create_player(session: Session, *, player_id: str | None, display_name: str = "Guest") -> Player:
    if player_id:
        existing = session.get(Player, player_id)
        if existing is not None:
            return existing
    from app.auth import generate_guest_username

    player = Player(
        username=generate_guest_username(),
        display_name=display_name or "Guest",
    )
    session.add(player)
    session.flush()
    return player


def create_match(
    session: Session,
    *,
    character_id: str,
    player_id: str,
    player_color: str = "random",
) -> Match:
    character = session.get(Character, character_id)
    if character is None or character.deleted_at is not None:
        raise MatchError(f"Character {character_id} not found")
    player = session.get(Player, player_id)
    if player is None:
        raise MatchError(f"Player {player_id} not found")

    if player_color == "random":
        chosen = random.choice([Color.WHITE, Color.BLACK])
    elif player_color == "white":
        chosen = Color.WHITE
    elif player_color == "black":
        chosen = Color.BLACK
    else:
        raise MatchError(f"Invalid player_color: {player_color}")

    match = Match(
        character_id=character.id,
        player_id=player.id,
        player_color=chosen,
        status=MatchStatus.IN_PROGRESS,
        initial_fen=START_FEN,
        current_fen=START_FEN,
        move_count=0,
        character_elo_at_start=character.current_elo,
        player_elo_at_start=player.elo,
        match_kind="human_vs_character",
    )
    session.add(match)
    session.flush()

    # Seed mood so the first engine turn can read it.
    _load_or_init_mood(session, match)
    return match


def get_match(session: Session, match_id: str) -> Match:
    match = session.get(Match, match_id)
    if match is None:
        raise MatchNotFound(match_id)
    return match


def create_agent_match(
    session: Session,
    *,
    agent_id: str,
    player_id: str,
    character_preset_key: str = "kenji_sato",
    player_color: str = "random",
) -> Match:
    """Create a match where the player's agent faces a preset character.

    The resulting Match has match_kind='agent_vs_character' and
    participant_agent_id set. The owning player_id is also recorded so
    post-match pipelines can apply Elo / memory updates.
    """
    from app.models.player_agent import PlayerAgent

    agent = session.get(PlayerAgent, agent_id)
    if agent is None or agent.archived_at is not None:
        raise MatchError(f"Agent {agent_id} not found or archived")
    if agent.owner_player_id != player_id:
        raise MatchError(f"Agent {agent_id} does not belong to player {player_id}")

    player = session.get(Player, player_id)
    if player is None:
        raise MatchError(f"Player {player_id} not found")

    character = session.execute(
        select(Character).where(Character.preset_key == character_preset_key)
    ).scalar_one_or_none()
    if character is None or character.deleted_at is not None:
        raise MatchError(f"Character preset '{character_preset_key}' not found")

    if player_color == "random":
        chosen = random.choice([Color.WHITE, Color.BLACK])
    elif player_color == "white":
        chosen = Color.WHITE
    elif player_color == "black":
        chosen = Color.BLACK
    else:
        raise MatchError(f"Invalid player_color: {player_color}")

    match = Match(
        character_id=character.id,
        player_id=player.id,
        participant_agent_id=agent_id,
        player_color=chosen,
        status=MatchStatus.IN_PROGRESS,
        initial_fen=START_FEN,
        current_fen=START_FEN,
        move_count=0,
        character_elo_at_start=character.current_elo,
        player_elo_at_start=agent.elo,
        match_kind="agent_vs_character",
    )
    session.add(match)
    session.flush()

    _load_or_init_mood(session, match)
    return match


# --- Turn loop ----------------------------------------------------------


def _persist_move(
    session: Session,
    *,
    match: Match,
    board_before: chess.Board,
    engine_result: MoveResult | None,
    uci: str,
    side: chess.Color,
    player_chat_before: str | None = None,
    mood_snapshot: dict | None = None,
    agent_chat_after: str | None = None,
    surfaced_memory_ids: list[str] | None = None,
) -> Move:
    move = chess.Move.from_uci(uci)
    if move not in board_before.legal_moves:
        raise IllegalMove(f"Illegal move: {uci}")
    san = board_before.san(move)
    board_after = board_before.copy()
    board_after.push(move)

    move_row = Move(
        match_id=match.id,
        move_number=match.move_count + 1,
        side=Color.WHITE if side == chess.WHITE else Color.BLACK,
        uci=uci,
        san=san,
        fen_after=board_after.fen(),
        engine_name=engine_result.engine_name if engine_result else None,
        time_taken_ms=engine_result.time_taken_ms if engine_result else None,
        eval_cp=engine_result.eval_cp if engine_result else None,
        considered_moves=[cm.model_dump() for cm in engine_result.considered_moves]
        if engine_result
        else [],
        thinking_depth=engine_result.thinking_depth if engine_result else None,
        player_chat_before=player_chat_before,
        mood_snapshot=mood_snapshot or {},
        agent_chat_after=agent_chat_after,
        surfaced_memory_ids=list(surfaced_memory_ids or []),
    )
    session.add(move_row)

    match.current_fen = board_after.fen()
    match.move_count += 1
    _finalize_if_over(match, board_after)

    session.flush()
    return move_row


@dataclass
class AgentTurnOutcome:
    """Bundle of what the agents produced during one character turn.

    Returned alongside the engine `Move` so the API layer can surface chat,
    emotion, and memory snippets to the client without re-querying.
    """

    surfaced: list[SurfacedMemory]
    soul: SoulResponse


async def _engine_turn(session: Session, match: Match) -> tuple[Move, AgentTurnOutcome | None]:
    board = board_with_history(match, session)  # history-aware for 3-fold detection
    character = session.get(Character, match.character_id)
    if character is None:
        raise MatchError("Character missing mid-match")

    raw, smoothed = _load_or_init_mood(session, match)
    smoothed = _apply_mood_step(match.id, raw, smoothed)

    available = frozenset(available_engines())
    if not available:
        raise EngineUnavailable("No engines available — install Maia-2 or Stockfish")

    context = MatchContext(
        move_number=board.fullmove_number,
        game_phase=_phase_from_board(board),
        player_color=match.player_color.value,
        engines_available=available,
    )
    config: EngineConfig = choose_engine_config(
        character=character, mood=smoothed, match_context=context
    )

    engine = get_engine(config.engine_name)

    def _run() -> MoveResult:
        return engine.get_move(board, config)

    result = await asyncio.to_thread(_run)

    # --- Run Subconscious + Soul BEFORE persisting the move so we can
    # attach chat + surfaced ids to the Move row in a single flush.
    board_after = board.copy()
    try:
        board_after.push(chess.Move.from_uci(result.move))
    except Exception:
        # If the engine returned an illegal move, bail and let _persist_move raise.
        board_after = board

    agent_outcome: AgentTurnOutcome | None = None
    try:
        agent_outcome = await asyncio.to_thread(
            _run_agents_sync,
            session,
            match,
            character,
            board_after,
            result,
            smoothed,
            raw,
        )
    except Exception as exc:
        logger.warning(
            "Agent pipeline failed (match=%s, move=%s); proceeding silently: %s",
            match.id,
            result.move,
            exc,
        )

    agent_chat = agent_outcome.soul.speak if agent_outcome else None
    surfaced_ids = [m.memory_id for m in agent_outcome.surfaced] if agent_outcome else []

    move_row = _persist_move(
        session,
        match=match,
        board_before=board,
        engine_result=result,
        uci=result.move,
        side=board.turn,
        mood_snapshot=smoothed.to_dict(),
        agent_chat_after=agent_chat,
        surfaced_memory_ids=surfaced_ids,
    )

    # Queue opponent note for post-match (Slice 2 will drain this).
    if agent_outcome and agent_outcome.soul.note_about_opponent:
        notes = list(match.extra_state.get("pending_opponent_notes", [])) if match.extra_state else []
        notes.append(
            {
                "note": agent_outcome.soul.note_about_opponent,
                "move_number": move_row.move_number,
                "ts": datetime.utcnow().isoformat(),
            }
        )
        # SQLAlchemy JSON: reassign to trigger dirty tracking.
        state = dict(match.extra_state or {})
        state["pending_opponent_notes"] = notes
        match.extra_state = state
        session.flush()

    return move_row, agent_outcome


def _load_last_chat_lines(session: Session, match_id: str, limit: int = 5) -> list[str]:
    """Return up to the last `limit` chat lines (player + agent) in chronological order."""
    stmt = (
        select(Move)
        .where(Move.match_id == match_id)
        .order_by(Move.move_number.desc())
        .limit(limit * 2)
    )
    rows = list(session.execute(stmt).scalars())
    rows.reverse()
    out: list[str] = []
    for mv in rows:
        if mv.player_chat_before:
            out.append(f"Player: {mv.player_chat_before}")
        if mv.agent_chat_after:
            out.append(f"You: {mv.agent_chat_after}")
    return out[-limit:]


def _compute_player_timings(session: Session, match: Match) -> tuple[float | None, float | None, float | None]:
    """Return (player_last_seconds, player_avg_seconds, elapsed_total_seconds).

    - player_last_seconds: time between the character's previous move
      (or match start if none) and the most recent player move, in seconds.
    - player_avg_seconds: mean of all such intervals across the player's
      moves so far this match.
    - elapsed_total_seconds: wall time from match start to now.

    All values are None until enough data exists (e.g. no player moves yet).
    """
    from datetime import datetime as _dt

    player_side = match.player_color
    player_moves = list(
        session.execute(
            select(Move)
            .where(Move.match_id == match.id, Move.side == player_side)
            .order_by(Move.move_number.asc())
        ).scalars()
    )
    if not player_moves:
        # No player moves yet (character opens as white).
        elapsed = (_dt.utcnow() - (match.started_at or _dt.utcnow())).total_seconds()
        return None, None, max(0.0, elapsed)

    # For each player move, its "think time" = its created_at minus the
    # previous character move's created_at (or match.started_at for the
    # very first player move).
    char_moves = list(
        session.execute(
            select(Move)
            .where(Move.match_id == match.id, Move.side != player_side)
            .order_by(Move.move_number.asc())
        ).scalars()
    )
    intervals: list[float] = []
    for pm in player_moves:
        prior_char = next((c for c in reversed(char_moves) if c.move_number < pm.move_number), None)
        prev_ts = prior_char.created_at if prior_char else (match.started_at or pm.created_at)
        delta = (pm.created_at - prev_ts).total_seconds()
        if delta >= 0:
            intervals.append(delta)

    last = intervals[-1] if intervals else None
    avg = (sum(intervals) / len(intervals)) if intervals else None
    elapsed = (_dt.utcnow() - (match.started_at or _dt.utcnow())).total_seconds()
    return last, avg, max(0.0, elapsed)


def _load_last_player_san(session: Session, match_id: str) -> str | None:
    match = session.get(Match, match_id)
    if match is None:
        return None
    player_side = match.player_color
    stmt = (
        select(Move)
        .where(Move.match_id == match_id, Move.side == player_side)
        .order_by(Move.move_number.desc())
        .limit(1)
    )
    mv = session.execute(stmt).scalar_one_or_none()
    return mv.san if mv else None


def _load_last_player_context(session: Session, match_id: str) -> tuple[str | None, str | None]:
    """Return (last_player_uci, last_player_chat) from the most recent player move.

    Used by the Subconscious to key its cache and to condition retrieval on
    what the opponent just did.

    Proper side tracking: the player plays `match.player_color`; the character
    plays the opposite. The previous heuristic (`player_chat_before not None or
    agent_chat_after is None`) misidentified Soul-silent character moves as
    player moves, polluting `last_player_uci` + `last_player_chat`.
    """
    match = session.get(Match, match_id)
    if match is None:
        return None, None
    player_side = match.player_color
    stmt = (
        select(Move)
        .where(Move.match_id == match_id, Move.side == player_side)
        .order_by(Move.move_number.desc())
        .limit(1)
    )
    mv = session.execute(stmt).scalar_one_or_none()
    if mv is None:
        return None, None
    return mv.uci, mv.player_chat_before


def _opponent_profile_for(
    session: Session, *, character_id: str, player_id: str
) -> OpponentProfile | None:
    stmt = select(OpponentProfile).where(
        OpponentProfile.character_id == character_id,
        OpponentProfile.player_id == player_id,
    )
    return session.execute(stmt).scalar_one_or_none()


def _profile_summary(profile: OpponentProfile | None) -> dict | None:
    if profile is None:
        return None
    return {
        "games_played": profile.games_played,
        "character_record": {
            "w": profile.games_won_by_character,
            "l": profile.games_lost_by_character,
            "d": profile.games_drawn,
        },
        "style_features": profile.style_features or {},
        "narrative_summary": profile.narrative_summary or "",
    }


def _head_to_head(profile: OpponentProfile | None) -> dict[str, int] | None:
    if profile is None or profile.games_played == 0:
        return None
    return {
        "w": profile.games_won_by_character,
        "l": profile.games_lost_by_character,
        "d": profile.games_drawn,
    }


def _run_agents_sync(
    session: Session,
    match: Match,
    character: Character,
    board_after: chess.Board,
    engine_result: MoveResult,
    smoothed: MoodState,
    raw: MoodState,
) -> AgentTurnOutcome:
    """Subconscious -> Soul -> mood-delta apply. Synchronous; called via to_thread."""
    board_summary = board_to_english(board_after, eval_cp=engine_result.eval_cp)

    last_player_uci, last_player_chat = _load_last_player_context(session, match.id)
    last_player_san = _load_last_player_san(session, match.id)
    character_color_str = "black" if match.player_color == Color.WHITE else "white"
    player_took_s, player_avg_s, elapsed_s = _compute_player_timings(session, match)

    # Pull the most recent SAN moves for retrieval context.
    last_move_rows = list(
        session.execute(
            select(Move)
            .where(Move.match_id == match.id)
            .order_by(Move.move_number.desc())
            .limit(6)
        ).scalars()
    )
    last_moves_san = [mv.san for mv in reversed(last_move_rows) if mv.san]

    profile = _opponent_profile_for(
        session, character_id=match.character_id, player_id=match.player_id
    )
    style_features = profile.style_features if profile else None

    sub_input = SubconsciousInput(
        character_id=match.character_id,
        match_id=match.id,
        current_turn=match.move_count + 1,  # about-to-persist move number
        board_summary=board_summary,
        mood=smoothed,
        last_player_uci=last_player_uci,
        last_player_chat=last_player_chat,
        last_moves_san=last_moves_san,
        recent_chat=_load_last_chat_lines(session, match.id),
        opening_label=None,  # ECO classification lands in Slice 2 via post-match
        current_player_id=match.player_id,
        opponent_style_features=style_features,
    )
    surfaced = run_subconscious(session, character, sub_input)

    # Derive SAN for the engine's move (the prior position is needed for this).
    pre_board = board_after.copy()
    try:
        last = pre_board.pop()
        engine_move_san = pre_board.san(last)
    except Exception:
        engine_move_san = engine_result.move

    soul_input = SoulInput(
        board=board_summary,
        mood=smoothed,
        surfaced_memories=surfaced,
        recent_chat=sub_input.recent_chat,
        engine_move_san=engine_move_san,
        engine_move_uci=engine_result.move,
        engine_eval_cp=engine_result.eval_cp,
        engine_considered=[cm.model_dump() for cm in engine_result.considered_moves],
        engine_time_ms=engine_result.time_taken_ms,
        move_number=match.move_count + 1,
        game_phase=board_summary.phase,
        opponent_profile_summary=_profile_summary(profile),
        head_to_head=_head_to_head(profile),
        player_just_spoke=bool(last_player_chat),
        last_player_chat=last_player_chat,
        match_id=match.id,
        character_color=character_color_str,
        opponent_last_san=last_player_san,
        opponent_last_uci=last_player_uci,
        player_took_seconds=player_took_s,
        player_average_seconds=player_avg_s,
        elapsed_total_seconds=elapsed_s,
    )

    soul_resp = run_soul(character, soul_input)

    # Apply the Soul's mood deltas to RAW mood; re-smooth; persist.
    deltas = soul_resp.mood_deltas.to_dict()
    new_raw = apply_deltas(raw, deltas)
    new_smoothed = smooth_mood(smoothed, new_raw)
    save_mood(match.id, new_raw, smoothed=False)
    save_mood(match.id, new_smoothed, smoothed=True)

    return AgentTurnOutcome(surfaced=surfaced, soul=soul_resp)


async def start_match_play(session: Session, match: Match) -> Move | None:
    """If the character is white, play the opening move before returning to the user.

    Agent output (chat, emotion, surfaced memories) is still persisted on the
    resulting Move row — clients read it via the moves endpoint. Returning just
    the Move keeps the signature compatible with Phase 2a callers.
    """
    if match.status != MatchStatus.IN_PROGRESS:
        return None
    board = board_with_history(match, session)
    if board.turn != _char_color(match):
        return None
    move, _outcome = await _engine_turn(session, match)
    return move


async def apply_player_move(
    session: Session,
    *,
    match_id: str,
    uci: str,
    player_chat: str | None = None,
) -> tuple[Move, Move | None, AgentTurnOutcome | None]:
    match = get_match(session, match_id)
    if match.status != MatchStatus.IN_PROGRESS:
        raise GameAlreadyOver(match.status.value)

    board = board_with_history(match, session)
    player_color = chess.WHITE if match.player_color == Color.WHITE else chess.BLACK
    if board.turn != player_color:
        raise NotYourTurn("It's not your turn")

    player_move = _persist_move(
        session,
        match=match,
        board_before=board,
        engine_result=None,
        uci=uci,
        side=player_color,
        player_chat_before=player_chat,
    )

    if match.status != MatchStatus.IN_PROGRESS:
        return player_move, None, None

    engine_move, agent_outcome = await _engine_turn(session, match)
    return player_move, engine_move, agent_outcome


def resign(session: Session, *, match_id: str) -> Match:
    """Clean concession: match ends RESIGNED, character wins on the actual board side.

    Distinct from disconnect-timeout (which uses `abandon_for_disconnect`):
    - RESIGNED: player chose to end it; full Elo + memory pipelines run normally.
    - ABANDONED: player never came back; rage-quit branch runs.
    """
    match = get_match(session, match_id)
    if match.status != MatchStatus.IN_PROGRESS:
        raise GameAlreadyOver(match.status.value)
    match.status = MatchStatus.RESIGNED
    # Character wins the side opposite the resigning player.
    match.result = (
        MatchResult.BLACK_WIN if match.player_color == Color.WHITE else MatchResult.WHITE_WIN
    )
    match.ended_at = datetime.utcnow()
    match.character_elo_at_end = match.character_elo_at_start
    _stash_trailing_pending_chat(match)
    session.flush()
    return match


def abandon_for_disconnect(session: Session, *, match_id: str) -> Match:
    """Disconnect-timeout path: match ends ABANDONED (rage-quit semantics)."""
    match = get_match(session, match_id)
    if match.status != MatchStatus.IN_PROGRESS:
        raise GameAlreadyOver(match.status.value)
    match.status = MatchStatus.ABANDONED
    match.result = MatchResult.ABANDONED
    match.ended_at = datetime.utcnow()
    match.character_elo_at_end = match.character_elo_at_start
    _stash_trailing_pending_chat(match)
    session.flush()
    return match


def _stash_trailing_pending_chat(match: Match) -> None:
    """On abrupt match end, preserve any un-persisted pending player chat.

    Moves anything in `extra_state.pending_player_chat` into `trailing_player_chat`
    so post-match features (narrative summary, memory_gen) can still read the
    final messages. Empties the pending buffer.
    """
    state = dict(match.extra_state or {})
    pending = list(state.get("pending_player_chat", []))
    if not pending:
        return
    trailing = list(state.get("trailing_player_chat", []))
    trailing.extend(pending)
    state["trailing_player_chat"] = trailing
    state["pending_player_chat"] = []
    match.extra_state = state


# --- Outcome helpers for the API ---------------------------------------


def player_outcome(match: Match) -> str | None:
    if match.status == MatchStatus.IN_PROGRESS:
        return None
    if match.status in (MatchStatus.RESIGNED, MatchStatus.ABANDONED):
        return "resigned"
    if match.result == MatchResult.DRAW:
        return "draw"
    player_side = match.player_color
    winning_side = Color.WHITE if match.result == MatchResult.WHITE_WIN else Color.BLACK
    return "win" if player_side == winning_side else "loss"
