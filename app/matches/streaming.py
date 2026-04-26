"""Streamed (event-emitting) match turn loop for Socket.IO (Phase 3b).

The REST path in `app.matches.service.apply_player_move` remains intact as a
fallback. This module provides the socket-aware variant: it runs engine and
Subconscious concurrently (so `memory_surfaced` fires before `agent_move`),
persists the player + character moves, and invokes per-phase emitters that the
Socket.IO layer wires to rooms.

Semantic shift (documented decision): the Subconscious now runs on the board
_after the player's move, before the engine's move_ — i.e. on the position
the character is about to respond to. In the Phase 2b REST pipeline it ran on
the post-engine board. This shift is what lets us surface memories while the
engine is still thinking. Board prose and context tokens are still well-defined
at this position, so retrieval quality is not materially affected.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Awaitable, Callable

import chess
from sqlalchemy import select

from app.agents.soul import SoulInput, run_soul
from app.agents.subconscious import SubconsciousInput, run_subconscious
from app.config import get_settings
from app.db import SessionLocal
from app.director.director import MatchContext, choose_engine_config
from app.director.mood import MoodState, apply_deltas, load_mood, save_mood, smooth_mood
from app.engine import EngineUnavailable, available_engines, get_engine
from app.engine.base import ConsideredMove, EngineConfig, MoveResult
from app.engine.diversity_guard import filter_shuffle_moves
from app.engine.board_abstraction import board_to_english
from app.matches import service as _svc
from app.matches.service import (
    GameAlreadyOver,
    IllegalMove,
    MatchNotFound,
    NotYourTurn,
)
from app.models.character import Character
from app.models.match import Color, Match, MatchResult, MatchStatus, Move, OpponentProfile, Player
from app.memory.inline_save import save_inline_memory
from app.schemas.agents import SoulResponse, SurfacedMemory

logger = logging.getLogger(__name__)


# --- Emitter bundle --------------------------------------------------------


@dataclass
class TurnEmitters:
    """Callbacks the streamed turn loop invokes at each phase.

    Each is an async callable with no-op default. The Socket.IO server supplies
    real emitters; tests can supply stubs that record events in-order.
    """

    on_player_move: Callable[[Move], Awaitable[None]]
    on_thinking: Callable[[float], Awaitable[None]]
    on_memory_surfaced: Callable[[list[SurfacedMemory]], Awaitable[None]]
    on_agent_move: Callable[[Move], Awaitable[None]]
    on_agent_chat: Callable[[SoulResponse], Awaitable[None]]
    on_mood_update: Callable[[MoodState], Awaitable[None]]
    on_match_ended: Callable[[str, str, str | None], Awaitable[None]]
    on_post_match_kickoff: Callable[[], Awaitable[None]]


# --- Helpers ---------------------------------------------------------------


def _phase_from_board(board: chess.Board) -> str:
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


def _estimate_eta_seconds(time_budget_seconds: float) -> float:
    """`eta_seconds` for agent_thinking = engine budget + fixed Soul overhead, rounded to 0.5s.

    Soft estimate. If Soul returns `speak=None`, user-visible "thinking" actually ends
    at `agent_move`, not `agent_chat`. UI should not treat as a hard countdown.
    """
    overhead = float(get_settings().agent_thinking_soul_overhead_seconds)
    total = time_budget_seconds + overhead
    return round(total * 2) / 2  # nearest 0.5


def _drain_pending_chat(match: Match) -> list[dict[str, Any]]:
    """Pop pending player chat off extra_state; caller must commit."""
    state = dict(match.extra_state or {})
    pending = list(state.get("pending_player_chat", []))
    state["pending_player_chat"] = []
    match.extra_state = state
    return pending


def _join_pending_chat(pending: list[dict[str, Any]]) -> str | None:
    """Join mid-thinking chat messages into a single transcript string.

    Separator: ' / '. Returns None when no messages — so callers can assign
    straight to a nullable Move.player_chat_before column.
    """
    texts = [entry.get("text", "") for entry in pending if entry.get("text")]
    return " / ".join(texts) if texts else None


def _finalize_outcome(match: Match, board: chess.Board) -> tuple[str, str, str | None] | None:
    """If the board terminated, set status/result on the match and return (reason, result, outcome).

    Does NOT commit. Caller owns the session.

    Patch Pass 2 Item 3: caller must pass a history-aware board (built via
    `board_with_history`) so 3-fold / 5-fold repetition + 75-move checks
    actually fire. Boards built from FEN alone have no move stack and
    silently miss every repetition draw.
    """
    if match.status != MatchStatus.IN_PROGRESS:
        # Already ended via resign or disconnect-timeout.
        if match.status == MatchStatus.RESIGNED:
            result_value = match.result.value if match.result else "abandoned"
            return "resign", result_value, "resigned"
        if match.status == MatchStatus.ABANDONED:
            return "disconnect_timeout", "abandoned", "resigned"
        return None

    # Mandatory draws (don't require a claim).
    if board.is_fivefold_repetition() or board.is_seventyfive_moves():
        match.status = MatchStatus.COMPLETED
        match.ended_at = datetime.utcnow()
        match.character_elo_at_end = match.character_elo_at_start
        match.result = MatchResult.DRAW
        reason = "fivefold_repetition" if board.is_fivefold_repetition() else "seventyfive_moves"
        return reason, "draw", "draw"

    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        return None

    match.status = MatchStatus.COMPLETED
    match.ended_at = datetime.utcnow()
    match.character_elo_at_end = match.character_elo_at_start

    if outcome.winner is None:
        match.result = MatchResult.DRAW
        term = outcome.termination.name
        if term == "STALEMATE":
            reason = "stalemate"
        elif term == "THREEFOLD_REPETITION":
            reason = "threefold_repetition"
        else:
            reason = "draw_rule"
        return reason, "draw", "draw"

    if outcome.winner == chess.WHITE:
        match.result = MatchResult.WHITE_WIN
    else:
        match.result = MatchResult.BLACK_WIN

    player_is_white = match.player_color == Color.WHITE
    player_won = (outcome.winner == chess.WHITE) == player_is_white
    return "checkmate", match.result.value, ("win" if player_won else "loss")


def _load_last_chat_lines(session, match_id: str, limit: int = 5) -> list[str]:
    rows = list(
        session.execute(
            select(Move)
            .where(Move.match_id == match_id)
            .order_by(Move.move_number.desc())
            .limit(limit * 2)
        ).scalars()
    )
    rows.reverse()
    out: list[str] = []
    for mv in rows:
        if mv.player_chat_before:
            out.append(f"Player: {mv.player_chat_before}")
        if mv.agent_chat_after:
            out.append(f"You: {mv.agent_chat_after}")
    return out[-limit:]


def _load_cross_chat_lines(
    session,
    match_id: str,
    own_color: str,
    opponent_name: str,
    limit: int = 5,
) -> list[str]:
    """Like `_load_last_chat_lines` but labels each side's chat distinctly.

    Used in agent_vs_character matches so each Soul can tell its own quips
    apart from the opponent's.  own_color is the SIDE whose chat becomes
    "You: ...", the other side's chat becomes "{opponent_name}: ...".
    """
    from app.models.match import Color

    own_side = Color.WHITE if own_color == "white" else Color.BLACK
    rows = list(
        session.execute(
            select(Move)
            .where(Move.match_id == match_id)
            .order_by(Move.move_number.desc())
            .limit(limit * 2)
        ).scalars()
    )
    rows.reverse()
    out: list[str] = []
    for mv in rows:
        if mv.player_chat_before:
            out.append(f"Player: {mv.player_chat_before}")
        if mv.agent_chat_after:
            if mv.side == own_side:
                out.append(f"You: {mv.agent_chat_after}")
            else:
                out.append(f"{opponent_name}: {mv.agent_chat_after}")
    return out[-limit:]


def _load_last_player_san(session, match_id: str) -> str | None:
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


def _load_last_player_context(session, match_id: str) -> tuple[str | None, str | None]:
    """See `app.matches.service._load_last_player_context` — mirrored here for
    the streaming turn loop (doesn't import from service to avoid cycles)."""
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


def _opponent_profile_for(session, *, character_id: str, player_id: str) -> OpponentProfile | None:
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


# --- Own-move extraction for anti-shuffle filter -------------------------


def _extract_own_recent_moves(board: chess.Board, lookback: int) -> list[chess.Move]:
    """Return the side-to-move's last `lookback` own moves from the board stack, newest first.

    After the player moves it's the engine's turn; own moves sit at stack positions
    -2, -4, -6 … (every other ply, skipping the opponent's intervening moves).
    """
    stack = board.move_stack
    out: list[chess.Move] = []
    for i in range(2, 2 * lookback + 1, 2):
        if len(stack) >= i:
            out.append(stack[-i])
    return out


# --- Player move persistence ----------------------------------------------


def _peek_pending_chat(match: Match) -> list[dict[str, Any]]:
    """Read pending_player_chat without clearing it. Distinct from _drain_* which
    empties the buffer."""
    state = match.extra_state or {}
    return list(state.get("pending_player_chat", []))


def _merge_player_chat(pending: list[dict[str, Any]], inline: str | None) -> str | None:
    """Combine mid-think buffered chat with any chat attached to the make_move event."""
    texts = [entry.get("text", "") for entry in pending if entry.get("text")]
    if inline:
        texts.append(inline)
    return " / ".join(texts) if texts else None


def _stash_trailing_chat(match: Match, pending: list[dict[str, Any]]) -> None:
    """Move any leftover pending chat into extra_state.trailing_player_chat.

    Used when the match ends before the next player move could claim these
    messages as its `player_chat_before`. The post-match pipeline can read
    this if it wants the final words.
    """
    if not pending:
        return
    state = dict(match.extra_state or {})
    trailing = list(state.get("trailing_player_chat", []))
    trailing.extend(pending)
    state["trailing_player_chat"] = trailing
    state["pending_player_chat"] = []
    match.extra_state = state


def _persist_player_move(session, match: Match, uci: str, player_chat: str | None) -> Move:
    # Patch Pass 2 Item 3: build the board from move history, not just the
    # current FEN, so the upcoming finalization can detect repetition draws.
    board = _svc.board_with_history(match, session)
    player_color = chess.WHITE if match.player_color == Color.WHITE else chess.BLACK
    if board.turn != player_color:
        raise NotYourTurn("It's not your turn")
    if match.status != MatchStatus.IN_PROGRESS:
        raise GameAlreadyOver(match.status.value)

    move = chess.Move.from_uci(uci)
    if move not in board.legal_moves:
        raise IllegalMove(f"Illegal move: {uci}")
    san = board.san(move)
    board.push(move)

    row = Move(
        match_id=match.id,
        move_number=match.move_count + 1,
        side=Color.WHITE if player_color == chess.WHITE else Color.BLACK,
        uci=uci,
        san=san,
        fen_after=board.fen(),
        considered_moves=[],
        player_chat_before=player_chat,
    )
    session.add(row)
    match.current_fen = board.fen()
    match.move_count += 1
    session.flush()
    return row


def _persist_engine_move(
    session,
    match: Match,
    engine_result: MoveResult,
    board_before: chess.Board,
    *,
    mood_snapshot: dict,
) -> Move:
    move = chess.Move.from_uci(engine_result.move)
    if move not in board_before.legal_moves:
        raise IllegalMove(f"Engine produced illegal move: {engine_result.move}")
    san = board_before.san(move)
    board_after = board_before.copy()
    board_after.push(move)

    engine_side = board_before.turn
    row = Move(
        match_id=match.id,
        move_number=match.move_count + 1,
        side=Color.WHITE if engine_side == chess.WHITE else Color.BLACK,
        uci=engine_result.move,
        san=san,
        fen_after=board_after.fen(),
        engine_name=engine_result.engine_name,
        time_taken_ms=engine_result.time_taken_ms,
        eval_cp=engine_result.eval_cp,
        considered_moves=[cm.model_dump() for cm in engine_result.considered_moves],
        thinking_depth=engine_result.thinking_depth,
        mood_snapshot=mood_snapshot,
        surfaced_memory_ids=[],
    )
    session.add(row)
    match.current_fen = board_after.fen()
    match.move_count += 1
    session.flush()
    return row


# --- Main entry point ------------------------------------------------------


# --- Item 5: chat-triggered immediate Soul response -----------------------


# In-flight turn registry: marks that `_run_engine_and_agents` is executing for
# a given match. Chat-triggered Soul calls skip this window so we don't
# double-fire on top of the turn's own Soul call.
_in_flight_turns: set[str] = set()


def _turn_in_flight(match_id: str) -> bool:
    return match_id in _in_flight_turns


# Per-match last-fired timestamp for the chat-triggered path (monotonic ms).
_chat_soul_last_ms: dict[str, int] = {}


def _chat_soul_rate_limit_ok(match_id: str, min_interval_ms: int) -> bool:
    """Token-spacing: returns True if the match hasn't fired a chat-triggered
    Soul call in the last `min_interval_ms`. Updates the timestamp on pass.
    """
    import time as _time

    now_ms = int(_time.monotonic() * 1000)
    last = _chat_soul_last_ms.get(match_id, 0)
    if now_ms - last < min_interval_ms:
        return False
    _chat_soul_last_ms[match_id] = now_ms
    return True


def reset_chat_soul_rate_limit() -> None:
    """Testing hook."""
    _chat_soul_last_ms.clear()
    _in_flight_turns.clear()


async def run_chat_triggered_soul(
    *,
    match_id: str,
    emit_chat: Callable[[SoulResponse], Awaitable[None]],
) -> bool:
    """Fire a lightweight Soul call when the player chats between character turns.

    No engine move, no mood persist, no Move-row side-effects. Just: character
    sheet + mood + opponent profile + last N chat messages + current board
    summary → SoulResponse. If it speaks, emit agent_chat.

    Returns True if the call fired (even if silent), False if rate-limited or
    skipped because a character turn is in flight.

    Rate limit: `chat_triggered_soul_min_interval_ms` per match. During an
    active character turn the regular buffering path handles the chat —
    this function no-ops in that window.
    """
    settings = get_settings()
    if _turn_in_flight(match_id):
        return False
    if not _chat_soul_rate_limit_ok(match_id, settings.chat_triggered_soul_min_interval_ms):
        return False

    with SessionLocal() as session:
        match = session.get(Match, match_id)
        if match is None or match.status != MatchStatus.IN_PROGRESS:
            return False
        character = session.get(Character, match.character_id)
        if character is None:
            return False

        smoothed = load_mood(match.id, smoothed=True)
        if smoothed is None:
            from app.director import initial_mood_for_character

            smoothed = initial_mood_for_character(character)

        board = _svc.board_with_history(match, session)
        board_summary = board_to_english(board, eval_cp=None)

        last_player_uci, last_player_chat = _load_last_player_context(session, match.id)
        last_player_san = _load_last_player_san(session, match.id)

        # Chat context: last N moves' chat plus any pending.
        recent_chat = _load_last_chat_lines(session, match.id)
        pending = _peek_pending_chat(match)
        for entry in pending:
            recent_chat.append(f"Player: {entry.get('text', '')}")

        profile = _opponent_profile_for(
            session, character_id=match.character_id, player_id=match.player_id
        )
        character_color_str = "black" if match.player_color == Color.WHITE else "white"
        player_took_s, player_avg_s, elapsed_s = _svc._compute_player_timings(session, match)

    # No engine move at chat-triggered time — point the prompt at the last
    # character move so the Soul can respond in context. If the character
    # hasn't moved yet, leave engine_move_* blank.
    with SessionLocal() as s2:
        last_char_move = s2.execute(
            select(Move)
            .where(
                Move.match_id == match_id,
                Move.side != (Color.WHITE if character_color_str == "black" else Color.BLACK),
            )
            .order_by(Move.move_number.desc())
            .limit(1)
        ).scalar_one_or_none()
    engine_move_san = last_char_move.san if last_char_move else ""
    engine_move_uci = last_char_move.uci if last_char_move else ""

    def _soul_call() -> SoulResponse:
        soul_input = SoulInput(
            board=board_summary,
            mood=smoothed,
            surfaced_memories=[],  # Subconscious skipped on the lightweight path
            recent_chat=recent_chat,
            engine_move_san=engine_move_san,
            engine_move_uci=engine_move_uci,
            engine_eval_cp=None,
            engine_considered=[],
            engine_time_ms=None,
            move_number=match.move_count,
            game_phase=board_summary.phase,
            opponent_profile_summary=_profile_summary(profile) if profile else None,
            head_to_head=_head_to_head(profile) if profile else None,
            player_just_spoke=True,
            last_player_chat=(pending[-1].get("text") if pending else last_player_chat),
            match_id=match_id,
            character_color=character_color_str,
            opponent_last_san=last_player_san,
            opponent_last_uci=last_player_uci,
            player_took_seconds=player_took_s,
            player_average_seconds=player_avg_s,
            elapsed_total_seconds=elapsed_s,
        )
        return run_soul(character, soul_input)

    try:
        soul_resp = await asyncio.to_thread(_soul_call)
    except Exception as exc:
        logger.warning(
            "Chat-triggered Soul failed (match=%s): %s — dropping",
            match_id, exc,
        )
        return True  # still "fired" — rate-limit already counted

    if soul_resp.speak:
        await emit_chat(soul_resp)
    return True


async def apply_player_move_streamed(
    *,
    match_id: str,
    uci: str,
    player_chat: str | None,
    emitters: TurnEmitters,
) -> None:
    """End-to-end streamed turn, emitting Socket.IO events through `emitters`.

    Opens its own DB sessions. Does not return any value — everything the caller
    wants to know flows through the emitters. Exceptions propagate (the caller —
    the socket handler — translates them to error events).
    """
    settings = get_settings()

    # --- Phase 1: persist player's move --------------------------------
    with SessionLocal() as session:
        match = session.get(Match, match_id)
        if match is None:
            raise MatchNotFound(match_id)
        # Drain the mid-think chat buffer and merge with any chat attached to
        # this make_move event. The joined string is the Move.player_chat_before
        # — so transcripts, spectators, and memory_gen all see player chat.
        pending_before = _drain_pending_chat(match)
        merged_chat = _merge_player_chat(pending_before, player_chat)
        player_row = _persist_player_move(session, match, uci, merged_chat)
        # Patch Pass 2 Item 3: history-aware for 3-fold detection.
        board_after_player = _svc.board_with_history(match, session)
        end_info = _finalize_outcome(match, board_after_player)
        if end_info is not None:
            # Match just ended on the player's move. Any chat still in pending
            # (none at this point since we just drained) or arriving later goes
            # to trailing_player_chat — there's no next Move to attach it to.
            _stash_trailing_chat(match, _drain_pending_chat(match))
        session.commit()
        session.refresh(player_row)

    await emitters.on_player_move(player_row)

    if end_info is not None:
        reason, result_value, outcome = end_info
        await emitters.on_match_ended(reason, result_value, outcome)
        await emitters.on_post_match_kickoff()
        return

    # --- Phase 2: kick off engine + subconscious concurrently ----------
    # Open a fresh session for the agent work. Most of it runs in `to_thread`
    # since the engine + Subconscious are blocking.
    await _run_engine_and_agents(match_id=match_id, emitters=emitters, settings=settings)


async def _run_engine_and_agents(
    *, match_id: str, emitters: TurnEmitters, settings
) -> None:
    _in_flight_turns.add(match_id)
    try:
        await _run_engine_and_agents_inner(match_id=match_id, emitters=emitters, settings=settings)
    finally:
        _in_flight_turns.discard(match_id)


async def _run_engine_and_agents_inner(
    *, match_id: str, emitters: TurnEmitters, settings
) -> None:
    with SessionLocal() as session:
        match = session.get(Match, match_id)
        if match is None:
            raise MatchNotFound(match_id)
        if match.status != MatchStatus.IN_PROGRESS:
            return

        character = session.get(Character, match.character_id)
        if character is None:
            raise _svc.MatchError("Character missing mid-match")

        # Mood init / load.
        smoothed = load_mood(match.id, smoothed=True)
        raw = load_mood(match.id, smoothed=False)
        if smoothed is None or raw is None:
            from app.director import initial_mood_for_character

            initial = initial_mood_for_character(character)
            save_mood(match.id, initial, smoothed=True)
            save_mood(match.id, initial, smoothed=False)
            smoothed = raw = initial
        smoothed = smooth_mood(smoothed, raw)
        save_mood(match.id, smoothed, smoothed=True)
        save_mood(match.id, raw, smoothed=False)

        # Patch Pass 2 Item 3: rebuild from move history so post-move
        # finalization can detect 3-fold / 5-fold repetition.
        board = _svc.board_with_history(match, session)

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

        # --- Prep Subconscious input on the PRE-ENGINE board --------------
        pre_engine_summary = board_to_english(board, eval_cp=None)
        last_player_uci, last_player_chat = _load_last_player_context(session, match.id)
        last_player_san = _load_last_player_san(session, match.id)
        character_color_str = "black" if match.player_color == Color.WHITE else "white"
        player_took_s, player_avg_s, elapsed_s = _svc._compute_player_timings(session, match)
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
        # For agent_vs_character, label each side's chat distinctly so Kenji's
        # Soul can tell its own quips from the agent's. Otherwise fall back to
        # the standard "You: ..." labeling used in human_vs_character matches.
        if match.match_kind == "agent_vs_character" and match.participant_agent_id:
            from app.models.player_agent import PlayerAgent as _PA
            _agent = session.get(_PA, match.participant_agent_id)
            agent_name = _agent.name if _agent else "Opponent"
            recent_chat = _load_cross_chat_lines(
                session, match.id,
                own_color=character_color_str,
                opponent_name=agent_name,
            )
        else:
            recent_chat = _load_last_chat_lines(session, match.id)

        # Peek — DON'T drain — chat sent during engine thinking. Those messages
        # will be claimed as player_chat_before on the *next* player move's
        # Phase 1 drain. Feeding them to Subconscious here is additive context.
        pending = _peek_pending_chat(match)
        for entry in pending:
            recent_chat.append(f"Player: {entry.get('text', '')}")

        # Final cap on recent_chat passed to retrieval.
        recent_chat = recent_chat[-(get_settings().pending_chat_max_messages + 5):]

        sub_input = SubconsciousInput(
            character_id=match.character_id,
            match_id=match.id,
            current_turn=match.move_count + 1,
            board_summary=pre_engine_summary,
            mood=smoothed,
            last_player_uci=last_player_uci,
            last_player_chat=last_player_chat,
            last_moves_san=last_moves_san,
            recent_chat=recent_chat,
            opening_label=None,
            current_player_id=match.player_id,
            opponent_style_features=profile.style_features if profile else None,
        )

        session.commit()  # persist pending-chat drain before fanning out to threads

        match_id_local = match.id
        character_id_local = match.character_id
        player_id_local = match.player_id
        mood_snapshot_for_move = smoothed.to_dict()
        eta = _estimate_eta_seconds(config.time_budget_seconds)

    # Announce thinking *before* doing the heavy work.
    await emitters.on_thinking(eta)

    # Fan out: engine call + Subconscious call in parallel threads.
    def _engine_call() -> MoveResult:
        return engine.get_move(board, config)

    def _subconscious_call() -> list[SurfacedMemory]:
        with SessionLocal() as s:
            char = s.get(Character, character_id_local)
            assert char is not None
            return run_subconscious(s, char, sub_input)

    engine_task = asyncio.create_task(asyncio.to_thread(_engine_call))
    sub_task = asyncio.create_task(asyncio.to_thread(_subconscious_call))

    surfaced: list[SurfacedMemory] = []
    engine_result: MoveResult | None = None

    # Wait on whichever finishes first; emit as they arrive.
    pending_tasks = {engine_task, sub_task}
    while pending_tasks:
        done, pending_tasks = await asyncio.wait(
            pending_tasks, return_when=asyncio.FIRST_COMPLETED
        )
        for task in done:
            if task is sub_task:
                try:
                    surfaced = await task
                except Exception as exc:
                    logger.warning(
                        "Subconscious failed (match=%s): %s — continuing without memories",
                        match_id_local, exc,
                    )
                    surfaced = []
                await emitters.on_memory_surfaced(surfaced)
            elif task is engine_task:
                engine_result = await task  # surfaces EngineUnavailable if it happened

    assert engine_result is not None  # engine_task is always awaited above

    # Anti-shuffle: skip candidates that cycle a piece back to a square it recently left.
    own_recent = _extract_own_recent_moves(board, config.shuffle_guard_lookback)
    top_cm = ConsideredMove(uci=engine_result.move, san=engine_result.san, eval_cp=engine_result.eval_cp)
    alts = [cm for cm in engine_result.considered_moves if cm.uci != engine_result.move]
    chosen = filter_shuffle_moves([top_cm] + alts, own_recent, board, config.shuffle_guard_lookback)
    if chosen.uci != engine_result.move:
        logger.info(
            "shuffle_guard: match=%s rejected %s → chose %s",
            match_id_local, engine_result.move, chosen.uci,
        )
        chosen_move = chess.Move.from_uci(chosen.uci)
        if chosen_move in board.legal_moves:
            engine_result = MoveResult(
                move=chosen.uci,
                san=board.san(chosen_move),
                eval_cp=chosen.eval_cp,
                considered_moves=engine_result.considered_moves,
                time_taken_ms=engine_result.time_taken_ms,
                engine_name=engine_result.engine_name,
                thinking_depth=engine_result.thinking_depth,
                raw=engine_result.raw,
            )
        else:
            logger.warning(
                "shuffle_guard: chosen %s is illegal on this board — keeping %s",
                chosen.uci, engine_result.move,
            )

    # --- Phase 3: persist engine move + emit agent_move ----------------
    with SessionLocal() as session:
        match = session.get(Match, match_id_local)
        assert match is not None
        # Patch Pass 2 Item 3: history-aware for 3-fold detection post-engine.
        board_before = _svc.board_with_history(match, session)
        engine_move_row = _persist_engine_move(
            session, match, engine_result, board_before,
            mood_snapshot=mood_snapshot_for_move,
        )
        session.commit()
        session.refresh(engine_move_row)

    await emitters.on_agent_move(engine_move_row)

    # --- Phase 4: run Soul (post-engine board) + mood update -----------
    post_engine_board = chess.Board(engine_move_row.fen_after)
    post_engine_summary = board_to_english(post_engine_board, eval_cp=engine_result.eval_cp)

    pre_board = post_engine_board.copy()
    try:
        last = pre_board.pop()
        engine_move_san = pre_board.san(last)
    except Exception:
        engine_move_san = engine_result.move

    def _soul_call() -> SoulResponse:
        soul_input = SoulInput(
            board=post_engine_summary,
            mood=smoothed,
            surfaced_memories=surfaced,
            recent_chat=recent_chat,
            engine_move_san=engine_move_san,
            engine_move_uci=engine_result.move,
            engine_eval_cp=engine_result.eval_cp,
            engine_considered=[cm.model_dump() for cm in engine_result.considered_moves],
            engine_time_ms=engine_result.time_taken_ms,
            move_number=engine_move_row.move_number,
            game_phase=post_engine_summary.phase,
            opponent_profile_summary=_profile_summary(profile) if profile else None,
            head_to_head=_head_to_head(profile) if profile else None,
            player_just_spoke=bool(last_player_chat) or bool(pending),
            last_player_chat=last_player_chat,
            match_id=match_id_local,
            character_color=character_color_str,
            opponent_last_san=last_player_san,
            opponent_last_uci=last_player_uci,
            player_took_seconds=player_took_s,
            player_average_seconds=player_avg_s,
            elapsed_total_seconds=elapsed_s,
        )
        with SessionLocal() as _s:
            _char = _s.get(Character, character_id_local)
            if _char is None:
                return SoulResponse()
            return run_soul(_char, soul_input)

    try:
        soul_resp = await asyncio.to_thread(_soul_call)
    except Exception as exc:
        logger.warning(
            "Soul failed (match=%s): %s — proceeding silently", match_id_local, exc,
        )
        soul_resp = SoulResponse()

    # Apply mood deltas + persist soul output onto the Move row.
    new_raw = apply_deltas(raw, soul_resp.mood_deltas.to_dict())
    new_smoothed = smooth_mood(smoothed, new_raw)
    save_mood(match_id_local, new_raw, smoothed=False)
    save_mood(match_id_local, new_smoothed, smoothed=True)

    surfaced_ids = [m.memory_id for m in surfaced]
    with SessionLocal() as session:
        match = session.get(Match, match_id_local)
        assert match is not None
        engine_row = session.get(Move, engine_move_row.id)
        assert engine_row is not None
        engine_row.agent_chat_after = soul_resp.speak
        engine_row.surfaced_memory_ids = surfaced_ids

        # Queue opponent note for post-match, same shape as the REST path.
        if soul_resp.note_about_opponent:
            state = dict(match.extra_state or {})
            notes = list(state.get("pending_opponent_notes", []))
            notes.append(
                {
                    "note": soul_resp.note_about_opponent,
                    "move_number": engine_row.move_number,
                    "ts": datetime.utcnow().isoformat(),
                }
            )
            state["pending_opponent_notes"] = notes
            match.extra_state = state

        # Did the engine's move end the game?
        # Patch Pass 2 Item 3: history-aware for 3-fold / 5-fold detection.
        board_after_engine = _svc.board_with_history(match, session)
        end_info = _finalize_outcome(match, board_after_engine)
        if end_info is not None:
            # Match ended on the engine's move. Any pending player chat has no
            # future Move to attach to — stash it for post-match use.
            _stash_trailing_chat(match, _drain_pending_chat(match))
        session.commit()

    # Inline memory — fire-and-forget, survives client disconnect.
    if soul_resp.save_memory:
        asyncio.create_task(
            save_inline_memory(
                soul_resp.save_memory,
                character_id=character_id_local,
                player_id=player_id_local,
                match_id=match_id_local,
            )
        )

    await emitters.on_agent_chat(soul_resp)
    await emitters.on_mood_update(new_smoothed)

    if end_info is not None:
        reason, result_value, outcome = end_info
        await emitters.on_match_ended(reason, result_value, outcome)
        await emitters.on_post_match_kickoff()
