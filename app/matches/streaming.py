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
from app.engine.base import EngineConfig, MoveResult
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


def _finalize_outcome(match: Match, board: chess.Board) -> tuple[str, str, str | None] | None:
    """If the board terminated, set status/result on the match and return (reason, result, outcome).

    Does NOT commit. Caller owns the session.
    """
    if match.status != MatchStatus.IN_PROGRESS:
        # Already ended via resign or disconnect-timeout.
        if match.status == MatchStatus.RESIGNED:
            result_value = match.result.value if match.result else "abandoned"
            return "resign", result_value, "resigned"
        if match.status == MatchStatus.ABANDONED:
            return "disconnect_timeout", "abandoned", "resigned"
        return None

    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        return None

    match.status = MatchStatus.COMPLETED
    match.ended_at = datetime.utcnow()
    match.character_elo_at_end = match.character_elo_at_start

    if outcome.winner is None:
        match.result = MatchResult.DRAW
        reason = "stalemate" if outcome.termination.name == "STALEMATE" else "draw_rule"
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


def _load_last_player_context(session, match_id: str) -> tuple[str | None, str | None]:
    rows = list(
        session.execute(
            select(Move)
            .where(Move.match_id == match_id)
            .order_by(Move.move_number.desc())
            .limit(6)
        ).scalars()
    )
    for mv in rows:
        if mv.player_chat_before is not None or mv.agent_chat_after is None:
            return mv.uci, mv.player_chat_before
    return None, None


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


# --- Player move persistence ----------------------------------------------


def _persist_player_move(session, match: Match, uci: str, player_chat: str | None) -> Move:
    board = chess.Board(match.current_fen)
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
        player_row = _persist_player_move(session, match, uci, player_chat)
        board_after_player = chess.Board(match.current_fen)
        end_info = _finalize_outcome(match, board_after_player)
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

        board = chess.Board(match.current_fen)

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
        recent_chat = _load_last_chat_lines(session, match.id)

        # Merge pending mid-think chat into recent_chat, then clear the buffer.
        pending = _drain_pending_chat(match)
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

    # --- Phase 3: persist engine move + emit agent_move ----------------
    with SessionLocal() as session:
        match = session.get(Match, match_id_local)
        assert match is not None
        board_before = chess.Board(match.current_fen)
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
        )
        return run_soul(character, soul_input)

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
        board_after_engine = chess.Board(match.current_fen)
        end_info = _finalize_outcome(match, board_after_engine)
        session.commit()

    await emitters.on_agent_chat(soul_resp)
    await emitters.on_mood_update(new_smoothed)

    if end_info is not None:
        reason, result_value, outcome = end_info
        await emitters.on_match_ended(reason, result_value, outcome)
        await emitters.on_post_match_kickoff()
