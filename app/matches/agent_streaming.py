"""Block 13 Commit 5 — server-side match loop for agent_vs_character.

Both sides are fully automated:
  - Agent's turns  (match.player_color side): engine + agent Soul.
  - Kenji's turns  (character side): existing _run_engine_and_agents pipeline.

Events flow to the /play Socket.IO room so the player can watch in real time.

Launched via asyncio.create_task from room_server.py after create_agent_match.
"""

from __future__ import annotations

import asyncio
import logging
import types

import chess
from sqlalchemy import select

from app.agents.soul import SoulInput, _fallback_response, run_agent_soul_for_room
from app.agents.subconscious import SubconsciousInput, run_subconscious
from app.config import get_settings
from app.db import SessionLocal
from app.director.director import MatchContext, choose_engine_config
from app.director.mood import (
    MoodState,
    apply_deltas,
    initial_mood_for_character,
    load_mood,
    save_mood,
    smooth_mood,
)
from app.engine import EngineUnavailable, available_engines, get_engine
from app.engine.board_abstraction import board_to_english
from app.matches import service as _svc
from app.matches.streaming import (
    TurnEmitters,
    _finalize_outcome,
    _load_last_chat_lines,
    _persist_engine_move,
    _run_engine_and_agents,
)
from app.models.match import Color, Match, MatchStatus, Move
from app.schemas.agents import SoulResponse

logger = logging.getLogger(__name__)

INTER_TURN_DELAY_S = 1.2  # pause between automated turns for UI readability
MAX_TURNS = 300            # safety cap — no game is longer than 150 moves per side


# --- Agent mood key ----------------------------------------------------------

def _agent_mood_key(match_id: str) -> str:
    """Separate mood namespace for the agent side to avoid clobbering Kenji's mood."""
    return f"{match_id}:agent"


def _load_or_init_agent_mood(agent_elo: int, match_id: str) -> tuple[MoodState, MoodState]:
    key = _agent_mood_key(match_id)
    smoothed = load_mood(key, smoothed=True)
    raw = load_mood(key, smoothed=False)
    if smoothed is None or raw is None:
        initial = MoodState()  # neutral baseline for agents
        save_mood(key, initial, smoothed=True)
        save_mood(key, initial, smoothed=False)
        return initial, initial
    return raw, smoothed


# --- Agent engine turn -------------------------------------------------------


async def _run_agent_engine_turn(
    *,
    match_id: str,
    agent_id: str,
    emitters: TurnEmitters,
    settings,
) -> bool:
    """Run one engine+Soul turn for the agent side.

    Returns True when the game ended after this move.
    Emits: on_thinking → on_player_move → (on_agent_side_chat if speaks).
    """
    with SessionLocal() as session:
        match = session.get(Match, match_id)
        if match is None or match.status != MatchStatus.IN_PROGRESS:
            return True

        from app.models.player_agent import PlayerAgent
        agent = session.get(PlayerAgent, agent_id)
        if agent is None:
            logger.warning("[agent_loop] agent %s not found, aborting loop", agent_id)
            return True

        raw, smoothed = _load_or_init_agent_mood(agent.elo, match_id)
        smoothed = smooth_mood(smoothed, raw)
        save_mood(_agent_mood_key(match_id), smoothed, smoothed=True)
        save_mood(_agent_mood_key(match_id), raw, smoothed=False)

        board = _svc.board_with_history(match, session)

        available = frozenset(available_engines())
        if not available:
            raise EngineUnavailable("No engines available")

        # Build an engine config for the agent's Elo.
        agent_proxy = types.SimpleNamespace(
            current_elo=agent.elo,
            floor_elo=max(800, agent.elo - 200),
            max_elo=agent.elo + 200,
            patience=5,
            aggression=5,
            adaptive=False,
        )
        context = MatchContext(
            move_number=board.fullmove_number,
            game_phase="opening" if board.fullmove_number < 10 else "middlegame",
            player_color=match.player_color.value,
            engines_available=available,
        )
        config = choose_engine_config(
            character=agent_proxy,  # type: ignore[arg-type]
            mood=smoothed,
            match_context=context,
        )
        engine = get_engine(config.engine_name)

        # Pre-engine board summary for Subconscious.
        pre_engine_summary = board_to_english(board, eval_cp=None)
        recent_chat = _load_last_chat_lines(session, match_id)
        last_player_moves = list(
            session.execute(
                select(Move)
                .where(Move.match_id == match_id)
                .order_by(Move.move_number.desc())
                .limit(6)
            ).scalars()
        )
        last_moves_san = [mv.san for mv in reversed(last_player_moves) if mv.san]

        match_id_local = match.id
        player_id_local = match.player_id
        mood_snapshot = smoothed.to_dict()

    # Emit thinking.
    eta = float(config.time_budget_seconds) + 3.0
    await emitters.on_thinking(eta)

    # Run Subconscious + engine concurrently.
    def _sub_sync():
        with SessionLocal() as s:
            from app.models.player_agent import PlayerAgent as _PA
            ag = s.get(_PA, agent_id)
            if ag is None:
                return []
            char_proxy = types.SimpleNamespace(name=ag.name)
            sub_input = SubconsciousInput(
                character_id="",  # ignored when agent_id is set
                agent_id=agent_id,
                match_id=match_id_local,
                current_turn=0,
                board_summary=pre_engine_summary,
                mood=smoothed,
                last_player_uci=None,
                last_player_chat=None,
                last_moves_san=last_moves_san,
                recent_chat=recent_chat,
                opening_label=None,
                current_player_id=player_id_local,
            )
            return run_subconscious(s, char_proxy, sub_input) or []  # type: ignore[arg-type]

    def _engine_sync():
        with SessionLocal() as s:
            match_row = s.get(Match, match_id_local)
            if match_row is None:
                return None
            b = _svc.board_with_history(match_row, s)
            return engine.get_move(b, config)

    surfaced_task = asyncio.create_task(asyncio.to_thread(_sub_sync))
    engine_task = asyncio.create_task(asyncio.to_thread(_engine_sync))

    surfaced, engine_result = await asyncio.gather(surfaced_task, engine_task)

    if engine_result is None:
        logger.warning("[agent_loop] engine returned None for match=%s", match_id)
        return False

    # --- Persist agent's move as the "player" move row ----------------------
    game_ended = False
    agent_move_row: Move | None = None
    with SessionLocal() as session:
        match = session.get(Match, match_id_local)
        if match is None or match.status != MatchStatus.IN_PROGRESS:
            return True

        board = _svc.board_with_history(match, session)
        agent_move_row = _persist_engine_move(
            session, match, engine_result, board, mood_snapshot=mood_snapshot
        )
        # Mark surface IDs.
        agent_move_row.surfaced_memory_ids = [m.memory_id for m in surfaced]
        board_after = _svc.board_with_history(match, session)
        end_info = _finalize_outcome(match, board_after)
        if end_info is not None:
            game_ended = True
        session.commit()
        session.refresh(agent_move_row)

    # Emit player_move_applied (agent IS the "player" in the match model).
    await emitters.on_player_move(agent_move_row)

    if game_ended:
        reason, result_value, outcome = end_info
        await emitters.on_match_ended(reason, result_value, outcome)
        await emitters.on_post_match_kickoff()
        return True

    # --- Agent Soul: generate chat -------------------------------------------
    def _soul_sync():
        with SessionLocal() as s:
            from app.models.player_agent import PlayerAgent as _PA
            ag = s.get(_PA, agent_id)
            if ag is None:
                return _fallback_response()
            from app.agents.prompts import build_agent_system_prompt
            system = build_agent_system_prompt(ag)
            # Derive SAN for the engine move from the pre-move board.
            try:
                with SessionLocal() as s2:
                    m = s2.get(Match, match_id_local)
                    if m is None:
                        engine_san = engine_result.move
                    else:
                        b_pre = _svc.board_with_history(m, s2)
                        b_pre.pop()  # undo last move to get pre-move board
                        engine_san = b_pre.san(chess.Move.from_uci(engine_result.move))
            except Exception:
                engine_san = engine_result.move

            with SessionLocal() as s3:
                m3 = s3.get(Match, match_id_local)
                post_board = _svc.board_with_history(m3, s3) if m3 else None
                board_summary = board_to_english(post_board or chess.Board(), eval_cp=engine_result.eval_cp)

            soul_inp = SoulInput(
                board=board_summary,
                mood=smoothed,
                surfaced_memories=surfaced,
                recent_chat=recent_chat,
                engine_move_san=engine_san,
                engine_move_uci=engine_result.move,
                engine_eval_cp=engine_result.eval_cp,
                engine_considered=[cm.model_dump() for cm in engine_result.considered_moves],
                engine_time_ms=engine_result.time_taken_ms,
                move_number=agent_move_row.move_number,
                game_phase="opening" if agent_move_row.move_number < 20 else "middlegame",
                player_just_spoke=False,
                last_player_chat=None,
                match_id=match_id_local,
                character_color="white" if match.player_color == Color.WHITE else "black",
            )
            return run_agent_soul_for_room(system, soul_inp)

    try:
        soul_resp: SoulResponse = await asyncio.wait_for(
            asyncio.to_thread(_soul_sync), timeout=30.0
        )
    except Exception:
        logger.exception("[agent_loop] agent Soul failed for match=%s", match_id)
        soul_resp = _fallback_response()

    # Persist Soul output (chat, mood deltas) on the move row.
    if soul_resp.speak or soul_resp.mood_deltas:
        with SessionLocal() as session:
            row = session.get(Move, agent_move_row.id)
            if row is not None:
                row.agent_chat_after = soul_resp.speak
                deltas = soul_resp.mood_deltas.to_dict()
                new_raw = apply_deltas(raw, deltas)
                new_smoothed = smooth_mood(smoothed, new_raw)
                save_mood(_agent_mood_key(match_id), new_raw, smoothed=False)
                save_mood(_agent_mood_key(match_id), new_smoothed, smoothed=True)
                session.commit()

    if soul_resp.speak:
        await emitters.on_agent_chat(soul_resp)

    return False


# --- Main loop ---------------------------------------------------------------


async def run_agent_match_loop(match_id: str, agent_id: str) -> None:
    """Full server-side loop for agent_vs_character matches.

    Runs until the match ends, errors out, or MAX_TURNS is hit.
    Emits events to the /play Socket.IO room.
    """
    from app.sockets.server import _build_turn_emitters
    settings = get_settings()
    emitters = _build_turn_emitters(match_id)

    logger.info("[agent_loop] starting for match=%s agent=%s", match_id, agent_id)

    # If agent has white, they move first.
    with SessionLocal() as session:
        match = session.get(Match, match_id)
        if match is None:
            logger.warning("[agent_loop] match %s not found", match_id)
            return

    for turn_num in range(MAX_TURNS):
        await asyncio.sleep(INTER_TURN_DELAY_S)

        with SessionLocal() as session:
            match = session.get(Match, match_id)
            if match is None or match.status != MatchStatus.IN_PROGRESS:
                break
            board = _svc.board_with_history(match, session)
            agent_chess_color = chess.WHITE if match.player_color == Color.WHITE else chess.BLACK

        if board.turn == agent_chess_color:
            # Agent's turn.
            try:
                ended = await _run_agent_engine_turn(
                    match_id=match_id,
                    agent_id=agent_id,
                    emitters=emitters,
                    settings=settings,
                )
            except Exception:
                logger.exception("[agent_loop] agent turn crashed for match=%s", match_id)
                break
        else:
            # Kenji's turn — use existing character pipeline.
            try:
                await _run_engine_and_agents(
                    match_id=match_id, emitters=emitters, settings=settings
                )
            except Exception:
                logger.exception("[agent_loop] Kenji turn crashed for match=%s", match_id)
                break
            # Check if game ended (emitters.on_match_ended already fired if so).
            with SessionLocal() as session:
                match = session.get(Match, match_id)
                ended = match is None or match.status != MatchStatus.IN_PROGRESS

        if ended:
            break
    else:
        logger.warning("[agent_loop] hit MAX_TURNS (%d) for match=%s", MAX_TURNS, match_id)

    logger.info("[agent_loop] finished for match=%s after %d turns", match_id, turn_num + 1)
