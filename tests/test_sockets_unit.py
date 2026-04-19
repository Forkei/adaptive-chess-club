"""Unit tests for Phase 3b primitives that don't need a real Socket.IO server.

Covers:
- Disconnect cooldown registry (start / cancel / timeout)
- Pending-chat FIFO buffer (message + char caps)
- Processor `status_callback` is invoked at each step boundary
"""

from __future__ import annotations

import asyncio

import pytest

from app.db import SessionLocal
from app.models.character import Character, CharacterState
from app.models.match import Color, Match, MatchStatus, Player
from app.sockets import disconnect as disconnect_registry
from app.sockets.server import _append_pending_chat
from app.sockets.events import match_room_name


def _mk_player(sess, username="p"):
    p = Player(username=username, display_name=username)
    sess.add(p)
    sess.flush()
    return p


def _mk_character(sess):
    c = Character(
        name="U", aggression=5, risk_tolerance=5, patience=5, trash_talk=5,
        target_elo=1400, current_elo=1400, floor_elo=1400, max_elo=1800,
        adaptive=True, state=CharacterState.READY,
    )
    sess.add(c); sess.flush()
    return c


def _mk_match(sess, player, character):
    import chess
    m = Match(
        character_id=character.id, player_id=player.id,
        player_color=Color.WHITE, status=MatchStatus.IN_PROGRESS,
        initial_fen=chess.STARTING_FEN, current_fen=chess.STARTING_FEN,
        move_count=0, character_elo_at_start=character.current_elo,
    )
    sess.add(m); sess.flush()
    return m


# --- Room naming ----------------------------------------------------------


def test_match_room_name_is_deterministic():
    assert match_room_name("abc") == "match:abc"
    assert match_room_name("abc") == match_room_name("abc")


# --- Disconnect cooldown registry -----------------------------------------


@pytest.mark.asyncio
async def test_cooldown_cancel_prevents_timeout_handler():
    calls: list[str] = []

    async def _on_timeout(match_id: str):
        calls.append(match_id)

    disconnect_registry.start(
        "match-a", player_id="p1", on_timeout=_on_timeout, seconds=5,
    )
    assert disconnect_registry.is_active("match-a")

    cancelled = disconnect_registry.cancel("match-a")
    assert cancelled is True
    assert not disconnect_registry.is_active("match-a")

    # Wait past the original timeout; handler must not have fired.
    await asyncio.sleep(0.05)
    assert calls == []


@pytest.mark.asyncio
async def test_cooldown_timeout_fires_handler_once():
    calls: list[str] = []

    async def _on_timeout(match_id: str):
        calls.append(match_id)

    disconnect_registry.start(
        "match-b", player_id="p1", on_timeout=_on_timeout, seconds=0,
    )
    # Give the event loop a tick to run the 0-second sleep.
    await asyncio.sleep(0.05)
    assert calls == ["match-b"]
    assert not disconnect_registry.is_active("match-b")


@pytest.mark.asyncio
async def test_cooldown_restart_replaces_existing():
    calls: list[str] = []

    async def _on_timeout(match_id: str):
        calls.append(match_id)

    disconnect_registry.start(
        "match-c", player_id="p1", on_timeout=_on_timeout, seconds=5,
    )
    disconnect_registry.start(
        "match-c", player_id="p1", on_timeout=_on_timeout, seconds=0,
    )
    await asyncio.sleep(0.05)
    # Only the second (0-second) fires; the first was replaced.
    assert calls == ["match-c"]


@pytest.fixture(autouse=True)
def _reset_registry():
    disconnect_registry.reset_all()
    yield
    disconnect_registry.reset_all()


# --- Pending chat buffer --------------------------------------------------


def test_pending_chat_fifo_eviction_by_message_count(monkeypatch):
    monkeypatch.setattr(
        "app.sockets.server.get_settings",
        lambda: type("S", (), {
            "player_chat_min_interval_ms": 500,
            "pending_chat_max_messages": 3,
            "pending_chat_max_chars": 10_000,
        })(),
    )
    with SessionLocal() as sess:
        p = _mk_player(sess, "pchat1")
        c = _mk_character(sess)
        m = _mk_match(sess, p, c)

        _append_pending_chat(m, "a")
        _append_pending_chat(m, "b")
        _append_pending_chat(m, "c")
        _append_pending_chat(m, "d")

        buf = m.extra_state["pending_player_chat"]
        assert [e["text"] for e in buf] == ["b", "c", "d"]


def test_pending_chat_fifo_eviction_by_total_chars(monkeypatch):
    monkeypatch.setattr(
        "app.sockets.server.get_settings",
        lambda: type("S", (), {
            "player_chat_min_interval_ms": 500,
            "pending_chat_max_messages": 100,
            "pending_chat_max_chars": 10,
        })(),
    )
    with SessionLocal() as sess:
        p = _mk_player(sess, "pchat2")
        c = _mk_character(sess)
        m = _mk_match(sess, p, c)

        _append_pending_chat(m, "hello")  # 5
        _append_pending_chat(m, "world")  # 10 total
        _append_pending_chat(m, "!")       # forces eviction: 11 -> drop "hello"

        buf = m.extra_state["pending_player_chat"]
        texts = [e["text"] for e in buf]
        assert "hello" not in texts
        total = sum(len(e["text"]) for e in buf)
        assert total <= 10


# --- Pending chat persistence on player move ------------------------------


def test_pending_chat_persists_to_player_move_on_drain():
    """Messages typed before a move should land on Move.player_chat_before.

    Simulates the Phase 1 flow without spinning up the socket: calls the
    streaming module's drain + merge helpers directly, then runs a player
    move and asserts the Move row carries the merged chat.
    """
    import asyncio
    import chess

    from app.db import SessionLocal
    from app.matches import streaming as stream
    from app.matches import service as match_service
    from app.matches.streaming import TurnEmitters

    with SessionLocal() as sess:
        p = _mk_player(sess, "pcp1")
        c = _mk_character(sess)
        m = _mk_match(sess, p, c)
        # Two messages typed while waiting for the engine (or just before
        # making a move).
        _append_pending_chat(m, "are you even trying")
        _append_pending_chat(m, "here goes e4")
        sess.commit()
        match_id = m.id

    async def _go():
        noop = lambda *_a, **_k: None

        async def _async_noop(*_a, **_k):
            return None

        emitters = TurnEmitters(
            on_player_move=_async_noop, on_thinking=_async_noop,
            on_memory_surfaced=_async_noop, on_agent_move=_async_noop,
            on_agent_chat=_async_noop, on_mood_update=_async_noop,
            on_match_ended=_async_noop, on_post_match_kickoff=_async_noop,
        )
        # Only run Phase 1 — player move — then stop. Easiest way: force the
        # engine path to fail fast. We do this by letting phase-2 run against
        # the mock engine (always available in tests) and just checking the
        # player Move row state after commit.
        try:
            await stream.apply_player_move_streamed(
                match_id=match_id, uci="e2e4", player_chat=None, emitters=emitters,
            )
        except Exception:
            # Phase 2 may fail in unit-test environments without engines — we
            # only care about Phase 1's persistence side-effect here.
            pass

    asyncio.run(_go())

    with SessionLocal() as sess:
        match = sess.get(Match, match_id)
        assert match is not None
        # Find the player's move (side=WHITE, move_number=1).
        player_move = next(mv for mv in match.moves if mv.move_number == 1)
        # Both messages should be on the player move's player_chat_before.
        assert player_move.player_chat_before is not None
        assert "are you even trying" in player_move.player_chat_before
        assert "here goes e4" in player_move.player_chat_before
        # Separator is " / ".
        assert " / " in player_move.player_chat_before
        # Pending buffer was drained.
        pending = (match.extra_state or {}).get("pending_player_chat", [])
        assert pending == []


def test_pending_chat_stashed_to_trailing_on_resign():
    """Resign mid-thinking should preserve un-persisted chat to trailing_player_chat."""
    with SessionLocal() as sess:
        p = _mk_player(sess, "pcp2")
        c = _mk_character(sess)
        m = _mk_match(sess, p, c)
        _append_pending_chat(m, "gg")
        sess.commit()
        match_id = m.id

    from app.matches import service as match_service

    with SessionLocal() as sess:
        match_service.resign(sess, match_id=match_id)
        sess.commit()

    with SessionLocal() as sess:
        match = sess.get(Match, match_id)
        assert match is not None
        pending = (match.extra_state or {}).get("pending_player_chat", [])
        trailing = (match.extra_state or {}).get("trailing_player_chat", [])
        assert pending == []
        assert any(e.get("text") == "gg" for e in trailing)


# --- Processor status callback --------------------------------------------


def test_processor_invokes_status_callback_between_steps(monkeypatch):
    """Run the processor with LLM + engine steps disabled; verify the callback
    fires `step_started`/`step_completed` + `pipeline_completed`.
    """
    from app.post_match.processor import (
        ALL_STEPS,
        ProcessorConfig,
        process_match_post_game,
    )
    import chess
    from app.models.match import MatchResult

    # Seed a completed match with no moves (keeps feature extraction cheap).
    with SessionLocal() as sess:
        char = _mk_character(sess)
        p = _mk_player(sess, "pcb")
        m = Match(
            character_id=char.id, player_id=p.id,
            player_color=Color.WHITE, status=MatchStatus.COMPLETED,
            result=MatchResult.WHITE_WIN,
            initial_fen=chess.STARTING_FEN, current_fen=chess.STARTING_FEN,
            move_count=0, character_elo_at_start=char.current_elo,
        )
        sess.add(m); sess.commit()
        match_id = m.id

    events: list[tuple[str, dict]] = []
    def _cb(event: str, payload: dict) -> None:
        events.append((event, payload))

    process_match_post_game(
        match_id,
        config=ProcessorConfig(run_engine_analysis=False, run_llm_steps=False),
        status_callback=_cb,
    )

    # At minimum we expect pipeline_completed + step_started/step_completed for
    # every non-LLM step (engine_analysis, feature_extraction, elo_ratchet).
    names = [e[0] for e in events]
    assert "pipeline_completed" in names
    # Step_started/completed pairs fired for the three non-LLM steps.
    started = [e[1].get("current_step") for e in events if e[0] == "step_started"]
    completed_steps_last = events[-1][1].get("steps_completed", [])
    assert "engine_analysis" in started
    assert "feature_extraction" in started
    assert "elo_ratchet" in started
    # pipeline_completed payload contains the full step list.
    assert set(ALL_STEPS).issubset(set(completed_steps_last))
