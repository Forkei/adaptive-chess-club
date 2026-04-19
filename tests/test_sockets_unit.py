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
