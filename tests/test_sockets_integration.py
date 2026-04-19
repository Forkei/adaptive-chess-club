"""Socket.IO integration tests for Phase 3b.

Spins up a uvicorn instance bound to localhost on an ephemeral port and drives
it through a real `socketio.AsyncClient`. The engines are forced to MockEngine
(deterministic); the Subconscious + Soul are stubbed so tests don't need an LLM.

Covers the three smoke-test scenarios from section M of the phase brief:
  M.5 — agent turn event ordering (memory_surfaced before agent_move)
  M.6 — disconnect → reconnect within window → match resumes
  M.7 — disconnect → timeout → match abandoned → post-match runs
  M.8 — player_chat while agent is thinking flows into next Subconscious call
"""

from __future__ import annotations

import asyncio
import contextlib
import socket as _socket
import time
from typing import Any

import pytest
import pytest_asyncio
import socketio
import uvicorn

from app.agents.subconscious import clear_cache as _clear_sub_cache
from app.db import SessionLocal
from app.engine.registry import reset_engines_for_testing
from app.models.character import Character, CharacterState
from app.models.match import Color, Match, MatchStatus, Player
from app.redis_client import reset_memory_store_for_testing
from app.schemas.agents import MoodDeltas, SoulResponse, SurfacedMemory
from app.sockets import disconnect as disconnect_registry


# --- Fixtures --------------------------------------------------------------


def _free_port() -> int:
    with _socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(autouse=True)
def _reset_infra():
    reset_engines_for_testing()
    reset_memory_store_for_testing()
    _clear_sub_cache()
    disconnect_registry.reset_all()
    yield
    disconnect_registry.reset_all()
    _clear_sub_cache()
    reset_memory_store_for_testing()
    reset_engines_for_testing()


@pytest.fixture
def _short_cooldown(monkeypatch):
    """Make disconnect cooldown 2 seconds so tests don't wait 5 minutes."""
    from app.config import get_settings

    settings = get_settings()
    monkeypatch.setattr(settings, "match_disconnect_cooldown_seconds", 2)
    # Also cut the chat rate limit so rapid test chats aren't throttled.
    monkeypatch.setattr(settings, "player_chat_min_interval_ms", 0)
    yield


@pytest.fixture
def _force_mock_engine(monkeypatch):
    import app.engine.registry as reg

    def _fake_available() -> list:
        reg._build_default_factories()
        return ["mock"]

    monkeypatch.setattr("app.matches.service.available_engines", _fake_available)
    monkeypatch.setattr("app.matches.streaming.available_engines", _fake_available)
    yield


@pytest.fixture
def _stub_agents(monkeypatch):
    """Replace Subconscious + Soul so tests don't require an LLM."""

    captured: dict[str, Any] = {"subconscious_inputs": [], "soul_inputs": []}

    def _fake_subconscious(session, character, inp, llm=None):
        captured["subconscious_inputs"].append(inp)
        # Return one canned memory so memory_surfaced has content.
        return [
            SurfacedMemory(
                memory_id="mem-1",
                narrative_text="A canned memory.",
                triggers=["t"],
                relevance_tags=["tag"],
                emotional_valence=0.1,
                scope="character_lore",
                score=0.9,
                retrieval_reason="for testing",
                from_cache=False,
            )
        ]

    def _fake_soul(character, inp):
        captured["soul_inputs"].append(inp)
        return SoulResponse(
            speak=f"hello from {character.name}",
            emotion="focused",
            emotion_intensity=0.4,
            mood_deltas=MoodDeltas(),
            referenced_memory_ids=["mem-1"],
        )

    monkeypatch.setattr("app.matches.streaming.run_subconscious", _fake_subconscious)
    monkeypatch.setattr("app.matches.streaming.run_soul", _fake_soul)
    yield captured


@pytest_asyncio.fixture
async def live_server(_short_cooldown, _force_mock_engine, _stub_agents):
    """Start uvicorn in-process against the combined ASGI app."""
    from app.main import app as asgi_app

    port = _free_port()
    config = uvicorn.Config(
        asgi_app, host="127.0.0.1", port=port, log_level="warning", lifespan="on",
    )
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve())

    # Wait until the server is actually accepting connections.
    for _ in range(200):
        if getattr(server, "started", False):
            break
        await asyncio.sleep(0.02)
    else:
        raise RuntimeError("uvicorn never started")

    yield port

    server.should_exit = True
    with contextlib.suppress(Exception):
        await asyncio.wait_for(task, timeout=5.0)


# --- Helpers ---------------------------------------------------------------


def _seed_match(player_color: str = "white") -> tuple[str, str]:
    """Return (player_id, match_id) for a freshly seeded match."""
    import chess
    with SessionLocal() as sess:
        char = Character(
            name="Sock Char", aggression=5, risk_tolerance=5, patience=5, trash_talk=5,
            target_elo=1400, current_elo=1400, floor_elo=1400, max_elo=1800,
            adaptive=True, state=CharacterState.READY,
        )
        sess.add(char)
        p = Player(username="ws_tester", display_name="Sock")
        sess.add(p)
        sess.flush()

        m = Match(
            character_id=char.id, player_id=p.id,
            player_color=Color.WHITE if player_color == "white" else Color.BLACK,
            status=MatchStatus.IN_PROGRESS,
            initial_fen=chess.STARTING_FEN, current_fen=chess.STARTING_FEN,
            move_count=0, character_elo_at_start=char.current_elo,
        )
        sess.add(m); sess.commit()
        return p.id, m.id


class _EventRecorder:
    """Attaches to an AsyncClient and captures every server→client event with a timestamp.

    `wait(name)` consumes one event of that name; calling it twice waits for two distinct
    events (backed by a per-name asyncio.Queue). `names_in_order()` returns the full
    chronological trace across all event types — used for ordering assertions.
    """

    EVENT_NAMES = (
        "match_state", "player_move_applied", "agent_thinking",
        "memory_surfaced", "agent_move", "agent_chat", "mood_update",
        "match_ended", "post_match_status", "post_match_complete",
        "match_paused", "match_resumed", "player_chat_echoed",
        "player_chat_rate_limited", "pong", "error",
    )

    def __init__(self, client: socketio.AsyncClient):
        self.client = client
        self.events: list[tuple[float, str, Any]] = []
        self._queues: dict[str, asyncio.Queue] = {n: asyncio.Queue() for n in self.EVENT_NAMES}
        for name in self.EVENT_NAMES:
            client.on(name, self._mk_handler(name), namespace="/play")

    def _mk_handler(self, name: str):
        async def _h(data):
            self.events.append((time.monotonic(), name, data))
            await self._queues[name].put(data)
        return _h

    async def wait(self, name: str, timeout: float = 10.0):
        return await asyncio.wait_for(self._queues[name].get(), timeout=timeout)

    def names_in_order(self) -> list[str]:
        return [n for _ts, n, _d in self.events]

    def count(self, name: str) -> int:
        return sum(1 for _ts, n, _d in self.events if n == name)


async def _connect(port: int, player_id: str, match_id: str) -> tuple[socketio.AsyncClient, _EventRecorder]:
    client = socketio.AsyncClient(reconnection=False)
    rec = _EventRecorder(client)
    # Pass the cookie via headers. python-socketio's asyncio client accepts headers= in connect().
    await client.connect(
        f"http://127.0.0.1:{port}",
        namespaces=["/play"],
        socketio_path="socket.io",
        headers={"Cookie": f"player_id={player_id}"},
        auth={"match_id": match_id},
        wait=True,
    )
    return client, rec


# --- Tests ----------------------------------------------------------------


@pytest.mark.asyncio
async def test_connect_emits_match_state(live_server):
    port = live_server
    player_id, match_id = _seed_match(player_color="white")

    client, rec = await _connect(port, player_id, match_id)
    try:
        state = await rec.wait("match_state", timeout=5.0)
        assert state["match_id"] == match_id
        assert state["status"] == "in_progress"
        assert state["player_color"] == "white"
    finally:
        await client.disconnect()


@pytest.mark.asyncio
async def test_make_move_event_ordering(live_server):
    """M.5 smoke: memory_surfaced must fire before agent_move in a character turn."""
    port = live_server
    player_id, match_id = _seed_match(player_color="white")

    client, rec = await _connect(port, player_id, match_id)
    try:
        await rec.wait("match_state", timeout=5.0)
        # Player plays e2-e4 (MockEngine will respond with the first legal move).
        await client.emit("make_move", {"uci": "e2e4"}, namespace="/play")

        await rec.wait("agent_thinking", timeout=5.0)
        await rec.wait("memory_surfaced", timeout=5.0)
        await rec.wait("agent_move", timeout=10.0)
        await rec.wait("agent_chat", timeout=10.0)

        names = rec.names_in_order()
        i_think = names.index("agent_thinking")
        i_mem = names.index("memory_surfaced")
        i_move = names.index("agent_move")
        i_chat = names.index("agent_chat")
        assert i_think < i_mem < i_move < i_chat, (
            f"Unexpected event order: {names}"
        )
    finally:
        await client.disconnect()


@pytest.mark.asyncio
async def test_disconnect_reconnect_within_window_resumes(live_server):
    """M.6 smoke: drop the socket, reconnect within cooldown → match_resumed + still in_progress."""
    port = live_server
    player_id, match_id = _seed_match(player_color="white")

    client1, rec1 = await _connect(port, player_id, match_id)
    await rec1.wait("match_state", timeout=5.0)
    await client1.disconnect()

    # Give the disconnect handler a tick to stamp the match + arm cooldown.
    await asyncio.sleep(0.1)

    # Reconnect well within the 2-second cooldown.
    client2, rec2 = await _connect(port, player_id, match_id)
    try:
        state = await rec2.wait("match_state", timeout=5.0)
        assert state["status"] == "in_progress"
        # match_resumed fires on the room (and reaches the new socket).
        await rec2.wait("match_resumed", timeout=2.0)

        # The match's extra_state should have its disconnect markers cleared.
        with SessionLocal() as s:
            m = s.get(Match, match_id)
            assert "disconnect_started_at" not in (m.extra_state or {})
    finally:
        await client2.disconnect()


@pytest.mark.asyncio
async def test_disconnect_timeout_abandons_match_and_runs_post_match(live_server, monkeypatch):
    """M.7 smoke: drop socket, wait past cooldown → match ABANDONED and post-match runs."""
    # Patch the processor launcher to capture kickoff without actually running LLM-backed steps.
    kickoffs: list[str] = []
    real_start = None

    from app.post_match import processor as proc_module

    def _fake_start(match_id, **kwargs):
        kickoffs.append(match_id)
        # Run a no-op config synchronously so we can assert the callback wiring
        # without background thread races in the test.
        status_cb = kwargs.get("status_callback")
        cfg = proc_module.ProcessorConfig(run_engine_analysis=False, run_llm_steps=False)
        proc_module.process_match_post_game(match_id, config=cfg, status_callback=status_cb)
        import threading
        t = threading.Thread(target=lambda: None, daemon=True)
        t.start()
        return t

    monkeypatch.setattr("app.post_match.processor.start_post_match_background", _fake_start)

    port = live_server
    player_id, match_id = _seed_match(player_color="white")

    client, rec = await _connect(port, player_id, match_id)
    await rec.wait("match_state", timeout=5.0)
    await client.disconnect()

    # Wait out the 2-second cooldown + a small buffer.
    await asyncio.sleep(3.0)

    assert kickoffs == [match_id]
    with SessionLocal() as s:
        m = s.get(Match, match_id)
        assert m.status == MatchStatus.ABANDONED


@pytest.mark.asyncio
async def test_chat_during_thinking_flows_into_next_subconscious_call(live_server, _stub_agents):
    """M.8 smoke: player chats while agent is thinking; the NEXT Subconscious input
    must include that text in its recent_chat context."""
    port = live_server
    player_id, match_id = _seed_match(player_color="white")

    captured = _stub_agents

    client, rec = await _connect(port, player_id, match_id)
    try:
        await rec.wait("match_state", timeout=5.0)

        # Kick off a turn.
        await client.emit("make_move", {"uci": "e2e4"}, namespace="/play")
        await rec.wait("agent_thinking", timeout=5.0)

        # Fire a chat message RIGHT after thinking begins, well before agent_move.
        await client.emit("player_chat", {"text": "I'm going to crush you this game"}, namespace="/play")
        await rec.wait("player_chat_echoed", timeout=5.0)

        # Finish this turn so the chat buffer gets drained.
        await rec.wait("agent_move", timeout=10.0)
        await rec.wait("agent_chat", timeout=10.0)
        # Kick off another turn so Subconscious runs again with the pending chat merged.
        # Pick any legal second move — e7e5 would be black's move, but MockEngine already
        # responded as black, so we play a white move. Whichever move Mock picked will
        # change the legality; we just try a few common responses.
        # Simpler: read current FEN and make any legal move.
        import chess as _chess
        with SessionLocal() as s:
            m = s.get(Match, match_id)
            board = _chess.Board(m.current_fen)
            mv = next(iter(board.legal_moves))
            uci = mv.uci()
        await client.emit("make_move", {"uci": uci}, namespace="/play")
        await rec.wait("agent_move", timeout=10.0)

        # Second Subconscious call should have seen our pending chat.
        assert len(captured["subconscious_inputs"]) >= 2
        second_recent = captured["subconscious_inputs"][1].recent_chat
        flattened = " ".join(second_recent)
        assert "going to crush you" in flattened
    finally:
        await client.disconnect()


@pytest.mark.live
@pytest.mark.asyncio
async def test_live_match_emits_memory_surfaced_before_agent_move(live_server):
    """Opt-in live test (requires RUN_LIVE_LLM_TESTS=1 + GEMINI_API_KEY + real engines).

    Confirms the headline UX property of Phase 3b: memory_surfaced arrives before
    agent_move. The deterministic `test_make_move_event_ordering` covers this with
    mocked agents; this variant exercises real Gemini + Maia-2/Stockfish.
    """
    import os
    if not os.environ.get("RUN_LIVE_LLM_TESTS"):
        pytest.skip("Set RUN_LIVE_LLM_TESTS=1 to run live Socket.IO match tests")
    if not os.environ.get("GEMINI_API_KEY"):
        pytest.skip("GEMINI_API_KEY not set")

    port = live_server
    player_id, match_id = _seed_match(player_color="white")
    client, rec = await _connect(port, player_id, match_id)
    try:
        await rec.wait("match_state", timeout=5.0)
        await client.emit("make_move", {"uci": "e2e4"}, namespace="/play")
        await rec.wait("agent_thinking", timeout=5.0)
        await rec.wait("memory_surfaced", timeout=30.0)
        await rec.wait("agent_move", timeout=60.0)

        names = rec.names_in_order()
        assert names.index("memory_surfaced") < names.index("agent_move"), names
    finally:
        await client.disconnect()


@pytest.mark.asyncio
async def test_chat_rate_limit_rejects_rapid_bursts(live_server, monkeypatch):
    from app.config import get_settings
    monkeypatch.setattr(get_settings(), "player_chat_min_interval_ms", 500)

    port = live_server
    player_id, match_id = _seed_match(player_color="white")

    client, rec = await _connect(port, player_id, match_id)
    try:
        await rec.wait("match_state", timeout=5.0)
        await client.emit("player_chat", {"text": "one"}, namespace="/play")
        await rec.wait("player_chat_echoed", timeout=5.0)
        await client.emit("player_chat", {"text": "two"}, namespace="/play")
        await rec.wait("player_chat_rate_limited", timeout=5.0)
    finally:
        await client.disconnect()
