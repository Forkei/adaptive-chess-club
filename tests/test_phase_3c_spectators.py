"""Socket.IO spectator integration tests (Phase 3c).

Spins up uvicorn on an ephemeral port (same pattern as
`test_sockets_integration.py`) and drives it with two real
`socketio.AsyncClient` instances — one participant, one spectator.

Verifies:
  - spectator joins via the /play namespace, receives match_state
  - spectator receives agent_move / agent_chat / memory_surfaced
  - spectator receives player_chat_broadcast when the participant sends player_chat
  - spectator_chat is broadcast to other sockets AND does NOT reach the Subconscious
  - participant attempting spectator_chat is rejected
  - spectator attempting make_move / resign / player_chat is rejected
  - spectator_count updates on join/leave
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
from app.models.character import Character, CharacterState, ContentRating, Visibility
from app.models.match import Color, Match, MatchStatus, Player
from app.redis_client import reset_memory_store_for_testing
from app.schemas.agents import MoodDeltas, SoulResponse, SurfacedMemory
from app.sockets import disconnect as disconnect_registry


# --- Fixtures (mirrors test_sockets_integration.py) -----------------------


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
    from app.config import get_settings
    s = get_settings()
    monkeypatch.setattr(s, "match_disconnect_cooldown_seconds", 2)
    monkeypatch.setattr(s, "player_chat_min_interval_ms", 0)
    yield


@pytest.fixture
def _force_mock_engine(monkeypatch):
    import app.engine.registry as reg
    def _fake_available() -> list:
        reg._build_default_factories(); return ["mock"]
    monkeypatch.setattr("app.matches.service.available_engines", _fake_available)
    monkeypatch.setattr("app.matches.streaming.available_engines", _fake_available)
    yield


@pytest.fixture
def _stub_agents(monkeypatch):
    captured: dict[str, Any] = {"subconscious_inputs": [], "soul_inputs": []}

    def _fake_subconscious(session, character, inp, llm=None):
        captured["subconscious_inputs"].append(inp)
        return [SurfacedMemory(
            memory_id="m1", narrative_text="memory", triggers=["t"], relevance_tags=["tag"],
            emotional_valence=0.1, scope="character_lore", score=0.9,
            retrieval_reason="test", from_cache=False,
        )]

    def _fake_soul(character, inp):
        captured["soul_inputs"].append(inp)
        return SoulResponse(
            speak="hi", emotion="focused", emotion_intensity=0.4,
            mood_deltas=MoodDeltas(), referenced_memory_ids=["m1"],
        )

    monkeypatch.setattr("app.matches.streaming.run_subconscious", _fake_subconscious)
    monkeypatch.setattr("app.matches.streaming.run_soul", _fake_soul)
    yield captured


@pytest_asyncio.fixture
async def live_server(_short_cooldown, _force_mock_engine, _stub_agents):
    from app.main import app as asgi_app
    port = _free_port()
    config = uvicorn.Config(asgi_app, host="127.0.0.1", port=port, log_level="warning", lifespan="on")
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve())
    for _ in range(200):
        if getattr(server, "started", False): break
        await asyncio.sleep(0.02)
    else:
        raise RuntimeError("uvicorn never started")
    yield port
    server.should_exit = True
    with contextlib.suppress(Exception):
        await asyncio.wait_for(task, timeout=5.0)


# --- Helpers --------------------------------------------------------------


def _seed_public_match() -> tuple[str, str, str]:
    """Returns (participant_id, spectator_id, match_id)."""
    import chess
    with SessionLocal() as sess:
        char = Character(
            name="SpecChar", aggression=5, risk_tolerance=5, patience=5, trash_talk=5,
            target_elo=1400, current_elo=1400, floor_elo=1400, max_elo=1800,
            adaptive=True, state=CharacterState.READY,
            content_rating=ContentRating.FAMILY, visibility=Visibility.PUBLIC,
        )
        sess.add(char)
        p1 = Player(username="participant", display_name="P1")
        p2 = Player(username="spectator", display_name="S1")
        sess.add_all([p1, p2]); sess.flush()
        m = Match(
            character_id=char.id, player_id=p1.id,
            player_color=Color.WHITE, status=MatchStatus.IN_PROGRESS,
            initial_fen=chess.STARTING_FEN, current_fen=chess.STARTING_FEN,
            move_count=0, character_elo_at_start=char.current_elo,
        )
        sess.add(m); sess.commit()
        return p1.id, p2.id, m.id


class _EventRecorder:
    EVENT_NAMES = (
        "match_state", "player_move_applied", "agent_thinking",
        "memory_surfaced", "agent_move", "agent_chat", "mood_update",
        "match_ended", "post_match_status", "post_match_complete",
        "match_paused", "match_resumed", "player_chat_echoed",
        "player_chat_rate_limited", "pong", "error",
        "spectator_chat_broadcast", "spectator_chat_echoed", "spectator_chat_rejected",
        "player_chat_broadcast", "spectator_joined", "spectator_left", "spectator_count",
    )

    def __init__(self, client: socketio.AsyncClient):
        self.client = client
        self.events: list[tuple[float, str, Any]] = []
        self._queues: dict[str, asyncio.Queue] = {n: asyncio.Queue() for n in self.EVENT_NAMES}
        for name in self.EVENT_NAMES:
            client.on(name, self._mk(name), namespace="/play")

    def _mk(self, name: str):
        async def _h(data):
            self.events.append((time.monotonic(), name, data))
            await self._queues[name].put(data)
        return _h

    async def wait(self, name: str, timeout: float = 10.0):
        return await asyncio.wait_for(self._queues[name].get(), timeout=timeout)

    def names(self) -> list[str]:
        return [n for _t, n, _d in self.events]


async def _connect(port: int, player_id: str, match_id: str) -> tuple[socketio.AsyncClient, _EventRecorder]:
    client = socketio.AsyncClient(reconnection=False)
    rec = _EventRecorder(client)
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
async def test_spectator_receives_match_state_and_events(live_server):
    port = live_server
    p_id, s_id, match_id = _seed_public_match()

    p_client, p_rec = await _connect(port, p_id, match_id)
    s_client, s_rec = await _connect(port, s_id, match_id)
    try:
        await s_rec.wait("match_state", timeout=5.0)
        await p_rec.wait("match_state", timeout=5.0)

        # Participant plays; spectator should see all the turn events.
        await p_client.emit("make_move", {"uci": "e2e4"}, namespace="/play")
        await s_rec.wait("player_move_applied", timeout=5.0)
        await s_rec.wait("memory_surfaced", timeout=5.0)
        await s_rec.wait("agent_move", timeout=10.0)
        await s_rec.wait("agent_chat", timeout=10.0)
    finally:
        await s_client.disconnect()
        await p_client.disconnect()


@pytest.mark.asyncio
async def test_spectator_rejected_from_participant_actions(live_server):
    port = live_server
    p_id, s_id, match_id = _seed_public_match()
    p_client, _ = await _connect(port, p_id, match_id)
    s_client, s_rec = await _connect(port, s_id, match_id)
    try:
        await s_rec.wait("match_state", timeout=5.0)
        await s_client.emit("make_move", {"uci": "e2e4"}, namespace="/play")
        err = await s_rec.wait("error", timeout=5.0)
        assert err["code"] == "spectator_cannot_move"

        await s_client.emit("player_chat", {"text": "hi"}, namespace="/play")
        err = await s_rec.wait("error", timeout=5.0)
        assert err["code"] == "spectator_cannot_player_chat"

        await s_client.emit("resign", namespace="/play")
        err = await s_rec.wait("error", timeout=5.0)
        assert err["code"] == "spectator_cannot_resign"
    finally:
        await s_client.disconnect()
        await p_client.disconnect()


@pytest.mark.asyncio
async def test_participant_rejected_from_spectator_chat(live_server):
    port = live_server
    p_id, _s_id, match_id = _seed_public_match()
    p_client, p_rec = await _connect(port, p_id, match_id)
    try:
        await p_rec.wait("match_state", timeout=5.0)
        await p_client.emit("spectator_chat", {"text": "sneaky"}, namespace="/play")
        rejected = await p_rec.wait("spectator_chat_rejected", timeout=5.0)
        assert rejected["reason"]
    finally:
        await p_client.disconnect()


@pytest.mark.asyncio
async def test_spectator_chat_does_not_reach_subconscious(live_server, _stub_agents):
    """Key Phase 3c invariant: spectator chat is invisible to the character."""
    port = live_server
    p_id, s_id, match_id = _seed_public_match()
    captured = _stub_agents

    p_client, p_rec = await _connect(port, p_id, match_id)
    s_client, s_rec = await _connect(port, s_id, match_id)
    try:
        await p_rec.wait("match_state", timeout=5.0)
        await s_rec.wait("match_state", timeout=5.0)

        # Spectator says something inflammatory.
        await s_client.emit("spectator_chat", {"text": "you should resign now LOL"}, namespace="/play")
        await s_rec.wait("spectator_chat_echoed", timeout=5.0)
        # Participant sees the broadcast — confirms the round trip worked.
        await p_rec.wait("spectator_chat_broadcast", timeout=5.0)

        # Participant now plays a move — the Subconscious MUST not see the
        # spectator's jab in its recent_chat context.
        await p_client.emit("make_move", {"uci": "e2e4"}, namespace="/play")
        await p_rec.wait("agent_move", timeout=10.0)

        assert len(captured["subconscious_inputs"]) == 1
        recent_chat_blob = " ".join(captured["subconscious_inputs"][0].recent_chat or [])
        assert "you should resign now LOL" not in recent_chat_blob

        # Also check the match's pending_player_chat buffer didn't pick it up.
        with SessionLocal() as s:
            m = s.get(Match, match_id)
            buf = (m.extra_state or {}).get("pending_player_chat", [])
            assert all("resign" not in e.get("text", "").lower() for e in buf)
    finally:
        await s_client.disconnect()
        await p_client.disconnect()


@pytest.mark.asyncio
async def test_player_chat_is_broadcast_to_spectators(live_server):
    port = live_server
    p_id, s_id, match_id = _seed_public_match()
    p_client, p_rec = await _connect(port, p_id, match_id)
    s_client, s_rec = await _connect(port, s_id, match_id)
    try:
        await p_rec.wait("match_state", timeout=5.0)
        await s_rec.wait("match_state", timeout=5.0)

        await p_client.emit("player_chat", {"text": "nice opening"}, namespace="/play")
        await p_rec.wait("player_chat_echoed", timeout=5.0)
        broadcast = await s_rec.wait("player_chat_broadcast", timeout=5.0)
        assert broadcast["text"] == "nice opening"
        assert broadcast["username"] == "participant"
    finally:
        await s_client.disconnect()
        await p_client.disconnect()


@pytest.mark.asyncio
async def test_spectator_count_increments_and_notifies(live_server):
    port = live_server
    p_id, s_id, match_id = _seed_public_match()
    p_client, p_rec = await _connect(port, p_id, match_id)
    try:
        await p_rec.wait("match_state", timeout=5.0)

        s_client, s_rec = await _connect(port, s_id, match_id)
        joined = await p_rec.wait("spectator_joined", timeout=5.0)
        assert joined["username"] == "spectator"

        count = await p_rec.wait("spectator_count", timeout=5.0)
        assert count["count"] == 1

        await s_client.disconnect()
        left = await p_rec.wait("spectator_left", timeout=5.0)
        assert left["username"] == "spectator"
        count2 = await p_rec.wait("spectator_count", timeout=5.0)
        assert count2["count"] == 0
    finally:
        await p_client.disconnect()


@pytest.mark.asyncio
async def test_spectator_rejected_from_private_character(live_server):
    """Spectator can't watch a match against a private character they don't own."""
    import chess
    with SessionLocal() as sess:
        owner = Player(username="pvt_owner", display_name="Owner")
        outsider = Player(username="pvt_outsider", display_name="Outsider")
        sess.add_all([owner, outsider]); sess.flush()
        priv = Character(
            name="Priv", aggression=5, risk_tolerance=5, patience=5, trash_talk=5,
            target_elo=1400, current_elo=1400, floor_elo=1400, max_elo=1800,
            adaptive=True, state=CharacterState.READY,
            content_rating=ContentRating.FAMILY, visibility=Visibility.PRIVATE,
            owner_id=owner.id,
        )
        sess.add(priv); sess.flush()
        m = Match(
            character_id=priv.id, player_id=owner.id,
            player_color=Color.WHITE, status=MatchStatus.IN_PROGRESS,
            initial_fen=chess.STARTING_FEN, current_fen=chess.STARTING_FEN,
            move_count=0, character_elo_at_start=priv.current_elo,
        )
        sess.add(m); sess.commit()
        match_id = m.id
        outsider_id = outsider.id

    client = socketio.AsyncClient(reconnection=False)
    with pytest.raises(Exception):
        await client.connect(
            f"http://127.0.0.1:{live_server}",
            namespaces=["/play"],
            socketio_path="socket.io",
            headers={"Cookie": f"player_id={outsider_id}"},
            auth={"match_id": match_id},
            wait=True,
        )
