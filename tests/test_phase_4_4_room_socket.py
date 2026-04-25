"""Phase 4.4 Block 2 — /room Socket.IO namespace tests.

Verifies the streaming pre-match chat path:
  R.1 — connect to /room, receive room_state with history
  R.2 — player_chat → player_chat_ack + agent_thinking + agent_chat (in order)
  R.3 — game_started fires when Soul emits start_game + match row created
  R.4 — two tabs: both sockets receive the same agent_chat event
  R.5 — reconnect: room_state serves full history on re-connect

Requires the same live uvicorn fixture pattern as test_sockets_integration.py.
LLM calls are stubbed so tests don't need GEMINI_API_KEY.
"""

from __future__ import annotations

import asyncio
import contextlib
import socket as _socket
from typing import Any

import pytest
import pytest_asyncio
import socketio
import uvicorn

from app.db import SessionLocal
from app.engine.registry import reset_engines_for_testing
from app.models.character import Character, CharacterState, ContentRating, Visibility
from app.models.chat import CharacterChatSession, ChatSessionStatus, CharacterChatTurn, ChatTurnRole
from app.models.match import Player
from app.redis_client import reset_memory_store_for_testing
from app.schemas.agents import MoodDeltas, SoulResponse


# --- Fixtures ---------------------------------------------------------------


def _free_port() -> int:
    with _socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(autouse=True)
def _reset_infra():
    reset_engines_for_testing()
    reset_memory_store_for_testing()
    yield
    reset_memory_store_for_testing()
    reset_engines_for_testing()


@pytest.fixture
def _stub_chat_agents(monkeypatch):
    """Stub Subconscious + Soul + greeting in the /room pipeline so no LLM needed."""

    def _fake_subconscious(session, character, inp, llm=None):
        return []

    def _fake_soul(character, inp):
        return SoulResponse(
            speak=f"hello, I'm {character.name}",
            emotion="neutral",
            emotion_intensity=0.3,
            mood_deltas=MoodDeltas(),
        )

    def _no_greeting(*args, **kwargs):
        return None

    monkeypatch.setattr("app.sockets.room_server.run_subconscious", _fake_subconscious)
    monkeypatch.setattr("app.sockets.room_server.run_soul", _fake_soul)
    monkeypatch.setattr("app.sockets.room_server.maybe_character_greets", _no_greeting)
    yield


@pytest.fixture
def _stub_start_game_agents(monkeypatch):
    """Soul always returns start_game so hand-off tests work."""

    def _fake_subconscious(session, character, inp, llm=None):
        return []

    def _fake_soul(character, inp):
        return SoulResponse(
            speak="alright, let's play",
            emotion="focused",
            emotion_intensity=0.5,
            mood_deltas=MoodDeltas(),
            game_action="start_game",
        )

    def _no_greeting(*args, **kwargs):
        return None

    monkeypatch.setattr("app.sockets.room_server.run_subconscious", _fake_subconscious)
    monkeypatch.setattr("app.sockets.room_server.run_soul", _fake_soul)
    monkeypatch.setattr("app.sockets.room_server.maybe_character_greets", _no_greeting)
    yield


@pytest_asyncio.fixture
async def live_server(_stub_chat_agents):
    """Start uvicorn in-process; yield port number."""
    from app.main import app as asgi_app

    port = _free_port()
    config = uvicorn.Config(
        asgi_app, host="127.0.0.1", port=port, log_level="warning", lifespan="on",
    )
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve())

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


@pytest_asyncio.fixture
async def live_server_start_game(_stub_start_game_agents):
    """Live server where Soul always returns start_game."""
    from app.main import app as asgi_app

    port = _free_port()
    config = uvicorn.Config(
        asgi_app, host="127.0.0.1", port=port, log_level="warning", lifespan="on",
    )
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve())

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


# --- DB seed helpers -------------------------------------------------------


def _seed_chat(*, add_turn: bool = False) -> tuple[str, str, str]:
    """Seed a player, character, and chat session. Return (player_id, char_id, session_id)."""
    with SessionLocal() as s:
        char = Character(
            name="RoomTest",
            short_description="Test character",
            backstory="A backstory.",
            voice_descriptor="calm",
            target_elo=1400, current_elo=1400, floor_elo=1400, max_elo=1800,
            adaptive=False, is_preset=False, owner_id=None,
            state=CharacterState.READY,
            visibility=Visibility.PUBLIC,
            content_rating=ContentRating.FAMILY,
        )
        s.add(char)
        player = Player(username="roomtest", display_name="RoomTest", elo=1200)
        s.add(player)
        s.flush()

        sess = CharacterChatSession(
            character_id=char.id,
            player_id=player.id,
            status=ChatSessionStatus.ACTIVE,
            pending_notes=[],
        )
        s.add(sess)
        s.flush()

        if add_turn:
            t = CharacterChatTurn(
                session_id=sess.id,
                turn_number=1,
                role=ChatTurnRole.CHARACTER,
                text="Welcome, stranger.",
            )
            s.add(t)

        s.commit()
        return player.id, char.id, sess.id


def _make_cookie(player_id: str) -> dict[str, str]:
    from app.auth import PLAYER_COOKIE
    return {"Cookie": f"{PLAYER_COOKIE}={player_id}"}


# --- Collector helper -------------------------------------------------------


class _EventCollector:
    """Subscribes to all events on an AsyncClient and records them."""

    def __init__(self, sio_client: socketio.AsyncClient):
        self._events: list[tuple[str, Any]] = []
        self._client = sio_client

        @sio_client.event(namespace="/room")
        def room_state(data):
            self._events.append(("room_state", data))

        @sio_client.event(namespace="/room")
        def player_chat_ack(data):
            self._events.append(("player_chat_ack", data))

        @sio_client.event(namespace="/room")
        def agent_thinking(data):
            self._events.append(("agent_thinking", data))

        @sio_client.event(namespace="/room")
        def agent_chat(data):
            self._events.append(("agent_chat", data))

        @sio_client.event(namespace="/room")
        def agent_error(data):
            self._events.append(("agent_error", data))

        @sio_client.event(namespace="/room")
        def game_started(data):
            self._events.append(("game_started", data))

        @sio_client.event(namespace="/room")
        def error(data):
            self._events.append(("error", data))

    def get(self, name: str) -> list[Any]:
        return [d for (n, d) in self._events if n == name]

    async def wait_for(self, name: str, *, timeout: float = 8.0) -> Any:
        deadline = asyncio.get_event_loop().time() + timeout
        while asyncio.get_event_loop().time() < deadline:
            hits = self.get(name)
            if hits:
                return hits[-1]
            await asyncio.sleep(0.05)
        raise TimeoutError(f"Event '{name}' not received within {timeout}s. Got: {[n for n,_ in self._events]}")


# --- Tests ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_room_connect_emits_room_state(live_server):
    """R.1 — connecting to /room emits room_state with empty history."""
    player_id, char_id, session_id = _seed_chat()
    port = live_server

    sio = socketio.AsyncClient()
    collector = _EventCollector(sio)

    await sio.connect(
        f"http://127.0.0.1:{port}",
        namespaces=["/room"],
        headers=_make_cookie(player_id),
        socketio_path="/socket.io",
        transports=["websocket"],
        auth={"character_id": char_id},
    )
    try:
        state = await collector.wait_for("room_state", timeout=5.0)
        assert state["session_id"] == session_id
        assert isinstance(state["turns"], list)
        assert state["character"]["id"] == char_id
    finally:
        await sio.disconnect()


@pytest.mark.asyncio
async def test_room_state_includes_existing_turns(live_server):
    """R.1b — room_state on reconnect includes persisted turns."""
    player_id, char_id, session_id = _seed_chat(add_turn=True)
    port = live_server

    sio = socketio.AsyncClient()
    collector = _EventCollector(sio)

    await sio.connect(
        f"http://127.0.0.1:{port}",
        namespaces=["/room"],
        headers=_make_cookie(player_id),
        socketio_path="/socket.io",
        transports=["websocket"],
        auth={"character_id": char_id},
    )
    try:
        state = await collector.wait_for("room_state", timeout=5.0)
        assert len(state["turns"]) >= 1
        assert state["turns"][0]["text"] == "Welcome, stranger."
    finally:
        await sio.disconnect()


@pytest.mark.asyncio
async def test_player_chat_produces_ack_and_agent_chat(live_server):
    """R.2 — player_chat → player_chat_ack + agent_thinking + agent_chat (in order).

    Uses add_turn=True so the session is non-empty on connect — this prevents
    _locked_fire_greeting from firing an agent_thinking before the player sends
    their message (which would make the ordering assertion non-deterministic).
    """
    player_id, char_id, _ = _seed_chat(add_turn=True)
    port = live_server

    sio = socketio.AsyncClient()
    collector = _EventCollector(sio)

    await sio.connect(
        f"http://127.0.0.1:{port}",
        namespaces=["/room"],
        headers=_make_cookie(player_id),
        socketio_path="/socket.io",
        transports=["websocket"],
        auth={"character_id": char_id},
    )
    try:
        await collector.wait_for("room_state", timeout=5.0)

        await sio.emit("player_chat", {"text": "hello there"}, namespace="/room")

        ack = await collector.wait_for("player_chat_ack", timeout=5.0)
        assert ack["text"] == "hello there"
        assert "turn_id" in ack

        await collector.wait_for("agent_thinking", timeout=5.0)
        chat = await collector.wait_for("agent_chat", timeout=10.0)
        assert "text" in chat
        assert len(chat["text"]) > 0

        # Order check: ack before thinking before chat.
        names = [n for (n, _) in collector._events]
        assert names.index("player_chat_ack") < names.index("agent_thinking")
        assert names.index("agent_thinking") < names.index("agent_chat")

    finally:
        await sio.disconnect()


@pytest.mark.asyncio
async def test_player_turn_persisted_to_db(live_server):
    """R.2b — player turn is written to DB synchronously before ack."""
    player_id, char_id, session_id = _seed_chat()
    port = live_server

    sio = socketio.AsyncClient()
    collector = _EventCollector(sio)

    await sio.connect(
        f"http://127.0.0.1:{port}",
        namespaces=["/room"],
        headers=_make_cookie(player_id),
        socketio_path="/socket.io",
        transports=["websocket"],
        auth={"character_id": char_id},
    )
    try:
        await collector.wait_for("room_state", timeout=5.0)
        await sio.emit("player_chat", {"text": "test persistence"}, namespace="/room")
        ack = await collector.wait_for("player_chat_ack", timeout=5.0)

        with SessionLocal() as s:
            from sqlalchemy import select
            turns = s.execute(
                select(CharacterChatTurn)
                .where(CharacterChatTurn.session_id == session_id)
            ).scalars().all()
        player_turns = [t for t in turns if t.role == ChatTurnRole.PLAYER]
        assert any(t.text == "test persistence" for t in player_turns)

    finally:
        await sio.disconnect()


@pytest.mark.asyncio
async def test_game_started_event_creates_match(live_server_start_game):
    """R.3 — when Soul returns start_game, game_started fires and match row is created."""
    player_id, char_id, _ = _seed_chat()
    port = live_server_start_game

    sio = socketio.AsyncClient()
    collector = _EventCollector(sio)

    await sio.connect(
        f"http://127.0.0.1:{port}",
        namespaces=["/room"],
        headers=_make_cookie(player_id),
        socketio_path="/socket.io",
        transports=["websocket"],
        auth={"character_id": char_id},
    )
    try:
        await collector.wait_for("room_state", timeout=5.0)
        await sio.emit("player_chat", {"text": "let's play"}, namespace="/room")

        game_ev = await collector.wait_for("game_started", timeout=12.0)
        assert "match_id" in game_ev
        assert game_ev["redirect_url"].startswith("/matches/")

        match_id = game_ev["match_id"]
        with SessionLocal() as s:
            from app.models.match import Match
            m = s.get(Match, match_id)
            assert m is not None
            assert m.status.value == "in_progress"
            assert m.move_count == 0  # opening move fires on /play connect, not here

    finally:
        await sio.disconnect()


@pytest.mark.asyncio
async def test_two_tabs_both_receive_agent_chat(live_server):
    """R.4 — two sockets for the same player + session both receive agent_chat."""
    player_id, char_id, _ = _seed_chat()
    port = live_server

    sio_a = socketio.AsyncClient()
    sio_b = socketio.AsyncClient()
    col_a = _EventCollector(sio_a)
    col_b = _EventCollector(sio_b)

    await sio_a.connect(
        f"http://127.0.0.1:{port}",
        namespaces=["/room"],
        headers=_make_cookie(player_id),
        socketio_path="/socket.io",
        transports=["websocket"],
        auth={"character_id": char_id},
    )
    await sio_b.connect(
        f"http://127.0.0.1:{port}",
        namespaces=["/room"],
        headers=_make_cookie(player_id),
        socketio_path="/socket.io",
        transports=["websocket"],
        auth={"character_id": char_id},
    )
    try:
        await col_a.wait_for("room_state", timeout=5.0)
        await col_b.wait_for("room_state", timeout=5.0)

        # Send from tab A.
        await sio_a.emit("player_chat", {"text": "multi-tab check"}, namespace="/room")

        # Both tabs should receive agent_chat.
        chat_a = await col_a.wait_for("agent_chat", timeout=10.0)
        chat_b = await col_b.wait_for("agent_chat", timeout=10.0)
        assert chat_a["text"] == chat_b["text"]

    finally:
        await sio_a.disconnect()
        await sio_b.disconnect()


@pytest.mark.asyncio
async def test_reconnect_room_state_has_full_history(live_server):
    """R.5 — disconnect + reconnect → room_state includes all prior turns."""
    player_id, char_id, session_id = _seed_chat()
    port = live_server

    # First connection: send a message, wait for agent_chat to persist.
    sio = socketio.AsyncClient()
    collector = _EventCollector(sio)

    await sio.connect(
        f"http://127.0.0.1:{port}",
        namespaces=["/room"],
        headers=_make_cookie(player_id),
        socketio_path="/socket.io",
        transports=["websocket"],
        auth={"character_id": char_id},
    )
    await collector.wait_for("room_state", timeout=5.0)
    await sio.emit("player_chat", {"text": "first message"}, namespace="/room")
    await collector.wait_for("agent_chat", timeout=10.0)
    await sio.disconnect()

    # Second connection: room_state must include both turns.
    sio2 = socketio.AsyncClient()
    col2 = _EventCollector(sio2)

    await sio2.connect(
        f"http://127.0.0.1:{port}",
        namespaces=["/room"],
        headers=_make_cookie(player_id),
        socketio_path="/socket.io",
        transports=["websocket"],
        auth={"character_id": char_id},
    )
    try:
        state2 = await col2.wait_for("room_state", timeout=5.0)
        # At least 2 turns: player "first message" + character response.
        assert len(state2["turns"]) >= 2
        player_texts = [t["text"] for t in state2["turns"] if t["role"] == "player"]
        assert "first message" in player_texts

    finally:
        await sio2.disconnect()


@pytest.mark.asyncio
async def test_auth_rejected_without_cookie(live_server):
    """Connecting without a player cookie must be refused (no room_state emitted)."""
    _, char_id, _ = _seed_chat()
    port = live_server

    sio = socketio.AsyncClient()
    connected = []

    @sio.event(namespace="/room")
    def room_state(data):
        connected.append(data)

    with pytest.raises(Exception):
        await sio.connect(
            f"http://127.0.0.1:{port}",
            namespaces=["/room"],
            socketio_path="/socket.io",
            transports=["websocket"],
            auth={"character_id": char_id},
        )

    assert not connected
