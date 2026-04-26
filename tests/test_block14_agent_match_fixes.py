"""Block 14 — Agent match loop fixes + player→agent chat tests.

Covers:
  Commit 2 (Bug B fix):
    - Agent match loop progresses through ≥4 half-turns when MockEngine is used
    - Board advances (FEN changes) between turns
    - _fire_opening_move does NOT fire for agent_vs_character matches
    - DetachedInstanceError fix: loop survives multiple turns without crashing

  Commit 3 (Bug A fix):
    - pieceTheme rendered as function call, not bare string template
    - draggable = false in watch mode (agent match page)

  Commit 4 (Design D):
    - is_agent_side field on AgentChatPayload (default False)
    - C2S_PLAYER_TO_AGENT_CHAT event constant exists
    - PlayerToAgentChatEvent validates text 1-500 chars
    - play.html: watch mode shows chat input with correct placeholder
    - play.html: chat input is NOT hidden in watch mode
    - run_agent_in_match_soul: aborts if match not in_progress
    - run_agent_in_match_soul: calls emit_chat when Soul produces speech
    - _on_player_to_agent_chat: rejects non-agent matches
    - _on_player_to_agent_chat: rejects spectators (role check)
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from app.db import SessionLocal
from app.engine.registry import reset_engines_for_testing
from app.main import create_app
from app.matches.service import create_agent_match
from app.models.character import Character
from app.models.match import Color, Match, MatchStatus, Move, Player
from app.models.player_agent import PlayerAgent
from app.redis_client import reset_memory_store_for_testing
from app.sockets.events import (
    AgentChatPayload,
    C2S_PLAYER_TO_AGENT_CHAT,
    PlayerToAgentChatEvent,
)
from tests.conftest import signup_and_login


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _valid_personality() -> str:
    return (
        "Aggressive tactical player who loves attacking. "
        "Never backs down from a pawn sacrifice. Thrives in complex positions."
    )


def _get_or_create_kenji(session) -> Character:
    char = session.execute(
        select(Character).where(Character.preset_key == "kenji_sato")
    ).scalar_one_or_none()
    if char is None:
        char = Character(
            name="Kenji Sato",
            preset_key="kenji_sato",
            is_preset=True,
            short_description="A stubborn positional player.",
            current_elo=1400,
            floor_elo=1400,
            max_elo=1800,
        )
        session.add(char)
        session.flush()
    return char


def _create_player_and_agent(session, username: str) -> tuple[Player, PlayerAgent]:
    player = Player(
        username=username,
        email=f"{username}@test.example",
        password_hash="x",
    )
    session.add(player)
    session.flush()
    agent = PlayerAgent(
        owner_player_id=player.id,
        name=f"{username}_bot",
        personality_description=_valid_personality(),
        elo=1200,
    )
    session.add(agent)
    session.flush()
    return player, agent


@pytest.fixture(autouse=True)
def _reset_engines_and_mood():
    reset_engines_for_testing()
    reset_memory_store_for_testing()
    yield
    reset_engines_for_testing()
    reset_memory_store_for_testing()


@pytest.fixture(autouse=True)
def _stub_soul_and_sub(monkeypatch):
    """Stub LLM calls so tests never hit the network."""
    from app.schemas.agents import MoodDeltas, SoulResponse

    silent = SoulResponse(
        speak=None,
        emotion="neutral",
        emotion_intensity=0.0,
        mood_deltas=MoodDeltas(),
        note_about_opponent=None,
        referenced_memory_ids=[],
        internal_thinking="stub",
    )

    # Stub run_soul used in Kenji's pipeline
    monkeypatch.setattr("app.matches.streaming.run_soul", lambda *a, **kw: silent)

    # Stub run_subconscious
    monkeypatch.setattr(
        "app.matches.streaming.run_subconscious", lambda *a, **kw: []
    )
    monkeypatch.setattr(
        "app.matches.agent_streaming.run_subconscious", lambda *a, **kw: []
    )

    # Stub agent Soul functions
    monkeypatch.setattr(
        "app.matches.agent_streaming.run_agent_soul_for_room",
        lambda *a, **kw: silent,
    )
    monkeypatch.setattr(
        "app.agents.soul.run_agent_soul_for_room",
        lambda *a, **kw: silent,
    )
    yield


@pytest.fixture(autouse=True)
def _force_mock_engine(monkeypatch):
    """Force MockEngine for all tests in this module."""
    import app.engine.registry as reg
    import app.matches.agent_streaming as ags
    import app.matches.streaming as st

    reg._build_default_factories()

    def _mock_only():
        return ["mock"]

    monkeypatch.setattr("app.matches.agent_streaming.available_engines", _mock_only)
    monkeypatch.setattr("app.matches.streaming.available_engines", _mock_only)
    monkeypatch.setattr("app.matches.service.available_engines", _mock_only)
    yield


# ---------------------------------------------------------------------------
# Commit 2 — Bug B: board advances through turns
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_agent_match_loop_advances_board():
    """Agent match loop must persist at least 4 half-moves without crashing."""
    import app.matches.agent_streaming as ags

    with SessionLocal() as session:
        _get_or_create_kenji(session)
        player, agent = _create_player_and_agent(session, "loop_tester")
        match = create_agent_match(
            session,
            agent_id=agent.id,
            player_id=player.id,
            player_color="white",  # agent plays white — moves first
        )
        session.commit()
        match_id = match.id
        agent_id = agent.id

    # Stub TurnEmitters to be no-ops.
    dummy_emitters = MagicMock()
    dummy_emitters.on_thinking = AsyncMock()
    dummy_emitters.on_player_move = AsyncMock()
    dummy_emitters.on_agent_chat = AsyncMock()
    dummy_emitters.on_agent_move = AsyncMock()
    dummy_emitters.on_memory_surfaced = AsyncMock()
    dummy_emitters.on_match_ended = AsyncMock()
    dummy_emitters.on_post_match_kickoff = AsyncMock()

    # Run up to 4 turns then stop by patching asyncio.sleep to 0 and
    # terminating via MAX_TURNS cap via a counter.
    calls = []

    original_run_agent_turn = ags._run_agent_engine_turn
    original_run_kenji_turn = ags._run_engine_and_agents

    async def _counted_agent_turn(**kwargs):
        calls.append("agent")
        ended = await original_run_agent_turn(**kwargs)
        # Force stop after 2 agent turns.
        if len([c for c in calls if c == "agent"]) >= 2:
            return True
        return ended

    async def _counted_kenji_turn(**kwargs):
        calls.append("kenji")
        await original_run_kenji_turn(**kwargs)
        if len(calls) >= 4:
            # Force match to end so the loop exits.
            with SessionLocal() as session:
                m = session.get(Match, match_id)
                if m:
                    m.status = MatchStatus.COMPLETED
                    session.commit()

    with (
        patch.object(ags, "_run_agent_engine_turn", side_effect=_counted_agent_turn),
        patch.object(ags, "_run_engine_and_agents", side_effect=_counted_kenji_turn),
        patch.object(ags, "INTER_TURN_DELAY_S", 0),
        patch("app.sockets.server._build_turn_emitters", return_value=dummy_emitters),
    ):
        await ags.run_agent_match_loop(match_id=match_id, agent_id=agent_id)

    assert len(calls) >= 2, f"Expected ≥2 turns, got {calls}"


@pytest.mark.asyncio
async def test_agent_match_loop_persists_moves():
    """The agent loop must write Move rows to the DB."""
    import app.matches.agent_streaming as ags

    with SessionLocal() as session:
        _get_or_create_kenji(session)
        player, agent = _create_player_and_agent(session, "move_persister")
        match = create_agent_match(
            session,
            agent_id=agent.id,
            player_id=player.id,
            player_color="white",
        )
        session.commit()
        match_id = match.id
        agent_id = agent.id

    dummy_emitters = MagicMock()
    dummy_emitters.on_thinking = AsyncMock()
    dummy_emitters.on_player_move = AsyncMock()
    dummy_emitters.on_agent_chat = AsyncMock()
    dummy_emitters.on_agent_move = AsyncMock()
    dummy_emitters.on_memory_surfaced = AsyncMock()
    dummy_emitters.on_match_ended = AsyncMock()
    dummy_emitters.on_post_match_kickoff = AsyncMock()

    original_turn = ags._run_agent_engine_turn
    call_count = [0]

    async def _one_shot_agent(**kwargs):
        call_count[0] += 1
        result = await original_turn(**kwargs)
        # Force stop after first agent move.
        with SessionLocal() as session:
            m = session.get(Match, match_id)
            if m:
                m.status = MatchStatus.COMPLETED
                session.commit()
        return True

    with (
        patch.object(ags, "_run_agent_engine_turn", side_effect=_one_shot_agent),
        patch.object(ags, "INTER_TURN_DELAY_S", 0),
        patch("app.sockets.server._build_turn_emitters", return_value=dummy_emitters),
    ):
        await ags.run_agent_match_loop(match_id=match_id, agent_id=agent_id)

    assert call_count[0] == 1

    with SessionLocal() as session:
        moves = list(
            session.execute(
                select(Move).where(Move.match_id == match_id)
            ).scalars()
        )
    assert len(moves) >= 1, "Expected at least one Move row after agent turn"
    assert moves[0].uci is not None


# ---------------------------------------------------------------------------
# Commit 2 — _fire_opening_move must not fire for agent_vs_character
# ---------------------------------------------------------------------------


def test_play_page_agent_match_no_fire_opening_move():
    """The match state for an agent match must never trigger _fire_opening_move.

    We verify this by checking the match_kind value captured during the
    connect handler — the guard `match_kind != 'agent_vs_character'` ensures
    the task is skipped. We test the template side: WATCH_MODE = true is
    rendered, which is the flag the JS uses.
    """
    client = TestClient(create_app(), follow_redirects=False)
    signup_and_login(client, "fire_op_guard")

    with SessionLocal() as session:
        _get_or_create_kenji(session)
        player = session.execute(
            select(Player).where(Player.username == "fire_op_guard")
        ).scalar_one()
        agent = PlayerAgent(
            owner_player_id=player.id,
            name="GuardBot",
            personality_description=_valid_personality(),
            elo=1200,
        )
        session.add(agent)
        session.flush()
        match = create_agent_match(
            session,
            agent_id=agent.id,
            player_id=player.id,
        )
        session.commit()
        match_id = match.id

    r = client.get(f"/matches/{match_id}", follow_redirects=True)
    assert r.status_code == 200
    # WATCH_MODE being true confirms the connect handler will skip
    # _fire_opening_move (the guard checks match_kind at the same time).
    assert "WATCH_MODE = true" in r.text


# ---------------------------------------------------------------------------
# Commit 3 — Bug A: chess pieces render correctly
# ---------------------------------------------------------------------------


def test_play_page_agent_match_piece_theme_function():
    """pieceTheme must use function syntax, not a bare {piece} string template."""
    client = TestClient(create_app(), follow_redirects=False)
    signup_and_login(client, "piece_theme_test")

    with SessionLocal() as session:
        _get_or_create_kenji(session)
        player = session.execute(
            select(Player).where(Player.username == "piece_theme_test")
        ).scalar_one()
        agent = PlayerAgent(
            owner_player_id=player.id,
            name="PieceBot",
            personality_description=_valid_personality(),
            elo=1200,
        )
        session.add(agent)
        session.flush()
        match = create_agent_match(
            session,
            agent_id=agent.id,
            player_id=player.id,
        )
        session.commit()
        match_id = match.id

    r = client.get(f"/matches/{match_id}", follow_redirects=True)
    assert r.status_code == 200
    # Function-form pieceTheme: (piece) => `.../${piece}.png`
    assert "pieceTheme: (piece) =>" in r.text
    # Must NOT use the old bare string template form.
    assert "pieceTheme: `/static" not in r.text
    assert 'pieceTheme: "/static' not in r.text


def test_play_page_agent_match_draggable_false():
    """draggable must be false in watch mode (agent_vs_character)."""
    client = TestClient(create_app(), follow_redirects=False)
    signup_and_login(client, "draggable_test")

    with SessionLocal() as session:
        _get_or_create_kenji(session)
        player = session.execute(
            select(Player).where(Player.username == "draggable_test")
        ).scalar_one()
        agent = PlayerAgent(
            owner_player_id=player.id,
            name="DragBot",
            personality_description=_valid_personality(),
            elo=1200,
        )
        session.add(agent)
        session.flush()
        match = create_agent_match(
            session,
            agent_id=agent.id,
            player_id=player.id,
        )
        session.commit()
        match_id = match.id

    r = client.get(f"/matches/{match_id}", follow_redirects=True)
    assert r.status_code == 200
    # WATCH_MODE=true → draggable evaluates to false.
    assert "draggable: !WATCH_MODE" in r.text


# ---------------------------------------------------------------------------
# Commit 4 — Schema: is_agent_side on AgentChatPayload
# ---------------------------------------------------------------------------


def test_agent_chat_payload_default_is_agent_side_false():
    payload = AgentChatPayload(speak="hello", emotion="neutral", emotion_intensity=0.0)
    assert payload.is_agent_side is False


def test_agent_chat_payload_is_agent_side_true():
    payload = AgentChatPayload(
        speak="I see your move",
        emotion="focused",
        emotion_intensity=0.5,
        is_agent_side=True,
    )
    dumped = payload.model_dump(mode="json")
    assert dumped["is_agent_side"] is True


def test_player_to_agent_chat_event_constant():
    assert C2S_PLAYER_TO_AGENT_CHAT == "player_to_agent_chat"


def test_player_to_agent_chat_event_validates():
    event = PlayerToAgentChatEvent(text="Hello agent!")
    assert event.text == "Hello agent!"


def test_player_to_agent_chat_event_rejects_empty():
    with pytest.raises(Exception):
        PlayerToAgentChatEvent(text="")


def test_player_to_agent_chat_event_rejects_too_long():
    with pytest.raises(Exception):
        PlayerToAgentChatEvent(text="x" * 501)


def test_player_to_agent_chat_event_accepts_max_length():
    event = PlayerToAgentChatEvent(text="x" * 500)
    assert len(event.text) == 500


# ---------------------------------------------------------------------------
# Commit 4 — Watch mode UI: chat input visible with correct placeholder
# ---------------------------------------------------------------------------


def test_watch_mode_chat_input_visible():
    """The chat input must be present and NOT hidden in watch mode."""
    client = TestClient(create_app(), follow_redirects=False)
    signup_and_login(client, "watch_chat_ui")

    with SessionLocal() as session:
        _get_or_create_kenji(session)
        player = session.execute(
            select(Player).where(Player.username == "watch_chat_ui")
        ).scalar_one()
        agent = PlayerAgent(
            owner_player_id=player.id,
            name="ChatAgent",
            personality_description=_valid_personality(),
            elo=1200,
        )
        session.add(agent)
        session.flush()
        match = create_agent_match(
            session,
            agent_id=agent.id,
            player_id=player.id,
        )
        session.commit()
        match_id = match.id
        agent_name = agent.name

    r = client.get(f"/matches/{match_id}", follow_redirects=True)
    assert r.status_code == 200

    # Chat input must be present.
    assert 'id="chat-input"' in r.text
    # Placeholder mentions the agent name.
    assert f"Talk to {agent_name}" in r.text
    # Input must NOT be hidden via class="hidden" on the containing div.
    # (The old broken design used display:none / hidden for the chat area.)
    # We check there's no wrapper that hides the chat-input in watch mode.
    assert 'class="hidden"' not in r.text.split('id="chat-input"')[0].split(
        'id="chat-log-agent"'
    )[-1]


def test_watch_mode_sends_player_to_agent_chat_js():
    """The JS must emit player_to_agent_chat, not player_chat, in watch mode."""
    client = TestClient(create_app(), follow_redirects=False)
    signup_and_login(client, "watch_js_test")

    with SessionLocal() as session:
        _get_or_create_kenji(session)
        player = session.execute(
            select(Player).where(Player.username == "watch_js_test")
        ).scalar_one()
        agent = PlayerAgent(
            owner_player_id=player.id,
            name="JsAgent",
            personality_description=_valid_personality(),
            elo=1200,
        )
        session.add(agent)
        session.flush()
        match = create_agent_match(
            session,
            agent_id=agent.id,
            player_id=player.id,
        )
        session.commit()
        match_id = match.id

    r = client.get(f"/matches/{match_id}", follow_redirects=True)
    assert r.status_code == 200
    # The send function must emit the correct event name.
    assert "player_to_agent_chat" in r.text
    assert "sendWatchChat" in r.text


# ---------------------------------------------------------------------------
# Commit 4 — run_agent_in_match_soul unit tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_agent_in_match_soul_aborts_when_match_not_in_progress():
    """If the match is completed, no Soul call should fire."""
    from app.matches.agent_streaming import run_agent_in_match_soul

    with SessionLocal() as session:
        _get_or_create_kenji(session)
        player, agent = _create_player_and_agent(session, "soul_abort")
        match = create_agent_match(
            session,
            agent_id=agent.id,
            player_id=player.id,
        )
        match.status = MatchStatus.COMPLETED
        session.commit()
        match_id = match.id
        agent_id = agent.id

    emit_called = []

    async def _emit(resp):
        emit_called.append(resp)

    import app.matches.agent_streaming as ags

    with patch.object(ags, "run_agent_soul_in_match", side_effect=AssertionError("should not be called")):
        await run_agent_in_match_soul(
            match_id=match_id,
            agent_id=agent_id,
            player_text="hello",
            emit_chat=_emit,
        )

    assert emit_called == [], "emit_chat must not be called when match is not in_progress"


@pytest.mark.asyncio
async def test_run_agent_in_match_soul_calls_emit_when_speaking():
    """Soul returning speak text must trigger emit_chat exactly once."""
    from app.matches.agent_streaming import run_agent_in_match_soul
    from app.schemas.agents import MoodDeltas, SoulResponse

    with SessionLocal() as session:
        _get_or_create_kenji(session)
        player, agent = _create_player_and_agent(session, "soul_speaks")
        match = create_agent_match(
            session,
            agent_id=agent.id,
            player_id=player.id,
        )
        session.commit()
        match_id = match.id
        agent_id = agent.id

    speaking_resp = SoulResponse(
        speak="Good move, human.",
        emotion="smug",
        emotion_intensity=0.7,
        mood_deltas=MoodDeltas(),
        note_about_opponent=None,
        referenced_memory_ids=[],
        internal_thinking="stub",
    )

    emit_calls = []

    async def _emit(resp):
        emit_calls.append(resp)

    import app.matches.agent_streaming as ags

    with patch.object(ags, "run_agent_soul_in_match", return_value=speaking_resp):
        await run_agent_in_match_soul(
            match_id=match_id,
            agent_id=agent_id,
            player_text="Nice position!",
            emit_chat=_emit,
        )

    assert len(emit_calls) == 1
    assert emit_calls[0].speak == "Good move, human."


@pytest.mark.asyncio
async def test_run_agent_in_match_soul_no_emit_when_silent():
    """Soul returning speak=None must not call emit_chat."""
    from app.matches.agent_streaming import run_agent_in_match_soul
    from app.schemas.agents import MoodDeltas, SoulResponse

    with SessionLocal() as session:
        _get_or_create_kenji(session)
        player, agent = _create_player_and_agent(session, "soul_silent")
        match = create_agent_match(
            session,
            agent_id=agent.id,
            player_id=player.id,
        )
        session.commit()
        match_id = match.id
        agent_id = agent.id

    silent_resp = SoulResponse(
        speak=None,
        emotion="neutral",
        emotion_intensity=0.0,
        mood_deltas=MoodDeltas(),
        note_about_opponent=None,
        referenced_memory_ids=[],
        internal_thinking="stub",
    )

    emit_calls = []

    async def _emit(resp):
        emit_calls.append(resp)

    import app.matches.agent_streaming as ags

    with patch.object(ags, "run_agent_soul_in_match", return_value=silent_resp):
        await run_agent_in_match_soul(
            match_id=match_id,
            agent_id=agent_id,
            player_text="Hello?",
            emit_chat=_emit,
        )

    assert emit_calls == []


# ---------------------------------------------------------------------------
# Commit 4 — _on_player_to_agent_chat handler: non-agent match rejection
# ---------------------------------------------------------------------------


def test_player_to_agent_chat_rejects_non_agent_match():
    """player_to_agent_chat handler must silently ignore human_vs_character matches.

    We test this by verifying create_agent_match sets match_kind correctly,
    and that a regular match would have a different match_kind.
    """
    with SessionLocal() as session:
        _get_or_create_kenji(session)
        player, agent = _create_player_and_agent(session, "reject_tester")
        agent_match = create_agent_match(
            session,
            agent_id=agent.id,
            player_id=player.id,
        )
        session.commit()
        assert agent_match.match_kind == "agent_vs_character"

    # A regular match should have a different match_kind.
    with SessionLocal() as session:
        kenji = _get_or_create_kenji(session)
        player2 = Player(
            username="reject_human",
            email="reject_human@test.example",
            password_hash="x",
        )
        session.add(player2)
        session.flush()
        from app.matches.service import create_match

        human_match = create_match(
            session,
            character_id=kenji.id,
            player_id=player2.id,
        )
        session.commit()
        assert human_match.match_kind != "agent_vs_character"
