"""Block 15 — Agent match quality fixes.

Tests for:
  Commit 2 (cross-side chat context):
    - _load_cross_chat_lines labels own moves "You:" and opponent with name
    - Kenji turn uses cross-chat in agent_vs_character matches

  Commit 3 (diversity guard for agent):
    - filter_shuffle_moves is called inside _run_agent_engine_turn

  Commit 4 (speaking cap):
    - Agent system prompt references 40% cap
    - Prompt includes "You:" counting instruction

  Commit 5 (pieceTheme function form):
    - pvp.html uses function form, not bare string template

  Commit 6 (post-match for agent_vs_character):
    - create_agent_match records agent.elo, not player.elo, as player_elo_at_start
    - _apply_agent_vs_character_elo updates PlayerAgent.elo (not Player.elo)
    - _apply_agent_vs_character_elo does NOT change Player.elo
    - save_inline_memory fires with agent_id when soul_resp.save_memory is set
    - play.html renders AGENT_ID in watch mode

  Commit 7 (UI race):
    - thinking-label span exists (not thinking-name + separate " is thinking" span)
    - setThinking rounds etaSeconds to 1 decimal via .toFixed(1)
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

import chess

from app.db import SessionLocal
from app.engine.registry import reset_engines_for_testing
from app.main import create_app
from app.matches.service import create_agent_match
from app.models.character import Character
from app.models.match import Color, Match, MatchStatus, Move, Player
from app.models.player_agent import PlayerAgent
from app.redis_client import reset_memory_store_for_testing
from tests.conftest import signup_and_login

_START_FEN = chess.Board().fen()


# ---------------------------------------------------------------------------
# Shared helpers
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
def _reset_state():
    reset_engines_for_testing()
    reset_memory_store_for_testing()
    yield
    reset_engines_for_testing()
    reset_memory_store_for_testing()


@pytest.fixture(autouse=True)
def _stub_llm(monkeypatch):
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
    monkeypatch.setattr("app.matches.streaming.run_soul", lambda *a, **kw: silent)
    monkeypatch.setattr("app.matches.streaming.run_subconscious", lambda *a, **kw: [])
    monkeypatch.setattr("app.matches.agent_streaming.run_subconscious", lambda *a, **kw: [])
    monkeypatch.setattr("app.matches.agent_streaming.run_agent_soul_in_match_move", lambda *a, **kw: silent)
    monkeypatch.setattr("app.matches.agent_streaming.run_agent_soul_for_room", lambda *a, **kw: silent)
    monkeypatch.setattr("app.agents.soul.run_agent_soul_for_room", lambda *a, **kw: silent)
    yield


@pytest.fixture(autouse=True)
def _mock_engine(monkeypatch):
    import app.engine.registry as reg
    reg._build_default_factories()

    def _mock_only():
        return ["mock"]

    monkeypatch.setattr("app.matches.agent_streaming.available_engines", _mock_only)
    monkeypatch.setattr("app.matches.streaming.available_engines", _mock_only)
    monkeypatch.setattr("app.matches.service.available_engines", _mock_only)
    yield


# ---------------------------------------------------------------------------
# Commit 2 — cross-side chat labeling
# ---------------------------------------------------------------------------


def test_load_cross_chat_lines_labels_own_as_you():
    """Moves made by the own_color side are labeled 'You:'."""
    from app.matches.streaming import _load_cross_chat_lines

    with SessionLocal() as session:
        _get_or_create_kenji(session)
        player, agent = _create_player_and_agent(session, "crosschat_own")
        match = create_agent_match(
            session,
            agent_id=agent.id,
            player_id=player.id,
            player_color="white",
        )
        # Insert a move for the agent (white = player_color) with chat.
        mv = Move(
            match_id=match.id,
            move_number=1,
            side=Color.WHITE,
            uci="e2e4",
            san="e4",
            fen_after=_START_FEN,
            agent_chat_after="I start boldly.",
        )
        session.add(mv)
        session.commit()
        match_id = match.id

    with SessionLocal() as session:
        lines = _load_cross_chat_lines(
            session, match_id,
            own_color="white",
            opponent_name="Kenji",
        )

    assert any("You: I start boldly." == line for line in lines)


def test_load_cross_chat_lines_labels_opponent_with_name():
    """Moves by the opponent side are labeled with the opponent's name."""
    from app.matches.streaming import _load_cross_chat_lines

    with SessionLocal() as session:
        _get_or_create_kenji(session)
        player, agent = _create_player_and_agent(session, "crosschat_opp")
        match = create_agent_match(
            session,
            agent_id=agent.id,
            player_id=player.id,
            player_color="white",  # agent = white, Kenji = black
        )
        # Black (Kenji) move with chat.
        mv = Move(
            match_id=match.id,
            move_number=1,
            side=Color.BLACK,
            uci="e7e5",
            san="e5",
            fen_after=_START_FEN,
            agent_chat_after="Your move.",
        )
        session.add(mv)
        session.commit()
        match_id = match.id

    with SessionLocal() as session:
        lines = _load_cross_chat_lines(
            session, match_id,
            own_color="white",
            opponent_name="Kenji",
        )

    assert any("Kenji: Your move." == line for line in lines)


def test_load_cross_chat_lines_does_not_label_opponent_as_you():
    """Opponent lines must never be labeled 'You:'."""
    from app.matches.streaming import _load_cross_chat_lines

    with SessionLocal() as session:
        _get_or_create_kenji(session)
        player, agent = _create_player_and_agent(session, "crosschat_noyou")
        match = create_agent_match(
            session,
            agent_id=agent.id,
            player_id=player.id,
            player_color="white",
        )
        mv = Move(
            match_id=match.id,
            move_number=1,
            side=Color.BLACK,  # opponent side
            uci="e7e5",
            san="e5",
            fen_after=_START_FEN,
            agent_chat_after="Counter-attack!",
        )
        session.add(mv)
        session.commit()
        match_id = match.id

    with SessionLocal() as session:
        lines = _load_cross_chat_lines(
            session, match_id,
            own_color="white",
            opponent_name="Kenji",
        )

    assert all("You:" not in line for line in lines), f"Unexpected 'You:' label: {lines}"


# ---------------------------------------------------------------------------
# Commit 4 — speaking cap in agent prompt
# ---------------------------------------------------------------------------


def test_agent_system_prompt_contains_40_percent_cap():
    """build_agent_system_prompt must mention the 40% cap."""
    from app.agents.prompts import build_agent_system_prompt

    with SessionLocal() as session:
        _, agent = _create_player_and_agent(session, "prompt_cap_tester")
        prompt = build_agent_system_prompt(agent)
        session.rollback()

    assert "40%" in prompt, "Prompt missing 40% speaking cap"


def test_agent_system_prompt_counts_you_lines():
    """Prompt must instruct the agent to count lines labeled 'You:'."""
    from app.agents.prompts import build_agent_system_prompt

    with SessionLocal() as session:
        _, agent = _create_player_and_agent(session, "prompt_you_tester")
        prompt = build_agent_system_prompt(agent)
        session.rollback()

    assert '"You:"' in prompt or "'You:'" in prompt or "labeled" in prompt, (
        "Prompt missing 'You:' counting instruction"
    )


# ---------------------------------------------------------------------------
# Commit 5 — pieceTheme function form across all board-rendering templates
#
# History: Block 15 originally only checked pvp.html, which is the PvP lobby
# template (lobby_routes.py). Agent matches render play.html; spectators use
# watch.html. All three must use function-form pieceTheme to avoid 404s.
# ---------------------------------------------------------------------------


def test_pvp_html_piece_theme_uses_function_form():
    """pvp.html (PvP lobby) pieceTheme must use function syntax."""
    from pathlib import Path

    path = Path("app/web/templates/pvp.html")
    content = path.read_text(encoding="utf-8")

    assert "pieceTheme: (piece) =>" in content, "pvp.html missing function-form pieceTheme"
    assert "pieceTheme: '/static" not in content, "pvp.html still uses bare string pieceTheme"


def test_play_html_piece_theme_uses_function_form():
    """play.html (agent + human match template) pieceTheme must use function syntax."""
    from pathlib import Path

    path = Path("app/web/templates/play.html")
    content = path.read_text(encoding="utf-8")

    assert "pieceTheme: (piece) =>" in content, "play.html missing function-form pieceTheme"
    assert "pieceTheme: '/static" not in content, "play.html still uses bare string pieceTheme"


def test_watch_html_piece_theme_uses_function_form():
    """watch.html (spectator template) pieceTheme must use function syntax."""
    from pathlib import Path

    path = Path("app/web/templates/watch.html")
    content = path.read_text(encoding="utf-8")

    assert "pieceTheme: (piece) =>" in content, "watch.html missing function-form pieceTheme"
    assert "pieceTheme: '/static" not in content, "watch.html still uses bare string pieceTheme"


# ---------------------------------------------------------------------------
# Commit 6 — service.py: player_elo_at_start = agent.elo
# ---------------------------------------------------------------------------


def test_create_agent_match_records_agent_elo_as_player_elo_at_start():
    """create_agent_match must store agent.elo (not player.elo) as player_elo_at_start."""
    with SessionLocal() as session:
        _get_or_create_kenji(session)
        player, agent = _create_player_and_agent(session, "elo_start_test")
        # Give agent and player different elos so we can tell them apart.
        agent.elo = 1350
        player.elo = 900
        session.flush()

        match = create_agent_match(
            session,
            agent_id=agent.id,
            player_id=player.id,
        )
        session.flush()

        assert match.player_elo_at_start == 1350, (
            f"Expected agent.elo=1350, got {match.player_elo_at_start}"
        )
        session.rollback()


# ---------------------------------------------------------------------------
# Commit 6 — _apply_agent_vs_character_elo
# ---------------------------------------------------------------------------


def test_apply_agent_vs_character_elo_updates_agent_not_player():
    """After the agent Elo step, PlayerAgent.elo must change but Player.elo must not."""
    from app.post_match.processor import _apply_agent_vs_character_elo
    from app.models.match import MatchResult

    with SessionLocal() as session:
        kenji = _get_or_create_kenji(session)
        player, agent = _create_player_and_agent(session, "elo_update_test")
        player.elo = 1200
        agent.elo = 1200
        session.flush()

        match = create_agent_match(
            session,
            agent_id=agent.id,
            player_id=player.id,
            player_color="white",
        )
        # Simulate agent won (agent = white → WHITE_WIN means agent won).
        match.status = MatchStatus.COMPLETED
        match.result = MatchResult.WHITE_WIN
        match.move_count = 40
        match.character_elo_at_start = kenji.current_elo
        match.player_elo_at_start = agent.elo
        session.flush()

        player_elo_before = player.elo
        agent_elo_before = agent.elo

        _apply_agent_vs_character_elo(session, match=match, analysis_moves=[])
        session.flush()

        # Agent elo must change.
        assert agent.elo != agent_elo_before, (
            f"Agent elo unchanged: {agent.elo}"
        )
        # Player elo must NOT change.
        assert player.elo == player_elo_before, (
            f"Player.elo changed from {player_elo_before} to {player.elo}"
        )

        session.rollback()


def test_apply_agent_vs_character_elo_agent_win_increases_elo():
    """Agent winning against Kenji should increase agent.elo."""
    from app.post_match.processor import _apply_agent_vs_character_elo
    from app.models.match import MatchResult

    with SessionLocal() as session:
        kenji = _get_or_create_kenji(session)
        player, agent = _create_player_and_agent(session, "agent_win_elo")
        agent.elo = 1200
        session.flush()

        match = create_agent_match(
            session,
            agent_id=agent.id,
            player_id=player.id,
            player_color="white",
        )
        match.status = MatchStatus.COMPLETED
        match.result = MatchResult.WHITE_WIN  # agent (white) wins
        match.move_count = 40
        match.character_elo_at_start = kenji.current_elo
        match.player_elo_at_start = agent.elo
        session.flush()

        elo_before = agent.elo
        _apply_agent_vs_character_elo(session, match=match, analysis_moves=[])
        session.flush()

        assert agent.elo > elo_before, f"Expected elo increase, got {elo_before} → {agent.elo}"
        session.rollback()


# ---------------------------------------------------------------------------
# Commit 6 — inline memory fires with agent_id
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_agent_engine_turn_fires_save_memory_with_agent_id():
    """When soul_resp.save_memory is set, save_inline_memory must fire with agent_id."""
    import app.matches.agent_streaming as ags
    from app.schemas.agents import InlineMemoryRequest, MoodDeltas, SoulResponse

    with SessionLocal() as session:
        _get_or_create_kenji(session)
        player, agent = _create_player_and_agent(session, "inline_mem_test")
        match = create_agent_match(
            session,
            agent_id=agent.id,
            player_id=player.id,
            player_color="white",
        )
        session.commit()
        match_id = match.id
        agent_id = agent.id

    save_mem_request = InlineMemoryRequest(
        type="observation",
        emotional_valence=0.5,
        triggers=["e4", "center control", "opening"],
        narrative_text="I opened with e4 and seized the center early.",
        relevance_tags=["opening"],
    )

    speaking_resp = SoulResponse(
        speak="Let's play!",
        emotion="focused",
        emotion_intensity=0.5,
        mood_deltas=MoodDeltas(),
        note_about_opponent=None,
        referenced_memory_ids=[],
        internal_thinking="stub",
        save_memory=save_mem_request,
    )

    dummy_emitters = MagicMock()
    dummy_emitters.on_thinking = AsyncMock()
    dummy_emitters.on_player_move = AsyncMock()
    dummy_emitters.on_agent_chat = AsyncMock()
    dummy_emitters.on_match_ended = AsyncMock()
    dummy_emitters.on_post_match_kickoff = AsyncMock()

    saved_calls = []

    async def _fake_save_inline(request, *, agent_id=None, player_id=None, match_id=None):
        saved_calls.append({"agent_id": agent_id, "player_id": player_id})

    fake_save = AsyncMock(side_effect=_fake_save_inline)

    with (
        patch.object(ags, "run_agent_soul_in_match_move", return_value=speaking_resp),
        patch.object(ags, "save_inline_memory", new=fake_save),
    ):
        ended = await ags._run_agent_engine_turn(
            match_id=match_id,
            agent_id=agent_id,
            emitters=dummy_emitters,
            settings=MagicMock(),
        )
        # Yield to the event loop so the create_task coroutine runs.
        await asyncio.sleep(0)

    assert not ended, "Match should not have ended after one agent move"
    assert len(saved_calls) == 1, f"save_inline_memory should fire once, got {saved_calls}"
    assert saved_calls[0]["agent_id"] == agent_id, (
        f"save_inline_memory called with agent_id={saved_calls[0]['agent_id']}, expected {agent_id}"
    )


# ---------------------------------------------------------------------------
# Commit 6 — play.html: AGENT_ID rendered in watch mode
# ---------------------------------------------------------------------------


def test_play_html_renders_agent_id_in_watch_mode():
    """play.html must render const AGENT_ID for agent_vs_character matches."""
    client = TestClient(create_app(), follow_redirects=False)
    signup_and_login(client, "agent_id_render")

    with SessionLocal() as session:
        _get_or_create_kenji(session)
        player = session.execute(
            select(Player).where(Player.username == "agent_id_render")
        ).scalar_one()
        agent = PlayerAgent(
            owner_player_id=player.id,
            name="IdBot",
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
        agent_id = agent.id

    r = client.get(f"/matches/{match_id}", follow_redirects=True)
    assert r.status_code == 200
    assert "AGENT_ID" in r.text, "AGENT_ID constant missing from play.html"
    assert agent_id in r.text, f"Agent ID {agent_id} not embedded in page"


# ---------------------------------------------------------------------------
# Commit 7 — UI race: thinking banner + float rounding
# ---------------------------------------------------------------------------


def test_play_html_thinking_label_span_exists():
    """play.html must use a single #thinking-label span, not split name + action spans."""
    from pathlib import Path

    content = Path("app/web/templates/play.html").read_text(encoding="utf-8")

    # New unified span.
    assert 'id="thinking-label"' in content, "thinking-label span missing"
    # Old split form must not exist.
    assert 'id="thinking-name"' not in content, "Old thinking-name span still present"


def test_play_html_setthinking_rounds_float():
    """setThinking must call .toFixed(1) to avoid raw float display."""
    from pathlib import Path

    content = Path("app/web/templates/play.html").read_text(encoding="utf-8")

    assert "toFixed(1)" in content, "setThinking missing .toFixed(1) for eta rounding"
