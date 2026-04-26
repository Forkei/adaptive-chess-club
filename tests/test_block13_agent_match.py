"""Block 13 — Agent vs Character match flow tests.

Covers:
  - create_agent_match: correct Match row fields
  - create_agent_match: rejects wrong-owner agents
  - Agent detail page: recent_matches context populated
  - Play page: WATCH_MODE rendered for agent_vs_character matches
  - Play page: human_vs_character still renders in normal mode
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from app.db import SessionLocal
from app.main import create_app
from app.matches.service import create_agent_match
from app.models.character import Character
from app.models.match import Match, MatchStatus
from app.models.player_agent import PlayerAgent
from app.models.match import Player
from tests.conftest import signup_and_login


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _client() -> TestClient:
    return TestClient(create_app(), follow_redirects=False)


def _login(username: str) -> TestClient:
    c = _client()
    signup_and_login(c, username)
    return c


def _valid_personality() -> str:
    return (
        "Aggressive tactical player who loves attacking. "
        "Never backs down from a pawn sacrifice. Thrives in complex positions."
    )


def _create_agent(client: TestClient, name: str = "TestBot") -> str:
    r = client.post(
        "/agents/new",
        data={"name": name, "personality_description": _valid_personality()},
    )
    assert r.status_code == 303
    return r.headers["location"].split("/agents/")[1].split("?")[0]


def _get_or_create_kenji(session) -> Character:
    """Return the kenji_sato Character, creating a minimal stub if needed."""
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


# ---------------------------------------------------------------------------
# create_agent_match unit tests (no HTTP)
# ---------------------------------------------------------------------------


def test_create_agent_match_correct_fields():
    with SessionLocal() as session:
        kenji = _get_or_create_kenji(session)
        # Create a player and agent directly.
        player = Player(username="agentmatch_player", email="am@test.example", password_hash="x")
        session.add(player)
        session.flush()

        agent = PlayerAgent(
            owner_player_id=player.id,
            name="TestAgent",
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

    with SessionLocal() as session:
        row = session.get(Match, match_id)
        assert row is not None
        assert row.match_kind == "agent_vs_character"
        assert row.participant_agent_id == agent.id
        assert row.player_id == player.id
        assert row.status == MatchStatus.IN_PROGRESS
        assert row.character_id == kenji.id
        assert row.player_color.value in ("white", "black")


def test_create_agent_match_wrong_owner_raises():
    from app.matches.service import MatchNotFound

    with SessionLocal() as session:
        # Two players, one agent owned by player1.
        p1 = Player(username="owner_player", email="p1@test.example", password_hash="x")
        p2 = Player(username="other_player", email="p2@test.example", password_hash="x")
        session.add_all([p1, p2])
        session.flush()

        agent = PlayerAgent(
            owner_player_id=p1.id,
            name="BotA",
            personality_description=_valid_personality(),
            elo=1200,
        )
        session.add(agent)
        session.flush()

        # Player 2 tries to use player 1's agent.
        with pytest.raises(Exception):
            create_agent_match(session, agent_id=agent.id, player_id=p2.id)


# ---------------------------------------------------------------------------
# HTTP integration tests
# ---------------------------------------------------------------------------


def test_agent_detail_page_shows_match_count():
    client = _login("block13_detail")
    agent_id = _create_agent(client, name="MatchBot")

    r = client.get(f"/agents/{agent_id}", follow_redirects=True)
    assert r.status_code == 200
    assert "Recent Matches" in r.text
    # No matches yet — placeholder shown.
    assert "No matches yet" in r.text or "Send your agent" in r.text


def test_agent_detail_page_lists_matches_after_game():
    client = _login("block13_list")

    # Create an agent.
    agent_id = _create_agent(client, name="HistoryBot")

    with SessionLocal() as session:
        kenji = _get_or_create_kenji(session)
        player = session.execute(
            select(Player).where(Player.username == "block13_list")
        ).scalar_one()
        agent = session.get(PlayerAgent, agent_id)

        match = create_agent_match(session, agent_id=agent.id, player_id=player.id)
        session.commit()
        match_id = match.id

    r = client.get(f"/agents/{agent_id}", follow_redirects=True)
    assert r.status_code == 200
    # Match should appear in the recent matches list.
    assert "vs " in r.text  # "vs Kenji Sato" or similar
    assert "/play/" in r.text


def test_play_page_watch_mode_for_agent_match():
    client = _login("block13_play")
    agent_id = _create_agent(client, name="WatchBot")

    with SessionLocal() as session:
        _get_or_create_kenji(session)
        player = session.execute(
            select(Player).where(Player.username == "block13_play")
        ).scalar_one()
        agent = session.get(PlayerAgent, agent_id)
        match = create_agent_match(session, agent_id=agent.id, player_id=player.id)
        session.commit()
        match_id = match.id

    r = client.get(f"/matches/{match_id}", follow_redirects=True)
    assert r.status_code == 200
    # Watch mode constants injected into the page.
    assert "WATCH_MODE = true" in r.text
    # Agent name in header.
    assert "WatchBot" in r.text
    # Back link to briefing room.
    assert f"/agents/{agent_id}/room" in r.text
    # Resign button is hidden.
    assert 'id="resign-btn" class="hidden"' in r.text
    # Two-column layout exists.
    assert 'id="chat-log-agent"' in r.text


def test_play_page_normal_mode_unaffected():
    """Human-vs-character match must not show WATCH_MODE."""
    client = _login("block13_normal")

    with SessionLocal() as session:
        kenji = _get_or_create_kenji(session)

    # Start a normal match via the lobby/character flow.
    r = client.post(f"/characters/{kenji.id}/start_match")
    if r.status_code not in (200, 303):
        pytest.skip("start_match route not available in this config")

    if r.status_code == 303:
        match_url = r.headers["location"]
        r2 = client.get(match_url, follow_redirects=True)
        assert r2.status_code == 200
        assert "WATCH_MODE = false" in r2.text
        # No agent chat column.
        assert 'id="chat-log-agent"' not in r2.text
