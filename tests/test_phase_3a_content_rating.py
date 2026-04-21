"""Phase 3a: content-rating filtering and LLM prompt injection."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.characters.content_rating_prompts import rating_prompt_fragment
from app.characters.memory_generator import build_prompt
from app.db import SessionLocal
from app.main import create_app
from app.models.character import (
    Character,
    CharacterState,
    ContentRating,
    Visibility,
    rating_allowed,
    rating_level,
)
from app.models.match import Player
from tests.conftest import signup_and_login


def _client() -> TestClient:
    return TestClient(create_app(), follow_redirects=False)


def _login(username: str, *, rating: ContentRating | None = None) -> TestClient:
    c = _client()
    signup_and_login(c, username)
    if rating is not None:
        c.post("/settings", data={"display_name": username, "max_content_rating": rating.value})
    return c


def _make_char(
    *,
    rating: ContentRating,
    owner_id: str | None = None,
    name: str | None = None,
    visibility: Visibility = Visibility.PUBLIC,
) -> str:
    with SessionLocal() as s:
        c = Character(
            name=name or f"R_{rating.value}",
            short_description="x",
            backstory="A rich backstory for memory generation.",
            voice_descriptor="test voice",
            target_elo=1400,
            current_elo=1400,
            floor_elo=1400,
            max_elo=1800,
            owner_id=owner_id,
            visibility=visibility,
            content_rating=rating,
            state=CharacterState.READY,
        )
        s.add(c)
        s.commit()
        s.refresh(c)
        return c.id


# --- rating ordering helpers ---------------------------------------------


def test_rating_level_order():
    assert rating_level(ContentRating.FAMILY) < rating_level(ContentRating.MATURE)
    assert rating_level(ContentRating.MATURE) < rating_level(ContentRating.UNRESTRICTED)


def test_rating_allowed_matrix():
    # family player sees only family
    assert rating_allowed(ContentRating.FAMILY, ContentRating.FAMILY)
    assert not rating_allowed(ContentRating.MATURE, ContentRating.FAMILY)
    assert not rating_allowed(ContentRating.UNRESTRICTED, ContentRating.FAMILY)
    # mature player sees family + mature
    assert rating_allowed(ContentRating.FAMILY, ContentRating.MATURE)
    assert rating_allowed(ContentRating.MATURE, ContentRating.MATURE)
    assert not rating_allowed(ContentRating.UNRESTRICTED, ContentRating.MATURE)
    # unrestricted sees all
    assert rating_allowed(ContentRating.FAMILY, ContentRating.UNRESTRICTED)
    assert rating_allowed(ContentRating.MATURE, ContentRating.UNRESTRICTED)
    assert rating_allowed(ContentRating.UNRESTRICTED, ContentRating.UNRESTRICTED)


# --- filtering on /api/characters ----------------------------------------


def test_family_player_sees_only_family_rated_characters():
    _make_char(rating=ContentRating.FAMILY, name="Fam1")
    _make_char(rating=ContentRating.MATURE, name="Mat1")
    _make_char(rating=ContentRating.UNRESTRICTED, name="Unr1")
    c = _login("famuser", rating=ContentRating.FAMILY)
    rows = c.get("/api/characters").json()
    ratings = {r["content_rating"] for r in rows}
    assert ratings == {"family"} or ratings == {"family"}  # only family


def test_mature_player_sees_family_and_mature():
    _make_char(rating=ContentRating.FAMILY, name="Fam2")
    _make_char(rating=ContentRating.MATURE, name="Mat2")
    _make_char(rating=ContentRating.UNRESTRICTED, name="Unr2")
    c = _login("matuser", rating=ContentRating.MATURE)
    rows = c.get("/api/characters").json()
    ratings = {r["content_rating"] for r in rows}
    assert "family" in ratings and "mature" in ratings
    assert "unrestricted" not in ratings


def test_unrestricted_player_sees_all():
    _make_char(rating=ContentRating.FAMILY, name="Fam3")
    _make_char(rating=ContentRating.MATURE, name="Mat3")
    _make_char(rating=ContentRating.UNRESTRICTED, name="Unr3")
    c = _login("unruser", rating=ContentRating.UNRESTRICTED)
    rows = c.get("/api/characters").json()
    ratings = {r["content_rating"] for r in rows}
    assert ratings == {"family", "mature", "unrestricted"}


def test_accessing_hidden_character_directly_returns_403_hidden_page():
    # User is family, character is mature.
    c = _login("famuser2", rating=ContentRating.FAMILY)
    char_id = _make_char(rating=ContentRating.MATURE, name="Hidden")
    r = c.get(f"/characters/{char_id}")
    assert r.status_code == 403
    assert "Hidden by your content preference" in r.text


def test_api_accessing_rating_hidden_character_returns_404():
    c = _login("famuser3", rating=ContentRating.FAMILY)
    char_id = _make_char(rating=ContentRating.MATURE, name="Hidden2")
    r = c.get(f"/api/characters/{char_id}")
    assert r.status_code == 404


# --- LLM prompt injection (assert via string contains) ------------------


def test_memory_generator_prompt_includes_family_rating_block():
    with SessionLocal() as s:
        # reuse an existing character from fixture
        c = Character(
            name="Family Test",
            short_description="",
            backstory="A backstory long enough to matter.",
            voice_descriptor="",
            target_elo=1400,
            current_elo=1400,
            floor_elo=1400,
            max_elo=1800,
            content_rating=ContentRating.FAMILY,
            state=CharacterState.READY,
        )
        s.add(c)
        s.commit()
        s.refresh(c)
        prompt = build_prompt(c, target=40, minimum=30, maximum=50)
    assert "CONTENT RATING: FAMILY" in prompt
    assert "kid-friendly" in prompt.lower() or "Kid-friendly" in prompt


def test_memory_generator_prompt_includes_mature_rating_block():
    with SessionLocal() as s:
        c = Character(
            name="Mature Test",
            short_description="",
            backstory="A backstory long enough to matter.",
            target_elo=1400,
            current_elo=1400,
            floor_elo=1400,
            max_elo=1800,
            content_rating=ContentRating.MATURE,
            state=CharacterState.READY,
        )
        s.add(c)
        s.commit()
        s.refresh(c)
        prompt = build_prompt(c, target=40, minimum=30, maximum=50)
    assert "CONTENT RATING: MATURE" in prompt


def test_soul_system_prompt_includes_rating_block():
    from app.agents.prompts import build_system_prompt

    with SessionLocal() as s:
        c = Character(
            name="Unr Test",
            short_description="",
            backstory="x",
            target_elo=1400,
            current_elo=1400,
            floor_elo=1400,
            max_elo=1800,
            content_rating=ContentRating.UNRESTRICTED,
            state=CharacterState.READY,
        )
        s.add(c)
        s.commit()
        s.refresh(c)
        prompt = build_system_prompt(c)
    assert "CONTENT RATING: UNRESTRICTED" in prompt


def test_post_match_memory_prompt_includes_rating():
    from app.post_match.memory_gen import _build_memory_prompt
    from app.models.match import Match, MatchResult, MatchStatus, Color

    with SessionLocal() as s:
        c = Character(
            name="PM Test",
            short_description="",
            backstory="x",
            target_elo=1400,
            current_elo=1400,
            floor_elo=1400,
            max_elo=1800,
            content_rating=ContentRating.MATURE,
            state=CharacterState.READY,
        )
        s.add(c)
        s.commit()
        s.refresh(c)
        from app.auth import generate_guest_username

        p = Player(username=generate_guest_username(), display_name="PMP")
        s.add(p)
        s.commit()
        s.refresh(p)
        match = Match(
            character_id=c.id,
            player_id=p.id,
            status=MatchStatus.COMPLETED,
            result=MatchResult.DRAW,
            player_color=Color.WHITE,
            initial_fen="x",
            current_fen="x",
            move_count=20,
            character_elo_at_start=1400,
        )
        s.add(match)
        s.commit()
        s.refresh(match)
        prompt = _build_memory_prompt(
            character=c,
            match=match,
            critical_moments=[],
            features_before=None,
            features_after={},
            opponent_notes=[],
            prior_memory_samples=[],
        )
    assert "CONTENT RATING: MATURE" in prompt


def test_rating_prompt_fragment_distinct_per_level():
    a = rating_prompt_fragment(ContentRating.FAMILY)
    b = rating_prompt_fragment(ContentRating.MATURE)
    c = rating_prompt_fragment(ContentRating.UNRESTRICTED)
    assert a != b != c
    assert "FAMILY" in a
    assert "MATURE" in b
    assert "UNRESTRICTED" in c
