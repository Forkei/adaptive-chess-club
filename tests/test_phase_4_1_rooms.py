"""Phase 4.1 — character rooms: theme metadata, route wiring, template rendering."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.characters.presets import ARCHIBALD, KENJI, MARGOT, PRESETS, VIKTOR
from app.characters.rooms import (
    ARCHIBALD_ROOM,
    DEFAULT_ROOM,
    EMOTIONS,
    KENJI_ROOM,
    MARGOT_ROOM,
    VIKTOR_ROOM,
    theme_for_character,
    theme_for_preset_key,
)
from app.db import SessionLocal
from app.main import create_app
from app.models.character import Character, CharacterState, ContentRating, Visibility
from app.models.match import Player
from tests.conftest import signup_and_login


# --- rooms module: unit tests --------------------------------------------


def test_each_preset_has_a_room_theme():
    """Every grandmaster preset must resolve to a non-default room."""
    for preset in PRESETS:
        room = theme_for_preset_key(preset.preset_key)
        assert room is not DEFAULT_ROOM, f"{preset.preset_key} falls through to default"


def test_default_theme_used_for_custom_character():
    class _Stub:
        preset_key = None

    room = theme_for_character(_Stub())
    assert room is DEFAULT_ROOM


def test_viktor_has_full_emotion_clip_set():
    """Viktor is the seeded emotion-video character; all 8 Soul emotions
    must map to a clip URL."""
    assert set(VIKTOR_ROOM.emotion_clips.keys()) == set(EMOTIONS)
    for url in VIKTOR_ROOM.emotion_clips.values():
        assert url.startswith("/static/characters/viktor_petrov/emotions/")


def test_presets_without_assets_fall_back_to_empty_clips():
    """Margot/Archibald still ship without emotion clips — the UI falls
    back to the text-only emotion indicator. Kenji now has assets
    (Phase 4.4d reel integration) so he's excluded from this check."""
    for r in (MARGOT_ROOM, ARCHIBALD_ROOM):
        assert r.emotion_clips == {}


def test_kenji_clips_cover_all_emotions():
    """Kenji's Phase 4.4d reel maps every Soul emotion to an MP4 URL."""
    assert set(KENJI_ROOM.emotion_clips.keys()) == set(EMOTIONS)
    for url in KENJI_ROOM.emotion_clips.values():
        assert url.startswith("/static/characters/kenji_sato/emotions/")


def test_css_vars_only_touch_mood_tokens():
    """Theme overrides must not collide with rating/conn-pill tokens."""
    forbidden_prefixes = ("--mp-rating-", "conn-")
    for r in (VIKTOR_ROOM, MARGOT_ROOM, KENJI_ROOM, ARCHIBALD_ROOM):
        for k in r.css_vars:
            for f in forbidden_prefixes:
                assert not k.startswith(f), f"{r.slug} overrides forbidden token {k}"


def test_preset_slugs_match_preset_keys():
    """Room slugs must equal preset_key so theme_for_preset_key lookups work."""
    mapping = {
        VIKTOR.preset_key: VIKTOR_ROOM.slug,
        MARGOT.preset_key: MARGOT_ROOM.slug,
        KENJI.preset_key: KENJI_ROOM.slug,
        ARCHIBALD.preset_key: ARCHIBALD_ROOM.slug,
    }
    for preset_key, slug in mapping.items():
        assert preset_key == slug


# --- template rendering: integration tests --------------------------------


def _client() -> TestClient:
    return TestClient(create_app(), follow_redirects=False)


def _seed_preset_character(preset_key: str = "viktor_petrov") -> str:
    """Create a minimum Character row mirroring a preset, so /characters/{id}
    resolves and pulls the correct room."""
    with SessionLocal() as s:
        c = Character(
            name="Viktor Petrov",
            short_description="Brooklyn bakery stairs.",
            backstory="A rich backstory for testing purposes.",
            voice_descriptor="Gruff",
            target_elo=2200,
            current_elo=2200,
            floor_elo=2200,
            max_elo=2200,
            adaptive=False,
            is_preset=True,
            preset_key=preset_key,
            owner_id=None,
            state=CharacterState.READY,
            visibility=Visibility.PUBLIC,
            content_rating=ContentRating.MATURE,
        )
        s.add(c)
        s.commit()
        s.refresh(c)
        return c.id


def test_detail_page_includes_viktor_theme_vars():
    c = _client()
    signup_and_login(c, "viewer1")
    # Viewer needs mature rating to see Viktor.
    c.post("/settings", data={"display_name": "viewer1", "max_content_rating": "mature"})

    cid = _seed_preset_character("viktor_petrov")
    r = c.get(f"/characters/{cid}")
    assert r.status_code == 200
    # Theme vars are rendered inline on <body>.
    assert 'data-theme="viktor_petrov"' in r.text
    assert "--mp-brass: #C58147" in r.text
    # Room tagline present.
    assert "Above the bakery" in r.text
    # Phase 4.4c — ambient is now always available: either a recorded
    # <audio id="mp-ambient"> track, or a synth fallback when no audio
    # file is shipped with the room. The toggle is always present.
    assert 'id="mp-ambient-toggle"' in r.text
    assert ('id="mp-ambient"' in r.text) or ("synth_ambient.js" in r.text)


def test_detail_page_custom_character_has_no_theme_attrs():
    """A non-preset character → DEFAULT_ROOM → no themed attrs, no tagline lede."""
    c = _client()
    signup_and_login(c, "viewer2")

    # Create a custom character owned by viewer2.
    with SessionLocal() as s:
        player = s.query(Player).filter(Player.username == "viewer2").one()
        char = Character(
            name="Custom",
            short_description="custom character",
            backstory="backstory text",
            voice_descriptor="voice",
            target_elo=1400,
            current_elo=1400,
            floor_elo=1400,
            max_elo=1400,
            adaptive=False,
            is_preset=False,
            owner_id=player.id,
            state=CharacterState.READY,
            visibility=Visibility.PUBLIC,
            content_rating=ContentRating.FAMILY,
        )
        s.add(char)
        s.commit()
        cid = char.id

    r = c.get(f"/characters/{cid}")
    assert r.status_code == 200
    # No data-theme attribute (or it's the default).
    assert 'data-theme="default"' not in r.text
    # Default room has no tagline lede.
    assert "You enter the room" not in r.text
    # Menu ambient audio plays on non-themed pages (base.html else branch).
    assert 'id="mp-ambient"' in r.text


def test_rooms_module_emotions_constant_matches_soul_schema():
    """Defensive: if someone reshapes the Soul Emotion literal, catch it."""
    from app.schemas.agents import Emotion
    from typing import get_args

    assert set(EMOTIONS) == set(get_args(Emotion))


def test_play_page_renders_emotion_stage_with_inline_src():
    """Fix 2 (demo-rescue): play.html must render the emotion-stage div
    with the neutral clip URL pre-populated in HTML so the stage isn't
    blank on first paint. Without this, we depend on the JS probe
    succeeding before the user gets a visual."""
    from app.matches.service import create_match
    from app.models.match import Match

    c = _client()
    signup_and_login(c, "play_emot_viewer")
    # Kenji is rated family → no settings bump required.
    cid = _seed_preset_character("kenji_sato")

    with SessionLocal() as s:
        player = s.query(Player).filter(Player.username == "play_emot_viewer").one()
        match = create_match(
            s, character_id=cid, player_id=player.id, player_color="white",
        )
        s.commit()
        match_id = match.id

    r = c.get(f"/matches/{match_id}")
    assert r.status_code == 200, r.text
    assert 'id="mp-emotion-stage"' in r.text
    # Neutral clip path must be in the rendered HTML — that's what
    # guarantees the stage shows a frame before JS runs.
    assert "/static/characters/kenji_sato/emotions/neutral.mp4" in r.text
    assert 'id="mp-emotion-a"' in r.text
