"""Phase 3a: character ownership, visibility, clone, preset protection."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.db import SessionLocal
from app.main import create_app
from app.models.character import Character, CharacterState, ContentRating, Visibility
from app.models.match import Player
from tests.conftest import signup_and_login


def _client() -> TestClient:
    return TestClient(create_app(), follow_redirects=False)


def _login(username: str) -> TestClient:
    c = _client()
    signup_and_login(c, username)
    return c


def _make_character(
    *,
    owner_id: str | None,
    visibility: Visibility = Visibility.PUBLIC,
    content_rating: ContentRating = ContentRating.FAMILY,
    is_preset: bool = False,
    name: str = "TestChar",
) -> str:
    with SessionLocal() as s:
        c = Character(
            name=name,
            short_description="x",
            backstory="x",
            target_elo=1400,
            current_elo=1400,
            floor_elo=1400,
            max_elo=1800,
            owner_id=owner_id,
            visibility=visibility,
            content_rating=content_rating,
            is_preset=is_preset,
            preset_key=f"test_{name}" if is_preset else None,
            state=CharacterState.READY,
        )
        s.add(c)
        s.commit()
        s.refresh(c)
        return c.id


def _player_id(c: TestClient) -> str:
    return c.get("/api/me").json()["id"]


# --- ownership: edit + delete -------------------------------------------


def test_owner_can_patch_character():
    alice = _login("alice")
    char_id = _make_character(owner_id=_player_id(alice))

    r = alice.patch(f"/api/characters/{char_id}", json={"short_description": "new desc"})
    assert r.status_code == 200
    assert r.json()["short_description"] == "new desc"


def test_non_owner_gets_403_on_patch():
    alice = _login("alice2")
    char_id = _make_character(owner_id=_player_id(alice))
    bob = _login("bob2")
    r = bob.patch(f"/api/characters/{char_id}", json={"short_description": "hacked"})
    assert r.status_code == 403


def test_owner_can_delete_character():
    alice = _login("alice3")
    char_id = _make_character(owner_id=_player_id(alice))
    r = alice.delete(f"/api/characters/{char_id}")
    assert r.status_code == 204
    # Now hidden even from the owner.
    r2 = alice.get(f"/api/characters/{char_id}")
    assert r2.status_code == 404


def test_non_owner_cannot_delete():
    alice = _login("alice4")
    char_id = _make_character(owner_id=_player_id(alice))
    bob = _login("bob4")
    r = bob.delete(f"/api/characters/{char_id}")
    assert r.status_code == 403


# --- visibility ---------------------------------------------------------


def test_private_character_hidden_from_non_owner():
    alice = _login("alice5")
    char_id = _make_character(
        owner_id=_player_id(alice), visibility=Visibility.PRIVATE
    )
    bob = _login("bob5")
    r = bob.get(f"/api/characters/{char_id}")
    assert r.status_code == 404
    # Not in the listing either.
    listing = bob.get("/api/characters").json()
    ids = [c["id"] for c in listing]
    assert char_id not in ids


def test_private_character_visible_to_owner():
    alice = _login("alice6")
    char_id = _make_character(
        owner_id=_player_id(alice), visibility=Visibility.PRIVATE
    )
    r = alice.get(f"/api/characters/{char_id}")
    assert r.status_code == 200
    listing = alice.get("/api/characters").json()
    ids = [c["id"] for c in listing]
    assert char_id in ids


# --- clone --------------------------------------------------------------


def test_clone_creates_character_owned_by_current_player():
    alice = _login("alice7")
    source_id = _make_character(owner_id=_player_id(alice), name="Source")

    bob = _login("bob7")
    r = bob.post(f"/api/characters/{source_id}/clone")
    assert r.status_code == 202
    clone = r.json()
    assert clone["id"] != source_id
    assert clone["owner_id"] == _player_id(bob)
    assert "clone" in clone["name"].lower()


def test_clone_resets_elo_state():
    alice = _login("alice8")
    # Source with mutated current/floor (simulating games played).
    with SessionLocal() as s:
        src = Character(
            name="Drifted",
            short_description="x",
            backstory="x",
            target_elo=1400,
            current_elo=1600,  # drifted up
            floor_elo=1500,
            max_elo=1800,
            owner_id=_player_id(alice),
            visibility=Visibility.PUBLIC,
            content_rating=ContentRating.FAMILY,
            state=CharacterState.READY,
        )
        s.add(src)
        s.commit()
        s.refresh(src)
        src_id = src.id

    r = alice.post(f"/api/characters/{src_id}/clone")
    assert r.status_code == 202
    clone = r.json()
    assert clone["current_elo"] == 1400  # reset to target
    assert clone["floor_elo"] == 1400


def test_clone_leaves_source_unchanged():
    alice = _login("alice9")
    source_id = _make_character(owner_id=_player_id(alice))
    bob = _login("bob9")
    bob.post(f"/api/characters/{source_id}/clone")
    # Source still owned by alice.
    with SessionLocal() as s:
        src = s.get(Character, source_id)
        assert src.owner_id == _player_id(alice)


def test_clone_while_source_is_still_generating_allowed():
    """Per phase_3a_decisions.md: clone is independent of source state."""
    alice = _login("alice10")
    # Source in GENERATING state.
    with SessionLocal() as s:
        src = Character(
            name="MidGen",
            short_description="x",
            backstory="x",
            target_elo=1400,
            current_elo=1400,
            floor_elo=1400,
            max_elo=1800,
            owner_id=_player_id(alice),
            visibility=Visibility.PUBLIC,
            content_rating=ContentRating.FAMILY,
            state=CharacterState.GENERATING_MEMORIES,  # still generating!
        )
        s.add(src)
        s.commit()
        s.refresh(src)
        src_id = src.id

    bob = _login("bob10")
    r = bob.post(f"/api/characters/{src_id}/clone")
    assert r.status_code == 202  # no 409


# --- preset protection --------------------------------------------------


def test_preset_cannot_be_deleted_even_by_any_user():
    preset_id = _make_character(owner_id=None, is_preset=True, name="FakePreset1")
    alice = _login("alice11")
    r = alice.delete(f"/api/characters/{preset_id}")
    assert r.status_code == 403


def test_preset_cannot_be_patched():
    preset_id = _make_character(owner_id=None, is_preset=True, name="FakePreset2")
    alice = _login("alice12")
    r = alice.patch(f"/api/characters/{preset_id}", json={"short_description": "pwnd"})
    assert r.status_code == 403
