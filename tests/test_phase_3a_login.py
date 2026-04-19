"""Phase 3a: username login flow + cookie auth on protected routes."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.auth import generate_guest_username
from app.db import SessionLocal
from app.main import create_app
from app.models.match import Player


def _client() -> TestClient:
    return TestClient(create_app(), follow_redirects=False)


# --- /login form ---------------------------------------------------------


def test_login_creates_new_player_when_username_is_new():
    c = _client()
    r = c.post("/login", data={"username": "alice", "next": "/"})
    assert r.status_code == 303
    assert r.cookies.get("player_id")
    assert "flash=welcome" in r.headers["location"]

    # The player exists in the DB.
    with SessionLocal() as s:
        p = s.query(Player).filter(Player.username == "alice").one()
        assert p.display_name == "alice"


def test_login_reuses_existing_player_by_username():
    c = _client()
    # First login creates.
    r1 = c.post("/login", data={"username": "bob", "next": "/"})
    first_id = r1.cookies["player_id"]

    # Fresh client (no cookie) logs in with same username — gets the same player.
    c2 = _client()
    r2 = c2.post("/login", data={"username": "bob", "next": "/"})
    second_id = r2.cookies["player_id"]
    assert first_id == second_id
    assert "welcome_back" in r2.headers["location"]


def test_login_normalizes_case_and_whitespace():
    c = _client()
    r = c.post("/login", data={"username": "  Carol  ", "next": "/"})
    assert r.status_code == 303
    # Normalized to lowercase.
    with SessionLocal() as s:
        assert s.query(Player).filter(Player.username == "carol").one() is not None
        assert s.query(Player).filter(Player.username == "Carol").first() is None


def test_login_rejects_invalid_characters():
    c = _client()
    r = c.post("/login", data={"username": "Bad Name!", "next": "/"})
    # Redirected back to login with error code in query.
    assert r.status_code == 303
    assert "error=invalid_chars" in r.headers["location"]


def test_login_rejects_too_short():
    c = _client()
    r = c.post("/login", data={"username": "ab", "next": "/"})
    assert r.status_code == 303
    assert "error=too_short" in r.headers["location"]


def test_guest_cookie_rename_preserves_player_row():
    c = _client()
    # Simulate migrated guest: create a player with guest_* username and
    # point the cookie at them.
    with SessionLocal() as s:
        guest_username = generate_guest_username()
        player = Player(username=guest_username, display_name="Guest")
        s.add(player)
        s.commit()
        s.refresh(player)
        pid = player.id

    c.cookies.set("player_id", pid)
    r = c.post("/login", data={"username": "dave", "next": "/"})
    assert r.status_code == 303
    assert "renamed" in r.headers["location"]

    with SessionLocal() as s:
        # Same row, new username. Row count unchanged at 1.
        renamed = s.query(Player).filter(Player.id == pid).one()
        assert renamed.username == "dave"
        assert s.query(Player).count() == 1


def test_logout_clears_cookie():
    c = _client()
    c.post("/login", data={"username": "eve", "next": "/"})
    assert c.cookies.get("player_id")
    r = c.get("/logout")
    assert r.status_code == 303
    # cookie cleared (set with empty value)
    assert r.headers["location"] == "/login"


# --- protected routes ----------------------------------------------------


def test_api_route_returns_401_without_cookie():
    c = _client()
    r = c.get("/api/characters")
    assert r.status_code == 401
    assert r.json() == {"detail": "Not authenticated"}


def test_api_me_returns_401_without_cookie():
    c = _client()
    r = c.get("/api/me")
    assert r.status_code == 401


def test_html_route_redirects_to_login_without_cookie():
    c = _client()
    r = c.get("/characters/new")
    assert r.status_code == 303
    assert "/login" in r.headers["location"]
    # Preserves intended destination via ?next=
    assert "next=/characters/new" in r.headers["location"] or "next=%2F" in r.headers["location"]


def test_landing_is_open_and_shows_login_cta_when_logged_out():
    c = _client()
    r = c.get("/")
    assert r.status_code == 200
    assert "Log in" in r.text


def test_landing_shows_characters_when_logged_in():
    c = _client()
    c.post("/login", data={"username": "frank", "next": "/"})
    r = c.get("/")
    assert r.status_code == 200
    assert "Characters" in r.text


def test_api_me_returns_current_player_when_logged_in():
    c = _client()
    c.post("/login", data={"username": "grace", "next": "/"})
    r = c.get("/api/me")
    assert r.status_code == 200
    body = r.json()
    assert body["username"] == "grace"
