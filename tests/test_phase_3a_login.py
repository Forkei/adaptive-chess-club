"""Username/email login flow + cookie auth on protected routes.

Originally Phase 3a (username-only). Updated for Phase 4.0a: /login now
requires credentials, /signup creates the account. Legacy rows without a
`password_hash` still log in with identifier-only (the migration path for
pre-4.0a dev accounts).
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.auth import generate_guest_username
from app.db import SessionLocal
from app.main import create_app
from app.models.match import Player
from tests.conftest import signup_and_login


def _client() -> TestClient:
    return TestClient(create_app(), follow_redirects=False)


# --- /signup -------------------------------------------------------------


def test_signup_creates_new_player_with_email_and_password():
    c = _client()
    r = signup_and_login(c, "alice")
    assert r.status_code == 303
    assert c.cookies.get("player_id")
    assert "flash=welcome" in r.headers["location"]

    with SessionLocal() as s:
        p = s.query(Player).filter(Player.username == "alice").one()
        assert p.email == "alice@test.example"
        assert p.password_hash  # hashed, not plaintext
        assert p.password_hash != "testpass123"


def test_signup_rejects_duplicate_username():
    c = _client()
    signup_and_login(c, "bob")
    c2 = _client()
    r = c2.post(
        "/signup",
        data={
            "username": "bob",
            "email": "different@test.example",
            "password": "testpass123",
            "password_confirm": "testpass123",
            "next": "/",
        },
    )
    assert r.status_code == 303
    assert "error=username_taken" in r.headers["location"]


def test_signup_rejects_duplicate_email():
    c = _client()
    signup_and_login(c, "carol")
    c2 = _client()
    r = c2.post(
        "/signup",
        data={
            "username": "carol2",
            "email": "carol@test.example",
            "password": "testpass123",
            "password_confirm": "testpass123",
            "next": "/",
        },
    )
    assert r.status_code == 303
    assert "error=email_taken" in r.headers["location"]


def test_signup_rejects_password_mismatch():
    c = _client()
    r = c.post(
        "/signup",
        data={
            "username": "dave",
            "email": "dave@test.example",
            "password": "testpass123",
            "password_confirm": "different!",
            "next": "/",
        },
    )
    assert r.status_code == 303
    assert "error=password_mismatch" in r.headers["location"]


def test_signup_rejects_short_password():
    c = _client()
    r = c.post(
        "/signup",
        data={
            "username": "eve",
            "email": "eve@test.example",
            "password": "short",
            "password_confirm": "short",
            "next": "/",
        },
    )
    assert r.status_code == 303
    assert "error=password_too_short" in r.headers["location"]


# --- /login --------------------------------------------------------------


def test_login_with_username_and_password():
    c = _client()
    signup_and_login(c, "frank")

    # Fresh client (no cookie) — log in with credentials.
    c2 = _client()
    r = c2.post(
        "/login",
        data={"identifier": "frank", "password": "testpass123", "next": "/"},
    )
    assert r.status_code == 303
    assert c2.cookies.get("player_id")
    assert "welcome_back" in r.headers["location"]


def test_login_with_email():
    c = _client()
    signup_and_login(c, "grace", email="grace@example.com")

    c2 = _client()
    r = c2.post(
        "/login",
        data={"identifier": "grace@example.com", "password": "testpass123", "next": "/"},
    )
    assert r.status_code == 303
    assert c2.cookies.get("player_id")


def test_login_rejects_wrong_password():
    c = _client()
    signup_and_login(c, "henry")
    c2 = _client()
    r = c2.post(
        "/login",
        data={"identifier": "henry", "password": "WRONG-pass", "next": "/"},
    )
    assert r.status_code == 303
    assert "error=bad_credentials" in r.headers["location"]
    assert not c2.cookies.get("player_id")


def test_login_rejects_unknown_user():
    c = _client()
    r = c.post(
        "/login",
        data={"identifier": "ghost", "password": "anything!", "next": "/"},
    )
    assert r.status_code == 303
    assert "error=bad_credentials" in r.headers["location"]


def test_login_rejects_blank_identifier():
    c = _client()
    r = c.post("/login", data={"identifier": "  ", "password": "x", "next": "/"})
    assert r.status_code == 303
    assert "error=identifier_required" in r.headers["location"]


# --- legacy accounts (no password_hash) ----------------------------------


def test_legacy_account_logs_in_with_identifier_only():
    """Pre-4.0a rows have NULL password_hash — should still log in."""
    with SessionLocal() as s:
        legacy = Player(username="legacy_user", display_name="Legacy")
        s.add(legacy)
        s.commit()

    c = _client()
    r = c.post("/login", data={"identifier": "legacy_user", "next": "/"})
    assert r.status_code == 303
    assert c.cookies.get("player_id")
    # Should hint they need to set a password.
    assert "flash=needs_password" in r.headers["location"]


def test_guest_upgrade_via_signup_preserves_row():
    """Logged-in guest posts /signup → row is upgraded in place."""
    c = _client()
    # Seed a guest row + cookie.
    with SessionLocal() as s:
        guest = Player(
            username=generate_guest_username(), display_name="Guest"
        )
        s.add(guest)
        s.commit()
        s.refresh(guest)
        pid = guest.id
    c.cookies.set("player_id", pid)

    r = c.post(
        "/signup",
        data={
            "username": "iris",
            "email": "iris@test.example",
            "password": "testpass123",
            "password_confirm": "testpass123",
            "next": "/",
        },
    )
    assert r.status_code == 303
    assert "flash=upgraded" in r.headers["location"]

    with SessionLocal() as s:
        upgraded = s.query(Player).filter(Player.id == pid).one()
        assert upgraded.username == "iris"
        assert upgraded.email == "iris@test.example"
        assert upgraded.password_hash
        assert s.query(Player).count() == 1


# --- logout + protected routes -------------------------------------------


def test_logout_clears_cookie():
    c = _client()
    signup_and_login(c, "jack")
    assert c.cookies.get("player_id")
    r = c.get("/logout")
    assert r.status_code == 303
    assert r.headers["location"] == "/login"


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
    assert "next=/characters/new" in r.headers["location"] or "next=%2F" in r.headers["location"]


def test_landing_is_open_and_shows_login_cta_when_logged_out():
    c = _client()
    r = c.get("/")
    assert r.status_code == 200
    assert "Log in" in r.text


def test_landing_shows_characters_when_logged_in():
    c = _client()
    signup_and_login(c, "karen")
    r = c.get("/")
    assert r.status_code == 200
    assert "Characters" in r.text


def test_api_me_returns_current_player_when_logged_in():
    c = _client()
    signup_and_login(c, "liam")
    r = c.get("/api/me")
    assert r.status_code == 200
    body = r.json()
    assert body["username"] == "liam"
