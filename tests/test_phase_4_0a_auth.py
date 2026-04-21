"""Phase 4.0a: password reset flow, /settings/password, /settings/email,
Argon2 hashing properties.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.auth import (
    hash_password,
    issue_password_reset_token,
    verify_password,
)
from app.db import SessionLocal
from app.main import create_app
from app.models.auth import PasswordResetToken
from app.models.match import Player
from tests.conftest import signup_and_login


def _client() -> TestClient:
    return TestClient(create_app(), follow_redirects=False)


# --- hashing primitives ---------------------------------------------------


def test_hash_password_produces_distinct_salted_hashes():
    h1 = hash_password("same-password")
    h2 = hash_password("same-password")
    assert h1 != h2
    assert h1.startswith("$argon2")
    assert verify_password("same-password", h1)
    assert verify_password("same-password", h2)


def test_verify_password_rejects_wrong():
    h = hash_password("correct-horse")
    assert verify_password("correct-horse", h) is True
    assert verify_password("wrong-horse", h) is False
    assert verify_password("", h) is False


def test_verify_password_with_null_hash_returns_false():
    assert verify_password("any", None) is False


# --- forgot-password / reset flow ----------------------------------------


def test_forgot_password_issues_token_for_known_email():
    c = _client()
    signup_and_login(c, "reset_user", email="reset@test.example")

    # Fresh client (simulating a logged-out request).
    c2 = _client()
    r = c2.post("/forgot-password", data={"email": "reset@test.example"})
    assert r.status_code == 303
    assert "flash=sent" in r.headers["location"]

    with SessionLocal() as s:
        tokens = s.query(PasswordResetToken).all()
        assert len(tokens) == 1
        assert tokens[0].used_at is None


def test_forgot_password_does_not_leak_unknown_email():
    c = _client()
    r = c.post("/forgot-password", data={"email": "nobody@nowhere.local"})
    assert r.status_code == 303
    # Same flash as success → no enumeration.
    assert "flash=sent" in r.headers["location"]

    with SessionLocal() as s:
        assert s.query(PasswordResetToken).count() == 0


def test_reset_password_end_to_end():
    # Sign up a user.
    c = _client()
    signup_and_login(c, "resetter", email="resetter@test.example")

    # Issue a token directly (avoids having to parse the dev outbox).
    with SessionLocal() as s:
        p = s.query(Player).filter(Player.username == "resetter").one()
        raw_token = issue_password_reset_token(s, p)
        s.commit()

    # Consume the token: POST the new password.
    c2 = _client()
    r = c2.post(
        f"/reset-password/{raw_token}",
        data={"password": "newstrongpass", "password_confirm": "newstrongpass"},
    )
    assert r.status_code == 303
    assert "flash=password_reset" in r.headers["location"]
    assert c2.cookies.get("player_id")

    # Old password no longer works.
    c3 = _client()
    r = c3.post(
        "/login",
        data={"identifier": "resetter", "password": "testpass123", "next": "/"},
    )
    assert "error=bad_credentials" in r.headers["location"]

    # New one does.
    c4 = _client()
    r = c4.post(
        "/login",
        data={"identifier": "resetter", "password": "newstrongpass", "next": "/"},
    )
    assert r.status_code == 303
    assert c4.cookies.get("player_id")


def test_reset_token_is_single_use():
    c = _client()
    signup_and_login(c, "onceonly", email="once@test.example")
    with SessionLocal() as s:
        p = s.query(Player).filter(Player.username == "onceonly").one()
        raw = issue_password_reset_token(s, p)
        s.commit()

    c2 = _client()
    r1 = c2.post(
        f"/reset-password/{raw}",
        data={"password": "firstpass1", "password_confirm": "firstpass1"},
    )
    assert r1.status_code == 303
    assert "password_reset" in r1.headers["location"]

    # Second use — rejected.
    c3 = _client()
    r2 = c3.post(
        f"/reset-password/{raw}",
        data={"password": "secondpass", "password_confirm": "secondpass"},
    )
    assert r2.status_code == 303
    assert "reset_token_invalid" in r2.headers["location"]


def test_reset_rejects_password_mismatch():
    c = _client()
    signup_and_login(c, "mismatch_user", email="mismatch_user@test.example")
    with SessionLocal() as s:
        p = s.query(Player).filter(Player.username == "mismatch_user").one()
        raw = issue_password_reset_token(s, p)
        s.commit()

    c2 = _client()
    r = c2.post(
        f"/reset-password/{raw}",
        data={"password": "aaaaaaaa", "password_confirm": "bbbbbbbb"},
    )
    assert r.status_code == 303
    assert "error=password_mismatch" in r.headers["location"]


# --- /settings/password ---------------------------------------------------


def test_settings_change_password_requires_current():
    c = _client()
    signup_and_login(c, "changer")
    r = c.post(
        "/settings/password",
        data={
            "current_password": "WRONG",
            "new_password": "brandnewpass",
            "new_password_confirm": "brandnewpass",
        },
    )
    assert r.status_code == 303
    assert "error=wrong_current_password" in r.headers["location"]


def test_settings_change_password_happy_path():
    c = _client()
    signup_and_login(c, "happy", password="origpass1")
    r = c.post(
        "/settings/password",
        data={
            "current_password": "origpass1",
            "new_password": "newerpass2",
            "new_password_confirm": "newerpass2",
        },
    )
    assert r.status_code == 303
    assert "flash=password_saved" in r.headers["location"]

    # Verify via a fresh login.
    c2 = _client()
    r = c2.post(
        "/login",
        data={"identifier": "happy", "password": "newerpass2", "next": "/"},
    )
    assert r.status_code == 303
    assert c2.cookies.get("player_id")


def test_settings_set_password_for_legacy_user_without_current():
    """Legacy (password_hash=NULL) accounts can set an initial password
    without providing a `current_password`."""
    with SessionLocal() as s:
        legacy = Player(username="legacy_set", display_name="Legacy")
        s.add(legacy)
        s.commit()
        s.refresh(legacy)
        pid = legacy.id

    c = _client()
    c.cookies.set("player_id", pid)
    r = c.post(
        "/settings/password",
        data={
            "current_password": "",
            "new_password": "firsttime8",
            "new_password_confirm": "firsttime8",
        },
    )
    assert r.status_code == 303
    assert "flash=password_saved" in r.headers["location"]

    with SessionLocal() as s:
        refreshed = s.query(Player).filter(Player.id == pid).one()
        assert refreshed.password_hash


# --- /settings/email ------------------------------------------------------


def test_settings_change_email_happy_path():
    c = _client()
    signup_and_login(c, "emailer", email="old@test.example")
    r = c.post("/settings/email", data={"new_email": "new@test.example"})
    assert r.status_code == 303
    assert "flash=email_saved" in r.headers["location"]
    with SessionLocal() as s:
        p = s.query(Player).filter(Player.username == "emailer").one()
        assert p.email == "new@test.example"


def test_settings_change_email_rejects_duplicate():
    c1 = _client()
    signup_and_login(c1, "first", email="first@test.example")
    c2 = _client()
    signup_and_login(c2, "second", email="second@test.example")

    r = c2.post("/settings/email", data={"new_email": "first@test.example"})
    assert r.status_code == 303
    assert "error=email_taken" in r.headers["location"]


def test_settings_change_email_rejects_invalid():
    c = _client()
    signup_and_login(c, "bad_email_setter")
    r = c.post("/settings/email", data={"new_email": "not-an-email"})
    assert r.status_code == 303
    assert "error=email_invalid" in r.headers["location"]
