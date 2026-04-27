"""Test-suite bootstrap.

Runs BEFORE any `app.*` module is imported so the env vars pick up the
test database. Without this, `app.db` would grab the production SQLite
URL at import time.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

_TMP_DIR = Path(tempfile.mkdtemp(prefix="metropolis_tests_"))
_TEST_DB_PATH = _TMP_DIR / "test.db"

os.environ.setdefault("DATABASE_URL", f"sqlite:///{_TEST_DB_PATH.as_posix()}")
os.environ.setdefault("GEMINI_API_KEY", "test-key-not-real")
os.environ.setdefault("LOG_DIR", str(_TMP_DIR / "logs"))

import pytest  # noqa: E402
from sqlalchemy import delete  # noqa: E402

from app.db import SessionLocal, engine, init_db  # noqa: E402
from app.models.auth import PasswordResetToken  # noqa: E402
from app.models.chat import CharacterChatSession, CharacterChatTurn  # noqa: E402
from app.models.character import Character  # noqa: E402
from app.models.clay_balance import ClayBalance  # noqa: E402
from app.models.clay_transaction import ClayTransaction  # noqa: E402
from app.models.evolution import CharacterEvolutionState  # noqa: E402
from app.models.lobby import (  # noqa: E402
    Lobby,
    LobbyMembership,
    MatchmakingQueue,
    PvpMatch,
)
from app.models.match import Match, MatchAnalysis, Move, OpponentProfile, Player  # noqa: E402
from app.models.memory import Memory  # noqa: E402
from app.models.player_agent import PlayerAgent  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _bootstrap_db() -> None:
    init_db()


@pytest.fixture(autouse=True)
def _clean_tables():
    """Wipe every row before each test for deterministic state."""
    with engine.begin() as conn:
        conn.execute(delete(MatchAnalysis))
        conn.execute(delete(Move))
        conn.execute(delete(ClayTransaction))
        conn.execute(delete(ClayBalance))
        conn.execute(delete(Match))
        conn.execute(delete(PlayerAgent))
        conn.execute(delete(OpponentProfile))
        conn.execute(delete(PvpMatch))
        conn.execute(delete(MatchmakingQueue))
        conn.execute(delete(LobbyMembership))
        conn.execute(delete(Lobby))
        conn.execute(delete(CharacterChatTurn))
        conn.execute(delete(CharacterChatSession))
        conn.execute(delete(CharacterEvolutionState))
        conn.execute(delete(PasswordResetToken))
        conn.execute(delete(Player))
        conn.execute(delete(Memory))
        conn.execute(delete(Character))
    yield


@pytest.fixture
def db_session():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


# --- Phase 4.0a auth helpers --------------------------------------------
#
# Pre-4.0a tests just POSTed /login with a username to "become" that user —
# the route auto-created the row. Post-4.0a, /login requires credentials.
# `signup_and_login` does the signup step so test setup stays a one-liner.


def signup_and_login(client, username: str, *, email: str | None = None,
                     password: str = "testpass123"):
    """Create a player via /signup and leave `client` logged in as them.

    Returns the response from POST /signup. Idempotent-ish: if the
    username already exists, falls back to POST /login with the same
    password (matches real user flow).
    """
    if email is None:
        email = f"{username}@test.example"
    r = client.post(
        "/signup",
        data={
            "username": username,
            "email": email,
            "password": password,
            "password_confirm": password,
            "next": "/",
        },
    )
    if r.status_code == 303 and "error=" in r.headers.get("location", ""):
        # Fall back to login — account already exists.
        r = client.post(
            "/login",
            data={"identifier": username, "password": password, "next": "/"},
        )
    return r


@pytest.fixture
def signup_login():
    return signup_and_login
