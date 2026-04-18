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
from app.models.character import Character  # noqa: E402
from app.models.match import Match, Move, OpponentProfile, Player  # noqa: E402
from app.models.memory import Memory  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _bootstrap_db() -> None:
    init_db()


@pytest.fixture(autouse=True)
def _clean_tables():
    """Wipe every row before each test for deterministic state."""
    with engine.begin() as conn:
        conn.execute(delete(Move))
        conn.execute(delete(Match))
        conn.execute(delete(OpponentProfile))
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
