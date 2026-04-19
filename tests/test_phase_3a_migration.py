"""Phase 3a: alembic migration coverage.

Two scenarios:

1. Fresh DB: baseline 0001 is a no-op, 0002 is idempotent — both apply
   cleanly to an empty DB where create_all already materialized the 3a
   schema.

2. Pre-3a DB: fabricate a SQLite DB with the old schema (no username on
   players, no owner_id/visibility/content_rating on characters), seed a
   few rows, then run alembic from 0001 -> 0002. Assert the backfill:
   every existing player gets a guest_<...> username, a legacy_system
   player is created, ownerless non-preset characters are reassigned to
   legacy_system, preset content ratings are applied.
"""

from __future__ import annotations

import os
import sqlite3
import tempfile
from pathlib import Path

from alembic import command
from alembic.config import Config


def _alembic_cfg(db_url: str) -> Config:
    repo_root = Path(__file__).resolve().parents[1]
    cfg = Config(str(repo_root / "alembic.ini"))
    cfg.set_main_option("script_location", str(repo_root / "alembic"))
    cfg.set_main_option("sqlalchemy.url", db_url)
    return cfg


def test_alembic_upgrade_head_on_fresh_db_is_idempotent(tmp_path, monkeypatch):
    """Fresh DB where create_all already ran: upgrade head must be safe."""
    db_path = tmp_path / "fresh.db"
    url = f"sqlite:///{db_path.as_posix()}"
    monkeypatch.setenv("DATABASE_URL", url)

    # Run app's create_all to materialize the current schema.
    from sqlalchemy import create_engine
    from app.models.base import Base
    from app.models import character, match, memory  # noqa: F401

    eng = create_engine(url)
    Base.metadata.create_all(bind=eng)
    eng.dispose()

    cfg = _alembic_cfg(url)
    command.upgrade(cfg, "head")

    # Verify no crash and version table exists.
    con = sqlite3.connect(db_path)
    try:
        cur = con.execute("SELECT version_num FROM alembic_version")
        version = cur.fetchone()[0]
        assert version == "0005_player_elo"
    finally:
        con.close()


def test_alembic_upgrade_backfills_pre_3a_db(tmp_path, monkeypatch):
    """Simulated pre-3a DB: columns missing. upgrade head adds + backfills."""
    db_path = tmp_path / "pre3a.db"
    url = f"sqlite:///{db_path.as_posix()}"

    # Hand-build the pre-3a schema. Just the tables the migration touches.
    con = sqlite3.connect(db_path)
    con.executescript(
        """
        CREATE TABLE players (
            id VARCHAR(36) PRIMARY KEY,
            display_name VARCHAR(80) NOT NULL DEFAULT 'Guest',
            created_at DATETIME NOT NULL
        );
        CREATE TABLE characters (
            id VARCHAR(36) PRIMARY KEY,
            name VARCHAR(120) NOT NULL,
            short_description VARCHAR(280) NOT NULL DEFAULT '',
            backstory TEXT NOT NULL DEFAULT '',
            avatar_emoji VARCHAR(8) NOT NULL DEFAULT '',
            aggression INTEGER NOT NULL DEFAULT 5,
            risk_tolerance INTEGER NOT NULL DEFAULT 5,
            patience INTEGER NOT NULL DEFAULT 5,
            trash_talk INTEGER NOT NULL DEFAULT 5,
            target_elo INTEGER NOT NULL DEFAULT 1400,
            adaptive BOOLEAN NOT NULL DEFAULT 0,
            current_elo INTEGER NOT NULL DEFAULT 1400,
            floor_elo INTEGER NOT NULL DEFAULT 1400,
            max_elo INTEGER NOT NULL DEFAULT 1800,
            opening_preferences TEXT NOT NULL DEFAULT '[]',
            voice_descriptor VARCHAR(280) NOT NULL DEFAULT '',
            quirks TEXT NOT NULL DEFAULT '',
            state VARCHAR(19) NOT NULL DEFAULT 'ready',
            memory_generation_started_at DATETIME,
            memory_generation_error TEXT,
            is_preset BOOLEAN NOT NULL DEFAULT 0,
            preset_key VARCHAR(64) UNIQUE,
            deleted_at DATETIME,
            created_at DATETIME NOT NULL,
            updated_at DATETIME NOT NULL
        );
        """
    )
    # Seed some rows.
    con.execute(
        "INSERT INTO players (id, display_name, created_at) VALUES "
        "('p1', 'Olivier', '2026-04-01'), ('p2', 'Tester', '2026-04-02')"
    )
    con.execute(
        "INSERT INTO characters (id, name, backstory, avatar_emoji, "
        "is_preset, preset_key, created_at, updated_at) VALUES "
        "('c_viktor', 'Viktor Volkov', 'x', '♜', 1, 'viktor_volkov', '2026-04-01', '2026-04-01'),"
        "('c_margot', 'Margot Lindqvist', 'x', '♗', 1, 'margot_lindqvist', '2026-04-01', '2026-04-01'),"
        "('c_user', 'User Character', 'x', '♟', 0, NULL, '2026-04-02', '2026-04-02')"
    )
    con.commit()
    con.close()

    # Stamp at baseline, then upgrade.
    monkeypatch.setenv("DATABASE_URL", url)
    cfg = _alembic_cfg(url)
    command.stamp(cfg, "0001_initial_baseline")
    command.upgrade(cfg, "head")

    # Verify: migrations applied.
    con = sqlite3.connect(db_path)
    try:
        # Every player got a username.
        rows = con.execute("SELECT id, username FROM players ORDER BY id").fetchall()
        usernames = {pid: u for pid, u in rows}
        assert usernames["p1"].startswith("guest_")
        assert usernames["p2"].startswith("guest_")
        # legacy_system Player created.
        legacy = con.execute(
            "SELECT id FROM players WHERE username = 'legacy_system'"
        ).fetchone()
        assert legacy is not None
        legacy_id = legacy[0]

        # User-created character was reassigned to legacy; presets stay NULL.
        ownerless_non_preset = con.execute(
            "SELECT owner_id FROM characters WHERE id = 'c_user'"
        ).fetchone()
        assert ownerless_non_preset[0] == legacy_id
        viktor_owner = con.execute(
            "SELECT owner_id, content_rating FROM characters WHERE id = 'c_viktor'"
        ).fetchone()
        assert viktor_owner[0] is None
        assert viktor_owner[1] == "mature"
        margot = con.execute(
            "SELECT content_rating FROM characters WHERE id = 'c_margot'"
        ).fetchone()
        assert margot[0] == "family"

        # max_content_rating present and defaulted.
        mcr = con.execute(
            "SELECT max_content_rating FROM players WHERE id = 'p1'"
        ).fetchone()
        assert mcr[0] == "family"

        # Visibility defaulted to public.
        vis = con.execute(
            "SELECT visibility FROM characters WHERE id = 'c_user'"
        ).fetchone()
        assert vis[0] == "public"
    finally:
        con.close()


def test_alembic_upgrade_rerun_is_idempotent(tmp_path, monkeypatch):
    """Running upgrade twice must not error."""
    db_path = tmp_path / "rerun.db"
    url = f"sqlite:///{db_path.as_posix()}"
    monkeypatch.setenv("DATABASE_URL", url)

    from sqlalchemy import create_engine
    from app.models.base import Base
    from app.models import character, match, memory  # noqa: F401

    eng = create_engine(url)
    Base.metadata.create_all(bind=eng)
    eng.dispose()

    cfg = _alembic_cfg(url)
    command.upgrade(cfg, "head")
    # Running again is fine.
    command.upgrade(cfg, "head")
