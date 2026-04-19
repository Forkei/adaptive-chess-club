"""Phase 3a: username login + character ownership + content rating.

Schema changes:
- players.username (unique, NOT NULL)  — backfilled as `guest_<short_uuid>`
- players.max_content_rating (enum, default 'family')
- characters.owner_id (FK players.id, nullable)
- characters.visibility (enum 'public'/'private', default 'public')
- characters.content_rating (enum 'family'/'mature'/'unrestricted', default 'family')

Data migration:
- Existing players without a username get `guest_<first 8 chars of id>`.
- A `legacy_system` Player is created (username=legacy_system,
  display_name='Legacy'). All existing non-preset characters whose
  owner_id is NULL are reassigned to this player so they survive the
  ownership rules. Presets remain owner_id=NULL.
- Preset content ratings applied: Viktor + Kenji -> mature; Margot +
  Archibald -> family. Seed code re-applies on startup, but we set them
  at migration time too for completeness.

The migration is idempotent: each column add is guarded against
already-existing columns (so running 0002 after a fresh `create_all`
that already produced the new columns is a no-op on the structural
side; the data step runs either way).

Revision ID: 0002_phase_3a
Revises: 0001_initial_baseline
"""
from __future__ import annotations

import uuid
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect

# revision identifiers, used by Alembic.
revision: str = "0002_phase_3a"
down_revision: Union[str, None] = "0001_initial_baseline"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


CONTENT_RATING_ENUM = sa.Enum(
    "family", "mature", "unrestricted", name="character_content_rating"
)
PLAYER_RATING_ENUM = sa.Enum(
    "family", "mature", "unrestricted", name="player_max_content_rating"
)
VISIBILITY_ENUM = sa.Enum("public", "private", name="character_visibility")


def _existing_columns(table: str) -> set[str]:
    bind = op.get_bind()
    insp = inspect(bind)
    if table not in insp.get_table_names():
        return set()
    return {c["name"] for c in insp.get_columns(table)}


def _short_uuid() -> str:
    return uuid.uuid4().hex[:8]


def upgrade() -> None:
    bind = op.get_bind()

    # --- players: username (unique, NOT NULL) + max_content_rating --------
    player_cols = _existing_columns("players")

    if "username" not in player_cols:
        with op.batch_alter_table("players") as batch:
            # Add as nullable first so we can backfill, then enforce NOT NULL +
            # unique below.
            batch.add_column(sa.Column("username", sa.String(length=24), nullable=True))

        # Backfill every existing player.
        rows = bind.execute(sa.text("SELECT id FROM players")).fetchall()
        used: set[str] = set()
        for (pid,) in rows:
            candidate = f"guest_{_short_uuid()}"
            while candidate in used:
                candidate = f"guest_{_short_uuid()}"
            used.add(candidate)
            bind.execute(
                sa.text("UPDATE players SET username = :u WHERE id = :id"),
                {"u": candidate, "id": pid},
            )

        with op.batch_alter_table("players") as batch:
            batch.alter_column("username", existing_type=sa.String(length=24), nullable=False)
            batch.create_unique_constraint("uq_players_username", ["username"])
            batch.create_index("ix_players_username", ["username"], unique=True)

    if "max_content_rating" not in player_cols:
        with op.batch_alter_table("players") as batch:
            batch.add_column(
                sa.Column(
                    "max_content_rating",
                    PLAYER_RATING_ENUM,
                    nullable=False,
                    server_default="family",
                )
            )

    # --- characters: owner_id + visibility + content_rating ---------------
    char_cols = _existing_columns("characters")

    if "owner_id" not in char_cols:
        # Add column without an inline FK — SQLite's batch_alter_table
        # insists on named constraints, and we don't need DB-level FK
        # enforcement (SQLAlchemy ORM holds the relationship; SQLite
        # doesn't enforce FKs by default anyway).
        with op.batch_alter_table("characters") as batch:
            batch.add_column(sa.Column("owner_id", sa.String(length=36), nullable=True))
            batch.create_index("ix_characters_owner_id", ["owner_id"])

    if "visibility" not in char_cols:
        with op.batch_alter_table("characters") as batch:
            batch.add_column(
                sa.Column(
                    "visibility",
                    VISIBILITY_ENUM,
                    nullable=False,
                    server_default="public",
                )
            )

    if "content_rating" not in char_cols:
        with op.batch_alter_table("characters") as batch:
            batch.add_column(
                sa.Column(
                    "content_rating",
                    CONTENT_RATING_ENUM,
                    nullable=False,
                    server_default="family",
                )
            )

    # --- data migration: legacy_system player + preset content ratings ----
    # Create legacy_system player if missing (idempotent).
    legacy_exists = bind.execute(
        sa.text("SELECT id FROM players WHERE username = :u"),
        {"u": "legacy_system"},
    ).fetchone()
    if legacy_exists is None:
        legacy_id = str(uuid.uuid4())
        bind.execute(
            sa.text(
                "INSERT INTO players (id, username, display_name, max_content_rating, created_at) "
                "VALUES (:id, :u, :dn, :r, CURRENT_TIMESTAMP)"
            ),
            {
                "id": legacy_id,
                "u": "legacy_system",
                "dn": "Legacy",
                "r": "unrestricted",
            },
        )
    else:
        legacy_id = legacy_exists[0]

    # Reassign existing ownerless, non-preset characters to legacy_system.
    bind.execute(
        sa.text(
            "UPDATE characters SET owner_id = :legacy "
            "WHERE owner_id IS NULL AND is_preset = 0"
        ),
        {"legacy": legacy_id},
    )

    # Preset content ratings (match presets.py).
    preset_ratings = {
        "viktor_volkov": "mature",
        "kenji_sato": "mature",
        "margot_lindqvist": "family",
        "archibald_finch": "family",
    }
    for key, rating in preset_ratings.items():
        bind.execute(
            sa.text(
                "UPDATE characters SET content_rating = :r WHERE preset_key = :k"
            ),
            {"r": rating, "k": key},
        )


def downgrade() -> None:
    # Best-effort reverse. Does not recreate lost data.
    char_cols = _existing_columns("characters")
    if "content_rating" in char_cols:
        with op.batch_alter_table("characters") as batch:
            batch.drop_column("content_rating")
    if "visibility" in char_cols:
        with op.batch_alter_table("characters") as batch:
            batch.drop_column("visibility")
    if "owner_id" in char_cols:
        with op.batch_alter_table("characters") as batch:
            try:
                batch.drop_index("ix_characters_owner_id")
            except Exception:
                pass
            batch.drop_column("owner_id")

    player_cols = _existing_columns("players")
    if "max_content_rating" in player_cols:
        with op.batch_alter_table("players") as batch:
            batch.drop_column("max_content_rating")
    if "username" in player_cols:
        with op.batch_alter_table("players") as batch:
            try:
                batch.drop_index("ix_players_username")
            except Exception:
                pass
            try:
                batch.drop_constraint("uq_players_username", type_="unique")
            except Exception:
                pass
            batch.drop_column("username")
