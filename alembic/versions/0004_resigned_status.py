"""Patch Pass 1: split MatchStatus.RESIGNED off from ABANDONED + profile counters.

Adds:
  - MatchStatus value 'resigned' (distinct from 'abandoned')
  - opponent_profiles.resigned_count, opponent_profiles.abandoned_count

Schema impact:
  - SQLite: match_status column is TEXT — the enum value is a no-op at the
    schema level, enforcement is in SQLAlchemy.
  - Postgres: future-us will need to ALTER TYPE match_status ADD VALUE 'resigned'.
    Deferred — we're on SQLite today. When the Postgres migration lands, it
    should do the ADD VALUE then backfill any ABANDONED rows that were actually
    clean resigns (we can't tell retroactively; leave them as ABANDONED per the
    Patch Pass 1 spec).

Revision ID: 0004_resigned_status
Revises: 0003_phase_3c
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect

revision: str = "0004_resigned_status"
down_revision: Union[str, None] = "0003_phase_3c"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _columns(table: str) -> set[str]:
    bind = op.get_bind()
    insp = inspect(bind)
    if table not in insp.get_table_names():
        return set()
    return {c["name"] for c in insp.get_columns(table)}


def upgrade() -> None:
    # Tolerant: pre-2b schemas lack this table. Skip rather than fail — the
    # app's `create_all` will build the full current schema on first boot.
    existing = _columns("opponent_profiles")
    if not existing:
        return
    if "resigned_count" not in existing:
        op.add_column(
            "opponent_profiles",
            sa.Column(
                "resigned_count", sa.Integer(), nullable=False, server_default="0"
            ),
        )
    if "abandoned_count" not in existing:
        op.add_column(
            "opponent_profiles",
            sa.Column(
                "abandoned_count", sa.Integer(), nullable=False, server_default="0"
            ),
        )


def downgrade() -> None:
    existing = _columns("opponent_profiles")
    if "abandoned_count" in existing:
        op.drop_column("opponent_profiles", "abandoned_count")
    if "resigned_count" in existing:
        op.drop_column("opponent_profiles", "resigned_count")
