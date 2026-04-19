"""Phase 3c: discovery + leaderboard indexes.

Adds indexes on matches.status and matches.ended_at to speed up:
  - discovery queries: `WHERE status = 'in_progress'` ordered by started_at desc
  - discovery queries: `WHERE status IN ('completed', 'abandoned')` ordered by ended_at desc
  - leaderboard windowed filters: `WHERE ended_at > :cutoff`

Idempotent: skips if the index already exists (SQLite reflection).

Revision ID: 0003_phase_3c
Revises: 0002_phase_3a
"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op
from sqlalchemy import inspect

# revision identifiers, used by Alembic.
revision: str = "0003_phase_3c"
down_revision: Union[str, None] = "0002_phase_3a"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _table_exists(table: str) -> bool:
    bind = op.get_bind()
    insp = inspect(bind)
    return table in insp.get_table_names()


def _existing_indexes(table: str) -> set[str]:
    bind = op.get_bind()
    insp = inspect(bind)
    if table not in insp.get_table_names():
        return set()
    return {ix["name"] for ix in insp.get_indexes(table)}


def upgrade() -> None:
    # Migration 0002's hand-fabricated pre-3a test DB only creates `players`
    # and `characters`; the `matches` table is materialized later by
    # create_all. Gracefully skip the indexing step on such DBs — it's a no-op
    # by intent, not an error.
    if not _table_exists("matches"):
        return
    existing = _existing_indexes("matches")
    if "ix_matches_status" not in existing:
        op.create_index("ix_matches_status", "matches", ["status"])
    if "ix_matches_ended_at" not in existing:
        op.create_index("ix_matches_ended_at", "matches", ["ended_at"])


def downgrade() -> None:
    if not _table_exists("matches"):
        return
    existing = _existing_indexes("matches")
    if "ix_matches_ended_at" in existing:
        op.drop_index("ix_matches_ended_at", table_name="matches")
    if "ix_matches_status" in existing:
        op.drop_index("ix_matches_status", table_name="matches")
