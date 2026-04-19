"""Initial baseline — represents the pre-3a schema.

Phase 3a is the point we adopted Alembic. This revision is deliberately a
no-op upgrade: the schema it represents (characters, players, matches,
moves, memories, opponent_profiles, match_analyses, with the column set
as of Phase 2b) was produced by the app's startup `create_all` path.

Usage:
- **Pre-3a dev DBs**: `alembic stamp 0001_initial_baseline` marks the DB
  as sitting at this revision, then `alembic upgrade head` applies 0002
  to add the Phase 3a columns and backfill data.
- **Fresh DBs**: `init_db()` (startup, `create_all`) builds the
  _current_ schema including 3a columns. Then `alembic stamp head` marks
  the DB at head; 0002 never runs against it because there's nothing to
  add.

Revision ID: 0001_initial_baseline
Revises:
"""
from __future__ import annotations

from typing import Sequence, Union

# revision identifiers, used by Alembic.
revision: str = "0001_initial_baseline"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # No-op: baseline. See module docstring.
    pass


def downgrade() -> None:
    # No-op: baseline has no structural changes to reverse.
    pass
