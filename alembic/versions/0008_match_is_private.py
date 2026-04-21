"""Phase 4.0b: add `matches.is_private` flag.

Private matches (code-gated PvP lobbies) skip the Elo ratchet and
evolution-learning pipeline to prevent Elo farming with friends. This
migration just adds the column, defaulted to False — existing rows stay
public (correct: no pre-4.0b match was ever private).

Revision ID: 0008_match_is_private
Revises: 0007_auth_credentials
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect

revision: str = "0008_match_is_private"
down_revision: Union[str, None] = "0007_auth_credentials"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _columns(table: str) -> set[str]:
    bind = op.get_bind()
    insp = inspect(bind)
    if table not in insp.get_table_names():
        return set()
    return {c["name"] for c in insp.get_columns(table)}


def upgrade() -> None:
    matches_cols = _columns("matches")
    if matches_cols and "is_private" not in matches_cols:
        op.add_column(
            "matches",
            sa.Column(
                "is_private",
                sa.Boolean(),
                nullable=False,
                server_default="0",
            ),
        )


def downgrade() -> None:
    if "is_private" in _columns("matches"):
        op.drop_column("matches", "is_private")
