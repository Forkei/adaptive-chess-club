"""Phase 4.3 — character_evolution_state.

One row per character. Drift applied after each post-match run.
See docs/phase_4_evolution.md.

Revision ID: 0012_character_evolution_state
Revises: 0011_lobby_time_control
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect

revision: str = "0012_character_evolution_state"
down_revision: Union[str, None] = "0011_lobby_time_control"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _has_table(name: str) -> bool:
    return name in inspect(op.get_bind()).get_table_names()


def upgrade() -> None:
    if not _has_table("character_evolution_state"):
        op.create_table(
            "character_evolution_state",
            sa.Column(
                "character_id",
                sa.String(length=36),
                sa.ForeignKey("characters.id", ondelete="CASCADE"),
                primary_key=True,
            ),
            sa.Column("slider_drift", sa.JSON(), nullable=False),
            sa.Column("opening_scores", sa.JSON(), nullable=False),
            sa.Column("trap_memory", sa.JSON(), nullable=False),
            sa.Column("tone_drift", sa.JSON(), nullable=False),
            sa.Column("matches_processed", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("last_match_id", sa.String(length=36), nullable=True),
            sa.Column("last_updated_at", sa.DateTime(), nullable=False),
        )


def downgrade() -> None:
    if _has_table("character_evolution_state"):
        op.drop_table("character_evolution_state")
