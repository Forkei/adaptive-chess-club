"""Block 12 — player_agents table.

One agent per row; owner_player_id FK to players.
Archived agents are soft-deleted (archived_at IS NOT NULL).

Revision ID: 0016_player_agents
Revises: 0015_feedback
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect


def _has_table(name: str) -> bool:
    return name in inspect(op.get_bind()).get_table_names()


revision: str = "0016_player_agents"
down_revision: Union[str, None] = "0015_feedback"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    if not _has_table("player_agents"):
        op.create_table(
            "player_agents",
            sa.Column("id", sa.String(36), primary_key=True),
            # index=True tells op.create_table() to emit CREATE INDEX automatically.
            # Do NOT also call op.create_index() for the same column — that would
            # double-create the index and raise "index already exists" on SQLite.
            sa.Column(
                "owner_player_id",
                sa.String(36),
                sa.ForeignKey("players.id", ondelete="CASCADE"),
                nullable=False,
                index=True,
            ),
            sa.Column("name", sa.String(60), nullable=False),
            sa.Column("personality_description", sa.Text, nullable=False),
            sa.Column("avatar_image_url", sa.Text, nullable=True),
            sa.Column("metropolis_token_id", sa.String(80), nullable=True, index=True),
            sa.Column("elo", sa.Integer, nullable=False, server_default="1200"),
            sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
            sa.Column("archived_at", sa.DateTime, nullable=True),
        )


def downgrade() -> None:
    if _has_table("player_agents"):
        op.drop_table("player_agents")
