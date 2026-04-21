"""Phase 4.2.5 — pre-match character chat tables.

Adds:
  - character_chat_sessions
  - character_chat_turns

Idempotent. Skip-if-exists.

Revision ID: 0010_chat_sessions
Revises: 0009_lobbies_and_pvp
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect

revision: str = "0010_chat_sessions"
down_revision: Union[str, None] = "0009_lobbies_and_pvp"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _has_table(name: str) -> bool:
    return name in inspect(op.get_bind()).get_table_names()


def upgrade() -> None:
    if not _has_table("character_chat_sessions"):
        op.create_table(
            "character_chat_sessions",
            sa.Column("id", sa.String(length=36), primary_key=True),
            sa.Column(
                "character_id",
                sa.String(length=36),
                sa.ForeignKey("characters.id", ondelete="CASCADE"),
                nullable=False,
                index=True,
            ),
            sa.Column(
                "player_id",
                sa.String(length=36),
                sa.ForeignKey("players.id", ondelete="CASCADE"),
                nullable=False,
                index=True,
            ),
            sa.Column("status", sa.String(length=16), nullable=False, server_default="active", index=True),
            sa.Column("handed_off_match_id", sa.String(length=36), nullable=True),
            sa.Column("pending_notes", sa.JSON(), nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.Column("updated_at", sa.DateTime(), nullable=False),
            sa.Column("ended_at", sa.DateTime(), nullable=True),
        )

    if not _has_table("character_chat_turns"):
        op.create_table(
            "character_chat_turns",
            sa.Column("id", sa.String(length=36), primary_key=True),
            sa.Column(
                "session_id",
                sa.String(length=36),
                sa.ForeignKey("character_chat_sessions.id", ondelete="CASCADE"),
                nullable=False,
                index=True,
            ),
            sa.Column("turn_number", sa.Integer(), nullable=False),
            sa.Column("role", sa.String(length=16), nullable=False),
            sa.Column("text", sa.Text(), nullable=False),
            sa.Column("emotion", sa.String(length=32), nullable=True),
            sa.Column("emotion_intensity", sa.Float(), nullable=True),
            sa.Column("game_action", sa.String(length=16), nullable=True),
            sa.Column("soul_raw", sa.JSON(), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=False),
        )


def downgrade() -> None:
    for tbl in ("character_chat_turns", "character_chat_sessions"):
        if _has_table(tbl):
            op.drop_table(tbl)
