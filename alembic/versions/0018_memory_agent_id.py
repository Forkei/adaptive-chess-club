"""Block 13 — add memories.agent_id; make memories.character_id nullable.

Agents have their own memory scope: memories with agent_id set and
character_id=NULL. The batch alter is required for SQLite to drop the
NOT NULL constraint on character_id.

Revision ID: 0018_memory_agent_id
Revises: 0017_agent_match_kind
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect


def _has_table(name: str) -> bool:
    return name in inspect(op.get_bind()).get_table_names()


def _has_column(table: str, column: str) -> bool:
    if not _has_table(table):
        return False
    cols = {r["name"] for r in inspect(op.get_bind()).get_columns(table)}
    return column in cols


def _has_index(table: str, index_name: str) -> bool:
    if not _has_table(table):
        return False
    idxs = {i["name"] for i in inspect(op.get_bind()).get_indexes(table)}
    return index_name in idxs


revision: str = "0018_memory_agent_id"
down_revision: Union[str, None] = "0017_agent_match_kind"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    if not _has_table("memories"):
        return

    needs_agent_id = not _has_column("memories", "agent_id")
    needs_nullable = False
    if _has_table("memories"):
        # Check if character_id is currently NOT NULL.
        cols = inspect(op.get_bind()).get_columns("memories")
        for col in cols:
            if col["name"] == "character_id" and col.get("nullable") is False:
                needs_nullable = True
                break

    if needs_agent_id or needs_nullable:
        with op.batch_alter_table("memories", recreate="always") as batch_op:
            if needs_nullable:
                batch_op.alter_column("character_id", nullable=True)
            if needs_agent_id:
                batch_op.add_column(
                    sa.Column("agent_id", sa.String(36), nullable=True)
                )

    if needs_agent_id and not _has_index("memories", "ix_memories_agent_id"):
        op.create_index("ix_memories_agent_id", "memories", ["agent_id"])


def downgrade() -> None:
    if not _has_table("memories"):
        return

    if _has_index("memories", "ix_memories_agent_id"):
        op.drop_index("ix_memories_agent_id", table_name="memories")

    with op.batch_alter_table("memories", recreate="always") as batch_op:
        if _has_column("memories", "agent_id"):
            batch_op.drop_column("agent_id")
        # Restore NOT NULL on character_id (note: existing agent rows would violate this).
        batch_op.alter_column("character_id", nullable=False)
