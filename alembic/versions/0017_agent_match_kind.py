"""Block 13 — add participant_agent_id + match_kind to matches.

Adds:
  - matches.participant_agent_id  (nullable FK → player_agents.id SET NULL)
  - matches.match_kind            (string, default "human_vs_character")
  - composite index ix_matches_agent_status on (participant_agent_id, status)

Backfills existing rows with match_kind = "human_vs_character". Idempotent.

Revision ID: 0017_agent_match_kind
Revises: 0016_player_agents
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect, text


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


revision: str = "0017_agent_match_kind"
down_revision: Union[str, None] = "0016_player_agents"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()

    if not _has_table("matches"):
        return  # Pre-schema DB: matches table doesn't exist yet; nothing to alter.

    if not _has_column("matches", "participant_agent_id"):
        with op.batch_alter_table("matches") as batch_op:
            batch_op.add_column(
                sa.Column("participant_agent_id", sa.String(36), nullable=True)
            )

    if not _has_column("matches", "match_kind"):
        with op.batch_alter_table("matches") as batch_op:
            batch_op.add_column(
                sa.Column(
                    "match_kind",
                    sa.String(32),
                    nullable=False,
                    server_default="human_vs_character",
                )
            )

    # Backfill any rows that are NULL (pre-migration rows will have NULL from
    # SQLite's column default not firing on existing data).
    bind.execute(
        text(
            "UPDATE matches SET match_kind = 'human_vs_character' "
            "WHERE match_kind IS NULL"
        )
    )

    if not _has_index("matches", "ix_matches_agent_status"):
        op.create_index(
            "ix_matches_agent_status",
            "matches",
            ["participant_agent_id", "status"],
        )


def downgrade() -> None:
    if _has_index("matches", "ix_matches_agent_status"):
        op.drop_index("ix_matches_agent_status", table_name="matches")

    with op.batch_alter_table("matches") as batch_op:
        if _has_column("matches", "match_kind"):
            batch_op.drop_column("match_kind")
        if _has_column("matches", "participant_agent_id"):
            batch_op.drop_column("participant_agent_id")
