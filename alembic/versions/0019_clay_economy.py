"""Block 17 — $CLAY economy: clay_balances, clay_transactions, match stake columns.

Idempotent — running twice is safe (all DDL is guarded by existence checks).

Revision ID: 0019_clay_economy
Revises: 0018_memory_agent_id
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


revision: str = "0019_clay_economy"
down_revision: Union[str, None] = "0018_memory_agent_id"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # --- clay_balances ---------------------------------------------------------
    if not _has_table("clay_balances"):
        op.create_table(
            "clay_balances",
            sa.Column(
                "player_id",
                sa.String(36),
                sa.ForeignKey("players.id", ondelete="CASCADE"),
                primary_key=True,
            ),
            sa.Column("balance", sa.Integer, nullable=False, server_default="0"),
            sa.Column("updated_at", sa.DateTime, nullable=False),
        )

    # --- clay_transactions -----------------------------------------------------
    if not _has_table("clay_transactions"):
        op.create_table(
            "clay_transactions",
            sa.Column("id", sa.String(36), primary_key=True),
            sa.Column(
                "player_id",
                sa.String(36),
                sa.ForeignKey("players.id", ondelete="CASCADE"),
                nullable=False,
            ),
            sa.Column("amount", sa.Integer, nullable=False),
            sa.Column("balance_after", sa.Integer, nullable=False),
            sa.Column("reason", sa.String(64), nullable=False),
            sa.Column(
                "related_match_id",
                sa.String(36),
                sa.ForeignKey("matches.id", ondelete="SET NULL"),
                nullable=True,
            ),
            sa.Column("created_at", sa.DateTime, nullable=False),
        )

    if not _has_index("clay_transactions", "ix_clay_transactions_player_id"):
        op.create_index(
            "ix_clay_transactions_player_id", "clay_transactions", ["player_id"]
        )
    if not _has_index("clay_transactions", "ix_clay_transactions_match_id"):
        op.create_index(
            "ix_clay_transactions_match_id",
            "clay_transactions",
            ["related_match_id"],
        )

    # --- matches: stake columns ------------------------------------------------
    if not _has_table("matches"):
        return

    needs_stake = not _has_column("matches", "stake_cents")
    needs_settled = not _has_column("matches", "stake_settled_at")

    if needs_stake or needs_settled:
        with op.batch_alter_table("matches") as batch_op:
            if needs_stake:
                batch_op.add_column(
                    sa.Column(
                        "stake_cents",
                        sa.Integer,
                        nullable=False,
                        server_default="0",
                    )
                )
            if needs_settled:
                batch_op.add_column(
                    sa.Column("stake_settled_at", sa.DateTime, nullable=True)
                )


def downgrade() -> None:
    if _has_index("clay_transactions", "ix_clay_transactions_player_id"):
        op.drop_index("ix_clay_transactions_player_id", "clay_transactions")
    if _has_index("clay_transactions", "ix_clay_transactions_match_id"):
        op.drop_index("ix_clay_transactions_match_id", "clay_transactions")

    if _has_table("clay_transactions"):
        op.drop_table("clay_transactions")
    if _has_table("clay_balances"):
        op.drop_table("clay_balances")

    if _has_table("matches"):
        with op.batch_alter_table("matches") as batch_op:
            if _has_column("matches", "stake_settled_at"):
                batch_op.drop_column("stake_settled_at")
            if _has_column("matches", "stake_cents"):
                batch_op.drop_column("stake_cents")
