"""Phase 4.0a: email + password auth columns + password_reset_tokens table.

Adds to `players`:
  - email (String(254), unique, nullable) — NULL for guests + legacy rows
  - password_hash (String(255), nullable) — NULL means "legacy, no password"
  - email_verified_at (DateTime, nullable)

Creates `password_reset_tokens` for the forgot-password flow.

Idempotent: skips any column/table that already exists. Safe to run
against a DB that already has Base.metadata.create_all() applied (via
init_db); safe to run twice.

Revision ID: 0007_auth_credentials
Revises: 0006_viktor_rename
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect

revision: str = "0007_auth_credentials"
down_revision: Union[str, None] = "0006_viktor_rename"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _columns(table: str) -> set[str]:
    bind = op.get_bind()
    insp = inspect(bind)
    if table not in insp.get_table_names():
        return set()
    return {c["name"] for c in insp.get_columns(table)}


def _has_table(name: str) -> bool:
    return name in inspect(op.get_bind()).get_table_names()


def upgrade() -> None:
    players_cols = _columns("players")
    if players_cols:
        if "email" not in players_cols:
            op.add_column(
                "players",
                sa.Column("email", sa.String(length=254), nullable=True),
            )
            op.create_index(
                "ix_players_email", "players", ["email"], unique=True
            )
        if "password_hash" not in players_cols:
            op.add_column(
                "players",
                sa.Column("password_hash", sa.String(length=255), nullable=True),
            )
        if "email_verified_at" not in players_cols:
            op.add_column(
                "players",
                sa.Column("email_verified_at", sa.DateTime(), nullable=True),
            )

    if not _has_table("password_reset_tokens"):
        # `index=True` on the player_id column makes alembic auto-create
        # `ix_password_reset_tokens_player_id`. `unique=True` on token_hash
        # creates a unique constraint, which SQLite materialises as an
        # index too. So the explicit create_index calls would duplicate.
        op.create_table(
            "password_reset_tokens",
            sa.Column("id", sa.String(length=36), primary_key=True),
            sa.Column(
                "player_id",
                sa.String(length=36),
                sa.ForeignKey("players.id", ondelete="CASCADE"),
                nullable=False,
                index=True,
            ),
            sa.Column(
                "token_hash",
                sa.String(length=64),
                nullable=False,
                unique=True,
                index=True,
            ),
            sa.Column("expires_at", sa.DateTime(), nullable=False),
            sa.Column("used_at", sa.DateTime(), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=False),
        )


def downgrade() -> None:
    if _has_table("password_reset_tokens"):
        op.drop_table("password_reset_tokens")

    players_cols = _columns("players")
    if "email_verified_at" in players_cols:
        op.drop_column("players", "email_verified_at")
    if "password_hash" in players_cols:
        op.drop_column("players", "password_hash")
    if "email" in players_cols:
        try:
            op.drop_index("ix_players_email", "players")
        except Exception:
            pass
        op.drop_column("players", "email")
