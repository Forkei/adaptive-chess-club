"""Phase 4.2 — lobbies, matchmaking, PvP matches.

Adds four tables (all skip-if-exists so this migration can run after
`Base.metadata.create_all` has already materialized the schema):

  - lobbies
  - lobby_memberships
  - matchmaking_queue
  - pvp_matches

Revision ID: 0009_lobbies_and_pvp
Revises: 0008_match_is_private
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect

revision: str = "0009_lobbies_and_pvp"
down_revision: Union[str, None] = "0008_match_is_private"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _has_table(name: str) -> bool:
    return name in inspect(op.get_bind()).get_table_names()


def upgrade() -> None:
    if not _has_table("lobbies"):
        op.create_table(
            "lobbies",
            sa.Column("id", sa.String(length=36), primary_key=True),
            sa.Column("code", sa.String(length=8), nullable=False, unique=True, index=True),
            sa.Column(
                "host_id",
                sa.String(length=36),
                sa.ForeignKey("players.id", ondelete="CASCADE"),
                nullable=False,
                index=True,
            ),
            sa.Column("is_private", sa.Boolean(), nullable=False, server_default="0"),
            sa.Column("allow_spectators", sa.Boolean(), nullable=False, server_default="1"),
            sa.Column("music_track", sa.String(length=64), nullable=True),
            sa.Column("music_volume", sa.Float(), nullable=False, server_default="0.5"),
            sa.Column("lights_brightness", sa.Float(), nullable=False, server_default="0.7"),
            sa.Column("lights_hue", sa.String(length=16), nullable=False, server_default="#C9A66B"),
            sa.Column("status", sa.String(length=16), nullable=False, server_default="open", index=True),
            sa.Column("current_match_id", sa.String(length=36), nullable=True),
            sa.Column("via_matchmaking", sa.Boolean(), nullable=False, server_default="0"),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.Column("updated_at", sa.DateTime(), nullable=False),
            sa.Column("closed_at", sa.DateTime(), nullable=True),
        )

    if not _has_table("lobby_memberships"):
        op.create_table(
            "lobby_memberships",
            sa.Column("id", sa.String(length=36), primary_key=True),
            sa.Column(
                "lobby_id",
                sa.String(length=36),
                sa.ForeignKey("lobbies.id", ondelete="CASCADE"),
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
            sa.Column("role", sa.String(length=16), nullable=False, server_default="guest"),
            sa.Column("joined_at", sa.DateTime(), nullable=False),
            sa.Column("left_at", sa.DateTime(), nullable=True),
        )

    if not _has_table("matchmaking_queue"):
        op.create_table(
            "matchmaking_queue",
            sa.Column("id", sa.String(length=36), primary_key=True),
            sa.Column(
                "player_id",
                sa.String(length=36),
                sa.ForeignKey("players.id", ondelete="CASCADE"),
                nullable=False,
                unique=True,
                index=True,
            ),
            sa.Column("queued_at", sa.DateTime(), nullable=False),
            sa.Column("elo_at_queue", sa.Integer(), nullable=False),
            sa.Column("band_expansion_step", sa.Integer(), nullable=False, server_default="0"),
            sa.Column(
                "matched_lobby_id",
                sa.String(length=36),
                sa.ForeignKey("lobbies.id", ondelete="SET NULL"),
                nullable=True,
            ),
            sa.Column("canceled_at", sa.DateTime(), nullable=True),
        )

    if not _has_table("pvp_matches"):
        op.create_table(
            "pvp_matches",
            sa.Column("id", sa.String(length=36), primary_key=True),
            sa.Column(
                "lobby_id",
                sa.String(length=36),
                sa.ForeignKey("lobbies.id", ondelete="CASCADE"),
                nullable=False,
                index=True,
            ),
            sa.Column(
                "white_player_id",
                sa.String(length=36),
                sa.ForeignKey("players.id", ondelete="CASCADE"),
                nullable=False,
                index=True,
            ),
            sa.Column(
                "black_player_id",
                sa.String(length=36),
                sa.ForeignKey("players.id", ondelete="CASCADE"),
                nullable=False,
                index=True,
            ),
            sa.Column("initial_fen", sa.String(length=120), nullable=False),
            sa.Column("current_fen", sa.String(length=120), nullable=False),
            sa.Column("moves", sa.JSON(), nullable=False),
            sa.Column("move_count", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("status", sa.String(length=16), nullable=False, server_default="in_progress", index=True),
            sa.Column("result", sa.String(length=16), nullable=True),
            sa.Column("is_private", sa.Boolean(), nullable=False, server_default="0"),
            sa.Column("white_elo_at_start", sa.Integer(), nullable=False),
            sa.Column("black_elo_at_start", sa.Integer(), nullable=False),
            sa.Column("white_elo_at_end", sa.Integer(), nullable=True),
            sa.Column("black_elo_at_end", sa.Integer(), nullable=True),
            sa.Column("extra_state", sa.JSON(), nullable=False),
            sa.Column("started_at", sa.DateTime(), nullable=False),
            sa.Column("ended_at", sa.DateTime(), nullable=True, index=True),
        )


def downgrade() -> None:
    for table in ("pvp_matches", "matchmaking_queue", "lobby_memberships", "lobbies"):
        if _has_table(table):
            op.drop_table(table)
