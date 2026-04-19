"""Patch Pass 2 Item 2: player Elo + proper expected-score math.

Adds:
  - players.elo (int, default 1200)
  - players.elo_floor (int, default 800)
  - players.elo_ceiling (int, default 2800)
  - matches.player_elo_at_start (int, nullable)
  - matches.player_elo_at_end (int, nullable)
  - match_analyses.player_elo_delta_applied (int, nullable)
  - match_analyses.player_floor_raised (bool, default False)

Idempotent: skips any column that already exists. Pre-existing rows keep
NULL on nullable cols. Player.elo backfills to 1200 via server_default.

Revision ID: 0005_player_elo
Revises: 0004_resigned_status
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect

revision: str = "0005_player_elo"
down_revision: Union[str, None] = "0004_resigned_status"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _columns(table: str) -> set[str]:
    bind = op.get_bind()
    insp = inspect(bind)
    if table not in insp.get_table_names():
        return set()
    return {c["name"] for c in insp.get_columns(table)}


def upgrade() -> None:
    players_cols = _columns("players")
    if players_cols:
        if "elo" not in players_cols:
            op.add_column(
                "players",
                sa.Column("elo", sa.Integer(), nullable=False, server_default="1200"),
            )
        if "elo_floor" not in players_cols:
            op.add_column(
                "players",
                sa.Column("elo_floor", sa.Integer(), nullable=False, server_default="800"),
            )
        if "elo_ceiling" not in players_cols:
            op.add_column(
                "players",
                sa.Column("elo_ceiling", sa.Integer(), nullable=False, server_default="2800"),
            )

    matches_cols = _columns("matches")
    if matches_cols:
        if "player_elo_at_start" not in matches_cols:
            op.add_column(
                "matches",
                sa.Column("player_elo_at_start", sa.Integer(), nullable=True),
            )
        if "player_elo_at_end" not in matches_cols:
            op.add_column(
                "matches",
                sa.Column("player_elo_at_end", sa.Integer(), nullable=True),
            )

    analysis_cols = _columns("match_analyses")
    if analysis_cols:
        if "player_elo_delta_applied" not in analysis_cols:
            op.add_column(
                "match_analyses",
                sa.Column("player_elo_delta_applied", sa.Integer(), nullable=True),
            )
        if "player_floor_raised" not in analysis_cols:
            op.add_column(
                "match_analyses",
                sa.Column(
                    "player_floor_raised", sa.Boolean(), nullable=False, server_default="0"
                ),
            )


def downgrade() -> None:
    analysis_cols = _columns("match_analyses")
    if "player_floor_raised" in analysis_cols:
        op.drop_column("match_analyses", "player_floor_raised")
    if "player_elo_delta_applied" in analysis_cols:
        op.drop_column("match_analyses", "player_elo_delta_applied")

    matches_cols = _columns("matches")
    if "player_elo_at_end" in matches_cols:
        op.drop_column("matches", "player_elo_at_end")
    if "player_elo_at_start" in matches_cols:
        op.drop_column("matches", "player_elo_at_start")

    players_cols = _columns("players")
    if "elo_ceiling" in players_cols:
        op.drop_column("players", "elo_ceiling")
    if "elo_floor" in players_cols:
        op.drop_column("players", "elo_floor")
    if "elo" in players_cols:
        op.drop_column("players", "elo")
