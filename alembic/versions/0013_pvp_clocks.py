"""Phase 4.4 — PvP clocks.

Adds time_control / increment_ms / white_clock_ms / black_clock_ms /
last_tick_at to pvp_matches so the server can enforce flag-fall.

Revision ID: 0013_pvp_clocks
Revises: 0012_character_evolution_state
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect

revision: str = "0013_pvp_clocks"
down_revision: Union[str, None] = "0012_character_evolution_state"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _columns(table: str) -> set[str]:
    bind = op.get_bind()
    insp = inspect(bind)
    if table not in insp.get_table_names():
        return set()
    return {c["name"] for c in insp.get_columns(table)}


def upgrade() -> None:
    cols = _columns("pvp_matches")
    if not cols:
        return
    if "time_control" not in cols:
        op.add_column(
            "pvp_matches",
            sa.Column(
                "time_control", sa.String(length=16),
                nullable=False, server_default="untimed",
            ),
        )
    if "increment_ms" not in cols:
        op.add_column(
            "pvp_matches",
            sa.Column("increment_ms", sa.Integer(), nullable=False, server_default="0"),
        )
    if "white_clock_ms" not in cols:
        op.add_column("pvp_matches", sa.Column("white_clock_ms", sa.Integer(), nullable=True))
    if "black_clock_ms" not in cols:
        op.add_column("pvp_matches", sa.Column("black_clock_ms", sa.Integer(), nullable=True))
    if "last_tick_at" not in cols:
        op.add_column("pvp_matches", sa.Column("last_tick_at", sa.DateTime(), nullable=True))


def downgrade() -> None:
    cols = _columns("pvp_matches")
    for c in ("last_tick_at", "black_clock_ms", "white_clock_ms", "increment_ms", "time_control"):
        if c in cols:
            op.drop_column("pvp_matches", c)
