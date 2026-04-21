"""Phase 4.2.5 — add `lobbies.time_control` (tabletop clock preset).

Just a display-only hint at 4.2.5 — no server ticking yet. Values:
"untimed" / "5+0" / "10+0" / "15+10".

Revision ID: 0011_lobby_time_control
Revises: 0010_chat_sessions
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect

revision: str = "0011_lobby_time_control"
down_revision: Union[str, None] = "0010_chat_sessions"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _columns(table: str) -> set[str]:
    bind = op.get_bind()
    insp = inspect(bind)
    if table not in insp.get_table_names():
        return set()
    return {c["name"] for c in insp.get_columns(table)}


def upgrade() -> None:
    cols = _columns("lobbies")
    if cols and "time_control" not in cols:
        op.add_column(
            "lobbies",
            sa.Column(
                "time_control",
                sa.String(length=16),
                nullable=False,
                server_default="untimed",
            ),
        )


def downgrade() -> None:
    if "time_control" in _columns("lobbies"):
        op.drop_column("lobbies", "time_control")
