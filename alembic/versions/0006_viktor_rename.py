"""Rename Viktor preset: Volkov -> Petrov.

Updates any existing character row with `preset_key='viktor_volkov'` to
`preset_key='viktor_petrov'` and `name='Viktor Petrov'` so the on-startup
idempotent seeder recognises the existing row instead of creating a
duplicate alongside it.

Idempotent: only touches rows that still carry the old key.

Revision ID: 0006_viktor_rename
Revises: 0005_player_elo
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect

revision: str = "0006_viktor_rename"
down_revision: Union[str, None] = "0005_player_elo"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _has_table(name: str) -> bool:
    return name in inspect(op.get_bind()).get_table_names()


def upgrade() -> None:
    if not _has_table("characters"):
        return
    op.execute(
        sa.text(
            "UPDATE characters "
            "SET preset_key = 'viktor_petrov', name = 'Viktor Petrov' "
            "WHERE preset_key = 'viktor_volkov'"
        )
    )


def downgrade() -> None:
    if not _has_table("characters"):
        return
    op.execute(
        sa.text(
            "UPDATE characters "
            "SET preset_key = 'viktor_volkov', name = 'Viktor Volkov' "
            "WHERE preset_key = 'viktor_petrov'"
        )
    )
