"""Single-character mode: default max_content_rating to mature.

Kenji is rated mature. New users defaulted to family, which blocked
them from seeing him. Change the default to mature and backfill all
existing family-rated players so they can access Kenji.

Revision ID: 0014_mature_default_rating
Revises: 0013_pvp_clocks
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0014_mature_default_rating"
down_revision: Union[str, None] = "0013_pvp_clocks"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Backfill existing players who are still at the 'family' default.
    op.execute(
        "UPDATE players SET max_content_rating = 'mature' WHERE max_content_rating = 'family'"
    )

    # Change the column server default so new rows land at 'mature'.
    with op.batch_alter_table("players") as batch_op:
        batch_op.alter_column(
            "max_content_rating",
            existing_type=sa.String(),
            server_default="mature",
            existing_nullable=False,
        )


def downgrade() -> None:
    op.execute(
        "UPDATE players SET max_content_rating = 'family' WHERE max_content_rating = 'mature'"
    )
    with op.batch_alter_table("players") as batch_op:
        batch_op.alter_column(
            "max_content_rating",
            existing_type=sa.String(),
            server_default="family",
            existing_nullable=False,
        )
