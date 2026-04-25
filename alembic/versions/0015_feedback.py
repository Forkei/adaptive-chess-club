"""Add feedback table.

Revision ID: 0015_feedback
Revises: 0014_mature_default_rating
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0015_feedback"
down_revision: Union[str, None] = "0014_mature_default_rating"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "feedback",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("text", sa.Text, nullable=False),
        sa.Column("rating", sa.Integer, nullable=True),
        sa.Column("username", sa.String(64), nullable=True),
        sa.Column("page_url", sa.String(512), nullable=True),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now(), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("feedback")
