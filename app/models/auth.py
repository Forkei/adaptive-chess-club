"""Phase 4.0a — auth-adjacent models (password reset).

Kept in a separate module from `match.py` so the `Player` row stays
focused on gameplay identity + ratings. Imported by `app.models.__init__`.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.utcnow()


class PasswordResetToken(Base):
    """One row per issued reset token.

    The raw token is emailed to the user; only the SHA-256 of it is stored
    in `token_hash` so a DB leak doesn't expose working reset links. Tokens
    are one-shot — `used_at` is set when the reset completes and subsequent
    uses are rejected.
    """

    __tablename__ = "password_reset_tokens"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    player_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("players.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    token_hash: Mapped[str] = mapped_column(
        String(64), nullable=False, unique=True, index=True
    )
    expires_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    used_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=_now)
