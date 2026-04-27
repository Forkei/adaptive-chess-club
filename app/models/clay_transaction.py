from __future__ import annotations

from datetime import datetime
from uuid import uuid4

from sqlalchemy import DateTime, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


def _uuid() -> str:
    return str(uuid4())


def _now() -> datetime:
    return datetime.utcnow()


class ClayTransaction(Base):
    """Immutable audit log for every $CLAY balance movement."""

    __tablename__ = "clay_transactions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    player_id: Mapped[str] = mapped_column(
        ForeignKey("players.id", ondelete="CASCADE"), nullable=False, index=True
    )
    # Signed: negative = debit (money left), positive = credit (money arrived).
    amount: Mapped[int] = mapped_column(Integer, nullable=False)
    # Balance after this transaction — snapshot for audit.
    balance_after: Mapped[int] = mapped_column(Integer, nullable=False)
    # e.g. "starting_grant", "match_stake", "match_win", "match_loss",
    # "match_draw_refund", "match_abandon_refund"
    reason: Mapped[str] = mapped_column(String(64), nullable=False)
    related_match_id: Mapped[str | None] = mapped_column(
        ForeignKey("matches.id", ondelete="SET NULL"), nullable=True, index=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=_now
    )
