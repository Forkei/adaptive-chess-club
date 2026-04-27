from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


def _now() -> datetime:
    return datetime.utcnow()


class ClayBalance(Base):
    __tablename__ = "clay_balances"

    player_id: Mapped[str] = mapped_column(
        ForeignKey("players.id", ondelete="CASCADE"), primary_key=True
    )
    # Integer cents — avoids float precision issues. 100 $CLAY = 10000 stored.
    balance: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=_now, onupdate=_now
    )
