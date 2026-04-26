from __future__ import annotations

from datetime import datetime
from uuid import uuid4

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base


def _uuid() -> str:
    return str(uuid4())


def _now() -> datetime:
    return datetime.utcnow()


class PlayerAgent(Base):
    __tablename__ = "player_agents"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    owner_player_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("players.id", ondelete="CASCADE"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(60), nullable=False)
    # Max 500 chars enforced at API/form layer; stored as-is after sanitization.
    personality_description: Mapped[str] = mapped_column(Text, nullable=False)
    avatar_image_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    metropolis_token_id: Mapped[str | None] = mapped_column(String(80), nullable=True, index=True)
    elo: Mapped[int] = mapped_column(Integer, nullable=False, default=1200)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=_now)
    # Soft delete — archived agents are hidden from listings but not purged.
    archived_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    owner = relationship("Player", back_populates="agents")
