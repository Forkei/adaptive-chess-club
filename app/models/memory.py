from __future__ import annotations

import enum
import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import JSON, DateTime, Enum, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base

if TYPE_CHECKING:
    from app.models.character import Character


class MemoryScope(str, enum.Enum):
    CHARACTER_LORE = "character_lore"
    OPPONENT_SPECIFIC = "opponent_specific"
    CROSS_PLAYER = "cross_player"
    MATCH_RECAP = "match_recap"


class MemoryType(str, enum.Enum):
    FORMATIVE = "formative"
    RIVALRY = "rivalry"
    TRAVEL = "travel"
    TRIUMPH = "triumph"
    DEFEAT = "defeat"
    HABIT = "habit"
    OPINION = "opinion"
    OBSERVATION = "observation"
    # Phase 4.3 — learned patterns the character has picked up over
    # matches (e.g. trap patterns that burned them once). Fed into the
    # Subconscious's normal retrieval flow.
    LEARNING = "learning"


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.utcnow()


class Memory(Base):
    __tablename__ = "memories"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)

    character_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("characters.id", ondelete="CASCADE"), nullable=False, index=True
    )
    # player_id / match_id are not FK'd to real tables yet (those arrive in Phase 2).
    # Nullable strings keep the schema future-proof without requiring those tables now.
    player_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    match_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)

    scope: Mapped[MemoryScope] = mapped_column(Enum(MemoryScope, name="memory_scope"), nullable=False)
    type: Mapped[MemoryType] = mapped_column(Enum(MemoryType, name="memory_type"), nullable=False)

    emotional_valence: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    triggers: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    narrative_text: Mapped[str] = mapped_column(Text, nullable=False)
    relevance_tags: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)

    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=_now)
    last_surfaced_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    surface_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # Phase 2b: semantic retrieval vector. NULL until the memory is embedded
    # (either at creation via `embed_and_persist` or by the backfill script).
    embedding: Mapped[list[float] | None] = mapped_column(JSON, nullable=True)

    character: Mapped["Character"] = relationship(back_populates="memories")
