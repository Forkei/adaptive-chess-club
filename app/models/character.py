from __future__ import annotations

import enum
import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import JSON, Boolean, DateTime, Enum, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base


class CharacterState(str, enum.Enum):
    GENERATING_MEMORIES = "generating_memories"
    READY = "ready"
    GENERATION_FAILED = "generation_failed"


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.utcnow()


class Character(Base):
    __tablename__ = "characters"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)

    name: Mapped[str] = mapped_column(String(120), nullable=False)
    short_description: Mapped[str] = mapped_column(String(280), nullable=False, default="")
    backstory: Mapped[str] = mapped_column(Text, nullable=False, default="")
    avatar_emoji: Mapped[str] = mapped_column(String(8), nullable=False, default="♟️")

    # Style sliders (1-10)
    aggression: Mapped[int] = mapped_column(Integer, nullable=False, default=5)
    risk_tolerance: Mapped[int] = mapped_column(Integer, nullable=False, default=5)
    patience: Mapped[int] = mapped_column(Integer, nullable=False, default=5)
    trash_talk: Mapped[int] = mapped_column(Integer, nullable=False, default=5)

    # Skill
    target_elo: Mapped[int] = mapped_column(Integer, nullable=False, default=1400)
    adaptive: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    # `current_elo` is the live effective rating used by the Director each match.
    # `floor_elo` ratchets up over time; current_elo never drops below it.
    # `max_elo` is the hard ceiling.
    current_elo: Mapped[int] = mapped_column(Integer, nullable=False, default=1400)
    floor_elo: Mapped[int] = mapped_column(Integer, nullable=False, default=1400)
    max_elo: Mapped[int] = mapped_column(Integer, nullable=False, default=1800)

    # Opening preferences: list of ECO codes (strings) stored as JSON.
    opening_preferences: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)

    # Voice + quirks
    voice_descriptor: Mapped[str] = mapped_column(String(280), nullable=False, default="")
    quirks: Mapped[str] = mapped_column(Text, nullable=False, default="")

    # Lifecycle
    state: Mapped[CharacterState] = mapped_column(
        Enum(CharacterState, name="character_state"),
        nullable=False,
        default=CharacterState.GENERATING_MEMORIES,
    )
    memory_generation_started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    memory_generation_error: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Presets / soft-delete
    is_preset: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    preset_key: Mapped[str | None] = mapped_column(String(64), nullable=True, unique=True)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=_now, onupdate=_now
    )

    memories: Mapped[list["Memory"]] = relationship(  # type: ignore[name-defined]  # noqa: F821
        back_populates="character",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "short_description": self.short_description,
            "backstory": self.backstory,
            "avatar_emoji": self.avatar_emoji,
            "aggression": self.aggression,
            "risk_tolerance": self.risk_tolerance,
            "patience": self.patience,
            "trash_talk": self.trash_talk,
            "target_elo": self.target_elo,
            "adaptive": self.adaptive,
            "current_elo": self.current_elo,
            "floor_elo": self.floor_elo,
            "max_elo": self.max_elo,
            "opening_preferences": list(self.opening_preferences or []),
            "voice_descriptor": self.voice_descriptor,
            "quirks": self.quirks,
            "state": self.state.value if isinstance(self.state, CharacterState) else self.state,
            "is_preset": self.is_preset,
            "preset_key": self.preset_key,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
