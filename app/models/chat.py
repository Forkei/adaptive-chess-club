"""Phase 4.2.5 — pre-match conversation storage.

When a player enters a character's room they can chat before any Match
row exists. We persist those turns so:

- the Soul + Subconscious have context on follow-up messages
- the post-match memory generator can ingest the conversation
- characters can "remember" what was said across sessions

`CharacterChatSession` is one row per (character, player, open-period).
Closed when the Soul emits `game_action="start_game"` (the session then
hands off to the Match) or when the player leaves the room.

`CharacterChatTurn` is one row per utterance (player or character).
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import JSON, DateTime, Enum, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.utcnow()


class ChatSessionStatus(str, enum.Enum):
    ACTIVE = "active"
    HANDED_OFF = "handed_off"   # Soul fired start_game; a Match now owns the player
    ABANDONED = "abandoned"     # player left without starting


class ChatTurnRole(str, enum.Enum):
    PLAYER = "player"
    CHARACTER = "character"


class CharacterChatSession(Base):
    __tablename__ = "character_chat_sessions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    character_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("characters.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    player_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("players.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    status: Mapped[ChatSessionStatus] = mapped_column(
        Enum(ChatSessionStatus, name="chat_session_status",
             values_callable=lambda e: [m.value for m in e]),
        nullable=False, default=ChatSessionStatus.ACTIVE, index=True,
    )

    # When the Soul handed off to a real Match, we remember which one.
    handed_off_match_id: Mapped[str | None] = mapped_column(
        String(36), nullable=True
    )
    # Notes the Soul dropped about the visitor during chat — flushed into
    # `match.extra_state["pending_opponent_notes"]` at hand-off so the
    # existing post-match memory pipeline can consume them unchanged.
    pending_notes: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)

    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=_now, onupdate=_now
    )
    ended_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    turns: Mapped[list["CharacterChatTurn"]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="CharacterChatTurn.turn_number",
        lazy="selectin",
    )


class CharacterChatTurn(Base):
    __tablename__ = "character_chat_turns"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    session_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("character_chat_sessions.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    turn_number: Mapped[int] = mapped_column(Integer, nullable=False)
    role: Mapped[ChatTurnRole] = mapped_column(
        Enum(ChatTurnRole, name="chat_turn_role",
             values_callable=lambda e: [m.value for m in e]),
        nullable=False,
    )
    text: Mapped[str] = mapped_column(Text, nullable=False)

    # For character turns, mirror the Soul's rendered state.
    emotion: Mapped[str | None] = mapped_column(String(32), nullable=True)
    emotion_intensity: Mapped[float | None] = mapped_column(Float, nullable=True)
    game_action: Mapped[str | None] = mapped_column(String(16), nullable=True)
    # Raw Soul-output dump, for debug + post-match memory reconstruction.
    soul_raw: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=_now)

    session: Mapped[CharacterChatSession] = relationship(back_populates="turns")
