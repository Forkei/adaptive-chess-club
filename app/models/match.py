from __future__ import annotations

import enum
import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base
from app.models.character import ContentRating

# Username rules (Phase 3a): lowercase letters, digits, underscore; 3-24 chars.
USERNAME_MIN_LEN = 3
USERNAME_MAX_LEN = 24
USERNAME_PATTERN = r"^[a-z0-9_]{3,24}$"
LEGACY_USERNAME = "legacy_system"
GUEST_USERNAME_PREFIX = "guest_"


class MatchStatus(str, enum.Enum):
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    ABANDONED = "abandoned"


class MatchResult(str, enum.Enum):
    WHITE_WIN = "white_win"
    BLACK_WIN = "black_win"
    DRAW = "draw"
    ABANDONED = "abandoned"


class Color(str, enum.Enum):
    WHITE = "white"
    BLACK = "black"


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.utcnow()


class Player(Base):
    __tablename__ = "players"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    # Phase 3a: username is the login identity. Unique, immutable in 3a
    # (can be renamed only when an auto-assigned `guest_*` username logs in
    # and claims a real one).
    username: Mapped[str] = mapped_column(
        String(USERNAME_MAX_LEN), nullable=False, unique=True, index=True
    )
    display_name: Mapped[str] = mapped_column(String(80), nullable=False, default="Guest")
    max_content_rating: Mapped[ContentRating] = mapped_column(
        Enum(ContentRating, name="player_max_content_rating"),
        nullable=False,
        default=ContentRating.FAMILY,
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=_now)

    matches: Mapped[list["Match"]] = relationship(back_populates="player", lazy="selectin")


class Match(Base):
    __tablename__ = "matches"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)

    character_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("characters.id", ondelete="CASCADE"), nullable=False, index=True
    )
    player_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("players.id", ondelete="CASCADE"), nullable=False, index=True
    )

    status: Mapped[MatchStatus] = mapped_column(
        Enum(MatchStatus, name="match_status"),
        nullable=False,
        default=MatchStatus.IN_PROGRESS,
    )
    result: Mapped[MatchResult | None] = mapped_column(
        Enum(MatchResult, name="match_result"), nullable=True
    )
    player_color: Mapped[Color] = mapped_column(
        Enum(Color, name="match_color"), nullable=False, default=Color.WHITE
    )

    initial_fen: Mapped[str] = mapped_column(String(120), nullable=False)
    current_fen: Mapped[str] = mapped_column(String(120), nullable=False)
    move_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    character_elo_at_start: Mapped[int] = mapped_column(Integer, nullable=False)
    character_elo_at_end: Mapped[int | None] = mapped_column(Integer, nullable=True)

    extra_state: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)

    started_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=_now)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    player: Mapped[Player] = relationship(back_populates="matches")
    moves: Mapped[list["Move"]] = relationship(
        back_populates="match",
        cascade="all, delete-orphan",
        order_by="Move.move_number",
        lazy="selectin",
    )


class Move(Base):
    __tablename__ = "moves"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    match_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("matches.id", ondelete="CASCADE"), nullable=False, index=True
    )

    move_number: Mapped[int] = mapped_column(Integer, nullable=False)
    side: Mapped[Color] = mapped_column(Enum(Color, name="move_side"), nullable=False)

    uci: Mapped[str] = mapped_column(String(8), nullable=False)
    san: Mapped[str] = mapped_column(String(16), nullable=False, default="")
    fen_after: Mapped[str] = mapped_column(String(120), nullable=False)

    engine_name: Mapped[str | None] = mapped_column(String(32), nullable=True)
    time_taken_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    eval_cp: Mapped[int | None] = mapped_column(Integer, nullable=True)
    considered_moves: Mapped[list[dict[str, Any]]] = mapped_column(JSON, nullable=False, default=list)
    thinking_depth: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Chat + memory surfacing columns exist in 2a (wire-compat) but stay NULL
    # / empty until 2b's Soul runs.
    player_chat_before: Mapped[str | None] = mapped_column(Text, nullable=True)
    agent_chat_after: Mapped[str | None] = mapped_column(Text, nullable=True)
    surfaced_memory_ids: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    mood_snapshot: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)

    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=_now)

    match: Mapped[Match] = relationship(back_populates="moves")


class OpponentProfile(Base):
    """Populated by the post-match processor (Phase 2b).

    Exists in 2a as an empty schema so gameplay code can reference it without
    forcing another migration next phase.
    """

    __tablename__ = "opponent_profiles"
    __table_args__ = (
        UniqueConstraint("character_id", "player_id", name="uq_character_player"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)

    character_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("characters.id", ondelete="CASCADE"), nullable=False, index=True
    )
    player_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("players.id", ondelete="CASCADE"), nullable=False, index=True
    )

    games_played: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    games_won_by_character: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    games_lost_by_character: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    games_drawn: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # Per-player chess style summary, e.g. aggression_index, blunder_rate, typical_opening_eco.
    style_features: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    narrative_summary: Mapped[str] = mapped_column(Text, nullable=False, default="")

    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=_now, onupdate=_now
    )
    last_match_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    # We bump this whenever the style summary decays past its window; helps the
    # Soul decide whether to trust the summary verbatim or treat it as stale.
    features_version: Mapped[int] = mapped_column(Integer, nullable=False, default=0)


class MatchAnalysisStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class MatchAnalysis(Base):
    """Per-match post-game processing state + outputs.

    One row per match. Created when the match finalizes; updated in-place
    as the background processor advances through its pipeline. The
    polling endpoint reads from here.
    """

    __tablename__ = "match_analyses"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    match_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("matches.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )

    status: Mapped[MatchAnalysisStatus] = mapped_column(
        Enum(MatchAnalysisStatus, name="match_analysis_status"),
        nullable=False,
        default=MatchAnalysisStatus.PENDING,
    )
    # Ordered list of step names ("engine_analysis", "feature_extraction",
    # "elo_ratchet", "memory_generation", "narrative_summary") — for UI progress.
    steps_completed: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Step outputs
    engine_analysis: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    features: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    critical_moments: Mapped[list[dict[str, Any]]] = mapped_column(JSON, nullable=False, default=list)
    elo_delta_raw: Mapped[float | None] = mapped_column(Float, nullable=True)
    elo_delta_applied: Mapped[int | None] = mapped_column(Integer, nullable=True)
    floor_raised: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    generated_memory_ids: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)

    started_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=_now)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
