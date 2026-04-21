"""Phase 4.3 — character evolution state.

One row per character (lazy-created on first post-match evolution run).
Holds cumulative drift across all post-match applications:

- slider_drift: signed deltas on the four personality sliders
- opening_scores: per-opening EMA of result signal in [-1, +1]
- trap_memory: list of trap patterns with fell_for / avoided counters
- tone_drift: confidence + tilt baselines (fed into MoodState initialisation)

See docs/phase_4_evolution.md for the math.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import JSON, DateTime, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


def _now() -> datetime:
    return datetime.utcnow()


class CharacterEvolutionState(Base):
    __tablename__ = "character_evolution_state"

    # Primary key = character_id (one row per character). FK enforces
    # deletion cleanup when a character is removed.
    character_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("characters.id", ondelete="CASCADE"),
        primary_key=True,
    )

    slider_drift: Mapped[dict[str, float]] = mapped_column(
        JSON, nullable=False, default=dict
    )
    opening_scores: Mapped[dict[str, float]] = mapped_column(
        JSON, nullable=False, default=dict
    )
    trap_memory: Mapped[list[dict[str, Any]]] = mapped_column(
        JSON, nullable=False, default=list
    )
    tone_drift: Mapped[dict[str, float]] = mapped_column(
        JSON, nullable=False, default=dict
    )

    matches_processed: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default="0"
    )
    # Idempotency guard — we skip re-running evolution on a match we've
    # already processed. Nullable on first-ever application.
    last_match_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    last_updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=_now, onupdate=_now
    )
