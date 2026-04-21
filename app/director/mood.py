"""Mood state + smoothing + persistence.

Four axes, each in [0, 1]:
- aggression  — willingness to attack
- confidence  — self-belief right now
- tilt        — frustration / rattled-ness
- engagement  — how invested in this specific game

Raw mood comes from character traits + Soul updates. Smoothed mood
(exponential, tau = 3 moves) is what the Director actually reads.
"""

from __future__ import annotations

import math
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.models.character import Character

# Exponential smoothing time constant, in moves.
SMOOTHING_TAU_MOVES = 3.0
# Corresponding per-step alpha = 1 - e^(-1/tau).
SMOOTHING_ALPHA = 1.0 - math.exp(-1.0 / SMOOTHING_TAU_MOVES)


class MoodState(BaseModel):
    model_config = ConfigDict(frozen=False)

    aggression: float = Field(..., ge=0.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    tilt: float = Field(..., ge=0.0, le=1.0)
    engagement: float = Field(..., ge=0.0, le=1.0)

    def to_dict(self) -> dict[str, float]:
        return {
            "aggression": self.aggression,
            "confidence": self.confidence,
            "tilt": self.tilt,
            "engagement": self.engagement,
        }


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def initial_mood_for_character(
    character: Character,
    *,
    tone_bias: dict[str, float] | None = None,
) -> MoodState:
    """Seed the mood from character sliders.

    Rationale: neutral initialization would flatten character differences in
    the opening phase. We derive a starting mood from the Phase 1 sliders
    so a high-aggression/high-trash-talk character enters the game already
    hot, not neutral.

    Phase 4.3 — an optional `tone_bias` dict may carry evolution-driven
    additive offsets for `confidence` and `tilt` baselines. Callers that
    have loaded a `CharacterEvolutionState` pass `tone_bias_for(state)`
    here; call-sites without an evolution context omit it and behaviour
    is unchanged.
    """
    aggression = _clamp(character.aggression / 10.0)
    engagement = _clamp(0.6 + (character.trash_talk - 5) * 0.05)
    confidence = 0.5
    tilt = 0.0
    if tone_bias:
        confidence = _clamp(confidence + float(tone_bias.get("confidence_baseline", 0.0)))
        tilt = _clamp(tilt + float(tone_bias.get("tilt_baseline", 0.0)) * -1.0)
        # Note: `tilt_baseline` is negative on loss streaks; multiplied by
        # -1 so a loss streak raises the tilt mood component.
    return MoodState(
        aggression=aggression,
        confidence=confidence,
        tilt=tilt,
        engagement=engagement,
    )


def smooth_mood(previous: MoodState, raw: MoodState, alpha: float = SMOOTHING_ALPHA) -> MoodState:
    """Exponential smoothing; `alpha` is the per-step weight on the new reading."""
    alpha = _clamp(alpha, 0.0, 1.0)
    return MoodState(
        aggression=_clamp((1 - alpha) * previous.aggression + alpha * raw.aggression),
        confidence=_clamp((1 - alpha) * previous.confidence + alpha * raw.confidence),
        tilt=_clamp((1 - alpha) * previous.tilt + alpha * raw.tilt),
        engagement=_clamp((1 - alpha) * previous.engagement + alpha * raw.engagement),
    )


def apply_deltas(mood: MoodState, deltas: dict[str, float]) -> MoodState:
    """Additive deltas (then clamp). The Soul will use this in Phase 2b."""
    return MoodState(
        aggression=_clamp(mood.aggression + deltas.get("aggression", 0.0)),
        confidence=_clamp(mood.confidence + deltas.get("confidence", 0.0)),
        tilt=_clamp(mood.tilt + deltas.get("tilt", 0.0)),
        engagement=_clamp(mood.engagement + deltas.get("engagement", 0.0)),
    )


# --- Persistence (Redis-backed, with in-memory fallback) ---


def _mood_key(match_id: str, *, smoothed: bool) -> str:
    suffix = "smoothed" if smoothed else "raw"
    return f"mood:{match_id}:{suffix}"


def load_mood(match_id: str, *, smoothed: bool = True) -> MoodState | None:
    from app.redis_client import get as kv_get

    raw = kv_get(_mood_key(match_id, smoothed=smoothed))
    if raw is None:
        return None
    return MoodState.model_validate(raw)


def save_mood(match_id: str, mood: MoodState, *, smoothed: bool, ttl_s: int = 60 * 60 * 24) -> None:
    from app.redis_client import set_ as kv_set

    kv_set(_mood_key(match_id, smoothed=smoothed), mood.to_dict(), ttl_s=ttl_s)


def mood_from_dict(d: dict[str, Any]) -> MoodState:
    return MoodState.model_validate(d)
