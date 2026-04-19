"""Pydantic schemas shared by the Subconscious + Soul agents.

These are the structured-output shapes the LLM must conform to, plus the
internal `SurfacedMemory` object passed between Subconscious and Soul.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


# --- Subconscious outputs --------------------------------------------------


class SurfacedMemory(BaseModel):
    """A memory the Subconscious has decided is worth giving to the Soul.

    `from_cache` is True when this item was returned from the per-match
    cache rather than freshly scored — the Soul uses this to frame the
    memory as "still top of mind" rather than "newly recalled".
    """

    model_config = ConfigDict(frozen=False)

    memory_id: str
    narrative_text: str
    triggers: list[str]
    relevance_tags: list[str]
    emotional_valence: float
    scope: str  # string form of MemoryScope — avoids enum import cycles in LLM prompts
    score: float
    retrieval_reason: str
    from_cache: bool = False


# --- Soul output -----------------------------------------------------------


class MoodDeltas(BaseModel):
    """Additive nudges to the raw mood. Each ±0.1 max per turn."""

    aggression: float = Field(default=0.0, ge=-0.1, le=0.1)
    confidence: float = Field(default=0.0, ge=-0.1, le=0.1)
    tilt: float = Field(default=0.0, ge=-0.1, le=0.1)
    engagement: float = Field(default=0.0, ge=-0.1, le=0.1)

    def to_dict(self) -> dict[str, float]:
        return {
            "aggression": self.aggression,
            "confidence": self.confidence,
            "tilt": self.tilt,
            "engagement": self.engagement,
        }


Emotion = Literal[
    "neutral",
    "pleased",
    "annoyed",
    "excited",
    "focused",
    "uncertain",
    "smug",
    "deflated",
]


class SoulResponse(BaseModel):
    """Structured output the Soul LLM must return every character turn.

    `speak` being None is the usual case — most moves are silent. The
    Soul still runs every turn because it also produces mood deltas and
    queues opponent notes regardless of whether it speaks.
    """

    model_config = ConfigDict(frozen=False)

    speak: str | None = Field(
        default=None,
        description="Chat message the character says aloud after this move. "
        "Leave null for silent moves (the usual case). 1-3 sentences typically.",
    )
    emotion: Emotion = Field(
        default="neutral",
        description="Current emotional state after seeing this move and context.",
    )
    emotion_intensity: float = Field(default=0.3, ge=0.0, le=1.0)
    mood_deltas: MoodDeltas = Field(default_factory=MoodDeltas)
    note_about_opponent: str | None = Field(
        default=None,
        description="A short observation about the opponent, queued for post-match "
        "processing. Optional; leave null if nothing new stood out. Not shown to the player.",
    )
    referenced_memory_ids: list[str] = Field(
        default_factory=list,
        description="IDs (from the surfaced memories shown in the prompt) that shaped "
        "this response. Empty if no memory was relevant.",
    )
    internal_thinking: str | None = Field(
        default=None,
        description="Optional debug trace — logged, never shown to the player. "
        "Use to explain the reasoning in 1-2 sentences. Can be null.",
    )
