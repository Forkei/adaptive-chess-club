"""Scoring helpers for the Subconscious.

Pure functions — no DB, no LLM, no global state. Unit-testable in isolation.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Literal

from app.director.mood import MoodState
from app.models.memory import Memory, MemoryScope

# --- Mood polarity bucketing ----------------------------------------------

MoodPolarity = Literal[
    "deflated",   # score < -0.4
    "tense",      # -0.4..-0.15
    "guarded",    # -0.15..0
    "steady",     # 0..0.15
    "confident",  # 0.15..0.4
    "dominant",   # > 0.4
]


def mood_polarity_score(mood: MoodState) -> float:
    """Composite polarity score in roughly [-1, 1].

    (confidence - 0.5) + (aggression - 0.5) * 0.3 - tilt * 0.8 + (engagement - 0.5) * 0.2

    The weighting says: confidence dominates, tilt drags hard, aggression
    and engagement are secondary. A deflated character has low confidence
    AND high tilt; a dominant one has high confidence AND low tilt.
    """
    return (
        (mood.confidence - 0.5)
        + (mood.aggression - 0.5) * 0.3
        - mood.tilt * 0.8
        + (mood.engagement - 0.5) * 0.2
    )


def mood_polarity_bucket(mood: MoodState) -> MoodPolarity:
    s = mood_polarity_score(mood)
    if s < -0.4:
        return "deflated"
    if s < -0.15:
        return "tense"
    if s < 0.0:
        return "guarded"
    if s < 0.15:
        return "steady"
    if s < 0.4:
        return "confident"
    return "dominant"


# --- Tokenization ----------------------------------------------------------

_WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9]+")


def tokenize(text: str | None) -> set[str]:
    if not text:
        return set()
    return {m.group(0).lower() for m in _WORD_RE.finditer(text)}


def build_context_tokens(
    *,
    board_prose: str,
    opening_label: str | None,
    opponent_style_features: dict | None,
    last_player_chat: str | None,
    tactical_themes: Iterable[str],
) -> set[str]:
    tokens: set[str] = set()
    tokens |= tokenize(board_prose)
    tokens |= tokenize(opening_label or "")
    tokens |= tokenize(last_player_chat or "")
    for theme in tactical_themes:
        tokens |= tokenize(theme)
    if opponent_style_features:
        for k, v in opponent_style_features.items():
            tokens |= tokenize(k)
            if isinstance(v, str):
                tokens |= tokenize(v)
    return tokens


def trigger_match_score(memory: Memory, context_tokens: set[str]) -> float:
    """Fraction of the memory's triggers that appear in the context tokens.

    Each trigger is tokenized itself so multi-word triggers ("Sicilian
    Najdorf") match on any shared word.
    """
    triggers = list(memory.triggers or [])
    if not triggers:
        return 0.0
    hits = 0
    for trig in triggers:
        trig_tokens = tokenize(trig)
        if trig_tokens & context_tokens:
            hits += 1
    return hits / len(triggers)


# --- Opponent relevance ----------------------------------------------------


def opponent_relevance_score(
    memory: Memory,
    *,
    current_player_id: str | None,
    opponent_style_features: dict | None,
) -> float:
    """1.0 for direct player match, 0.5 for archetype-matched cross_player, else 0."""
    if current_player_id and memory.player_id == current_player_id:
        return 1.0
    if memory.scope == MemoryScope.CROSS_PLAYER:
        if not opponent_style_features:
            return 0.25  # mild prior — cross_player is broadly applicable
        # Check whether any memory trigger/tag overlaps the opponent's features.
        mem_terms: set[str] = set()
        for t in memory.triggers or []:
            mem_terms |= tokenize(t)
        for t in memory.relevance_tags or []:
            mem_terms |= tokenize(t)
        feat_terms: set[str] = set()
        for k, v in opponent_style_features.items():
            feat_terms |= tokenize(k)
            if isinstance(v, str):
                feat_terms |= tokenize(v)
        if mem_terms & feat_terms:
            return 0.5
        return 0.2
    return 0.0


# --- Mood alignment --------------------------------------------------------


def mood_alignment_score(memory: Memory, mood: MoodState) -> float:
    """score = 1 - |mood_polarity - sign(valence)| / 2 in [0, 1]."""
    polarity = max(-1.0, min(1.0, mood_polarity_score(mood)))
    valence = memory.emotional_valence
    if valence > 0:
        val_sign = 1.0
    elif valence < 0:
        val_sign = -1.0
    else:
        val_sign = 0.0
    return 1.0 - abs(polarity - val_sign) / 2.0


# --- Recency penalty -------------------------------------------------------


def recency_penalty(memory: Memory, *, now: datetime | None = None) -> float:
    """Exponential-ish penalty on re-surfacing.

    `penalty = min(1.0, surface_count * 0.1) * recency_factor`
    `recency_factor = max(0, 1 - days_since_last_surface / 7)`

    Never-surfaced memories (surface_count=0 OR last_surfaced_at=None)
    incur zero penalty.
    """
    if memory.surface_count <= 0 or memory.last_surfaced_at is None:
        return 0.0
    reference = now or datetime.utcnow()
    days = (reference - memory.last_surfaced_at).total_seconds() / 86400.0
    recency_factor = max(0.0, 1.0 - days / 7.0)
    surface_factor = min(1.0, memory.surface_count * 0.1)
    return surface_factor * recency_factor


# --- Weighted aggregation --------------------------------------------------


@dataclass(frozen=True)
class RetrievalWeights:
    semantic: float = 0.35
    trigger: float = 0.25
    opponent: float = 0.15
    mood: float = 0.10
    recency: float = -0.15


DEFAULT_WEIGHTS = RetrievalWeights()


@dataclass
class ScoreBreakdown:
    memory_id: str
    semantic: float
    trigger: float
    opponent: float
    mood: float
    recency: float
    total: float

    def dominant_axis(self) -> str:
        """Axis with the largest *positive* contribution — used to
        auto-generate a retrieval_reason when the LLM re-rank is skipped."""
        contribs = {
            "semantic": self.semantic * DEFAULT_WEIGHTS.semantic,
            "trigger": self.trigger * DEFAULT_WEIGHTS.trigger,
            "opponent": self.opponent * DEFAULT_WEIGHTS.opponent,
            "mood": self.mood * DEFAULT_WEIGHTS.mood,
        }
        return max(contribs, key=contribs.get)


def aggregate_scores(
    *,
    semantic: float,
    trigger: float,
    opponent: float,
    mood: float,
    recency: float,
    weights: RetrievalWeights = DEFAULT_WEIGHTS,
) -> float:
    return (
        weights.semantic * semantic
        + weights.trigger * trigger
        + weights.opponent * opponent
        + weights.mood * mood
        + weights.recency * recency
    )
