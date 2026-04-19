"""Unit tests for retrieval scoring helpers (app/agents/retrieval.py).

Pure functions — no DB, no LLM, fully deterministic.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from app.agents.retrieval import (
    build_context_tokens,
    mood_alignment_score,
    mood_polarity_bucket,
    mood_polarity_score,
    opponent_relevance_score,
    recency_penalty,
    tokenize,
    trigger_match_score,
)
from app.director.mood import MoodState
from app.models.memory import Memory, MemoryScope, MemoryType


def _mem(
    *,
    triggers=None,
    tags=None,
    valence=0.0,
    player_id=None,
    scope=MemoryScope.CHARACTER_LORE,
    surface_count=0,
    last_surfaced_at=None,
):
    m = Memory(
        character_id="c1",
        player_id=player_id,
        scope=scope,
        type=MemoryType.OBSERVATION,
        emotional_valence=valence,
        triggers=triggers or [],
        narrative_text="x",
        relevance_tags=tags or [],
    )
    m.id = "mem-xyz"
    m.surface_count = surface_count
    m.last_surfaced_at = last_surfaced_at
    return m


# --- mood polarity ---------------------------------------------------------


def test_mood_polarity_dominant_high_confidence_low_tilt():
    mood = MoodState(aggression=0.8, confidence=0.95, tilt=0.0, engagement=0.8)
    assert mood_polarity_bucket(mood) == "dominant"
    assert mood_polarity_score(mood) > 0.4


def test_mood_polarity_deflated_low_confidence_high_tilt():
    mood = MoodState(aggression=0.2, confidence=0.1, tilt=0.9, engagement=0.2)
    assert mood_polarity_bucket(mood) == "deflated"
    assert mood_polarity_score(mood) < -0.4


def test_mood_polarity_steady_is_near_zero():
    mood = MoodState(aggression=0.5, confidence=0.55, tilt=0.0, engagement=0.5)
    assert mood_polarity_bucket(mood) == "steady"


def test_mood_polarity_buckets_cover_all_six():
    samples = [
        MoodState(aggression=0.2, confidence=0.1, tilt=0.9, engagement=0.2),  # deflated
        MoodState(aggression=0.3, confidence=0.3, tilt=0.5, engagement=0.3),  # tense
        MoodState(aggression=0.4, confidence=0.4, tilt=0.1, engagement=0.4),  # guarded
        MoodState(aggression=0.5, confidence=0.55, tilt=0.0, engagement=0.5),  # steady
        MoodState(aggression=0.6, confidence=0.7, tilt=0.0, engagement=0.6),  # confident
        MoodState(aggression=0.9, confidence=0.95, tilt=0.0, engagement=0.9),  # dominant
    ]
    buckets = {mood_polarity_bucket(m) for m in samples}
    assert {"deflated", "confident", "dominant", "steady"} <= buckets


# --- trigger match ---------------------------------------------------------


def test_trigger_match_zero_when_no_triggers():
    assert trigger_match_score(_mem(), set()) == 0.0


def test_trigger_match_counts_overlap_per_trigger():
    mem = _mem(triggers=["Sicilian Najdorf", "Reykjavik", "queen sacrifice"])
    ctx = tokenize("we are playing the najdorf line today")
    score = trigger_match_score(mem, ctx)
    # 1 of 3 triggers has a matching token.
    assert abs(score - 1 / 3) < 1e-6


def test_trigger_match_multiword_partial_match_counts():
    mem = _mem(triggers=["opposite-side castling"])
    ctx = tokenize("opposite side pawn storm")
    assert trigger_match_score(mem, ctx) == 1.0


# --- opponent relevance ----------------------------------------------------


def test_opponent_relevance_direct_player_match_is_one():
    mem = _mem(player_id="p1")
    score = opponent_relevance_score(mem, current_player_id="p1", opponent_style_features=None)
    assert score == 1.0


def test_opponent_relevance_cross_player_with_matching_archetype_scores_half():
    mem = _mem(triggers=["blitz hustler"], scope=MemoryScope.CROSS_PLAYER, player_id=None)
    score = opponent_relevance_score(
        mem,
        current_player_id="p1",
        opponent_style_features={"archetype": "blitz hustler"},
    )
    assert score == 0.5


def test_opponent_relevance_unrelated_memory_is_zero():
    mem = _mem(player_id="other")
    assert opponent_relevance_score(mem, current_player_id="p1", opponent_style_features=None) == 0.0


# --- mood alignment --------------------------------------------------------


def test_mood_alignment_positive_mood_positive_valence_high():
    mood = MoodState(aggression=0.8, confidence=0.95, tilt=0.0, engagement=0.8)
    mem = _mem(valence=0.8)
    score = mood_alignment_score(mem, mood)
    # Polarity caps around +0.75 for max mood, so alignment with positive
    # valence ≈ 1 - |0.75 - 1|/2 ≈ 0.875. Anything >0.8 is aligned.
    assert score >= 0.8


def test_mood_alignment_opposite_polarity_is_zero():
    mood = MoodState(aggression=0.2, confidence=0.05, tilt=0.95, engagement=0.2)  # deflated
    mem = _mem(valence=0.9)
    score = mood_alignment_score(mem, mood)
    assert score < 0.1


# --- recency penalty -------------------------------------------------------


def test_recency_penalty_zero_if_never_surfaced():
    mem = _mem(surface_count=0, last_surfaced_at=None)
    assert recency_penalty(mem) == 0.0


def test_recency_penalty_decays_over_7_days():
    now = datetime.utcnow()
    recent = _mem(surface_count=3, last_surfaced_at=now - timedelta(hours=1))
    older = _mem(surface_count=3, last_surfaced_at=now - timedelta(days=5))
    ancient = _mem(surface_count=3, last_surfaced_at=now - timedelta(days=9))

    r = recency_penalty(recent, now=now)
    o = recency_penalty(older, now=now)
    a = recency_penalty(ancient, now=now)

    assert r > o > 0
    assert a == 0.0  # past 7 days, fully decayed


def test_recency_penalty_caps_surface_count_contribution():
    now = datetime.utcnow()
    ten = _mem(surface_count=10, last_surfaced_at=now)
    fifty = _mem(surface_count=50, last_surfaced_at=now)
    # Both clamp to 1.0 * recency_factor, so identical.
    assert abs(recency_penalty(ten, now=now) - recency_penalty(fifty, now=now)) < 1e-9


# --- context token building ------------------------------------------------


def test_tokenize_splits_on_hyphens_and_punctuation():
    # Hyphens split: "opposite-side" becomes two tokens, so triggers like
    # "opposite-side castling" match prose like "opposite side pawn storm".
    assert tokenize("Sicilian-Najdorf E4!") == {"sicilian", "najdorf", "e4"}


def test_build_context_tokens_combines_all_sources():
    tokens = build_context_tokens(
        board_prose="king in center, open e file",
        opening_label="Ruy Lopez",
        opponent_style_features={"archetype": "blitz hustler"},
        last_player_chat="I love tactics",
        tactical_themes=["e7 knight pinned"],
    )
    assert "king" in tokens
    assert "ruy" in tokens and "lopez" in tokens
    assert "blitz" in tokens and "hustler" in tokens
    assert "tactics" in tokens
    assert "pinned" in tokens
