from __future__ import annotations

import math

from app.director.mood import (
    MoodState,
    SMOOTHING_ALPHA,
    SMOOTHING_TAU_MOVES,
    apply_deltas,
    initial_mood_for_character,
    smooth_mood,
)
from app.models.character import Character


def _char(**overrides) -> Character:
    d: dict = dict(
        name="T",
        aggression=5,
        risk_tolerance=5,
        patience=5,
        trash_talk=5,
        target_elo=1400,
        current_elo=1400,
        floor_elo=1400,
        max_elo=1800,
    )
    d.update(overrides)
    return Character(**d)


def test_initial_mood_maps_aggression_slider():
    mood_low = initial_mood_for_character(_char(aggression=1))
    mood_mid = initial_mood_for_character(_char(aggression=5))
    mood_high = initial_mood_for_character(_char(aggression=10))
    assert abs(mood_low.aggression - 0.1) < 1e-6
    assert abs(mood_mid.aggression - 0.5) < 1e-6
    assert abs(mood_high.aggression - 1.0) < 1e-6


def test_initial_mood_has_neutral_confidence_and_zero_tilt():
    m = initial_mood_for_character(_char())
    assert m.confidence == 0.5
    assert m.tilt == 0.0


def test_trash_talk_biases_engagement():
    quiet = initial_mood_for_character(_char(trash_talk=1))
    baseline = initial_mood_for_character(_char(trash_talk=5))
    loud = initial_mood_for_character(_char(trash_talk=10))
    # +0.05 per point above 5, -0.05 per point below.
    assert abs(baseline.engagement - 0.6) < 1e-6
    assert abs(quiet.engagement - (0.6 - 0.05 * 4)) < 1e-6
    assert abs(loud.engagement - (0.6 + 0.05 * 5)) < 1e-6


def test_trash_talk_engagement_clamped():
    # Extreme cases shouldn't push out of [0,1].
    m = initial_mood_for_character(_char(trash_talk=10))
    assert 0 <= m.engagement <= 1
    m = initial_mood_for_character(_char(trash_talk=1))
    assert 0 <= m.engagement <= 1


def test_smoothing_alpha_matches_time_constant():
    expected = 1.0 - math.exp(-1.0 / SMOOTHING_TAU_MOVES)
    assert abs(SMOOTHING_ALPHA - expected) < 1e-9


def test_smoothing_is_weighted_average():
    prev = MoodState(aggression=0.0, confidence=0.0, tilt=0.0, engagement=0.0)
    raw = MoodState(aggression=1.0, confidence=1.0, tilt=1.0, engagement=1.0)
    out = smooth_mood(prev, raw, alpha=0.25)
    for v in out.to_dict().values():
        assert abs(v - 0.25) < 1e-9


def test_smoothing_converges_over_many_steps():
    prev = MoodState(aggression=0.0, confidence=0.0, tilt=0.0, engagement=0.0)
    raw = MoodState(aggression=0.8, confidence=0.8, tilt=0.8, engagement=0.8)
    current = prev
    for _ in range(30):
        current = smooth_mood(current, raw)
    for v in current.to_dict().values():
        # With tau=3, 30 steps is many tau — should be close to raw.
        assert abs(v - 0.8) < 0.01


def test_apply_deltas_adds_and_clamps():
    m = MoodState(aggression=0.5, confidence=0.5, tilt=0.0, engagement=0.5)
    out = apply_deltas(m, {"aggression": 0.3, "tilt": 0.2, "confidence": -0.1})
    assert abs(out.aggression - 0.8) < 1e-9
    assert abs(out.tilt - 0.2) < 1e-9
    assert abs(out.confidence - 0.4) < 1e-9
    # Clamp.
    out2 = apply_deltas(m, {"confidence": -2.0, "tilt": 2.0})
    assert out2.confidence == 0.0
    assert out2.tilt == 1.0
