from __future__ import annotations

from app.director.director import MatchContext, choose_engine_config, compute_effective_elo
from app.director.mood import MoodState
from app.models.character import Character


def _char(**overrides) -> Character:
    d: dict = dict(
        name="T",
        aggression=5,
        risk_tolerance=5,
        patience=5,
        trash_talk=5,
        target_elo=1500,
        current_elo=1500,
        floor_elo=1500,
        max_elo=1900,
        adaptive=False,
    )
    d.update(overrides)
    return Character(**d)


def _context(engines=("mock",)) -> MatchContext:
    return MatchContext(
        move_number=5,
        game_phase="opening",
        player_color="white",
        engines_available=frozenset(engines),
    )


def _mood(**overrides) -> MoodState:
    d: dict = dict(aggression=0.5, confidence=0.5, tilt=0.0, engagement=0.6)
    d.update(overrides)
    return MoodState(**d)


def test_effective_elo_adds_confidence_and_subtracts_tilt():
    char = _char(current_elo=1500, floor_elo=1000, max_elo=2000)
    neutral = compute_effective_elo(char, _mood(confidence=0.5, tilt=0.0))
    confident = compute_effective_elo(char, _mood(confidence=1.0, tilt=0.0))
    tilted = compute_effective_elo(char, _mood(confidence=0.5, tilt=1.0))
    # Confidence=1 adds 100 over confidence=0.5.
    assert confident - neutral == 50  # (1.0 - 0.5) * 100
    # Tilt=1 subtracts 150.
    assert neutral - tilted == 150


def test_effective_elo_clamped_to_floor_max():
    char = _char(current_elo=1500, floor_elo=1400, max_elo=1600)
    cold = compute_effective_elo(char, _mood(confidence=0.0, tilt=1.0))
    hot = compute_effective_elo(char, _mood(confidence=1.0, tilt=0.0))
    assert cold == 1400
    assert hot == 1600


def test_beast_mode_triggers_stockfish_above_2100():
    char = _char(current_elo=2200, floor_elo=2000, max_elo=2400, adaptive=False)
    cfg = choose_engine_config(
        character=char,
        mood=_mood(),
        match_context=_context(engines=("maia2", "stockfish")),
    )
    assert cfg.engine_name == "stockfish"


def test_high_aggression_high_confidence_triggers_stockfish():
    char = _char(aggression=10, current_elo=1600, floor_elo=1500, max_elo=1800)
    cfg = choose_engine_config(
        character=char,
        mood=_mood(confidence=0.9),
        match_context=_context(engines=("maia2", "stockfish")),
    )
    assert cfg.engine_name == "stockfish"


def test_maia_default_in_normal_range():
    char = _char(current_elo=1500, floor_elo=1400, max_elo=1800)
    cfg = choose_engine_config(
        character=char,
        mood=_mood(),
        match_context=_context(engines=("maia2", "stockfish")),
    )
    assert cfg.engine_name == "maia2"
    assert cfg.maia_elo_bucket is not None and 1100 <= cfg.maia_elo_bucket <= 1900


def test_stockfish_fallback_below_maia_range():
    char = _char(current_elo=1000, floor_elo=900, max_elo=1100)
    cfg = choose_engine_config(
        character=char,
        mood=_mood(),
        match_context=_context(engines=("maia2", "stockfish")),
    )
    assert cfg.engine_name == "stockfish"


def test_falls_back_to_stockfish_when_maia_missing():
    char = _char(current_elo=1500)
    cfg = choose_engine_config(
        character=char, mood=_mood(), match_context=_context(engines=("stockfish",))
    )
    assert cfg.engine_name == "stockfish"


def test_falls_back_to_mock_when_only_mock_available():
    char = _char(current_elo=1500)
    cfg = choose_engine_config(
        character=char, mood=_mood(), match_context=_context(engines=("mock",))
    )
    assert cfg.engine_name == "mock"


def test_time_budget_scales_with_patience():
    slow = _char(patience=10)
    fast = _char(patience=1)
    cfg_slow = choose_engine_config(character=slow, mood=_mood(aggression=0), match_context=_context())
    cfg_fast = choose_engine_config(character=fast, mood=_mood(aggression=0), match_context=_context())
    assert cfg_slow.time_budget_seconds > cfg_fast.time_budget_seconds
    assert abs(cfg_fast.time_budget_seconds - 0.5) < 0.01
    assert abs(cfg_slow.time_budget_seconds - 5.0) < 0.01


def test_aggression_discounts_time_budget():
    char = _char(patience=5)
    calm = choose_engine_config(character=char, mood=_mood(aggression=0.0), match_context=_context())
    rabid = choose_engine_config(character=char, mood=_mood(aggression=1.0), match_context=_context())
    assert rabid.time_budget_seconds < calm.time_budget_seconds
