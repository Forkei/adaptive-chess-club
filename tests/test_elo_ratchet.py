from __future__ import annotations

from app.director.elo import (
    ELO_DELTA_CAP,
    ELO_DELTA_GAIN,
    FLOOR_RAISE_STEP,
    apply_elo_ratchet,
    outcome_delta,
)


def _ratchet(**kwargs):
    defaults = dict(
        current_elo=1500,
        floor_elo=1400,
        max_elo=1800,
        elo_delta_raw=200.0,
        recent_current_elos=[],
        adaptive=True,
    )
    defaults.update(kwargs)
    return apply_elo_ratchet(**defaults)


def test_non_adaptive_never_moves():
    r = _ratchet(adaptive=False, elo_delta_raw=1000.0)
    assert r.new_current_elo == 1500
    assert r.new_floor_elo == 1400
    assert r.current_elo_change == 0
    assert r.floor_elo_raised is False


def test_clean_win_bumps_current_elo_proportionally():
    r = _ratchet(elo_delta_raw=200.0)
    # 0.1 * 200 = 20 — under the ±30 cap.
    assert r.current_elo_change == 20
    assert r.new_current_elo == 1520


def test_big_win_clamped_to_cap():
    r = _ratchet(elo_delta_raw=1000.0)
    assert r.current_elo_change == ELO_DELTA_CAP
    assert r.new_current_elo == 1500 + ELO_DELTA_CAP


def test_big_loss_clamped_to_cap():
    r = _ratchet(current_elo=1600, floor_elo=1400, elo_delta_raw=-1000.0)
    assert r.current_elo_change == -ELO_DELTA_CAP
    assert r.new_current_elo == 1600 - ELO_DELTA_CAP


def test_cant_drop_below_floor():
    r = _ratchet(current_elo=1410, floor_elo=1400, elo_delta_raw=-500.0)
    assert r.new_current_elo == 1400
    assert r.current_elo_change == -10  # only 10 of room


def test_cant_exceed_max():
    r = _ratchet(current_elo=1780, max_elo=1800, elo_delta_raw=500.0)
    assert r.new_current_elo == 1800
    assert r.current_elo_change == 20


def test_floor_raises_after_three_consecutive_strong_showings():
    floor = 1400
    # Previous two matches: already hovering 100+ above floor.
    recent = [1510, 1520]
    # This match: also above floor + 100. Third in window → floor bump.
    r = apply_elo_ratchet(
        current_elo=1510,
        floor_elo=floor,
        max_elo=1800,
        elo_delta_raw=100.0,      # minor win — new current ~1520
        recent_current_elos=recent,
        adaptive=True,
    )
    assert r.floor_elo_raised is True
    assert r.new_floor_elo == floor + FLOOR_RAISE_STEP


def test_floor_does_not_raise_when_window_incomplete():
    # Only one recent — window isn't full.
    r = apply_elo_ratchet(
        current_elo=1600,
        floor_elo=1400,
        max_elo=1800,
        elo_delta_raw=100.0,
        recent_current_elos=[1500],
        adaptive=True,
    )
    assert r.floor_elo_raised is False
    assert r.new_floor_elo == 1400


def test_floor_does_not_raise_when_one_match_below_threshold():
    # Two above, one below — window fails.
    r = apply_elo_ratchet(
        current_elo=1510,
        floor_elo=1400,
        max_elo=1800,
        elo_delta_raw=50.0,
        recent_current_elos=[1505, 1450],  # 1450 < floor+100
        adaptive=True,
    )
    assert r.floor_elo_raised is False


def test_outcome_delta_signs():
    assert outcome_delta(character_won=True, is_draw=False) == 200.0
    assert outcome_delta(character_won=False, is_draw=False) == -200.0
    assert outcome_delta(character_won=None, is_draw=True) == 0.0


def test_elo_delta_gain_matches_constants():
    # A win with outcome_delta(+200) via apply_elo_ratchet should yield
    # a current_elo change of 20 (10% of 200), matching the spec's 0.1 gain.
    r = _ratchet(elo_delta_raw=200.0)
    assert r.current_elo_change == int(200.0 * ELO_DELTA_GAIN)
