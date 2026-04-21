"""Phase 4.3 — agent evolution tests.

Covers pure math functions, the apply_evolution entry point, and
long-horizon clamp guarantees.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta

import pytest

from app.db import SessionLocal
from app.models.character import (
    Character,
    CharacterState,
    ContentRating,
    Visibility,
)
from app.models.evolution import CharacterEvolutionState
from app.models.match import Color, Match, MatchResult, MatchStatus, Player
from app.models.memory import Memory, MemoryType
from app.post_match import evolution as ev


# --- seed helpers --------------------------------------------------------


def _mk_character(s, **over) -> Character:
    defaults = dict(
        name="TestChar", short_description="x",
        backstory="A rich backstory." * 4,
        voice_descriptor="voice",
        target_elo=1500, current_elo=1500, floor_elo=1400, max_elo=1800,
        adaptive=True, is_preset=False, owner_id=None,
        state=CharacterState.READY,
        visibility=Visibility.PUBLIC,
        content_rating=ContentRating.FAMILY,
        aggression=5, risk_tolerance=5, patience=5, trash_talk=5,
    )
    defaults.update(over)
    c = Character(**defaults)
    s.add(c); s.commit(); s.refresh(c)
    return c


def _mk_player(s, username="p", elo=1500) -> Player:
    p = Player(username=username, display_name=username, elo=elo)
    s.add(p); s.commit(); s.refresh(p)
    return p


def _mk_match(
    s, *, character, player, result=MatchResult.WHITE_WIN,
    player_color=Color.WHITE, status=MatchStatus.COMPLETED,
    char_elo_start=1500, player_elo_start=1500, is_private=False,
    ended_at=None, move_count=30,
) -> Match:
    m = Match(
        character_id=character.id, player_id=player.id,
        status=status, result=result, player_color=player_color,
        initial_fen="startpos", current_fen="startpos",
        move_count=move_count,
        character_elo_at_start=char_elo_start, player_elo_at_start=player_elo_start,
        is_private=is_private,
        ended_at=ended_at or datetime.utcnow(),
    )
    s.add(m); s.commit(); s.refresh(m)
    return m


# --- slider drift math --------------------------------------------------


def test_slider_nudge_lost_cautious_bumps_aggression():
    nudge = ev.select_slider_nudge(
        won=False, lost=True, char_acpl=15,
        trash_talk_base=5, trash_talk_drift=0.0,
    )
    assert nudge == ("aggression", +ev.SLIDER_DELTA_STEP)


def test_slider_nudge_lost_reckless_bumps_patience():
    nudge = ev.select_slider_nudge(
        won=False, lost=True, char_acpl=120,
        trash_talk_base=5, trash_talk_drift=0.0,
    )
    assert nudge == ("patience", +ev.SLIDER_DELTA_STEP)


def test_slider_nudge_on_a_decisive_win_returns_none():
    """Wins don't nudge a slider — they nudge tone. Ensures the character
    doesn't drift just because they played a weak opponent and won."""
    nudge = ev.select_slider_nudge(
        won=True, lost=False, char_acpl=25,
        trash_talk_base=5, trash_talk_drift=0.0,
    )
    assert nudge is None


def test_slider_nudge_homeostasis_pulls_trash_talk_back():
    """If trash_talk_drift has drifted positive, homeostasis should
    pull it back toward zero rather than letting it keep climbing."""
    nudge = ev.select_slider_nudge(
        won=True, lost=False, char_acpl=25,
        trash_talk_base=5, trash_talk_drift=1.5,
    )
    assert nudge is not None
    slider, delta = nudge
    assert slider == "trash_talk"
    assert delta < 0


def test_apply_slider_drift_clamps_cumulatively():
    drift = {}
    for _ in range(30):
        drift = ev.apply_slider_drift(drift, ("aggression", +ev.SLIDER_DELTA_STEP))
    # 30 × +0.5 = 15.0, should clamp to SLIDER_DRIFT_CLAMP.
    assert drift["aggression"] == ev.SLIDER_DRIFT_CLAMP


# --- opening ema --------------------------------------------------------


def test_opening_ema_moves_toward_signal():
    openings = ev.opening_ema_step({}, opening_label="Sicilian Najdorf", signal=1.0)
    assert openings["Sicilian Najdorf"] == pytest.approx(ev.OPENING_EMA_ALPHA)
    openings = ev.opening_ema_step(openings, opening_label="Sicilian Najdorf", signal=1.0)
    assert openings["Sicilian Najdorf"] > ev.OPENING_EMA_ALPHA


def test_opening_ema_clamps_and_ignores_empty_label():
    openings = ev.opening_ema_step({}, opening_label=None, signal=1.0)
    assert openings == {}


# --- trap detection + memory -------------------------------------------


def test_detect_trap_when_character_blunders_early():
    cms = [
        {"ply": 6, "side": "white", "eval_loss_cp": 500, "pattern": "scholar_mate"},
    ]
    trap = ev.detect_trap(critical_moments=cms, character_is_white=True)
    assert trap is not None
    assert trap["fell_for"] is True
    assert trap["pattern"] == "scholar_mate"


def test_detect_trap_ignores_opponent_blunder():
    """If the OPPONENT blundered early, we might still return an entry
    (as a trick the character used) but with fell_for=False."""
    cms = [{"ply": 4, "side": "black", "eval_loss_cp": 600, "pattern": "gambit_bluff"}]
    trap = ev.detect_trap(critical_moments=cms, character_is_white=True)
    assert trap is not None
    assert trap["fell_for"] is False


def test_detect_trap_none_for_late_blunder():
    cms = [{"ply": 25, "side": "white", "eval_loss_cp": 800, "pattern": "late_mistake"}]
    trap = ev.detect_trap(critical_moments=cms, character_is_white=True)
    assert trap is None


def test_update_trap_memory_first_time_sets_brand_new_flag():
    entries, brand_new = ev.update_trap_memory(
        [], detected={"pattern": "scholar", "fell_for": True, "ply": 6, "eval_loss_cp": 500},
        now=datetime.utcnow(),
    )
    assert brand_new is True
    assert len(entries) == 1
    assert entries[0]["pattern"] == "scholar"
    assert entries[0]["fell_for"] == 1


def test_update_trap_memory_second_time_bumps_counter():
    first, _ = ev.update_trap_memory(
        [], detected={"pattern": "scholar", "fell_for": True, "ply": 6, "eval_loss_cp": 500},
        now=datetime.utcnow(),
    )
    second, brand_new = ev.update_trap_memory(
        first, detected={"pattern": "scholar", "fell_for": True, "ply": 8, "eval_loss_cp": 450},
        now=datetime.utcnow(),
    )
    assert brand_new is False
    assert second[0]["fell_for"] == 2


# --- tone drift ---------------------------------------------------------


def test_tone_drift_moves_toward_streak_target():
    after = ev.tone_ema_step({}, win_streak=5, loss_streak=0)
    assert after["confidence_baseline"] > 0
    # Single step can't reach the target.
    assert after["confidence_baseline"] < ev.TONE_CLAMP


def test_tone_drift_clamps_over_many_steps():
    tone = {}
    for _ in range(500):
        tone = ev.tone_ema_step(tone, win_streak=10, loss_streak=0)
    assert tone["confidence_baseline"] <= ev.TONE_CLAMP + 1e-9


# --- apply_evolution: end-to-end ---------------------------------------


def test_apply_evolution_skips_private_match():
    with SessionLocal() as s:
        char = _mk_character(s)
        p = _mk_player(s, "ev_priv")
        m = _mk_match(s, character=char, player=p, is_private=True)
        summary = ev.apply_evolution(
            s, match=m, analysis_moves=[], critical_moments=[]
        )
        assert summary.skipped_private is True
        assert s.get(CharacterEvolutionState, char.id) is None


def test_apply_evolution_creates_state_on_first_run():
    with SessionLocal() as s:
        char = _mk_character(s)
        p = _mk_player(s, "ev_first", elo=1500)
        m = _mk_match(s, character=char, player=p, player_color=Color.WHITE,
                      result=MatchResult.WHITE_WIN)
        summary = ev.apply_evolution(s, match=m, analysis_moves=[], critical_moments=[])
        assert summary.skipped_private is False
        state = s.get(CharacterEvolutionState, char.id)
        assert state is not None
        assert state.matches_processed == 1
        assert state.last_match_id == m.id


def test_apply_evolution_is_idempotent():
    with SessionLocal() as s:
        char = _mk_character(s)
        p = _mk_player(s, "ev_id")
        m = _mk_match(s, character=char, player=p)
        ev.apply_evolution(s, match=m, analysis_moves=[], critical_moments=[])
        before_state = s.get(CharacterEvolutionState, char.id)
        mp_before = before_state.matches_processed
        drift_before = dict(before_state.slider_drift or {})

        summary = ev.apply_evolution(s, match=m, analysis_moves=[], critical_moments=[])
        assert summary.skipped_idempotent is True
        after_state = s.get(CharacterEvolutionState, char.id)
        assert after_state.matches_processed == mp_before
        assert (after_state.slider_drift or {}) == drift_before


def test_apply_evolution_records_trap_and_creates_learning_memory():
    with SessionLocal() as s:
        char = _mk_character(s)
        # Character played black (player_color=white). Player (white) wins
        # → character lost. Character's blunder on ply 6.
        p = _mk_player(s, "ev_trap")
        m = _mk_match(
            s, character=char, player=p,
            player_color=Color.WHITE, result=MatchResult.WHITE_WIN,
        )
        cms = [{"ply": 6, "side": "black", "eval_loss_cp": 550, "pattern": "opening_pin"}]
        summary = ev.apply_evolution(s, match=m, analysis_moves=[], critical_moments=cms)
        assert summary.trap_detected is not None
        assert summary.new_learning_memory_id is not None
        mem = s.get(Memory, summary.new_learning_memory_id)
        assert mem.type == MemoryType.LEARNING


def test_apply_evolution_second_trap_hit_bumps_counter_but_no_new_memory():
    with SessionLocal() as s:
        char = _mk_character(s)
        p = _mk_player(s, "ev_trap2")
        cms = [{"ply": 6, "side": "black", "eval_loss_cp": 550, "pattern": "opening_pin"}]

        m1 = _mk_match(
            s, character=char, player=p,
            player_color=Color.WHITE, result=MatchResult.WHITE_WIN,
            ended_at=datetime.utcnow() - timedelta(minutes=5),
        )
        ev.apply_evolution(s, match=m1, analysis_moves=[], critical_moments=cms)

        m2 = _mk_match(
            s, character=char, player=p,
            player_color=Color.WHITE, result=MatchResult.WHITE_WIN,
        )
        summary = ev.apply_evolution(s, match=m2, analysis_moves=[], critical_moments=cms)
        assert summary.new_learning_memory_id is None
        state = s.get(CharacterEvolutionState, char.id)
        entry = next((e for e in state.trap_memory if e["pattern"] == "opening_pin"), None)
        assert entry is not None
        assert entry["fell_for"] == 2


def test_50_match_simulation_respects_cumulative_clamps():
    """Long-horizon test: run the pipeline on 50 matches and confirm the
    character hasn't drifted outside its identity range."""
    with SessionLocal() as s:
        char = _mk_character(s, aggression=3, patience=8)  # calm, patient base
        p = _mk_player(s, "long", elo=1700)

        now = datetime.utcnow()
        for i in range(50):
            ended = now - timedelta(minutes=50 - i)
            m = _mk_match(
                s, character=char, player=p,
                player_color=Color.WHITE,
                result=MatchResult.WHITE_WIN,  # character loses every match (they're black)
                char_elo_start=1500, player_elo_start=1700,
                ended_at=ended,
            )
            # Half the matches — simulate reckless play (high ACPL).
            moves = [{"side": "black", "eval_loss_cp": 100}] * 15 if i % 2 else []
            ev.apply_evolution(s, match=m, analysis_moves=moves, critical_moments=[])

        state = s.get(CharacterEvolutionState, char.id)
        for slider in ("aggression", "risk_tolerance", "patience", "trash_talk"):
            assert abs(state.slider_drift.get(slider, 0.0)) <= ev.SLIDER_DRIFT_CLAMP
        assert abs(state.tone_drift.get("tilt_baseline", 0.0)) <= ev.TONE_CLAMP + 1e-9
        assert state.matches_processed == 50


# --- integration helpers (sliders + tone) ------------------------------


def test_effective_sliders_applies_drift_and_clamps_1_to_10():
    with SessionLocal() as s:
        char = _mk_character(s, aggression=9)
        state = CharacterEvolutionState(
            character_id=char.id,
            slider_drift={"aggression": +2.0},
            opening_scores={}, trap_memory=[], tone_drift={},
            matches_processed=0, last_match_id=None,
        )
        eff = ev.effective_sliders(char, state)
        assert eff["aggression"] == 10  # clamped, not 11


def test_tone_bias_for_none_state_returns_zeros():
    bias = ev.tone_bias_for(None)
    assert bias["confidence_baseline"] == 0.0
    assert bias["tilt_baseline"] == 0.0
