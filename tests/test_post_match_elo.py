"""compute_elo_delta — Patch Pass 2 Item 2 spec (expected-score formula)."""

from __future__ import annotations

import math

from app.models.match import Color, Match, MatchResult, MatchStatus
from app.post_match.elo_apply import (
    K_ESTABLISHED,
    K_NEW,
    compute_elo_delta,
    expected_score,
)


def _match(
    result: MatchResult,
    status=MatchStatus.COMPLETED,
    player_color=Color.WHITE,
    move_count: int = 30,
    char_elo_start: int = 1500,
    player_elo_start: int = 1200,
) -> Match:
    return Match(
        id="m-1",
        character_id="c-1",
        player_id="p-1",
        status=status,
        result=result,
        player_color=player_color,
        initial_fen="startpos",
        current_fen="startpos",
        move_count=move_count,
        character_elo_at_start=char_elo_start,
        player_elo_at_start=player_elo_start,
    )


def _moves(char_losses, opp_losses, *, char_color="black"):
    out = []
    mn = 1
    opp_color = "white" if char_color == "black" else "black"
    for loss in char_losses:
        out.append({"move_number": mn, "side": char_color, "eval_loss_cp": loss})
        mn += 1
    for loss in opp_losses:
        out.append({"move_number": mn, "side": opp_color, "eval_loss_cp": loss})
        mn += 1
    return out


def test_expected_score_equal_ratings_is_half():
    assert math.isclose(expected_score(1500, 1500), 0.5, abs_tol=1e-9)


def test_expected_score_favourite_wins_big():
    # 400-point advantage → expected ~0.909
    assert expected_score(1900, 1500) > 0.9


def test_equal_ratings_draw_is_zero_change():
    # char 1500 vs player 1500, draw, no analysis → actual=0.5, expected=0.5
    match = _match(MatchResult.DRAW, player_color=Color.WHITE, move_count=30,
                   char_elo_start=1500, player_elo_start=1500)
    comp = compute_elo_delta(match=match, analysis_moves=[])
    assert math.isclose(comp.elo_delta_raw, 0.0, abs_tol=0.01)
    assert math.isclose(comp.player_elo_delta_raw, 0.0, abs_tol=0.01)


def test_underdog_win_big_gain():
    # char is 2200, player is 1500, but char LOSES → massive negative delta.
    # Expected for char = ~0.983, actual = 0.0, K=32 (new).
    # elo_change = 32 * (0 - 0.983) ≈ -31.5
    match = _match(MatchResult.WHITE_WIN, player_color=Color.WHITE, move_count=40,
                   char_elo_start=2200, player_elo_start=1500)
    comp = compute_elo_delta(match=match, analysis_moves=[])
    assert comp.k_factor == K_NEW
    assert comp.outcome_delta < -30
    # Player upset → big positive gain.
    assert comp.player_elo_delta_raw > 30


def test_favourite_beats_underdog_small_gain():
    # char 1800 vs player 1200 → expected ~0.969. Char wins, actual=1.
    # change ≈ 32 * (1 - 0.969) ≈ 1.
    match = _match(MatchResult.BLACK_WIN, player_color=Color.WHITE, move_count=40,
                   char_elo_start=1800, player_elo_start=1200)
    comp = compute_elo_delta(match=match, analysis_moves=[])
    # Character is BLACK (player is WHITE), BLACK_WIN → char wins.
    assert comp.actual_score == 1.0
    assert 0 < comp.outcome_delta < 3


def test_short_match_scaling():
    # 5-ply match: final delta multiplied by 0.3.
    match = _match(MatchResult.BLACK_WIN, player_color=Color.WHITE, move_count=5,
                   char_elo_start=1500, player_elo_start=1500)
    comp = compute_elo_delta(match=match, analysis_moves=[])
    assert comp.short_match_halved is True
    # Equal ratings, win, K=32 → raw ≈ 16 before scale, ≈ 4.8 after.
    assert 4.0 < comp.elo_delta_raw < 5.5


def test_zero_move_resign_clamps_to_plus_minus_3():
    match = _match(MatchResult.BLACK_WIN, status=MatchStatus.RESIGNED,
                   player_color=Color.WHITE, move_count=0,
                   char_elo_start=1500, player_elo_start=1500)
    comp = compute_elo_delta(match=match, analysis_moves=[])
    assert comp.short_match_halved is True
    assert abs(comp.elo_delta_raw) <= 3.0
    assert abs(comp.player_elo_delta_raw) <= 3.0


def test_rage_quit_skips_quality_but_keeps_outcome():
    # Player rage-quits → ABANDONED, character wins.
    match = _match(MatchResult.ABANDONED, status=MatchStatus.ABANDONED,
                   player_color=Color.WHITE, move_count=20,
                   char_elo_start=1500, player_elo_start=1500)
    # Provide analysis moves that WOULD shift quality, to confirm they're
    # ignored.
    moves = _moves([0] * 10, [300] * 10)
    comp = compute_elo_delta(match=match, analysis_moves=moves)
    assert comp.rage_quit_skipped_quality is True
    assert comp.move_quality_delta == 0.0
    # Actual=1.0, expected=0.5, K=32 → raw=+16 exact.
    assert math.isclose(comp.elo_delta_raw, 16.0, abs_tol=0.01)


def test_move_quality_is_capped_at_pm_10():
    # Equal ratings, draw, but opponent blundered massively.
    char_losses = [0] * 10
    opp_losses = [300] * 10  # 3000cp diff → /100 = 30 → cap 10.
    match = _match(MatchResult.DRAW, move_count=20,
                   char_elo_start=1500, player_elo_start=1500)
    comp = compute_elo_delta(match=match, analysis_moves=_moves(char_losses, opp_losses))
    assert comp.move_quality_delta == 10.0
    # Player mirror = -10.
    assert math.isclose(comp.player_elo_delta_raw, -10.0, abs_tol=0.1)


def test_k_factor_switches_after_30_games():
    match = _match(MatchResult.DRAW, move_count=30)
    comp = compute_elo_delta(
        match=match, analysis_moves=[],
        character_games_played=30, player_games_played=5,
    )
    assert comp.k_factor == K_ESTABLISHED
    assert comp.player_k_factor == K_NEW
