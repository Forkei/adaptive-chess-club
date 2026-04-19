"""compute_elo_delta — spec formula."""

from __future__ import annotations

from app.models.match import Color, Match, MatchResult, MatchStatus
from app.post_match.elo_apply import compute_elo_delta


def _match(result: MatchResult, status=MatchStatus.COMPLETED, player_color=Color.WHITE) -> Match:
    m = Match(
        id="m-1",
        character_id="c-1",
        player_id="p-1",
        status=status,
        result=result,
        player_color=player_color,
        initial_fen="startpos",
        current_fen="startpos",
        move_count=30,
        character_elo_at_start=1500,
    )
    return m


def _moves(char_losses, opp_losses, *, char_color="black"):
    """Build analysis_moves with per-side eval_loss lists."""
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


def test_clean_win_outcome_only_when_no_moves():
    match = _match(MatchResult.BLACK_WIN, player_color=Color.WHITE)  # char is black, wins
    comp = compute_elo_delta(match=match, analysis_moves=[])
    # Empty analysis → move_quality = 0, short match halving kicks in
    # (< 10 half-moves). outcome=+200, halved to +100.
    assert comp.outcome_delta == 200.0
    assert comp.move_quality_delta == 0.0
    assert comp.elo_delta_raw == 100.0
    assert comp.short_match_halved is True


def test_loss_with_character_blundering():
    match = _match(MatchResult.WHITE_WIN, player_color=Color.WHITE)  # char black, loses
    # Character lost a total of 250cp, opponent 50cp: opp - char = -200 / 10 = -20.
    moves = _moves([100, 150], [20, 30], char_color="black")  # 4 half-moves — short
    comp = compute_elo_delta(match=match, analysis_moves=moves)
    assert comp.outcome_delta == -200.0
    assert comp.move_quality_delta == -20.0
    # Halved since 4 < 10.
    assert comp.elo_delta_raw == (-200.0 + -20.0) / 2.0


def test_draw_with_opponent_blundering_still_gives_positive_quality():
    match = _match(MatchResult.DRAW, player_color=Color.WHITE)
    # Character loss 0, opponent loss 500 → quality = 500/10 = 50.
    # 14 half-moves total, not short.
    char_losses = [0] * 7
    opp_losses = [100, 100, 100, 100, 100, 0, 0]
    moves = _moves(char_losses, opp_losses, char_color="black")
    comp = compute_elo_delta(match=match, analysis_moves=moves)
    assert comp.outcome_delta == 0.0
    assert comp.move_quality_delta == 50.0
    assert comp.elo_delta_raw == 50.0
    assert comp.short_match_halved is False


def test_rage_quit_skips_quality():
    # Player rage-quits → character wins; only outcome counts.
    match = _match(MatchResult.ABANDONED, status=MatchStatus.ABANDONED)
    moves = _moves([0, 0, 0, 0, 0], [200, 200, 200, 200, 200])  # 10 — long
    comp = compute_elo_delta(match=match, analysis_moves=moves)
    assert comp.rage_quit_skipped_quality is True
    # outcome is +200 (player abandoned → char wins).
    assert comp.outcome_delta == 200.0
    assert comp.move_quality_delta == 0.0
    # Not halved because abandoned matches bypass the short-match rule.
    assert comp.short_match_halved is False
    assert comp.elo_delta_raw == 200.0


def test_move_quality_delta_capped_at_100():
    match = _match(MatchResult.DRAW, player_color=Color.WHITE)
    # Opponent's loss would drive quality >100 — cap.
    char_losses = [0] * 6
    opp_losses = [300] * 8  # 8*300 = 2400 → /10 = 240 → cap to 100.
    moves = _moves(char_losses, opp_losses, char_color="black")
    comp = compute_elo_delta(match=match, analysis_moves=moves)
    assert comp.move_quality_delta == 100.0


def test_move_quality_delta_floored_at_minus_100():
    match = _match(MatchResult.WHITE_WIN, player_color=Color.WHITE)
    char_losses = [300] * 8
    opp_losses = [0] * 6
    moves = _moves(char_losses, opp_losses, char_color="black")
    comp = compute_elo_delta(match=match, analysis_moves=moves)
    assert comp.move_quality_delta == -100.0
