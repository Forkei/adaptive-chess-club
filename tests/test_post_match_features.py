"""Feature extraction + merge."""

from __future__ import annotations

from app.post_match.features import extract_features, merge_features


def _move(mn, side, san, uci="e2e4"):
    return {"move_number": mn, "side": side, "uci": uci, "san": san}


def test_extract_features_basic():
    moves = [
        _move(1, "white", "e4"),
        _move(1, "black", "c5"),
        _move(2, "white", "Nf3"),
        _move(2, "black", "d6"),
        _move(3, "white", "d4"),
        _move(3, "black", "cxd4"),  # capture — aggressive for black
    ]
    analysis = {"moves": []}
    feats = extract_features(
        moves=moves, player_color="black", analysis=analysis, abandoned=False
    )
    assert feats["typical_opening_name"] == "Sicilian Defense"
    assert feats["typical_opening_group"] == "king_pawn_open"
    assert feats["total_player_moves"] == 3
    # 1 capture out of 3 moves.
    assert abs(feats["aggression_index"] - 1 / 3) < 1e-3


def test_extract_features_empty_moves():
    feats = extract_features(moves=[], player_color="white", analysis={}, abandoned=True)
    assert feats["typical_opening_name"] == "unknown"
    assert feats["total_player_moves"] == 0
    assert feats["abandoned_last"] is True


def test_extract_features_phase_strengths_from_analysis():
    moves = [_move(mn, "white", "e4") for mn in range(1, 12)]
    moves += [_move(mn, "black", "e5") for mn in range(1, 12)]
    analysis = {
        "moves": [
            # Player (white) losses spread across phases.
            {"move_number": 2, "side": "white", "eval_loss_cp": 100, "is_blunder": False},
            {"move_number": 8, "side": "white", "eval_loss_cp": 50, "is_blunder": False},
            {"move_number": 25, "side": "white", "eval_loss_cp": 200, "is_blunder": True},
        ]
    }
    feats = extract_features(
        moves=moves, player_color="white", analysis=analysis, abandoned=False
    )
    # Opening (<20) phase averages 100+50=150/2=75.
    assert feats["phase_strengths"]["opening"] == 75.0
    # Middlegame (20..40) has one loss of 200.
    assert feats["phase_strengths"]["middlegame"] == 200.0
    # Endgame empty → None.
    assert feats["phase_strengths"]["endgame"] is None


def test_merge_features_weighted_running_average():
    prev = {
        "aggression_index": 0.2,
        "blunder_rate": 1.0,
        "games_sampled": 4,
        "typical_opening_name": "Sicilian Defense",
        "typical_opening_eco": "B20",
        "typical_opening_group": "king_pawn_open",
        "phase_strengths": {"opening": 100.0, "middlegame": 120.0, "endgame": 80.0},
        "preferred_trades": {"N": 2, "B": 1},
        "total_player_moves": 100,
    }
    new = {
        "aggression_index": 0.5,
        "blunder_rate": 2.0,
        "games_sampled": 1,
        "typical_opening_name": "French Defense",
        "typical_opening_eco": "C00",
        "typical_opening_group": "king_pawn_open",
        "time_trouble_blunders": 2,
        "phase_strengths": {"opening": 50.0, "middlegame": 100.0, "endgame": None},
        "preferred_trades": {"N": 3, "Q": 1},
        "total_player_moves": 30,
        "abandoned_last": False,
    }
    merged = merge_features(prev, new, prior_games=4)
    # Weight = 1/5 for new.
    assert abs(merged["aggression_index"] - (0.2 * 0.8 + 0.5 * 0.2)) < 1e-6
    # Opening updates to most recent.
    assert merged["typical_opening_name"] == "French Defense"
    # Preferred trades accumulate.
    assert merged["preferred_trades"]["N"] == 5  # 2 + 3
    assert merged["preferred_trades"]["Q"] == 1
    assert merged["games_sampled"] == 5
    assert merged["total_player_moves"] == 130


def test_merge_features_first_match():
    """No prior profile → merged == new."""
    new = {"aggression_index": 0.4, "games_sampled": 1}
    merged = merge_features(None, new, prior_games=0)
    assert merged["aggression_index"] == 0.4
