"""Opening classifier (app/characters/openings.classify_opening)."""

from __future__ import annotations

from app.characters.openings import classify_opening


def test_empty_is_unknown():
    out = classify_opening([])
    assert out == {"eco": "unknown", "name": "unknown", "group": "unknown"}


def test_kings_pawn_game():
    out = classify_opening(["e4", "e5"])
    assert out["group"] == "king_pawn_open"
    assert out["name"] == "King's Pawn Game"


def test_sicilian():
    out = classify_opening(["e4", "c5"])
    assert out["name"] == "Sicilian Defense"
    assert out["group"] == "king_pawn_open"


def test_ruy_lopez_prefers_longer_prefix():
    """e4 e5 matches King's Pawn, but Nf3 Nc6 Bb5 extends to Ruy Lopez."""
    out = classify_opening(["e4", "e5", "Nf3", "Nc6", "Bb5", "a6"])
    assert out["name"] == "Ruy Lopez"
    assert out["eco"] == "C60"


def test_kings_gambit_beats_kings_pawn_game():
    out = classify_opening(["e4", "e5", "f4"])
    assert out["name"] == "King's Gambit"
    assert out["group"] == "gambit"


def test_queens_gambit():
    out = classify_opening(["d4", "d5", "c4"])
    assert out["name"] == "Queen's Gambit"


def test_kings_indian_defense():
    out = classify_opening(["d4", "Nf6", "c4", "g6", "Nc3", "Bg7"])
    assert out["name"] == "King's Indian Defense"
    assert out["group"] == "indian"


def test_english_opening():
    out = classify_opening(["c4"])
    assert out["name"] == "English Opening"
    assert out["group"] == "flank"


def test_unknown_when_no_prefix_matches():
    # First move not in the table (Nc3 as opener is unusual).
    out = classify_opening(["Nc3"])
    assert out == {"eco": "unknown", "name": "unknown", "group": "unknown"}
