"""Opponent-style feature extraction.

Pure function from (moves, analysis) → dict of features, suitable for
merging into OpponentProfile.style_features with a running average
weighted by games_played.

Computed features:
- `aggression_index`: fraction of player moves that are captures / checks / sacrifices
- `typical_opening_eco`: family-level ECO code via openings.classify_opening
- `typical_opening_name`: human-readable opening name
- `blunder_rate`: blunders per 40 moves (eval_loss >= 200cp)
- `time_trouble_blunders`: count of blunders in the last 10 moves of the game
- `preferred_trades`: piece-exchange pattern counts
- `phase_strengths`: average eval_loss per phase (opening / middlegame / endgame)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import chess

from app.characters.openings import classify_opening

logger = logging.getLogger(__name__)

BLUNDER_THRESHOLD_CP = 200
TIME_TROUBLE_WINDOW = 10


def _phase_of(move_number: int, is_endgame_fen_based: bool = False) -> str:
    """Coarse phase by move number. Analysis layer has a better phase
    via material, but we use move_number here for simplicity + parity
    with the director._phase_from_board heuristic."""
    if move_number < 10:
        return "opening"
    if is_endgame_fen_based:
        return "endgame"
    return "middlegame"


def _is_capture_or_check(san: str) -> bool:
    """Quick SAN-based classifier. `x` = capture, `+`/`#` = check/mate."""
    if not san:
        return False
    return ("x" in san) or ("+" in san) or ("#" in san)


def _is_promotion(san: str) -> bool:
    return "=" in (san or "")


def extract_features(
    *,
    moves: list[dict[str, Any]],
    player_color: str,
    analysis: dict[str, Any] | None,
    abandoned: bool,
) -> dict[str, Any]:
    """Compute style features for the PLAYER (the human), not the character.

    `moves` is the list of Move rows (dicts). `player_color` is
    "white"/"black". `analysis` is the engine_analysis dict from
    post_match.analysis (may be empty/skipped — we degrade).
    """
    player_moves = [m for m in moves if _move_side_str(m) == player_color]
    total_moves = len(player_moves)

    if total_moves == 0:
        return _zero_features(abandoned=abandoned)

    # Opening classification from the first 10 half-moves of the actual game.
    opening_san_seq = [m.get("san") for m in moves[:10] if m.get("san")]
    opening = classify_opening(opening_san_seq)

    # Aggression: captures + checks + promotions.
    aggressive = sum(
        1
        for m in player_moves
        if _is_capture_or_check(m.get("san") or "") or _is_promotion(m.get("san") or "")
    )
    aggression_index = round(aggressive / total_moves, 3)

    # Blunders, time-trouble blunders, and phase strengths from analysis.
    analysis_moves = (analysis or {}).get("moves") or []
    # Filter to player's moves only.
    by_mnum: dict[int, dict[str, Any]] = {}
    for am in analysis_moves:
        if am.get("side") == player_color:
            by_mnum[am.get("move_number")] = am

    blunders = sum(1 for am in by_mnum.values() if am.get("is_blunder"))
    blunder_rate = round((blunders / max(1, total_moves)) * 40, 2)

    # Time-trouble: blunders in the LAST 10 half-moves overall, restricted to player's side.
    last_window_nums = {m.get("move_number") for m in moves[-TIME_TROUBLE_WINDOW:]}
    tt_blunders = sum(
        1
        for am in by_mnum.values()
        if am.get("is_blunder") and am.get("move_number") in last_window_nums
    )

    # Phase strengths: mean eval_loss per phase for player's moves.
    phase_losses: dict[str, list[int]] = {"opening": [], "middlegame": [], "endgame": []}
    total_plies = len(moves)
    for am in by_mnum.values():
        mnum = am.get("move_number") or 0
        # Coarse phase: opening if move_number < 20, endgame if move_number > 40, else middle.
        if mnum < 20:
            phase = "opening"
        elif mnum > 40:
            phase = "endgame"
        else:
            phase = "middlegame"
        phase_losses[phase].append(int(am.get("eval_loss_cp") or 0))
    phase_strengths = {
        phase: (
            round(sum(losses) / len(losses), 1)
            if losses
            else None
        )
        for phase, losses in phase_losses.items()
    }

    # Preferred trades: count captures of each piece type (SAN-based heuristic).
    preferred_trades: dict[str, int] = {}
    for m in player_moves:
        san = m.get("san") or ""
        if "x" in san and san:
            piece = san[0] if san[0].isupper() else "P"  # uppercase = Q/R/B/N/K; pawn captures start with file letter
            preferred_trades[piece] = preferred_trades.get(piece, 0) + 1

    return {
        "games_sampled": 1,
        "aggression_index": aggression_index,
        "typical_opening_eco": opening["eco"],
        "typical_opening_name": opening["name"],
        "typical_opening_group": opening["group"],
        "blunder_rate": blunder_rate,
        "time_trouble_blunders": tt_blunders,
        "preferred_trades": preferred_trades,
        "phase_strengths": phase_strengths,
        "total_player_moves": total_moves,
        "abandoned_last": abandoned,
    }


def merge_features(
    previous: dict[str, Any] | None,
    new: dict[str, Any],
    *,
    prior_games: int,
) -> dict[str, Any]:
    """Merge `new` into `previous` as a weighted running average.

    `prior_games` is the count BEFORE this match, used as the weight on
    `previous`. So after one more match, the new values contribute
    `1/(prior_games+1)` of the combined average.
    """
    if not previous:
        return dict(new)
    merged: dict[str, Any] = dict(previous)
    n = max(prior_games, 0)
    weight_new = 1.0 / (n + 1.0) if n >= 0 else 1.0

    def _avg(a: float | None, b: float | None) -> float | None:
        if a is None and b is None:
            return None
        if a is None:
            return b
        if b is None:
            return a
        return round(a * (1 - weight_new) + b * weight_new, 3)

    for key in ("aggression_index", "blunder_rate"):
        merged[key] = _avg(previous.get(key), new.get(key))

    # Integer-ish features: sum (they're counts/this-match values).
    merged["time_trouble_blunders_last"] = new.get("time_trouble_blunders", 0)

    # Opening: take most recent (it drifts over time naturally).
    for key in ("typical_opening_eco", "typical_opening_name", "typical_opening_group"):
        merged[key] = new.get(key) or previous.get(key) or "unknown"

    # Phase strengths: merge dict-of-floats.
    prev_ps = previous.get("phase_strengths") or {}
    new_ps = new.get("phase_strengths") or {}
    merged_ps = {}
    for phase in ("opening", "middlegame", "endgame"):
        merged_ps[phase] = _avg(prev_ps.get(phase), new_ps.get(phase))
    merged["phase_strengths"] = merged_ps

    # Preferred trades: accumulate counts.
    prev_pt = previous.get("preferred_trades") or {}
    new_pt = new.get("preferred_trades") or {}
    combined_pt = dict(prev_pt)
    for k, v in new_pt.items():
        combined_pt[k] = combined_pt.get(k, 0) + v
    merged["preferred_trades"] = combined_pt

    merged["games_sampled"] = (previous.get("games_sampled") or 0) + 1
    merged["abandoned_last"] = new.get("abandoned_last", False)
    merged["total_player_moves"] = (previous.get("total_player_moves") or 0) + new.get(
        "total_player_moves", 0
    )
    return merged


def _zero_features(*, abandoned: bool) -> dict[str, Any]:
    return {
        "games_sampled": 1,
        "aggression_index": 0.0,
        "typical_opening_eco": "unknown",
        "typical_opening_name": "unknown",
        "typical_opening_group": "unknown",
        "blunder_rate": 0.0,
        "time_trouble_blunders": 0,
        "preferred_trades": {},
        "phase_strengths": {"opening": None, "middlegame": None, "endgame": None},
        "total_player_moves": 0,
        "abandoned_last": abandoned,
    }


def _move_side_str(move: dict[str, Any]) -> str:
    side = move.get("side")
    if hasattr(side, "value"):
        return side.value  # Color enum
    return str(side) if side else ""
