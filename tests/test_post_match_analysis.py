"""Post-match engine analysis + critical moment detection.

Uses a fake engine that returns canned evals so we can pin the math
without invoking Stockfish.
"""

from __future__ import annotations

from typing import Any

import chess
import pytest

from app.engine.base import EngineConfig, MoveResult
from app.post_match.analysis import (
    analyze_match_moves,
    identify_critical_moments,
)


class _FakeEngine:
    """Returns deterministic `MoveResult`s based on a script.

    `script` is a list of (best_uci, eval_cp_white_pov) pairs, consumed
    in order. Extra calls reuse the last entry.
    """

    name = "stockfish"

    def __init__(self, script: list[tuple[str, int]]):
        self._script = script
        self._i = 0

    def get_move(self, board: chess.Board, config: EngineConfig) -> MoveResult:
        entry = self._script[min(self._i, len(self._script) - 1)]
        self._i += 1
        best_uci, eval_cp = entry
        # Pick any legal move as a fallback if script's move isn't legal here.
        try:
            move = chess.Move.from_uci(best_uci)
            if move not in board.legal_moves:
                move = next(iter(board.legal_moves))
                best_uci = move.uci()
        except Exception:
            move = next(iter(board.legal_moves))
            best_uci = move.uci()
        san = board.san(move)
        return MoveResult(
            move=best_uci,
            san=san,
            eval_cp=eval_cp,
            considered_moves=[],
            time_taken_ms=1,
            engine_name="stockfish",
            thinking_depth=10,
        )


def test_analyze_empty_moves():
    out = analyze_match_moves([], engine=_FakeEngine([("e2e4", 10)]))
    assert out["status"] == "completed"
    assert out["moves"] == []


def test_analyze_single_clean_move():
    # Script: 3 calls (start-eval, best-from-start, eval-after-played)
    engine = _FakeEngine([("e2e4", 20), ("e2e4", 25), ("e7e5", 0)])
    out = analyze_match_moves(
        moves=[{"move_number": 1, "side": "white", "uci": "e2e4", "san": "e4"}],
        engine=engine,
    )
    assert out["status"] == "completed"
    moves = out["moves"]
    assert len(moves) == 1
    m = moves[0]
    assert m["san"] == "e4"
    assert m["is_blunder"] is False
    # eval_loss should be small (played move was best, no loss).
    assert m["eval_loss_cp"] < 50


def test_analyze_detects_blunder():
    """Engine sees best +400cp, player plays something that leaves eval at -100cp.

    `_analyze_position` returns the script eval for each call. We need:
    - start-eval (arbitrary)
    - best-from-start (from white-POV, say +400cp — very good for white)
    - eval-after-played (from white-POV, -100cp — now black is better)
    - So the mover (white) lost 400 - (-100) = 500cp, clamped to 300.
    """
    engine = _FakeEngine(
        [
            ("e2e4", 0),    # initial
            ("d1h5", 400),  # best from pre-move — Qh5, +400cp
            ("e2e4", -100),  # after the played move — black now at +100cp
        ]
    )
    out = analyze_match_moves(
        moves=[{"move_number": 1, "side": "white", "uci": "e2e4", "san": "e4"}],
        engine=engine,
    )
    m = out["moves"][0]
    assert m["is_blunder"] is True
    assert m["eval_loss_cp"] == 300  # clamped


def test_critical_moments_filters_and_ranks():
    analysis = {
        "moves": [
            {"move_number": 1, "side": "white", "san": "e4", "eval_loss_cp": 10,
             "eval_before_cp": 20, "eval_after_cp": 15, "best_move_uci": "e2e4"},
            {"move_number": 8, "side": "black", "san": "Bxa2??", "eval_loss_cp": 300,
             "eval_before_cp": 0, "eval_after_cp": -400, "best_move_uci": "a7a6"},
            {"move_number": 15, "side": "white", "san": "Nf3", "eval_loss_cp": 150,
             "eval_before_cp": -10, "eval_after_cp": -180, "best_move_uci": "c2c3"},
        ]
    }
    out = identify_critical_moments(analysis, max_moments=3)
    assert len(out) == 2  # the 10cp move is filtered
    assert out[0]["san"] == "Bxa2??"
    assert "blunder" in out[0]["label"].lower()


def test_skipped_engine_returns_skipped_status(monkeypatch):
    """If stockfish isn't in available_engines, analysis skips gracefully."""
    monkeypatch.setattr("app.post_match.analysis.available_engines", lambda: ["mock"])
    out = analyze_match_moves(
        moves=[{"move_number": 1, "side": "white", "uci": "e2e4", "san": "e4"}]
    )
    assert out["status"] == "skipped"
    assert out["moves"] == []
