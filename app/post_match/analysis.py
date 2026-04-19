"""Engine analysis pass — replay every position through Stockfish.

For each move in the match we produce:
- `eval_before_cp`: eval of the position before the move (from the side-to-move's POV)
- `eval_after_cp`: eval of the position after the move
- `best_move_uci`: Stockfish's best move from the pre-move position
- `best_eval_cp`: Stockfish's eval after the best move
- `eval_loss_cp`: clamp(max(0, best_eval_cp - eval_after_cp_for_mover), 0, 300)
  Signed from the mover's perspective — how much a blunder was this move.

Critical moments = moves with eval_loss_cp > 100 (blunders) OR where the
position eval flipped sign by >= 150cp.

We budget 0.3s/move (predictable on Docker) rather than fixed depth —
the latter produced wild variance on slow CI containers during 2a.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import chess

from app.engine import available_engines, get_engine
from app.engine.base import EngineConfig

logger = logging.getLogger(__name__)

ANALYSIS_TIME_PER_MOVE_S = 0.3
ANALYSIS_ENGINE_ELO = 2800      # ceiling — we want the strongest reads
BLUNDER_CP_THRESHOLD = 200
CRITICAL_SWING_CP = 150


@dataclass
class MoveAnalysis:
    move_number: int
    side: str           # "white" or "black"
    uci: str
    san: str
    eval_before_cp: int | None
    eval_after_cp: int | None
    best_move_uci: str | None
    best_eval_cp: int | None
    eval_loss_cp: int   # clamp(0, 300) — relative to what was possible
    is_blunder: bool


def _eval_from_mover_pov(eval_cp: int | None, side_to_move: chess.Color) -> int | None:
    """Stockfish gives eval from white's POV. Convert to the mover's POV."""
    if eval_cp is None:
        return None
    return eval_cp if side_to_move == chess.WHITE else -eval_cp


def _analyze_position(
    engine,
    board: chess.Board,
    time_budget_s: float,
) -> tuple[str | None, int | None]:
    """Return (best_uci_from_here, eval_cp_from_white_pov) for this position.

    Uses the provided engine at max strength. Wraps config construction so
    tests can inject a stub engine.
    """
    cfg = EngineConfig(
        target_elo=ANALYSIS_ENGINE_ELO,
        time_budget_seconds=time_budget_s,
        engine_name="stockfish",
    )
    try:
        res = engine.get_move(board, cfg)
    except Exception as exc:
        logger.warning("analysis engine call failed at %s: %s", board.fen(), exc)
        return None, None
    best_uci = res.move
    eval_cp_white = res.eval_cp
    # Stockfish returns eval from the side-to-move POV (per python package).
    # The wrapper in stockfish_engine.py already normalizes via `get_evaluation`
    # which returns from white-to-move POV. Keep whatever the wrapper gives us.
    return best_uci, eval_cp_white


def analyze_match_moves(
    moves: list[dict[str, Any]],
    *,
    initial_fen: str = chess.STARTING_FEN,
    engine=None,
    time_per_move_s: float = ANALYSIS_TIME_PER_MOVE_S,
) -> dict[str, Any]:
    """Run Stockfish over every position.

    `moves` is a list of dicts with at least `{uci, san, side}` keys
    (matching `Move` row fields). Returns a dict with a per-move list
    and aggregate stats.

    Gracefully degrades if the engine can't be obtained: returns an
    empty analysis with `status="skipped"` so downstream steps can use
    lighter signals.
    """
    if engine is None:
        available = set(available_engines())
        if "stockfish" not in available:
            logger.warning("Stockfish unavailable — skipping engine analysis")
            return {"status": "skipped", "reason": "stockfish_unavailable", "moves": []}
        engine = get_engine("stockfish")

    board = chess.Board(initial_fen)
    per_move: list[MoveAnalysis] = []

    # Evaluate the starting position once (eval_before for move #1).
    _, prev_eval_white = _analyze_position(engine, board, time_per_move_s)

    started = time.perf_counter()
    for mv in moves:
        uci = mv.get("uci")
        san = mv.get("san") or ""
        side = mv.get("side")
        if uci is None:
            continue
        side_to_move = board.turn
        eval_before_mover = _eval_from_mover_pov(prev_eval_white, side_to_move)

        # Best move from THIS position (before the actual move is played).
        best_uci, best_eval_white = _analyze_position(engine, board, time_per_move_s)
        best_eval_mover = _eval_from_mover_pov(best_eval_white, side_to_move)

        # Apply the actual move.
        try:
            move = chess.Move.from_uci(uci)
            if move not in board.legal_moves:
                logger.warning("analysis: illegal move %s at position %s — stopping", uci, board.fen())
                break
            board.push(move)
        except Exception as exc:
            logger.warning("analysis: move push failed (%s): %s", uci, exc)
            break

        # Eval after the actual move — still from white's POV by convention.
        _, after_eval_white = _analyze_position(engine, board, time_per_move_s)
        after_eval_mover = _eval_from_mover_pov(after_eval_white, side_to_move)

        # eval_loss = how much worse the played eval is compared to the best move,
        # from the mover's perspective. Clamp to [0, 300].
        if best_eval_mover is None or after_eval_mover is None:
            eval_loss = 0
        else:
            raw_loss = best_eval_mover - after_eval_mover
            eval_loss = max(0, min(300, raw_loss))

        per_move.append(
            MoveAnalysis(
                move_number=int(mv.get("move_number") or 0),
                side=side if isinstance(side, str) else getattr(side, "value", "white"),
                uci=uci,
                san=san,
                eval_before_cp=eval_before_mover,
                eval_after_cp=after_eval_mover,
                best_move_uci=best_uci,
                best_eval_cp=best_eval_mover,
                eval_loss_cp=int(eval_loss),
                is_blunder=eval_loss >= BLUNDER_CP_THRESHOLD,
            )
        )
        prev_eval_white = after_eval_white

    elapsed = time.perf_counter() - started
    return {
        "status": "completed",
        "elapsed_s": round(elapsed, 2),
        "moves": [
            {
                "move_number": m.move_number,
                "side": m.side,
                "uci": m.uci,
                "san": m.san,
                "eval_before_cp": m.eval_before_cp,
                "eval_after_cp": m.eval_after_cp,
                "best_move_uci": m.best_move_uci,
                "best_eval_cp": m.best_eval_cp,
                "eval_loss_cp": m.eval_loss_cp,
                "is_blunder": m.is_blunder,
            }
            for m in per_move
        ],
    }


def identify_critical_moments(analysis: dict[str, Any], max_moments: int = 6) -> list[dict[str, Any]]:
    """Pick the moves with the largest eval swings.

    Returns up to `max_moments` items, each a dict suitable for UI
    display and for feeding into the memory-generation prompt.
    """
    moves = analysis.get("moves") or []
    scored: list[tuple[int, dict[str, Any]]] = []
    for m in moves:
        loss = m.get("eval_loss_cp") or 0
        before = m.get("eval_before_cp") or 0
        after = m.get("eval_after_cp") or 0
        swing = abs((after or 0) - (before or 0))
        # Only consider genuinely notable moments: blunder-level loss OR
        # a meaningful eval swing. Filters out routine inaccuracies.
        if loss < 50 and swing < CRITICAL_SWING_CP:
            continue
        score = loss * 2 + max(0, swing - CRITICAL_SWING_CP)
        if score <= 0:
            continue
        scored.append((score, m))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [
        {
            "move_number": m["move_number"],
            "side": m["side"],
            "san": m["san"],
            "eval_loss_cp": m["eval_loss_cp"],
            "best_move_uci": m["best_move_uci"],
            "label": _moment_label(m),
        }
        for _, m in scored[:max_moments]
    ]


def _moment_label(m: dict[str, Any]) -> str:
    loss = m.get("eval_loss_cp") or 0
    if loss >= 300:
        return f"{m['side']} blundered on move {m['move_number']} ({m['san']})"
    if loss >= 200:
        return f"{m['side']} made a serious mistake on move {m['move_number']} ({m['san']})"
    if loss >= 100:
        return f"{m['side']}'s move {m['move_number']} ({m['san']}) was inaccurate"
    return f"sharp moment on move {m['move_number']} ({m['san']})"
