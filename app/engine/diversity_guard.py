"""Anti-shuffle filter: skips engine candidates that cycle a piece back to a recently vacated square."""

from __future__ import annotations

import chess

from app.engine.base import ConsideredMove


def _is_shuffle(move: chess.Move, board: chess.Board, recent: list[chess.Move]) -> bool:
    """True if `move` returns a piece to a square it recently left."""
    piece = board.piece_at(move.from_square)
    if piece is None:
        return False
    for r in recent:
        if move.to_square != r.from_square:
            continue
        # Candidate goes to where a piece recently came from.
        # Guard against false positives when a different piece type targets the same square.
        recent_mover = board.piece_at(r.to_square)
        if recent_mover is not None and recent_mover.piece_type == piece.piece_type:
            return True
    return False


def filter_shuffle_moves(
    candidates: list[ConsideredMove],
    own_recent_moves: list[chess.Move],
    board: chess.Board,
    lookback_plies: int = 6,
) -> ConsideredMove:
    """Return the first non-shuffle candidate; fall back to candidates[0] if all are flagged."""
    if not candidates:
        raise ValueError("candidates must not be empty")
    recent = own_recent_moves[:lookback_plies]
    for candidate in candidates:
        move = chess.Move.from_uci(candidate.uci)
        if not _is_shuffle(move, board, recent):
            return candidate
    return candidates[0]
