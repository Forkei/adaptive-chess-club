"""Deterministic engine used in tests and in dev when real engines are missing.

Phase 2a shipped a naive "pick alphabetically-first UCI move" strategy which
produced pathological shuffling in the opening (e.g. Nf3-Ng1-Nf3-Ng1 for
six plies). Phase 4.0c fixes this by filtering out:

  1. Direct reversals of the agent's last N own moves (avoids 2-move cycles).
  2. Moves that would produce a positional repetition (3-fold).

If the remaining legal set is empty the engine falls back to the raw
alphabetical pick so it always returns *some* legal move.
"""

from __future__ import annotations

import time

import chess

from app.engine.base import ChessEngine, ConsideredMove, EngineConfig, MoveResult

# How far back in own-move history to look for reversals. 3 = last three of
# our own moves; any move that reverses one of those is filtered. This
# handles A→B→A→B cycles *and* A→B→C→A sortie patterns.
REVERSAL_LOOKBACK = 3


class MockEngine(ChessEngine):
    name = "mock"

    @classmethod
    def is_available(cls) -> bool:  # always available
        return True

    def get_move(self, board: chess.Board, config: EngineConfig) -> MoveResult:
        started = time.perf_counter()
        legal = sorted(board.legal_moves, key=lambda m: m.uci())
        if not legal:
            raise ValueError("No legal moves — game is already over")

        chosen = _pick_non_shuffle(board, legal)
        san = board.san(chosen)

        considered = []
        for alt in legal[:3]:
            considered.append(
                ConsideredMove(
                    uci=alt.uci(),
                    san=board.san(alt),
                    eval_cp=None,
                    probability=1.0 / len(legal),
                )
            )

        return MoveResult(
            move=chosen.uci(),
            san=san,
            eval_cp=None,
            considered_moves=considered,
            time_taken_ms=int((time.perf_counter() - started) * 1000),
            engine_name="mock",
            thinking_depth=None,
        )


def _recent_own_moves(board: chess.Board, depth: int) -> list[chess.Move]:
    """Return up to `depth` of this side's most recent own moves, newest first.

    Stack layout: opponent-move, own-move, opponent-move, own-move, ...
    Own moves sit at indices -2, -4, -6, ... when it's our turn (since the
    most recent pushed move was the opponent's).
    """
    stack = board.move_stack
    out: list[chess.Move] = []
    for i in range(2, 2 * depth + 1, 2):
        if len(stack) >= i:
            out.append(stack[-i])
    return out


def _is_reversal(move: chess.Move, prior: chess.Move) -> bool:
    """True iff `move` directly reverses `prior` (same piece, swap squares).

    Promotions are never treated as reversals (a promoted move can't undo).
    """
    if move.promotion is not None or prior.promotion is not None:
        return False
    return move.from_square == prior.to_square and move.to_square == prior.from_square


def _would_repeat_position(board: chess.Board, move: chess.Move) -> bool:
    """Would pushing `move` create a 3-fold repetition?"""
    board.push(move)
    try:
        return board.is_repetition(3)
    finally:
        board.pop()


def _pick_non_shuffle(board: chess.Board, legal: list[chess.Move]) -> chess.Move:
    """Choose the alphabetically-first legal move that isn't a shuffle
    reversal of any of our recent own moves and doesn't force repetition.
    Falls back to `legal[0]` if every option trips the filters.
    """
    recents = _recent_own_moves(board, REVERSAL_LOOKBACK)
    for move in legal:
        if any(_is_reversal(move, r) for r in recents):
            continue
        if _would_repeat_position(board, move):
            continue
        return move
    return legal[0]
