"""Deterministic engine used in tests and in dev when real engines are missing.

Picks the move that sorts first by UCI string from the legal move list.
Stable across runs, no dependencies beyond python-chess.
"""

from __future__ import annotations

import time

import chess

from app.engine.base import ChessEngine, ConsideredMove, EngineConfig, MoveResult


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

        chosen = legal[0]
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
