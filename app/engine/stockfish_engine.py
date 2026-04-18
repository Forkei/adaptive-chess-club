"""Stockfish engine wrapper.

Uses the `stockfish` pip package which drives the Stockfish UCI binary.
The binary comes from `apt-get install stockfish` in Docker, or from the
`STOCKFISH_PATH` env var for local dev.
"""

from __future__ import annotations

import logging
import os
import shutil
import threading
import time

import chess

from app.engine.base import ChessEngine, ConsideredMove, EngineConfig, EngineUnavailable, MoveResult

logger = logging.getLogger(__name__)


def _find_stockfish_binary() -> str | None:
    explicit = os.environ.get("STOCKFISH_PATH")
    if explicit and os.path.exists(explicit):
        return explicit
    on_path = shutil.which("stockfish")
    if on_path:
        return on_path
    # Windows common names — Docker prod won't hit these but dev might.
    for candidate in ("stockfish.exe", "stockfish-windows-x86-64.exe"):
        hit = shutil.which(candidate)
        if hit:
            return hit
    return None


class StockfishEngine(ChessEngine):
    name = "stockfish"

    @classmethod
    def is_available(cls) -> bool:
        return _find_stockfish_binary() is not None

    def __init__(self, path: str | None = None):
        binary = path or _find_stockfish_binary()
        if binary is None:
            raise EngineUnavailable("No Stockfish binary found")
        self._path = binary
        self._lock = threading.Lock()
        try:
            from stockfish import Stockfish  # type: ignore
        except ImportError as exc:
            raise EngineUnavailable(f"stockfish python package missing: {exc}") from exc

        # One long-lived subprocess. Stockfish in this wrapper isn't thread-safe,
        # so we guard every call with `_lock`.
        self._sf = Stockfish(path=binary, depth=15, parameters={"Threads": 1, "Hash": 32})
        logger.info("Initialized Stockfish at %s", binary)

    @staticmethod
    def _elo_to_skill_level(elo: int) -> int:
        # Stockfish UCI_Elo is usable, but `Skill Level` (0..20) gives a smoother
        # feel at extremes. We set both.
        if elo >= 2400:
            return 20
        if elo <= 800:
            return 0
        return max(0, min(20, round((elo - 800) / 80)))

    def _apply_config(self, config: EngineConfig) -> None:
        target = max(1320, min(3190, config.target_elo))  # Stockfish UCI_Elo valid range
        self._sf.update_engine_parameters(
            {
                "UCI_LimitStrength": "true",
                "UCI_Elo": target,
                "Skill Level": config.skill_level
                if config.skill_level is not None
                else self._elo_to_skill_level(target),
                "Contempt": config.contempt if config.contempt is not None else 0,
            }
        )
        if config.depth:
            self._sf.set_depth(config.depth)

    def get_move(self, board: chess.Board, config: EngineConfig) -> MoveResult:
        started = time.perf_counter()
        with self._lock:
            self._apply_config(config)
            self._sf.set_fen_position(board.fen())

            time_budget_ms = max(50, int(config.time_budget_seconds * 1000))
            best_uci = self._sf.get_best_move_time(time_budget_ms)
            if best_uci is None:
                # Stalemate / game over; caller should check before asking.
                raise RuntimeError("Stockfish returned no move — position may be terminal")

            evaluation = self._sf.get_evaluation()  # {"type": "cp"|"mate", "value": int}
            eval_cp: int | None = None
            if evaluation and evaluation.get("type") == "cp":
                eval_cp = int(evaluation.get("value", 0))
            elif evaluation and evaluation.get("type") == "mate":
                mate_in = int(evaluation.get("value", 0))
                eval_cp = 100000 if mate_in > 0 else -100000

            top_moves_raw = []
            try:
                top_moves_raw = self._sf.get_top_moves(3) or []
            except Exception:
                pass

        best_move = chess.Move.from_uci(best_uci)
        san = board.san(best_move)

        considered: list[ConsideredMove] = []
        for tm in top_moves_raw[:3]:
            uci = tm.get("Move")
            if not uci:
                continue
            try:
                alt_san = board.san(chess.Move.from_uci(uci))
            except Exception:
                alt_san = None
            centipawn = tm.get("Centipawn")
            mate = tm.get("Mate")
            alt_cp: int | None = None
            if centipawn is not None:
                alt_cp = int(centipawn)
            elif mate is not None:
                alt_cp = 100000 if mate > 0 else -100000
            considered.append(
                ConsideredMove(uci=uci, san=alt_san, eval_cp=alt_cp, probability=None)
            )

        depth = None
        try:
            depth = int(self._sf.get_parameters().get("Depth", 0)) or None
        except Exception:
            pass

        return MoveResult(
            move=best_uci,
            san=san,
            eval_cp=eval_cp,
            considered_moves=considered,
            time_taken_ms=int((time.perf_counter() - started) * 1000),
            engine_name="stockfish",
            thinking_depth=depth,
        )

    def close(self) -> None:
        try:
            # The stockfish package's __del__ handles subprocess teardown, but
            # be explicit when we know we're done.
            del self._sf
        except Exception:
            pass
