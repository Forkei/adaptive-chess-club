"""Maia-2 engine wrapper.

Maia-2 is the CSSLab NeurIPS 2024 skill-aware model. It takes an Elo
bucket in [1100, 1900] and returns a distribution over legal moves.

We lazy-import the `maia2` package so `app.engine` can be imported in
environments where torch / maia2 aren't installed (tests, the
character-only pages, etc.).
"""

from __future__ import annotations

import importlib.util
import logging
import os
import threading
import time

import chess

from app.engine.base import ChessEngine, ConsideredMove, EngineConfig, EngineUnavailable, MoveResult

logger = logging.getLogger(__name__)

# Maia-2 is trained on these Elo buckets, inclusive.
MAIA_MIN_ELO = 1100
MAIA_MAX_ELO = 1900


def _clamp_to_maia_bucket(elo: int) -> int:
    # Maia-2 buckets are in 100-Elo steps from 1100 to 1900.
    clamped = max(MAIA_MIN_ELO, min(MAIA_MAX_ELO, elo))
    # Round to nearest 100.
    return int(round(clamped / 100.0)) * 100


class Maia2Engine(ChessEngine):
    name = "maia2"

    _model = None
    _prepared = None
    _load_lock = threading.Lock()

    @classmethod
    def is_available(cls) -> bool:
        if importlib.util.find_spec("maia2") is None:
            return False
        # Don't attempt to load weights during probe — too slow. A real load
        # happens on first `get_move` call.
        return True

    def __init__(self, model_type: str = "rapid", device: str = "cpu"):
        self.model_type = model_type
        self.device = device
        self._inf_lock = threading.Lock()

    def _ensure_loaded(self) -> None:
        if Maia2Engine._model is not None:
            return
        with Maia2Engine._load_lock:
            if Maia2Engine._model is not None:
                return
            cache_dir = os.environ.get("MAIA2_CACHE_DIR")
            if cache_dir:
                os.environ.setdefault("HF_HOME", cache_dir)
            try:
                from maia2 import inference, model as maia_model  # type: ignore
            except ImportError as exc:
                raise EngineUnavailable(f"maia2 package import failed: {exc}") from exc

            logger.info("Loading Maia-2 %s model on %s", self.model_type, self.device)
            try:
                Maia2Engine._model = maia_model.from_pretrained(
                    type=self.model_type, device=self.device
                )
                Maia2Engine._prepared = inference.prepare()
            except Exception as exc:
                raise EngineUnavailable(f"Maia-2 model load failed: {exc}") from exc

    def get_move(self, board: chess.Board, config: EngineConfig) -> MoveResult:
        started = time.perf_counter()
        self._ensure_loaded()

        from maia2 import inference  # type: ignore

        elo_bucket = config.maia_elo_bucket or _clamp_to_maia_bucket(config.target_elo)

        with self._inf_lock:
            move_probs, _win_prob, _value = inference.inference_each(
                Maia2Engine._model,
                Maia2Engine._prepared,
                board.fen(),
                elo_self=elo_bucket,
                elo_oppo=elo_bucket,
            )

        if not move_probs:
            raise RuntimeError("Maia-2 returned no move probabilities")

        # Maia-2 returns a dict {uci: prob}. Highest prob wins.
        legal_ucis = {m.uci() for m in board.legal_moves}
        filtered = {uci: p for uci, p in move_probs.items() if uci in legal_ucis}
        if not filtered:
            # Model returned only illegal moves (shouldn't happen with prepared data,
            # but defend against it). Fall back to highest-prob legal move by prefix.
            filtered = {m.uci(): 1.0 / max(1, len(legal_ucis)) for m in board.legal_moves}

        ranked = sorted(filtered.items(), key=lambda kv: kv[1], reverse=True)
        best_uci, _best_prob = ranked[0]
        best_move = chess.Move.from_uci(best_uci)
        san = board.san(best_move)

        considered: list[ConsideredMove] = []
        for uci, prob in ranked[:3]:
            try:
                alt_move = chess.Move.from_uci(uci)
                alt_san = board.san(alt_move)
            except Exception:
                alt_san = None
            considered.append(
                ConsideredMove(
                    uci=uci,
                    san=alt_san,
                    eval_cp=None,
                    probability=float(prob),
                )
            )

        return MoveResult(
            move=best_uci,
            san=san,
            eval_cp=None,
            considered_moves=considered,
            time_taken_ms=int((time.perf_counter() - started) * 1000),
            engine_name="maia2",
            thinking_depth=None,
            raw={"elo_bucket": elo_bucket},
        )
