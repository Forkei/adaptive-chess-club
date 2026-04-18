"""Single-source-of-truth for engine construction.

`get_engine(name)` returns a cached instance. Availability is probed
lazily the first time an engine is requested so we never import torch /
stockfish at module load.
"""

from __future__ import annotations

import logging
import threading
from typing import Callable

from app.engine.base import ChessEngine, EngineName, EngineUnavailable

logger = logging.getLogger(__name__)

_Factory = Callable[[], ChessEngine]

_FACTORIES: dict[str, _Factory] = {}
_INSTANCES: dict[str, ChessEngine] = {}
_LOCK = threading.Lock()


def register_engine(name: EngineName, factory: _Factory) -> None:
    _FACTORIES[name] = factory


def _build_default_factories() -> None:
    """Lazy-defined so importing app.engine.registry doesn't force a torch import."""

    if _FACTORIES:
        return

    def _mock() -> ChessEngine:
        from app.engine.mock_engine import MockEngine

        return MockEngine()

    def _maia2() -> ChessEngine:
        from app.engine.maia2_engine import Maia2Engine

        if not Maia2Engine.is_available():
            raise EngineUnavailable("Maia-2 is not installed or its cache is empty")
        return Maia2Engine()

    def _stockfish() -> ChessEngine:
        from app.engine.stockfish_engine import StockfishEngine

        if not StockfishEngine.is_available():
            raise EngineUnavailable("Stockfish binary not found on PATH / STOCKFISH_PATH")
        return StockfishEngine()

    register_engine("mock", _mock)
    register_engine("maia2", _maia2)
    register_engine("stockfish", _stockfish)


def get_engine(name: EngineName) -> ChessEngine:
    """Return a process-wide cached instance of the named engine.

    Raises `EngineUnavailable` when the engine can't be constructed.
    """

    _build_default_factories()
    with _LOCK:
        if name in _INSTANCES:
            return _INSTANCES[name]
        factory = _FACTORIES.get(name)
        if factory is None:
            raise EngineUnavailable(f"Unknown engine: {name}")
        instance = factory()
        _INSTANCES[name] = instance
        logger.info("Initialized engine: %s", name)
        return instance


def available_engines() -> list[EngineName]:
    """Which engines can be constructed right now? Probes each one."""

    _build_default_factories()
    out: list[EngineName] = []
    for name in _FACTORIES:
        try:
            _ = get_engine(name)  # type: ignore[arg-type]
            out.append(name)  # type: ignore[arg-type]
        except EngineUnavailable:
            continue
        except Exception as exc:  # pragma: no cover — log-and-skip
            logger.warning("Engine %s failed probe: %s", name, exc)
    return out


def reset_engines_for_testing() -> None:  # pragma: no cover — test helper
    with _LOCK:
        for inst in _INSTANCES.values():
            try:
                inst.close()
            except Exception:
                pass
        _INSTANCES.clear()
