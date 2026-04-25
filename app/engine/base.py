from __future__ import annotations

import abc
from typing import Any, Literal

import chess
from pydantic import BaseModel, ConfigDict, Field


EngineName = Literal["maia2", "stockfish", "mock"]


class EngineUnavailable(RuntimeError):
    """Raised when the requested engine isn't installed / initialized."""


class EngineConfig(BaseModel):
    """Everything the Director hands down to the Body.

    Serializable so we can persist it with a Move row for debugging.
    """

    target_elo: int = Field(..., ge=400, le=3000)
    time_budget_seconds: float = Field(2.0, ge=0.1, le=60.0)
    engine_name: EngineName = "maia2"
    # Stockfish knobs (ignored by Maia-2 / mock)
    skill_level: int | None = Field(None, ge=0, le=20)
    contempt: int | None = Field(None, ge=-100, le=100)
    depth: int | None = Field(None, ge=1, le=40)
    # For Maia-2, which takes an Elo bucket
    maia_elo_bucket: int | None = Field(None, ge=1100, le=1900)
    shuffle_guard_lookback: int = Field(6, ge=0, le=20)


class ConsideredMove(BaseModel):
    """One of the top-K alternatives the engine looked at."""

    uci: str
    san: str | None = None
    eval_cp: int | None = None       # Stockfish only
    probability: float | None = None  # Maia-2 only


class MoveResult(BaseModel):
    """Output of a single engine call."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    move: str  # uci
    san: str
    eval_cp: int | None = None
    considered_moves: list[ConsideredMove] = Field(default_factory=list)
    time_taken_ms: int
    engine_name: EngineName
    thinking_depth: int | None = None
    raw: dict[str, Any] | None = None  # engine-specific debug bag; not persisted to Move


class ChessEngine(abc.ABC):
    """Abstract interface every engine wrapper implements."""

    name: EngineName

    @classmethod
    @abc.abstractmethod
    def is_available(cls) -> bool:
        """Quick, cheap check — does this engine have everything it needs?"""

    @abc.abstractmethod
    def get_move(self, board: chess.Board, config: EngineConfig) -> MoveResult:
        """Select a move for the side to move. Blocking; call via asyncio.to_thread."""

    def close(self) -> None:  # pragma: no cover — default noop
        """Optional resource cleanup. Engines with subprocesses override this."""
        return None
