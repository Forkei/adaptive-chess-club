"""Deterministic mood/engine orchestration — no LLM calls here.

The Director is a pure function from (character, mood, opponent profile,
match context) to an `EngineConfig`. Keeping it purely deterministic
means it's trivially testable and introspectable: you can read off from
the slider values exactly what the engine will do.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.director.mood import MoodState
from app.engine.base import EngineConfig, EngineName
from app.engine.maia2_engine import MAIA_MAX_ELO, MAIA_MIN_ELO, _clamp_to_maia_bucket
from app.models.character import Character

# --- Rule constants (tweakable without changing code structure) ---

BEAST_MODE_ELO_THRESHOLD = 2100          # above this, always Stockfish
BEAST_MODE_AGGRESSION_SLIDER = 9         # high-aggression characters...
BEAST_MODE_CONFIDENCE_GATE = 0.7         # ...when confident enough...

MAIA_LOWER_FALLBACK_ELO = MAIA_MIN_ELO   # below this, fall through to Stockfish

CONFIDENCE_BONUS_PER_UNIT = 100.0        # +100 Elo at confidence=1
TILT_PENALTY_PER_UNIT = 150.0            # -150 Elo at tilt=1

TIME_BUDGET_BASE_PATIENCE_1 = 0.5        # seconds
TIME_BUDGET_BASE_PATIENCE_10 = 5.0       # seconds
TIME_BUDGET_AGGRESSION_DISCOUNT = 0.3    # 1.0 aggression knocks 30% off
TIME_BUDGET_CAP = 8.0                    # hard ceiling


@dataclass(frozen=True)
class MatchContext:
    move_number: int
    game_phase: str          # opening / middlegame / endgame
    player_color: str        # "white" | "black"
    engines_available: frozenset[str]


@dataclass(frozen=True)
class OpponentProfileSummary:
    """Whatever slice of the Phase 2b OpponentProfile matters here.

    Unused by the 2a Director rules — accepted as a parameter so the signature
    doesn't change when 2b wires it up.
    """

    aggression_index: float | None = None
    blunder_rate: float | None = None
    typical_opening_eco: str | None = None


def compute_effective_elo(character: Character, mood: MoodState) -> int:
    bonus = CONFIDENCE_BONUS_PER_UNIT * mood.confidence - TILT_PENALTY_PER_UNIT * mood.tilt
    raw = int(round(character.current_elo + bonus))
    return max(character.floor_elo, min(character.max_elo, raw))


def _time_budget(character: Character, mood: MoodState) -> float:
    # Patience 1 -> 0.5s; patience 10 -> 5.0s. Linear in between.
    span = TIME_BUDGET_BASE_PATIENCE_10 - TIME_BUDGET_BASE_PATIENCE_1
    base = TIME_BUDGET_BASE_PATIENCE_1 + (character.patience - 1) * span / 9
    scaled = base * (1.0 - TIME_BUDGET_AGGRESSION_DISCOUNT * mood.aggression)
    return max(0.1, min(TIME_BUDGET_CAP, scaled))


def _select_engine(
    *, character: Character, effective_elo: int, mood: MoodState, engines_available: frozenset[str]
) -> EngineName:
    # Hard fallback first: if Maia-2 is out, use Stockfish.
    if "maia2" not in engines_available and "stockfish" in engines_available:
        return "stockfish"
    # Beast mode: the character is strong enough or rabid enough that we want
    # Stockfish's crispness over Maia's human-likeness.
    in_beast_band = effective_elo > BEAST_MODE_ELO_THRESHOLD
    in_beast_mood = (
        character.aggression >= BEAST_MODE_AGGRESSION_SLIDER
        and mood.confidence > BEAST_MODE_CONFIDENCE_GATE
    )
    if (in_beast_band or in_beast_mood) and "stockfish" in engines_available:
        return "stockfish"
    # Below Maia's trained range, Stockfish handles low skill better.
    if effective_elo < MAIA_LOWER_FALLBACK_ELO and "stockfish" in engines_available:
        return "stockfish"
    # Default: Maia-2 when available, else Stockfish, else mock (tests / dev).
    if "maia2" in engines_available:
        return "maia2"
    if "stockfish" in engines_available:
        return "stockfish"
    return "mock"


def choose_engine_config(
    *,
    character: Character,
    mood: MoodState,
    opponent_profile: OpponentProfileSummary | None = None,
    match_context: MatchContext,
) -> EngineConfig:
    """Pure function — no side effects, no DB reads.

    Callers: the match service at every engine turn.
    """

    effective_elo = compute_effective_elo(character, mood)
    engine = _select_engine(
        character=character,
        effective_elo=effective_elo,
        mood=mood,
        engines_available=match_context.engines_available,
    )
    time_budget = _time_budget(character, mood)

    cfg = EngineConfig(
        target_elo=effective_elo,
        time_budget_seconds=time_budget,
        engine_name=engine,
    )
    if engine == "maia2":
        cfg = cfg.model_copy(update={"maia_elo_bucket": _clamp_to_maia_bucket(effective_elo)})
    elif engine == "stockfish":
        # Patient characters want depth; aggressive ones want speed. Skill_level is
        # optional — setting None lets the engine derive from UCI_Elo.
        cfg = cfg.model_copy(update={"contempt": 0})
    return cfg
