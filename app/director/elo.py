"""Elo ratchet helper.

Pure functions. The post-match processor (Phase 2b) will call these after
each game — in 2a they exist so tests + the Director's effective-Elo
math can share one source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass

# Rule knobs live in code so tests exercise the same values as prod.
ELO_DELTA_GAIN = 0.1          # 10% of signed outcome delta
ELO_DELTA_CAP = 30            # per-match clamp on |current_elo change|
FLOOR_RAISE_MARGIN = 100      # current must hover this far above floor...
FLOOR_RAISE_WINDOW = 3        # ...across this many recent matches...
FLOOR_RAISE_STEP = 25         # ...to promote the floor this much.


@dataclass(frozen=True)
class RatchetResult:
    new_current_elo: int
    new_floor_elo: int
    current_elo_change: int
    floor_elo_raised: bool


def apply_elo_ratchet(
    *,
    current_elo: int,
    floor_elo: int,
    max_elo: int,
    elo_delta_raw: float,
    recent_current_elos: list[int],
    adaptive: bool,
) -> RatchetResult:
    """Apply post-match Elo movement and maybe raise the floor.

    `elo_delta_raw` is the signed outcome signal (positive = character played
    better than expected; negative = worse). We apply a fixed 10% gain and
    clamp to ±30 per match, then clamp the result to [floor, max].

    Non-adaptive characters don't move at all.
    """

    if not adaptive:
        return RatchetResult(
            new_current_elo=current_elo,
            new_floor_elo=floor_elo,
            current_elo_change=0,
            floor_elo_raised=False,
        )

    scaled = int(round(elo_delta_raw * ELO_DELTA_GAIN))
    change = max(-ELO_DELTA_CAP, min(ELO_DELTA_CAP, scaled))
    new_current = max(floor_elo, min(max_elo, current_elo + change))
    actual_change = new_current - current_elo

    # Floor raise: look at the last N recent currents (inclusive of this match).
    window = recent_current_elos[-(FLOOR_RAISE_WINDOW - 1) :] + [new_current]
    floor_raised = False
    if len(window) == FLOOR_RAISE_WINDOW and all(
        c >= floor_elo + FLOOR_RAISE_MARGIN for c in window
    ):
        new_floor = min(max_elo, floor_elo + FLOOR_RAISE_STEP)
        floor_raised = new_floor != floor_elo
    else:
        new_floor = floor_elo

    return RatchetResult(
        new_current_elo=new_current,
        new_floor_elo=new_floor,
        current_elo_change=actual_change,
        floor_elo_raised=floor_raised,
    )


def outcome_delta(
    *,
    character_won: bool | None,
    is_draw: bool,
    move_quality_bonus: float = 0.0,
) -> float:
    """Translate a match outcome into a signed 'delta' value.

    +200 for a clean win, -200 for a clean loss, small bonus for draw
    depending on who was favoured (Phase 2b post-match supplies that).
    `move_quality_bonus` is a Phase 2b signal — in 2a it's always zero.
    """
    if is_draw:
        return move_quality_bonus
    if character_won is None:
        return 0.0
    return (200.0 if character_won else -200.0) + move_quality_bonus
