"""Compute and apply Elo delta for a finished match.

Per spec (Phase 2b):
- outcome_delta: win=+200, draw=0, loss=-200 (character's POV)
- move_quality_delta: per-move blunder_magnitude = clamp(0, eval_loss, 300).
  Sum character total + opponent total. Delta = clamp((opp - char) / 10, -100, 100).
- elo_delta_raw = outcome_delta + move_quality_delta
- If match length < 10 half-moves: halve raw before the multiplier.
- If match.status == abandoned (disconnect-timeout / rage-quit):
  use outcome_delta only, skip move_quality. A RESIGNED match runs the
  normal move_quality math — the player chose to end, not walk away.
- apply_elo_ratchet then clamps with ±30 and ±10% gain; floor ratchet if
  current > floor+100 for last 3 matches.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.director.elo import RatchetResult, apply_elo_ratchet
from app.models.character import Character
from app.models.match import Match, MatchResult, MatchStatus

logger = logging.getLogger(__name__)

MOVE_QUALITY_MAX_PER_MOVE = 300
MOVE_QUALITY_DELTA_CAP = 100
MOVE_QUALITY_DIVISOR = 10.0
SHORT_MATCH_HALF_MOVES = 10


@dataclass(frozen=True)
class EloComputation:
    outcome_delta: float
    move_quality_delta: float
    elo_delta_raw: float
    short_match_halved: bool
    rage_quit_skipped_quality: bool


def _character_color_is_white(match: Match) -> bool:
    # Character plays opposite the player.
    return match.player_color.value == "black"


def _character_won(match: Match) -> bool | None:
    """Returns None for draw / abandoned / not terminal."""
    if match.result is None:
        return None
    if match.result == MatchResult.DRAW:
        return None
    char_is_white = _character_color_is_white(match)
    if match.result == MatchResult.WHITE_WIN:
        return char_is_white
    if match.result == MatchResult.BLACK_WIN:
        return not char_is_white
    return None


def compute_elo_delta(
    *,
    match: Match,
    analysis_moves: list[dict[str, Any]],
) -> EloComputation:
    """Compute `elo_delta_raw` without applying it.

    `analysis_moves` is the `engine_analysis["moves"]` list — one entry
    per move with at least `side` and `eval_loss_cp`. When empty (engine
    skipped), move_quality is zero and only outcome contributes.
    """
    if match.result == MatchResult.DRAW:
        outcome = 0.0
    else:
        won = _character_won(match)
        if won is None:
            # ABANDONED status (disconnect-timeout) sets result=MatchResult.ABANDONED.
            # RESIGNED status sets result=WHITE_WIN/BLACK_WIN so it goes through the
            # `won` branch normally. Both paths score the character as winning.
            if match.result == MatchResult.ABANDONED:
                outcome = 200.0
            else:
                outcome = 0.0
        else:
            outcome = 200.0 if won else -200.0

    is_abandoned = match.status == MatchStatus.ABANDONED

    if is_abandoned:
        # Rage-quit: use outcome only.
        move_quality = 0.0
    else:
        char_color = "white" if _character_color_is_white(match) else "black"
        char_total = 0
        opp_total = 0
        for am in analysis_moves:
            side = am.get("side")
            loss = max(0, min(MOVE_QUALITY_MAX_PER_MOVE, int(am.get("eval_loss_cp") or 0)))
            if side == char_color:
                char_total += loss
            else:
                opp_total += loss
        move_quality = max(
            -MOVE_QUALITY_DELTA_CAP,
            min(MOVE_QUALITY_DELTA_CAP, (opp_total - char_total) / MOVE_QUALITY_DIVISOR),
        )

    raw = outcome + move_quality
    short = len(analysis_moves) < SHORT_MATCH_HALF_MOVES and not is_abandoned
    if short:
        raw = raw / 2.0

    return EloComputation(
        outcome_delta=outcome,
        move_quality_delta=move_quality,
        elo_delta_raw=raw,
        short_match_halved=short,
        rage_quit_skipped_quality=is_abandoned,
    )


def _recent_current_elos(session: Session, character_id: str, limit: int) -> list[int]:
    """Return up to `limit` most-recent `character_elo_at_end` values from
    completed prior matches, oldest-first. The `apply_elo_ratchet` helper
    expects the window to be oldest→newest."""
    stmt = (
        select(Match.character_elo_at_end)
        .where(Match.character_id == character_id)
        .where(Match.character_elo_at_end.is_not(None))
        .order_by(Match.ended_at.desc())
        .limit(limit)
    )
    rows = [r[0] for r in session.execute(stmt).all() if r[0] is not None]
    rows.reverse()
    return rows


def apply_to_character(
    session: Session,
    *,
    match: Match,
    elo_delta_raw: float,
) -> RatchetResult:
    """Apply the computed delta to the character + persist on the match.

    Reads/writes happen via the passed session; caller commits.
    """
    character = session.get(Character, match.character_id)
    if character is None:
        raise RuntimeError(f"Character {match.character_id} missing during post-match apply")

    recent = _recent_current_elos(session, character.id, limit=2)
    result = apply_elo_ratchet(
        current_elo=character.current_elo,
        floor_elo=character.floor_elo,
        max_elo=character.max_elo,
        elo_delta_raw=elo_delta_raw,
        recent_current_elos=recent,
        adaptive=character.adaptive,
    )
    character.current_elo = result.new_current_elo
    character.floor_elo = result.new_floor_elo
    match.character_elo_at_end = result.new_current_elo
    session.flush()
    return result
