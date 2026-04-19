"""Compute and apply Elo delta for a finished match.

Patch Pass 2 Item 2 overhaul: swapped the flat win=+200 / loss=-200 model for
the standard Elo expected-score formula, driven by the relative ratings of
the character and the player. Both sides now rate — a character beating a
600-rated player gains almost nothing; beating a 2200-rated player is
worth much more.

Formula:
    E_character = 1 / (1 + 10 ** ((E_player - E_character) / 400))
    actual_character ∈ {1.0 win, 0.5 draw, 0.0 loss}  # character POV
    change_character = K * (actual_character - E_character)

K-factor:
    32 for "new" (<30 games played by that rater)
    16 once established

Move quality stays but as a smaller adjustment (±10 cap, not ±100):
    move_quality_delta = clamp((opp_total - char_total) / 100, -10, +10)
    Applied on top of the expected-score change. Skipped on rage-quit.

Short-game scaling:
    < 10 ply  → final_delta *= 0.3
    < 2  ply  → final_delta clamped to ±3  (zero-move resign)

The player's expected score is the mirror: E_player = 1 - E_character.
Both Character and Player ratchets follow the same current/floor/ceiling
shape. Player has no "adaptive" toggle — they always float.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.director.elo import (
    ELO_DELTA_CAP,
    FLOOR_RAISE_MARGIN,
    FLOOR_RAISE_STEP,
    FLOOR_RAISE_WINDOW,
    RatchetResult,
    apply_elo_ratchet,
)
from app.models.character import Character
from app.models.match import Match, MatchResult, MatchStatus, Player

logger = logging.getLogger(__name__)

MOVE_QUALITY_MAX_PER_MOVE = 300
MOVE_QUALITY_CAP_ITEM_2 = 10            # secondary signal, lighter weight than before
MOVE_QUALITY_DIVISOR_ITEM_2 = 100.0
SHORT_MATCH_SCALE = 0.3                 # < 10 ply
SHORT_MATCH_HALF_MOVES = 10
ZERO_MOVE_CLAMP = 3                     # < 2 ply
ZERO_MOVE_HALF_MOVES = 2
K_NEW = 32
K_ESTABLISHED = 16
K_SWITCH_GAMES = 30


@dataclass(frozen=True)
class EloComputation:
    """Result of the character-side computation.

    `outcome_delta` here is the expected-score-based Elo change BEFORE move
    quality / short-game scaling, in Elo points. `move_quality_delta` is the
    small secondary bonus. `elo_delta_raw` is what goes into the ratchet.
    """

    outcome_delta: float
    move_quality_delta: float
    elo_delta_raw: float
    short_match_halved: bool       # kept as the field name for UI compat; semantics = "scaled"
    rage_quit_skipped_quality: bool
    expected_score: float
    actual_score: float
    k_factor: int
    # Player-side counterpart — mirror of the character computation.
    player_elo_delta_raw: float = 0.0
    player_expected_score: float = 0.0
    player_actual_score: float = 0.0
    player_k_factor: int = K_NEW


def _character_color_is_white(match: Match) -> bool:
    return match.player_color.value == "black"


def _character_won(match: Match) -> bool | None:
    """Returns True/False for decisive, None for draw."""
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


def _k_factor(games_played: int) -> int:
    return K_NEW if games_played < K_SWITCH_GAMES else K_ESTABLISHED


def expected_score(rating_a: int, rating_b: int) -> float:
    """Standard Elo expected score of A vs B."""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def compute_elo_delta(
    *,
    match: Match,
    analysis_moves: list[dict[str, Any]],
    character_elo: int | None = None,
    player_elo: int | None = None,
    character_games_played: int = 0,
    player_games_played: int = 0,
) -> EloComputation:
    """Compute the per-match Elo movement for both sides.

    The session-aware caller (`apply_to_both`) supplies ratings + games-played
    counts. Defaults make it easy to call from tests with only a `Match`.
    """
    char_elo = character_elo if character_elo is not None else match.character_elo_at_start
    p_elo = player_elo if player_elo is not None else (match.player_elo_at_start or 1200)

    # Expected + actual (character POV).
    e_char = expected_score(char_elo, p_elo)
    e_player = 1.0 - e_char

    is_abandoned = match.status == MatchStatus.ABANDONED

    if match.result == MatchResult.DRAW:
        actual_char = 0.5
    else:
        won = _character_won(match)
        if won is None:
            # ABANDONED → character wins.
            actual_char = 1.0 if is_abandoned else 0.0
        else:
            actual_char = 1.0 if won else 0.0
    actual_player = 1.0 - actual_char

    k_char = _k_factor(character_games_played)
    k_player = _k_factor(player_games_played)

    outcome_char = k_char * (actual_char - e_char)
    outcome_player = k_player * (actual_player - e_player)

    # Move quality: lighter weight. Skip on rage-quit.
    if is_abandoned:
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
        raw_quality = (opp_total - char_total) / MOVE_QUALITY_DIVISOR_ITEM_2
        move_quality = max(-MOVE_QUALITY_CAP_ITEM_2, min(MOVE_QUALITY_CAP_ITEM_2, raw_quality))

    # Character raw delta before short-game scaling.
    raw_char = outcome_char + move_quality
    # Player raw delta mirrors move-quality sign (if the character played well,
    # that reflects badly on the player and vice versa).
    raw_player = outcome_player - move_quality

    # Short / zero-move scaling. Move count is stored as plies (half-moves).
    plies = match.move_count or 0
    short = False
    if plies < ZERO_MOVE_HALF_MOVES and not is_abandoned:
        raw_char = max(-ZERO_MOVE_CLAMP, min(ZERO_MOVE_CLAMP, raw_char))
        raw_player = max(-ZERO_MOVE_CLAMP, min(ZERO_MOVE_CLAMP, raw_player))
        short = True
    elif plies < SHORT_MATCH_HALF_MOVES and not is_abandoned:
        raw_char *= SHORT_MATCH_SCALE
        raw_player *= SHORT_MATCH_SCALE
        short = True

    return EloComputation(
        outcome_delta=outcome_char,
        move_quality_delta=move_quality,
        elo_delta_raw=raw_char,
        short_match_halved=short,
        rage_quit_skipped_quality=is_abandoned,
        expected_score=e_char,
        actual_score=actual_char,
        k_factor=k_char,
        player_elo_delta_raw=raw_player,
        player_expected_score=e_player,
        player_actual_score=actual_player,
        player_k_factor=k_player,
    )


def _recent_current_elos(session: Session, character_id: str, limit: int) -> list[int]:
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


def _recent_player_elos(session: Session, player_id: str, limit: int) -> list[int]:
    stmt = (
        select(Match.player_elo_at_end)
        .where(Match.player_id == player_id)
        .where(Match.player_elo_at_end.is_not(None))
        .order_by(Match.ended_at.desc())
        .limit(limit)
    )
    rows = [r[0] for r in session.execute(stmt).all() if r[0] is not None]
    rows.reverse()
    return rows


def _games_played(session: Session, *, character_id: str | None = None,
                  player_id: str | None = None) -> int:
    """Completed / terminal matches for the given side."""
    stmt = select(func.count(Match.id)).where(
        Match.status.in_([MatchStatus.COMPLETED, MatchStatus.RESIGNED, MatchStatus.ABANDONED])
    )
    if character_id:
        stmt = stmt.where(Match.character_id == character_id)
    if player_id:
        stmt = stmt.where(Match.player_id == player_id)
    return int(session.execute(stmt).scalar() or 0)


def _apply_player_ratchet(
    *,
    current_elo: int,
    floor_elo: int,
    ceiling_elo: int,
    elo_delta_raw: float,
    recent_player_elos: list[int],
) -> RatchetResult:
    """Player-side ratchet — same shape as the character ratchet but with
    no `adaptive` toggle (players always float), no 0.1 gain scalar (we
    apply the full Elo change from the formula), and `ceiling_elo` as the
    upper clamp instead of `max_elo`.
    """
    change = int(round(elo_delta_raw))
    change = max(-ELO_DELTA_CAP, min(ELO_DELTA_CAP, change))
    new_current = max(floor_elo, min(ceiling_elo, current_elo + change))
    actual_change = new_current - current_elo

    window = recent_player_elos[-(FLOOR_RAISE_WINDOW - 1):] + [new_current]
    floor_raised = False
    if len(window) == FLOOR_RAISE_WINDOW and all(
        c >= floor_elo + FLOOR_RAISE_MARGIN for c in window
    ):
        new_floor = min(ceiling_elo, floor_elo + FLOOR_RAISE_STEP)
        floor_raised = new_floor != floor_elo
    else:
        new_floor = floor_elo

    return RatchetResult(
        new_current_elo=new_current,
        new_floor_elo=new_floor,
        current_elo_change=actual_change,
        floor_elo_raised=floor_raised,
    )


def _apply_character_ratchet_item2(
    *,
    current_elo: int,
    floor_elo: int,
    max_elo: int,
    elo_delta_raw: float,
    recent_current_elos: list[int],
    adaptive: bool,
) -> RatchetResult:
    """Character ratchet using the expected-score delta directly (no 0.1 scalar)."""
    if not adaptive:
        return RatchetResult(
            new_current_elo=current_elo,
            new_floor_elo=floor_elo,
            current_elo_change=0,
            floor_elo_raised=False,
        )
    change = int(round(elo_delta_raw))
    change = max(-ELO_DELTA_CAP, min(ELO_DELTA_CAP, change))
    new_current = max(floor_elo, min(max_elo, current_elo + change))
    actual_change = new_current - current_elo

    window = recent_current_elos[-(FLOOR_RAISE_WINDOW - 1):] + [new_current]
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


@dataclass(frozen=True)
class BothSidesResult:
    character: RatchetResult
    player: RatchetResult


def apply_to_both(
    session: Session,
    *,
    match: Match,
    analysis_moves: list[dict[str, Any]],
) -> tuple[EloComputation, BothSidesResult]:
    """Compute + apply the ratchet to both character and player.

    Persists Elo state on Character + Player, stamps `match.character_elo_at_end`
    and `match.player_elo_at_end`. Caller commits.
    """
    character = session.get(Character, match.character_id)
    if character is None:
        raise RuntimeError(f"Character {match.character_id} missing during post-match apply")
    player = session.get(Player, match.player_id)
    if player is None:
        raise RuntimeError(f"Player {match.player_id} missing during post-match apply")

    char_games = _games_played(session, character_id=character.id)
    player_games = _games_played(session, player_id=player.id)

    # Snapshot player_elo_at_start on the match if missing (older match rows).
    if match.player_elo_at_start is None:
        match.player_elo_at_start = player.elo

    computation = compute_elo_delta(
        match=match,
        analysis_moves=analysis_moves,
        character_elo=character.current_elo,
        player_elo=player.elo,
        character_games_played=char_games,
        player_games_played=player_games,
    )

    char_recent = _recent_current_elos(session, character.id, limit=2)
    char_result = _apply_character_ratchet_item2(
        current_elo=character.current_elo,
        floor_elo=character.floor_elo,
        max_elo=character.max_elo,
        elo_delta_raw=computation.elo_delta_raw,
        recent_current_elos=char_recent,
        adaptive=character.adaptive,
    )
    character.current_elo = char_result.new_current_elo
    character.floor_elo = char_result.new_floor_elo
    match.character_elo_at_end = char_result.new_current_elo

    player_recent = _recent_player_elos(session, player.id, limit=2)
    player_result = _apply_player_ratchet(
        current_elo=player.elo,
        floor_elo=player.elo_floor,
        ceiling_elo=player.elo_ceiling,
        elo_delta_raw=computation.player_elo_delta_raw,
        recent_player_elos=player_recent,
    )
    player.elo = player_result.new_current_elo
    player.elo_floor = player_result.new_floor_elo
    match.player_elo_at_end = player_result.new_current_elo

    session.flush()
    return computation, BothSidesResult(character=char_result, player=player_result)


# Backwards-compat shim: existing callers of `apply_to_character` still work,
# but they now also apply the player-side ratchet under the hood.
def apply_to_character(
    session: Session,
    *,
    match: Match,
    elo_delta_raw: float,  # kept for signature compat — ignored, recomputed
) -> RatchetResult:
    """Deprecated: prefer `apply_to_both`.

    Retained because some call sites (and tests that patch it) import this
    symbol directly. Signature-compatible; the `elo_delta_raw` argument is
    effectively ignored — the full recompute uses live ratings + game counts.
    """
    _, both = apply_to_both(session, match=match, analysis_moves=[])
    return both.character
