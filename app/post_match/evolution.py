"""Phase 4.3 — post-match agent evolution.

Called as a new pipeline step between `elo_ratchet` and
`memory_generation` in `app/post_match/processor.py`.

Pure data — no LLM calls, no network. Idempotent via `last_match_id`.
Skips entirely on private matches (anti-cheat guardrail from Phase 4.0b
memory).

Math + scope: see docs/phase_4_evolution.md.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.character import Character
from app.models.evolution import CharacterEvolutionState
from app.models.match import Match, MatchResult, MatchStatus
from app.models.memory import Memory, MemoryScope, MemoryType

logger = logging.getLogger(__name__)

# --- constants (tunable; cited in the design doc) ------------------------

# Slider-drift bounds (|drift| <= SLIDER_DRIFT_CLAMP per slider).
SLIDER_DRIFT_CLAMP = 2.0
# Max magnitude of a single-match slider delta.
SLIDER_DELTA_STEP = 0.5

# Opening-book EMA.
OPENING_EMA_ALPHA = 0.1
OPENING_SCORE_CLAMP = 1.0

# Trap detection thresholds.
TRAP_PLY_THRESHOLD = 12  # first critical moment must be within this many plies
TRAP_SWING_CP = 400      # centipawns of lost eval that mark a "trap"

# Tone drift (additive to MoodState at Soul call).
TONE_EMA_ALPHA = 0.05
TONE_CLAMP = 0.3
TONE_STREAK_DIVISOR = 5.0

# Opponent-strength scalar clamp — Phase 4.3 decision: strong opponents
# amplify drift; very weak opponents attenuate rather than zero.
OPP_STRENGTH_LOW = 0.3
OPP_STRENGTH_HIGH = 2.0

ACPL_RECKLESS_THRESHOLD = 80
ACPL_CAUTIOUS_THRESHOLD = 30


# --- helpers -------------------------------------------------------------


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _get_or_create_state(session: Session, character_id: str) -> CharacterEvolutionState:
    row = session.get(CharacterEvolutionState, character_id)
    if row is not None:
        return row
    row = CharacterEvolutionState(
        character_id=character_id,
        slider_drift={},
        opening_scores={},
        trap_memory=[],
        tone_drift={"confidence_baseline": 0.0, "tilt_baseline": 0.0},
        matches_processed=0,
        last_match_id=None,
    )
    session.add(row)
    session.flush()
    return row


def _character_pov_result(match: Match) -> int:
    """+1 = character won, 0 = draw, -1 = character lost. ABANDONED =
    character won (match is_private exclusion catches farm risk
    separately)."""
    if match.result is None:
        return 0
    if match.result == MatchResult.DRAW:
        return 0
    char_is_white = match.player_color.value == "black"
    if match.result == MatchResult.WHITE_WIN:
        return 1 if char_is_white else -1
    if match.result == MatchResult.BLACK_WIN:
        return 1 if not char_is_white else -1
    if match.result == MatchResult.ABANDONED and match.status == MatchStatus.ABANDONED:
        return 1
    return 0


def _opponent_strength(match: Match) -> float:
    """player_elo / character_elo_at_start, clamped. Used to amplify or
    attenuate drift based on who the character was actually fighting."""
    p_elo = match.player_elo_at_start or 1200
    c_elo = match.character_elo_at_start or 1200
    if c_elo <= 0:
        return 1.0
    return _clamp(p_elo / c_elo, OPP_STRENGTH_LOW, OPP_STRENGTH_HIGH)


# --- per-channel math (pure functions; easy to unit-test) ----------------


def select_slider_nudge(
    *, won: bool, lost: bool, char_acpl: float, trash_talk_base: int,
    trash_talk_drift: float,
) -> tuple[str, float] | None:
    """Pick at most one slider to nudge for this match. Returns
    (slider_name, delta) or None.

    Rules (documented in design doc):
    - lost + played cautiously (low ACPL) → aggression +delta
    - lost + played recklessly (high ACPL) → patience +delta
    - won + opponent stronger (checked upstream via multiplier)
      → tone drift (not a slider); returned as None here
    - trash_talk drifts back toward base over time if it's already
      skewed — homeostasis, prevents runaway drift in one direction.
    """
    if lost:
        if char_acpl < ACPL_CAUTIOUS_THRESHOLD:
            return ("aggression", +SLIDER_DELTA_STEP)
        if char_acpl > ACPL_RECKLESS_THRESHOLD:
            return ("patience", +SLIDER_DELTA_STEP)

    # Homeostasis on trash_talk when drift exceeds half the clamp.
    if abs(trash_talk_drift) > SLIDER_DRIFT_CLAMP * 0.5:
        # Pull back toward 0 by one step.
        return ("trash_talk", -SLIDER_DELTA_STEP if trash_talk_drift > 0 else +SLIDER_DELTA_STEP)

    return None


def apply_slider_drift(
    prev: dict[str, float], nudge: tuple[str, float] | None
) -> dict[str, float]:
    """Apply a single-slider nudge with cumulative clamp."""
    out = dict(prev or {})
    if nudge is None:
        return out
    slider, delta = nudge
    cur = out.get(slider, 0.0)
    out[slider] = _clamp(cur + delta, -SLIDER_DRIFT_CLAMP, +SLIDER_DRIFT_CLAMP)
    return out


def opening_ema_step(
    prev: dict[str, float], *, opening_label: str | None, signal: float
) -> dict[str, float]:
    """Update the per-opening EMA for the opening played this match."""
    out = dict(prev or {})
    if not opening_label:
        return out
    prev_score = out.get(opening_label, 0.0)
    new_score = (1.0 - OPENING_EMA_ALPHA) * prev_score + OPENING_EMA_ALPHA * signal
    out[opening_label] = _clamp(new_score, -OPENING_SCORE_CLAMP, +OPENING_SCORE_CLAMP)
    return out


def detect_trap(
    *, critical_moments: list[dict[str, Any]], character_is_white: bool,
) -> dict[str, Any] | None:
    """Return a trap descriptor (pattern str + metadata) if the first
    critical moment within TRAP_PLY_THRESHOLD plies fell on the character
    and cost ≥ TRAP_SWING_CP. Otherwise None.

    `critical_moments` is the list stored on `match_analyses.critical_moments`
    by the analysis step.
    """
    char_color = "white" if character_is_white else "black"
    for cm in sorted(critical_moments, key=lambda c: c.get("ply") or 10_000):
        ply = cm.get("ply")
        if ply is None or ply > TRAP_PLY_THRESHOLD:
            break
        # Who blundered on this move? The critical_moments schema stores
        # the side-to-move-on-this-ply in "side"; eval_loss is in cp.
        side = cm.get("side")
        loss_cp = int(cm.get("eval_loss_cp") or 0)
        if loss_cp < TRAP_SWING_CP:
            continue
        if side != char_color:
            # The OPPONENT blundered; character got the swing. That's a
            # trick they learned to play, not a trap they fell for.
            return {
                "pattern": cm.get("pattern") or f"early-{side}-blunder-ply-{ply}",
                "fell_for": False,
                "ply": ply,
                "eval_loss_cp": loss_cp,
            }
        return {
            "pattern": cm.get("pattern") or f"early-{char_color}-trap-ply-{ply}",
            "fell_for": True,
            "ply": ply,
            "eval_loss_cp": loss_cp,
        }
    return None


def update_trap_memory(
    prev: list[dict[str, Any]], *, detected: dict[str, Any] | None, now: datetime,
) -> tuple[list[dict[str, Any]], bool]:
    """Return (new_list, is_brand_new_pattern). Caller uses the second
    item to decide whether to generate a learning Memory row.
    """
    out = [dict(e) for e in (prev or [])]
    if detected is None:
        return out, False
    pattern = detected["pattern"]
    iso = now.isoformat() + "Z"
    for entry in out:
        if entry.get("pattern") == pattern:
            if detected["fell_for"]:
                entry["fell_for"] = int(entry.get("fell_for", 0)) + 1
            else:
                entry["avoided"] = int(entry.get("avoided", 0)) + 1
            entry["last_seen_at"] = iso
            return out, False
    # New pattern.
    out.append({
        "pattern": pattern,
        "fell_for": 1 if detected["fell_for"] else 0,
        "avoided": 0 if detected["fell_for"] else 1,
        "last_seen_at": iso,
        "first_seen_ply": detected.get("ply"),
        "first_eval_loss_cp": detected.get("eval_loss_cp"),
    })
    return out, True


def tone_ema_step(
    prev: dict[str, float], *, win_streak: int, loss_streak: int,
) -> dict[str, float]:
    out = dict(prev or {})
    conf = float(out.get("confidence_baseline", 0.0))
    tilt = float(out.get("tilt_baseline", 0.0))
    target_conf = _clamp(win_streak / TONE_STREAK_DIVISOR, -TONE_CLAMP, +TONE_CLAMP)
    target_tilt = _clamp(-loss_streak / TONE_STREAK_DIVISOR, -TONE_CLAMP, +TONE_CLAMP)
    out["confidence_baseline"] = _clamp(
        (1.0 - TONE_EMA_ALPHA) * conf + TONE_EMA_ALPHA * target_conf,
        -TONE_CLAMP, +TONE_CLAMP,
    )
    out["tilt_baseline"] = _clamp(
        (1.0 - TONE_EMA_ALPHA) * tilt + TONE_EMA_ALPHA * target_tilt,
        -TONE_CLAMP, +TONE_CLAMP,
    )
    return out


def recent_streaks(session: Session, character_id: str, limit: int = 20) -> tuple[int, int]:
    """Return (win_streak, loss_streak) looking back through the
    character's most recent finished matches (excluding the current
    in-flight one).
    """
    rows = session.execute(
        select(Match.result, Match.player_color)
        .where(Match.character_id == character_id)
        .where(Match.ended_at.is_not(None))
        .where(Match.status.in_([MatchStatus.COMPLETED, MatchStatus.RESIGNED, MatchStatus.ABANDONED]))
        .order_by(Match.ended_at.desc())
        .limit(limit)
    ).all()
    win_streak = 0
    loss_streak = 0
    for (res, player_color) in rows:
        if res is None or res == MatchResult.DRAW:
            break
        char_is_white = player_color.value == "black"
        char_won = (
            (res == MatchResult.WHITE_WIN and char_is_white)
            or (res == MatchResult.BLACK_WIN and not char_is_white)
            or (res == MatchResult.ABANDONED)  # abandon counts as char win
        )
        if char_won and loss_streak == 0:
            win_streak += 1
        elif not char_won and win_streak == 0:
            loss_streak += 1
        else:
            break
    return win_streak, loss_streak


# --- main entry ----------------------------------------------------------


@dataclass
class EvolutionSummary:
    """What happened this call. For logging + tests."""

    skipped_private: bool = False
    skipped_idempotent: bool = False
    slider_updated: tuple[str, float] | None = None
    opening_updated: str | None = None
    trap_detected: dict[str, Any] | None = None
    new_learning_memory_id: str | None = None
    tone_before: dict[str, float] | None = None
    tone_after: dict[str, float] | None = None


def _per_side_acpl(analysis_moves: Iterable[dict[str, Any]], *, char_color: str) -> float:
    total = 0
    count = 0
    for m in analysis_moves:
        if m.get("side") != char_color:
            continue
        total += max(0, int(m.get("eval_loss_cp") or 0))
        count += 1
    return (total / count) if count else 0.0


def apply_evolution(
    session: Session,
    *,
    match: Match,
    analysis_moves: list[dict[str, Any]],
    critical_moments: list[dict[str, Any]],
    opening_label: str | None = None,
) -> EvolutionSummary:
    """Entry point — call once per finished match.

    Safe to re-run on the same match (idempotent via `last_match_id`).
    Safe on private matches (returns early, state untouched).
    """
    summary = EvolutionSummary()

    if match.is_private:
        logger.info("[evolution] match %s is_private → skipping entirely", match.id)
        summary.skipped_private = True
        return summary

    character = session.get(Character, match.character_id)
    if character is None:
        logger.warning("[evolution] character missing for match %s", match.id)
        return summary

    state = _get_or_create_state(session, character.id)
    if state.last_match_id == match.id:
        logger.info("[evolution] already processed match %s — no-op", match.id)
        summary.skipped_idempotent = True
        return summary

    pov = _character_pov_result(match)
    signal = pov * _opponent_strength(match)
    char_is_white = match.player_color.value == "black"
    char_color = "white" if char_is_white else "black"
    char_acpl = _per_side_acpl(analysis_moves, char_color=char_color)

    # Slider drift.
    nudge = select_slider_nudge(
        won=pov > 0,
        lost=pov < 0,
        char_acpl=char_acpl,
        trash_talk_base=character.trash_talk,
        trash_talk_drift=float(state.slider_drift.get("trash_talk", 0.0)),
    )
    new_slider = apply_slider_drift(state.slider_drift, nudge)
    if new_slider != state.slider_drift:
        state.slider_drift = new_slider
    summary.slider_updated = nudge

    # Opening-book EMA.
    if opening_label:
        new_openings = opening_ema_step(
            state.opening_scores, opening_label=opening_label, signal=float(signal)
        )
        if new_openings != state.opening_scores:
            state.opening_scores = new_openings
            summary.opening_updated = opening_label

    # Trap memory.
    detected = detect_trap(
        critical_moments=critical_moments, character_is_white=char_is_white
    )
    summary.trap_detected = detected
    new_trap_memory, brand_new = update_trap_memory(
        state.trap_memory, detected=detected, now=datetime.utcnow()
    )
    if new_trap_memory != state.trap_memory:
        state.trap_memory = new_trap_memory

    # Learning memory row for first-ever sighting of a trap.
    if brand_new and detected is not None and detected.get("fell_for"):
        memo = _make_learning_memory(character, detected)
        session.add(memo)
        session.flush()
        summary.new_learning_memory_id = memo.id

    # Tone drift.
    win_streak, loss_streak = recent_streaks(session, character.id)
    summary.tone_before = dict(state.tone_drift or {})
    state.tone_drift = tone_ema_step(
        state.tone_drift, win_streak=win_streak, loss_streak=loss_streak
    )
    summary.tone_after = dict(state.tone_drift)

    # Stamp + commit.
    state.last_match_id = match.id
    state.matches_processed = int(state.matches_processed or 0) + 1
    state.last_updated_at = datetime.utcnow()
    session.flush()

    logger.info(
        "[evolution] match=%s char=%s slider_nudge=%s opening=%s trap=%s "
        "win_streak=%s loss_streak=%s",
        match.id,
        character.id,
        nudge,
        summary.opening_updated,
        (detected or {}).get("pattern"),
        win_streak,
        loss_streak,
    )
    return summary


def _make_learning_memory(character: Character, detected: dict[str, Any]) -> Memory:
    """Create a first-sighting learning memory. The narrative is
    deterministic — we want this to be cheap, not LLM-generated.
    The post-match narrative step can still elaborate later.
    """
    pattern = detected["pattern"]
    ply = detected.get("ply")
    loss_cp = detected.get("eval_loss_cp")
    narrative = (
        f"An opponent caught me with {pattern} around move "
        f"{max(1, (ply or 0) // 2)}; I lost about {loss_cp}cp in one blow. "
        "Next time an opening looks like that, I should slow down."
    )
    return Memory(
        character_id=character.id,
        scope=MemoryScope.CHARACTER_LORE,
        type=MemoryType.LEARNING,
        emotional_valence=-0.35,
        triggers=[pattern, "trap", "opening"],
        narrative_text=narrative,
        relevance_tags=["learning", "opening", "caution"],
    )


# --- integration helpers used by the Director / Soul wiring --------------


def effective_sliders(character: Character, state: CharacterEvolutionState | None) -> dict[str, int]:
    """Return the four sliders with cumulative drift applied, rounded to
    the nearest integer and clamped to [1, 10] (the slider range).
    `None` state means "no evolution yet" — use base.
    """
    drift = (state.slider_drift if state else {}) or {}
    out: dict[str, int] = {}
    for name in ("aggression", "risk_tolerance", "patience", "trash_talk"):
        base = int(getattr(character, name))
        delta = float(drift.get(name, 0.0))
        out[name] = int(round(max(1, min(10, base + delta))))
    return out


def tone_bias_for(state: CharacterEvolutionState | None) -> dict[str, float]:
    """Return confidence/tilt biases for feeding into MoodState init.
    `None` state → zeros.
    """
    drift = (state.tone_drift if state else {}) or {}
    return {
        "confidence_baseline": float(drift.get("confidence_baseline", 0.0)),
        "tilt_baseline": float(drift.get("tilt_baseline", 0.0)),
    }
