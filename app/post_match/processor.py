"""Post-match orchestrator.

Ties together: engine analysis → feature extraction → Elo ratchet → memory
generation → narrative summary update. Each step runs in its own try/except
so a failure in one doesn't kill the rest; MatchAnalysis is updated
incrementally so the polling endpoint reflects progress.

Runs synchronously when called directly (easy to test, easy to profile).
The public `start_post_match_background` spawns a daemon thread for
the HTTP hot path — same pattern as Phase 1's memory_generator.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable

from sqlalchemy import select
from sqlalchemy.orm import Session

# Phase 3b: optional per-step status callback. Invoked with `(event, payload)`
# where event ∈ {"step_started", "step_completed", "pipeline_completed",
# "pipeline_failed"}. Failures inside the callback are swallowed — the pipeline
# owns completion, not the observer.
StatusCallback = Callable[[str, dict], None]

from app.db import SessionLocal, session_scope
from app.llm.client import LLMClient, LLMError, get_llm_client
from app.memory.crud import bulk_create
from app.models.character import Character
from app.models.match import (
    Match,
    MatchAnalysis,
    MatchAnalysisStatus,
    MatchResult,
    MatchStatus,
    Move,
    OpponentProfile,
)
from app.models.memory import Memory
from app.post_match.analysis import analyze_match_moves, identify_critical_moments
from app.post_match.elo_apply import (
    BothSidesResult,
    apply_to_both,
    compute_elo_delta,
    _apply_character_ratchet_item2,
    _games_played,
    _recent_current_elos,
)
from app.post_match.evolution import apply_evolution
from app.post_match.features import extract_features, merge_features
from app.post_match.memory_gen import generate_match_memories, update_narrative_summary

logger = logging.getLogger(__name__)

# Steps in execution order. Mirrored in the UI.
STEP_ENGINE_ANALYSIS = "engine_analysis"
STEP_FEATURES = "feature_extraction"
STEP_ELO_RATCHET = "elo_ratchet"
# Phase 4.3 — evolution runs AFTER Elo and BEFORE memory generation so
# newly-detected traps can be referenced in the LLM memory narratives.
STEP_EVOLUTION = "evolution"
STEP_MEMORY_GEN = "memory_generation"
STEP_NARRATIVE = "narrative_summary"

ALL_STEPS = [
    STEP_ENGINE_ANALYSIS,
    STEP_FEATURES,
    STEP_ELO_RATCHET,
    STEP_EVOLUTION,
    STEP_MEMORY_GEN,
    STEP_NARRATIVE,
]


# --- Helpers ---------------------------------------------------------------


def _ensure_analysis_row(session: Session, match_id: str) -> MatchAnalysis:
    existing = session.execute(
        select(MatchAnalysis).where(MatchAnalysis.match_id == match_id)
    ).scalar_one_or_none()
    if existing is not None:
        return existing
    row = MatchAnalysis(match_id=match_id, status=MatchAnalysisStatus.PENDING)
    session.add(row)
    session.flush()
    return row


def _get_analysis(session: Session, match_id: str) -> MatchAnalysis:
    row = session.execute(
        select(MatchAnalysis).where(MatchAnalysis.match_id == match_id)
    ).scalar_one()
    return row


def _move_rows_to_dicts(moves: list[Move]) -> list[dict[str, Any]]:
    return [
        {
            "move_number": m.move_number,
            "side": m.side.value if hasattr(m.side, "value") else str(m.side),
            "uci": m.uci,
            "san": m.san,
            "eval_cp": m.eval_cp,
            "time_taken_ms": m.time_taken_ms,
        }
        for m in moves
    ]


MAX_AUDIT_OPPONENT_NOTES = 50


def _drain_opponent_notes(match: Match) -> list[dict[str, Any]]:
    """Pop `pending_opponent_notes` off match.extra_state and return them.

    Keeps the drained notes in `drained_opponent_notes` for audit, capped at
    `MAX_AUDIT_OPPONENT_NOTES` with FIFO eviction so a pathologically long
    correspondence game can't grow this field unboundedly.

    Caller is expected to commit the mutation.
    """
    state = dict(match.extra_state or {})
    notes = list(state.get("pending_opponent_notes", []))
    state["pending_opponent_notes"] = []
    prior_audit = list(state.get("drained_opponent_notes", []))
    combined = prior_audit + notes
    if len(combined) > MAX_AUDIT_OPPONENT_NOTES:
        combined = combined[-MAX_AUDIT_OPPONENT_NOTES:]
    state["drained_opponent_notes"] = combined
    match.extra_state = state
    return notes


def _player_color_for_player(match: Match) -> str:
    return match.player_color.value


def _mark_step(session: Session, analysis: MatchAnalysis, step: str) -> None:
    steps = list(analysis.steps_completed or [])
    if step not in steps:
        steps.append(step)
        analysis.steps_completed = steps
        session.flush()


# --- Agent-vs-character Elo helper -----------------------------------------


def _apply_agent_vs_character_elo(
    session: Session,
    *,
    match: Match,
    analysis_moves: list[dict[str, Any]],
) -> tuple[Any, BothSidesResult]:
    """Compute + apply Elo for an agent_vs_character match.

    Mirrors apply_to_both but updates PlayerAgent.elo instead of Player.elo
    so the human watcher's rating is left untouched.
    """
    from app.director.elo import ELO_DELTA_CAP, RatchetResult
    from app.models.player_agent import PlayerAgent

    character = session.get(Character, match.character_id)
    if character is None:
        raise RuntimeError(f"Character {match.character_id} missing")
    agent = session.get(PlayerAgent, match.participant_agent_id)
    if agent is None:
        raise RuntimeError(f"PlayerAgent {match.participant_agent_id} missing")

    char_games = _games_played(session, character_id=character.id)
    match.player_elo_at_start = agent.elo  # correct the mis-set value from service.py

    computation = compute_elo_delta(
        match=match,
        analysis_moves=analysis_moves,
        character_elo=character.current_elo,
        player_elo=agent.elo,
        character_games_played=char_games,
        player_games_played=0,
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

    # Agent elo: simple clamped change, no floor/ceiling beyond 400–3000.
    raw_agent = computation.player_elo_delta_raw
    capped = int(round(max(-ELO_DELTA_CAP, min(ELO_DELTA_CAP, raw_agent))))
    new_agent_elo = max(400, min(3000, agent.elo + capped))
    actual_agent_change = new_agent_elo - agent.elo
    agent.elo = new_agent_elo
    match.player_elo_at_end = new_agent_elo
    session.flush()

    agent_result = RatchetResult(
        new_current_elo=new_agent_elo,
        new_floor_elo=400,
        current_elo_change=actual_agent_change,
        floor_elo_raised=False,
    )
    return computation, BothSidesResult(character=char_result, player=agent_result)


# --- Main pipeline ---------------------------------------------------------


@dataclass
class ProcessorConfig:
    run_engine_analysis: bool = True
    # When True, memory generation + narrative summary are skipped
    # (used by tests without a real LLM).
    run_llm_steps: bool = True


def _safe_notify(callback: StatusCallback | None, event: str, payload: dict) -> None:
    if callback is None:
        return
    try:
        callback(event, payload)
    except Exception:
        logger.exception("Post-match status_callback crashed for event=%s", event)


def process_match_post_game(
    match_id: str,
    *,
    llm: LLMClient | None = None,
    config: ProcessorConfig | None = None,
    status_callback: StatusCallback | None = None,
) -> None:
    """Run the full post-match pipeline for `match_id`.

    Safe to call from a thread. Each step catches its own exceptions;
    a catastrophic failure bubbles up to the outer handler which marks
    the analysis FAILED.

    `status_callback` (Phase 3b) is invoked on each step transition so the
    Socket.IO layer can stream progress to connected clients. It is called
    from the worker thread; callers bridging to an async event loop should
    do so via `asyncio.run_coroutine_threadsafe`.
    """
    cfg = config or ProcessorConfig()

    try:
        _run_pipeline(match_id, llm=llm, cfg=cfg, status_callback=status_callback)
        _safe_notify(
            status_callback,
            "pipeline_completed",
            {"steps_completed": list(ALL_STEPS), "match_id": match_id},
        )
    except Exception as exc:
        logger.exception("Post-match pipeline crashed for match=%s", match_id)
        _safe_notify(
            status_callback,
            "pipeline_failed",
            {"error": f"{type(exc).__name__}: {exc}", "match_id": match_id},
        )
        # Best-effort mark as failed.
        try:
            with session_scope() as session:
                row = session.execute(
                    select(MatchAnalysis).where(MatchAnalysis.match_id == match_id)
                ).scalar_one_or_none()
                if row is None:
                    row = MatchAnalysis(match_id=match_id)
                    session.add(row)
                row.status = MatchAnalysisStatus.FAILED
                row.error = f"{type(exc).__name__}: {exc}"
                row.completed_at = datetime.utcnow()
        except Exception:
            logger.exception("Failed to persist post-match FAILED state")


def _run_pipeline(
    match_id: str,
    *,
    llm: LLMClient | None,
    cfg: ProcessorConfig,
    status_callback: StatusCallback | None = None,
) -> None:
    completed: list[str] = []

    def _start(step: str) -> None:
        _safe_notify(
            status_callback,
            "step_started",
            {"current_step": step, "steps_completed": list(completed), "match_id": match_id},
        )

    def _finish(step: str) -> None:
        if step not in completed:
            completed.append(step)
        _safe_notify(
            status_callback,
            "step_completed",
            {"current_step": step, "steps_completed": list(completed), "match_id": match_id},
        )
    # Initialize / resume.
    with session_scope() as session:
        match = session.get(Match, match_id)
        if match is None:
            logger.warning("Post-match: match %s not found", match_id)
            return
        if match.status not in (MatchStatus.COMPLETED, MatchStatus.RESIGNED, MatchStatus.ABANDONED):
            logger.warning("Post-match: match %s not terminal (status=%s) — skipping", match_id, match.status)
            return
        analysis = _ensure_analysis_row(session, match_id)
        if analysis.status == MatchAnalysisStatus.COMPLETED:
            logger.info("Post-match: analysis for %s already complete — skipping", match_id)
            return
        analysis.status = MatchAnalysisStatus.RUNNING
        analysis.started_at = analysis.started_at or datetime.utcnow()
        analysis.error = None
        analysis_id = analysis.id
        moves_for_analysis = _move_rows_to_dicts(list(match.moves))
        character_id = match.character_id
        player_id = match.player_id
        initial_fen = match.initial_fen
        is_agent_match = (
            match.match_kind == "agent_vs_character"
            and match.participant_agent_id is not None
        )
        participant_agent_id = match.participant_agent_id

    # --- Step 1: engine analysis -------------------------------------------
    engine_result: dict[str, Any] = {}
    critical_moments: list[dict[str, Any]] = []
    _start(STEP_ENGINE_ANALYSIS)
    try:
        if cfg.run_engine_analysis:
            engine_result = analyze_match_moves(
                moves_for_analysis, initial_fen=initial_fen
            )
            critical_moments = identify_critical_moments(engine_result)
        else:
            engine_result = {"status": "skipped", "reason": "disabled_by_config", "moves": []}
        with session_scope() as session:
            a = _get_analysis(session, match_id)
            a.engine_analysis = engine_result
            a.critical_moments = critical_moments
            _mark_step(session, a, STEP_ENGINE_ANALYSIS)
        _finish(STEP_ENGINE_ANALYSIS)
    except Exception as exc:
        logger.exception("engine_analysis failed for match=%s: %s", match_id, exc)

    # --- Step 2: feature extraction ----------------------------------------
    features_before: dict[str, Any] | None = None
    features_after: dict[str, Any] = {}
    _start(STEP_FEATURES)
    try:
        with session_scope() as session:
            match = session.get(Match, match_id)
            profile = _get_or_create_profile(session, character_id, player_id)
            features_before = dict(profile.style_features) if profile.style_features else None
            # `abandoned_last` means the player walked away — disconnect timeout only.
            # A clean resign is a legitimate finish; don't flag it as abandonment.
            abandoned = match.status == MatchStatus.ABANDONED
            new_features = extract_features(
                moves=_move_rows_to_dicts(list(match.moves)),
                player_color=_player_color_for_player(match),
                analysis=engine_result,
                abandoned=abandoned,
            )
            merged = merge_features(
                features_before,
                new_features,
                prior_games=profile.games_played,
            )
            profile.style_features = merged
            profile.last_match_at = datetime.utcnow()
            profile.features_version = (profile.features_version or 0) + 1
            # Update W/L/D + games_played. Result already set on match.
            _bump_profile_records(profile, match)
            features_after = dict(merged)

            a = _get_analysis(session, match_id)
            a.features = features_after
            _mark_step(session, a, STEP_FEATURES)
        _finish(STEP_FEATURES)
    except Exception as exc:
        logger.exception("feature_extraction failed for match=%s: %s", match_id, exc)

    # --- Step 3: Elo ratchet -----------------------------------------------
    _start(STEP_ELO_RATCHET)
    try:
        with session_scope() as session:
            match = session.get(Match, match_id)
            if is_agent_match:
                computation, both = _apply_agent_vs_character_elo(
                    session, match=match,
                    analysis_moves=engine_result.get("moves") or [],
                )
            else:
                computation, both = apply_to_both(
                    session, match=match, analysis_moves=engine_result.get("moves") or []
                )
            a = _get_analysis(session, match_id)
            a.elo_delta_raw = computation.elo_delta_raw
            a.elo_delta_applied = both.character.current_elo_change
            a.floor_raised = both.character.floor_elo_raised
            a.player_elo_delta_applied = both.player.current_elo_change
            a.player_floor_raised = both.player.floor_elo_raised
            _mark_step(session, a, STEP_ELO_RATCHET)
        _finish(STEP_ELO_RATCHET)
    except Exception as exc:
        logger.exception("elo_ratchet failed for match=%s: %s", match_id, exc)

    # --- Step 3.6: $CLAY wager settlement (one-shot guard via stake_settled_at) ---
    try:
        with session_scope() as session:
            match = session.get(Match, match_id)
            if match and match.stake_cents > 0 and match.stake_settled_at is None:
                _settle_wager(session, match)
    except Exception as exc:
        logger.exception("wager_settlement failed for match=%s: %s", match_id, exc)

    # --- Step 3.5: Evolution (pure data, no LLM, private matches skipped) --
    _start(STEP_EVOLUTION)
    try:
        with session_scope() as session:
            match = session.get(Match, match_id)
            analysis_row = _get_analysis(session, match_id)
            opening_label = (analysis_row.features or {}).get("opening_label")
            apply_evolution(
                session,
                match=match,
                analysis_moves=engine_result.get("moves") or [],
                critical_moments=list(analysis_row.critical_moments or []),
                opening_label=opening_label,
            )
            _mark_step(session, analysis_row, STEP_EVOLUTION)
        _finish(STEP_EVOLUTION)
    except Exception as exc:
        logger.exception("evolution failed for match=%s: %s", match_id, exc)

    # --- Steps 4 + 5: memory gen + narrative update (LLM) ------------------
    llm_client = llm
    if cfg.run_llm_steps and llm_client is None:
        try:
            llm_client = get_llm_client()
        except LLMError as exc:
            logger.warning("Post-match: LLM unavailable (%s) — skipping steps 4+5", exc)
            llm_client = None
            cfg = ProcessorConfig(run_engine_analysis=cfg.run_engine_analysis, run_llm_steps=False)

    if cfg.run_llm_steps and llm_client is not None:
        generated_ids: list[str] = []
        _start(STEP_MEMORY_GEN)
        try:
            with session_scope() as session:
                match = session.get(Match, match_id)
                character = session.get(Character, character_id)
                # Drain opponent notes from match.extra_state.
                opponent_notes = _drain_opponent_notes(match)
                prior_memories = list(
                    session.execute(
                        select(Memory).where(Memory.character_id == character_id).limit(20)
                    ).scalars()
                )
                memory_creates = generate_match_memories(
                    character=character,
                    match=match,
                    critical_moments=critical_moments,
                    features_before=features_before,
                    features_after=features_after,
                    opponent_notes=opponent_notes,
                    prior_memories=prior_memories,
                    llm=llm_client,
                )
                if memory_creates:
                    rows = bulk_create(
                        session,
                        character_id=character_id,
                        items=memory_creates,
                        embed=True,
                    )
                    generated_ids = [r.id for r in rows]

                a = _get_analysis(session, match_id)
                a.generated_memory_ids = generated_ids
                _mark_step(session, a, STEP_MEMORY_GEN)
            _finish(STEP_MEMORY_GEN)
        except Exception as exc:
            logger.exception("memory_generation failed for match=%s: %s", match_id, exc)

        _start(STEP_NARRATIVE)
        try:
            with session_scope() as session:
                match = session.get(Match, match_id)
                character = session.get(Character, character_id)
                profile = _get_or_create_profile(session, character_id, player_id)
                new_summary = update_narrative_summary(
                    character=character,
                    match=match,
                    features=features_after or dict(profile.style_features or {}),
                    previous_summary=profile.narrative_summary or None,
                    llm=llm_client,
                )
                if new_summary:
                    profile.narrative_summary = new_summary
                a = _get_analysis(session, match_id)
                _mark_step(session, a, STEP_NARRATIVE)
            _finish(STEP_NARRATIVE)
        except Exception as exc:
            logger.exception("narrative_summary failed for match=%s: %s", match_id, exc)

    # --- Finalize ----------------------------------------------------------
    with session_scope() as session:
        a = _get_analysis(session, match_id)
        a.status = MatchAnalysisStatus.COMPLETED
        a.completed_at = datetime.utcnow()


# --- Internal helpers ------------------------------------------------------


def _settle_wager(session: Session, match: Match) -> None:
    """Settle the $CLAY wager for a match. Called at most once (one-shot guard).

    Determines outcome relative to the player/agent (not the character), then:
      Win  → credit 2× stake (stake back + equal win).
      Loss → no credit (stake already debited at match creation; log a marker txn).
      Draw / Abandoned → refund the stake.

    Sets match.stake_settled_at within the caller's session so the guard is
    committed atomically with any credit/debit.
    """
    from datetime import datetime
    from app.economy.clay_ledger import get_ledger

    stake = match.stake_cents
    player_id = match.player_id
    match_id = match.id

    # Determine outcome from first principles (player_outcome() conflates
    # RESIGNED and ABANDONED; we want to refund abandoned matches per spec).
    if match.status == MatchStatus.ABANDONED:
        raw_outcome = "abandoned"
    elif match.status in (MatchStatus.COMPLETED, MatchStatus.RESIGNED):
        if match.result == MatchResult.DRAW:
            raw_outcome = "draw"
        elif match.result == MatchResult.ABANDONED:
            raw_outcome = "abandoned"
        else:
            from app.models.match import Color as _Color
            player_side = match.player_color
            winning_side = (
                _Color.WHITE if match.result == MatchResult.WHITE_WIN else _Color.BLACK
            )
            raw_outcome = "win" if player_side == winning_side else "loss"
    else:
        logger.warning("_settle_wager: match %s is not terminal (status=%s)", match_id, match.status)
        return

    ledger = get_ledger()

    if raw_outcome == "win":
        ledger.credit(player_id, stake * 2, reason="match_win", match_id=match_id)
        logger.info(
            "wager settled: win player=%s match=%s +%d cents", player_id, match_id, stake * 2
        )
    elif raw_outcome in ("draw", "abandoned"):
        ledger.credit(player_id, stake, reason="match_draw_refund", match_id=match_id)
        logger.info(
            "wager settled: %s player=%s match=%s refund %d cents",
            raw_outcome, player_id, match_id, stake,
        )
    else:
        # Loss — stake was already debited. Record an audit marker with amount=0
        # so transaction history shows what happened.
        ledger.credit(player_id, 0, reason="match_loss", match_id=match_id)
        logger.info(
            "wager settled: loss player=%s match=%s (stake %d forfeited)",
            player_id, match_id, stake,
        )

    match.stake_settled_at = datetime.utcnow()
    session.flush()


def _get_or_create_profile(session: Session, character_id: str, player_id: str) -> OpponentProfile:
    row = session.execute(
        select(OpponentProfile).where(
            OpponentProfile.character_id == character_id,
            OpponentProfile.player_id == player_id,
        )
    ).scalar_one_or_none()
    if row is not None:
        return row
    row = OpponentProfile(
        character_id=character_id,
        player_id=player_id,
        games_played=0,
        style_features={},
        narrative_summary="",
    )
    session.add(row)
    session.flush()
    return row


def _bump_profile_records(profile: OpponentProfile, match: Match) -> None:
    profile.games_played = (profile.games_played or 0) + 1
    if match.status == MatchStatus.ABANDONED:
        # Disconnect-timeout / rage-quit → character won.
        profile.games_won_by_character = (profile.games_won_by_character or 0) + 1
        profile.abandoned_count = (profile.abandoned_count or 0) + 1
        return
    if match.status == MatchStatus.RESIGNED:
        # Clean resign → character won.
        profile.games_won_by_character = (profile.games_won_by_character or 0) + 1
        profile.resigned_count = (profile.resigned_count or 0) + 1
        return
    # Determine winner from result + player_color.
    from app.models.match import MatchResult

    if match.result == MatchResult.DRAW:
        profile.games_drawn = (profile.games_drawn or 0) + 1
        return
    char_is_white = match.player_color.value == "black"
    char_won = (
        (match.result == MatchResult.WHITE_WIN and char_is_white)
        or (match.result == MatchResult.BLACK_WIN and not char_is_white)
    )
    if char_won:
        profile.games_won_by_character = (profile.games_won_by_character or 0) + 1
    else:
        profile.games_lost_by_character = (profile.games_lost_by_character or 0) + 1


# --- Threaded launcher -----------------------------------------------------


def start_post_match_background(
    match_id: str,
    *,
    llm: LLMClient | None = None,
    config: ProcessorConfig | None = None,
    status_callback: StatusCallback | None = None,
) -> threading.Thread:
    """Spawn a daemon thread running `process_match_post_game`.

    Idempotent from the caller's perspective: if an analysis row already
    exists and is COMPLETED, the thread no-ops immediately. Returns the
    thread so tests can `join()` it.
    """
    # Short-circuit duplicate kickoff: quick DB peek.
    try:
        with SessionLocal() as session:
            existing = session.execute(
                select(MatchAnalysis).where(MatchAnalysis.match_id == match_id)
            ).scalar_one_or_none()
            if existing and existing.status == MatchAnalysisStatus.COMPLETED:
                logger.debug("post-match already complete for %s; skipping spawn", match_id)
                # Still fire the completion event — a new observer (e.g. a
                # socket that reconnected post-completion) may be waiting.
                _safe_notify(
                    status_callback,
                    "pipeline_completed",
                    {"steps_completed": list(ALL_STEPS), "match_id": match_id},
                )
                return _noop_thread()
            if existing and existing.status == MatchAnalysisStatus.RUNNING:
                logger.debug("post-match already running for %s; skipping spawn", match_id)
                return _noop_thread()
    except Exception as exc:
        logger.warning("Failed to peek MatchAnalysis for %s: %s — spawning anyway", match_id, exc)

    t = threading.Thread(
        target=process_match_post_game,
        args=(match_id,),
        kwargs={"llm": llm, "config": config, "status_callback": status_callback},
        daemon=True,
        name=f"postmatch-{match_id[:8]}",
    )
    t.start()
    return t


def _noop_thread() -> threading.Thread:
    t = threading.Thread(target=lambda: None, daemon=True)
    t.start()
    return t
