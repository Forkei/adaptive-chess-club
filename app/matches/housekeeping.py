"""Periodic housekeeping for match lifecycle (Patch Pass 1).

Two sweeps live here and are exposed as a single public entry point so `app.main`
can run them once at startup and on a timer:

1. **Stale match reaper** — matches stuck in `IN_PROGRESS` long past any real
   activity get marked ABANDONED. Default criteria: `started_at` older than
   `STALE_MATCH_REAPER_HOURS` *and* `move_count < STALE_MATCH_MOVE_THRESHOLD`.
   Narrow on purpose — we don't want to kill an active correspondence-style game.
2. **Stuck-analysis sweep** — `MatchAnalysis` rows stuck in `RUNNING` past
   `STUCK_ANALYSIS_MINUTES` are marked `FAILED`. Prevents zombie rows after a
   server crash (the `start_post_match_background` short-circuit skips RUNNING
   rows, so without this they'd never be reprocessed).

3. **Disconnect restoration** — on startup only, walk every IN_PROGRESS match
   with `extra_state.disconnect_started_at` set. If the cooldown window already
   expired, finish the abandon path now; if still in-window, re-arm an asyncio
   cooldown task for the remaining time.

Kept deliberately conservative: favour false-negatives (leaving something alive)
over false-positives (killing live work). All actions are logged.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta

from sqlalchemy import select

from app.config import get_settings
from app.db import SessionLocal
from app.models.match import Match, MatchAnalysis, MatchAnalysisStatus, MatchStatus

logger = logging.getLogger(__name__)


# --- Individual sweeps ----------------------------------------------------


def reap_stale_matches() -> int:
    """Mark old-and-empty IN_PROGRESS matches as ABANDONED. Returns count reaped."""
    settings = get_settings()
    cutoff = datetime.utcnow() - timedelta(hours=settings.stale_match_reaper_hours)
    threshold = settings.stale_match_move_threshold
    reaped = 0
    with SessionLocal() as session:
        stmt = select(Match).where(
            Match.status == MatchStatus.IN_PROGRESS,
            Match.started_at < cutoff,
            Match.move_count < threshold,
        )
        for match in session.execute(stmt).scalars():
            match.status = MatchStatus.ABANDONED
            from app.models.match import MatchResult
            match.result = MatchResult.ABANDONED
            match.ended_at = datetime.utcnow()
            match.character_elo_at_end = match.character_elo_at_start
            reaped += 1
            logger.info(
                "Reaped stale match %s (age=%s, moves=%d)",
                match.id,
                datetime.utcnow() - match.started_at,
                match.move_count,
            )
        if reaped:
            session.commit()
    return reaped


def fail_stuck_analyses() -> int:
    """Mark `MatchAnalysis` rows stuck in RUNNING as FAILED. Returns count failed."""
    settings = get_settings()
    cutoff = datetime.utcnow() - timedelta(minutes=settings.stuck_analysis_minutes)
    failed = 0
    with SessionLocal() as session:
        stmt = select(MatchAnalysis).where(
            MatchAnalysis.status == MatchAnalysisStatus.RUNNING,
            MatchAnalysis.started_at < cutoff,
        )
        for row in session.execute(stmt).scalars():
            row.status = MatchAnalysisStatus.FAILED
            row.error = (
                f"Auto-failed by housekeeping sweep: RUNNING since {row.started_at} "
                f"(cutoff {settings.stuck_analysis_minutes}m)."
            )
            row.completed_at = datetime.utcnow()
            failed += 1
            logger.warning(
                "Marked stuck MatchAnalysis %s as FAILED (match=%s, started_at=%s)",
                row.id,
                row.match_id,
                row.started_at,
            )
        if failed:
            session.commit()
    return failed


async def restore_disconnect_cooldowns() -> tuple[int, int]:
    """On startup, re-arm in-window cooldowns and finish expired ones.

    Returns (rearmed_count, expired_count).
    """
    from app.sockets import disconnect as disconnect_registry

    settings = get_settings()
    window = timedelta(seconds=settings.match_disconnect_cooldown_seconds)
    now = datetime.utcnow()
    rearmed = 0
    expired = 0
    candidates: list[tuple[str, str, datetime]] = []

    with SessionLocal() as session:
        stmt = select(Match).where(Match.status == MatchStatus.IN_PROGRESS)
        for match in session.execute(stmt).scalars():
            state = match.extra_state or {}
            started_iso = state.get("disconnect_started_at")
            player_id = state.get("disconnect_player_id")
            if not started_iso or not player_id:
                continue
            try:
                started = datetime.fromisoformat(started_iso)
            except ValueError:
                logger.warning(
                    "Skipping disconnect restore for match=%s: bad timestamp %r",
                    match.id, started_iso,
                )
                continue
            candidates.append((match.id, player_id, started + window))

    # Do the actual work outside the session scope so the async callback can
    # open its own sessions cleanly.
    from app.sockets.server import _on_disconnect_timeout
    for match_id, player_id, deadline in candidates:
        remaining = (deadline - now).total_seconds()
        if remaining <= 0:
            # Window already expired during downtime. Finish the abandon path now.
            try:
                await _on_disconnect_timeout(match_id)
            except Exception:
                logger.exception(
                    "Disconnect restore: expired cooldown handler failed for %s", match_id,
                )
            expired += 1
        else:
            disconnect_registry.start(
                match_id,
                player_id=player_id,
                on_timeout=_on_disconnect_timeout,
                seconds=int(remaining),
            )
            rearmed += 1
            logger.info(
                "Restored disconnect cooldown for match=%s (%.0fs remaining)",
                match_id, remaining,
            )
    return rearmed, expired


# --- Orchestration --------------------------------------------------------


async def run_once() -> None:
    """One sweep of all non-startup-only housekeeping. Safe on every tick."""
    try:
        reap_stale_matches()
    except Exception:
        logger.exception("reap_stale_matches failed")
    try:
        fail_stuck_analyses()
    except Exception:
        logger.exception("fail_stuck_analyses failed")


async def run_startup() -> None:
    """Startup-time sweep. Includes the one-shot disconnect restore."""
    await run_once()
    try:
        await restore_disconnect_cooldowns()
    except Exception:
        logger.exception("restore_disconnect_cooldowns failed")


async def periodic_loop() -> None:
    """Background coroutine — runs `run_once` every housekeeping_interval_seconds.

    Cancellation-safe: on CancelledError, exits promptly without work.
    """
    settings = get_settings()
    interval = max(30, int(settings.housekeeping_interval_seconds))
    while True:
        try:
            await asyncio.sleep(interval)
        except asyncio.CancelledError:
            logger.info("Housekeeping periodic loop cancelled")
            raise
        try:
            await run_once()
        except Exception:
            logger.exception("Housekeeping tick failed; continuing")


# --- PvP flag-fall sweep --------------------------------------------------
# Runs at a much faster cadence than the main housekeeping loop (clocks
# need ~1s resolution to feel right). Kept separate so long sweeps in
# housekeeping don't delay flag detection.

PVP_FLAGFALL_INTERVAL_SECONDS = 1.0


async def pvp_flagfall_loop() -> None:
    """Tight loop that reaps PvP matches whose side-to-move has flagged."""
    from app.lobbies.pvp_service import flagfall_sweep

    while True:
        try:
            await asyncio.sleep(PVP_FLAGFALL_INTERVAL_SECONDS)
        except asyncio.CancelledError:
            logger.info("PvP flag-fall loop cancelled")
            raise
        try:
            with SessionLocal() as s:
                flagged = flagfall_sweep(s)
                if flagged:
                    logger.info("[pvp-clock] flagged %d match(es): %s", len(flagged), flagged)
        except Exception:
            logger.exception("pvp_flagfall_loop tick failed; continuing")
