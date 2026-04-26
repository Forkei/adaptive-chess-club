"""Build the `status_callback` that the post-match processor invokes between steps.

Kept in its own module so the processor doesn't import the Socket.IO layer
directly — the callback is opt-in and optional.
"""

from __future__ import annotations

import logging
from typing import Callable

from app.sockets.bridge import emit_post_match_complete, emit_post_match_status

logger = logging.getLogger(__name__)


def _read_agent_elo_info(match_id: str) -> tuple[int | None, int | None, str | None]:
    """Read agent elo delta, new elo, and agent_id from DB for the complete event.

    Returns (agent_elo_delta, agent_elo_new, agent_id) or (None, None, None).
    """
    try:
        from app.db import SessionLocal
        from app.models.match import Match, MatchAnalysis
        from sqlalchemy import select

        with SessionLocal() as session:
            match = session.get(Match, match_id)
            if match is None or match.match_kind != "agent_vs_character":
                return None, None, None
            agent_id = match.participant_agent_id
            analysis = session.execute(
                select(MatchAnalysis).where(MatchAnalysis.match_id == match_id)
            ).scalar_one_or_none()
            if analysis is None:
                return None, None, agent_id
            delta = analysis.player_elo_delta_applied
            new_elo = match.player_elo_at_end
            return delta, new_elo, agent_id
    except Exception:
        logger.warning("_read_agent_elo_info failed for match=%s", match_id, exc_info=True)
        return None, None, None


def build_processor_callback(match_id: str) -> Callable[[str, dict], None]:
    """Return a callback that emits Socket.IO status events for this match.

    Called from the processor's daemon thread; dispatches via the async bridge.
    """

    def _callback(event: str, payload: dict) -> None:
        try:
            if event == "step_started":
                emit_post_match_status(
                    match_id,
                    status="running",
                    steps_completed=payload.get("steps_completed", []),
                    current_step=payload.get("current_step"),
                )
            elif event == "step_completed":
                emit_post_match_status(
                    match_id,
                    status="running",
                    steps_completed=payload.get("steps_completed", []),
                    current_step=None,
                )
            elif event == "pipeline_completed":
                emit_post_match_status(
                    match_id,
                    status="completed",
                    steps_completed=payload.get("steps_completed", []),
                )
                agent_elo_delta, agent_elo_new, agent_id = _read_agent_elo_info(match_id)
                emit_post_match_complete(
                    match_id,
                    summary_url=f"/matches/{match_id}/summary",
                    agent_elo_delta=agent_elo_delta,
                    agent_elo_new=agent_elo_new,
                    agent_id=agent_id,
                )
            elif event == "pipeline_failed":
                emit_post_match_status(
                    match_id,
                    status="failed",
                    steps_completed=payload.get("steps_completed", []),
                    error=payload.get("error"),
                )
        except Exception:
            # Never let a socket emit fail break the processor.
            logger.exception("processor_callback emit failed for match=%s", match_id)

    return _callback
