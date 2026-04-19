"""Build the `status_callback` that the post-match processor invokes between steps.

Kept in its own module so the processor doesn't import the Socket.IO layer
directly — the callback is opt-in and optional.
"""

from __future__ import annotations

import logging
from typing import Callable

from app.sockets.bridge import emit_post_match_complete, emit_post_match_status

logger = logging.getLogger(__name__)


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
                emit_post_match_complete(
                    match_id,
                    summary_url=f"/matches/{match_id}/summary",
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
