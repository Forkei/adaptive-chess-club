"""Bridge between threaded (sync) code and the Socket.IO async event loop.

The post-match processor runs in a daemon thread. When it needs to emit a
socket event (status update, completion), it calls into here rather than
directly awaiting. We capture the main event loop once at app startup and
schedule coroutines onto it via `asyncio.run_coroutine_threadsafe`.

If the loop is shutting down (app terminating) the emit fails silently —
processor completion is more important than status delivery to a socket that
may already be gone.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from app.sockets.events import (
    NAMESPACE,
    S2C_POST_MATCH_COMPLETE,
    S2C_POST_MATCH_STATUS,
    PostMatchCompletePayload,
    PostMatchStatusPayload,
    match_room_name,
)

logger = logging.getLogger(__name__)


# Captured at app startup. Kept as a simple global — there's one event loop per
# uvicorn worker, and multi-worker deployment is out of scope (Phase 4 concern).
_loop: asyncio.AbstractEventLoop | None = None


def set_main_loop(loop: asyncio.AbstractEventLoop | None) -> None:
    global _loop
    _loop = loop


def get_main_loop() -> asyncio.AbstractEventLoop | None:
    return _loop


def _run_threadsafe(coro) -> None:
    """Dispatch `coro` onto the main loop from a non-async thread. Best-effort.

    We deliberately do NOT wait for the result — the caller is a background
    pipeline and shouldn't block on socket I/O.
    """
    loop = _loop
    if loop is None:
        # App never captured the loop, or we're in a test without Socket.IO wired.
        logger.debug("bridge: no main loop captured; dropping emit")
        coro.close()
        return
    try:
        asyncio.run_coroutine_threadsafe(coro, loop)
    except RuntimeError as exc:
        # Loop closing / closed. Drop the emit; the processor keeps going.
        logger.warning("bridge: failed to schedule emit (loop shutting down?): %s", exc)
        try:
            coro.close()
        except Exception:
            pass


# --- Public helpers used by the post-match processor ----------------------


async def _emit_status(match_id: str, payload: PostMatchStatusPayload) -> None:
    # Import locally to avoid a circular import between server <-> bridge at module load.
    from app.sockets.server import sio

    await sio.emit(
        S2C_POST_MATCH_STATUS,
        payload.model_dump(mode="json"),
        room=match_room_name(match_id),
        namespace=NAMESPACE,
    )


async def _emit_complete(match_id: str, payload: PostMatchCompletePayload) -> None:
    from app.sockets.server import sio

    await sio.emit(
        S2C_POST_MATCH_COMPLETE,
        payload.model_dump(mode="json"),
        room=match_room_name(match_id),
        namespace=NAMESPACE,
    )


def emit_post_match_status(
    match_id: str,
    *,
    status: str,
    steps_completed: list[str],
    current_step: str | None = None,
    error: str | None = None,
) -> None:
    """Called by the post-match processor's `status_callback` from a worker thread."""
    payload = PostMatchStatusPayload(
        match_id=match_id,
        status=status,  # type: ignore[arg-type]
        steps_completed=list(steps_completed),
        current_step=current_step,
        error=error,
    )
    _run_threadsafe(_emit_status(match_id, payload))


def emit_post_match_complete(match_id: str, *, summary_url: str) -> None:
    payload = PostMatchCompletePayload(match_id=match_id, summary_url=summary_url)
    _run_threadsafe(_emit_complete(match_id, payload))
