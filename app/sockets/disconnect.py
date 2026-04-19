"""Per-match disconnect cooldown registry.

When a player's socket drops, we stamp `match.extra_state.disconnect_started_at`
and schedule an asyncio task that waits `cooldown_seconds` then abandons the
match. If the player reconnects inside the window, the task is cancelled.

In-process single-worker assumption (Phase 4 concern for Redis). The registry
is a plain dict keyed by match_id.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

from app.config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class _CooldownEntry:
    task: asyncio.Task
    player_id: str
    started_at: datetime
    deadline: datetime


_entries: dict[str, _CooldownEntry] = {}


def cooldown_seconds() -> int:
    return int(get_settings().match_disconnect_cooldown_seconds)


def is_active(match_id: str) -> bool:
    entry = _entries.get(match_id)
    if entry is None:
        return False
    if entry.task.done():
        # Leftover from a completed task; clean up.
        _entries.pop(match_id, None)
        return False
    return True


def deadline_for(match_id: str) -> datetime | None:
    entry = _entries.get(match_id)
    return entry.deadline if entry else None


def start(
    match_id: str,
    *,
    player_id: str,
    on_timeout,
    seconds: int | None = None,
) -> datetime:
    """Start (or restart) a cooldown task. Returns the deadline.

    `on_timeout` is an async callable `(match_id) -> None` invoked if the cooldown
    fires without a `cancel`. Must be idempotent — it may get called once at most,
    but callers should not rely on exclusivity.
    """
    cancel(match_id)
    duration = cooldown_seconds() if seconds is None else seconds
    started = datetime.utcnow()
    deadline = started + timedelta(seconds=duration)

    async def _runner() -> None:
        try:
            await asyncio.sleep(duration)
        except asyncio.CancelledError:
            logger.debug("Disconnect cooldown cancelled for match %s", match_id)
            raise
        try:
            await on_timeout(match_id)
        except Exception:
            logger.exception("Disconnect timeout handler failed for match %s", match_id)
        finally:
            _entries.pop(match_id, None)

    task = asyncio.create_task(_runner(), name=f"disconnect-cooldown-{match_id[:8]}")
    _entries[match_id] = _CooldownEntry(
        task=task, player_id=player_id, started_at=started, deadline=deadline
    )
    logger.info(
        "Disconnect cooldown started for match=%s player=%s duration=%ss",
        match_id,
        player_id,
        duration,
    )
    return deadline


def cancel(match_id: str) -> bool:
    entry = _entries.pop(match_id, None)
    if entry is None:
        return False
    if not entry.task.done():
        entry.task.cancel()
    logger.info("Disconnect cooldown cancelled for match=%s", match_id)
    return True


def reset_all() -> None:
    """Test helper — cancel every pending cooldown task."""
    for match_id in list(_entries.keys()):
        cancel(match_id)
