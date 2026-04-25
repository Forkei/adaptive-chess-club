"""
In-process per-key async locks.

For single-worker deployments. When we move to multi-worker, swap the
implementation to Redis-backed (redis-py asyncio + SET NX EX pattern)
without changing the call sites.

Memory note: registries grow monotonically — one Lock per unique
session/match id, never freed. At current scale (dozens of sessions) this
is negligible. At scale, add TTL eviction (e.g. a WeakValueDictionary or a
periodic sweep keyed on last-access time). Do not fix now.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator

_chat_session_locks: dict[str, asyncio.Lock] = {}
_match_locks: dict[str, asyncio.Lock] = {}


def _get_lock(registry: dict[str, asyncio.Lock], key: str) -> asyncio.Lock:
    lock = registry.get(key)
    if lock is None:
        lock = asyncio.Lock()
        registry[key] = lock
    return lock


@asynccontextmanager
async def chat_session_lock(session_id: str) -> AsyncIterator[None]:
    """Serialize background work for a single chat session."""
    lock = _get_lock(_chat_session_locks, session_id)
    async with lock:
        yield


@asynccontextmanager
async def match_lock(match_id: str) -> AsyncIterator[None]:
    """Serialize background work for a single match (e.g., opening move)."""
    lock = _get_lock(_match_locks, match_id)
    async with lock:
        yield
