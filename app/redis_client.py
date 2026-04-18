"""Minimal Redis KV wrapper with JSON values + TTL.

Used in 2a for mood state and in 2b for recent-moves caches. If
`REDIS_URL` is unset or the connection fails, falls back to a thread-safe
in-process dict so local dev without Docker still works (with a warning).
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from app.config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class _MemoryValue:
    payload: Any
    expires_at: float | None  # epoch seconds, None = never


@dataclass
class _InMemoryStore:
    values: dict[str, _MemoryValue] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def get(self, key: str) -> Any | None:
        with self.lock:
            v = self.values.get(key)
            if v is None:
                return None
            if v.expires_at is not None and v.expires_at < time.time():
                del self.values[key]
                return None
            return v.payload

    def set(self, key: str, value: Any, ttl_s: int | None) -> None:
        expires = time.time() + ttl_s if ttl_s else None
        with self.lock:
            self.values[key] = _MemoryValue(payload=value, expires_at=expires)

    def delete(self, key: str) -> None:
        with self.lock:
            self.values.pop(key, None)

    def clear(self) -> None:
        with self.lock:
            self.values.clear()


_MEMORY_STORE = _InMemoryStore()
_REAL_CLIENT: Any = None
_CLIENT_LOCK = threading.Lock()
_WARNED = False


def _get_real_client() -> Any | None:
    global _REAL_CLIENT, _WARNED
    settings = get_settings()
    url = getattr(settings, "redis_url", "") or ""
    if not url:
        if not _WARNED:
            logger.warning(
                "REDIS_URL is unset — mood/cache state falls back to in-process memory. "
                "This is fine for dev, not for multi-process deployment."
            )
            _WARNED = True
        return None
    if _REAL_CLIENT is not None:
        return _REAL_CLIENT
    with _CLIENT_LOCK:
        if _REAL_CLIENT is not None:
            return _REAL_CLIENT
        try:
            import redis  # type: ignore

            client = redis.Redis.from_url(url, decode_responses=True)
            client.ping()
            _REAL_CLIENT = client
            logger.info("Connected to Redis at %s", url)
            return client
        except Exception as exc:
            if not _WARNED:
                logger.warning("Redis unavailable (%s); using in-memory fallback", exc)
                _WARNED = True
            return None


def get(key: str) -> Any | None:
    client = _get_real_client()
    if client is None:
        return _MEMORY_STORE.get(key)
    try:
        raw = client.get(key)
    except Exception as exc:
        logger.warning("Redis GET failed (%s); falling back to memory for this call", exc)
        return _MEMORY_STORE.get(key)
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def set_(key: str, value: Any, *, ttl_s: int | None = None) -> None:
    payload = json.dumps(value, default=str)
    client = _get_real_client()
    if client is None:
        _MEMORY_STORE.set(key, value, ttl_s)
        return
    try:
        if ttl_s:
            client.set(key, payload, ex=ttl_s)
        else:
            client.set(key, payload)
    except Exception as exc:
        logger.warning("Redis SET failed (%s); falling back to memory for this call", exc)
        _MEMORY_STORE.set(key, value, ttl_s)


def delete(key: str) -> None:
    client = _get_real_client()
    if client is None:
        _MEMORY_STORE.delete(key)
        return
    try:
        client.delete(key)
    except Exception:
        _MEMORY_STORE.delete(key)


def reset_memory_store_for_testing() -> None:  # pragma: no cover — test helper
    _MEMORY_STORE.clear()
