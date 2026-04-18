"""JSONL call log for LLM interactions.

Separate from the app logger because these records are structured and
used for debugging prompt quality, cost, and latency — not for ops logs.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from app.config import get_settings

_LOCK = threading.Lock()


def _log_path() -> Path:
    return get_settings().log_path / "llm_calls.jsonl"


def log_call(record: dict[str, Any]) -> None:
    """Append one JSON line to the LLM call log. Best-effort; never raises."""
    try:
        payload = {"timestamp": datetime.utcnow().isoformat() + "Z", **record}
        line = json.dumps(payload, default=str, ensure_ascii=False)
        path = _log_path()
        with _LOCK:
            with path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
    except Exception:
        # Logging must never break a live call. Swallow silently.
        pass
