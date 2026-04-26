"""Prompt-injection sanitizer for player-supplied personality descriptions.

The personality text is embedded verbatim into an LLM system prompt.
This module strips lines that look like injection attempts before
the text is stored in the DB. It is not foolproof — a determined
adversary can phrase injections to evade pattern matching — but it
raises the bar materially.

Two-tier approach:
  1. Hard-drop prefixes that are almost never legitimate in personality
     descriptions (e.g. "you are", "act as", "system:").
  2. Soft-drop prefixes that are sometimes legitimate (e.g. "ignore") but
     only when the line also contains injection-specific keywords such as
     "instructions", "previous", "prompt", "constraints".

This correctly handles the false-positive case:
  "Ignore the queen sacrifice trap"  → KEPT  (no injection keyword)
  "Ignore all previous instructions" → DROPPED ("previous" + "instructions")
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Hard drops — any line starting with these is removed unconditionally.
_HARD_DROP_STARTS: tuple[str, ...] = (
    "you are",
    "you're",
    "system:",
    "system prompt",
    "new instruction",
    "new task",
    "act as",
    "pretend to be",
    "roleplay as",
    "the player is",
    "the user is",
)

# Soft-drop candidates — only flagged when combined with an injection keyword.
_SOFT_DROP_STARTS: tuple[str, ...] = (
    "ignore",
    "disregard",
    "forget",
    "instead",
)

# If any of these words appear in a soft-drop candidate line, it's dropped.
_INJECTION_KEYWORDS: tuple[str, ...] = (
    "instruction",
    "previous",
    "prior",
    "above",
    "system prompt",
    "prompt",
    "constraint",
    "forget everything",
    "disregard all",
    "new rule",
    "new task",
)


def sanitize_personality(raw: str) -> str:
    """Return the personality text with injection-attempt lines removed.

    Logs a warning when lines are dropped so ops can review patterns.
    """
    lines = raw.strip().split("\n")
    cleaned: list[str] = []
    dropped: list[str] = []

    for line in lines:
        stripped = line.strip().lower()

        # Hard drop.
        if any(stripped.startswith(prefix) for prefix in _HARD_DROP_STARTS):
            dropped.append(line.strip()[:80])
            continue

        # Soft drop: only when injection keywords are also present.
        if any(stripped.startswith(prefix) for prefix in _SOFT_DROP_STARTS):
            if any(kw in stripped for kw in _INJECTION_KEYWORDS):
                dropped.append(line.strip()[:80])
                continue

        cleaned.append(line)

    if dropped:
        logger.warning(
            "personality_sanitizer: dropped %d suspicious line(s): %s",
            len(dropped),
            dropped,
        )

    return "\n".join(cleaned).strip()
