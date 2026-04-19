"""Opt-in live match end-to-end test.

Only runs when BOTH:
  - env var RUN_LIVE_LLM_TESTS=1
  - a real GEMINI_API_KEY is set (not the conftest placeholder)

Exercises the full Slice 1 + Slice 2 + Slice 3 pipeline against a real
Gemini call and whichever engine is locally available. For the Slice 3
voice vibe-check, run `scripts/live_match_demo.py` which captures a
richer markdown report.

Run it with:
    RUN_LIVE_LLM_TESTS=1 pytest -m live tests/test_live_match.py
"""

from __future__ import annotations

import asyncio
import os

import pytest

pytestmark = pytest.mark.live


def _should_skip() -> tuple[bool, str]:
    if os.environ.get("RUN_LIVE_LLM_TESTS") != "1":
        return True, "RUN_LIVE_LLM_TESTS not set"
    key = os.environ.get("GEMINI_API_KEY", "")
    if not key or key == "test-key-not-real":
        return True, "real GEMINI_API_KEY not present"
    return False, ""


def test_live_match_end_to_end():
    skip, reason = _should_skip()
    if skip:
        pytest.skip(reason)

    from scripts.live_match_demo import PRESET_KEYS, _ensure_character, run_live_match

    # Archibald: preset Elo 1600, non-adaptive. Reasonable choice for vibe check.
    character_id = _ensure_character(PRESET_KEYS["archibald"])
    result = asyncio.run(run_live_match(character_id, max_moves=10, rng_seed=11))

    # Completion + post-match ran.
    assert result.final_result
    # At least one new memory generated OR narrative summary populated.
    assert result.generated_memories or result.narrative_summary, (
        "post-match produced nothing — LLM outage or prompt failure?"
    )
    # Elo computation wired up (delta may be 0 for non-adaptive characters).
    assert "outcome" in result.elo_breakdown
    # At least one turn had a Soul response (even if silent).
    assert len(result.turns) >= 1
