"""Opt-in test that hits the real Gemini API.

Only runs when BOTH:
  - env var RUN_LIVE_LLM_TESTS=1
  - a real GEMINI_API_KEY is present (not the test placeholder)

Run it with:
    RUN_LIVE_LLM_TESTS=1 GEMINI_API_KEY=... pytest -m live
"""

from __future__ import annotations

import os

import pytest
from sqlalchemy import select

from app.characters.memory_generator import generate_and_store
from app.config import get_settings
from app.db import SessionLocal
from app.models.character import Character, CharacterState
from app.models.memory import Memory, MemoryScope, MemoryType

pytestmark = pytest.mark.live


def _should_skip() -> tuple[bool, str]:
    if os.environ.get("RUN_LIVE_LLM_TESTS") != "1":
        return True, "RUN_LIVE_LLM_TESTS not set"
    key = os.environ.get("GEMINI_API_KEY", "")
    if not key or key == "test-key-not-real":
        return True, "real GEMINI_API_KEY not present"
    return False, ""


def test_live_generation_produces_valid_memories():
    skip, reason = _should_skip()
    if skip:
        pytest.skip(reason)

    # The lru_cache in get_settings + get_llm_client may be holding the
    # test-time API key. Refresh them against the current environment.
    from app.config import get_settings as _s
    from app.llm.client import get_llm_client as _c

    _s.cache_clear()
    _c.cache_clear()

    settings = get_settings()

    with SessionLocal() as s:
        char = Character(
            name="Anja Berg",
            short_description="A quiet Norwegian endgame specialist",
            backstory=(
                "Grew up in Trondheim. Learned from her uncle, a retired engineer "
                "who played correspondence chess. Spent her twenties studying Capablanca games."
            ),
            aggression=3,
            risk_tolerance=4,
            patience=9,
            trash_talk=2,
            target_elo=2200,
            adaptive=False,
            opening_preferences=["Catalan Opening", "English Opening"],
            voice_descriptor="calm, precise Scandinavian English",
            quirks="writes notes in tiny handwriting between games",
            state=CharacterState.GENERATING_MEMORIES,
            is_preset=False,
        )
        s.add(char)
        s.commit()
        character_id = char.id

    count = generate_and_store(character_id)
    assert settings.memory_gen_min // 2 <= count <= settings.memory_gen_max + 10

    with SessionLocal() as s:
        char = s.get(Character, character_id)
        assert char.state == CharacterState.READY
        rows = list(s.execute(select(Memory).where(Memory.character_id == character_id)).scalars())

    assert len(rows) == count
    for m in rows:
        assert isinstance(m.scope, MemoryScope)
        assert isinstance(m.type, MemoryType)
        assert -1.0 <= m.emotional_valence <= 1.0
        assert m.narrative_text and len(m.narrative_text) >= 20
        assert isinstance(m.triggers, list) and m.triggers
        assert isinstance(m.relevance_tags, list) and m.relevance_tags
