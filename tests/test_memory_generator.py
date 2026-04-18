from __future__ import annotations

from sqlalchemy import select

from app.characters.memory_generator import GeneratedMemory, build_prompt, generate_and_store
from app.db import SessionLocal
from app.models.character import Character, CharacterState
from app.models.memory import Memory, MemoryScope, MemoryType


class FakeLLMClient:
    """Duck-typed stand-in for `LLMClient` used in tests."""

    def __init__(self, memories: list[GeneratedMemory] | Exception):
        self._result = memories

    def generate_structured(self, **kwargs):
        if isinstance(self._result, Exception):
            raise self._result
        return self._result


def _make_character(session) -> str:
    char = Character(
        name="Test Character",
        short_description="A test",
        backstory="Born in a test. Lived in a test. Died in a test.",
        aggression=6,
        risk_tolerance=6,
        patience=5,
        trash_talk=4,
        target_elo=1600,
        adaptive=False,
        opening_preferences=["Ruy Lopez", "Sicilian Najdorf"],
        voice_descriptor="test voice",
        quirks="never blinks",
        state=CharacterState.GENERATING_MEMORIES,
    )
    session.add(char)
    session.commit()
    return char.id


def _fake_memories(n: int = 30) -> list[GeneratedMemory]:
    out: list[GeneratedMemory] = []
    types = list(MemoryType)
    for i in range(n):
        out.append(
            GeneratedMemory(
                scope=MemoryScope.CHARACTER_LORE,
                type=types[i % len(types)],
                emotional_valence=(i % 5 - 2) / 2.0,  # spread -1..1
                triggers=[f"trigger_{i}", f"alt_{i}"],
                narrative_text=f"This is the {i}-th fake memory. It has enough length to pass validation.",
                relevance_tags=[f"tag_{i % 3}"],
            )
        )
    return out


def test_build_prompt_contains_character_details():
    char = Character(
        name="Vera",
        short_description="quiet",
        backstory="a long backstory here",
        aggression=9,
        risk_tolerance=2,
        patience=3,
        trash_talk=1,
        target_elo=2000,
        adaptive=True,
        opening_preferences=["Ruy Lopez"],
        voice_descriptor="calm",
        quirks="hums",
    )
    prompt = build_prompt(char, target=40, minimum=30, maximum=50)
    assert "Vera" in prompt
    assert "a long backstory here" in prompt
    assert "Ruy Lopez" in prompt
    assert "2000" in prompt
    assert "adapts" in prompt.lower()  # adaptive line


def test_generate_and_store_persists_and_marks_ready():
    with SessionLocal() as s:
        character_id = _make_character(s)

    fake = FakeLLMClient(_fake_memories(30))
    count = generate_and_store(character_id, client=fake)
    assert count == 30

    with SessionLocal() as s:
        char = s.get(Character, character_id)
        assert char is not None
        assert char.state == CharacterState.READY
        assert char.memory_generation_error is None

        memories = list(s.execute(select(Memory).where(Memory.character_id == character_id)).scalars())
        assert len(memories) == 30
        # Spread check — we should see multiple distinct types.
        assert len({m.type for m in memories}) >= 4


def test_generate_and_store_failure_marks_character_failed():
    with SessionLocal() as s:
        character_id = _make_character(s)

    fake = FakeLLMClient(RuntimeError("fake Gemini outage"))
    import pytest

    with pytest.raises(RuntimeError):
        generate_and_store(character_id, client=fake)

    with SessionLocal() as s:
        char = s.get(Character, character_id)
        assert char is not None
        assert char.state == CharacterState.GENERATION_FAILED
        assert char.memory_generation_error is not None
        assert "fake Gemini outage" in char.memory_generation_error


def test_generate_and_store_rejects_too_few():
    """If the LLM returns a suspiciously tiny batch, we fail the character."""
    with SessionLocal() as s:
        character_id = _make_character(s)

    fake = FakeLLMClient(_fake_memories(3))  # below min//2
    import pytest

    with pytest.raises(Exception):
        generate_and_store(character_id, client=fake)

    with SessionLocal() as s:
        char = s.get(Character, character_id)
        assert char.state == CharacterState.GENERATION_FAILED
