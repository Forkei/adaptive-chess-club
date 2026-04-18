from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.models.memory import MemoryScope, MemoryType
from app.schemas.memory import MemoryCreate


def _valid_payload(**overrides) -> dict:
    payload: dict = dict(
        scope=MemoryScope.CHARACTER_LORE,
        type=MemoryType.FORMATIVE,
        emotional_valence=0.0,
        triggers=["grandfather", "Minsk"],
        narrative_text="I remember my grandfather teaching me in a cold room.",
        relevance_tags=["childhood"],
    )
    payload.update(overrides)
    return payload


def test_valid_payload_parses():
    m = MemoryCreate(**_valid_payload())
    assert m.narrative_text.startswith("I remember")
    assert m.scope == MemoryScope.CHARACTER_LORE
    assert m.type == MemoryType.FORMATIVE


def test_rejects_out_of_range_valence():
    with pytest.raises(ValidationError):
        MemoryCreate(**_valid_payload(emotional_valence=1.5))
    with pytest.raises(ValidationError):
        MemoryCreate(**_valid_payload(emotional_valence=-1.5))


def test_rejects_invalid_enum():
    with pytest.raises(ValidationError):
        MemoryCreate(**_valid_payload(scope="not_a_scope"))
    with pytest.raises(ValidationError):
        MemoryCreate(**_valid_payload(type="not_a_type"))


def test_rejects_empty_narrative():
    with pytest.raises(ValidationError):
        MemoryCreate(**_valid_payload(narrative_text=""))


def test_triggers_stripped_and_deduped():
    m = MemoryCreate(**_valid_payload(triggers=["  Sicilian ", "sicilian", "Najdorf", ""]))
    assert m.triggers == ["Sicilian", "Najdorf"]


def test_accepts_enum_values_as_strings():
    m = MemoryCreate(**_valid_payload(scope="character_lore", type="opinion"))
    assert m.scope == MemoryScope.CHARACTER_LORE
    assert m.type == MemoryType.OPINION
