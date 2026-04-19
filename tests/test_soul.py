"""Soul agent tests.

Covers: schema validation, referenced_memory_ids sanitization (must be
subset of surfaced set), silent fallback on LLM failure.
"""

from __future__ import annotations

from typing import Any

import chess
import pytest

from app.agents.soul import SoulInput, run_soul
from app.director.mood import MoodState
from app.engine.board_abstraction import board_to_english
from app.llm.client import LLMError
from app.models.character import Character, CharacterState
from app.schemas.agents import MoodDeltas, SoulResponse, SurfacedMemory


def _char() -> Character:
    return Character(
        id="char-1",
        name="Tester",
        short_description="t",
        backstory="once upon a time",
        voice_descriptor="calm and dry",
        quirks="taps pocket watch",
        aggression=5,
        risk_tolerance=5,
        patience=5,
        trash_talk=5,
        target_elo=1600,
        current_elo=1600,
        floor_elo=1600,
        max_elo=1800,
        state=CharacterState.READY,
    )


def _surfaced(ids: list[str]) -> list[SurfacedMemory]:
    return [
        SurfacedMemory(
            memory_id=mid,
            narrative_text=f"memory content for {mid}",
            triggers=["foo"],
            relevance_tags=["bar"],
            emotional_valence=0.1,
            scope="character_lore",
            score=0.5,
            retrieval_reason=f"reason-{mid}",
            from_cache=False,
        )
        for mid in ids
    ]


def _soul_input(memories: list[SurfacedMemory]) -> SoulInput:
    board = chess.Board()
    board.push_san("e4")
    return SoulInput(
        board=board_to_english(board),
        mood=MoodState(aggression=0.5, confidence=0.5, tilt=0.0, engagement=0.5),
        surfaced_memories=memories,
        recent_chat=[],
        engine_move_san="e4",
        engine_move_uci="e2e4",
        engine_eval_cp=20,
        engine_considered=[],
        engine_time_ms=200,
        move_number=1,
        game_phase="opening",
        match_id="m-1",
    )


class _FakeLLMReturning:
    def __init__(self, resp: SoulResponse | Exception):
        self._resp = resp

    def generate_structured(self, **kwargs):
        if isinstance(self._resp, Exception):
            raise self._resp
        return self._resp


def test_soul_passes_through_valid_response():
    surfaced = _surfaced(["m1", "m2"])
    resp = SoulResponse(
        speak="checking",
        emotion="focused",
        emotion_intensity=0.4,
        mood_deltas=MoodDeltas(confidence=0.05),
        referenced_memory_ids=["m1"],
    )
    out = run_soul(_char(), _soul_input(surfaced), llm=_FakeLLMReturning(resp))
    assert out.speak == "checking"
    assert out.emotion == "focused"
    assert out.referenced_memory_ids == ["m1"]


def test_soul_drops_hallucinated_memory_ids():
    surfaced = _surfaced(["m1", "m2"])
    resp = SoulResponse(
        speak=None,
        emotion="neutral",
        emotion_intensity=0.2,
        referenced_memory_ids=["m1", "not-surfaced", "also-hallucinated"],
    )
    out = run_soul(_char(), _soul_input(surfaced), llm=_FakeLLMReturning(resp))
    assert out.referenced_memory_ids == ["m1"]


def test_soul_falls_back_silent_on_llm_error():
    surfaced = _surfaced(["m1"])
    fake = _FakeLLMReturning(LLMError("boom"))
    out = run_soul(_char(), _soul_input(surfaced), llm=fake)
    assert out.speak is None
    assert out.emotion == "neutral"
    assert out.mood_deltas.to_dict() == {
        "aggression": 0.0,
        "confidence": 0.0,
        "tilt": 0.0,
        "engagement": 0.0,
    }
    assert out.referenced_memory_ids == []


def test_soul_mood_deltas_clamp_at_pm_0_1():
    # Pydantic validation should reject deltas outside ±0.1 per axis.
    with pytest.raises(Exception):
        MoodDeltas(aggression=0.2)
    with pytest.raises(Exception):
        MoodDeltas(tilt=-0.15)


def test_soul_response_accepts_silent_output():
    r = SoulResponse()
    assert r.speak is None
    assert r.emotion == "neutral"
    assert r.referenced_memory_ids == []
