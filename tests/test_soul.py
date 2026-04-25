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
        self.last_prompt: str | None = None

    def generate_structured(self, **kwargs):
        self.last_prompt = kwargs.get("prompt")
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


# --- Prompt content regression: chess-authority rule ----------------------


def test_system_prompt_contains_hard_move_prediction_rule():
    """The 'HARD RULE: NEVER PREDICT YOUR OWN MOVES' block must be present and
    must appear BEFORE the CHARACTER SHEET so the LLM reads it first."""
    from app.agents.prompts import build_system_prompt

    prompt = build_system_prompt(_char())

    assert "HARD RULE: NEVER PREDICT YOUR OWN MOVES" in prompt
    # Positive reinforcement phrasing must be present.
    assert "You are not the engine" in prompt
    assert "deflect" in prompt
    # The rule should appear before the character sheet.
    assert prompt.index("HARD RULE") < prompt.index("CHARACTER SHEET")


def test_soul_prompt_includes_timing_when_supplied():
    """Patch Pass 2 Item 4: timing data must reach the prompt with the
    accuse-slow threshold guidance attached."""
    surfaced = _surfaced([])
    inp = SoulInput(
        board=_soul_input(surfaced).board,
        mood=MoodState(aggression=0.5, confidence=0.5, tilt=0.0, engagement=0.5),
        surfaced_memories=surfaced,
        recent_chat=[],
        engine_move_san="Nf3",
        engine_move_uci="g1f3",
        engine_eval_cp=10,
        engine_considered=[],
        engine_time_ms=200,
        move_number=5,
        game_phase="opening",
        match_id="m-timing",
        character_color="white",
        opponent_last_san="e5",
        opponent_last_uci="e7e5",
        player_took_seconds=12.3,
        player_average_seconds=8.7,
        elapsed_total_seconds=420.0,
    )
    resp = SoulResponse(speak=None, emotion="neutral", emotion_intensity=0.2)
    fake = _FakeLLMReturning(resp)
    run_soul(_char(), inp, llm=fake)

    assert fake.last_prompt is not None
    prompt = fake.last_prompt
    assert "=== TIMING ===" in prompt
    assert "PLAYER TOOK: 12.3 seconds" in prompt
    assert "PLAYER'S AVERAGE MOVE TIME SO FAR: 8.7 seconds" in prompt
    assert "ELAPSED TOTAL MATCH TIME: 7.0 minutes" in prompt
    assert "Do not accuse the player of being slow" in prompt


def test_soul_prompt_omits_timing_block_when_no_data():
    surfaced = _surfaced([])
    inp = _soul_input(surfaced)
    resp = SoulResponse(speak=None, emotion="neutral", emotion_intensity=0.2)
    fake = _FakeLLMReturning(resp)
    run_soul(_char(), inp, llm=fake)
    assert fake.last_prompt is not None
    # No timing section when no data supplied.
    assert "=== TIMING ===" not in fake.last_prompt


def test_soul_prompt_labels_engine_move_and_player_move_unambiguously():
    """Regression for the Ng1 misattribution bug.

    The Soul prompt must make it impossible to confuse the engine's move
    (the character's own move) with the opponent's last move. Both must
    be labeled, with color reminders in multiple places.
    """
    surfaced = _surfaced([])
    inp = SoulInput(
        board=_soul_input(surfaced).board,
        mood=MoodState(aggression=0.5, confidence=0.5, tilt=0.0, engagement=0.5),
        surfaced_memories=surfaced,
        recent_chat=[],
        engine_move_san="Ng1",
        engine_move_uci="f3g1",
        engine_eval_cp=-20,
        engine_considered=[],
        engine_time_ms=150,
        move_number=8,
        game_phase="opening",
        match_id="m-1",
        character_color="white",
        opponent_last_san="Nc6",
        opponent_last_uci="b8c6",
    )
    resp = SoulResponse(speak=None, emotion="neutral", emotion_intensity=0.2)
    fake = _FakeLLMReturning(resp)
    run_soul(_char(), inp, llm=fake)

    assert fake.last_prompt is not None
    prompt = fake.last_prompt
    # Engine move is unambiguously attributed to the character.
    assert "YOUR OWN MOVE YOU JUST PLAYED: Ng1" in prompt
    assert "you are playing white" in prompt
    # Player move is unambiguously attributed to the opponent.
    assert "OPPONENT'S LAST MOVE: Nc6" in prompt
    assert "they are playing black" in prompt
    # No neutral phrasing that could let the model conflate sides.
    assert "Your move just played" not in prompt  # old ambiguous phrasing gone
    # Color reminders appear multiple times (header + per-section + instructions).
    assert prompt.count("white") >= 2
    assert prompt.count("black") >= 2


# --- Regression: game_action start_game gating (Issue 2) -------------------


def _pre_match_soul_input(player_text: str) -> SoulInput:
    """A SoulInput shaped like the pre-match room pipeline."""
    board = chess.Board()
    return SoulInput(
        board=board_to_english(board),
        mood=MoodState(aggression=0.6, confidence=0.5, tilt=0.1, engagement=0.7),
        surfaced_memories=[],
        recent_chat=[f"player: {player_text}"],
        engine_move_san="",
        engine_move_uci="",
        engine_eval_cp=None,
        engine_considered=None,
        engine_time_ms=None,
        move_number=0,
        game_phase="pre-match",
        player_just_spoke=True,
        last_player_chat=player_text,
        match_id="chat:test-session",
        character_color="white",
    )


def test_game_action_prompt_forbids_vague_trigger():
    """The system prompt must explicitly prohibit start_game for questions
    and small talk. If it doesn't, the LLM will start games on 'what's my rating?'.
    """
    from app.agents.prompts import build_system_prompt

    system = build_system_prompt(_char())
    assert "Do NOT emit start_game when" in system or "Do NOT use start_game" in system, (
        "System prompt must contain explicit start_game guard for questions/small talk"
    )
    assert "what's my rating" in system.lower() or "asking a question" in system.lower(), (
        "System prompt must list question-asking as an example that does NOT trigger start_game"
    )
    assert "small talk" in system.lower() or "greeting" in system.lower(), (
        "System prompt must list small talk / greetings as non-triggers for start_game"
    )


def test_game_action_not_started_when_fake_llm_obeys_rules():
    """Integration: if the LLM obeys the rules (returns 'none' for a question),
    run_soul must pass that through unchanged."""
    resp = SoulResponse(
        speak="I don't know your rating, but I'll find out on the board.",
        emotion="focused",
        emotion_intensity=0.4,
        game_action="none",
    )
    fake = _FakeLLMReturning(resp)
    out = run_soul(_char(), _pre_match_soul_input("what's my rating?"), llm=fake)
    assert out.game_action == "none", (
        "Soul must not escalate game_action beyond what the LLM returned"
    )


def test_game_action_prompt_contains_explicit_examples():
    """The prompt must contain concrete do-not-trigger examples so the LLM
    has clear guidance, not just a vague rule."""
    from app.agents.prompts import build_system_prompt

    system = build_system_prompt(_char())
    # Must list at least one explicit readiness phrase (positive case).
    assert any(phrase in system for phrase in ("let's play", "let's go", "I'm ready", "ready")), (
        "System prompt must list explicit player phrases that DO trigger start_game"
    )
    # Must forbid the impatience pattern that triggered the bug.
    assert "impatient" in system.lower() or "bored" in system.lower() or "wanting to push" in system.lower() or "push" in system.lower(), (
        "System prompt must explicitly say the character's impatience is not a valid start_game trigger"
    )


# --- Regression: no chess move claims (Issue 3) ----------------------------


def test_chess_authority_rules_in_system_prompt():
    """The system prompt must contain the move-claim prohibition so the LLM
    knows it cannot predict or announce specific future moves."""
    from app.agents.prompts import build_system_prompt

    system = build_system_prompt(_char())
    assert "King's Gambit" in system or "I'm opening with" in system, (
        "System prompt must include a concrete forbidden-example like 'I'm opening with the King's Gambit'"
    )
    assert "engine" in system.lower() and ("pick" in system.lower() or "choos" in system.lower() or "decid" in system.lower()), (
        "System prompt must explain that the ENGINE chooses moves, not the Soul"
    )
    assert (
        "DO NOT say" in system
        or "Do not say" in system
        or "don't say" in system.lower()
        or "FORBIDDEN" in system
        or "never say" in system.lower()
    ), "System prompt must contain a clear prohibition on move-claim phrases"


def test_chess_authority_rule_allows_style_claims():
    """The prohibition must be surgical: general style claims ('I like aggressive play')
    are fine. The system prompt should distinguish style from specific move predictions."""
    from app.agents.prompts import build_system_prompt

    system = build_system_prompt(_char())
    # The rules should allow style/tendency language.
    assert "style" in system.lower() or "tendency" in system.lower() or "general" in system.lower(), (
        "System prompt should clarify that style claims are allowed (just not specific move predictions)"
    )
