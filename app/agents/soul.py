"""The Soul agent.

Called every character turn AFTER the engine move AND after the
Subconscious has returned surfaced memories. Produces a `SoulResponse`:
chat (nullable), emotion + intensity, mood deltas, opponent note,
referenced memory ids.

The Soul always runs — "silent move" is a valid output. This keeps cost
predictable and lets the LLM judge whether each move is worth speaking
about with full context (rather than a pre-gate coin flip).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from pydantic import TypeAdapter

from app.agents.prompts import build_system_prompt, build_user_prompt
from app.director.mood import MoodState
from app.engine.board_abstraction import BoardSummary
from app.llm.client import LLMClient, LLMError, get_llm_client
from app.models.character import Character
from app.schemas.agents import MoodDeltas, SoulResponse, SurfacedMemory

logger = logging.getLogger(__name__)

_SOUL_ADAPTER = TypeAdapter(SoulResponse)


@dataclass
class SoulInput:
    board: BoardSummary
    mood: MoodState
    surfaced_memories: list[SurfacedMemory]
    recent_chat: list[str]
    engine_move_san: str
    engine_move_uci: str
    engine_eval_cp: int | None
    engine_considered: list[dict[str, Any]] | None
    engine_time_ms: int | None
    move_number: int
    game_phase: str
    opponent_profile_summary: dict | None = None
    head_to_head: dict[str, int] | None = None
    player_just_spoke: bool = False
    last_player_chat: str | None = None
    match_id: str = ""


def _fallback_response() -> SoulResponse:
    """Silent, neutral response used when the LLM is unavailable or errors out."""
    return SoulResponse(
        speak=None,
        emotion="neutral",
        emotion_intensity=0.2,
        mood_deltas=MoodDeltas(),
        note_about_opponent=None,
        referenced_memory_ids=[],
        internal_thinking="llm unavailable; silent fallback",
    )


def _sanitize(resp: SoulResponse, surfaced: list[SurfacedMemory]) -> SoulResponse:
    """Enforce the invariant that referenced_memory_ids is a subset of surfaced set.

    The LLM occasionally hallucinates IDs; we drop any that aren't in the
    surfaced list. (Structured-output validation catches shape errors,
    not this semantic constraint.)
    """
    surfaced_ids = {m.memory_id for m in surfaced}
    cleaned = [mid for mid in resp.referenced_memory_ids if mid in surfaced_ids]
    if cleaned != list(resp.referenced_memory_ids):
        dropped = set(resp.referenced_memory_ids) - surfaced_ids
        logger.debug("Soul dropped %d hallucinated memory ids: %s", len(dropped), dropped)
        return resp.model_copy(update={"referenced_memory_ids": cleaned})
    return resp


def run_soul(
    character: Character,
    inp: SoulInput,
    *,
    llm: LLMClient | None = None,
) -> SoulResponse:
    """Call the LLM and return a validated (and sanitized) SoulResponse.

    On LLM failure, returns a silent neutral fallback so gameplay never
    stalls.
    """
    system = build_system_prompt(character)
    user = build_user_prompt(
        character=character,
        board=inp.board,
        mood=inp.mood,
        surfaced_memories=inp.surfaced_memories,
        recent_chat=inp.recent_chat,
        engine_move_san=inp.engine_move_san,
        engine_move_uci=inp.engine_move_uci,
        engine_eval_cp=inp.engine_eval_cp,
        engine_considered=inp.engine_considered,
        engine_time_ms=inp.engine_time_ms,
        move_number=inp.move_number,
        game_phase=inp.game_phase,
        opponent_profile_summary=inp.opponent_profile_summary,
        head_to_head=inp.head_to_head,
        player_just_spoke=inp.player_just_spoke,
        last_player_chat=inp.last_player_chat,
    )
    prompt = f"{system}\n\n---\n\n{user}"

    client = llm
    if client is None:
        try:
            client = get_llm_client()
        except LLMError as exc:
            logger.warning("Soul: LLM unavailable (%s); returning silent fallback", exc)
            return _fallback_response()

    try:
        raw = client.generate_structured(
            prompt=prompt,
            response_schema=SoulResponse,
            response_adapter=_SOUL_ADAPTER,
            temperature=0.85,
            max_output_tokens=800,
            call_tag=f"soul:{inp.match_id}",
        )
    except LLMError as exc:
        logger.warning("Soul LLM call failed (%s); returning silent fallback", exc)
        return _fallback_response()
    except Exception as exc:
        logger.exception("Soul unexpected error (%s); returning silent fallback", exc)
        return _fallback_response()

    if not isinstance(raw, SoulResponse):
        # In case the adapter fell back to a dict.
        try:
            raw = _SOUL_ADAPTER.validate_python(raw)
        except Exception as exc:
            logger.warning("Soul output failed validation (%s); silent fallback", exc)
            return _fallback_response()

    return _sanitize(raw, inp.surfaced_memories)
