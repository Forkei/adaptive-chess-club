"""Post-match memory generation + narrative summary update.

Two separate LLM calls, separate models:

1. `generate_match_memories`: 1-3 new memories scoped MATCH_RECAP or
   OPPONENT_SPECIFIC, derived from character voice + match summary +
   critical moments + opponent profile delta + rage-quit flag + queued
   opponent notes. Validates each memory has ≥3 triggers; retries once
   with explicit feedback if any are weak.

2. `update_narrative_summary`: LLM rewrites
   OpponentProfile.narrative_summary in ≤3 first-person sentences given
   the updated features and last match context.
"""

from __future__ import annotations

import logging
import random
from typing import Any

from pydantic import BaseModel, Field, TypeAdapter

from app.characters.content_rating_prompts import rating_prompt_fragment
from app.characters.style import style_to_prompt_fragments
from app.llm.client import LLMClient, LLMError
from app.models.character import Character
from app.models.match import Match, MatchResult, MatchStatus
from app.models.memory import Memory, MemoryScope, MemoryType
from app.schemas.memory import MemoryCreate

logger = logging.getLogger(__name__)

MIN_TRIGGERS_PER_MEMORY = 3
MEMORY_RETRY_LIMIT = 1       # one retry if validation fails


# --- LLM schema ------------------------------------------------------------


class _MatchMemory(BaseModel):
    """Shape the LLM must return per memory."""

    scope: MemoryScope = Field(
        description="Usually match_recap. Use opponent_specific when the memory is "
        "distinctively about this player's style or behavior."
    )
    type: MemoryType
    emotional_valence: float = Field(..., ge=-1.0, le=1.0)
    triggers: list[str] = Field(
        ...,
        description=(
            "At least 3, ideally 5-8. Cover: opening names, tactical themes "
            "('opposite-side castling', 'endgame bishop vs knight'), opponent "
            "traits mentioned ('plays fast', 'avoids captures'), board features, "
            "keywords the opponent might use in chat ('tactics', 'gambit'). "
            "Concrete and specific, not generic."
        ),
    )
    narrative_text: str = Field(
        ...,
        min_length=20,
        description="First-person memory in the character's voice. 2-4 sentences.",
    )
    relevance_tags: list[str] = Field(
        ...,
        description="2-6 chess-concept tags: 'endgame', 'kingside attack', 'time pressure'.",
    )


_MEMORY_LIST_ADAPTER = TypeAdapter(list[_MatchMemory])


class _NarrativeOut(BaseModel):
    summary: str = Field(
        ...,
        min_length=10,
        max_length=400,
        description="First-person summary of this opponent, at most 3 sentences. "
        "Should feel like a character jotting a note to themselves.",
    )


_NARRATIVE_ADAPTER = TypeAdapter(_NarrativeOut)


# --- Prompts ---------------------------------------------------------------


def _outcome_phrase(match: Match) -> str:
    if match.status == MatchStatus.ABANDONED:
        return "The player abandoned the match (rage quit — disconnected without coming back)."
    if match.status == MatchStatus.RESIGNED:
        # Character won via the player conceding — treat as a clean win, not a rage-quit.
        return "The player resigned, handing you the win. They chose to stop rather than play on."
    if match.result == MatchResult.DRAW:
        return "The match ended in a draw."
    if match.result is None:
        return "The match ended unresolved."
    # Character won iff result color matches the character's color.
    char_is_white = match.player_color.value == "black"
    char_won = (
        (match.result == MatchResult.WHITE_WIN and char_is_white)
        or (match.result == MatchResult.BLACK_WIN and not char_is_white)
    )
    return "You WON this match." if char_won else "You LOST this match."


def _sample_voice_examples(memories: list[Memory], k: int = 3) -> list[str]:
    """Return a small few-shot sample to anchor voice."""
    if not memories:
        return []
    pool = [m.narrative_text for m in memories if m.narrative_text]
    if not pool:
        return []
    random.shuffle(pool)
    return pool[:k]


def _build_memory_prompt(
    *,
    character: Character,
    match: Match,
    critical_moments: list[dict[str, Any]],
    features_before: dict[str, Any] | None,
    features_after: dict[str, Any],
    opponent_notes: list[dict[str, Any]],
    prior_memory_samples: list[str],
    retry_feedback: str | None = None,
) -> str:
    frags = style_to_prompt_fragments(character)
    rage_quit = match.status == MatchStatus.ABANDONED
    resigned = match.status == MatchStatus.RESIGNED
    outcome_line = _outcome_phrase(match)

    parts: list[str] = []
    parts.append(
        f"You are writing 1-3 new memories that {character.name} keeps from a match "
        f"they just finished. Memories must be in their voice — first-person, true to "
        f"how they speak and think. Each memory should be something a future version "
        f"of the character might actually recall during another game."
    )
    parts.append("")
    parts.append(rating_prompt_fragment(character.content_rating))
    parts.append("")
    parts.append(f"CHARACTER: {character.name}")
    parts.append(f"Voice / tone: {character.voice_descriptor or '(infer from backstory)'}")
    parts.append(
        f"Style: aggression={frags['aggression']}; risk_tolerance={frags['risk_tolerance']}; "
        f"patience={frags['patience']}; trash_talk={frags['trash_talk']}."
    )
    parts.append(f"Quirks: {character.quirks or '(none)'}")
    parts.append("")
    parts.append("MATCH OUTCOME")
    parts.append(outcome_line)
    parts.append(f"Move count (half-moves): {match.move_count}")
    if rage_quit:
        parts.append(
            "Because the player rage-quit, your memory should reflect the character's "
            "honest reaction — bitter, amused, or dismissive depending on who they are. "
            "Do not pretend it was a normal finish."
        )
    elif resigned:
        parts.append(
            "The player resigned — a clean concession, not a walk-out. Your memory should "
            "treat this as a legitimate finish. React authentically to winning that way "
            "(respectful, smug, bored, whatever fits the character), but don't frame them "
            "as a rage-quitter."
        )
    parts.append("")
    if critical_moments:
        parts.append("CRITICAL MOMENTS (from engine analysis)")
        for cm in critical_moments:
            parts.append(f"  - {cm.get('label', '')}")
        parts.append("")
    parts.append("OPPONENT STYLE (what we know about this specific player)")
    if features_before:
        parts.append(f"  Before this match: {_compact_features(features_before)}")
    else:
        parts.append("  Before this match: first time playing this opponent.")
    parts.append(f"  After this match: {_compact_features(features_after)}")
    parts.append("")
    if opponent_notes:
        parts.append("NOTES YOU THOUGHT DURING THE MATCH")
        for note in opponent_notes:
            txt = note.get("note") or note.get("text") or ""
            mv = note.get("move_number")
            parts.append(f"  - (move {mv}) {txt}")
        parts.append("")
    if prior_memory_samples:
        parts.append("SAMPLE OF YOUR EXISTING MEMORIES (for voice calibration)")
        for sample in prior_memory_samples:
            parts.append(f"  - \"{sample}\"")
        parts.append("")
    parts.append("WHAT TO PRODUCE")
    parts.append(
        f"Return a JSON array of 1-3 memories. DO NOT pad — if only one thing was "
        f"memorable, return only one. Each memory MUST have at least "
        f"{MIN_TRIGGERS_PER_MEMORY} triggers (ideally 5-8), rich enough for later "
        f"retrieval: opening names, tactical themes, opponent traits you noticed, "
        f"board features, keywords the player might use in future chat. Prefer "
        f"scope=match_recap, or opponent_specific when the memory is distinctively "
        f"about this player rather than the match events."
    )
    if retry_feedback:
        parts.append("")
        parts.append(f"RETRY FEEDBACK: {retry_feedback}")
    parts.append("")
    parts.append("Output ONLY the JSON array. No prose, no markdown fences.")
    return "\n".join(parts)


def _compact_features(f: dict[str, Any]) -> str:
    keys = (
        "typical_opening_name",
        "aggression_index",
        "blunder_rate",
        "phase_strengths",
    )
    parts = []
    for k in keys:
        if k in f:
            parts.append(f"{k}={f[k]}")
    return ", ".join(parts) if parts else str(f)


def _validate_triggers(memories: list[_MatchMemory]) -> list[str]:
    weak: list[str] = []
    for i, m in enumerate(memories):
        if len(m.triggers) < MIN_TRIGGERS_PER_MEMORY:
            weak.append(
                f"memory #{i + 1} has only {len(m.triggers)} triggers "
                f"(need ≥{MIN_TRIGGERS_PER_MEMORY})"
            )
    return weak


def generate_match_memories(
    *,
    character: Character,
    match: Match,
    critical_moments: list[dict[str, Any]],
    features_before: dict[str, Any] | None,
    features_after: dict[str, Any],
    opponent_notes: list[dict[str, Any]],
    prior_memories: list[Memory],
    llm: LLMClient,
) -> list[MemoryCreate]:
    """Run the LLM, validate, retry once if triggers are weak.

    Returns a list of MemoryCreate ready for `memory.crud.bulk_create`.
    Returns an empty list if both attempts fail — caller logs and moves on.
    """
    samples = _sample_voice_examples(prior_memories)

    for attempt in range(1 + MEMORY_RETRY_LIMIT):
        feedback: str | None = None
        if attempt > 0:
            feedback = (
                "The previous attempt had memories with too few triggers. "
                "Every memory must have at least 3 concrete, specific triggers. "
                "Opening names, positional themes, opponent behaviors, keywords "
                "the opponent might use in chat. Not generic words."
            )
        prompt = _build_memory_prompt(
            character=character,
            match=match,
            critical_moments=critical_moments,
            features_before=features_before,
            features_after=features_after,
            opponent_notes=opponent_notes,
            prior_memory_samples=samples,
            retry_feedback=feedback,
        )
        try:
            raw: list[_MatchMemory] = llm.generate_structured(
                prompt=prompt,
                response_schema=list[_MatchMemory],
                response_adapter=_MEMORY_LIST_ADAPTER,
                temperature=0.9,
                max_output_tokens=1600,
                call_tag=f"post_match_mem:{match.id}:attempt{attempt + 1}",
            )
        except LLMError as exc:
            logger.warning("Post-match memory LLM failed (attempt %d): %s", attempt + 1, exc)
            if attempt < MEMORY_RETRY_LIMIT:
                continue
            return []

        if not raw:
            logger.warning("Post-match memory LLM returned empty list (attempt %d)", attempt + 1)
            if attempt < MEMORY_RETRY_LIMIT:
                continue
            return []

        weak = _validate_triggers(raw)
        if weak:
            logger.info(
                "Post-match memory attempt %d had weak triggers: %s",
                attempt + 1,
                "; ".join(weak),
            )
            if attempt < MEMORY_RETRY_LIMIT:
                continue
            # On final attempt, accept but drop memories with <MIN_TRIGGERS_PER_MEMORY.
            raw = [m for m in raw if len(m.triggers) >= MIN_TRIGGERS_PER_MEMORY]
            if not raw:
                return []

        # Convert to MemoryCreate with match_id + player_id attached.
        creates: list[MemoryCreate] = []
        for m in raw[:3]:
            creates.append(
                MemoryCreate(
                    scope=m.scope,
                    type=m.type,
                    emotional_valence=m.emotional_valence,
                    triggers=list(m.triggers),
                    narrative_text=m.narrative_text,
                    relevance_tags=list(m.relevance_tags),
                    player_id=match.player_id,
                    match_id=match.id,
                )
            )
        return creates

    return []


# --- Narrative summary -----------------------------------------------------


def _build_narrative_prompt(
    *,
    character: Character,
    features: dict[str, Any],
    outcome_line: str,
    previous_summary: str | None,
) -> str:
    frags = style_to_prompt_fragments(character)
    parts: list[str] = []
    parts.append(
        f"You are {character.name}. You keep a short private note to yourself about each "
        f"recurring opponent — first-person, at most 3 sentences, in your voice. Write or "
        f"update that note."
    )
    parts.append("")
    parts.append(rating_prompt_fragment(character.content_rating))
    parts.append("")
    parts.append(f"Your style: aggression={frags['aggression']}; patience={frags['patience']}.")
    parts.append(f"Voice / tone: {character.voice_descriptor or '(infer)'}")
    parts.append("")
    parts.append(f"Latest match outcome: {outcome_line}")
    parts.append("")
    parts.append("CURRENT OPPONENT PROFILE (aggregated across all games)")
    parts.append(_compact_features(features))
    parts.append("")
    if previous_summary:
        parts.append("PREVIOUS NOTE (to update, not wholesale replace)")
        parts.append(previous_summary)
        parts.append("")
    parts.append(
        "Return a JSON object with a single `summary` field. At most 3 sentences. "
        "First-person, your voice. Do NOT pretend to know things the features don't support."
    )
    return "\n".join(parts)


def update_narrative_summary(
    *,
    character: Character,
    match: Match,
    features: dict[str, Any],
    previous_summary: str | None,
    llm: LLMClient,
) -> str | None:
    """Return the new narrative summary, or None on LLM failure."""
    prompt = _build_narrative_prompt(
        character=character,
        features=features,
        outcome_line=_outcome_phrase(match),
        previous_summary=previous_summary,
    )
    try:
        out: _NarrativeOut = llm.generate_structured(
            prompt=prompt,
            response_schema=_NarrativeOut,
            response_adapter=_NARRATIVE_ADAPTER,
            temperature=0.8,
            max_output_tokens=400,
            call_tag=f"post_match_narrative:{match.id}",
        )
    except LLMError as exc:
        logger.warning("Post-match narrative LLM failed: %s", exc)
        return None

    if isinstance(out, _NarrativeOut):
        return out.summary
    # Fallback: dict result
    try:
        val = _NARRATIVE_ADAPTER.validate_python(out)
        return val.summary
    except Exception as exc:
        logger.warning("Narrative summary validation failed: %s", exc)
        return None
