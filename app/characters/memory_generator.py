"""Backstory → memories generator.

Takes a saved Character, builds a prompt, calls Gemini with a Pydantic
list schema, validates, and persists the memories. On success the
character moves to `ready`; on failure it moves to `generation_failed`
with an error message.
"""

from __future__ import annotations

import logging
from datetime import datetime

from pydantic import BaseModel, Field, TypeAdapter
from sqlalchemy.orm import Session

from app.characters.openings import by_eco, by_name
from app.characters.style import style_to_prompt_fragments
from app.config import get_settings
from app.db import session_scope
from app.llm.client import LLMClient, LLMError, get_llm_client
from app.memory.crud import bulk_create
from app.models.character import Character, CharacterState
from app.models.memory import MemoryScope, MemoryType
from app.schemas.memory import MemoryCreate

logger = logging.getLogger(__name__)


class GeneratedMemory(BaseModel):
    """Shape the LLM must return for each memory.

    This is the Pydantic model passed to google-genai's `response_schema`.
    Deliberately narrower than `MemoryCreate` — the LLM never sets
    `player_id`/`match_id` for character lore.
    """

    scope: MemoryScope
    type: MemoryType
    emotional_valence: float = Field(..., ge=-1.0, le=1.0)
    triggers: list[str] = Field(
        ...,
        description=(
            "Retrieval cues: opening names, piece patterns, opponent traits, "
            "board features, keywords. 3-8 items. Concrete and specific."
        ),
    )
    narrative_text: str = Field(
        ...,
        min_length=20,
        description="First-person memory matching the character's voice. 2-5 sentences.",
    )
    relevance_tags: list[str] = Field(
        ...,
        description=(
            "Chess concepts this memory connects to: 'endgame', 'kingside attack', "
            "'sicilian', 'zugzwang', etc. 2-6 items."
        ),
    )


_MEMORY_LIST_ADAPTER = TypeAdapter(list[GeneratedMemory])


_FEW_SHOT = """\
Example of good memories (voice: gruff Russian grandmaster):

[
  {
    "scope": "character_lore",
    "type": "formative",
    "emotional_valence": -0.3,
    "triggers": ["grandfather", "Minsk", "first loss", "queen sacrifice"],
    "narrative_text": "My grandfather let me win the first ten games. Then on the eleventh he crushed me with a queen sacrifice I did not see coming. I was nine. I cried and refused to speak to him for a week. It was the best thing he ever did for me.",
    "relevance_tags": ["childhood", "sacrifice", "queen sacrifice", "lesson"]
  },
  {
    "scope": "character_lore",
    "type": "rivalry",
    "emotional_valence": 0.2,
    "triggers": ["Kasparov", "Linares 1993", "Najdorf", "draw"],
    "narrative_text": "I drew with Kasparov once, in Linares, playing the Najdorf. He offered a handshake and a single word afterwards: 'Nerves.' I still do not know if he meant mine or his.",
    "relevance_tags": ["tournament", "najdorf", "sicilian", "elite"]
  },
  {
    "scope": "character_lore",
    "type": "opinion",
    "emotional_valence": -0.6,
    "triggers": ["London System", "boring", "white opening"],
    "narrative_text": "The London System is chess for accountants. I would rather lose a brilliant game than win a dull one with d4-Bf4-e3. Do not play it against me — I will not respect you for it.",
    "relevance_tags": ["openings", "london", "d4", "style"]
  }
]
"""


def _opening_names(preferences: list[str]) -> list[str]:
    names: list[str] = []
    for ident in preferences:
        op = by_name(ident) or by_eco(ident)
        if op is not None:
            names.append(op["name"])
        else:
            names.append(ident)
    return names


def build_prompt(character: Character, *, target: int, minimum: int, maximum: int) -> str:
    fragments = style_to_prompt_fragments(character)
    openings = _opening_names(list(character.opening_preferences or []))
    openings_line = ", ".join(openings) if openings else "no strong preferences"
    elo_line = (
        f"Starting around {character.target_elo} Elo; skill adapts to the opponent over time."
        if character.adaptive
        else f"Consistently plays around {character.target_elo} Elo."
    )

    return f"""\
You are writing the personal memories of a fictional chess-playing character. These memories
will be stored in a database and later surfaced to the character during chat and play, to
give their voice depth and continuity. Write them in first person, as if the character is
quietly recalling episodes from their own life.

CHARACTER PROFILE
Name: {character.name}
Voice / tone: {character.voice_descriptor or "not specified — infer from the backstory"}
Short description: {character.short_description or "(none)"}
Backstory:
\"\"\"
{character.backstory or "(no backstory provided — invent one consistent with the profile)"}
\"\"\"

Quirks: {character.quirks or "(none specified)"}

Playing style:
- Aggression: {fragments["aggression"]}
- Risk tolerance: {fragments["risk_tolerance"]}
- Patience: {fragments["patience"]}
- Trash talk / table manner: {fragments["trash_talk"]}

Chess level: {elo_line}
Favourite / typical openings: {openings_line}

WHAT TO PRODUCE
Return a JSON array of between {minimum} and {maximum} memories (aim for ~{target}). Strive
for variety across:
- `type`: mix formative, rivalry, travel, triumph, defeat, habit, opinion, observation.
  Do NOT cluster in just one or two types.
- `emotional_valence` (-1.0..1.0): span the full range. Include bitter memories, warm ones,
  ambivalent ones, and a few near-zero neutral observations. Avoid a single emotional tone.
- `scope`: most memories should be "character_lore". You may also include a handful of
  "cross_player" observations (general opinions about kinds of opponents the character has
  met over their career) — but do NOT invent specific named opponents for those.

WRITING STYLE
- First-person, matching the character's voice and quirks.
- 2-5 sentences per memory. Specific, sensory, particular. NOT generic.
- Name concrete places, years, opponents (when appropriate for character_lore), tournaments,
  openings, positional motifs. Specificity is the whole point — these memories exist to be
  retrieved by triggers, so they need hooks.
- Avoid starting every memory the same way ("I remember...", "Once...").

TRIGGERS AND RELEVANCE TAGS
- `triggers`: 3-8 short strings that would cue retrieval. Opening names ("Sicilian Najdorf"),
  places ("Reykjavik"), opponent archetypes ("blitz hustler"), positional features
  ("opposite-side castling"), themes ("time trouble"), people from the backstory.
- `relevance_tags`: 2-6 chess-concept tags. Examples: "endgame", "kingside attack",
  "zugzwang", "pawn structure", "opening theory", "time pressure".

{_FEW_SHOT}

Now produce the memories for {character.name}. Output ONLY the JSON array — no surrounding
prose, no markdown fences.
"""


def generate_and_store(character_id: str, *, client: LLMClient | None = None) -> int:
    """Generate memories for `character_id` and persist them.

    Moves character state to `ready` on success, `generation_failed` on error.
    Returns the number of memories persisted.
    """
    settings = get_settings()
    started = datetime.utcnow()

    with session_scope() as session:
        character = session.get(Character, character_id)
        if character is None:
            raise LLMError(f"Character {character_id} not found")
        character.state = CharacterState.GENERATING_MEMORIES
        character.memory_generation_started_at = started
        character.memory_generation_error = None
        prompt = build_prompt(
            character,
            target=settings.memory_gen_target,
            minimum=settings.memory_gen_min,
            maximum=settings.memory_gen_max,
        )

    try:
        llm = client or get_llm_client()
        memories_raw: list[GeneratedMemory] = llm.generate_structured(
            prompt=prompt,
            response_schema=list[GeneratedMemory],
            response_adapter=_MEMORY_LIST_ADAPTER,
            temperature=0.95,
            call_tag=f"memory_gen:{character_id}",
        )

        if len(memories_raw) < max(1, settings.memory_gen_min // 2):
            raise LLMError(
                f"LLM returned only {len(memories_raw)} memories; "
                f"minimum acceptable is {settings.memory_gen_min // 2}"
            )

        creates: list[MemoryCreate] = []
        skipped = 0
        for raw in memories_raw:
            try:
                creates.append(
                    MemoryCreate(
                        scope=raw.scope,
                        type=raw.type,
                        emotional_valence=raw.emotional_valence,
                        triggers=list(raw.triggers),
                        narrative_text=raw.narrative_text,
                        relevance_tags=list(raw.relevance_tags),
                    )
                )
            except Exception as exc:
                skipped += 1
                logger.warning("Discarding malformed memory: %s", exc)

        if not creates:
            raise LLMError("All generated memories failed validation")

        with session_scope() as session:
            bulk_create(session, character_id=character_id, items=creates)
            char = session.get(Character, character_id)
            if char is not None:
                char.state = CharacterState.READY
                char.memory_generation_error = None

        logger.info(
            "Generated %d memories for character %s (skipped %d malformed)",
            len(creates),
            character_id,
            skipped,
        )
        return len(creates)

    except Exception as exc:
        logger.exception("Memory generation failed for character %s", character_id)
        with session_scope() as session:
            char = session.get(Character, character_id)
            if char is not None:
                char.state = CharacterState.GENERATION_FAILED
                char.memory_generation_error = f"{type(exc).__name__}: {exc}"
        raise
