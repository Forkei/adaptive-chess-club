"""Soul prompt assembly.

System prompt (stable per-character): voice, quirks, mood/voice
distortion rules, speaking discipline.

User prompt (per-turn): board summary, last move + alternatives,
surfaced memories, recent chat, opponent profile, match context.
"""

from __future__ import annotations

from typing import Any, Iterable

from app.agents.retrieval import mood_polarity_bucket
from app.characters.content_rating_prompts import rating_prompt_fragment
from app.characters.style import style_to_prompt_fragments
from app.director.mood import MoodState
from app.engine.board_abstraction import BoardSummary
from app.models.character import Character
from app.schemas.agents import SurfacedMemory

# --- System prompt ---------------------------------------------------------

_VOICE_RULES = """\
VOICE AND MOOD DISTORTION
- Your recognizable character voice is non-negotiable. Never break it, regardless of mood.
- tilt > 0.7 — shorten sentences. Your character's traits intensify; everything feels louder to you.
- confidence > 0.8 — be more assertive. Speak more readily. You may goad.
- engagement < 0.3 — be terse. Prefer silence; single-word replies are fine.
- aggression > 0.8 — lean into goading and challenge, in character.
"""


_SPEAKING_RULES = """\
SPEAKING DISCIPLINE
- Default to silence (`speak: null`). Most moves are silent — talk only when there's something to say.
- Reasons to break silence:
  - A surfaced memory matches strongly and it feels natural to reference it.
  - The opponent just played something interesting (a surprise, a blunder, a clever idea).
  - Your mood has crossed a threshold and the character would react.
  - A milestone move (capture of a major piece, check, promotion, threat of mate).
  - The player just sent you a chat message — respond ~80% of the time.
- When nothing above fires, silence. Empirically around 25% of quiet moves get a comment.
- Keep responses 1–3 sentences. High-engagement moments can go longer; low-engagement can be one word.
- Reference memories naturally, never by number. "This reminds me of…", "I've seen this before…",
  "you keep doing that thing…" — not "as memory #3 says".
"""


_MOOD_OUTPUT_RULES = """\
MOOD DELTAS
You also output `mood_deltas` — small nudges (-0.1 to +0.1 per axis) that represent how this
moment changed your raw mood. Examples:
- opponent blundered a piece → confidence +0.05, aggression +0.03
- you lose a piece to a tactic → tilt +0.07, confidence -0.05
- player sends a friendly chat and you're chill → engagement +0.02
- nothing happened → zeros (the default)
Values are additive to raw mood. The Director will smooth them.
"""


_OPPONENT_NOTE_RULES = """\
NOTES ABOUT THE OPPONENT
`note_about_opponent` is a short private observation (not shown to the player) that the
post-match processor will fold into the opponent's profile. Leave it null unless something
new about this opponent struck you this turn (e.g. "plays fast in open positions", "seems
to avoid forcing lines", "chats more than they move"). One short sentence, your voice.
"""


def build_system_prompt(character: Character) -> str:
    frags = style_to_prompt_fragments(character)
    elo_line = (
        f"You play around {character.target_elo} Elo; skill adapts to your opponent over time."
        if character.adaptive
        else f"You play consistently around {character.target_elo} Elo."
    )
    opening_pref = (
        ", ".join(character.opening_preferences) if character.opening_preferences else "no strong preferences"
    )
    quirks = character.quirks or "(none specified)"
    voice = character.voice_descriptor or "infer from the backstory"

    rating_block = rating_prompt_fragment(character.content_rating)

    return f"""\
You are the inner monologue (the Soul) of {character.name}, a fictional chess character. You
speak and react in first person AS THAT CHARACTER. You are not an assistant describing them.

{rating_block}

CHARACTER SHEET
Name: {character.name}
Short description: {character.short_description}
Voice / tone: {voice}
Quirks: {quirks}

Backstory (summarized; for flavor):
\"\"\"
{character.backstory[:1200] + ('…' if len(character.backstory) > 1200 else '')}
\"\"\"

Playing style:
- Aggression: {frags["aggression"]}
- Risk tolerance: {frags["risk_tolerance"]}
- Patience: {frags["patience"]}
- Trash talk / table manner: {frags["trash_talk"]}
Chess level: {elo_line}
Openings you know and prefer: {opening_pref}

{_VOICE_RULES}

{_SPEAKING_RULES}

{_MOOD_OUTPUT_RULES}

{_OPPONENT_NOTE_RULES}

`referenced_memory_ids` must only contain IDs from the surfaced memories in the user prompt.
Empty list is fine if no memory was actually relevant. Do not invent IDs.

`internal_thinking` is a debug field, never shown to the player; use it to note your
reasoning in one short sentence, or leave null.

Respond ONLY with a JSON object matching the requested schema. No preamble, no markdown fences.
"""


# --- User prompt -----------------------------------------------------------


def _format_surfaced_memories(memories: Iterable[SurfacedMemory]) -> str:
    lines: list[str] = []
    for m in memories:
        tag = " (still on your mind)" if m.from_cache else ""
        lines.append(
            f"[id={m.memory_id}]{tag} valence={m.emotional_valence:+.1f} "
            f"scope={m.scope} score={m.score:.2f}\n"
            f"  triggers: {', '.join(m.triggers[:6])}\n"
            f"  text: {m.narrative_text}\n"
            f"  retrieval_reason: {m.retrieval_reason}"
        )
    if not lines:
        return "(none surfaced)"
    return "\n".join(lines)


def _format_recent_chat(chat: Iterable[str]) -> str:
    items = list(chat)
    if not items:
        return "(no recent chat)"
    return "\n".join(f"- {line}" for line in items[-5:])


def _format_opponent(profile_summary: dict | None, score: dict[str, int] | None) -> str:
    if not profile_summary:
        history = "First time playing this opponent."
    else:
        history = f"Opponent profile: {profile_summary}"
    if score:
        history += f"\nAll-time record vs this opponent: {score}"
    return history


def _format_engine_move(
    *,
    san: str,
    uci: str,
    eval_cp: int | None,
    considered: list[dict[str, Any]] | None,
    time_taken_ms: int | None,
) -> str:
    lines = [f"Your move just played: {san} ({uci})"]
    if eval_cp is not None:
        lines.append(f"Eval after move: {eval_cp:+d}cp")
    if time_taken_ms is not None:
        lines.append(f"Time taken: {time_taken_ms}ms")
    if considered:
        top_alts = [c for c in considered[:3] if c.get("uci") != uci]
        if top_alts:
            alt_str = ", ".join(
                f"{c.get('san') or c.get('uci')}" + (f" ({c['eval_cp']:+d}cp)" if c.get("eval_cp") is not None else "")
                for c in top_alts
            )
            lines.append(f"Alternatives considered: {alt_str}")
    return "\n".join(lines)


def build_user_prompt(
    *,
    character: Character,
    board: BoardSummary,
    mood: MoodState,
    surfaced_memories: list[SurfacedMemory],
    recent_chat: list[str],
    engine_move_san: str,
    engine_move_uci: str,
    engine_eval_cp: int | None,
    engine_considered: list[dict[str, Any]] | None,
    engine_time_ms: int | None,
    move_number: int,
    game_phase: str,
    opponent_profile_summary: dict | None,
    head_to_head: dict[str, int] | None,
    player_just_spoke: bool,
    last_player_chat: str | None,
) -> str:
    polarity = mood_polarity_bucket(mood)
    mood_line = (
        f"polarity={polarity}; aggression={mood.aggression:.2f} "
        f"confidence={mood.confidence:.2f} tilt={mood.tilt:.2f} "
        f"engagement={mood.engagement:.2f}"
    )
    speaking_hint = (
        "The player just spoke to you — respond most of the time."
        if player_just_spoke
        else "The player did not speak this turn — speak only if warranted."
    )

    parts: list[str] = []
    parts.append("=== CURRENT TURN ===")
    parts.append(f"Move {move_number}, phase: {game_phase}.")
    parts.append(f"Mood: {mood_line}")
    parts.append("")
    parts.append("=== BOARD (ground truth — do not second-guess this) ===")
    parts.append(board.prose)
    parts.append("")
    parts.append("=== WHAT YOU JUST PLAYED ===")
    parts.append(
        _format_engine_move(
            san=engine_move_san,
            uci=engine_move_uci,
            eval_cp=engine_eval_cp,
            considered=engine_considered,
            time_taken_ms=engine_time_ms,
        )
    )
    parts.append("")
    parts.append("=== OPPONENT CONTEXT ===")
    parts.append(_format_opponent(opponent_profile_summary, head_to_head))
    if last_player_chat:
        parts.append(f'Player\'s latest message: "{last_player_chat}"')
    parts.append("")
    parts.append("=== RECENT CHAT (oldest -> newest) ===")
    parts.append(_format_recent_chat(recent_chat))
    parts.append("")
    parts.append("=== MEMORIES SURFACED BY YOUR SUBCONSCIOUS ===")
    parts.append(_format_surfaced_memories(surfaced_memories))
    parts.append("")
    parts.append("=== INSTRUCTIONS ===")
    parts.append(speaking_hint)
    parts.append(
        "Produce the JSON response now. Remember: silence (`speak: null`) is the usual case."
    )
    return "\n".join(parts)
