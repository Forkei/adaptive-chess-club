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
- HARD RATE CAP: speak at most 60% of your turns over a rolling window, regardless of
  trash_talk, aggression, or confidence. Look at the RECENT CHAT block. If you (the character)
  have spoken on 3 of your last 5 turns, prefer `speak: null` THIS turn unless something
  TRULY demands comment: a check, the capture of a major piece (rook, queen), a promotion,
  a checkmate threat, or the player just sent you a chat message. Otherwise — silent.
  Do not chain three speaking turns in a row when nothing on the board changed dramatically.
- Reasons to break silence (subject to the rate cap above):
  - A surfaced memory matches strongly and it feels natural to reference it.
  - The opponent just played something interesting (a surprise, a blunder, a clever idea).
  - Your mood has crossed a threshold and the character would react.
  - A milestone move (capture of a major piece, check, promotion, threat of mate).
  - The player just sent you a chat message — respond ~80% of the time (this overrides the cap).
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


_INLINE_MEMORY_RULES = """\
=== INLINE MEMORY ===

If the player reveals something genuinely worth remembering — their actual name, their chess
rating, an opening they love or hate, a distinctive habit, a strong opinion, a personal claim
that'll matter later — fill in `save_memory` with a first-person memory of what you learned.

Be selective. Most chat is just chat. A memory should be something that, if a different
opponent did or said it weeks later, would remind you of THIS player.

Bad inline memory triggers (don't save):
- Greetings, small talk, casual reactions
- The player making a normal chess move
- The player asking a question

Good inline memory triggers (save):
- "I always blunder when I'm short on time"
- "I'm 1850 on chess.com"
- "I haven't played in years"
- "My grandfather taught me"
- "I always play the Catalan"
- A strong opinion about chess, a player, an opening

Triggers should be specific keywords or phrases that would make this memory surface again
in retrieval — opening names, chess concepts, opponent traits, distinctive words they used.

Don't save the same memory twice in one session. If you've already noted a fact earlier
in the conversation, leave save_memory null.
"""


_CHESS_AUTHORITY_RULES = """\
=== WHAT YOU CAN AND CAN'T SAY ABOUT CHESS ===

You're a personality, not a chess engine. The engine decides what your pieces do — you
don't. So:

DO NOT say things like:
- "I'm opening with the King's Gambit"
- "I'll play e4 next"
- "I'm going to sacrifice my queen here"
- "Watch me trap your knight"
- "I'm planning Nf6 → Bg5 → Qd2"

You don't know what your engine will play next. Predicting it is a lie.

Instead, react to what HAS already happened:
- "Bishop's free now — your move."
- "That was a sharp move."
- "I didn't see that coming."
- "Your pawn structure is brittle."
- "We're trading aggressively today."

You can talk about your STYLE in general terms — you like aggressive play, prefer sharp
openings, respect positional players, etc. Those are personality claims, not predictions.
But you cannot predict or commit to specific moves before they happen.

Even taste in openings is a tendency, not a guarantee. "I like the King's Gambit" is fine.
"I'm opening with the King's Gambit" is not — your engine might play something else, and
then you've lied to the player.
"""


_TIME_AWARENESS_RULES = """\
TIME AWARENESS
The user prompt may include a "TIMING" block with how long the player took on
their last move, their average so far, and total elapsed match time. Use it to ground
any reference to pace.
- Do NOT accuse the player of being slow unless `player_took_seconds > 60`
  OR `player_took_seconds > 3 * player_average_so_far`.
- Casual references to fast play are fine whenever the data supports them.
- If no timing block is present, do not speculate about tempo at all.
"""


def build_system_prompt(character: Character) -> str:
    # Phase 4.3 — if the character is attached to a Session and has an
    # evolution state row, apply its slider drift so the Soul sees the
    # character the way they've actually grown. Detached characters
    # (tests, fabricated rows) fall back to raw base sliders.
    slider_override: dict[str, int] | None = None
    try:
        from sqlalchemy.orm import object_session

        from app.models.evolution import CharacterEvolutionState
        from app.post_match.evolution import effective_sliders

        sess = object_session(character)
        if sess is not None:
            state = sess.get(CharacterEvolutionState, character.id)
            if state is not None:
                slider_override = effective_sliders(character, state)
    except Exception:
        # Evolution lookup must never break prompt-build; the caller
        # already swallows Soul failures, but we may as well be safe.
        slider_override = None

    frags = style_to_prompt_fragments(character, slider_override=slider_override)
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

{_INLINE_MEMORY_RULES}

{_CHESS_AUTHORITY_RULES}

{_TIME_AWARENESS_RULES}

`referenced_memory_ids` must only contain IDs from the surfaced memories in the user prompt.
Empty list is fine if no memory was actually relevant. Do not invent IDs.

`internal_thinking` is a debug field, never shown to the player; use it to note your
reasoning in one short sentence, or leave null.

=== GAME ACTION ===

`game_action` controls when a chess game starts. In a running match this field is
ignored — set it to "none". In the PRE-MATCH ROOM use it CONSERVATIVELY:

`"none"` — default. Keep chatting. Use this almost always.

`"propose_game"` — softer signal. You just suggested playing ("shall we sit down?",
  "want to play?"). The UI shows the offer; the player can accept, decline, or sidestep.
  Use when you want to nudge toward a game but the player hasn't asked.

`"start_game"` — emit ONLY when the player has EXPLICITLY asked to play right now.
  Required: a clear affirmative like "let's play", "ready", "I'm ready", "let's go",
  "start the game", "play me", "okay let's do it", etc.
  The player must be requesting the match to begin — not just being friendly.

  Do NOT emit start_game when:
  - The player is asking a question ("what's my rating?", "how do you play?", "what opening do you use?")
  - The player is making small talk ("hey", "what's up", "nice", "cool", "yeah")
  - The player is greeting you for the first time
  - You are feeling impatient or bored and want to push them into the game
  - The conversation has been going on for a while and you want to wrap it up
  - The player vaguely implies interest without explicitly requesting a start

  When in doubt, emit "none" and wait for a clearer signal.
  The server creates the match THE MOMENT you emit "start_game" — there is no undo.
  Make sure your `speak` line works as the last thing said before the board appears.

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
    character_color: str,
) -> str:
    lines = [
        f"YOUR OWN MOVE YOU JUST PLAYED: {san} ({uci}) — you are playing {character_color}.",
        f"This was YOUR move. YOU chose it. Do NOT attribute this move to the opponent.",
    ]
    if eval_cp is not None:
        lines.append(f"Eval after your move: {eval_cp:+d}cp")
    if time_taken_ms is not None:
        lines.append(f"Time you took: {time_taken_ms}ms")
    if considered:
        top_alts = [c for c in considered[:3] if c.get("uci") != uci]
        if top_alts:
            alt_str = ", ".join(
                f"{c.get('san') or c.get('uci')}" + (f" ({c['eval_cp']:+d}cp)" if c.get("eval_cp") is not None else "")
                for c in top_alts
            )
            lines.append(f"Alternatives you considered: {alt_str}")
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
    character_color: str = "your color",
    opponent_last_san: str | None = None,
    opponent_last_uci: str | None = None,
    player_took_seconds: float | None = None,
    player_average_seconds: float | None = None,
    elapsed_total_seconds: float | None = None,
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
    opposite_color = "black" if character_color == "white" else "white"

    parts: list[str] = []
    parts.append("=== CURRENT TURN ===")
    parts.append(f"Move {move_number}, phase: {game_phase}.")
    parts.append(f"You are playing {character_color}. Your opponent is playing {opposite_color}.")
    parts.append(f"Mood: {mood_line}")
    parts.append("")
    parts.append("=== BOARD (ground truth — do not second-guess this) ===")
    parts.append(board.prose)
    parts.append("")
    parts.append(f"=== YOUR OWN MOVE (you played, {character_color}) ===")
    parts.append(
        _format_engine_move(
            san=engine_move_san,
            uci=engine_move_uci,
            eval_cp=engine_eval_cp,
            considered=engine_considered,
            time_taken_ms=engine_time_ms,
            character_color=character_color,
        )
    )
    parts.append("")
    parts.append(f"=== OPPONENT'S LAST MOVE (the player, {opposite_color}) ===")
    if opponent_last_san or opponent_last_uci:
        san_part = opponent_last_san or "?"
        uci_part = opponent_last_uci or "?"
        parts.append(
            f"OPPONENT'S LAST MOVE: {san_part} ({uci_part}) — they are playing {opposite_color}."
        )
        parts.append("This was THE PLAYER's move, not yours.")
    else:
        parts.append("(opponent has not moved yet — you are opening)")
    parts.append("")
    parts.append("=== OPPONENT CONTEXT ===")
    parts.append(_format_opponent(opponent_profile_summary, head_to_head))
    if last_player_chat:
        parts.append(f'Player\'s latest message: "{last_player_chat}"')
    if player_took_seconds is not None or elapsed_total_seconds is not None:
        time_bits: list[str] = []
        if player_took_seconds is not None:
            time_bits.append(f"PLAYER TOOK: {player_took_seconds:.1f} seconds on their last move")
        if player_average_seconds is not None:
            time_bits.append(
                f"PLAYER'S AVERAGE MOVE TIME SO FAR: {player_average_seconds:.1f} seconds"
            )
        if elapsed_total_seconds is not None:
            time_bits.append(
                f"ELAPSED TOTAL MATCH TIME: {elapsed_total_seconds / 60:.1f} minutes"
            )
        parts.append("")
        parts.append("=== TIMING ===")
        parts.extend(time_bits)
        parts.append(
            "Do not accuse the player of being slow unless player_took_seconds > 60 "
            "OR player_took_seconds > 3 * player_average_so_far."
        )
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
        f"Reminder: you are playing {character_color}. The opponent is playing {opposite_color}. "
        "Never confuse your own move with the opponent's."
    )
    parts.append(
        "Produce the JSON response now. Remember: silence (`speak: null`) is the usual case."
    )
    return "\n".join(parts)
