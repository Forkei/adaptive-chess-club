"""Phase 4.2.5 — pre-match chat service.

The room you enter when you click a character card. The player talks to
the agent (full Soul, Subconscious, memory, emotion). No board exists
yet. The Soul decides when to shift from chat into a game by emitting
`game_action="start_game"` in its response.

Reuses the existing Soul + Subconscious runners unchanged — we just
pass them a synthetic board state (startpos) and a chat-only user
prompt, and bypass the match-persistence path.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import chess
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.agents.soul import SoulInput, run_soul
from app.agents.subconscious import SubconsciousInput, run_subconscious
from app.director.mood import (
    MoodState,
    apply_deltas,
    load_mood,
    save_mood,
    smooth_mood,
)
from app.engine.board_abstraction import board_to_english
from app.models.evolution import CharacterEvolutionState
from app.post_match.evolution import tone_bias_for
from app.matches.service import create_match as create_pve_match
from app.models.character import Character, CharacterState
from app.models.chat import (
    CharacterChatSession,
    CharacterChatTurn,
    ChatSessionStatus,
    ChatTurnRole,
)
from app.models.match import Match, Player
from app.schemas.agents import SoulResponse, SurfacedMemory

logger = logging.getLogger(__name__)

# Keep the last N turns in the Soul's `recent_chat` field. Small number —
# Soul cost scales with prompt size and the system prompt is already big.
RECENT_CHAT_WINDOW = 6

# Probability the character opens the conversation when the room has
# zero turns. Not 100% so it doesn't feel scripted.
GREETING_PROBABILITY = 0.7


# --- session + turn helpers -----------------------------------------------


def get_or_create_session(
    session: Session, *, character: Character, player: Player
) -> CharacterChatSession:
    """One active session per (character, player). Reused across page
    refreshes so conversation history doesn't vanish.
    """
    row = session.execute(
        select(CharacterChatSession)
        .where(CharacterChatSession.character_id == character.id)
        .where(CharacterChatSession.player_id == player.id)
        .where(CharacterChatSession.status == ChatSessionStatus.ACTIVE)
        .order_by(CharacterChatSession.created_at.desc())
    ).scalar_one_or_none()
    if row is not None:
        return row
    row = CharacterChatSession(
        character_id=character.id,
        player_id=player.id,
        status=ChatSessionStatus.ACTIVE,
        pending_notes=[],
    )
    session.add(row)
    session.commit()
    session.refresh(row)
    return row


def get_turns(session: Session, chat_session: CharacterChatSession) -> list[CharacterChatTurn]:
    return list(
        session.execute(
            select(CharacterChatTurn)
            .where(CharacterChatTurn.session_id == chat_session.id)
            .order_by(CharacterChatTurn.turn_number.asc())
        ).scalars()
    )


def _next_turn_number(session: Session, chat_session: CharacterChatSession) -> int:
    from sqlalchemy import func

    n = session.execute(
        select(func.max(CharacterChatTurn.turn_number)).where(
            CharacterChatTurn.session_id == chat_session.id
        )
    ).scalar()
    return (n or 0) + 1


def _append_player_turn(
    session: Session, chat_session: CharacterChatSession, text: str
) -> CharacterChatTurn:
    turn = CharacterChatTurn(
        session_id=chat_session.id,
        turn_number=_next_turn_number(session, chat_session),
        role=ChatTurnRole.PLAYER,
        text=text,
    )
    session.add(turn)
    chat_session.updated_at = datetime.utcnow()
    session.commit()
    session.refresh(turn)
    return turn


def _append_character_turn(
    session: Session,
    chat_session: CharacterChatSession,
    *,
    text: str,
    soul_response: SoulResponse,
) -> CharacterChatTurn:
    turn = CharacterChatTurn(
        session_id=chat_session.id,
        turn_number=_next_turn_number(session, chat_session),
        role=ChatTurnRole.CHARACTER,
        text=text,
        emotion=soul_response.emotion,
        emotion_intensity=soul_response.emotion_intensity,
        game_action=soul_response.game_action,
        soul_raw=soul_response.model_dump(),
    )
    session.add(turn)
    chat_session.updated_at = datetime.utcnow()
    session.commit()
    session.refresh(turn)
    return turn


# --- Soul invocation ------------------------------------------------------


def _recent_chat_lines(turns: list[CharacterChatTurn]) -> list[str]:
    """Render the last N turns as 'player: ...' / 'character: ...' strings
    for the Soul's recent_chat field."""
    out: list[str] = []
    for t in turns[-RECENT_CHAT_WINDOW:]:
        prefix = "player" if t.role == ChatTurnRole.PLAYER else "character"
        out.append(f"{prefix}: {t.text}")
    return out


def _chat_mood_key(chat_session_id: str) -> str:
    """Mood persistence key used during a pre-match chat session. Re-uses
    the same Redis/in-memory store as match moods, so continuity across
    chat turns works without a new table."""
    return f"chat:{chat_session_id}"


def _default_mood_for(
    character: Character, *, session: Session | None = None
) -> MoodState:
    """Build a mood anchored on the character's sliders, biased by
    Phase 4.3 tone drift if we can see the evolution state.
    """
    tone_bias: dict[str, float] = {}
    if session is not None:
        state = session.get(CharacterEvolutionState, character.id)
        tone_bias = tone_bias_for(state)
    aggression = max(0.0, min(1.0, character.aggression / 10.0))
    confidence = max(0.0, min(1.0, 0.5 + float(tone_bias.get("confidence_baseline", 0.0))))
    # tilt_baseline is negative when on a loss streak — invert so loss
    # streak raises the chat mood's tilt.
    tilt = max(0.0, min(1.0, 0.0 - float(tone_bias.get("tilt_baseline", 0.0))))
    return MoodState(
        aggression=aggression,
        confidence=confidence,
        tilt=tilt,
        engagement=0.3,
    )


def _load_or_init_chat_mood(
    session: Session, chat_session: CharacterChatSession, character: Character
) -> MoodState:
    stored = load_mood(_chat_mood_key(chat_session.id), smoothed=True)
    if stored is not None:
        return stored
    initial = _default_mood_for(character, session=session)
    save_mood(_chat_mood_key(chat_session.id), initial, smoothed=True)
    return initial


def _idle_board_summary():
    """A BoardSummary for the starting position. The Soul knows there's
    no game yet from the game_action docs + the empty move history."""
    return board_to_english(chess.Board())


@dataclass
class ChatTurnResult:
    """Return shape of `handle_player_message`. Exactly one of
    `handed_off_match_id` or `None` is set depending on whether the
    Soul emitted `start_game`.
    """

    player_turn: CharacterChatTurn
    character_turn: CharacterChatTurn
    soul_response: SoulResponse
    surfaced_memories: list[SurfacedMemory]
    handed_off_match_id: str | None


def handle_player_message(
    session: Session,
    *,
    chat_session: CharacterChatSession,
    character: Character,
    player: Player,
    text: str,
) -> ChatTurnResult:
    """Full chat turn:
      1. persist player turn
      2. Subconscious → surfaced memories
      3. Soul → response
      4. persist character turn
      5. if game_action=='start_game', create a Match + hand off session
    """
    text = (text or "").strip()
    if not text:
        raise ValueError("empty message")
    if len(text) > 2000:
        text = text[:2000]

    if chat_session.status != ChatSessionStatus.ACTIVE:
        raise RuntimeError(f"chat session not active: {chat_session.status}")

    player_turn = _append_player_turn(session, chat_session, text)

    turns = get_turns(session, chat_session)
    recent_chat = _recent_chat_lines(turns[:-1])  # don't double-include the message we just added
    board = _idle_board_summary()
    # Phase 4.3g — persistent chat mood. Starts from sliders + evolution
    # tone bias; Soul's mood_deltas compound across turns within a
    # session.
    mood = _load_or_init_chat_mood(session, chat_session, character)

    # Subconscious — cheap when there's no FEN to work from; still useful
    # for memory retrieval based on the player's chat.
    sub_out: list[SurfacedMemory] = []
    try:
        sub_input = SubconsciousInput(
            character_id=character.id,
            match_id=f"chat:{chat_session.id}",
            current_turn=player_turn.turn_number,
            board_summary=board,
            mood=mood,
            last_player_uci=None,
            last_player_chat=text,
            last_moves_san=[],
            recent_chat=recent_chat + [f"player: {text}"],
            opening_label=None,
            current_player_id=player.id,
            opponent_style_features=None,
        )
        sub_out = run_subconscious(session, character, sub_input) or []
    except Exception:
        logger.exception("[chat] subconscious failed for session=%s", chat_session.id)

    # Soul.
    try:
        soul_input = SoulInput(
            board=board,
            mood=mood,
            surfaced_memories=sub_out,
            recent_chat=recent_chat + [f"player: {text}"],
            engine_move_san="",  # no move yet
            engine_move_uci="",
            engine_eval_cp=None,
            engine_considered=None,
            engine_time_ms=None,
            move_number=0,
            game_phase="pre-match",
            opponent_profile_summary=None,
            head_to_head=None,
            player_just_spoke=True,
            last_player_chat=text,
            match_id=f"chat:{chat_session.id}",
            character_color="white",
        )
        soul_resp = run_soul(character, soul_input)
    except Exception:
        logger.exception("[chat] soul failed for session=%s", chat_session.id)
        soul_resp = SoulResponse(speak=None, emotion="neutral", emotion_intensity=0.2)

    # Phase 4.3g — compound the Soul's mood_deltas onto the chat mood
    # and persist so the next turn starts from the evolved state.
    try:
        raw_next = apply_deltas(mood, soul_resp.mood_deltas.to_dict())
        smoothed = smooth_mood(mood, raw_next)
        save_mood(_chat_mood_key(chat_session.id), smoothed, smoothed=True)
    except Exception:
        logger.exception("[chat] failed to persist mood for session=%s", chat_session.id)

    # Always produce a character turn, even when the Soul chose silence —
    # an empty bubble would confuse the UI. Use a subtle ellipsis so the
    # user knows the character acknowledged them.
    reply_text = soul_resp.speak or "…"
    character_turn = _append_character_turn(
        session, chat_session, text=reply_text, soul_response=soul_resp
    )

    # Stash opponent notes on the session so they flow into match.extra_state
    # at hand-off.
    if soul_resp.note_about_opponent:
        chat_session.pending_notes = (chat_session.pending_notes or []) + [
            soul_resp.note_about_opponent
        ]
        session.commit()

    # Hand-off.
    handed_off: str | None = None
    if soul_resp.game_action == "start_game":
        handed_off = _hand_off_to_match(session, chat_session, character, player)

    return ChatTurnResult(
        player_turn=player_turn,
        character_turn=character_turn,
        soul_response=soul_resp,
        surfaced_memories=list(sub_out),
        handed_off_match_id=handed_off,
    )


def _hand_off_to_match(
    session: Session,
    chat_session: CharacterChatSession,
    character: Character,
    player: Player,
) -> str:
    """Create a fresh Match, mark the session handed-off, forward notes
    so the post-match memory pipeline sees the pre-match conversation.
    """
    match = create_pve_match(
        session, character_id=character.id, player_id=player.id, player_color="random"
    )
    # Forward any pending opponent notes into the match's extra_state.
    notes = list(chat_session.pending_notes or [])
    extra = dict(match.extra_state or {})
    prior = list(extra.get("pending_opponent_notes") or [])
    extra["pending_opponent_notes"] = prior + notes
    extra["pre_match_chat_session_id"] = chat_session.id
    match.extra_state = extra

    chat_session.status = ChatSessionStatus.HANDED_OFF
    chat_session.handed_off_match_id = match.id
    chat_session.ended_at = datetime.utcnow()

    session.commit()
    logger.info(
        "[chat] session=%s handed off to match=%s (notes=%d)",
        chat_session.id, match.id, len(notes),
    )
    return match.id


def _opponent_context(session: Session, character: Character, player: Player) -> str | None:
    """Build a short English string describing prior history with this
    player — feeds into the greeting prompt so the character can say
    "back for another one?" vs "first time here?" organically.
    """
    from app.models.match import MatchResult, OpponentProfile

    profile = session.execute(
        select(OpponentProfile)
        .where(OpponentProfile.character_id == character.id)
        .where(OpponentProfile.player_id == player.id)
    ).scalar_one_or_none()
    if profile is None or profile.games_played == 0:
        return None
    wins = profile.games_won_by_character or 0
    losses = profile.games_lost_by_character or 0
    draws = profile.games_played - wins - losses
    bits = [f"You've faced {player.username} {profile.games_played} time(s)"]
    record = f"(W {wins} / L {losses}"
    if draws:
        record += f" / D {draws}"
    record += ")"
    bits.append(record)
    if profile.narrative_summary:
        bits.append(profile.narrative_summary.strip())
    return " ".join(bits)


def maybe_character_greets(
    session: Session,
    chat_session: CharacterChatSession,
    character: Character,
    player: Player,
) -> CharacterChatTurn | None:
    """Fire an unsolicited character greeting when the player first
    enters an empty chat room. Returns the persisted turn, or None.

    Preconditions:
    - Session has zero turns (first arrival)
    - Random roll under GREETING_PROBABILITY
    - Character is READY

    Memories get surfaced by the Subconscious before the Soul speaks,
    flavored by any prior OpponentProfile with this player, so Viktor
    can rib a repeat loser or reset warily with a stranger.
    """
    import random

    if chat_session.status != ChatSessionStatus.ACTIVE:
        return None
    if character.state != CharacterState.READY:
        return None

    existing = get_turns(session, chat_session)
    if existing:
        return None
    if random.random() > GREETING_PROBABILITY:
        return None

    board = _idle_board_summary()
    mood = _load_or_init_chat_mood(session, chat_session, character)

    # Prime the Subconscious. No `last_player_chat` (player hasn't spoken),
    # but opponent profile + player identity still steer memory retrieval.
    surfaced: list[SurfacedMemory] = []
    try:
        sub_input = SubconsciousInput(
            character_id=character.id,
            match_id=f"chat:{chat_session.id}",
            current_turn=0,
            board_summary=board,
            mood=mood,
            last_player_uci=None,
            last_player_chat=None,
            last_moves_san=[],
            recent_chat=[],
            opening_label=None,
            current_player_id=player.id,
            opponent_style_features=None,
        )
        surfaced = run_subconscious(session, character, sub_input) or []
    except Exception:
        logger.exception("[chat] greeting subconscious failed for session=%s", chat_session.id)

    opponent_ctx = _opponent_context(session, character, player)

    try:
        soul_input = SoulInput(
            board=board,
            mood=mood,
            surfaced_memories=surfaced,
            recent_chat=[],
            engine_move_san="",
            engine_move_uci="",
            engine_eval_cp=None,
            engine_considered=None,
            engine_time_ms=None,
            move_number=0,
            game_phase="pre-match",
            opponent_profile_summary=opponent_ctx,
            head_to_head=None,
            # Player has NOT spoken — this is a proactive greeting. The Soul
            # prompt's pre-match docs call for a one-line opener if the
            # character feels like speaking first.
            player_just_spoke=False,
            last_player_chat=None,
            match_id=f"chat:{chat_session.id}",
            character_color="white",
        )
        soul_resp = run_soul(character, soul_input)
    except Exception:
        logger.exception("[chat] greeting soul failed for session=%s", chat_session.id)
        return None

    if not soul_resp.speak:
        # Soul decided silence — don't force a bubble. Player has to say
        # hi first. That's fine; it's probabilistic by design.
        return None

    turn = _append_character_turn(
        session, chat_session, text=soul_resp.speak, soul_response=soul_resp,
    )
    # Persist mood evolution from this greeting so the next turn reads it.
    try:
        raw_next = apply_deltas(mood, soul_resp.mood_deltas.to_dict())
        smoothed = smooth_mood(mood, raw_next)
        save_mood(_chat_mood_key(chat_session.id), smoothed, smoothed=True)
    except Exception:
        logger.exception("[chat] greeting mood persist failed for session=%s", chat_session.id)
    logger.info(
        "[chat] character greeted first in session=%s (char=%s player=%s)",
        chat_session.id, character.id, player.id,
    )
    return turn


def close_session(session: Session, chat_session: CharacterChatSession) -> None:
    """Called when the player leaves the room without starting a game."""
    if chat_session.status != ChatSessionStatus.ACTIVE:
        return
    chat_session.status = ChatSessionStatus.ABANDONED
    chat_session.ended_at = datetime.utcnow()
    session.commit()
