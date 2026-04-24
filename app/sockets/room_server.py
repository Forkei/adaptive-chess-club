"""Socket.IO /room namespace — streaming pre-match chat (Phase 4.4 Block 2).

Replaces the synchronous HTTP chat path with a streaming Socket.IO pipeline.
The client connects here when entering a character's detail page. All chat
flows via socket events; HTTP POST /chat stays as a 202 fallback for
clients that can't use sockets.

Namespace: /room
Room per session: chat:{chat_session_id}

Client → Server events:
    player_chat  {text: string}

Server → Client events:
    room_state         — on connect; full history + session metadata
    player_chat_ack    — immediately after player_chat: {turn_id, text, created_at}
    agent_thinking     — before Subconscious/Soul pipeline starts
    agent_chat         — Soul response: {text, emotion, emotion_intensity, game_action}
    agent_error        — pipeline failure: {message}
    game_started       — Soul emitted start_game: {match_id, redirect_url}
    error              — auth/protocol errors: {code, message}
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from http.cookies import SimpleCookie
from typing import Any
from urllib.parse import parse_qs

from app.agents.soul import SoulInput, run_soul
from app.agents.subconscious import SubconsciousInput, run_subconscious
from app.auth import PLAYER_COOKIE
from app.db import SessionLocal
from app.models.character import Character, CharacterState, Visibility, rating_allowed
from app.models.chat import CharacterChatSession, ChatSessionStatus
from app.models.match import Player
from app.schemas.agents import SoulResponse
from app.sockets.server import sio

logger = logging.getLogger(__name__)

ROOM_NAMESPACE = "/room"


def chat_room(session_id: str) -> str:
    return f"chat:{session_id}"


# --- Event constants -------------------------------------------------------

C2S_PLAYER_CHAT = "player_chat"
S2C_ROOM_STATE = "room_state"
S2C_PLAYER_CHAT_ACK = "player_chat_ack"
S2C_AGENT_THINKING = "agent_thinking"
S2C_AGENT_CHAT = "agent_chat"
S2C_AGENT_ERROR = "agent_error"
S2C_GAME_STARTED = "game_started"
S2C_ERROR = "error"


# --- Auth helpers (same pattern as lobby_server.py) -----------------------


def _parse_cookie_header(environ: dict[str, Any]) -> str | None:
    raw = environ.get("HTTP_COOKIE")
    if not raw:
        return None
    try:
        jar = SimpleCookie()
        jar.load(raw)
    except Exception:
        return None
    morsel = jar.get(PLAYER_COOKIE)
    return morsel.value if morsel else None


def _query_param(auth: Any, environ: dict[str, Any], key: str) -> str | None:
    if isinstance(auth, dict) and auth.get(key):
        return str(auth[key])
    qs = environ.get("QUERY_STRING", "")
    if not qs:
        return None
    parsed = parse_qs(qs)
    vals = parsed.get(key)
    return vals[0] if vals else None


async def _send_error(sid: str, code: str, message: str) -> None:
    await sio.emit(
        S2C_ERROR,
        {"code": code, "message": message},
        to=sid,
        namespace=ROOM_NAMESPACE,
    )


# --- Connect / disconnect --------------------------------------------------


@sio.on("connect", namespace=ROOM_NAMESPACE)
async def _on_connect(sid, environ, auth):
    player_id = _parse_cookie_header(environ)
    if not player_id:
        logger.info("Room socket rejected: no cookie (sid=%s)", sid)
        return False

    character_id = _query_param(auth, environ, "character_id")
    if not character_id:
        logger.info("Room socket rejected: no character_id (sid=%s)", sid)
        return False

    with SessionLocal() as session:
        player = session.get(Player, player_id)
        if player is None:
            logger.info("Room socket rejected: invalid player cookie (sid=%s)", sid)
            return False

        character = session.get(Character, character_id)
        if character is None or character.deleted_at is not None:
            logger.info("Room socket rejected: character not found (sid=%s char=%s)", sid, character_id)
            return False
        if character.visibility == Visibility.PRIVATE and character.owner_id != player.id:
            logger.info("Room socket rejected: character private (sid=%s)", sid)
            return False
        if not rating_allowed(character.content_rating, player.max_content_rating):
            logger.info("Room socket rejected: content rating (sid=%s)", sid)
            return False
        if character.state != CharacterState.READY:
            await _send_error(sid, "character_not_ready", "Character is still preparing.")
            return False

        from app.characters.chat_service import (
            get_or_create_session as _get_chat_session,
            get_turns,
        )

        chat_session = _get_chat_session(session, character=character, player=player)
        turns = get_turns(session, chat_session)
        is_empty = len(turns) == 0
        turns_data = [
            {
                "id": t.id,
                "turn_number": t.turn_number,
                "role": t.role.value,
                "text": t.text,
                "emotion": t.emotion,
                "emotion_intensity": t.emotion_intensity,
                "game_action": t.game_action,
                "created_at": t.created_at.isoformat() + "Z",
            }
            for t in turns
        ]
        room_state_payload = {
            "session_id": chat_session.id,
            "status": chat_session.status.value,
            "handed_off_match_id": chat_session.handed_off_match_id,
            "character": {
                "id": character.id,
                "name": character.name,
                "avatar_emoji": character.avatar_emoji,
                "state": character.state.value,
            },
            "turns": turns_data,
        }
        chat_session_id = chat_session.id

    await sio.save_session(
        sid,
        {
            "player_id": player_id,
            "character_id": character_id,
            "chat_session_id": chat_session_id,
        },
        namespace=ROOM_NAMESPACE,
    )
    await sio.enter_room(sid, chat_room(chat_session_id), namespace=ROOM_NAMESPACE)

    await sio.emit(
        S2C_ROOM_STATE,
        room_state_payload,
        to=sid,
        namespace=ROOM_NAMESPACE,
    )
    logger.info(
        "Room socket connect sid=%s player=%s character=%s session=%s",
        sid, player_id, character_id, chat_session_id,
    )

    # Greeting: fires asynchronously AFTER room_state so it never blocks the
    # handshake. The greeting race (Block 3) is unchanged — we just moved the
    # trigger from HTTP GET to here without adding any lock.
    if is_empty:
        asyncio.create_task(
            _fire_greeting(chat_session_id, character_id, player_id),
            name=f"greeting-{chat_session_id}",
        )


@sio.on("disconnect", namespace=ROOM_NAMESPACE)
async def _on_disconnect(sid):
    try:
        sess = await sio.get_session(sid, namespace=ROOM_NAMESPACE)
    except KeyError:
        return
    logger.info(
        "Room socket disconnect sid=%s session=%s",
        sid, sess.get("chat_session_id"),
    )


# --- Greeting background task ----------------------------------------------


def maybe_character_greets(*args, **kwargs):
    """Module-level shim so tests can monkeypatch 'app.sockets.room_server.maybe_character_greets'."""
    from app.characters.chat_service import maybe_character_greets as _real
    return _real(*args, **kwargs)


async def _fire_greeting(chat_session_id: str, character_id: str, player_id: str) -> None:
    """Run the character greeting asynchronously after room_state is sent.

    Uses asyncio.to_thread so the synchronous Soul/Subconscious calls don't
    block the event loop. Any greeting turn will arrive at the client as an
    agent_chat event (since it's appended to the session, not included in
    the initial room_state).
    """
    def _greet_sync():
        with SessionLocal() as s:
            chat_session = s.get(CharacterChatSession, chat_session_id)
            character = s.get(Character, character_id)
            player = s.get(Player, player_id)
            if chat_session is None or character is None or player is None:
                return None
            return maybe_character_greets(s, chat_session, character, player)

    try:
        turn = await asyncio.to_thread(_greet_sync)
    except Exception:
        logger.exception("[room] greeting failed for session=%s (non-fatal)", chat_session_id)
        return

    if turn is not None:
        await sio.emit(
            S2C_AGENT_CHAT,
            {
                "text": turn.text,
                "emotion": turn.emotion,
                "emotion_intensity": turn.emotion_intensity,
                "game_action": turn.game_action,
            },
            room=chat_room(chat_session_id),
            namespace=ROOM_NAMESPACE,
        )


# --- Event: player_chat ---------------------------------------------------


@sio.on(C2S_PLAYER_CHAT, namespace=ROOM_NAMESPACE)
async def _on_player_chat(sid, data):
    sess = await sio.get_session(sid, namespace=ROOM_NAMESPACE)
    chat_session_id = sess.get("chat_session_id")
    character_id = sess.get("character_id")
    player_id = sess.get("player_id")

    if not chat_session_id or not character_id or not player_id:
        await _send_error(sid, "no_session", "Socket has no bound chat session.")
        return

    if not isinstance(data, dict) or not data.get("text"):
        await _send_error(sid, "bad_payload", "Missing text field.")
        return

    text = str(data["text"]).strip()[:2000]
    if not text:
        await _send_error(sid, "bad_payload", "Empty message.")
        return

    # Persist player turn immediately, emit ack, then fire background pipeline.
    with SessionLocal() as session:
        chat_session = session.get(CharacterChatSession, chat_session_id)
        if chat_session is None or chat_session.status != ChatSessionStatus.ACTIVE:
            await _send_error(sid, "session_not_active", "Chat session is not active.")
            return

        from app.characters.chat_service import _append_player_turn
        player_turn = _append_player_turn(session, chat_session, text)
        turn_id = player_turn.id
        created_at_iso = player_turn.created_at.isoformat() + "Z"

    await sio.emit(
        S2C_PLAYER_CHAT_ACK,
        {"turn_id": turn_id, "text": text, "created_at": created_at_iso},
        to=sid,
        namespace=ROOM_NAMESPACE,
    )

    asyncio.create_task(
        run_room_agent_pipeline(
            chat_session_id=chat_session_id,
            character_id=character_id,
            player_id=player_id,
            player_text=text,
        )
    )


# --- Background pipeline --------------------------------------------------


async def run_room_agent_pipeline(
    *,
    chat_session_id: str,
    character_id: str,
    player_id: str,
    player_text: str,
) -> None:
    """Async Subconscious → Soul pipeline for a pre-match chat turn.

    Emits agent_thinking before starting, agent_chat when the Soul responds,
    and game_started if the Soul emits game_action="start_game". Each DB
    operation uses its own SessionLocal() context.

    Called by the socket `player_chat` handler (via asyncio.create_task) and
    by the HTTP fallback POST /characters/{id}/chat path (also as a task).
    """
    room = chat_room(chat_session_id)

    await sio.emit(
        S2C_AGENT_THINKING,
        {},
        room=room,
        namespace=ROOM_NAMESPACE,
    )

    # --- Load context -------------------------------------------------------
    with SessionLocal() as session:
        chat_session = session.get(CharacterChatSession, chat_session_id)
        if chat_session is None or chat_session.status != ChatSessionStatus.ACTIVE:
            logger.warning("[room] pipeline aborted: session %s not active", chat_session_id)
            return

        from app.characters.chat_service import (
            _chat_mood_key,
            _idle_board_summary,
            _load_or_init_chat_mood,
            _recent_chat_lines,
            get_turns,
        )

        turns = get_turns(session, chat_session)
        # recent_chat excludes the player turn we just appended (the last one).
        recent_chat = _recent_chat_lines(turns[:-1]) if turns else []
        board = _idle_board_summary()
        mood = _load_or_init_chat_mood(session, chat_session, session.get(Character, character_id))

        sub_input = SubconsciousInput(
            character_id=character_id,
            match_id=f"chat:{chat_session_id}",
            current_turn=turns[-1].turn_number if turns else 1,
            board_summary=board,
            mood=mood,
            last_player_uci=None,
            last_player_chat=player_text,
            last_moves_san=[],
            recent_chat=recent_chat + [f"player: {player_text}"],
            opening_label=None,
            current_player_id=player_id,
            opponent_style_features=None,
        )

    # --- Subconscious -------------------------------------------------------
    surfaced = []
    try:
        def _sub_call():
            with SessionLocal() as s:
                char = s.get(Character, character_id)
                if char is None:
                    return []
                return run_subconscious(s, char, sub_input) or []

        surfaced = await asyncio.to_thread(_sub_call)
    except Exception:
        logger.exception("[room] subconscious failed for session=%s", chat_session_id)

    # --- Soul ---------------------------------------------------------------
    try:
        def _soul_call():
            with SessionLocal() as s:
                char = s.get(Character, character_id)
                if char is None:
                    return SoulResponse(speak=None, emotion="neutral", emotion_intensity=0.2)
                soul_input = SoulInput(
                    board=board,
                    mood=mood,
                    surfaced_memories=surfaced,
                    recent_chat=recent_chat + [f"player: {player_text}"],
                    engine_move_san="",
                    engine_move_uci="",
                    engine_eval_cp=None,
                    engine_considered=None,
                    engine_time_ms=None,
                    move_number=0,
                    game_phase="pre-match",
                    opponent_profile_summary=None,
                    head_to_head=None,
                    player_just_spoke=True,
                    last_player_chat=player_text,
                    match_id=f"chat:{chat_session_id}",
                    character_color="white",
                )
                return run_soul(char, soul_input)

        soul_resp: SoulResponse = await asyncio.to_thread(_soul_call)
    except Exception:
        logger.exception("[room] soul failed for session=%s", chat_session_id)
        soul_resp = SoulResponse(speak=None, emotion="neutral", emotion_intensity=0.2)

    # --- Persist character turn + update mood --------------------------------
    try:
        with SessionLocal() as session:
            chat_session = session.get(CharacterChatSession, chat_session_id)
            if chat_session is None:
                return

            from app.characters.chat_service import _append_character_turn
            reply_text = soul_resp.speak or "…"
            _append_character_turn(session, chat_session, text=reply_text, soul_response=soul_resp)

            if soul_resp.note_about_opponent:
                chat_session.pending_notes = (chat_session.pending_notes or []) + [
                    soul_resp.note_about_opponent
                ]
                session.commit()

        # Compound mood deltas — same logic as the HTTP path.
        from app.director.mood import apply_deltas, save_mood, smooth_mood
        from app.characters.chat_service import _chat_mood_key as _mood_key

        try:
            raw_next = apply_deltas(mood, soul_resp.mood_deltas.to_dict())
            new_smoothed = smooth_mood(mood, raw_next)
            save_mood(_mood_key(chat_session_id), new_smoothed, smoothed=True)
        except Exception:
            logger.exception("[room] mood persist failed for session=%s", chat_session_id)

    except Exception:
        logger.exception("[room] character turn persist failed for session=%s", chat_session_id)

    # --- Emit agent_chat -----------------------------------------------------
    await sio.emit(
        S2C_AGENT_CHAT,
        {
            "text": soul_resp.speak or "…",
            "emotion": soul_resp.emotion,
            "emotion_intensity": soul_resp.emotion_intensity,
            "game_action": soul_resp.game_action,
        },
        room=room,
        namespace=ROOM_NAMESPACE,
    )

    # --- Hand off to match if Soul said start_game ---------------------------
    if soul_resp.game_action == "start_game":
        try:
            with SessionLocal() as session:
                chat_session = session.get(CharacterChatSession, chat_session_id)
                character = session.get(Character, character_id)
                player = session.get(Player, player_id)
                if chat_session is None or character is None or player is None:
                    return

                from app.characters.chat_service import _hand_off_to_match
                match_id = _hand_off_to_match(session, chat_session, character, player)

            await sio.emit(
                S2C_GAME_STARTED,
                {
                    "match_id": match_id,
                    "redirect_url": f"/matches/{match_id}",
                },
                room=room,
                namespace=ROOM_NAMESPACE,
            )
            logger.info(
                "[room] game_started emitted session=%s match=%s",
                chat_session_id, match_id,
            )
        except Exception:
            logger.exception("[room] hand-off failed for session=%s", chat_session_id)
            await sio.emit(
                S2C_AGENT_ERROR,
                {"message": "Failed to start the game. Try saying 'let's play' again."},
                room=room,
                namespace=ROOM_NAMESPACE,
            )
