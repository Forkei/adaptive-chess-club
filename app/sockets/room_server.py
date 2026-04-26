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
import time
from datetime import datetime
from http.cookies import SimpleCookie
from typing import Any
from urllib.parse import parse_qs

from app.agents.soul import SoulInput, _fallback_response, run_agent_soul_for_room, run_soul
from app.agents.subconscious import SubconsciousInput, run_subconscious
from app.auth import PLAYER_COOKIE
from app.db import SessionLocal
from app.models.character import Character, CharacterState, Visibility, rating_allowed
from app.models.chat import CharacterChatSession, ChatSessionStatus
from app.models.match import Player
from app.config import get_settings
from app.memory.inline_save import save_inline_memory
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
S2C_MATCH_VS_KENJI_STARTED = "match_vs_kenji_started"
S2C_GREETING_DONE = "greeting_done"
S2C_PLAYER_CHAT_RATE_LIMITED = "player_chat_rate_limited"
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


# Per-SID rate-limit tracker. Keyed by socket ID so no save_session await
# is needed, which avoids a yield-before-ack timing issue with the greeting task.
_room_chat_last_ms: dict[str, int] = {}


def _room_rate_limit_ok(sid: str) -> tuple[bool, int]:
    """Token-spacing check for the /room namespace. Returns (ok, retry_after_ms).

    Uses a module-level dict instead of sio_session so no await is needed,
    which keeps player_chat_ack emitted before the greeting task can fire
    agent_thinking.
    """
    min_interval_ms = int(get_settings().player_chat_min_interval_ms)
    now_ms = int(time.monotonic() * 1000)
    last = _room_chat_last_ms.get(sid, 0)
    delta = now_ms - last
    if delta < min_interval_ms:
        return False, min_interval_ms - delta
    _room_chat_last_ms[sid] = now_ms
    return True, 0


# --- Connect / disconnect --------------------------------------------------


@sio.on("connect", namespace=ROOM_NAMESPACE)
async def _on_connect(sid, environ, auth):
    player_id = _parse_cookie_header(environ)
    if not player_id:
        logger.info("Room socket rejected: no cookie (sid=%s)", sid)
        return False

    # Agent room mode: agent_id param takes priority over character_id.
    agent_id = _query_param(auth, environ, "agent_id")
    if agent_id:
        return await _on_connect_agent(sid, environ, auth, player_id, agent_id)

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
            "mode": "character",
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

    # Greeting: fires asynchronously so it never blocks the handshake.
    # _locked_fire_greeting acquires chat_session_lock before running so
    # two simultaneous connects to an empty session yield exactly one greeting.
    if is_empty:
        asyncio.create_task(
            _locked_fire_greeting(chat_session_id, character_id, player_id),
            name=f"greeting-{chat_session_id}",
        )


async def _on_connect_agent(
    sid: str,
    environ: dict,
    auth: Any,
    player_id: str,
    agent_id: str,
) -> bool | None:
    """Handle connect for agent room mode (/room?agent_id=...)."""
    with SessionLocal() as session:
        player = session.get(Player, player_id)
        if player is None:
            logger.info("Agent room rejected: invalid player cookie (sid=%s)", sid)
            return False

        from app.models.player_agent import PlayerAgent
        agent = session.get(PlayerAgent, agent_id)
        if agent is None or agent.archived_at is not None:
            logger.info("Agent room rejected: agent not found (sid=%s agent=%s)", sid, agent_id)
            return False
        if agent.owner_player_id != player.id:
            logger.info("Agent room rejected: not owner (sid=%s agent=%s)", sid, agent_id)
            return False

    await sio.save_session(
        sid,
        {
            "player_id": player_id,
            "agent_id": agent_id,
            "recent_chat": [],
            "mode": "agent",
        },
        namespace=ROOM_NAMESPACE,
    )

    await sio.emit(
        S2C_ROOM_STATE,
        {"agent_id": agent_id, "turns": []},
        to=sid,
        namespace=ROOM_NAMESPACE,
    )
    logger.info("Agent room connect sid=%s player=%s agent=%s", sid, player_id, agent_id)


@sio.on("disconnect", namespace=ROOM_NAMESPACE)
async def _on_disconnect(sid):
    _room_chat_last_ms.pop(sid, None)
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

    Emits agent_thinking {source: "greeting"} immediately so the client
    disables input and knows the character is about to speak first. When the
    LLM pipeline completes, emits either agent_chat (character spoke) or
    greeting_done (character stayed silent / player already sent a message).
    greeting_done is the definitive "greeting phase over" signal.
    """
    room = chat_room(chat_session_id)

    await sio.emit(
        S2C_AGENT_THINKING,
        {"source": "greeting"},
        room=room,
        namespace=ROOM_NAMESPACE,
    )

    greeted = False
    try:
        def _greet_sync():
            with SessionLocal() as s:
                chat_session = s.get(CharacterChatSession, chat_session_id)
                character = s.get(Character, character_id)
                player = s.get(Player, player_id)
                if chat_session is None or character is None or player is None:
                    return None
                return maybe_character_greets(s, chat_session, character, player)

        turn = await asyncio.to_thread(_greet_sync)

        if turn is not None:
            greeted = True
            await sio.emit(
                S2C_AGENT_CHAT,
                {
                    "text": turn.text,
                    "emotion": turn.emotion,
                    "emotion_intensity": turn.emotion_intensity,
                    "game_action": turn.game_action,
                    "source": "greeting",
                },
                room=room,
                namespace=ROOM_NAMESPACE,
            )
    except Exception:
        logger.exception("[room] greeting failed for session=%s (non-fatal)", chat_session_id)

    if not greeted:
        # Character stayed silent (probability miss, player beat us, or failure).
        # Signal the client so it unlocks the input.
        await sio.emit(S2C_GREETING_DONE, {}, room=room, namespace=ROOM_NAMESPACE)


async def _locked_fire_greeting(
    chat_session_id: str, character_id: str, player_id: str
) -> None:
    """Greeting entry point used by _on_connect.

    Acquires chat_session_lock before checking whether a greeting is still
    needed. Two simultaneous connects (two tabs, same session) both create a
    task here; only the first finds an empty session and proceeds. The second
    acquires the lock after the first finishes, sees a greeting turn already
    persisted, and returns early — emitting nothing extra to the room.
    """
    from app.concurrency.locks import chat_session_lock

    async with chat_session_lock(chat_session_id):
        with SessionLocal() as s:
            from app.characters.chat_service import get_turns
            cs = s.get(CharacterChatSession, chat_session_id)
            if cs is None:
                return
            if get_turns(s, cs):
                return  # Another connect already greeted.
        await _fire_greeting(chat_session_id, character_id, player_id)


async def _run_pipeline_locked(
    *,
    chat_session_id: str,
    character_id: str,
    player_id: str,
    player_text: str,
) -> None:
    """Serialize consecutive player_chat pipelines via chat_session_lock.

    Acquires the same lock as _locked_fire_greeting so a greeting in flight
    cannot interleave with a chat pipeline, and vice versa. If two messages
    arrive before the first pipeline commits, the second queues behind the
    first and runs in order — correct serial state, no racy mood/note writes.

    Starvation note: if a pipeline takes 8 s and three messages queue up,
    they process serially over 24 s. Correct behavior; the 30 s LLM timeout
    inside run_room_agent_pipeline is the backstop against infinite queuing.
    """
    from app.concurrency.locks import chat_session_lock

    async with chat_session_lock(chat_session_id):
        await run_room_agent_pipeline(
            chat_session_id=chat_session_id,
            character_id=character_id,
            player_id=player_id,
            player_text=player_text,
        )


# --- Event: player_chat ---------------------------------------------------


@sio.on(C2S_PLAYER_CHAT, namespace=ROOM_NAMESPACE)
async def _on_player_chat(sid, data):
    sess = await sio.get_session(sid, namespace=ROOM_NAMESPACE)
    mode = sess.get("mode", "character")

    if not isinstance(data, dict) or not data.get("text"):
        await _send_error(sid, "bad_payload", "Missing text field.")
        return

    text = str(data["text"]).strip()[:2000]
    if not text:
        await _send_error(sid, "bad_payload", "Empty message.")
        return

    # Rate limit: max 1 message per PLAYER_CHAT_MIN_INTERVAL_MS per socket.
    ok, retry_after = _room_rate_limit_ok(sid)
    if not ok:
        await sio.emit(
            S2C_PLAYER_CHAT_RATE_LIMITED,
            {"retry_after_ms": retry_after},
            to=sid,
            namespace=ROOM_NAMESPACE,
        )
        return

    if mode == "agent":
        agent_id = sess.get("agent_id")
        player_id = sess.get("player_id")
        recent_chat = list(sess.get("recent_chat") or [])
        if not agent_id or not player_id:
            await _send_error(sid, "no_session", "Socket has no bound agent session.")
            return

        import uuid as _uuid
        turn_id = str(_uuid.uuid4())
        created_at_iso = datetime.utcnow().isoformat() + "Z"
        await sio.emit(
            S2C_PLAYER_CHAT_ACK,
            {"turn_id": turn_id, "text": text, "created_at": created_at_iso},
            to=sid,
            namespace=ROOM_NAMESPACE,
        )

        new_chat = recent_chat + [f"player: {text}"]
        await sio.save_session(
            sid,
            {**sess, "recent_chat": new_chat[-20:]},
            namespace=ROOM_NAMESPACE,
        )

        asyncio.create_task(
            run_agent_room_pipeline(
                sid=sid,
                agent_id=agent_id,
                player_id=player_id,
                player_text=text,
                recent_chat=recent_chat,
            ),
            name=f"agent-pipeline-{agent_id}",
        )
        return

    # --- Character room mode (existing) ---
    chat_session_id = sess.get("chat_session_id")
    character_id = sess.get("character_id")
    player_id = sess.get("player_id")

    if not chat_session_id or not character_id or not player_id:
        await _send_error(sid, "no_session", "Socket has no bound chat session.")
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
        _run_pipeline_locked(
            chat_session_id=chat_session_id,
            character_id=character_id,
            player_id=player_id,
            player_text=text,
        ),
        name=f"pipeline-{chat_session_id}",
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

        # Load the opponent profile so Soul doesn't see "First time playing this opponent"
        # when there are already memories / match history for this player.
        from sqlalchemy import select as _select_room
        from app.models.match import OpponentProfile as _OpponentProfile
        _prof = session.execute(
            _select_room(_OpponentProfile)
            .where(_OpponentProfile.character_id == character_id)
            .where(_OpponentProfile.player_id == player_id)
        ).scalar_one_or_none()
        opponent_profile_summary: dict | None = None
        head_to_head: dict | None = None
        if _prof and _prof.games_played > 0:
            opponent_profile_summary = {
                "games_played": _prof.games_played,
                "narrative": _prof.narrative_summary or "(few interactions so far)",
            }
            head_to_head = {
                "character_wins": _prof.games_won_by_character,
                "character_losses": _prof.games_lost_by_character,
                "draws": _prof.games_drawn,
            }

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
            opponent_style_features=_prof.style_features if _prof else None,
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

        surfaced = await asyncio.wait_for(asyncio.to_thread(_sub_call), timeout=30.0)
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
                    opponent_profile_summary=opponent_profile_summary,
                    head_to_head=head_to_head,
                    player_just_spoke=True,
                    last_player_chat=player_text,
                    match_id=f"chat:{chat_session_id}",
                    character_color="white",
                )
                return run_soul(char, soul_input)

        soul_resp: SoulResponse = await asyncio.wait_for(asyncio.to_thread(_soul_call), timeout=30.0)
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

    # --- Inline memory (fire-and-forget) -------------------------------------
    if soul_resp.save_memory:
        asyncio.create_task(
            save_inline_memory(
                soul_resp.save_memory,
                character_id=character_id,
                player_id=player_id,
                match_id=None,
            )
        )

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


# --- Agent room pipeline --------------------------------------------------


async def run_agent_room_pipeline(
    *,
    sid: str,
    agent_id: str,
    player_id: str,
    player_text: str,
    recent_chat: list[str],
) -> None:
    """Async Subconscious → Soul pipeline for agent pre-match chat.

    Emits agent_thinking, agent_chat (with game_action), and on
    start_match_vs_kenji emits match_vs_kenji_started to the client.
    """
    await sio.emit(S2C_AGENT_THINKING, {}, to=sid, namespace=ROOM_NAMESPACE)

    from app.characters.chat_service import _idle_board_summary
    from app.director.mood import MoodState

    board = _idle_board_summary()
    mood = MoodState(aggression=0.5, confidence=0.5, tilt=0.0, engagement=0.5)
    chat_context = recent_chat + [f"player: {player_text}"]

    # --- Subconscious -------------------------------------------------------
    surfaced = []
    try:
        sub_input = SubconsciousInput(
            character_id="",  # ignored when agent_id is set
            agent_id=agent_id,
            match_id=f"agent-room:{agent_id}",
            current_turn=1,
            board_summary=board,
            mood=mood,
            last_player_uci=None,
            last_player_chat=player_text,
            last_moves_san=[],
            recent_chat=chat_context,
            opening_label=None,
            current_player_id=player_id,
        )

        def _sub_call():
            import types
            with SessionLocal() as s:
                from app.models.player_agent import PlayerAgent
                ag = s.get(PlayerAgent, agent_id)
                if ag is None:
                    return []
                char_proxy = types.SimpleNamespace(name=ag.name)
                return run_subconscious(s, char_proxy, sub_input) or []  # type: ignore[arg-type]

        surfaced = await asyncio.wait_for(asyncio.to_thread(_sub_call), timeout=30.0)
    except Exception:
        logger.exception("[room/agent] subconscious failed for agent=%s", agent_id)

    # --- Soul ---------------------------------------------------------------
    soul_resp: SoulResponse
    try:
        def _soul_call():
            with SessionLocal() as s:
                from app.models.player_agent import PlayerAgent
                ag = s.get(PlayerAgent, agent_id)
                if ag is None:
                    return _fallback_response()
                from app.agents.prompts import build_agent_system_prompt
                system = build_agent_system_prompt(ag)
                soul_inp = SoulInput(
                    board=board,
                    mood=mood,
                    surfaced_memories=surfaced,
                    recent_chat=chat_context,
                    engine_move_san="",
                    engine_move_uci="",
                    engine_eval_cp=None,
                    engine_considered=None,
                    engine_time_ms=None,
                    move_number=0,
                    game_phase="pre-match",
                    player_just_spoke=True,
                    last_player_chat=player_text,
                    match_id=f"agent-room:{agent_id}",
                    character_color="white",
                )
                return run_agent_soul_for_room(system, soul_inp)

        soul_resp = await asyncio.wait_for(asyncio.to_thread(_soul_call), timeout=30.0)
    except Exception:
        logger.exception("[room/agent] soul failed for agent=%s", agent_id)
        soul_resp = _fallback_response()

    # --- Inline memory (fire-and-forget) ------------------------------------
    if soul_resp.save_memory:
        asyncio.create_task(
            save_inline_memory(
                soul_resp.save_memory,
                agent_id=agent_id,
                player_id=player_id,
                match_id=None,
            )
        )

    # --- Emit agent_chat ----------------------------------------------------
    await sio.emit(
        S2C_AGENT_CHAT,
        {
            "text": soul_resp.speak or "…",
            "emotion": soul_resp.emotion,
            "emotion_intensity": soul_resp.emotion_intensity,
            "game_action": soul_resp.game_action,
        },
        to=sid,
        namespace=ROOM_NAMESPACE,
    )

    # --- Handle start_match_vs_kenji ----------------------------------------
    if soul_resp.game_action == "start_match_vs_kenji":
        try:
            def _create_match():
                from app.matches.service import create_agent_match
                with SessionLocal() as s:
                    match = create_agent_match(
                        s,
                        agent_id=agent_id,
                        player_id=player_id,
                        character_preset_key="kenji_sato",
                    )
                    match_id = match.id
                    s.commit()
                return match_id

            match_id = await asyncio.to_thread(_create_match)
            await sio.emit(
                S2C_MATCH_VS_KENJI_STARTED,
                {
                    "match_id": match_id,
                    "redirect_url": f"/matches/{match_id}",
                },
                to=sid,
                namespace=ROOM_NAMESPACE,
            )
            logger.info(
                "[room/agent] match_vs_kenji_started agent=%s player=%s match=%s",
                agent_id, player_id, match_id,
            )
            # Launch the automated match loop (Commit 5).
            from app.matches.agent_streaming import run_agent_match_loop
            asyncio.create_task(
                run_agent_match_loop(match_id, agent_id),
                name=f"agent-loop-{match_id}",
            )
        except Exception:
            logger.exception("[room/agent] match creation failed for agent=%s", agent_id)
            await sio.emit(
                S2C_AGENT_ERROR,
                {"message": "Failed to start the match. Try telling your agent to go again."},
                to=sid,
                namespace=ROOM_NAMESPACE,
            )
