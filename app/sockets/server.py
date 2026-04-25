"""Socket.IO server: /play namespace, cookie auth, per-match rooms, event handlers.

Architectural notes:

- Single `AsyncServer` (`async_mode="asgi"`) mounted alongside the FastAPI app.
- One namespace (`/play`) for gameplay. The client joins with `match_id` as a query param.
- One room per match (`match:<match_id>`). Server-driven `emit(..., room=...)` broadcasts to
  all sockets attached to that match (in 3b that's just the one player; 3c will add spectators).
- Cookie auth: the Socket.IO handshake carries the browser's `Cookie` header, we parse
  `player_id`, resolve the `Player`, and bind it to the Socket.IO session. No cookie ⇒
  reject connection.
- Content rating changes mid-match do NOT cut the connection — the choice was implicit when
  the player clicked "play". (Section F of the phase brief.)
- Single-worker assumption: disconnect timers + per-match room membership live in-process.
  Multi-worker needs Redis pub/sub, which is deferred to Phase 4.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime
from http.cookies import SimpleCookie
from typing import Any

import socketio
from sqlalchemy import select

from app.auth import PLAYER_COOKIE
from app.config import get_settings
from app.db import SessionLocal
from app.models.character import Character, Visibility, rating_allowed
from app.models.match import Match, MatchStatus, Move, Player
from app.sockets import disconnect as disconnect_registry
from app.sockets.events import (
    C2S_MAKE_MOVE,
    C2S_PING,
    C2S_PLAYER_CHAT,
    C2S_REQUEST_STATE,
    C2S_RESIGN,
    C2S_SPECTATOR_CHAT,
    NAMESPACE,
    S2C_AGENT_CHAT,
    S2C_AGENT_MOVE,
    S2C_AGENT_THINKING,
    S2C_ERROR,
    S2C_MATCH_ENDED,
    S2C_MATCH_PAUSED,
    S2C_MATCH_RESUMED,
    S2C_MATCH_STATE,
    S2C_MEMORY_SURFACED,
    S2C_MOOD_UPDATE,
    S2C_PLAYER_CHAT_BROADCAST,
    S2C_PLAYER_CHAT_ECHOED,
    S2C_PLAYER_CHAT_RATE_LIMITED,
    S2C_PLAYER_MOVE_APPLIED,
    S2C_PONG,
    S2C_SPECTATOR_CHAT_BROADCAST,
    S2C_SPECTATOR_CHAT_ECHOED,
    S2C_SPECTATOR_CHAT_REJECTED,
    S2C_SPECTATOR_COUNT,
    S2C_SPECTATOR_JOINED,
    S2C_SPECTATOR_LEFT,
    AgentChatPayload,
    AgentMovePayload,
    AgentThinkingPayload,
    ErrorPayload,
    MakeMoveEvent,
    MatchEndedPayload,
    MatchPausedPayload,
    MatchResumedPayload,
    MatchStatePayload,
    MemorySurfacedItem,
    MemorySurfacedPayload,
    MoodUpdatePayload,
    MoveSnapshot,
    PlayerChatBroadcastPayload,
    PlayerChatEchoedPayload,
    PlayerChatEvent,
    PlayerChatRateLimitedPayload,
    PlayerMoveAppliedPayload,
    PongPayload,
    SpectatorChatBroadcastPayload,
    SpectatorChatEvent,
    SpectatorChatRejectedPayload,
    SpectatorCountPayload,
    SpectatorJoinedPayload,
    SpectatorLeftPayload,
    match_room_name,
)

logger = logging.getLogger(__name__)


# --- Server instance -------------------------------------------------------

sio = socketio.AsyncServer(
    async_mode="asgi",
    # Same-origin only: play.html is served from the same FastAPI app.
    cors_allowed_origins=[],
    logger=False,
    engineio_logger=False,
)


def match_room(match_id: str) -> str:
    return match_room_name(match_id)


def build_asgi_app(fastapi_app):
    """Wrap a FastAPI app in Socket.IO's ASGI envelope.

    The returned ASGI app delegates WebSocket / socket.io traffic to `sio` and
    everything else (HTTP, static) to `fastapi_app`. Use this as the uvicorn entry.
    """
    return socketio.ASGIApp(sio, other_asgi_app=fastapi_app)


# --- Helpers ---------------------------------------------------------------


def _parse_cookie_header(environ: dict[str, Any]) -> str | None:
    raw = environ.get("HTTP_COOKIE")
    if not raw:
        return None
    cookie = SimpleCookie()
    try:
        cookie.load(raw)
    except Exception:
        return None
    morsel = cookie.get(PLAYER_COOKIE)
    return morsel.value if morsel else None


def _resolve_player(session, player_id: str) -> Player | None:
    return session.get(Player, player_id)


def _resolve_match(session, match_id: str) -> Match | None:
    return session.get(Match, match_id)


def _query_param(auth: Any, environ: dict[str, Any], key: str) -> str | None:
    # socket.io-client sends query via the HTTP handshake. python-socketio exposes
    # the query string in environ['QUERY_STRING']. `auth` carries anything the client
    # put in `socketio.connect(..., auth={...})`.
    if isinstance(auth, dict) and auth.get(key):
        return str(auth[key])
    qs = environ.get("QUERY_STRING", "")
    if not qs:
        return None
    from urllib.parse import parse_qs

    parsed = parse_qs(qs)
    vals = parsed.get(key)
    return vals[0] if vals else None


async def _send_error(sid: str, code: str, message: str) -> None:
    await sio.emit(
        S2C_ERROR,
        ErrorPayload(code=code, message=message).model_dump(mode="json"),
        to=sid,
        namespace=NAMESPACE,
    )


def _move_to_snapshot(m: Move) -> MoveSnapshot:
    return MoveSnapshot(
        move_number=m.move_number,
        side="white" if m.side.value == "white" else "black",
        uci=m.uci,
        san=m.san,
        fen_after=m.fen_after,
        engine_name=m.engine_name,
        time_taken_ms=m.time_taken_ms,
        eval_cp=m.eval_cp,
        player_chat_before=m.player_chat_before,
        agent_chat_after=m.agent_chat_after,
    )


def _build_match_state(session, match: Match) -> MatchStatePayload:
    moves = [
        _move_to_snapshot(m)
        for m in sorted(match.moves, key=lambda x: x.move_number)
    ]
    last_agent = next(
        (m for m in reversed(moves) if m.agent_chat_after), None
    )
    # Mood snapshot — pull the last persisted move's mood_snapshot if present, else empty.
    mood_snapshot: dict[str, float] = {}
    for m in reversed(list(match.moves)):
        if m.mood_snapshot:
            mood_snapshot = {k: float(v) for k, v in m.mood_snapshot.items() if isinstance(v, (int, float))}
            break

    deadline = disconnect_registry.deadline_for(match.id)
    return MatchStatePayload(
        match_id=match.id,
        status=match.status.value,  # type: ignore[arg-type]
        result=match.result.value if match.result else None,  # type: ignore[arg-type]
        player_color=match.player_color.value,  # type: ignore[arg-type]
        current_fen=match.current_fen,
        move_count=match.move_count,
        moves=moves,
        mood=mood_snapshot,
        last_agent_chat=last_agent.agent_chat_after if last_agent else None,
        last_emotion=None,
        disconnect_cooldown_seconds=disconnect_registry.cooldown_seconds() if deadline else None,
        disconnect_deadline=deadline,
    )


# --- Rate-limit + pending-chat buffer helpers -----------------------------


def _rate_limit_ok(sio_session: dict[str, Any]) -> tuple[bool, int]:
    """Token-spacing check. Returns (ok, retry_after_ms)."""
    settings = get_settings()
    min_interval_ms = int(settings.player_chat_min_interval_ms)
    now_ms = int(time.monotonic() * 1000)
    last = sio_session.get("last_chat_ms", 0)
    delta = now_ms - last
    if delta < min_interval_ms:
        return False, min_interval_ms - delta
    sio_session["last_chat_ms"] = now_ms
    return True, 0


def _append_pending_chat(match: Match, text: str) -> None:
    """Append `text` to match.extra_state.pending_player_chat with FIFO caps."""
    settings = get_settings()
    max_msgs = int(settings.pending_chat_max_messages)
    max_chars = int(settings.pending_chat_max_chars)

    state = dict(match.extra_state or {})
    buf = list(state.get("pending_player_chat", []))
    buf.append({"text": text, "ts": datetime.utcnow().isoformat()})

    # FIFO evict by message count.
    evicted = 0
    while len(buf) > max_msgs:
        buf.pop(0)
        evicted += 1
    # FIFO evict by total chars.
    def _total_chars(b):
        return sum(len(entry.get("text", "")) for entry in b)
    while buf and _total_chars(buf) > max_chars:
        buf.pop(0)
        evicted += 1
    if evicted:
        logger.debug(
            "pending_player_chat: evicted %d message(s) for match=%s (caps=%d msgs / %d chars)",
            evicted, match.id, max_msgs, max_chars,
        )

    state["pending_player_chat"] = buf
    match.extra_state = state


# --- Role helpers ---------------------------------------------------------

ROLE_PARTICIPANT = "participant"
ROLE_SPECTATOR = "spectator"

# Per-match spectator count cache. The server has the room membership already;
# this mirror keeps counts cheap to broadcast.
_spectator_counts: dict[str, int] = {}


def _spectator_count(match_id: str) -> int:
    return _spectator_counts.get(match_id, 0)


def _bump_spectator_count(match_id: str, delta: int) -> int:
    new = max(0, _spectator_counts.get(match_id, 0) + delta)
    _spectator_counts[match_id] = new
    return new


# --- Connect / disconnect --------------------------------------------------


@sio.on("connect", namespace=NAMESPACE)
async def _on_connect(sid, environ, auth):
    player_id = _parse_cookie_header(environ)
    if not player_id:
        logger.info("Socket.IO rejected: no cookie (sid=%s)", sid)
        return False

    match_id = _query_param(auth, environ, "match_id")
    if not match_id:
        logger.info("Socket.IO rejected: no match_id param (sid=%s)", sid)
        return False

    username = ""
    role = ROLE_PARTICIPANT
    with SessionLocal() as session:
        player = _resolve_player(session, player_id)
        if player is None:
            logger.info("Socket.IO rejected: invalid player cookie (sid=%s)", sid)
            return False
        username = player.username

        match = _resolve_match(session, match_id)
        if match is None:
            logger.info("Socket.IO rejected: match not found (sid=%s match=%s)", sid, match_id)
            return False

        if match.player_id == player.id:
            role = ROLE_PARTICIPANT
        else:
            # Spectator gate: character must be visible to the viewer.
            character = session.get(Character, match.character_id)
            if character is None or character.deleted_at is not None:
                logger.info("Socket.IO rejected spectator: character missing (sid=%s)", sid)
                return False
            if character.visibility == Visibility.PRIVATE and character.owner_id != player.id:
                logger.info(
                    "Socket.IO rejected spectator: character %s private (sid=%s player=%s)",
                    character.id, sid, player.id,
                )
                return False
            if not rating_allowed(character.content_rating, player.max_content_rating):
                logger.info(
                    "Socket.IO rejected spectator: rating (sid=%s player=%s)", sid, player.id
                )
                return False
            # 3b spec: abandoned matches aren't spectatable.
            if match.status == MatchStatus.ABANDONED:
                logger.info("Socket.IO rejected spectator: match abandoned (sid=%s)", sid)
                return False
            role = ROLE_SPECTATOR

        state_payload = _build_match_state(session, match)

    await sio.save_session(
        sid,
        {
            "player_id": player_id,
            "match_id": match_id,
            "username": username,
            "role": role,
            "last_chat_ms": 0,
        },
        namespace=NAMESPACE,
    )
    await sio.enter_room(sid, match_room(match_id), namespace=NAMESPACE)

    # Cancel any in-flight disconnect cooldown — the participant is back.
    resumed = False
    if role == ROLE_PARTICIPANT:
        resumed = disconnect_registry.cancel(match_id)

    await sio.emit(
        S2C_MATCH_STATE,
        state_payload.model_dump(mode="json"),
        to=sid,
        namespace=NAMESPACE,
    )
    if resumed:
        await sio.emit(
            S2C_MATCH_RESUMED,
            MatchResumedPayload(match_id=match_id).model_dump(mode="json"),
            room=match_room(match_id),
            namespace=NAMESPACE,
        )
        # Clear disconnect markers on the match row.
        with SessionLocal() as session:
            match = _resolve_match(session, match_id)
            if match is not None:
                state = dict(match.extra_state or {})
                state.pop("disconnect_started_at", None)
                state.pop("disconnect_player_id", None)
                match.extra_state = state
                session.commit()

    if role == ROLE_SPECTATOR:
        count = _bump_spectator_count(match_id, +1)
        await sio.emit(
            S2C_SPECTATOR_JOINED,
            SpectatorJoinedPayload(username=username).model_dump(mode="json"),
            room=match_room(match_id),
            namespace=NAMESPACE,
        )
        await sio.emit(
            S2C_SPECTATOR_COUNT,
            SpectatorCountPayload(count=count).model_dump(mode="json"),
            room=match_room(match_id),
            namespace=NAMESPACE,
        )
    else:
        # Let the just-connected participant know how many spectators are already watching.
        count = _spectator_count(match_id)
        if count:
            await sio.emit(
                S2C_SPECTATOR_COUNT,
                SpectatorCountPayload(count=count).model_dump(mode="json"),
                to=sid,
                namespace=NAMESPACE,
            )

    # Commit 5: if the character plays white and the match is fresh (move_count==0),
    # fire the opening engine turn as a background task. The player's board will
    # then show agent_thinking → agent_move → agent_chat via the normal streaming
    # path rather than blocking here.
    if (
        role == ROLE_PARTICIPANT
        and state_payload.status == "in_progress"
        and state_payload.move_count == 0
        and state_payload.player_color == "black"  # character is white
    ):
        asyncio.create_task(_fire_opening_move(match_id), name=f"opening-{match_id}")

    logger.info(
        "Socket.IO connect sid=%s player=%s match=%s role=%s", sid, player_id, match_id, role,
    )


async def _fire_opening_move(match_id: str) -> None:
    """Background task: plays the character's opening move when they have white.

    Acquires match_lock so two near-simultaneous connects (e.g. page refresh
    before the engine responds) cannot both fire the opening move. The re-check
    inside the lock ensures the first task to finish wins; the second sees
    move_count > 0 and exits without touching the engine or agents.
    """
    from app.concurrency.locks import match_lock
    from app.config import get_settings
    from app.matches.streaming import _run_engine_and_agents

    emitters = _build_turn_emitters(match_id)
    async with match_lock(match_id):
        with SessionLocal() as session:
            match = _resolve_match(session, match_id)
            if match is None or match.move_count > 0:
                return  # Already played by a parallel task.
        try:
            await _run_engine_and_agents(
                match_id=match_id, emitters=emitters, settings=get_settings()
            )
        except Exception:
            logger.exception("Opening move background task failed for match=%s", match_id)


@sio.on("disconnect", namespace=NAMESPACE)
async def _on_disconnect(sid):
    try:
        session_dict = await sio.get_session(sid, namespace=NAMESPACE)
    except KeyError:
        return
    match_id = session_dict.get("match_id")
    player_id = session_dict.get("player_id")
    role = session_dict.get("role", ROLE_PARTICIPANT)
    username = session_dict.get("username", "")
    if not match_id or not player_id:
        return

    if role == ROLE_SPECTATOR:
        count = _bump_spectator_count(match_id, -1)
        await sio.emit(
            S2C_SPECTATOR_LEFT,
            SpectatorLeftPayload(username=username).model_dump(mode="json"),
            room=match_room(match_id),
            namespace=NAMESPACE,
        )
        await sio.emit(
            S2C_SPECTATOR_COUNT,
            SpectatorCountPayload(count=count).model_dump(mode="json"),
            room=match_room(match_id),
            namespace=NAMESPACE,
        )
        logger.info("Socket.IO spectator disconnect sid=%s match=%s", sid, match_id)
        return

    # Participant disconnect: arm the cooldown if the match is still active.
    with SessionLocal() as session:
        match = _resolve_match(session, match_id)
        if match is None or match.status != MatchStatus.IN_PROGRESS:
            return
        if match.player_id != player_id:
            return
        state = dict(match.extra_state or {})
        state["disconnect_started_at"] = datetime.utcnow().isoformat()
        state["disconnect_player_id"] = player_id
        match.extra_state = state
        session.commit()

    deadline = disconnect_registry.start(
        match_id,
        player_id=player_id,
        on_timeout=_on_disconnect_timeout,
    )
    await sio.emit(
        S2C_MATCH_PAUSED,
        MatchPausedPayload(
            match_id=match_id,
            deadline=deadline,
            cooldown_seconds=disconnect_registry.cooldown_seconds(),
        ).model_dump(mode="json"),
        room=match_room(match_id),
        namespace=NAMESPACE,
    )
    logger.info("Socket.IO participant disconnect sid=%s match=%s — cooldown armed", sid, match_id)


async def _on_disconnect_timeout(match_id: str) -> None:
    """Called by the cooldown task when the player never reconnected.

    Marks the match abandoned (character wins) and kicks off post-match processing.
    """
    from app.matches import service as match_service
    from app.post_match.processor import start_post_match_background
    from app.sockets.processor_callback import build_processor_callback

    with SessionLocal() as session:
        match = _resolve_match(session, match_id)
        if match is None:
            return
        if match.status != MatchStatus.IN_PROGRESS:
            return
        match_service.abandon_for_disconnect(session, match_id=match_id)
        session.commit()
        session.refresh(match)
        end_payload = MatchEndedPayload(
            match_id=match_id,
            result="abandoned",
            reason="disconnect_timeout",
            player_outcome="resigned",
        )

    await sio.emit(
        S2C_MATCH_ENDED,
        end_payload.model_dump(mode="json"),
        room=match_room(match_id),
        namespace=NAMESPACE,
    )
    start_post_match_background(
        match_id,
        status_callback=build_processor_callback(match_id),
    )


# --- Event: ping -----------------------------------------------------------


@sio.on(C2S_PING, namespace=NAMESPACE)
async def _on_ping(sid, _data=None):
    await sio.emit(
        S2C_PONG,
        PongPayload(ts=datetime.utcnow()).model_dump(mode="json"),
        to=sid,
        namespace=NAMESPACE,
    )


# --- Event: request_state --------------------------------------------------


@sio.on(C2S_REQUEST_STATE, namespace=NAMESPACE)
async def _on_request_state(sid, _data=None):
    sess = await sio.get_session(sid, namespace=NAMESPACE)
    match_id = sess.get("match_id")
    if not match_id:
        await _send_error(sid, "no_match", "Socket has no bound match.")
        return
    with SessionLocal() as session:
        match = _resolve_match(session, match_id)
        if match is None:
            await _send_error(sid, "match_not_found", "Match vanished.")
            return
        payload = _build_match_state(session, match)
    await sio.emit(
        S2C_MATCH_STATE,
        payload.model_dump(mode="json"),
        to=sid,
        namespace=NAMESPACE,
    )


# --- Event: player_chat ---------------------------------------------------


@sio.on(C2S_PLAYER_CHAT, namespace=NAMESPACE)
async def _on_player_chat(sid, data):
    sess = await sio.get_session(sid, namespace=NAMESPACE)
    match_id = sess.get("match_id")
    role = sess.get("role", ROLE_PARTICIPANT)
    username = sess.get("username", "")
    if not match_id:
        await _send_error(sid, "no_match", "Socket has no bound match.")
        return
    if role != ROLE_PARTICIPANT:
        # Spectators speak via `spectator_chat`, not `player_chat`.
        await _send_error(sid, "spectator_cannot_player_chat",
                          "Spectators must use spectator_chat.")
        return

    try:
        event = PlayerChatEvent.model_validate(data or {})
    except Exception as exc:
        await _send_error(sid, "bad_payload", f"Invalid player_chat payload: {exc}")
        return

    ok, retry_after = _rate_limit_ok(sess)
    await sio.save_session(sid, sess, namespace=NAMESPACE)
    if not ok:
        await sio.emit(
            S2C_PLAYER_CHAT_RATE_LIMITED,
            PlayerChatRateLimitedPayload(retry_after_ms=retry_after).model_dump(mode="json"),
            to=sid,
            namespace=NAMESPACE,
        )
        return

    received_at = datetime.utcnow()
    with SessionLocal() as session:
        match = _resolve_match(session, match_id)
        if match is None or match.status != MatchStatus.IN_PROGRESS:
            await _send_error(sid, "match_not_active", "Match is not in progress.")
            return
        _append_pending_chat(match, event.text)
        session.commit()

    await sio.emit(
        S2C_PLAYER_CHAT_ECHOED,
        PlayerChatEchoedPayload(text=event.text, received_at=received_at).model_dump(mode="json"),
        to=sid,
        namespace=NAMESPACE,
    )
    # Broadcast to spectators so they see the dialogue with the character.
    # Participant's own socket also receives this but the client dedupes against
    # the optimistic bubble it already rendered.
    await sio.emit(
        S2C_PLAYER_CHAT_BROADCAST,
        PlayerChatBroadcastPayload(
            username=username, text=event.text, timestamp=received_at,
        ).model_dump(mode="json"),
        room=match_room(match_id),
        skip_sid=sid,
        namespace=NAMESPACE,
    )

    # Patch Pass 2 Item 5: fire a lightweight Soul call if no character turn
    # is in flight, so the character can respond to chat that arrives between
    # turns instead of always waiting for the next move. Rate-limited per
    # match; during an active turn this is a no-op (the existing buffering
    # path handles chat merged into the next Subconscious call).
    from app.matches.streaming import run_chat_triggered_soul

    async def _emit_chat(soul_response) -> None:
        await sio.emit(
            S2C_AGENT_CHAT,
            AgentChatPayload(
                speak=soul_response.speak,
                emotion=soul_response.emotion,
                emotion_intensity=soul_response.emotion_intensity,
                referenced_memory_ids=list(soul_response.referenced_memory_ids or []),
            ).model_dump(mode="json"),
            room=match_room(match_id),
            namespace=NAMESPACE,
        )

    # Fire-and-forget: don't block the chat handler on Soul latency.
    asyncio.create_task(
        run_chat_triggered_soul(match_id=match_id, emit_chat=_emit_chat)
    )


# --- Event: spectator_chat ------------------------------------------------


@sio.on(C2S_SPECTATOR_CHAT, namespace=NAMESPACE)
async def _on_spectator_chat(sid, data):
    sess = await sio.get_session(sid, namespace=NAMESPACE)
    match_id = sess.get("match_id")
    role = sess.get("role", ROLE_PARTICIPANT)
    username = sess.get("username", "")
    if not match_id:
        await _send_error(sid, "no_match", "Socket has no bound match.")
        return
    if role != ROLE_SPECTATOR:
        # Participants have their own channel; spectator_chat would bypass the
        # Subconscious buffer which is the whole point — reject loudly.
        await sio.emit(
            S2C_SPECTATOR_CHAT_REJECTED,
            SpectatorChatRejectedPayload().model_dump(mode="json"),
            to=sid,
            namespace=NAMESPACE,
        )
        return

    try:
        event = SpectatorChatEvent.model_validate(data or {})
    except Exception as exc:
        await _send_error(sid, "bad_payload", f"Invalid spectator_chat payload: {exc}")
        return

    ok, retry_after = _rate_limit_ok(sess)
    await sio.save_session(sid, sess, namespace=NAMESPACE)
    if not ok:
        await sio.emit(
            S2C_PLAYER_CHAT_RATE_LIMITED,
            PlayerChatRateLimitedPayload(retry_after_ms=retry_after).model_dump(mode="json"),
            to=sid,
            namespace=NAMESPACE,
        )
        return

    now = datetime.utcnow()
    await sio.emit(
        S2C_SPECTATOR_CHAT_ECHOED,
        PlayerChatEchoedPayload(text=event.text, received_at=now).model_dump(mode="json"),
        to=sid,
        namespace=NAMESPACE,
    )
    # Broadcast to the whole room — participant + other spectators. Crucially,
    # `spectator_chat` is NOT written into `match.extra_state.pending_player_chat`,
    # so the character never sees it.
    await sio.emit(
        S2C_SPECTATOR_CHAT_BROADCAST,
        SpectatorChatBroadcastPayload(
            username=username, text=event.text, timestamp=now,
        ).model_dump(mode="json"),
        room=match_room(match_id),
        skip_sid=sid,
        namespace=NAMESPACE,
    )


# --- Event: make_move ------------------------------------------------------


@sio.on(C2S_MAKE_MOVE, namespace=NAMESPACE)
async def _on_make_move(sid, data):
    sess = await sio.get_session(sid, namespace=NAMESPACE)
    match_id = sess.get("match_id")
    role = sess.get("role", ROLE_PARTICIPANT)
    if not match_id:
        await _send_error(sid, "no_match", "Socket has no bound match.")
        return
    if role != ROLE_PARTICIPANT:
        await _send_error(sid, "spectator_cannot_move", "Spectators cannot make moves.")
        return

    try:
        event = MakeMoveEvent.model_validate(data or {})
    except Exception as exc:
        await _send_error(sid, "bad_payload", f"Invalid make_move payload: {exc}")
        return

    from app.matches import service as match_service
    from app.matches.streaming import TurnEmitters, apply_player_move_streamed

    emitters = _build_turn_emitters(match_id)

    try:
        await apply_player_move_streamed(
            match_id=match_id,
            uci=event.uci,
            player_chat=event.chat,
            emitters=emitters,
        )
    except match_service.IllegalMove as exc:
        await _send_error(sid, "illegal_move", str(exc))
        return
    except match_service.NotYourTurn as exc:
        await _send_error(sid, "not_your_turn", str(exc))
        return
    except match_service.GameAlreadyOver as exc:
        await _send_error(sid, "game_over", str(exc))
        return
    except match_service.MatchNotFound:
        await _send_error(sid, "match_not_found", "Match not found.")
        return
    except Exception as exc:
        logger.exception("make_move crashed for match=%s", match_id)
        await _send_error(sid, "internal_error", f"{type(exc).__name__}: {exc}")
        return


# --- Event: resign ---------------------------------------------------------


@sio.on(C2S_RESIGN, namespace=NAMESPACE)
async def _on_resign(sid, _data=None):
    sess = await sio.get_session(sid, namespace=NAMESPACE)
    match_id = sess.get("match_id")
    role = sess.get("role", ROLE_PARTICIPANT)
    if not match_id:
        await _send_error(sid, "no_match", "Socket has no bound match.")
        return
    if role != ROLE_PARTICIPANT:
        await _send_error(sid, "spectator_cannot_resign", "Spectators cannot resign.")
        return

    from app.matches import service as match_service
    from app.post_match.processor import start_post_match_background
    from app.sockets.processor_callback import build_processor_callback

    with SessionLocal() as session:
        try:
            match = match_service.resign(session, match_id=match_id)
            session.commit()
            session.refresh(match)
            result_value = match.result.value  # white_win or black_win
        except match_service.GameAlreadyOver as exc:
            await _send_error(sid, "game_over", str(exc))
            return
        except match_service.MatchNotFound:
            await _send_error(sid, "match_not_found", "Match not found.")
            return

    await sio.emit(
        S2C_MATCH_ENDED,
        MatchEndedPayload(
            match_id=match_id,
            result=result_value,  # type: ignore[arg-type]
            reason="resign",
            player_outcome="resigned",
        ).model_dump(mode="json"),
        room=match_room(match_id),
        namespace=NAMESPACE,
    )
    start_post_match_background(
        match_id,
        status_callback=build_processor_callback(match_id),
    )


# --- Emitter factory: passed into the match-service streamed turn ----------


def _build_turn_emitters(match_id: str):
    """Create a `TurnEmitters` bundle that fires Socket.IO events at the right moments."""
    from app.matches.streaming import TurnEmitters

    async def _on_player_move(move: Move) -> None:
        await sio.emit(
            S2C_PLAYER_MOVE_APPLIED,
            PlayerMoveAppliedPayload(
                move_number=move.move_number,
                uci=move.uci,
                san=move.san,
                fen_after=move.fen_after,
                player_chat_before=move.player_chat_before,
            ).model_dump(mode="json"),
            room=match_room(match_id),
            namespace=NAMESPACE,
        )

    async def _on_thinking(eta_seconds: float) -> None:
        await sio.emit(
            S2C_AGENT_THINKING,
            AgentThinkingPayload(eta_seconds=eta_seconds).model_dump(mode="json"),
            room=match_room(match_id),
            namespace=NAMESPACE,
        )

    async def _on_memory_surfaced(surfaced) -> None:
        items = [
            MemorySurfacedItem(
                memory_id=m.memory_id,
                retrieval_reason=m.retrieval_reason,
                narrative_snippet=m.narrative_text,
                from_cache=m.from_cache,
            )
            for m in surfaced
        ]
        await sio.emit(
            S2C_MEMORY_SURFACED,
            MemorySurfacedPayload(items=items).model_dump(mode="json"),
            room=match_room(match_id),
            namespace=NAMESPACE,
        )

    async def _on_agent_move(move: Move) -> None:
        await sio.emit(
            S2C_AGENT_MOVE,
            AgentMovePayload(
                move_number=move.move_number,
                uci=move.uci,
                san=move.san,
                fen_after=move.fen_after,
                time_taken_ms=move.time_taken_ms,
                engine_name=move.engine_name,
                eval_cp=move.eval_cp,
            ).model_dump(mode="json"),
            room=match_room(match_id),
            namespace=NAMESPACE,
        )

    async def _on_agent_chat(soul_response) -> None:
        if not soul_response.speak:
            return
        await sio.emit(
            S2C_AGENT_CHAT,
            AgentChatPayload(
                speak=soul_response.speak,
                emotion=soul_response.emotion,
                emotion_intensity=soul_response.emotion_intensity,
                referenced_memory_ids=list(soul_response.referenced_memory_ids or []),
            ).model_dump(mode="json"),
            room=match_room(match_id),
            namespace=NAMESPACE,
        )

    async def _on_mood_update(mood) -> None:
        await sio.emit(
            S2C_MOOD_UPDATE,
            MoodUpdatePayload(mood=mood.to_dict()).model_dump(mode="json"),
            room=match_room(match_id),
            namespace=NAMESPACE,
        )

    async def _on_match_ended(reason: str, result_value: str, player_outcome: str | None) -> None:
        await sio.emit(
            S2C_MATCH_ENDED,
            MatchEndedPayload(
                match_id=match_id,
                result=result_value,  # type: ignore[arg-type]
                reason=reason,  # type: ignore[arg-type]
                player_outcome=player_outcome,  # type: ignore[arg-type]
            ).model_dump(mode="json"),
            room=match_room(match_id),
            namespace=NAMESPACE,
        )

    async def _on_post_match_kickoff() -> None:
        from app.post_match.processor import start_post_match_background
        from app.sockets.processor_callback import build_processor_callback

        start_post_match_background(
            match_id,
            status_callback=build_processor_callback(match_id),
        )

    return TurnEmitters(
        on_player_move=_on_player_move,
        on_thinking=_on_thinking,
        on_memory_surfaced=_on_memory_surfaced,
        on_agent_move=_on_agent_move,
        on_agent_chat=_on_agent_chat,
        on_mood_update=_on_mood_update,
        on_match_ended=_on_match_ended,
        on_post_match_kickoff=_on_post_match_kickoff,
    )
