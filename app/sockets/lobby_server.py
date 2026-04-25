"""Socket.IO /lobby namespace — real-time lobby state (Phase 4.4).

Pushes door/lock/music/lights/seat/match-start changes to everyone in
the lobby room the moment the host commits them. Replaces the 1.5s HTTP
polling loop in `lobby.html`.

Room name: ``lobby:<lobby_id>``. One per lobby, joined on WebSocket
connect. HTTP routes (controls, seat claim, start_match) call the
`broadcast_*` helpers below after committing their DB change.

The underlying `AsyncServer` (`app.sockets.server.sio`) is shared with
the ``/play`` namespace — this module only registers handlers on a
separate namespace path.
"""

from __future__ import annotations

import asyncio
import logging
from http.cookies import SimpleCookie
from typing import Any

from app.auth import PLAYER_COOKIE
from app.db import SessionLocal
from app.lobbies.service import get_lobby, LobbyError
from app.models.lobby import Lobby, LobbyMembership, LobbyRole
from app.models.match import Player
from app.sockets.bridge import get_main_loop
from app.sockets.server import sio

logger = logging.getLogger(__name__)

NAMESPACE = "/lobby"


# --- events (constants, so handlers + callers can't drift) ----------------

S2C_LOBBY_STATE = "lobby_state"
S2C_CONTROLS_CHANGED = "controls_changed"
S2C_SEAT_CHANGED = "seat_changed"
S2C_MATCH_STARTED = "match_started"
S2C_LOBBY_CLOSED = "lobby_closed"
S2C_MEMBER_COUNT = "member_count"
S2C_ERROR = "error"


def lobby_room(lobby_id: str) -> str:
    return f"lobby:{lobby_id}"


# --- auth helpers (duplicated from server.py rather than cross-imported to
#     avoid pulling in the /play namespace wiring) -------------------------


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
    for pair in qs.split("&"):
        if "=" not in pair:
            continue
        k, v = pair.split("=", 1)
        if k == key:
            return v
    return None


# --- state serializer ------------------------------------------------------


def _serialize_lobby(lob: Lobby) -> dict[str, Any]:
    return {
        "id": lob.id,
        "code": lob.code,
        "status": lob.status.value,
        "is_private": lob.is_private,
        "allow_spectators": lob.allow_spectators,
        "music_track": lob.music_track,
        "music_volume": lob.music_volume,
        "lights_brightness": lob.lights_brightness,
        "lights_hue": lob.lights_hue,
        "time_control": lob.time_control,
        "host_id": lob.host_id,
        "current_match_id": lob.current_match_id,
    }


def _serialize_members(session, lobby_id: str) -> list[dict[str, Any]]:
    from sqlalchemy import select

    rows = session.execute(
        select(LobbyMembership, Player)
        .join(Player, Player.id == LobbyMembership.player_id)
        .where(LobbyMembership.lobby_id == lobby_id)
        .where(LobbyMembership.left_at.is_(None))
    ).all()
    out = []
    for mem, pl in rows:
        out.append({
            "player_id": pl.id,
            "username": pl.username,
            "role": mem.role.value,
        })
    return out


# --- connect / disconnect -------------------------------------------------


@sio.on("connect", namespace=NAMESPACE)
async def _on_connect(sid, environ, auth):
    player_id = _parse_cookie_header(environ)
    if not player_id:
        logger.info("[lobby-sock] reject: no cookie (sid=%s)", sid)
        return False

    lobby_id = _query_param(auth, environ, "lobby_id")
    if not lobby_id:
        logger.info("[lobby-sock] reject: no lobby_id (sid=%s)", sid)
        return False

    with SessionLocal() as session:
        player = session.get(Player, player_id)
        if player is None:
            return False
        try:
            lob = get_lobby(session, lobby_id)
        except LobbyError:
            return False
        # Members always allowed. Non-members allowed only when
        # allow_spectators is True (matches the HTTP gate).
        is_member = session.execute(
            # any active membership?
            # pragmatic: just check the list
            __import__("sqlalchemy").select(LobbyMembership).where(
                LobbyMembership.lobby_id == lob.id,
                LobbyMembership.player_id == player.id,
                LobbyMembership.left_at.is_(None),
            )
        ).first() is not None
        if not is_member and not lob.allow_spectators:
            logger.info(
                "[lobby-sock] reject: spectators locked (sid=%s lobby=%s)", sid, lob.id
            )
            return False

        state = _serialize_lobby(lob)
        members = _serialize_members(session, lob.id)

    await sio.save_session(
        sid,
        {"player_id": player_id, "lobby_id": lobby_id, "is_member": is_member},
        namespace=NAMESPACE,
    )
    await sio.enter_room(sid, lobby_room(lobby_id), namespace=NAMESPACE)
    await sio.emit(
        S2C_LOBBY_STATE,
        {"lobby": state, "members": members},
        to=sid,
        namespace=NAMESPACE,
    )
    # Update member count for everyone.
    await sio.emit(
        S2C_MEMBER_COUNT,
        {"count": len(members), "lobby_id": lobby_id},
        room=lobby_room(lobby_id),
        namespace=NAMESPACE,
    )
    logger.info("[lobby-sock] connected sid=%s lobby=%s player=%s", sid, lobby_id, player_id)
    return True


@sio.on("disconnect", namespace=NAMESPACE)
async def _on_disconnect(sid):
    try:
        sess = await sio.get_session(sid, namespace=NAMESPACE)
    except KeyError:
        return
    lobby_id = sess.get("lobby_id") if sess else None
    if lobby_id:
        await sio.leave_room(sid, lobby_room(lobby_id), namespace=NAMESPACE)
        logger.info("[lobby-sock] disconnected sid=%s lobby=%s", sid, lobby_id)


# --- sync broadcast helpers (called from HTTP routes) ---------------------


def _dispatch(coro) -> None:
    """Schedule a coroutine on the main loop from sync context.

    Mirrors `bridge._run_threadsafe` but doesn't require importing bridge's
    private helper. Silent no-op if the loop isn't up (tests, shutdown).
    """
    loop = get_main_loop()
    if loop is None:
        try:
            coro.close()
        except Exception:
            pass
        return
    try:
        asyncio.run_coroutine_threadsafe(coro, loop)
    except RuntimeError:
        try:
            coro.close()
        except Exception:
            pass


async def _emit_controls(lobby_id: str, payload: dict[str, Any]) -> None:
    await sio.emit(
        S2C_CONTROLS_CHANGED, payload,
        room=lobby_room(lobby_id), namespace=NAMESPACE,
    )


async def _emit_seat(lobby_id: str, payload: dict[str, Any]) -> None:
    await sio.emit(
        S2C_SEAT_CHANGED, payload,
        room=lobby_room(lobby_id), namespace=NAMESPACE,
    )


async def _emit_match_started(lobby_id: str, payload: dict[str, Any]) -> None:
    await sio.emit(
        S2C_MATCH_STARTED, payload,
        room=lobby_room(lobby_id), namespace=NAMESPACE,
    )


async def _emit_closed(lobby_id: str) -> None:
    await sio.emit(
        S2C_LOBBY_CLOSED, {"lobby_id": lobby_id},
        room=lobby_room(lobby_id), namespace=NAMESPACE,
    )


def broadcast_controls_update(lob: Lobby) -> None:
    """Called by the controls-patch HTTP route after commit."""
    _dispatch(_emit_controls(lob.id, _serialize_lobby(lob)))


def broadcast_seat_changed(lob: Lobby, *, members: list[dict[str, Any]]) -> None:
    """Called on seat claim / leave / role change."""
    _dispatch(_emit_seat(lob.id, {"lobby_id": lob.id, "members": members}))


def broadcast_match_started(lob: Lobby, *, match_id: str) -> None:
    """Called by `/lobby/{id}/start` after `start_match` returns."""
    _dispatch(_emit_match_started(lob.id, {"lobby_id": lob.id, "match_id": match_id}))


def broadcast_lobby_closed(lob: Lobby) -> None:
    _dispatch(_emit_closed(lob.id))
