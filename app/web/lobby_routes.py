"""Phase 4.2h — HTTP routes for lobbies, matchmaking, and PvP play.

Lives in a separate module from `routes.py` because the existing file
is already large. All routes are under `/lobby/*`, `/lobbies`,
`/matchmaking`, or `/pvp/*`.
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from app.auth import get_optional_player, require_player
from app.db import get_session
from app.lobbies import matchmaking as mm
from app.lobbies import pvp_service as pvp
from app.lobbies.service import (
    KNOWN_MUSIC_TRACKS,
    ControlsPatch,
    CreateLobbyIn,
    LobbyClosed,
    LobbyError,
    LobbyForbidden,
    LobbyFull,
    LobbyInvalidControl,
    LobbyNotFound,
    active_members,
    close_lobby,
    create_lobby,
    current_active_lobby_for,
    get_lobby,
    get_lobby_by_code,
    join_lobby,
    join_lobby_by_code,
    leave_lobby,
    list_public_open_lobbies,
    update_controls,
)
from app.models.lobby import Lobby, LobbyStatus, PvpMatchStatus
from app.models.match import Player

logger = logging.getLogger(__name__)

_TEMPLATE_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(_TEMPLATE_DIR))

router = APIRouter(tags=["lobby"])


def _lobby_view_context(session: Session, lobby, *, player: Player) -> dict:
    members = active_members(session, lobby.id)
    # Hydrate player objects for display.
    rows = []
    for m in members:
        p = session.get(Player, m.player_id)
        if p is None:
            continue
        rows.append({
            "player_id": p.id,
            "username": p.username,
            "display_name": p.display_name,
            "elo": p.elo,
            "role": m.role.value,
            "is_you": p.id == player.id,
        })
    return {
        "lobby": lobby,
        "members": rows,
        "is_host": lobby.host_id == player.id,
        "known_music_tracks": KNOWN_MUSIC_TRACKS,
    }


# --- browse + create ------------------------------------------------------


@router.get("/lobbies", response_class=HTMLResponse)
def lobbies_index(
    request: Request,
    player: Player = Depends(require_player),
    session: Session = Depends(get_session),
) -> HTMLResponse:
    lobbies = list_public_open_lobbies(session)
    # Member counts per lobby for display.
    entries = []
    for lob in lobbies:
        mems = active_members(session, lob.id)
        host = session.get(Player, lob.host_id)
        entries.append({
            "lobby": lob,
            "member_count": len(mems),
            "host_username": host.username if host else "—",
        })
    current = current_active_lobby_for(session, player.id)
    return templates.TemplateResponse(
        request,
        "lobbies.html",
        {
            "player": player,
            "entries": entries,
            "current_lobby": current,
        },
    )


@router.get("/lobby/new", response_class=HTMLResponse)
def lobby_new_form(
    request: Request,
    player: Player = Depends(require_player),
) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "lobby_new.html",
        {"player": player, "known_music_tracks": KNOWN_MUSIC_TRACKS},
    )


@router.post("/lobby/new")
def lobby_new_submit(
    is_private: str = Form(""),
    allow_spectators: str = Form("on"),
    music_track: str = Form(""),
    lights_hue: str = Form("#C9A66B"),
    lights_brightness: float = Form(0.7),
    music_volume: float = Form(0.5),
    player: Player = Depends(require_player),
    session: Session = Depends(get_session),
):
    try:
        lob = create_lobby(
            session,
            host=player,
            spec=CreateLobbyIn(
                is_private=bool(is_private),
                allow_spectators=bool(allow_spectators),
                music_track=(music_track or None),
                music_volume=music_volume,
                lights_brightness=lights_brightness,
                lights_hue=lights_hue,
            ),
        )
    except LobbyInvalidControl as exc:
        return RedirectResponse(
            url=f"/lobby/new?error={exc.code}", status_code=303
        )
    return RedirectResponse(url=f"/lobby/{lob.id}", status_code=303)


@router.get("/lobby/join", response_class=HTMLResponse)
def lobby_join_form(
    request: Request,
    code: str = "",
    error: str | None = None,
    player: Player = Depends(require_player),
) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "lobby_join.html",
        {"player": player, "code": code, "error": error},
    )


@router.post("/lobby/join")
def lobby_join_submit(
    code: str = Form(...),
    player: Player = Depends(require_player),
    session: Session = Depends(get_session),
):
    try:
        lob, _ = join_lobby_by_code(session, code, player)
    except LobbyNotFound:
        return RedirectResponse(url=f"/lobby/join?code={code}&error=not_found", status_code=303)
    except LobbyFull:
        return RedirectResponse(url=f"/lobby/join?code={code}&error=full", status_code=303)
    except LobbyClosed:
        return RedirectResponse(url=f"/lobby/join?code={code}&error=closed", status_code=303)
    return RedirectResponse(url=f"/lobby/{lob.id}", status_code=303)


# --- the room -------------------------------------------------------------


@router.get("/lobby/{lobby_id}", response_class=HTMLResponse, response_model=None)
def lobby_room(
    request: Request,
    lobby_id: str,
    player: Player = Depends(require_player),
    session: Session = Depends(get_session),
) -> HTMLResponse | RedirectResponse:
    try:
        lob = get_lobby(session, lobby_id)
    except LobbyNotFound:
        raise HTTPException(status_code=404, detail="Lobby not found")

    # Players not in the lobby — bounce them to the join flow. Private
    # lobbies require the code path; public lobbies we'll just seat them.
    members = active_members(session, lob.id)
    seated = any(m.player_id == player.id for m in members)
    if not seated:
        if lob.status == LobbyStatus.CLOSED:
            raise HTTPException(status_code=404, detail="Lobby has been closed")
        if lob.is_private:
            return RedirectResponse(url="/lobby/join?error=private", status_code=303)
        # Auto-seat on a public lobby visit. Failure (full) redirects.
        try:
            join_lobby(session, lob, player)
        except LobbyFull:
            return RedirectResponse(url="/lobbies?error=full", status_code=303)

    # Phase 4.2.5 — zero-button auto-start: if both seats are filled and
    # no match is running, spawn one now with random colors. The board
    # then appears to both players with no "Start match" button needed.
    # Controls (door/lock/radio/lamp) remain available; the lobby page
    # is only a pass-through at that point.
    if (
        lob.status == LobbyStatus.OPEN
        and lob.current_match_id is None
        and len(active_members(session, lob.id)) == 2
    ):
        try:
            host = session.get(Player, lob.host_id)
            if host is not None:
                match = pvp.start_match(session, lob, by=host, white_choice="random")
                return RedirectResponse(url=f"/pvp/{match.id}", status_code=303)
        except Exception:
            logger.exception("auto-start failed for lobby %s", lob.id)

    # If an active match is in progress, redirect participants to the board.
    if lob.status == LobbyStatus.IN_MATCH and lob.current_match_id:
        return RedirectResponse(url=f"/pvp/{lob.current_match_id}", status_code=303)

    ctx = _lobby_view_context(session, lob, player=player)
    ctx["player"] = player
    return templates.TemplateResponse(request, "lobby.html", ctx)


@router.post("/lobby/{lobby_id}/leave")
def lobby_leave(
    lobby_id: str,
    player: Player = Depends(require_player),
    session: Session = Depends(get_session),
):
    try:
        lob = get_lobby(session, lobby_id)
    except LobbyNotFound:
        return RedirectResponse(url="/lobbies", status_code=303)
    leave_lobby(session, lob, player)
    return RedirectResponse(url="/lobbies", status_code=303)


@router.post("/lobby/{lobby_id}/close")
def lobby_close(
    lobby_id: str,
    player: Player = Depends(require_player),
    session: Session = Depends(get_session),
):
    lob = get_lobby(session, lobby_id)
    try:
        close_lobby(session, lob, by=player)
    except LobbyForbidden:
        return RedirectResponse(url=f"/lobby/{lobby_id}?error=not_host", status_code=303)
    return RedirectResponse(url="/lobbies", status_code=303)


@router.get("/lobby/{lobby_id}/state")
def lobby_state(
    lobby_id: str,
    player: Player = Depends(require_player),
    session: Session = Depends(get_session),
):
    """JSON snapshot for the lobby page poller."""
    try:
        lob = get_lobby(session, lobby_id)
    except LobbyNotFound:
        return JSONResponse({"error": "not_found"}, status_code=404)
    members = active_members(session, lob.id)
    return JSONResponse({
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
        "member_count": len(members),
    })


@router.post("/lobby/{lobby_id}/controls")
def lobby_controls(
    lobby_id: str,
    is_private: str | None = Form(None),
    allow_spectators: str | None = Form(None),
    music_track: str | None = Form(None),
    music_volume: float | None = Form(None),
    lights_brightness: float | None = Form(None),
    lights_hue: str | None = Form(None),
    time_control: str | None = Form(None),
    player: Player = Depends(require_player),
    session: Session = Depends(get_session),
):
    lob = get_lobby(session, lobby_id)

    def _bool(v: str | None) -> bool | None:
        if v is None:
            return None
        return v.lower() in ("1", "true", "on", "yes")

    patch = ControlsPatch(
        is_private=_bool(is_private),
        allow_spectators=_bool(allow_spectators),
        music_track=(music_track if music_track is not None else None),
        music_volume=music_volume,
        lights_brightness=lights_brightness,
        lights_hue=lights_hue,
        time_control=time_control,
    )
    try:
        update_controls(session, lob, by=player, patch=patch)
    except LobbyForbidden:
        return JSONResponse({"error": "not_host"}, status_code=403)
    except LobbyInvalidControl as exc:
        return JSONResponse({"error": exc.code, "message": str(exc)}, status_code=400)
    except LobbyClosed:
        return JSONResponse({"error": "closed"}, status_code=409)
    # Broadcast via socket layer (Phase 4.2g) — importing here to avoid
    # a module-level cycle; absence is tolerated for test isolation.
    try:
        from app.sockets.lobby_server import broadcast_controls_update  # type: ignore

        broadcast_controls_update(lob)
    except Exception:
        pass
    return JSONResponse({
        "is_private": lob.is_private,
        "allow_spectators": lob.allow_spectators,
        "music_track": lob.music_track,
        "music_volume": lob.music_volume,
        "lights_brightness": lob.lights_brightness,
        "lights_hue": lob.lights_hue,
        "time_control": lob.time_control,
    })


# --- start + play ---------------------------------------------------------


@router.post("/lobby/{lobby_id}/start")
def lobby_start_match(
    lobby_id: str,
    white_choice: str = Form("random"),
    player: Player = Depends(require_player),
    session: Session = Depends(get_session),
):
    lob = get_lobby(session, lobby_id)
    try:
        match = pvp.start_match(session, lob, by=player, white_choice=white_choice)
    except LobbyForbidden:
        return RedirectResponse(url=f"/lobby/{lobby_id}?error=not_host", status_code=303)
    except LobbyError as exc:
        return RedirectResponse(url=f"/lobby/{lobby_id}?error={exc.code}", status_code=303)
    return RedirectResponse(url=f"/pvp/{match.id}", status_code=303)


@router.get("/pvp/{match_id}", response_class=HTMLResponse, response_model=None)
def pvp_play_page(
    request: Request,
    match_id: str,
    player: Player = Depends(require_player),
    session: Session = Depends(get_session),
) -> HTMLResponse | RedirectResponse:
    try:
        match = pvp.get_match(session, match_id)
    except pvp.PvpMatchNotFound:
        raise HTTPException(status_code=404, detail="Match not found")

    lob = session.get(Lobby, match.lobby_id)
    if player.id not in (match.white_player_id, match.black_player_id):
        # Spectator path — gated by lobby.allow_spectators. Socket-level
        # spectator entry (Phase 4.2g) adds richer features; HTTP just
        # gives the static snapshot + read-only view.
        if lob is None or not lob.allow_spectators:
            raise HTTPException(status_code=403, detail="Spectators are not allowed in this lobby")

    white = session.get(Player, match.white_player_id)
    black = session.get(Player, match.black_player_id)
    return templates.TemplateResponse(
        request,
        "pvp.html",
        {
            "player": player,
            "match": match,
            "white": white,
            "black": black,
            "lobby": lob,
            "is_white": player.id == match.white_player_id,
            "is_black": player.id == match.black_player_id,
            "is_spectator": player.id not in (match.white_player_id, match.black_player_id),
        },
    )


@router.post("/pvp/{match_id}/move")
def pvp_play_move(
    match_id: str,
    uci: str = Form(...),
    player: Player = Depends(require_player),
    session: Session = Depends(get_session),
):
    match = pvp.get_match(session, match_id)
    try:
        applied = pvp.apply_move(session, match, by=player, uci=uci)
    except pvp.PvpError as exc:
        return JSONResponse({"error": exc.code, "message": str(exc)}, status_code=400)
    return JSONResponse({
        "move": applied.move,
        "fen_after": applied.fen_after,
        "game_over": applied.game_over,
        "result": applied.result.value if applied.result else None,
        "reason": applied.reason,
    })


@router.get("/pvp/{match_id}/state")
def pvp_state(
    match_id: str,
    player: Player = Depends(require_player),
    session: Session = Depends(get_session),
):
    """Polled by the PvP board for opponent moves + end-of-game. The
    `/lobby` socket namespace (4.2g) supersedes this once wired."""
    try:
        match = pvp.get_match(session, match_id)
    except pvp.PvpMatchNotFound:
        return JSONResponse({"error": "not_found"}, status_code=404)
    # Gate spectators on lobby.allow_spectators (same rule as the page GET).
    if player.id not in (match.white_player_id, match.black_player_id):
        lob = session.get(Lobby, match.lobby_id)
        if lob is None or not lob.allow_spectators:
            return JSONResponse({"error": "forbidden"}, status_code=403)
    reason = (match.extra_state or {}).get("reason")
    return JSONResponse({
        "status": match.status.value,
        "result": match.result.value if match.result else None,
        "reason": reason,
        "current_fen": match.current_fen,
        "moves": match.moves,
        "move_count": match.move_count,
    })


@router.post("/pvp/{match_id}/resign")
def pvp_play_resign(
    match_id: str,
    player: Player = Depends(require_player),
    session: Session = Depends(get_session),
):
    match = pvp.get_match(session, match_id)
    try:
        result = pvp.resign(session, match, by=player)
    except pvp.PvpError as exc:
        return JSONResponse({"error": exc.code}, status_code=400)
    return JSONResponse({"result": result.value})


# --- matchmaking ----------------------------------------------------------


@router.get("/matchmaking", response_class=HTMLResponse)
def matchmaking_page(
    request: Request,
    player: Player = Depends(require_player),
    session: Session = Depends(get_session),
) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "matchmaking.html",
        {"player": player},
    )


@router.post("/matchmaking/enqueue")
def matchmaking_enqueue(
    player: Player = Depends(require_player),
    session: Session = Depends(get_session),
):
    try:
        mm.enqueue(session, player)
    except mm.AlreadyQueued:
        pass
    # Try a match step synchronously so a user already in the queue can
    # pair immediately without waiting for the periodic worker.
    mm.try_match_step(session)
    return JSONResponse({"queued": True})


@router.post("/matchmaking/cancel")
def matchmaking_cancel(
    player: Player = Depends(require_player),
    session: Session = Depends(get_session),
):
    mm.cancel(session, player)
    return JSONResponse({"canceled": True})


@router.get("/matchmaking/poll")
def matchmaking_poll(
    player: Player = Depends(require_player),
    session: Session = Depends(get_session),
):
    # Opportunistic match attempt every poll — cheap.
    mm.try_match_step(session)
    r = mm.poll(session, player)
    return JSONResponse({
        "queued": r.queued,
        "matched_lobby_id": r.matched_lobby_id,
        "band_step": r.band_step,
        "waited_seconds": r.waited_seconds,
    })
