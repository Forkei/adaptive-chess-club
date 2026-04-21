"""Phase 4.2b — lobby service layer.

Create / join / leave / control-toggle + lookups. Called by HTTP
routes, the Socket.IO `/lobby` namespace, and the matchmaking worker.

All operations take a `Session` and persist by calling `session.commit()`
themselves — matches how `app/matches/service.py` works. Caller doesn't
need to manage transactions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.lobbies.codes import generate_unique_code
from app.models.lobby import (
    Lobby,
    LobbyMembership,
    LobbyRole,
    LobbyStatus,
)
from app.models.match import Player

logger = logging.getLogger(__name__)

LOBBY_MAX_MEMBERS = 2  # two-seat room — host + one opponent.

# Music track options the radio widget knows about. Add to this list as
# real audio files are dropped into /static/audio/. Unknown values are
# accepted server-side (we don't want to churn this list for every new
# track); client validates against its dropdown.
KNOWN_MUSIC_TRACKS = (
    "none",
    "felt_dim",
    "cafe_murmur",
    "rain_window",
    "synth_drift",
    "mid_century_jazz",
)


# --- errors ---------------------------------------------------------------


class LobbyError(Exception):
    code: str = "lobby_error"

    def __init__(self, message: str = "") -> None:
        super().__init__(message or self.code)


class LobbyNotFound(LobbyError):
    code = "lobby_not_found"


class LobbyFull(LobbyError):
    code = "lobby_full"


class LobbyClosed(LobbyError):
    code = "lobby_closed"


class LobbyForbidden(LobbyError):
    code = "lobby_forbidden"


class LobbyInvalidControl(LobbyError):
    code = "lobby_invalid_control"


# --- helpers --------------------------------------------------------------


def _code_exists(session: Session, code: str) -> bool:
    return session.execute(
        select(Lobby.id).where(Lobby.code == code)
    ).first() is not None


def active_members(session: Session, lobby_id: str) -> list[LobbyMembership]:
    return list(
        session.execute(
            select(LobbyMembership)
            .where(LobbyMembership.lobby_id == lobby_id)
            .where(LobbyMembership.left_at.is_(None))
            .order_by(LobbyMembership.joined_at.asc())
        ).scalars()
    )


def current_active_lobby_for(session: Session, player_id: str) -> Lobby | None:
    """Which non-closed lobby a player is currently seated in (if any).

    Players can only be seated in one lobby at a time — enforced here,
    not via a DB constraint (easier to evolve).
    """
    row = session.execute(
        select(Lobby)
        .join(LobbyMembership, LobbyMembership.lobby_id == Lobby.id)
        .where(LobbyMembership.player_id == player_id)
        .where(LobbyMembership.left_at.is_(None))
        .where(Lobby.status != LobbyStatus.CLOSED)
    ).scalar_one_or_none()
    return row


def get_lobby(session: Session, lobby_id: str) -> Lobby:
    lob = session.get(Lobby, lobby_id)
    if lob is None:
        raise LobbyNotFound()
    return lob


def get_lobby_by_code(session: Session, code: str) -> Lobby:
    code = (code or "").strip().upper()
    if not code:
        raise LobbyNotFound()
    lob = session.execute(
        select(Lobby).where(Lobby.code == code)
    ).scalar_one_or_none()
    if lob is None:
        raise LobbyNotFound()
    return lob


# --- create / join / leave ------------------------------------------------


@dataclass(frozen=True)
class CreateLobbyIn:
    is_private: bool = False
    allow_spectators: bool = True
    music_track: str | None = None
    music_volume: float = 0.5
    lights_brightness: float = 0.7
    lights_hue: str = "#C9A66B"
    via_matchmaking: bool = False


def create_lobby(session: Session, host: Player, spec: CreateLobbyIn) -> Lobby:
    """Create a new lobby with `host` seated and `status=OPEN`.

    If `host` is already in another active lobby, they're removed from
    it first (one-lobby-at-a-time rule).
    """
    _ensure_player_free(session, host.id)

    code = generate_unique_code(lambda c: _code_exists(session, c))
    lob = Lobby(
        code=code,
        host_id=host.id,
        is_private=bool(spec.is_private),
        allow_spectators=bool(spec.allow_spectators),
        music_track=spec.music_track,
        music_volume=_clamp(spec.music_volume, 0.0, 1.0),
        lights_brightness=_clamp(spec.lights_brightness, 0.0, 1.0),
        lights_hue=_validate_hex(spec.lights_hue),
        status=LobbyStatus.OPEN,
        via_matchmaking=bool(spec.via_matchmaking),
    )
    session.add(lob)
    session.flush()  # populate lob.id before creating membership

    session.add(
        LobbyMembership(
            lobby_id=lob.id,
            player_id=host.id,
            role=LobbyRole.HOST,
        )
    )
    session.commit()
    session.refresh(lob)
    return lob


def join_lobby(session: Session, lobby: Lobby, player: Player) -> LobbyMembership:
    """Seat `player` as a guest in `lobby`. Idempotent for re-joins after a
    network blip (if the player has an existing active membership, return
    that instead of creating a duplicate).
    """
    if lobby.status == LobbyStatus.CLOSED:
        raise LobbyClosed()

    existing_here = next(
        (m for m in active_members(session, lobby.id) if m.player_id == player.id),
        None,
    )
    if existing_here is not None:
        return existing_here

    # Remove from any OTHER lobby first.
    _ensure_player_free(session, player.id, but_keep_lobby_id=lobby.id)

    members = active_members(session, lobby.id)
    if len(members) >= LOBBY_MAX_MEMBERS:
        raise LobbyFull()

    m = LobbyMembership(
        lobby_id=lobby.id,
        player_id=player.id,
        role=LobbyRole.GUEST,
    )
    session.add(m)
    session.commit()
    session.refresh(m)
    return m


def join_lobby_by_code(session: Session, code: str, player: Player) -> tuple[Lobby, LobbyMembership]:
    lob = get_lobby_by_code(session, code)
    mem = join_lobby(session, lob, player)
    return lob, mem


def leave_lobby(session: Session, lobby: Lobby, player: Player) -> None:
    """Remove `player` from `lobby`. If the host leaves and another
    member remains, host role transfers to the longest-seated remaining
    member. If the last person leaves, the lobby is closed.
    """
    mems = active_members(session, lobby.id)
    mine = next((m for m in mems if m.player_id == player.id), None)
    if mine is None:
        return  # not a member; no-op

    mine.left_at = datetime.utcnow()
    remaining = [m for m in mems if m.id != mine.id]

    if not remaining:
        lobby.status = LobbyStatus.CLOSED
        lobby.closed_at = datetime.utcnow()
    elif mine.role == LobbyRole.HOST:
        # Transfer host to the next-earliest member.
        new_host = sorted(remaining, key=lambda m: m.joined_at)[0]
        new_host.role = LobbyRole.HOST
        lobby.host_id = new_host.player_id

    session.commit()


def close_lobby(session: Session, lobby: Lobby, *, by: Player) -> None:
    """Host-only: dissolve the lobby. Kicks anyone still inside."""
    if lobby.host_id != by.id:
        raise LobbyForbidden("only the host may close the lobby")
    for m in active_members(session, lobby.id):
        m.left_at = datetime.utcnow()
    lobby.status = LobbyStatus.CLOSED
    lobby.closed_at = datetime.utcnow()
    session.commit()


# --- controls -------------------------------------------------------------


@dataclass(frozen=True)
class ControlsPatch:
    """Partial update to a lobby's ambient controls. Any field left None
    is not touched. Validation is enforced on whichever fields are present.
    """

    is_private: bool | None = None
    allow_spectators: bool | None = None
    music_track: str | None = None
    music_volume: float | None = None
    lights_brightness: float | None = None
    lights_hue: str | None = None
    time_control: str | None = None


TIME_CONTROL_PRESETS: tuple[str, ...] = ("untimed", "5+0", "10+0", "15+10")


def update_controls(
    session: Session, lobby: Lobby, *, by: Player, patch: ControlsPatch
) -> Lobby:
    """Host-only. Apply a ControlsPatch and broadcast via the caller
    (socket handler) after returning.
    """
    if lobby.host_id != by.id:
        raise LobbyForbidden("only the host can change room controls")
    if lobby.status == LobbyStatus.CLOSED:
        raise LobbyClosed()

    if patch.is_private is not None:
        lobby.is_private = bool(patch.is_private)
    if patch.allow_spectators is not None:
        lobby.allow_spectators = bool(patch.allow_spectators)
    if patch.music_track is not None:
        # Accept any non-empty short string — we don't hard-validate against
        # KNOWN_MUSIC_TRACKS so new files can be dropped in without a deploy.
        if len(patch.music_track) > 64:
            raise LobbyInvalidControl("music_track too long")
        lobby.music_track = patch.music_track or None
    if patch.music_volume is not None:
        lobby.music_volume = _clamp(patch.music_volume, 0.0, 1.0)
    if patch.lights_brightness is not None:
        lobby.lights_brightness = _clamp(patch.lights_brightness, 0.0, 1.0)
    if patch.lights_hue is not None:
        lobby.lights_hue = _validate_hex(patch.lights_hue)
    if patch.time_control is not None:
        if patch.time_control not in TIME_CONTROL_PRESETS:
            raise LobbyInvalidControl(
                f"time_control must be one of {TIME_CONTROL_PRESETS}"
            )
        lobby.time_control = patch.time_control

    lobby.updated_at = datetime.utcnow()
    session.commit()
    session.refresh(lobby)
    return lobby


# --- browse --------------------------------------------------------------


def list_public_open_lobbies(session: Session, *, limit: int = 50) -> list[Lobby]:
    """Public lobbies (is_private=False) currently OPEN, newest-first.

    IN_MATCH lobbies are excluded because there's no useful action a
    browsing user can take on one (can't join — full — and spectating
    has its own entry point).
    """
    rows = session.execute(
        select(Lobby)
        .where(Lobby.is_private.is_(False))
        .where(Lobby.status == LobbyStatus.OPEN)
        .order_by(Lobby.created_at.desc())
        .limit(limit)
    ).scalars()
    return list(rows)


# --- internals ------------------------------------------------------------


def _ensure_player_free(
    session: Session, player_id: str, but_keep_lobby_id: str | None = None
) -> None:
    """Leave any currently-active lobby (except `but_keep_lobby_id`).

    Mirrors the implicit "one lobby at a time" rule without adding a DB
    constraint we might regret. Called inside create_lobby / join_lobby.
    """
    active = session.execute(
        select(LobbyMembership, Lobby)
        .join(Lobby, Lobby.id == LobbyMembership.lobby_id)
        .where(LobbyMembership.player_id == player_id)
        .where(LobbyMembership.left_at.is_(None))
        .where(Lobby.status != LobbyStatus.CLOSED)
    ).all()
    now = datetime.utcnow()
    for mem, lob in active:
        if but_keep_lobby_id is not None and lob.id == but_keep_lobby_id:
            continue
        mem.left_at = now
        # Mirror the host-transfer / close-on-empty logic from leave_lobby.
        siblings = [
            m for m in active_members(session, lob.id)
            if m.id != mem.id
        ]
        if not siblings:
            lob.status = LobbyStatus.CLOSED
            lob.closed_at = now
        elif mem.role == LobbyRole.HOST:
            new_host = sorted(siblings, key=lambda m: m.joined_at)[0]
            new_host.role = LobbyRole.HOST
            lob.host_id = new_host.player_id


def _clamp(x: float, lo: float, hi: float) -> float:
    try:
        x = float(x)
    except (TypeError, ValueError):
        raise LobbyInvalidControl("value must be a number")
    if x < lo or x > hi:
        raise LobbyInvalidControl(f"value must be between {lo} and {hi}")
    return x


def _validate_hex(value: str) -> str:
    v = (value or "").strip()
    if not v.startswith("#") or len(v) not in (4, 7):
        raise LobbyInvalidControl("lights_hue must be #RGB or #RRGGBB")
    try:
        int(v[1:], 16)
    except ValueError:
        raise LobbyInvalidControl("lights_hue must be a valid hex colour")
    # Normalise to uppercase 7-char form if already #RRGGBB; short form left as-is.
    if len(v) == 7:
        return "#" + v[1:].upper()
    return v.upper()
