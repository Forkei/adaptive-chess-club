"""Phase 4.2 — player-vs-player lobbies + matchmaking + PvP matches.

Concepts:

- `Lobby`: a persistent room a host creates. Holds the ambient state
  (music, lights), the door/lock toggles (`is_private`,
  `allow_spectators`), and a membership roster. Matches are played
  *inside* a lobby — one lobby can host many matches over time.

- `LobbyMembership`: junction table, one row per player currently (or
  previously) in the lobby. `left_at` being NULL means "still here".

- `MatchmakingQueue`: a row per player who asked to be paired. A
  periodic reaper walks it, finds compatible Elo pairs, creates a
  public lobby, stamps `matched_lobby_id` on both entries.

- `PvpMatch`: a finished-or-running match inside a lobby. Separate
  from the existing `Match` table (which is player-vs-character);
  PvpMatch has two player FKs, no character, no engine analysis.
  Moves live as a JSON list on the row — PvP doesn't need the per-move
  eval/memory/mood side-data that `Move` carries.
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.utcnow()


class LobbyStatus(str, enum.Enum):
    OPEN = "open"            # accepting players / waiting to start
    IN_MATCH = "in_match"    # a PvpMatch is currently running
    CLOSED = "closed"        # dissolved; no further joins


class LobbyRole(str, enum.Enum):
    HOST = "host"
    GUEST = "guest"


class Lobby(Base):
    __tablename__ = "lobbies"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)

    # Short uppercase alphanumeric invite code. Unique across all open lobbies.
    # Both public and private lobbies carry a code — it's the invite-by-URL
    # path. Public lobbies also appear in the browser.
    code: Mapped[str] = mapped_column(String(8), nullable=False, unique=True, index=True)

    host_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("players.id", ondelete="CASCADE"), nullable=False, index=True
    )

    # "Door" — private lobbies do not show up in the public browser; only
    # the invite code / URL gets a player in.
    is_private: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default="0"
    )
    # "Lock" — when False, spectator connections are rejected at the socket
    # handshake. Host can toggle mid-match.
    allow_spectators: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True, server_default="1"
    )

    # --- Ambient controls (Phase 4.2e) -----------------------------------
    # All optional; clients default to a sensible room feel when NULL.
    music_track: Mapped[str | None] = mapped_column(String(64), nullable=True)
    music_volume: Mapped[float] = mapped_column(
        Float, nullable=False, default=0.5, server_default="0.5"
    )
    lights_brightness: Mapped[float] = mapped_column(
        Float, nullable=False, default=0.7, server_default="0.7"
    )
    # Hex colour (#RRGGBB) for the hue tint. Keeps the column format-agnostic.
    lights_hue: Mapped[str] = mapped_column(
        String(16), nullable=False, default="#C9A66B", server_default="#C9A66B"
    )

    # Phase 4.2.5 — tabletop clock preset. Accepted values:
    # "untimed" / "5+0" / "10+0" / "15+10". Display-only at 4.2.5 (no
    # server-side ticking / timeout yet).
    time_control: Mapped[str] = mapped_column(
        String(16), nullable=False, default="untimed", server_default="untimed"
    )

    status: Mapped[LobbyStatus] = mapped_column(
        Enum(LobbyStatus, name="lobby_status", values_callable=lambda e: [m.value for m in e]),
        nullable=False,
        default=LobbyStatus.OPEN,
        index=True,
    )

    # The PvpMatch that this lobby is currently running (if status=IN_MATCH).
    # Reference stored by id rather than FK to avoid a cycle during match
    # creation; service layer maintains consistency.
    current_match_id: Mapped[str | None] = mapped_column(String(36), nullable=True)

    # Lobbies created via matchmaking carry a pointer so we can show "auto-paired
    # from the queue" in the UI. Nullable for host-created lobbies.
    via_matchmaking: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default="0"
    )

    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=_now, onupdate=_now
    )
    closed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    memberships: Mapped[list["LobbyMembership"]] = relationship(
        back_populates="lobby", cascade="all, delete-orphan", lazy="selectin"
    )


class LobbyMembership(Base):
    __tablename__ = "lobby_memberships"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    lobby_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("lobbies.id", ondelete="CASCADE"), nullable=False, index=True
    )
    player_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("players.id", ondelete="CASCADE"), nullable=False, index=True
    )
    role: Mapped[LobbyRole] = mapped_column(
        Enum(LobbyRole, name="lobby_role", values_callable=lambda e: [m.value for m in e]),
        nullable=False,
        default=LobbyRole.GUEST,
    )
    joined_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=_now)
    # Nullable — still-present members have NULL. Leaving sets it.
    left_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    lobby: Mapped[Lobby] = relationship(back_populates="memberships")


class MatchmakingQueue(Base):
    __tablename__ = "matchmaking_queue"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    player_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("players.id", ondelete="CASCADE"),
        nullable=False, unique=True, index=True,  # only one active entry per player
    )
    queued_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=_now)
    elo_at_queue: Mapped[int] = mapped_column(Integer, nullable=False)

    # How many times the Elo band around `elo_at_queue` has been widened.
    # 0 = ±50, 1 = ±100, 2 = ±200, 3 = ±400, 4+ = open. Tunable in service.
    band_expansion_step: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default="0"
    )

    # Set by the matcher when a pair is found. The player's client polls
    # the queue; once `matched_lobby_id` is non-null, it navigates to the
    # lobby. The row stays for audit; a new enqueue creates a new row.
    matched_lobby_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("lobbies.id", ondelete="SET NULL"), nullable=True
    )

    # User-initiated cancel. Non-null rows are ignored by the matcher.
    canceled_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class PvpMatchStatus(str, enum.Enum):
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    RESIGNED = "resigned"
    ABANDONED = "abandoned"


class PvpMatchResult(str, enum.Enum):
    WHITE_WIN = "white_win"
    BLACK_WIN = "black_win"
    DRAW = "draw"
    ABANDONED = "abandoned"


class PvpMatch(Base):
    """Chess game between two players inside a lobby.

    Moves are stored as an ordered list of JSON objects on this row:

        [
          {"uci": "e2e4", "san": "e4", "fen_after": "...",
           "side": "white", "time_taken_ms": 0, "ts": "<iso>"},
          ...
        ]

    Kept inline because PvP matches don't need the richer per-move
    metadata (eval_cp, surfaced_memory_ids, agent_chat_after) that the
    PvE `Move` table stores.
    """

    __tablename__ = "pvp_matches"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    lobby_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("lobbies.id", ondelete="CASCADE"), nullable=False, index=True
    )

    white_player_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("players.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    black_player_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("players.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )

    initial_fen: Mapped[str] = mapped_column(String(120), nullable=False)
    current_fen: Mapped[str] = mapped_column(String(120), nullable=False)
    moves: Mapped[list[dict[str, Any]]] = mapped_column(JSON, nullable=False, default=list)
    move_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    status: Mapped[PvpMatchStatus] = mapped_column(
        Enum(PvpMatchStatus, name="pvp_match_status",
             values_callable=lambda e: [m.value for m in e]),
        nullable=False, default=PvpMatchStatus.IN_PROGRESS, index=True,
    )
    result: Mapped[PvpMatchResult | None] = mapped_column(
        Enum(PvpMatchResult, name="pvp_match_result",
             values_callable=lambda e: [m.value for m in e]),
        nullable=True,
    )

    # Mirrors lobby.is_private at the moment the match starts so later
    # toggles on the lobby don't retroactively flip Elo treatment.
    is_private: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default="0"
    )

    white_elo_at_start: Mapped[int] = mapped_column(Integer, nullable=False)
    black_elo_at_start: Mapped[int] = mapped_column(Integer, nullable=False)
    white_elo_at_end: Mapped[int | None] = mapped_column(Integer, nullable=True)
    black_elo_at_end: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Holder for disconnect bookkeeping (disconnect_player_id, deadline_iso).
    extra_state: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)

    # Phase 4.4 — clocks. Copied from Lobby.time_control at match start so
    # later lobby edits don't mutate a live game. `None` on both clock_ms
    # fields means untimed; flag-fall logic skips that branch.
    time_control: Mapped[str] = mapped_column(
        String(16), nullable=False, default="untimed", server_default="untimed"
    )
    increment_ms: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default="0"
    )
    white_clock_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    black_clock_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # Wall-clock instant at which the side-to-move's timer started ticking.
    # Updated after each move application.
    last_tick_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    started_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=_now)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True, index=True)
