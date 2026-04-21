"""Phase 4.2c — matchmaking queue.

Players call `enqueue(player)` and poll `poll(player)`. A background
worker (`try_match_step`) walks the queue periodically, finds pairs
whose Elo bands overlap, creates a public lobby, and stamps both
entries with `matched_lobby_id`. The poller then notices and redirects
the client into the lobby.

Band widening: every `WIDEN_AFTER_SECONDS` of unmatched time the
player's "band_expansion_step" counter is bumped and the search
tolerance grows. After step 4 we match anyone remaining.

    step  0:  ±50
    step  1:  ±100
    step  2:  ±200
    step  3:  ±400
    step 4+:  ∞ (match anyone)

Design notes:
- The queue is backed by a SQL table so it survives a server restart.
  In-process, a periodic asyncio task drives `try_match_step`.
- A player can only have one active (non-canceled, un-matched) entry
  at a time — enforced via a unique index on `player_id`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.lobbies.service import CreateLobbyIn, create_lobby, join_lobby
from app.models.lobby import MatchmakingQueue
from app.models.match import Player

logger = logging.getLogger(__name__)

# Band steps — exposed as a constant so tests + UI can align.
BANDS: tuple[int, ...] = (50, 100, 200, 400)
WIDEN_AFTER_SECONDS: int = 20
# Matches involving step >= len(BANDS) skip the Elo check entirely.
OPEN_STEP: int = len(BANDS)


class MatchmakingError(Exception):
    code = "matchmaking_error"

    def __init__(self, message: str = "") -> None:
        super().__init__(message or self.code)


class AlreadyQueued(MatchmakingError):
    code = "matchmaking_already_queued"


# --- enqueue / cancel / poll ----------------------------------------------


def _active_entry(session: Session, player_id: str) -> MatchmakingQueue | None:
    """Return the player's current active queue entry, if any.

    Active = canceled_at IS NULL AND matched_lobby_id IS NULL.
    """
    return session.execute(
        select(MatchmakingQueue)
        .where(MatchmakingQueue.player_id == player_id)
        .where(MatchmakingQueue.canceled_at.is_(None))
        .where(MatchmakingQueue.matched_lobby_id.is_(None))
    ).scalar_one_or_none()


def enqueue(session: Session, player: Player) -> MatchmakingQueue:
    """Add `player` to the queue. Raises `AlreadyQueued` if they already
    have an active entry.
    """
    existing = _active_entry(session, player.id)
    if existing is not None:
        raise AlreadyQueued()

    entry = MatchmakingQueue(
        player_id=player.id,
        elo_at_queue=int(player.elo),
        band_expansion_step=0,
    )
    session.add(entry)
    session.commit()
    session.refresh(entry)
    return entry


def cancel(session: Session, player: Player) -> bool:
    """Mark the player's active entry canceled. Returns True iff there
    was one to cancel.
    """
    entry = _active_entry(session, player.id)
    if entry is None:
        return False
    entry.canceled_at = datetime.utcnow()
    session.commit()
    return True


@dataclass(frozen=True)
class PollResult:
    queued: bool
    matched_lobby_id: str | None
    band_step: int
    waited_seconds: float


def poll(session: Session, player: Player) -> PollResult:
    """Return the current state for the player's active entry, or a
    `queued=False, matched_lobby_id=None` sentinel if they aren't in
    the queue at all.
    """
    entry = _active_entry(session, player.id)
    if entry is None:
        # Maybe we just got matched — look for a recent matched row.
        recent_match = session.execute(
            select(MatchmakingQueue)
            .where(MatchmakingQueue.player_id == player.id)
            .where(MatchmakingQueue.matched_lobby_id.is_not(None))
            .where(MatchmakingQueue.canceled_at.is_(None))
            .order_by(MatchmakingQueue.queued_at.desc())
            .limit(1)
        ).scalar_one_or_none()
        if recent_match is not None:
            waited = (datetime.utcnow() - recent_match.queued_at).total_seconds()
            return PollResult(
                queued=False,
                matched_lobby_id=recent_match.matched_lobby_id,
                band_step=recent_match.band_expansion_step,
                waited_seconds=waited,
            )
        return PollResult(
            queued=False, matched_lobby_id=None, band_step=0, waited_seconds=0.0
        )
    waited = (datetime.utcnow() - entry.queued_at).total_seconds()
    return PollResult(
        queued=True,
        matched_lobby_id=None,
        band_step=entry.band_expansion_step,
        waited_seconds=waited,
    )


# --- matcher worker --------------------------------------------------------


def _all_active(session: Session) -> list[MatchmakingQueue]:
    return list(
        session.execute(
            select(MatchmakingQueue)
            .where(MatchmakingQueue.canceled_at.is_(None))
            .where(MatchmakingQueue.matched_lobby_id.is_(None))
            .order_by(MatchmakingQueue.queued_at.asc())
        ).scalars()
    )


def _bump_expansion_if_due(entry: MatchmakingQueue, now: datetime) -> bool:
    """Raise `band_expansion_step` based on wait time. Returns True iff
    it changed.
    """
    waited = (now - entry.queued_at).total_seconds()
    desired = min(OPEN_STEP, int(waited // WIDEN_AFTER_SECONDS))
    if desired > entry.band_expansion_step:
        entry.band_expansion_step = desired
        return True
    return False


def _pair_ok(a: MatchmakingQueue, b: MatchmakingQueue) -> bool:
    """Is the Elo gap between `a` and `b` within the tighter of their
    current bands? Either side at OPEN_STEP means any gap is OK.
    """
    step = min(a.band_expansion_step, b.band_expansion_step)
    if step >= OPEN_STEP:
        return True
    gap = abs(a.elo_at_queue - b.elo_at_queue)
    return gap <= BANDS[step]


def try_match_step(session: Session) -> list[str]:
    """Run one sweep of the matcher. Returns a list of lobby IDs
    created during this step.

    Greedy pairing: walk the queue oldest-first, for each unpaired
    entry try to pair it with the oldest compatible other entry.
    """
    now = datetime.utcnow()
    entries = _all_active(session)
    changed_steps = False
    for e in entries:
        if _bump_expansion_if_due(e, now):
            changed_steps = True
    if changed_steps:
        session.commit()

    created: list[str] = []
    paired_ids: set[str] = set()
    for i, a in enumerate(entries):
        if a.id in paired_ids:
            continue
        for b in entries[i + 1:]:
            if b.id in paired_ids:
                continue
            if a.player_id == b.player_id:
                continue  # shouldn't happen (unique index), guard anyway
            if not _pair_ok(a, b):
                continue
            lobby_id = _create_lobby_and_seat_both(session, a, b)
            paired_ids.add(a.id)
            paired_ids.add(b.id)
            created.append(lobby_id)
            break
    return created


def _create_lobby_and_seat_both(
    session: Session, a: MatchmakingQueue, b: MatchmakingQueue
) -> str:
    """Create a public lobby with `a`'s player as host, seat `b`'s
    player as guest, stamp both queue entries with the lobby id.
    """
    player_a = session.get(Player, a.player_id)
    player_b = session.get(Player, b.player_id)
    if player_a is None or player_b is None:
        # One of them vanished mid-flight (account deletion, corrupt row).
        # Mark both as canceled; they'll requeue naturally.
        a.canceled_at = datetime.utcnow()
        b.canceled_at = datetime.utcnow()
        session.commit()
        return ""

    lob = create_lobby(
        session,
        host=player_a,
        spec=CreateLobbyIn(
            is_private=False,
            allow_spectators=True,
            via_matchmaking=True,
        ),
    )
    join_lobby(session, lob, player_b)
    a.matched_lobby_id = lob.id
    b.matched_lobby_id = lob.id
    session.commit()
    logger.info(
        "[matchmaking] paired %s (elo %s, step %s) with %s (elo %s, step %s) → lobby %s",
        player_a.id, a.elo_at_queue, a.band_expansion_step,
        player_b.id, b.elo_at_queue, b.band_expansion_step,
        lob.id,
    )
    return lob.id


# --- housekeeping ---------------------------------------------------------


def reap_stale(session: Session, *, older_than_minutes: int = 30) -> int:
    """Mark old, never-matched, never-canceled entries as canceled.

    A player who closed their browser without clicking cancel would
    otherwise sit forever in the queue. Called from the periodic sweep.
    """
    cutoff = datetime.utcnow() - timedelta(minutes=older_than_minutes)
    stale = session.execute(
        select(MatchmakingQueue)
        .where(MatchmakingQueue.canceled_at.is_(None))
        .where(MatchmakingQueue.matched_lobby_id.is_(None))
        .where(MatchmakingQueue.queued_at < cutoff)
    ).scalars()
    n = 0
    now = datetime.utcnow()
    for e in stale:
        e.canceled_at = now
        n += 1
    if n:
        session.commit()
    return n
