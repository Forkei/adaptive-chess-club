"""Phase 4.2c — matchmaking queue unit tests."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from app.db import SessionLocal
from app.lobbies import matchmaking as mm
from app.lobbies.service import active_members
from app.models.lobby import Lobby, MatchmakingQueue
from app.models.match import Player


def _mk_player(s, username: str, elo: int = 1200) -> Player:
    p = Player(username=username, display_name=username, elo=elo)
    s.add(p)
    s.commit()
    s.refresh(p)
    return p


# --- enqueue / cancel / poll ----------------------------------------------


def test_enqueue_adds_entry_with_current_elo():
    with SessionLocal() as s:
        p = _mk_player(s, "mm_1", elo=1500)
        entry = mm.enqueue(s, p)
        assert entry.player_id == p.id
        assert entry.elo_at_queue == 1500
        assert entry.band_expansion_step == 0
        assert entry.matched_lobby_id is None
        assert entry.canceled_at is None


def test_enqueue_twice_raises():
    with SessionLocal() as s:
        p = _mk_player(s, "mm_2")
        mm.enqueue(s, p)
        with pytest.raises(mm.AlreadyQueued):
            mm.enqueue(s, p)


def test_cancel_ends_queue_entry():
    with SessionLocal() as s:
        p = _mk_player(s, "mm_3")
        mm.enqueue(s, p)
        assert mm.cancel(s, p) is True
        assert mm.cancel(s, p) is False  # already canceled


def test_poll_when_not_queued():
    with SessionLocal() as s:
        p = _mk_player(s, "mm_unq")
        r = mm.poll(s, p)
        assert r.queued is False
        assert r.matched_lobby_id is None


def test_poll_when_queued_reports_wait_time():
    with SessionLocal() as s:
        p = _mk_player(s, "mm_poll")
        mm.enqueue(s, p)
        r = mm.poll(s, p)
        assert r.queued is True
        assert r.waited_seconds >= 0


# --- band widening + pairing ----------------------------------------------


def test_try_match_pairs_close_elo_immediately():
    with SessionLocal() as s:
        a = _mk_player(s, "mm_a1", elo=1500)
        b = _mk_player(s, "mm_b1", elo=1530)  # within ±50
        mm.enqueue(s, a)
        mm.enqueue(s, b)
        lobbies = mm.try_match_step(s)
        assert len(lobbies) == 1
        # Both entries now have matched_lobby_id.
        rows = s.query(MatchmakingQueue).filter(
            MatchmakingQueue.player_id.in_([a.id, b.id])
        ).all()
        assert all(r.matched_lobby_id == lobbies[0] for r in rows)

        # The lobby exists + both players are members.
        lob = s.get(Lobby, lobbies[0])
        assert lob is not None
        assert lob.via_matchmaking is True
        member_ids = {m.player_id for m in active_members(s, lob.id)}
        assert member_ids == {a.id, b.id}


def test_try_match_does_not_pair_out_of_band():
    with SessionLocal() as s:
        a = _mk_player(s, "mm_a2", elo=1200)
        b = _mk_player(s, "mm_b2", elo=1600)  # 400 gap > step-0 band of 50
        mm.enqueue(s, a)
        mm.enqueue(s, b)
        lobbies = mm.try_match_step(s)
        assert lobbies == []


def test_band_widens_over_time_and_pairs():
    with SessionLocal() as s:
        a = _mk_player(s, "mm_a3", elo=1200)
        b = _mk_player(s, "mm_b3", elo=1600)  # 400 gap
        mm.enqueue(s, a)
        mm.enqueue(s, b)
        # Backdate their queued_at so the matcher thinks they've been
        # waiting long enough to jump to step 3 (±400).
        for row in s.query(MatchmakingQueue).all():
            row.queued_at = datetime.utcnow() - timedelta(
                seconds=mm.WIDEN_AFTER_SECONDS * 3 + 1
            )
        s.commit()

        lobbies = mm.try_match_step(s)
        assert len(lobbies) == 1
        # Both entries were bumped to step 3.
        for row in s.query(MatchmakingQueue).all():
            assert row.band_expansion_step >= 3


def test_open_step_pairs_anyone():
    with SessionLocal() as s:
        a = _mk_player(s, "mm_a4", elo=800)
        b = _mk_player(s, "mm_b4", elo=2500)  # huge gap
        mm.enqueue(s, a)
        mm.enqueue(s, b)
        # Push both to OPEN_STEP (4+).
        for row in s.query(MatchmakingQueue).all():
            row.queued_at = datetime.utcnow() - timedelta(
                seconds=mm.WIDEN_AFTER_SECONDS * mm.OPEN_STEP + 10
            )
        s.commit()
        lobbies = mm.try_match_step(s)
        assert len(lobbies) == 1


def test_canceled_entries_are_skipped():
    with SessionLocal() as s:
        a = _mk_player(s, "mm_ax", elo=1500)
        b = _mk_player(s, "mm_bx", elo=1510)
        mm.enqueue(s, a)
        mm.enqueue(s, b)
        mm.cancel(s, a)
        lobbies = mm.try_match_step(s)
        assert lobbies == []


def test_pair_resolution_prefers_oldest():
    with SessionLocal() as s:
        a = _mk_player(s, "mm_old", elo=1500)
        b = _mk_player(s, "mm_mid", elo=1510)
        c = _mk_player(s, "mm_new", elo=1520)
        mm.enqueue(s, a)
        mm.enqueue(s, b)
        mm.enqueue(s, c)
        # Backdate a so it's oldest, b middle, c youngest.
        base = datetime.utcnow()
        for row in s.query(MatchmakingQueue).all():
            if row.player_id == a.id:
                row.queued_at = base - timedelta(seconds=30)
            elif row.player_id == b.id:
                row.queued_at = base - timedelta(seconds=20)
            else:
                row.queued_at = base - timedelta(seconds=5)
        s.commit()

        lobbies = mm.try_match_step(s)
        assert len(lobbies) == 1
        # a and b paired (both old), c left waiting.
        lob = s.get(Lobby, lobbies[0])
        members = {m.player_id for m in active_members(s, lob.id)}
        assert members == {a.id, b.id}
        c_row = s.query(MatchmakingQueue).filter_by(player_id=c.id).one()
        assert c_row.matched_lobby_id is None


def test_poll_after_match_returns_lobby_id():
    with SessionLocal() as s:
        a = _mk_player(s, "mm_pa", elo=1500)
        b = _mk_player(s, "mm_pb", elo=1510)
        mm.enqueue(s, a)
        mm.enqueue(s, b)
        mm.try_match_step(s)
        r = mm.poll(s, a)
        assert r.queued is False
        assert r.matched_lobby_id is not None


def test_reap_stale_cancels_old_entries():
    with SessionLocal() as s:
        old = _mk_player(s, "mm_reap")
        mm.enqueue(s, old)
        # Backdate beyond threshold.
        for row in s.query(MatchmakingQueue).all():
            row.queued_at = datetime.utcnow() - timedelta(minutes=90)
        s.commit()

        n = mm.reap_stale(s, older_than_minutes=60)
        assert n == 1
        row = s.query(MatchmakingQueue).filter_by(player_id=old.id).one()
        assert row.canceled_at is not None
