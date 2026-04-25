"""Phase 4.4 — PvP clocks: parser, tick, flag-fall."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from app.db import SessionLocal
from app.lobbies import pvp_service as pvp
from app.lobbies.service import CreateLobbyIn, create_lobby, join_lobby, update_controls, ControlsPatch
from app.models.lobby import PvpMatch, PvpMatchResult, PvpMatchStatus
from app.models.match import Player


def _mk_player(s, username: str) -> Player:
    p = Player(username=username, display_name=username, elo=1200)
    s.add(p); s.commit(); s.refresh(p)
    return p


def _two_seats(s, *, time_control: str = "5+0"):
    host = _mk_player(s, f"h_{id(s)%10000}")
    guest = _mk_player(s, f"g_{id(s)%10000}")
    lob = create_lobby(s, host, CreateLobbyIn())
    join_lobby(s, lob, guest)
    if time_control != "untimed":
        update_controls(s, lob, by=host, patch=ControlsPatch(time_control=time_control))
    return lob, host, guest


# --- parser -----------------------------------------------------------------


@pytest.mark.parametrize("tc, expected", [
    ("5+0",   (300_000, 0)),
    ("10+0",  (600_000, 0)),
    ("15+10", (900_000, 10_000)),
    ("untimed", (None, 0)),
    (None, (None, 0)),
    ("", (None, 0)),
    ("nonsense", (None, 0)),
    ("-1+0", (None, 0)),
])
def test_parse_time_control(tc, expected):
    assert pvp._parse_time_control(tc) == expected


# --- start_match clock init ------------------------------------------------


def test_start_match_sets_clocks_from_lobby_time_control():
    with SessionLocal() as s:
        lob, host, _ = _two_seats(s, time_control="5+0")
        match = pvp.start_match(s, lob, by=host)
        assert match.time_control == "5+0"
        assert match.increment_ms == 0
        assert match.white_clock_ms == 300_000
        assert match.black_clock_ms == 300_000
        assert match.last_tick_at is not None


def test_start_match_untimed_leaves_clocks_null():
    with SessionLocal() as s:
        lob, host, _ = _two_seats(s, time_control="untimed")
        match = pvp.start_match(s, lob, by=host)
        assert match.time_control == "untimed"
        assert match.white_clock_ms is None
        assert match.black_clock_ms is None
        assert match.last_tick_at is None


# --- tick + increment ------------------------------------------------------


def test_apply_move_subtracts_mover_clock():
    with SessionLocal() as s:
        lob, host, guest = _two_seats(s, time_control="5+0")
        match = pvp.start_match(s, lob, by=host, white_choice="white")
        # Rewind last_tick_at so the elapsed is a concrete 2000ms.
        match.last_tick_at = datetime.utcnow() - timedelta(milliseconds=2000)
        s.commit()

        white_before = match.white_clock_ms
        pvp.apply_move(s, match, by=host, uci="e2e4")
        s.refresh(match)
        # Elapsed ~2s off white's clock; black untouched.
        assert white_before - match.white_clock_ms >= 1800
        assert match.black_clock_ms == 300_000


def test_apply_move_credits_fischer_increment():
    with SessionLocal() as s:
        lob, host, _ = _two_seats(s, time_control="15+10")
        match = pvp.start_match(s, lob, by=host, white_choice="white")
        match.last_tick_at = datetime.utcnow() - timedelta(milliseconds=1000)
        s.commit()
        pvp.apply_move(s, match, by=host, uci="e2e4")
        s.refresh(match)
        # 15m - 1s + 10s increment = 15m9s = 909_000 ms (± a few ms of drift)
        assert match.white_clock_ms >= 908_000
        assert match.white_clock_ms <= 910_000


def test_apply_move_flagfall_when_elapsed_exceeds_clock():
    """If the mover takes longer than their remaining time, the move is
    rejected and the opponent wins on time."""
    with SessionLocal() as s:
        lob, host, guest = _two_seats(s, time_control="5+0")
        match = pvp.start_match(s, lob, by=host, white_choice="white")
        # Force white's clock down + last_tick to be long ago.
        match.white_clock_ms = 100
        match.last_tick_at = datetime.utcnow() - timedelta(seconds=5)
        s.commit()
        result = pvp.apply_move(s, match, by=host, uci="e2e4")
        assert result.game_over is True
        assert result.reason == "time"
        assert result.result == PvpMatchResult.BLACK_WIN
        s.refresh(match)
        assert match.status == PvpMatchStatus.COMPLETED
        # Move NOT recorded on flag-fall.
        assert match.move_count == 0


# --- flag-fall sweep -------------------------------------------------------


def test_flagfall_sweep_flags_idle_match():
    with SessionLocal() as s:
        lob, host, guest = _two_seats(s, time_control="5+0")
        match = pvp.start_match(s, lob, by=host, white_choice="white")
        # Simulate white stepped away with 50ms on the clock.
        match.white_clock_ms = 50
        match.last_tick_at = datetime.utcnow() - timedelta(seconds=1)
        s.commit()
        flagged = pvp.flagfall_sweep(s)
        assert match.id in flagged
        s.refresh(match)
        assert match.status == PvpMatchStatus.COMPLETED
        assert match.result == PvpMatchResult.BLACK_WIN


def test_flagfall_sweep_noop_for_untimed():
    with SessionLocal() as s:
        lob, host, _ = _two_seats(s, time_control="untimed")
        match = pvp.start_match(s, lob, by=host)
        flagged = pvp.flagfall_sweep(s)
        assert match.id not in flagged
        s.refresh(match)
        assert match.status == PvpMatchStatus.IN_PROGRESS


def test_flagfall_sweep_noop_when_time_remains():
    with SessionLocal() as s:
        lob, host, _ = _two_seats(s, time_control="5+0")
        match = pvp.start_match(s, lob, by=host)
        # Fresh match — clock is full, last_tick just set. No flag.
        flagged = pvp.flagfall_sweep(s)
        assert match.id not in flagged
