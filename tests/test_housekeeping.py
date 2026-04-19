"""Tests for the Patch Pass 1 housekeeping sweeps."""

from __future__ import annotations

from datetime import datetime, timedelta

import chess
import pytest

from app.db import SessionLocal
from app.matches import housekeeping
from app.models.character import Character, CharacterState
from app.models.match import (
    Color,
    Match,
    MatchAnalysis,
    MatchAnalysisStatus,
    MatchResult,
    MatchStatus,
    Player,
)


def _mk_character(sess) -> Character:
    c = Character(
        name="H", aggression=5, risk_tolerance=5, patience=5, trash_talk=5,
        target_elo=1400, current_elo=1400, floor_elo=1400, max_elo=1800,
        adaptive=True, state=CharacterState.READY,
    )
    sess.add(c)
    sess.flush()
    return c


def _mk_player(sess, username: str) -> Player:
    p = Player(username=username, display_name=username)
    sess.add(p)
    sess.flush()
    return p


def test_reap_stale_matches_marks_old_empty_matches_abandoned():
    with SessionLocal() as sess:
        p = _mk_player(sess, "hk_stale1")
        c = _mk_character(sess)
        # 2 hours old, 0 moves — should be reaped.
        old_match = Match(
            character_id=c.id, player_id=p.id,
            player_color=Color.WHITE, status=MatchStatus.IN_PROGRESS,
            initial_fen=chess.STARTING_FEN, current_fen=chess.STARTING_FEN,
            move_count=0, character_elo_at_start=1400,
            started_at=datetime.utcnow() - timedelta(hours=2),
        )
        sess.add(old_match)
        # 2 hours old but 5 moves — keep (active game).
        active_match = Match(
            character_id=c.id, player_id=p.id,
            player_color=Color.BLACK, status=MatchStatus.IN_PROGRESS,
            initial_fen=chess.STARTING_FEN, current_fen=chess.STARTING_FEN,
            move_count=5, character_elo_at_start=1400,
            started_at=datetime.utcnow() - timedelta(hours=2),
        )
        sess.add(active_match)
        # Fresh match, 0 moves — keep (still young).
        fresh_match = Match(
            character_id=c.id, player_id=p.id,
            player_color=Color.WHITE, status=MatchStatus.IN_PROGRESS,
            initial_fen=chess.STARTING_FEN, current_fen=chess.STARTING_FEN,
            move_count=0, character_elo_at_start=1400,
            started_at=datetime.utcnow() - timedelta(minutes=5),
        )
        sess.add(fresh_match)
        sess.commit()
        ids = (old_match.id, active_match.id, fresh_match.id)

    reaped = housekeeping.reap_stale_matches()
    assert reaped == 1

    with SessionLocal() as sess:
        old = sess.get(Match, ids[0])
        active = sess.get(Match, ids[1])
        fresh = sess.get(Match, ids[2])
        assert old.status == MatchStatus.ABANDONED
        assert old.result == MatchResult.ABANDONED
        assert old.ended_at is not None
        assert active.status == MatchStatus.IN_PROGRESS
        assert fresh.status == MatchStatus.IN_PROGRESS


def test_fail_stuck_analyses_marks_old_running_rows_failed():
    with SessionLocal() as sess:
        p = _mk_player(sess, "hk_stuck1")
        c = _mk_character(sess)
        m = Match(
            character_id=c.id, player_id=p.id,
            player_color=Color.WHITE, status=MatchStatus.COMPLETED,
            initial_fen=chess.STARTING_FEN, current_fen=chess.STARTING_FEN,
            move_count=30, character_elo_at_start=1400,
        )
        sess.add(m)
        sess.flush()
        # Stuck for 15 minutes — should be failed.
        stuck = MatchAnalysis(
            match_id=m.id, status=MatchAnalysisStatus.RUNNING,
            started_at=datetime.utcnow() - timedelta(minutes=15),
        )
        sess.add(stuck)
        sess.commit()
        stuck_id = stuck.id

    failed = housekeeping.fail_stuck_analyses()
    assert failed == 1

    with SessionLocal() as sess:
        row = sess.get(MatchAnalysis, stuck_id)
        assert row.status == MatchAnalysisStatus.FAILED
        assert row.error is not None
        assert "housekeeping" in row.error.lower()
        assert row.completed_at is not None


def test_fail_stuck_analyses_leaves_fresh_running_alone():
    with SessionLocal() as sess:
        p = _mk_player(sess, "hk_stuck2")
        c = _mk_character(sess)
        m = Match(
            character_id=c.id, player_id=p.id,
            player_color=Color.WHITE, status=MatchStatus.COMPLETED,
            initial_fen=chess.STARTING_FEN, current_fen=chess.STARTING_FEN,
            move_count=30, character_elo_at_start=1400,
        )
        sess.add(m)
        sess.flush()
        # 2 minutes old — still fresh.
        fresh = MatchAnalysis(
            match_id=m.id, status=MatchAnalysisStatus.RUNNING,
            started_at=datetime.utcnow() - timedelta(minutes=2),
        )
        sess.add(fresh)
        sess.commit()
        fresh_id = fresh.id

    failed = housekeeping.fail_stuck_analyses()
    assert failed == 0

    with SessionLocal() as sess:
        row = sess.get(MatchAnalysis, fresh_id)
        assert row.status == MatchAnalysisStatus.RUNNING


@pytest.mark.asyncio
async def test_restore_disconnect_rearms_in_window_cooldowns(monkeypatch):
    """A match dropped 30s ago with a 300s window gets re-armed (~270s remaining)."""
    from app.sockets import disconnect as disconnect_registry

    disconnect_registry.reset_all()
    with SessionLocal() as sess:
        p = _mk_player(sess, "hk_disc1")
        c = _mk_character(sess)
        m = Match(
            character_id=c.id, player_id=p.id,
            player_color=Color.WHITE, status=MatchStatus.IN_PROGRESS,
            initial_fen=chess.STARTING_FEN, current_fen=chess.STARTING_FEN,
            move_count=3, character_elo_at_start=1400,
            extra_state={
                "disconnect_started_at": (datetime.utcnow() - timedelta(seconds=30)).isoformat(),
                "disconnect_player_id": p.id,
            },
        )
        sess.add(m)
        sess.commit()
        match_id = m.id

    rearmed, expired = await housekeeping.restore_disconnect_cooldowns()
    assert rearmed == 1
    assert expired == 0
    assert disconnect_registry.is_active(match_id)
    disconnect_registry.reset_all()
