"""Unit tests for Phase 3c discovery + leaderboard + hall-of-fame queries.

No sockets — these exercise the SQL aggregations directly against an
in-memory SQLite DB seeded with canned matches.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import chess
import pytest

from app.db import SessionLocal
from app.discovery import (
    character_leaderboard,
    hall_of_fame_for_character,
    list_live_matches,
    list_recent_matches,
    player_leaderboard,
)
from app.discovery.leaderboard import MIN_MATCHES_FOR_LEADERBOARD
from app.models.character import Character, CharacterState, ContentRating, Visibility
from app.models.match import Color, Match, MatchResult, MatchStatus, Player


# --- Fixtures -------------------------------------------------------------


def _mk_player(sess, username: str, max_rating: ContentRating = ContentRating.FAMILY) -> Player:
    p = Player(username=username, display_name=username, max_content_rating=max_rating)
    sess.add(p); sess.flush(); return p


def _mk_character(
    sess,
    *,
    name: str,
    owner_id: str | None = None,
    content_rating: ContentRating = ContentRating.FAMILY,
    visibility: Visibility = Visibility.PUBLIC,
    elo: int = 1400,
) -> Character:
    c = Character(
        name=name, aggression=5, risk_tolerance=5, patience=5, trash_talk=5,
        target_elo=elo, current_elo=elo, floor_elo=elo, max_elo=elo + 400,
        adaptive=True, state=CharacterState.READY,
        owner_id=owner_id, content_rating=content_rating, visibility=visibility,
    )
    sess.add(c); sess.flush(); return c


def _mk_match(
    sess,
    *,
    player: Player,
    character: Character,
    status: MatchStatus,
    result: MatchResult | None = None,
    player_color: Color = Color.WHITE,
    ended_at: datetime | None = None,
) -> Match:
    m = Match(
        character_id=character.id,
        player_id=player.id,
        player_color=player_color,
        status=status,
        result=result,
        initial_fen=chess.STARTING_FEN,
        current_fen=chess.STARTING_FEN,
        move_count=10,
        character_elo_at_start=character.current_elo,
        ended_at=ended_at,
    )
    sess.add(m); sess.flush(); return m


# --- Discovery: live / recent ---------------------------------------------


def test_live_matches_excludes_viewers_own():
    with SessionLocal() as sess:
        viewer = _mk_player(sess, "viewer")
        opponent = _mk_player(sess, "opponent")
        char = _mk_character(sess, name="X")
        _mk_match(sess, player=viewer, character=char, status=MatchStatus.IN_PROGRESS)
        _mk_match(sess, player=opponent, character=char, status=MatchStatus.IN_PROGRESS)
        sess.commit()

        live = list_live_matches(sess, viewer=viewer)
        assert len(live) == 1
        assert live[0].player_id == opponent.id


def test_live_matches_filters_by_rating():
    with SessionLocal() as sess:
        family_viewer = _mk_player(sess, "fam", max_rating=ContentRating.FAMILY)
        other = _mk_player(sess, "other", max_rating=ContentRating.MATURE)
        family_char = _mk_character(sess, name="Family", content_rating=ContentRating.FAMILY)
        mature_char = _mk_character(sess, name="Mature", content_rating=ContentRating.MATURE)
        _mk_match(sess, player=other, character=family_char, status=MatchStatus.IN_PROGRESS)
        _mk_match(sess, player=other, character=mature_char, status=MatchStatus.IN_PROGRESS)
        sess.commit()

        rows = list_live_matches(sess, viewer=family_viewer)
        names = {r.character_name for r in rows}
        assert "Mature" not in names
        assert "Family" in names


def test_live_matches_hides_private_character_from_non_owner():
    with SessionLocal() as sess:
        viewer = _mk_player(sess, "viewer")
        owner = _mk_player(sess, "owner")
        other = _mk_player(sess, "other")
        priv = _mk_character(sess, name="Priv", owner_id=owner.id, visibility=Visibility.PRIVATE)
        _mk_match(sess, player=other, character=priv, status=MatchStatus.IN_PROGRESS)
        sess.commit()

        rows_viewer = list_live_matches(sess, viewer=viewer)
        rows_owner = list_live_matches(sess, viewer=owner)
        assert rows_viewer == []
        assert len(rows_owner) == 1


def test_recent_matches_includes_abandoned_and_viewer_own():
    with SessionLocal() as sess:
        viewer = _mk_player(sess, "viewer")
        char = _mk_character(sess, name="C")
        _mk_match(
            sess, player=viewer, character=char,
            status=MatchStatus.COMPLETED, result=MatchResult.WHITE_WIN,
            ended_at=datetime.utcnow(),
        )
        _mk_match(
            sess, player=viewer, character=char,
            status=MatchStatus.ABANDONED, result=MatchResult.ABANDONED,
            ended_at=datetime.utcnow(),
        )
        sess.commit()

        rows = list_recent_matches(sess, viewer=viewer)
        statuses = [r.status for r in rows]
        assert "completed" in statuses
        assert "abandoned" in statuses


# --- Leaderboard ----------------------------------------------------------


def _populate_char_winrate(
    sess, character: Character, player: Player,
    wins: int = 0, losses: int = 0, draws: int = 0, abandoned: int = 0,
    ended_offset: timedelta = timedelta(days=1),
):
    base = datetime.utcnow() - ended_offset
    # Character is BLACK when player_color == WHITE.
    # Character wins: white player loses to black char -> BLACK_WIN.
    for _ in range(wins):
        _mk_match(sess, player=player, character=character,
                  status=MatchStatus.COMPLETED, result=MatchResult.BLACK_WIN,
                  player_color=Color.WHITE, ended_at=base)
    for _ in range(losses):
        _mk_match(sess, player=player, character=character,
                  status=MatchStatus.COMPLETED, result=MatchResult.WHITE_WIN,
                  player_color=Color.WHITE, ended_at=base)
    for _ in range(draws):
        _mk_match(sess, player=player, character=character,
                  status=MatchStatus.COMPLETED, result=MatchResult.DRAW,
                  player_color=Color.WHITE, ended_at=base)
    for _ in range(abandoned):
        _mk_match(sess, player=player, character=character,
                  status=MatchStatus.ABANDONED, result=MatchResult.ABANDONED,
                  player_color=Color.WHITE, ended_at=base)


def test_character_leaderboard_ranks_by_win_rate():
    with SessionLocal() as sess:
        viewer = _mk_player(sess, "viewer")
        opp = _mk_player(sess, "opp")
        hot = _mk_character(sess, name="Hot")
        cold = _mk_character(sess, name="Cold")
        _populate_char_winrate(sess, hot, opp, wins=8, losses=2)
        _populate_char_winrate(sess, cold, opp, wins=3, losses=7)
        sess.commit()

        rows = character_leaderboard(sess, viewer=viewer, window="all")
        assert [r.character_name for r in rows] == ["Hot", "Cold"]
        assert rows[0].win_rate == 0.8
        assert rows[1].win_rate == 0.3


def test_character_leaderboard_threshold_5_matches():
    with SessionLocal() as sess:
        viewer = _mk_player(sess, "viewer")
        opp = _mk_player(sess, "opp")
        below = _mk_character(sess, name="Below")
        above = _mk_character(sess, name="Above")
        _populate_char_winrate(sess, below, opp, wins=2, losses=2)  # 4 matches
        _populate_char_winrate(sess, above, opp, wins=3, losses=2)  # 5 matches
        sess.commit()

        rows = character_leaderboard(sess, viewer=viewer, window="all")
        names = [r.character_name for r in rows]
        assert "Below" not in names
        assert "Above" in names


def test_character_leaderboard_abandoned_counts_as_character_win():
    with SessionLocal() as sess:
        viewer = _mk_player(sess, "viewer")
        opp = _mk_player(sess, "opp")
        provoker = _mk_character(sess, name="Provoker")
        _populate_char_winrate(sess, provoker, opp, wins=1, losses=1, abandoned=3)  # 5 matches
        sess.commit()

        rows = character_leaderboard(sess, viewer=viewer, window="all")
        assert len(rows) == 1
        r = rows[0]
        assert r.total_matches == 5
        assert r.wins == 4  # 1 real win + 3 abandoned
        assert r.losses == 1


def test_character_leaderboard_window_filter():
    with SessionLocal() as sess:
        viewer = _mk_player(sess, "viewer")
        opp = _mk_player(sess, "opp")
        recent_char = _mk_character(sess, name="Recent")
        old_char = _mk_character(sess, name="Old")
        _populate_char_winrate(sess, recent_char, opp, wins=5, losses=0, ended_offset=timedelta(days=1))
        _populate_char_winrate(sess, old_char, opp, wins=5, losses=0, ended_offset=timedelta(days=100))
        sess.commit()

        rows_week = character_leaderboard(sess, viewer=viewer, window="7d")
        names_week = [r.character_name for r in rows_week]
        assert names_week == ["Recent"]

        rows_all = character_leaderboard(sess, viewer=viewer, window="all")
        names_all = {r.character_name for r in rows_all}
        assert "Recent" in names_all and "Old" in names_all


def test_character_leaderboard_hides_mature_from_family_viewer():
    with SessionLocal() as sess:
        fam = _mk_player(sess, "fam", max_rating=ContentRating.FAMILY)
        opp = _mk_player(sess, "opp")
        fam_char = _mk_character(sess, name="Fam", content_rating=ContentRating.FAMILY)
        mat_char = _mk_character(sess, name="Mat", content_rating=ContentRating.MATURE)
        _populate_char_winrate(sess, fam_char, opp, wins=5, losses=0)
        _populate_char_winrate(sess, mat_char, opp, wins=5, losses=0)
        sess.commit()

        rows = character_leaderboard(sess, viewer=fam, window="all")
        names = [r.character_name for r in rows]
        assert "Mat" not in names


def test_player_leaderboard_abandoned_counts_as_player_loss():
    """Mirror of the character-side: player ABANDONED = player loss (character wins)."""
    with SessionLocal() as sess:
        viewer = _mk_player(sess, "viewer")
        hero = _mk_player(sess, "hero")
        char = _mk_character(sess, name="C")
        _populate_char_winrate(sess, char, hero, wins=1, losses=1, abandoned=3)
        sess.commit()

        rows = player_leaderboard(sess, viewer=viewer, window="all")
        hero_row = next(r for r in rows if r.username == "hero")
        assert hero_row.total_matches == 5
        # Player wins are the character's losses from `_populate_char_winrate` above.
        assert hero_row.wins == 1
        assert hero_row.losses == 4  # 1 real loss + 3 abandoned


def test_hall_of_fame_top_by_wins():
    with SessionLocal() as sess:
        viewer = _mk_player(sess, "viewer")
        a = _mk_player(sess, "alice")
        b = _mk_player(sess, "bob")
        char = _mk_character(sess, name="C")
        _populate_char_winrate(sess, char, a, wins=1, losses=3)  # alice beat char 3 times
        _populate_char_winrate(sess, char, b, wins=5, losses=0)  # bob never beat char
        sess.commit()

        hof = hall_of_fame_for_character(sess, character_id=char.id, limit=10)
        # Player wins = character losses — bob has 0 wins, alice has 3.
        alice_row = next(r for r in hof if r.username == "alice")
        bob_row = next(r for r in hof if r.username == "bob")
        assert alice_row.wins == 3
        assert bob_row.wins == 0
        assert hof[0].username == "alice"
