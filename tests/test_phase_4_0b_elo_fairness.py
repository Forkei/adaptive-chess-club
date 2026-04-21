"""Phase 4.0b: opponent-quality weighting + easy-win soft cap + private match
Elo exclusion in `apply_to_both`.
"""

from __future__ import annotations

from datetime import datetime

from app.db import SessionLocal
from app.models.character import Character, CharacterState, ContentRating, Visibility
from app.models.match import Color, Match, MatchResult, MatchStatus, Player
from app.post_match.elo_apply import (
    EASY_WIN_ELO_GAP,
    apply_to_both,
    easy_win_scale,
    opp_acpl_quality_factor,
)


# --- unit tests on the pure functions -------------------------------------


def test_opp_acpl_quality_factor_endpoints():
    # Clean opponent (low ACPL) → no penalty.
    assert opp_acpl_quality_factor(10) == 1.0
    # Catastrophic ACPL → floor.
    assert opp_acpl_quality_factor(1000) == 0.1
    # Interpolation between anchors is monotonic.
    assert opp_acpl_quality_factor(40) > opp_acpl_quality_factor(120) > opp_acpl_quality_factor(300)


def test_easy_win_scale_progression():
    assert easy_win_scale(0) == 1.0
    assert easy_win_scale(2) == 1.0
    assert easy_win_scale(3) == 0.5
    assert easy_win_scale(4) < easy_win_scale(3)
    # Never below floor.
    assert easy_win_scale(50) >= 0.1


# --- integration helpers --------------------------------------------------


def _seed_character(
    session,
    *,
    current_elo: int = 1500,
    floor_elo: int = 1400,
    max_elo: int = 1800,
    adaptive: bool = True,
) -> str:
    c = Character(
        name="TestChar",
        short_description="x",
        backstory="A rich backstory for testing.",
        voice_descriptor="voice",
        target_elo=current_elo,
        current_elo=current_elo,
        floor_elo=floor_elo,
        max_elo=max_elo,
        adaptive=adaptive,
        is_preset=False,
        owner_id=None,
        state=CharacterState.READY,
        visibility=Visibility.PUBLIC,
        content_rating=ContentRating.FAMILY,
    )
    session.add(c)
    session.commit()
    session.refresh(c)
    return c.id


def _seed_player(session, *, elo: int = 1200) -> str:
    p = Player(
        username=f"p_{elo}_{datetime.utcnow().timestamp()}",
        display_name="Player",
        elo=elo,
    )
    session.add(p)
    session.commit()
    session.refresh(p)
    return p.id


def _seed_match(
    session,
    *,
    character_id: str,
    player_id: str,
    result: MatchResult,
    player_color: Color = Color.WHITE,
    move_count: int = 30,
    char_elo_start: int = 1500,
    player_elo_start: int = 1200,
    is_private: bool = False,
    status: MatchStatus = MatchStatus.COMPLETED,
    ended_at: datetime | None = None,
) -> Match:
    m = Match(
        character_id=character_id,
        player_id=player_id,
        status=status,
        result=result,
        player_color=player_color,
        initial_fen="startpos",
        current_fen="startpos",
        move_count=move_count,
        character_elo_at_start=char_elo_start,
        player_elo_at_start=player_elo_start,
        is_private=is_private,
        ended_at=ended_at or datetime.utcnow(),
    )
    session.add(m)
    session.commit()
    session.refresh(m)
    return m


def _moves(char_losses, opp_losses, *, char_color="black"):
    out = []
    mn = 1
    opp_color = "white" if char_color == "black" else "black"
    for loss in char_losses:
        out.append({"move_number": mn, "side": char_color, "eval_loss_cp": loss})
        mn += 1
    for loss in opp_losses:
        out.append({"move_number": mn, "side": opp_color, "eval_loss_cp": loss})
        mn += 1
    return out


# --- private match exclusion ----------------------------------------------


def test_private_match_produces_zero_delta_for_both_sides():
    with SessionLocal() as s:
        cid = _seed_character(s, current_elo=1500)
        pid = _seed_player(s, elo=1200)
        m = _seed_match(
            s,
            character_id=cid,
            player_id=pid,
            result=MatchResult.BLACK_WIN,  # char is black, black wins → char wins
            char_elo_start=1500,
            player_elo_start=1200,
            is_private=True,
        )
        _, both = apply_to_both(s, match=m, analysis_moves=_moves([0] * 15, [0] * 15))
        assert both.character.current_elo_change == 0
        assert both.player.current_elo_change == 0


def test_public_match_same_conditions_does_apply_delta():
    """Control: identical setup but is_private=False → non-zero change."""
    with SessionLocal() as s:
        cid = _seed_character(s, current_elo=1500)
        pid = _seed_player(s, elo=1200)
        m = _seed_match(
            s,
            character_id=cid,
            player_id=pid,
            result=MatchResult.BLACK_WIN,
            char_elo_start=1500,
            player_elo_start=1200,
            is_private=False,
        )
        _, both = apply_to_both(s, match=m, analysis_moves=_moves([0] * 15, [0] * 15))
        # Char was the favourite but still gets a little gain from winning.
        assert both.character.current_elo_change != 0


# --- opponent-quality weighting -------------------------------------------


def test_blunder_fest_reduces_winner_gain():
    """Same Elo, character wins. Compare gain when opponent ACPL low vs high."""
    with SessionLocal() as s:
        # Clean-opponent variant.
        cid_clean = _seed_character(s, current_elo=1500)
        pid_clean = _seed_player(s, elo=1500)
        m_clean = _seed_match(
            s,
            character_id=cid_clean,
            player_id=pid_clean,
            result=MatchResult.BLACK_WIN,
            char_elo_start=1500,
            player_elo_start=1500,
        )
        _, clean = apply_to_both(
            s, match=m_clean, analysis_moves=_moves([20] * 15, [20] * 15)
        )

    with SessionLocal() as s:
        # Blunder-fest variant.
        cid_bad = _seed_character(s, current_elo=1500)
        pid_bad = _seed_player(s, elo=1500)
        m_bad = _seed_match(
            s,
            character_id=cid_bad,
            player_id=pid_bad,
            result=MatchResult.BLACK_WIN,
            char_elo_start=1500,
            player_elo_start=1500,
        )
        # Opponent averaging ~300cp loss per move = massive blunder-fest.
        _, bad = apply_to_both(
            s, match=m_bad, analysis_moves=_moves([20] * 15, [300] * 15)
        )

    assert clean.character.current_elo_change > bad.character.current_elo_change


def test_losing_side_delta_is_unaffected_by_acpl_weighting():
    """Weighting only touches positive deltas; loser loses full."""
    with SessionLocal() as s:
        cid = _seed_character(s, current_elo=1500)
        pid = _seed_player(s, elo=1500)
        m = _seed_match(
            s,
            character_id=cid,
            player_id=pid,
            result=MatchResult.BLACK_WIN,
            char_elo_start=1500,
            player_elo_start=1500,
        )
        # Player was the loser (BLACK_WIN, player is WHITE).
        _, r = apply_to_both(
            s, match=m, analysis_moves=_moves([10] * 15, [300] * 15)
        )
        # Player loses; negative delta is NOT scaled down to something small.
        assert r.player.current_elo_change < 0


# --- easy-win soft cap ----------------------------------------------------


def test_easy_win_streak_reduces_later_gains():
    """Character stomps many weak opponents in a row. Later wins should gain less
    than the first.
    """
    with SessionLocal() as s:
        cid = _seed_character(s, current_elo=1800, floor_elo=1400, max_elo=1800)

    gains: list[int] = []
    for i in range(5):
        with SessionLocal() as s:
            # Each opponent is much weaker (gap > EASY_WIN_ELO_GAP).
            pid = _seed_player(s, elo=1800 - EASY_WIN_ELO_GAP - 100)
            # Win (character is black, BLACK_WIN).
            now = datetime(2026, 4, 1, 10, i)
            m = _seed_match(
                s,
                character_id=cid,
                player_id=pid,
                result=MatchResult.BLACK_WIN,
                char_elo_start=1800,
                player_elo_start=1800 - EASY_WIN_ELO_GAP - 100,
                ended_at=now,
            )
            _, both = apply_to_both(
                s, match=m, analysis_moves=_moves([20] * 15, [20] * 15)
            )
            gains.append(both.character.current_elo_change)
            # Reset char_current_elo so we can keep testing a "this is an easy
            # win vs a weak opp" scenario without the character climbing up.
            char = s.get(Character, cid)
            char.current_elo = 1800
            char.floor_elo = 1400
            s.commit()

    # First two free, third onward reduced.
    assert gains[0] >= gains[2]
    assert gains[2] >= gains[4]
    # Meaningful gap (the scale should reduce by enough to matter).
    assert gains[2] <= gains[0] * 0.75 + 1
