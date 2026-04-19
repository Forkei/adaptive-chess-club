"""Discovery + leaderboard + hall-of-fame JSON endpoints (Phase 3c)."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.auth import require_player
from app.db import get_session
from app.discovery import (
    LeaderboardWindow,
    character_leaderboard,
    hall_of_fame_for_character,
    list_live_matches,
    list_recent_matches,
    player_leaderboard,
)
from app.models.character import Character, ContentRating, Visibility, rating_allowed
from app.models.match import Player
from app.schemas.discovery import (
    CharacterLeaderboardEntry,
    CharacterLeaderboardResponse,
    HallOfFameEntry,
    HallOfFameResponse,
    MatchSummaryOut,
    MatchSummaryPage,
    PlayerLeaderboardEntry,
    PlayerLeaderboardResponse,
)

router = APIRouter(prefix="/api", tags=["discovery"])


def _as_summary(s) -> MatchSummaryOut:
    return MatchSummaryOut(
        match_id=s.match_id,
        character_id=s.character_id,
        character_name=s.character_name,
        character_avatar=s.character_avatar,
        character_content_rating=s.character_content_rating,
        character_visibility=s.character_visibility,
        player_id=s.player_id,
        player_username=s.player_username,
        status=s.status,
        result=s.result,
        move_count=s.move_count,
        player_color=s.player_color,
        started_at=s.started_at,
        ended_at=s.ended_at,
    )


@router.get("/matches/live", response_model=MatchSummaryPage)
def get_live_matches(
    limit: int = Query(20, ge=1, le=50),
    offset: int = Query(0, ge=0),
    player: Player = Depends(require_player),
    session: Session = Depends(get_session),
) -> MatchSummaryPage:
    rows = list_live_matches(session, viewer=player, limit=limit, offset=offset)
    return MatchSummaryPage(items=[_as_summary(r) for r in rows])


@router.get("/matches/recent", response_model=MatchSummaryPage)
def get_recent_matches(
    limit: int = Query(20, ge=1, le=50),
    offset: int = Query(0, ge=0),
    player: Player = Depends(require_player),
    session: Session = Depends(get_session),
) -> MatchSummaryPage:
    rows = list_recent_matches(session, viewer=player, limit=limit, offset=offset)
    return MatchSummaryPage(items=[_as_summary(r) for r in rows])


@router.get("/leaderboard/characters", response_model=CharacterLeaderboardResponse)
def get_character_leaderboard(
    window: Literal["all", "30d", "7d"] = Query("all"),
    player: Player = Depends(require_player),
    session: Session = Depends(get_session),
) -> CharacterLeaderboardResponse:
    rows = character_leaderboard(session, viewer=player, window=window)
    return CharacterLeaderboardResponse(
        window=window,
        rows=[
            CharacterLeaderboardEntry(
                rank=r.rank,
                character_id=r.character_id,
                character_name=r.character_name,
                character_avatar=r.character_avatar,
                current_elo=r.current_elo,
                total_matches=r.total_matches,
                wins=r.wins,
                losses=r.losses,
                draws=r.draws,
                win_rate=round(r.win_rate, 4),
            )
            for r in rows
        ],
    )


@router.get("/leaderboard/players", response_model=PlayerLeaderboardResponse)
def get_player_leaderboard(
    window: Literal["all", "30d", "7d"] = Query("all"),
    player: Player = Depends(require_player),
    session: Session = Depends(get_session),
) -> PlayerLeaderboardResponse:
    rows = player_leaderboard(session, viewer=player, window=window)
    return PlayerLeaderboardResponse(
        window=window,
        rows=[
            PlayerLeaderboardEntry(
                rank=r.rank,
                player_id=r.player_id,
                username=r.username,
                display_name=r.display_name,
                total_matches=r.total_matches,
                wins=r.wins,
                losses=r.losses,
                draws=r.draws,
                win_rate=round(r.win_rate, 4),
            )
            for r in rows
        ],
    )


@router.get("/characters/{character_id}/hall_of_fame", response_model=HallOfFameResponse)
def get_hall_of_fame(
    character_id: str,
    player: Player = Depends(require_player),
    session: Session = Depends(get_session),
) -> HallOfFameResponse:
    character = session.get(Character, character_id)
    if character is None or character.deleted_at is not None:
        raise HTTPException(status_code=404, detail="Character not found")
    if character.visibility == Visibility.PRIVATE and character.owner_id != player.id:
        raise HTTPException(status_code=404, detail="Character not found")
    if not rating_allowed(character.content_rating, player.max_content_rating):
        raise HTTPException(status_code=403, detail="Content rating exceeds your preference.")

    rows = hall_of_fame_for_character(session, character_id=character_id)
    return HallOfFameResponse(
        character_id=character_id,
        rows=[
            HallOfFameEntry(
                rank=r.rank,
                player_id=r.player_id,
                username=r.username,
                display_name=r.display_name,
                wins=r.wins,
                total_matches=r.total_matches,
                win_rate=round(r.win_rate, 4),
            )
            for r in rows
        ],
    )
