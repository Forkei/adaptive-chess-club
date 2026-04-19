"""Pydantic schemas for Phase 3c discovery + leaderboard + hall-of-fame responses."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class MatchSummaryOut(BaseModel):
    match_id: str
    character_id: str
    character_name: str
    character_avatar: str
    character_content_rating: str
    character_visibility: str
    player_id: str
    player_username: str
    status: str
    result: str | None
    move_count: int
    player_color: str
    started_at: datetime
    ended_at: datetime | None


class MatchSummaryPage(BaseModel):
    items: list[MatchSummaryOut] = Field(default_factory=list)


class CharacterLeaderboardEntry(BaseModel):
    rank: int
    character_id: str
    character_name: str
    character_avatar: str
    current_elo: int
    total_matches: int
    wins: int
    losses: int
    draws: int
    win_rate: float


class CharacterLeaderboardResponse(BaseModel):
    window: Literal["all", "30d", "7d"]
    rows: list[CharacterLeaderboardEntry] = Field(default_factory=list)


class PlayerLeaderboardEntry(BaseModel):
    rank: int
    player_id: str
    username: str
    display_name: str
    total_matches: int
    wins: int
    losses: int
    draws: int
    win_rate: float


class PlayerLeaderboardResponse(BaseModel):
    window: Literal["all", "30d", "7d"]
    rows: list[PlayerLeaderboardEntry] = Field(default_factory=list)


class HallOfFameEntry(BaseModel):
    rank: int
    player_id: str
    username: str
    display_name: str
    wins: int
    total_matches: int
    win_rate: float


class HallOfFameResponse(BaseModel):
    character_id: str
    rows: list[HallOfFameEntry] = Field(default_factory=list)
