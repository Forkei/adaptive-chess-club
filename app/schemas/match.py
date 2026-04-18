from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.models.match import Color, MatchResult, MatchStatus


class PlayerRead(BaseModel):
    model_config = ConfigDict(from_attributes=True, use_enum_values=False)

    id: str
    display_name: str
    created_at: datetime


class PlayerCreate(BaseModel):
    display_name: str = Field("Guest", min_length=1, max_length=80)


class ConsideredMove(BaseModel):
    uci: str
    san: str | None = None
    eval_cp: int | None = None
    probability: float | None = None


class MoveRead(BaseModel):
    model_config = ConfigDict(from_attributes=True, use_enum_values=False)

    id: str
    move_number: int
    side: Color
    uci: str
    san: str
    fen_after: str
    engine_name: str | None
    time_taken_ms: int | None
    eval_cp: int | None
    considered_moves: list[dict[str, Any]]
    thinking_depth: int | None
    agent_chat_after: str | None
    player_chat_before: str | None
    surfaced_memory_ids: list[str]
    mood_snapshot: dict[str, Any]
    created_at: datetime


class MatchCreate(BaseModel):
    character_id: str
    # "random" picks uniformly; otherwise explicit.
    player_color: Literal["white", "black", "random"] = "random"


class MatchRead(BaseModel):
    model_config = ConfigDict(from_attributes=True, use_enum_values=False)

    id: str
    character_id: str
    player_id: str
    status: MatchStatus
    result: MatchResult | None
    player_color: Color
    initial_fen: str
    current_fen: str
    move_count: int
    character_elo_at_start: int
    character_elo_at_end: int | None
    started_at: datetime
    ended_at: datetime | None


class MoveSubmit(BaseModel):
    uci: str = Field(..., min_length=4, max_length=6)


class MoveResponse(BaseModel):
    """Returned after POST /api/matches/{id}/move.

    The player's move is applied first, then (if the game isn't over)
    the engine plays too. `agent_move` is NULL when it's now the
    player's turn again (shouldn't happen) or the game ended on the
    player's move.
    """

    match: MatchRead
    player_move: MoveRead
    agent_move: MoveRead | None
    game_over: bool
    outcome: str | None  # "win" / "loss" / "draw" from the player's POV


class MoveList(BaseModel):
    total: int
    items: list[MoveRead]
