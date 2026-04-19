from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.models.character import ContentRating
from app.models.match import Color, MatchResult, MatchStatus


class PlayerRead(BaseModel):
    model_config = ConfigDict(from_attributes=True, use_enum_values=False)

    id: str
    username: str
    display_name: str
    max_content_rating: ContentRating = ContentRating.FAMILY
    created_at: datetime


class PlayerCreate(BaseModel):
    display_name: str = Field("Guest", min_length=1, max_length=80)


class PlayerSettingsUpdate(BaseModel):
    """Fields editable on the /settings page."""

    display_name: str | None = Field(None, min_length=1, max_length=80)
    max_content_rating: ContentRating | None = None


class ConsideredMove(BaseModel):
    uci: str
    san: str | None = None
    eval_cp: int | None = None
    probability: float | None = None


class SurfacedMemorySnippet(BaseModel):
    """Compact view the UI can render as the 'memory ribbon' glimpse."""

    memory_id: str
    narrative_text: str
    retrieval_reason: str
    from_cache: bool = False


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
    # Phase 2b: optional chat message the player attaches to this move.
    # Stored on the resulting Move.player_chat_before and visible to the
    # Subconscious next turn.
    chat: str | None = Field(default=None, max_length=500)


class AgentTurnInfo(BaseModel):
    """Phase 2b addition — the Soul's output rendered for the client.

    Attached alongside `agent_move` so the UI can show the emotion indicator,
    chat message, and memory-ribbon glimpses without scraping the move row.
    """

    speak: str | None = None
    emotion: str = "neutral"
    emotion_intensity: float = 0.0
    surfaced_memories: list[SurfacedMemorySnippet] = Field(default_factory=list)


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
    agent_turn: AgentTurnInfo | None = None  # only present when the agent moved
    game_over: bool
    outcome: str | None  # "win" / "loss" / "draw" from the player's POV


class MoveList(BaseModel):
    total: int
    items: list[MoveRead]


# --- Phase 2b: post-match status polling ----------------------------------


class GeneratedMemorySnippet(BaseModel):
    """Compact view of a memory the post-match LLM just produced — shown
    to the player on the match summary page."""

    memory_id: str
    narrative_text: str
    triggers: list[str]
    emotional_valence: float


class PostMatchStatus(BaseModel):
    """Response for GET /api/matches/{id}/post_match_status."""

    match_id: str
    status: Literal["none", "pending", "running", "completed", "failed"]
    steps_completed: list[str] = Field(default_factory=list)
    error: str | None = None
    # Populated when complete.
    elo_delta_applied: int | None = None
    floor_raised: bool = False
    critical_moments: list[dict[str, Any]] = Field(default_factory=list)
    features: dict[str, Any] = Field(default_factory=dict)
    generated_memories: list[GeneratedMemorySnippet] = Field(default_factory=list)
    narrative_summary: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
