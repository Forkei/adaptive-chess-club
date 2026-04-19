"""Socket.IO event payload schemas (Phase 3b).

Every client↔server event has a Pydantic model here so the contract is self-documenting.
Handlers validate incoming events with `MODEL.model_validate` and emit outgoing events with
`MODEL(...).model_dump(mode="json")`.

Naming: client→server events are the `C2S_*` constants; server→client events are `S2C_*`.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


# --- Namespace + rooms -----------------------------------------------------

NAMESPACE = "/play"


def match_room_name(match_id: str) -> str:
    """Deterministic Socket.IO room name per match."""
    return f"match:{match_id}"


# --- Client -> Server ------------------------------------------------------

C2S_MAKE_MOVE = "make_move"
C2S_PLAYER_CHAT = "player_chat"
C2S_RESIGN = "resign"
C2S_PING = "ping"
C2S_REQUEST_STATE = "request_state"  # optional — state is also emitted on connect

# Phase 3c: spectators have their own chat channel that never reaches the Subconscious.
C2S_SPECTATOR_CHAT = "spectator_chat"


class MakeMoveEvent(BaseModel):
    uci: str = Field(..., min_length=4, max_length=6)
    chat: str | None = Field(default=None, max_length=500)


class PlayerChatEvent(BaseModel):
    text: str = Field(..., min_length=1, max_length=500)


class SpectatorChatEvent(BaseModel):
    text: str = Field(..., min_length=1, max_length=500)


# --- Server -> Client ------------------------------------------------------

S2C_MATCH_STATE = "match_state"
S2C_PLAYER_MOVE_APPLIED = "player_move_applied"
S2C_AGENT_THINKING = "agent_thinking"
S2C_MEMORY_SURFACED = "memory_surfaced"
S2C_AGENT_MOVE = "agent_move"
S2C_AGENT_CHAT = "agent_chat"
S2C_MOOD_UPDATE = "mood_update"
S2C_MATCH_ENDED = "match_ended"
S2C_POST_MATCH_STATUS = "post_match_status"
S2C_POST_MATCH_COMPLETE = "post_match_complete"
S2C_PONG = "pong"
S2C_MATCH_RESUMED = "match_resumed"
S2C_MATCH_PAUSED = "match_paused"
S2C_PLAYER_CHAT_ECHOED = "player_chat_echoed"
S2C_PLAYER_CHAT_RATE_LIMITED = "player_chat_rate_limited"
S2C_ERROR = "error"

# Phase 3c: spectator-oriented events.
S2C_SPECTATOR_CHAT_BROADCAST = "spectator_chat_broadcast"
S2C_SPECTATOR_CHAT_ECHOED = "spectator_chat_echoed"
S2C_SPECTATOR_CHAT_REJECTED = "spectator_chat_rejected"
S2C_PLAYER_CHAT_BROADCAST = "player_chat_broadcast"
S2C_SPECTATOR_JOINED = "spectator_joined"
S2C_SPECTATOR_LEFT = "spectator_left"
S2C_SPECTATOR_COUNT = "spectator_count"


class _FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=False)


class MoveSnapshot(_FrozenModel):
    move_number: int
    side: Literal["white", "black"]
    uci: str
    san: str
    fen_after: str
    engine_name: str | None = None
    time_taken_ms: int | None = None
    eval_cp: int | None = None
    player_chat_before: str | None = None
    agent_chat_after: str | None = None


class MatchStatePayload(_FrozenModel):
    """Full current state, emitted on connect and on request_state."""

    match_id: str
    status: Literal["in_progress", "completed", "abandoned"]
    result: Literal["white_win", "black_win", "draw", "abandoned"] | None
    player_color: Literal["white", "black"]
    current_fen: str
    move_count: int
    moves: list[MoveSnapshot]
    mood: dict[str, float] = Field(default_factory=dict)
    last_agent_chat: str | None = None
    last_emotion: str | None = None
    disconnect_cooldown_seconds: int | None = None
    disconnect_deadline: datetime | None = None


class PlayerMoveAppliedPayload(_FrozenModel):
    move_number: int
    uci: str
    san: str
    fen_after: str
    player_chat_before: str | None = None


class AgentThinkingPayload(_FrozenModel):
    eta_seconds: float = Field(
        ...,
        description="Approximate total turn latency (engine + Soul overhead). UI should not treat as a hard countdown.",
    )


class MemorySurfacedItem(_FrozenModel):
    memory_id: str
    retrieval_reason: str
    narrative_snippet: str
    from_cache: bool = False


class MemorySurfacedPayload(_FrozenModel):
    items: list[MemorySurfacedItem]


class AgentMovePayload(_FrozenModel):
    move_number: int
    uci: str
    san: str
    fen_after: str
    time_taken_ms: int | None = None
    engine_name: str | None = None
    eval_cp: int | None = None


class AgentChatPayload(_FrozenModel):
    speak: str
    emotion: str = "neutral"
    emotion_intensity: float = 0.0
    referenced_memory_ids: list[str] = Field(default_factory=list)


class MoodUpdatePayload(_FrozenModel):
    mood: dict[str, float]


MatchEndReason = Literal[
    "checkmate", "stalemate", "resign", "disconnect_timeout", "agreement", "draw_rule"
]


class MatchEndedPayload(_FrozenModel):
    match_id: str
    result: Literal["white_win", "black_win", "draw", "abandoned"]
    reason: MatchEndReason
    player_outcome: Literal["win", "loss", "draw", "resigned"] | None = None


class PostMatchStatusPayload(_FrozenModel):
    match_id: str
    status: Literal["pending", "running", "completed", "failed"]
    steps_completed: list[str] = Field(default_factory=list)
    current_step: str | None = None
    error: str | None = None


class PostMatchCompletePayload(_FrozenModel):
    match_id: str
    summary_url: str


class PongPayload(_FrozenModel):
    ts: datetime


class MatchResumedPayload(_FrozenModel):
    match_id: str


class MatchPausedPayload(_FrozenModel):
    match_id: str
    deadline: datetime
    cooldown_seconds: int


class PlayerChatEchoedPayload(_FrozenModel):
    text: str
    received_at: datetime


class PlayerChatRateLimitedPayload(_FrozenModel):
    retry_after_ms: int
    reason: str = "too_many_chat_messages"


class ErrorPayload(_FrozenModel):
    code: str
    message: str


# --- Phase 3c spectator payloads ------------------------------------------


class SpectatorChatBroadcastPayload(_FrozenModel):
    username: str
    text: str
    timestamp: datetime


class PlayerChatBroadcastPayload(_FrozenModel):
    username: str
    text: str
    timestamp: datetime


class SpectatorJoinedPayload(_FrozenModel):
    username: str


class SpectatorLeftPayload(_FrozenModel):
    username: str


class SpectatorCountPayload(_FrozenModel):
    count: int


class SpectatorChatRejectedPayload(_FrozenModel):
    reason: str = "participants_cannot_use_spectator_chat"


__all__ = [
    "NAMESPACE",
    "match_room_name",
    # client -> server
    "C2S_MAKE_MOVE",
    "C2S_PLAYER_CHAT",
    "C2S_RESIGN",
    "C2S_PING",
    "C2S_REQUEST_STATE",
    "C2S_SPECTATOR_CHAT",
    "MakeMoveEvent",
    "PlayerChatEvent",
    "SpectatorChatEvent",
    # server -> client names
    "S2C_MATCH_STATE",
    "S2C_PLAYER_MOVE_APPLIED",
    "S2C_AGENT_THINKING",
    "S2C_MEMORY_SURFACED",
    "S2C_AGENT_MOVE",
    "S2C_AGENT_CHAT",
    "S2C_MOOD_UPDATE",
    "S2C_MATCH_ENDED",
    "S2C_POST_MATCH_STATUS",
    "S2C_POST_MATCH_COMPLETE",
    "S2C_PONG",
    "S2C_MATCH_RESUMED",
    "S2C_MATCH_PAUSED",
    "S2C_PLAYER_CHAT_ECHOED",
    "S2C_PLAYER_CHAT_RATE_LIMITED",
    "S2C_ERROR",
    "S2C_SPECTATOR_CHAT_BROADCAST",
    "S2C_SPECTATOR_CHAT_ECHOED",
    "S2C_SPECTATOR_CHAT_REJECTED",
    "S2C_PLAYER_CHAT_BROADCAST",
    "S2C_SPECTATOR_JOINED",
    "S2C_SPECTATOR_LEFT",
    "S2C_SPECTATOR_COUNT",
    # server -> client payloads
    "MoveSnapshot",
    "MatchStatePayload",
    "PlayerMoveAppliedPayload",
    "AgentThinkingPayload",
    "MemorySurfacedItem",
    "MemorySurfacedPayload",
    "AgentMovePayload",
    "AgentChatPayload",
    "MoodUpdatePayload",
    "MatchEndedPayload",
    "MatchEndReason",
    "PostMatchStatusPayload",
    "PostMatchCompletePayload",
    "PongPayload",
    "MatchResumedPayload",
    "MatchPausedPayload",
    "PlayerChatEchoedPayload",
    "PlayerChatRateLimitedPayload",
    "ErrorPayload",
    "SpectatorChatBroadcastPayload",
    "PlayerChatBroadcastPayload",
    "SpectatorJoinedPayload",
    "SpectatorLeftPayload",
    "SpectatorCountPayload",
    "SpectatorChatRejectedPayload",
]
