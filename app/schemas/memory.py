from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.models.memory import MemoryScope, MemoryType


class MemoryCreate(BaseModel):
    """Input schema for creating a memory. Used by the LLM generator too."""

    model_config = ConfigDict(use_enum_values=False)

    scope: MemoryScope
    type: MemoryType
    emotional_valence: float = Field(..., ge=-1.0, le=1.0)
    triggers: list[str] = Field(default_factory=list)
    narrative_text: str = Field(..., min_length=1)
    relevance_tags: list[str] = Field(default_factory=list)
    player_id: str | None = None
    match_id: str | None = None

    @field_validator("triggers", "relevance_tags")
    @classmethod
    def _strip_and_dedup(cls, v: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for item in v:
            s = item.strip()
            if s and s.lower() not in seen:
                seen.add(s.lower())
                out.append(s)
        return out


class MemoryRead(BaseModel):
    model_config = ConfigDict(from_attributes=True, use_enum_values=False)

    id: str
    character_id: str
    player_id: str | None
    match_id: str | None
    scope: MemoryScope
    type: MemoryType
    emotional_valence: float
    triggers: list[str]
    narrative_text: str
    relevance_tags: list[str]
    created_at: datetime
    last_surfaced_at: datetime | None
    surface_count: int
