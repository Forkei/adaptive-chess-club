from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.models.character import CharacterState


class CharacterCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)
    short_description: str = Field("", max_length=280)
    backstory: str = Field("", max_length=8000)
    avatar_emoji: str = Field("♟️", max_length=8)

    aggression: int = Field(5, ge=1, le=10)
    risk_tolerance: int = Field(5, ge=1, le=10)
    patience: int = Field(5, ge=1, le=10)
    trash_talk: int = Field(5, ge=1, le=10)

    target_elo: int = Field(1400, ge=600, le=2600)
    adaptive: bool = False
    # `max_elo` optional on create: defaults to target_elo + 400 if not given.
    max_elo: int | None = Field(None, ge=600, le=3000)

    opening_preferences: list[str] = Field(default_factory=list)
    voice_descriptor: str = Field("", max_length=280)
    quirks: str = Field("", max_length=4000)

    @field_validator("opening_preferences")
    @classmethod
    def _clean_openings(cls, v: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for item in v:
            s = item.strip()
            if s and s not in seen:
                seen.add(s)
                out.append(s)
        return out


class MemoryCountsByScope(BaseModel):
    character_lore: int = 0
    opponent_specific: int = 0
    cross_player: int = 0
    match_recap: int = 0


class MemoryCountsByType(BaseModel):
    formative: int = 0
    rivalry: int = 0
    travel: int = 0
    triumph: int = 0
    defeat: int = 0
    habit: int = 0
    opinion: int = 0
    observation: int = 0


class CharacterSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True, use_enum_values=False)

    id: str
    name: str
    short_description: str
    avatar_emoji: str
    state: CharacterState
    is_preset: bool
    target_elo: int
    current_elo: int
    floor_elo: int
    max_elo: int
    adaptive: bool
    created_at: datetime


class CharacterRead(BaseModel):
    model_config = ConfigDict(from_attributes=True, use_enum_values=False)

    id: str
    name: str
    short_description: str
    backstory: str
    avatar_emoji: str

    aggression: int
    risk_tolerance: int
    patience: int
    trash_talk: int

    target_elo: int
    current_elo: int
    floor_elo: int
    max_elo: int
    adaptive: bool

    opening_preferences: list[str]
    voice_descriptor: str
    quirks: str

    state: CharacterState
    memory_generation_started_at: datetime | None
    memory_generation_error: str | None
    is_preset: bool

    created_at: datetime
    updated_at: datetime


class CharacterDetail(CharacterRead):
    memory_count: int = 0
    memory_counts_by_scope: MemoryCountsByScope = Field(default_factory=MemoryCountsByScope)
    memory_counts_by_type: MemoryCountsByType = Field(default_factory=MemoryCountsByType)
