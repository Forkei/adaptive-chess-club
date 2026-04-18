from __future__ import annotations

import logging
from datetime import datetime

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import get_session
from app.memory.crud import counts_by_scope, counts_by_type, list_for_character
from app.models.character import Character, CharacterState
from app.models.memory import MemoryScope, MemoryType
from app.schemas.character import (
    CharacterCreate,
    CharacterDetail,
    CharacterRead,
    CharacterSummary,
    MemoryCountsByScope,
    MemoryCountsByType,
)
from app.schemas.memory import MemoryRead

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/characters", tags=["characters"])


def _run_generation_bg(character_id: str) -> None:
    from app.characters.memory_generator import generate_and_store

    try:
        generate_and_store(character_id)
    except Exception:
        logger.exception("Memory generation failed for %s", character_id)


@router.get("", response_model=list[CharacterSummary])
def list_characters(session: Session = Depends(get_session)) -> list[CharacterSummary]:
    rows = session.execute(
        select(Character).where(Character.deleted_at.is_(None)).order_by(Character.created_at.desc())
    ).scalars()
    return [CharacterSummary.model_validate(c) for c in rows]


@router.get("/{character_id}", response_model=CharacterDetail)
def get_character(character_id: str, session: Session = Depends(get_session)) -> CharacterDetail:
    character = session.get(Character, character_id)
    if character is None or character.deleted_at is not None:
        raise HTTPException(status_code=404, detail="Character not found")

    scope_counts = counts_by_scope(session, character_id=character_id)
    type_counts = counts_by_type(session, character_id=character_id)

    detail = CharacterDetail.model_validate(character).model_copy(
        update={
            "memory_count": sum(scope_counts.values()),
            "memory_counts_by_scope": MemoryCountsByScope(**scope_counts),
            "memory_counts_by_type": MemoryCountsByType(**type_counts),
        }
    )
    return detail


@router.post("", response_model=CharacterRead, status_code=status.HTTP_202_ACCEPTED)
def create_character(
    payload: CharacterCreate,
    background: BackgroundTasks,
    session: Session = Depends(get_session),
) -> CharacterRead:
    character = Character(
        name=payload.name,
        short_description=payload.short_description,
        backstory=payload.backstory,
        avatar_emoji=payload.avatar_emoji,
        aggression=payload.aggression,
        risk_tolerance=payload.risk_tolerance,
        patience=payload.patience,
        trash_talk=payload.trash_talk,
        target_elo=payload.target_elo,
        current_elo=payload.target_elo,
        floor_elo=payload.target_elo,
        max_elo=payload.max_elo if payload.max_elo is not None else payload.target_elo + 400,
        adaptive=payload.adaptive,
        opening_preferences=list(payload.opening_preferences),
        voice_descriptor=payload.voice_descriptor,
        quirks=payload.quirks,
        state=CharacterState.GENERATING_MEMORIES,
        memory_generation_started_at=datetime.utcnow(),
        is_preset=False,
    )
    session.add(character)
    session.commit()
    session.refresh(character)

    background.add_task(_run_generation_bg, character.id)
    return CharacterRead.model_validate(character)


@router.get("/{character_id}/memories", response_model=dict)
def list_memories(
    character_id: str,
    scope: MemoryScope | None = None,
    type: MemoryType | None = None,
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    session: Session = Depends(get_session),
) -> dict:
    character = session.get(Character, character_id)
    if character is None or character.deleted_at is not None:
        raise HTTPException(status_code=404, detail="Character not found")

    rows, total = list_for_character(
        session,
        character_id=character_id,
        scope=scope,
        type_=type,
        offset=offset,
        limit=limit,
    )
    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "items": [MemoryRead.model_validate(r).model_dump(mode="json") for r in rows],
    }


@router.delete("/{character_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_character(character_id: str, session: Session = Depends(get_session)) -> None:
    character = session.get(Character, character_id)
    if character is None or character.deleted_at is not None:
        raise HTTPException(status_code=404, detail="Character not found")
    if character.is_preset:
        raise HTTPException(status_code=403, detail="Preset characters cannot be deleted")
    character.deleted_at = datetime.utcnow()
    session.commit()
