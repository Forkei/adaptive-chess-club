from __future__ import annotations

import logging
from datetime import datetime

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status
from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from app.auth import require_player
from app.db import get_session
from app.memory.crud import counts_by_scope, counts_by_type, list_for_character
from app.models.character import (
    Character,
    CharacterState,
    ContentRating,
    Visibility,
    rating_allowed,
    rating_level,
)
from app.models.match import Player
from app.models.memory import MemoryScope, MemoryType
from app.schemas.character import (
    CharacterCreate,
    CharacterDetail,
    CharacterRead,
    CharacterSummary,
    CharacterUpdate,
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


def _allowed_rating_values(player_max: ContentRating) -> list[ContentRating]:
    max_idx = rating_level(player_max)
    return [r for r in (ContentRating.FAMILY, ContentRating.MATURE, ContentRating.UNRESTRICTED)
            if rating_level(r) <= max_idx]


def _visible_to(character: Character, player: Player) -> bool:
    if character.deleted_at is not None:
        return False
    if character.visibility == Visibility.PRIVATE and character.owner_id != player.id:
        return False
    if not rating_allowed(character.content_rating, player.max_content_rating):
        return False
    return True


@router.get("", response_model=list[CharacterSummary])
def list_characters(
    player: Player = Depends(require_player),
    session: Session = Depends(get_session),
) -> list[CharacterSummary]:
    allowed = _allowed_rating_values(player.max_content_rating)
    rows = session.execute(
        select(Character)
        .where(Character.deleted_at.is_(None))
        .where(Character.content_rating.in_(allowed))
        .where(
            or_(
                Character.visibility == Visibility.PUBLIC,
                Character.owner_id == player.id,
            )
        )
        .order_by(Character.created_at.desc())
    ).scalars()
    return [CharacterSummary.model_validate(c) for c in rows]


@router.get("/{character_id}", response_model=CharacterDetail)
def get_character(
    character_id: str,
    player: Player = Depends(require_player),
    session: Session = Depends(get_session),
) -> CharacterDetail:
    character = session.get(Character, character_id)
    if character is None or character.deleted_at is not None:
        raise HTTPException(status_code=404, detail="Character not found")
    if not _visible_to(character, player):
        # Private-not-owner and rating-filtered both surface as 404 — we
        # don't leak existence details via auth errors.
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
    player: Player = Depends(require_player),
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
        visibility=payload.visibility,
        content_rating=payload.content_rating,
        owner_id=player.id,
        state=CharacterState.GENERATING_MEMORIES,
        memory_generation_started_at=datetime.utcnow(),
        is_preset=False,
    )
    session.add(character)
    session.commit()
    session.refresh(character)

    background.add_task(_run_generation_bg, character.id)
    return CharacterRead.model_validate(character)


@router.patch("/{character_id}", response_model=CharacterRead)
def update_character(
    character_id: str,
    payload: CharacterUpdate,
    player: Player = Depends(require_player),
    session: Session = Depends(get_session),
) -> CharacterRead:
    character = session.get(Character, character_id)
    if character is None or character.deleted_at is not None:
        raise HTTPException(status_code=404, detail="Character not found")
    if character.is_preset:
        raise HTTPException(status_code=403, detail="Presets cannot be edited. Clone first.")
    if character.owner_id != player.id:
        raise HTTPException(status_code=403, detail="Not your character.")

    data = payload.model_dump(exclude_unset=True, exclude_none=True)
    for field, value in data.items():
        setattr(character, field, value)
    character.updated_at = datetime.utcnow()
    session.commit()
    session.refresh(character)
    return CharacterRead.model_validate(character)


@router.post("/{character_id}/regenerate_memories", status_code=status.HTTP_202_ACCEPTED)
def regenerate_memories(
    character_id: str,
    background: BackgroundTasks,
    player: Player = Depends(require_player),
    session: Session = Depends(get_session),
) -> dict:
    character = session.get(Character, character_id)
    if character is None or character.deleted_at is not None:
        raise HTTPException(status_code=404, detail="Character not found")
    if character.is_preset:
        raise HTTPException(status_code=403, detail="Presets cannot be regenerated.")
    if character.owner_id != player.id:
        raise HTTPException(status_code=403, detail="Not your character.")

    # Drop existing non-match memories — backstory-derived lore. Keep
    # MATCH_RECAP / OPPONENT_SPECIFIC rows (real play history).
    from app.models.memory import Memory, MemoryScope as MS

    session.query(Memory).filter(
        Memory.character_id == character_id,
        Memory.scope.in_([MS.CHARACTER_LORE, MS.CROSS_PLAYER]),
    ).delete(synchronize_session=False)
    character.state = CharacterState.GENERATING_MEMORIES
    character.memory_generation_started_at = datetime.utcnow()
    character.memory_generation_error = None
    session.commit()

    background.add_task(_run_generation_bg, character_id)
    return {"status": "accepted", "character_id": character_id}


@router.post("/{character_id}/clone", response_model=CharacterRead, status_code=status.HTTP_202_ACCEPTED)
def clone_character(
    character_id: str,
    background: BackgroundTasks,
    player: Player = Depends(require_player),
    session: Session = Depends(get_session),
) -> CharacterRead:
    source = session.get(Character, character_id)
    if source is None or source.deleted_at is not None:
        raise HTTPException(status_code=404, detail="Character not found")
    if not _visible_to(source, player):
        raise HTTPException(status_code=404, detail="Character not found")

    clone = Character(
        name=f"{source.name} (clone)",
        short_description=source.short_description,
        backstory=source.backstory,
        avatar_emoji=source.avatar_emoji,
        aggression=source.aggression,
        risk_tolerance=source.risk_tolerance,
        patience=source.patience,
        trash_talk=source.trash_talk,
        target_elo=source.target_elo,
        current_elo=source.target_elo,  # fresh Elo state
        floor_elo=source.target_elo,
        max_elo=source.max_elo,
        adaptive=source.adaptive,
        opening_preferences=list(source.opening_preferences or []),
        voice_descriptor=source.voice_descriptor,
        quirks=source.quirks,
        visibility=Visibility.PUBLIC,
        content_rating=source.content_rating,
        owner_id=player.id,
        is_preset=False,
        preset_key=None,
        state=CharacterState.GENERATING_MEMORIES,
        memory_generation_started_at=datetime.utcnow(),
    )
    session.add(clone)
    session.commit()
    session.refresh(clone)

    # Fresh memory generation — independent of the source (even if the
    # source is still generating; see phase_3a_decisions.md).
    background.add_task(_run_generation_bg, clone.id)
    return CharacterRead.model_validate(clone)


@router.get("/{character_id}/memories", response_model=dict)
def list_memories(
    character_id: str,
    scope: MemoryScope | None = None,
    type: MemoryType | None = None,
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    player: Player = Depends(require_player),
    session: Session = Depends(get_session),
) -> dict:
    character = session.get(Character, character_id)
    if character is None or character.deleted_at is not None:
        raise HTTPException(status_code=404, detail="Character not found")
    if not _visible_to(character, player):
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
def delete_character(
    character_id: str,
    player: Player = Depends(require_player),
    session: Session = Depends(get_session),
) -> None:
    character = session.get(Character, character_id)
    if character is None or character.deleted_at is not None:
        raise HTTPException(status_code=404, detail="Character not found")
    if character.is_preset:
        raise HTTPException(status_code=403, detail="Preset characters cannot be deleted")
    if character.owner_id != player.id:
        raise HTTPException(status_code=403, detail="Not your character.")
    character.deleted_at = datetime.utcnow()
    session.commit()
