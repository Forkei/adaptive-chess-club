from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models.memory import Memory, MemoryScope, MemoryType
from app.schemas.memory import MemoryCreate


def bulk_create(
    session: Session, *, character_id: str, items: Iterable[MemoryCreate]
) -> list[Memory]:
    rows = [
        Memory(
            character_id=character_id,
            player_id=item.player_id,
            match_id=item.match_id,
            scope=item.scope,
            type=item.type,
            emotional_valence=item.emotional_valence,
            triggers=list(item.triggers),
            narrative_text=item.narrative_text,
            relevance_tags=list(item.relevance_tags),
        )
        for item in items
    ]
    session.add_all(rows)
    session.flush()
    return rows


def list_for_character(
    session: Session,
    *,
    character_id: str,
    scope: MemoryScope | None = None,
    type_: MemoryType | None = None,
    offset: int = 0,
    limit: int = 50,
) -> tuple[list[Memory], int]:
    stmt = select(Memory).where(Memory.character_id == character_id)
    count_stmt = select(func.count()).select_from(Memory).where(Memory.character_id == character_id)
    if scope is not None:
        stmt = stmt.where(Memory.scope == scope)
        count_stmt = count_stmt.where(Memory.scope == scope)
    if type_ is not None:
        stmt = stmt.where(Memory.type == type_)
        count_stmt = count_stmt.where(Memory.type == type_)
    total = session.execute(count_stmt).scalar_one()
    rows = list(
        session.execute(
            stmt.order_by(Memory.created_at.desc()).offset(offset).limit(limit)
        ).scalars()
    )
    return rows, total


def counts_by_scope(session: Session, *, character_id: str) -> Counter[str]:
    rows = session.execute(
        select(Memory.scope, func.count()).where(Memory.character_id == character_id).group_by(Memory.scope)
    ).all()
    out: Counter[str] = Counter()
    for scope, n in rows:
        key = scope.value if isinstance(scope, MemoryScope) else str(scope)
        out[key] = int(n)
    return out


def counts_by_type(session: Session, *, character_id: str) -> Counter[str]:
    rows = session.execute(
        select(Memory.type, func.count()).where(Memory.character_id == character_id).group_by(Memory.type)
    ).all()
    out: Counter[str] = Counter()
    for typ, n in rows:
        key = typ.value if isinstance(typ, MemoryType) else str(typ)
        out[key] = int(n)
    return out
