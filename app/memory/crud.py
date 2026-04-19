from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Iterable

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models.memory import Memory, MemoryScope, MemoryType
from app.schemas.memory import MemoryCreate

logger = logging.getLogger(__name__)


def bulk_create(
    session: Session,
    *,
    character_id: str,
    items: Iterable[MemoryCreate],
    embed: bool = True,
) -> list[Memory]:
    """Create memories and (by default) embed them in a single pass.

    Embedding is best-effort: on failure we log and continue, leaving the
    `embedding` column NULL for those rows so the backfill script can
    pick them up later.
    """
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

    if embed and rows:
        try:
            from app.memory.embeddings import build_memory_embedding_input, embed_texts

            inputs = [
                build_memory_embedding_input(
                    narrative_text=r.narrative_text,
                    triggers=list(r.triggers or []),
                    relevance_tags=list(r.relevance_tags or []),
                )
                for r in rows
            ]
            vectors = embed_texts(inputs)
            for row, vec in zip(rows, vectors):
                row.embedding = vec
            session.flush()
        except Exception as exc:
            logger.warning(
                "Inline embedding failed for %d memories (character=%s): %s. "
                "Memories persisted without embeddings — run the backfill script.",
                len(rows),
                character_id,
                exc,
            )

    return rows


def get_by_ids(session: Session, ids: Iterable[str]) -> list[Memory]:
    id_list = list(ids)
    if not id_list:
        return []
    return list(session.execute(select(Memory).where(Memory.id.in_(id_list))).scalars())


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
