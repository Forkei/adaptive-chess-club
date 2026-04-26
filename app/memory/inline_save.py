"""Fire-and-forget inline memory persistence triggered by Soul mid-conversation.

Call via ``asyncio.create_task(save_inline_memory(...))``.  The coroutine opens
its own DB session so the caller's session is already closed by the time this
runs.  Failures are logged at WARNING — inline memory loss is visible in logs
but never propagates to the caller.
"""

from __future__ import annotations

import logging
import math

from sqlalchemy import select

from app.db import SessionLocal
from app.memory.embeddings import build_memory_embedding_input, embed_texts
from app.models.memory import Memory, MemoryScope, MemoryType
from app.schemas.agents import InlineMemoryRequest
from app.schemas.memory import MemoryCreate

logger = logging.getLogger(__name__)

DEDUP_COSINE_THRESHOLD = 0.92


def _cosine_sim(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return dot / (mag_a * mag_b)


def _is_near_duplicate(
    session,
    *,
    character_id: str | None = None,
    agent_id: str | None = None,
    player_id: str,
    proposed_embedding: list[float],
) -> bool:
    if agent_id is not None:
        scope_filter = Memory.agent_id == agent_id
    elif character_id is not None:
        scope_filter = Memory.character_id == character_id
    else:
        raise ValueError("_is_near_duplicate requires character_id or agent_id")
    stmt = (
        select(Memory.embedding)
        .where(
            scope_filter,
            Memory.player_id == player_id,
            Memory.scope == MemoryScope.OPPONENT_SPECIFIC,
            Memory.embedding.is_not(None),
        )
    )
    for (emb,) in session.execute(stmt):
        if emb and _cosine_sim(proposed_embedding, emb) > DEDUP_COSINE_THRESHOLD:
            return True
    return False


async def save_inline_memory(
    request: InlineMemoryRequest,
    *,
    character_id: str | None = None,
    agent_id: str | None = None,
    player_id: str,
    match_id: str | None = None,
) -> None:
    """Embed and persist a Soul-generated inline memory.

    Fire-and-forget: call as ``asyncio.create_task(save_inline_memory(...))``.
    Survives client disconnect. Logs and returns silently on any error.

    Pass either ``character_id`` (character room) or ``agent_id`` (agent room).
    """
    scope_label = agent_id or character_id
    try:
        embed_input = build_memory_embedding_input(
            narrative_text=request.narrative_text,
            triggers=list(request.triggers),
            relevance_tags=list(request.relevance_tags),
        )
        vectors = embed_texts([embed_input])
        proposed_embedding = vectors[0]

        with SessionLocal() as session:
            if _is_near_duplicate(
                session,
                character_id=character_id,
                agent_id=agent_id,
                player_id=player_id,
                proposed_embedding=proposed_embedding,
            ):
                logger.info(
                    "inline_memory_dedup_skipped player=%s scope=%s match=%s",
                    player_id, scope_label, match_id,
                )
                return

            item = MemoryCreate(
                scope=MemoryScope.OPPONENT_SPECIFIC,
                type=MemoryType(request.type),
                emotional_valence=request.emotional_valence,
                triggers=list(request.triggers),
                narrative_text=request.narrative_text,
                relevance_tags=list(request.relevance_tags),
                player_id=player_id,
                match_id=match_id,
            )
            # bulk_create handles embedding; pass embed=False since we already have it.
            row = Memory(
                character_id=character_id,
                agent_id=agent_id,
                player_id=item.player_id,
                match_id=item.match_id,
                scope=item.scope,
                type=item.type,
                emotional_valence=item.emotional_valence,
                triggers=list(item.triggers),
                narrative_text=item.narrative_text,
                relevance_tags=list(item.relevance_tags),
                embedding=proposed_embedding,
                surface_count=0,
                last_surfaced_at=None,
            )
            session.add(row)
            session.commit()

        logger.info(
            "inline_memory_saved player=%s scope=%s match=%s triggers=%s",
            player_id, scope_label, match_id, list(request.triggers),
        )

    except Exception:
        logger.warning(
            "inline_memory_failed player=%s scope=%s match=%s — skipping",
            player_id, scope_label, match_id,
            exc_info=True,
        )
