"""Interface over memory embeddings.

Currently a thin wrapper around the embedding column on Memory: cosine
similarity is computed in-Python with numpy. The corpus is small
(~40-50 memories per character, <5 characters preset), so in-memory
scoring is sub-millisecond.

Phase 3 can swap the implementation for sqlite-vec without touching
callers — the interface (`upsert`, `search`, `get_embedding`) is stable.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.memory import Memory, MemoryScope

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VectorHit:
    memory_id: str
    score: float


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    va = np.asarray(a, dtype=np.float32)
    vb = np.asarray(b, dtype=np.float32)
    na = float(np.linalg.norm(va))
    nb = float(np.linalg.norm(vb))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


def upsert(session: Session, *, memory_id: str, embedding: list[float]) -> None:
    """Write `embedding` onto the memory row. Caller commits."""
    row = session.get(Memory, memory_id)
    if row is None:
        raise ValueError(f"Memory {memory_id} not found")
    row.embedding = list(embedding)
    session.flush()


def search(
    session: Session,
    *,
    query_embedding: list[float],
    k: int,
    character_id: str | None = None,
    agent_id: str | None = None,
    scope: MemoryScope | None = None,
    player_id: str | None = None,
    include_null_player: bool = True,
) -> list[VectorHit]:
    """Return top-k memories ranked by cosine similarity.

    Pass either `character_id` (for character-scoped memories) or `agent_id`
    (for agent-scoped memories, Block 13+). Exactly one must be provided.

    Filters:
    - `scope`: restrict to a specific MemoryScope if given
    - `player_id`: restrict to memories attached to this player; when
      `include_null_player=True` (default) memories with `player_id` NULL
      are also kept (character_lore, cross_player-style general memories)

    Memories without an embedding are skipped silently — the caller should
    run the backfill script to cover them.
    """
    if agent_id is not None:
        stmt = select(Memory).where(Memory.agent_id == agent_id)
    elif character_id is not None:
        stmt = select(Memory).where(Memory.character_id == character_id)
    else:
        raise ValueError("search() requires either character_id or agent_id")
    if scope is not None:
        stmt = stmt.where(Memory.scope == scope)
    if player_id is not None:
        if include_null_player:
            stmt = stmt.where((Memory.player_id == player_id) | (Memory.player_id.is_(None)))
        else:
            stmt = stmt.where(Memory.player_id == player_id)

    rows = list(session.execute(stmt).scalars())
    if not rows:
        return []

    # Keep only rows that have been embedded.
    candidates: list[tuple[str, list[float]]] = []
    for m in rows:
        if m.embedding:
            candidates.append((m.id, list(m.embedding)))

    if not candidates:
        return []

    # Vectorized cosine over the candidate set.
    q = np.asarray(query_embedding, dtype=np.float32)
    q_norm = float(np.linalg.norm(q))
    if q_norm == 0.0:
        return []

    mat = np.asarray([c[1] for c in candidates], dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1)
    # Avoid divide-by-zero.
    safe_norms = np.where(norms == 0.0, 1.0, norms)
    sims = (mat @ q) / (safe_norms * q_norm)

    ranked = sorted(
        (VectorHit(memory_id=cid, score=float(sims[i])) for i, (cid, _) in enumerate(candidates)),
        key=lambda h: h.score,
        reverse=True,
    )
    return ranked[:k]


def get_embedding(session: Session, memory_id: str) -> list[float] | None:
    row = session.get(Memory, memory_id)
    if row is None:
        return None
    return list(row.embedding) if row.embedding else None


def ensure_embedding_column(bind) -> None:
    """Idempotently add the `embedding` column to `memories` on existing DBs.

    `Base.metadata.create_all` only creates missing tables, not missing
    columns. Phase 2b adds a column to a pre-existing table; this helper
    bridges the gap without requiring a proper migration tool in 2b.
    """
    with bind.begin() as conn:
        dialect = conn.dialect.name
        if dialect != "sqlite":
            # Other backends will need proper migrations; bail loudly.
            logger.warning(
                "ensure_embedding_column: dialect=%s not SQLite — skipping; "
                "ensure the `memories.embedding` column exists via your migration tool.",
                dialect,
            )
            return
        cols = conn.exec_driver_sql("PRAGMA table_info(memories)").fetchall()
        names = {row[1] for row in cols}
        if "embedding" not in names:
            logger.info("Adding memories.embedding column (Phase 2b migration).")
            conn.exec_driver_sql("ALTER TABLE memories ADD COLUMN embedding JSON")
