"""Backfill embeddings for memories created before Phase 2b.

Walks all memories with `embedding IS NULL`, batch-encodes their
(narrative_text | triggers | relevance_tags) input, and writes the
vectors back. Safe to run multiple times — already-embedded rows are
skipped.

Usage:
    python -m scripts.backfill_embeddings
    python -m scripts.backfill_embeddings --batch-size 32
"""

from __future__ import annotations

import argparse
import logging
import sys

from sqlalchemy import select

from app.db import SessionLocal, init_db
from app.memory.embeddings import build_memory_embedding_input, embed_texts
from app.models.memory import Memory

logger = logging.getLogger(__name__)


def backfill(batch_size: int = 32, dry_run: bool = False) -> int:
    """Returns the number of memories newly embedded."""
    init_db()

    updated = 0
    with SessionLocal() as session:
        rows = list(
            session.execute(
                select(Memory).where(Memory.embedding.is_(None))
            ).scalars()
        )
        if not rows:
            logger.info("No memories to backfill.")
            return 0

        logger.info("Backfilling %d memories…", len(rows))
        for i in range(0, len(rows), batch_size):
            batch = rows[i : i + batch_size]
            inputs = [
                build_memory_embedding_input(
                    narrative_text=m.narrative_text,
                    triggers=list(m.triggers or []),
                    relevance_tags=list(m.relevance_tags or []),
                )
                for m in batch
            ]
            if dry_run:
                logger.info("[dry-run] would embed batch of %d", len(batch))
                continue
            vectors = embed_texts(inputs)
            for m, vec in zip(batch, vectors):
                m.embedding = vec
                updated += 1
            session.commit()
            logger.info("Backfilled %d / %d", min(i + batch_size, len(rows)), len(rows))

    logger.info("Done. Embedded %d memories.", updated)
    return updated


def _main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    backfill(batch_size=args.batch_size, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(_main())
