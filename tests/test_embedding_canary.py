"""Canary test: real end-to-end embedding pipeline.

Purpose: catch environment-level failures in the embedding stack (broken
sentence-transformers import, corrupt package metadata, missing model
cache, etc.). Silent-embed-failure is what drove the Patch Pass 1 hotfix
— this test asserts the happy path actually embeds, rather than mocking
it out.

Runs in the non-live suite by default — the all-MiniLM-L6-v2 model (~90MB)
is downloaded once to the sentence-transformers cache and reused. If
the cache is missing, the first run pulls it; subsequent runs are fast.
Set EMBEDDING_CANARY_SKIP=1 to skip in environments without network
access on first run.
"""

from __future__ import annotations

import os

import pytest

from app.db import SessionLocal
from app.memory.crud import bulk_create
from app.models.character import Character, CharacterState
from app.models.memory import MemoryScope, MemoryType
from app.schemas.memory import MemoryCreate


def _skip_if_opted_out() -> None:
    if os.environ.get("EMBEDDING_CANARY_SKIP") == "1":
        pytest.skip("EMBEDDING_CANARY_SKIP=1 set")


def _mk_character(sess) -> Character:
    c = Character(
        name="Canary", aggression=5, risk_tolerance=5, patience=5, trash_talk=5,
        target_elo=1400, current_elo=1400, floor_elo=1400, max_elo=1800,
        adaptive=True, state=CharacterState.READY,
    )
    sess.add(c)
    sess.flush()
    return c


def test_bulk_create_embeds_memories_end_to_end():
    """Create a memory through the real pipeline; assert embedding is populated.

    This is the canary — if sentence-transformers can't import (e.g. because
    of the requests-2.32.5 corrupt-metadata issue we hit in Patch Pass 1),
    the call silently logs and returns memories with embedding=None. We
    hard-fail here so CI notices.
    """
    _skip_if_opted_out()

    with SessionLocal() as sess:
        char = _mk_character(sess)
        items = [
            MemoryCreate(
                scope=MemoryScope.CHARACTER_LORE,
                type=MemoryType.OPINION,
                emotional_valence=0.0,
                triggers=["opening", "e4", "aggression"],
                relevance_tags=["tactical", "opening"],
                narrative_text="I always open with the King's Gambit when I smell fear.",
                player_id=None, match_id=None,
            ),
            MemoryCreate(
                scope=MemoryScope.CHARACTER_LORE,
                type=MemoryType.OPINION,
                emotional_valence=0.2,
                triggers=["endgame", "king", "pawn"],
                relevance_tags=["endgame"],
                narrative_text="King-and-pawn endings are the only honest part of chess.",
                player_id=None, match_id=None,
            ),
        ]
        rows = bulk_create(sess, character_id=char.id, items=items, embed=True)
        sess.commit()
        char_id = char.id

    # Reload to confirm persistence.
    with SessionLocal() as sess:
        from sqlalchemy import select
        from app.models.memory import Memory

        persisted = list(
            sess.execute(select(Memory).where(Memory.character_id == char_id)).scalars()
        )
        assert len(persisted) == 2
        for m in persisted:
            assert m.embedding is not None, (
                f"Memory {m.id} has NULL embedding — the embedding pipeline is silently "
                f"failing. Check app/memory/crud.py warning logs for the real error."
            )
            # all-MiniLM-L6-v2 is 384 dims; other models would differ but we
            # pin to this one in embeddings.py.
            assert len(m.embedding) == 384
            # Sanity: not all zeros.
            assert any(abs(x) > 1e-6 for x in m.embedding[:10])
