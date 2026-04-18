from __future__ import annotations

from sqlalchemy import select

from app.characters.presets import PRESETS
from app.characters.seed import seed_presets
from app.db import SessionLocal
from app.models.character import Character


def test_seed_presets_creates_all():
    results = seed_presets(run_generation=False)
    assert set(results.keys()) == {p.preset_key for p in PRESETS}
    assert all(v is True for v in results.values())

    with SessionLocal() as s:
        rows = list(s.execute(select(Character)).scalars())
        assert len(rows) == len(PRESETS)
        preset_keys = {r.preset_key for r in rows}
        assert preset_keys == {p.preset_key for p in PRESETS}
        assert all(r.is_preset for r in rows)


def test_seed_presets_is_idempotent():
    seed_presets(run_generation=False)
    first_ids = {}
    with SessionLocal() as s:
        for r in s.execute(select(Character)).scalars():
            first_ids[r.preset_key] = r.id

    results2 = seed_presets(run_generation=False)
    assert all(v is False for v in results2.values())

    with SessionLocal() as s:
        rows = list(s.execute(select(Character)).scalars())
        assert len(rows) == len(PRESETS)
        for r in rows:
            assert r.id == first_ids[r.preset_key]


def test_seed_presets_skips_generation_when_disabled():
    seed_presets(run_generation=False)
    with SessionLocal() as s:
        rows = list(s.execute(select(Character)).scalars())
        for r in rows:
            # Nothing persisted yet — seed ran without generation.
            assert len(list(r.memories)) == 0
