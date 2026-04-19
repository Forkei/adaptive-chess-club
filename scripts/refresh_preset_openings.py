"""Refresh preset characters' `opening_preferences` in the database.

The preset seed logic (app/characters/seed.py) skips existing preset rows,
so tweaking a PresetSpec does not flow to already-seeded DBs. This script
is the escape hatch: for every preset in `app.characters.presets.PRESETS`,
find the matching DB row by `preset_key` and overwrite
`opening_preferences` from the spec. Everything else — memories,
backstory, voice, Elo state — is left alone.

Usage:
    python -m scripts.refresh_preset_openings
    python scripts/refresh_preset_openings.py

Idempotent: rows that already match the spec are skipped. Exits
non-zero if any preset_key in the spec list has no matching DB row
(suggests the seed never ran).
"""

from __future__ import annotations

import logging
import sys

from sqlalchemy import select

from app.characters.presets import PRESETS
from app.db import session_scope
from app.models.character import Character

logger = logging.getLogger(__name__)


def refresh() -> int:
    updated = 0
    missing: list[str] = []
    for spec in PRESETS:
        target = list(spec.opening_preferences)
        with session_scope() as session:
            row = session.execute(
                select(Character).where(Character.preset_key == spec.preset_key)
            ).scalar_one_or_none()
            if row is None:
                missing.append(spec.preset_key)
                continue
            current = list(row.opening_preferences or [])
            if current == target:
                print(f"[skip] {spec.preset_key}: already up to date")
                continue
            row.opening_preferences = target
            print(f"[update] {spec.preset_key}: {current!r} -> {target!r}")
            updated += 1
    if missing:
        print(f"[warn] preset rows missing (seed not run?): {missing}")
    print(f"Done. {updated} preset(s) updated.")
    return 0 if not missing else 2


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    sys.exit(refresh())
