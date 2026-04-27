"""One-time backfill: grant starting $CLAY to all existing players who haven't
received it yet.

Idempotent — safe to run multiple times. Players who already have a
'starting_grant' transaction are skipped.

Usage (from repo root with venv active):
    python -m scripts.grant_starting_clay
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make sure the app package is importable when running as a script.
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import get_settings
from app.db import SessionLocal, init_db
from app.economy.clay_ledger import get_ledger
from app.models.match import Player


def main() -> None:
    init_db()
    settings = get_settings()
    amount = settings.starting_clay_grant
    ledger = get_ledger()

    with SessionLocal() as session:
        players = list(session.query(Player).all())

    print(f"Checking {len(players)} player(s) for missing starting $CLAY grant…")
    granted = 0
    skipped = 0

    for player in players:
        existing = ledger.transactions_for_player(
            player.id, limit=1, reason="starting_grant"
        )
        if existing:
            skipped += 1
            continue
        try:
            ledger.credit(player.id, amount, reason="starting_grant")
            print(f"  Granted {amount} cents to @{player.username} ({player.id})")
            granted += 1
        except Exception as exc:
            print(f"  ERROR granting to @{player.username}: {exc}", file=sys.stderr)

    print(f"\nDone. Granted: {granted}  Already had grant: {skipped}")


if __name__ == "__main__":
    main()
