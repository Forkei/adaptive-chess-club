"""Boot-time migration script (called from entrypoint.sh).

On a fresh database: `alembic upgrade head` fails because the 0001 baseline
is a no-op and 0002 tries to ALTER tables that don't exist yet. Instead we
run SQLAlchemy create_all (which produces the full current schema in one
shot) and then stamp Alembic at head so future upgrades work normally.

On an existing database: run `alembic upgrade head` in the normal way.
"""
from __future__ import annotations

import subprocess
import sys


def main() -> None:
    from sqlalchemy import inspect

    from app.db import engine, init_db

    with engine.connect() as conn:
        tables = inspect(conn).get_table_names()
    engine.dispose()

    if not tables:
        print("[migrate] Fresh database — running create_all + alembic stamp head", flush=True)
        init_db()
        subprocess.run(["alembic", "stamp", "head"], check=True)
    else:
        print(f"[migrate] Existing database ({len(tables)} tables) — alembic upgrade head", flush=True)
        subprocess.run(["alembic", "upgrade", "head"], check=True)


if __name__ == "__main__":
    sys.exit(main())
