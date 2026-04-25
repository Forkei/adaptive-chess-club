"""Alembic environment.

Pulls the database URL from `app.config.get_settings()` so migrations use
the same DB as the app. Target metadata comes from `app.models.base.Base`
with all model modules imported for side effects.
"""

from __future__ import annotations

from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

# Import the app's models so their metadata is populated before we read it.
from app.config import get_settings
from app.models import character, feedback, match, memory  # noqa: F401 — register tables
from app.models.base import Base

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# If a caller (e.g. a test) already supplied sqlalchemy.url via
# config.set_main_option("sqlalchemy.url", ...), respect it. Otherwise
# derive from the app's Settings. The placeholder in alembic.ini
# ("driver://user:pass@localhost/dbname") is recognized as unset.
_current = config.get_main_option("sqlalchemy.url") or ""
if not _current or _current.startswith("driver://"):
    config.set_main_option("sqlalchemy.url", get_settings().database_url)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_as_batch=True,  # SQLite-friendly ALTERs
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            render_as_batch=True,  # SQLite-friendly ALTERs
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
