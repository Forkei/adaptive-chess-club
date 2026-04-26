from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from app.config import get_settings

_settings = get_settings()

_connect_args: dict = {}
if _settings.database_url.startswith("sqlite"):
    _connect_args["check_same_thread"] = False
    _connect_args["timeout"] = 30  # seconds to wait for write lock

engine = create_engine(
    _settings.database_url,
    connect_args=_connect_args,
    future=True,
)

if _settings.database_url.startswith("sqlite"):
    @event.listens_for(engine, "connect")
    def _set_sqlite_pragmas(dbapi_conn, _record):
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL")   # concurrent reads during writes
        cur.execute("PRAGMA synchronous=NORMAL")  # safe with WAL, faster than FULL
        cur.close()

SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
    future=True,
)


@contextmanager
def session_scope() -> Iterator[Session]:
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_session() -> Iterator[Session]:
    """FastAPI dependency."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def init_db() -> None:
    from app.models.base import Base  # noqa: F401 — import side effect
    from app.models import auth, character, chat, evolution, feedback, lobby, match, memory, player_agent  # noqa: F401

    Base.metadata.create_all(bind=engine)

    # Phase 2b: make sure columns added in 2b exist on pre-existing rows.
    from app.memory.vector_store import ensure_embedding_column

    ensure_embedding_column(engine)
