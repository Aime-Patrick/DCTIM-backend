"""SQLAlchemy engine and session factory for the RAG module.

Usage
-----
Call `build_engine(database_url)` once at startup and pass the resulting
engine to `SessionFactory`. Each request should obtain a short-lived session
via the `get_session` FastAPI dependency.
"""
from __future__ import annotations

from collections.abc import Generator

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker


def build_engine(database_url: str) -> Engine:
    """Create a synchronous SQLAlchemy engine with sane defaults.

    `pool_pre_ping=True` ensures stale connections recycled by Docker or the
    database are detected before use rather than surfacing as cryptic errors.
    """
    return create_engine(
        database_url,
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=10,
        echo=False,
    )


def build_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)


def get_session(session_factory: sessionmaker[Session]) -> Generator[Session, None, None]:
    """FastAPI dependency that yields a database session per request."""
    session: Session = session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def check_db_connection(engine: Engine) -> None:
    """Raise if the database is unreachable. Called during application startup."""
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
