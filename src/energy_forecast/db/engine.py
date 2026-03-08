"""Database engine and session factories (async + sync)."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Generator
from typing import TYPE_CHECKING

from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import Session, sessionmaker

if TYPE_CHECKING:
    from sqlalchemy import Engine
    from sqlalchemy.ext.asyncio import AsyncEngine

    from energy_forecast.config import DatabaseConfig


# ---------------------------------------------------------------------------
# Async (serving pipeline — asyncpg)
# ---------------------------------------------------------------------------


def create_db_engine(
    database_url: str,
    db_config: DatabaseConfig,
) -> AsyncEngine:
    """Create an async SQLAlchemy engine with pool configuration.

    Args:
        database_url: PostgreSQL connection string (asyncpg driver).
        db_config: Pool configuration from settings.
    """
    return create_async_engine(
        database_url,
        pool_size=db_config.pool_size,
        max_overflow=db_config.max_overflow,
        pool_timeout=db_config.pool_timeout,
        pool_recycle=db_config.pool_recycle,
        pool_pre_ping=db_config.pool_pre_ping,
        echo=db_config.echo,
    )


def create_session_factory(
    engine: AsyncEngine,
) -> async_sessionmaker[AsyncSession]:
    """Create an async session factory bound to the engine."""
    return async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )


async def get_session(
    session_factory: async_sessionmaker[AsyncSession],
) -> AsyncGenerator[AsyncSession, None]:
    """Yield an async session with auto-commit/rollback.

    Usage as FastAPI dependency via app.state.session_factory.
    """
    async with session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


# ---------------------------------------------------------------------------
# Sync (training pipeline — psycopg2)
# ---------------------------------------------------------------------------


def create_sync_engine(database_url_sync: str) -> Engine:
    """Create a sync SQLAlchemy engine for training pipeline.

    Args:
        database_url_sync: PostgreSQL connection string (psycopg2 driver).
    """
    return create_engine(
        database_url_sync,
        pool_size=1,
        max_overflow=0,
        pool_pre_ping=True,
    )


def create_sync_session_factory(engine: Engine) -> sessionmaker[Session]:
    """Create a sync session factory bound to the engine."""
    return sessionmaker(engine, expire_on_commit=False)


def get_sync_session(
    session_factory: sessionmaker[Session],
) -> Generator[Session, None, None]:
    """Yield a sync session with auto-commit/rollback."""
    with session_factory() as session:
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
