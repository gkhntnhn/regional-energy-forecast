"""Tests for src/energy_forecast/db/engine.py."""

from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import Session, sessionmaker

from energy_forecast.db.engine import (
    create_db_engine,
    create_session_factory,
    create_sync_engine,
    create_sync_session_factory,
    get_session,
    get_sync_session,
)


def _sqlite_sync_engine() -> Engine:
    """Create a plain SQLite in-memory engine (no pool params)."""
    return create_engine("sqlite:///:memory:")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_db_config() -> MagicMock:
    """Create a MagicMock that behaves like DatabaseConfig."""
    cfg = MagicMock()
    cfg.pool_size = 2
    cfg.max_overflow = 2
    cfg.pool_timeout = 30
    cfg.pool_recycle = 300
    cfg.pool_pre_ping = True
    cfg.echo = False
    return cfg


# ---------------------------------------------------------------------------
# Tests for create_db_engine (async)
# ---------------------------------------------------------------------------


class TestCreateDbEngine:
    """Test suite for create_db_engine."""

    @patch("energy_forecast.db.engine.create_async_engine")
    def test_returns_async_engine(self, mock_create: MagicMock) -> None:
        """Test create_db_engine calls create_async_engine and returns result."""
        sentinel = MagicMock(spec=AsyncEngine)
        mock_create.return_value = sentinel

        cfg = _make_db_config()
        result = create_db_engine("postgresql+asyncpg://user:pw@host/db", cfg)

        assert result is sentinel
        mock_create.assert_called_once_with(
            "postgresql+asyncpg://user:pw@host/db",
            pool_size=cfg.pool_size,
            max_overflow=cfg.max_overflow,
            pool_timeout=cfg.pool_timeout,
            pool_recycle=cfg.pool_recycle,
            pool_pre_ping=cfg.pool_pre_ping,
            echo=cfg.echo,
        )

    @patch("energy_forecast.db.engine.create_async_engine")
    def test_forwards_db_config_values(self, mock_create: MagicMock) -> None:
        """Test that all DatabaseConfig fields are forwarded to engine."""
        cfg = _make_db_config()
        cfg.pool_size = 5
        cfg.max_overflow = 10
        cfg.echo = True

        create_db_engine("postgresql+asyncpg://localhost/test", cfg)

        call_kwargs = mock_create.call_args.kwargs
        assert call_kwargs["pool_size"] == 5
        assert call_kwargs["max_overflow"] == 10
        assert call_kwargs["echo"] is True


# ---------------------------------------------------------------------------
# Tests for create_session_factory (async)
# ---------------------------------------------------------------------------


class TestCreateSessionFactory:
    """Test suite for create_session_factory."""

    def test_returns_async_sessionmaker(self) -> None:
        """Test create_session_factory returns an async_sessionmaker."""
        engine = create_async_engine("sqlite+aiosqlite:///:memory:")
        try:
            factory = create_session_factory(engine)
            assert isinstance(factory, async_sessionmaker)
        finally:
            engine.sync_engine.dispose()

    @pytest.mark.asyncio
    async def test_factory_creates_async_session(self) -> None:
        """Test the factory produces AsyncSession instances."""
        engine = create_async_engine("sqlite+aiosqlite:///:memory:")
        try:
            factory = create_session_factory(engine)
            async with factory() as session:
                assert isinstance(session, AsyncSession)
        finally:
            await engine.dispose()


# ---------------------------------------------------------------------------
# Tests for get_session (async generator)
# ---------------------------------------------------------------------------


class TestGetSession:
    """Test suite for get_session async generator."""

    @pytest.mark.asyncio
    async def test_yields_session_and_commits(self) -> None:
        """Test get_session yields a session and commits on normal exit."""
        engine = create_async_engine("sqlite+aiosqlite:///:memory:")
        factory = create_session_factory(engine)
        try:
            gen = get_session(factory)
            session = await gen.__anext__()
            assert isinstance(session, AsyncSession)
            # Normal close triggers commit path (line 70)
            with pytest.raises(StopAsyncIteration):
                await gen.__anext__()
        finally:
            await engine.dispose()

    @pytest.mark.asyncio
    async def test_rolls_back_on_exception(self) -> None:
        """Test get_session rolls back when an exception is raised."""
        engine = create_async_engine("sqlite+aiosqlite:///:memory:")
        factory = create_session_factory(engine)
        try:
            gen = get_session(factory)
            session = await gen.__anext__()
            assert isinstance(session, AsyncSession)
            # Throw an exception to trigger rollback path (lines 71-73)
            with pytest.raises(ValueError, match="test error"):
                await gen.athrow(ValueError("test error"))
        finally:
            await engine.dispose()

    @pytest.mark.asyncio
    async def test_session_executes_query(self) -> None:
        """Test the yielded session can execute SQL queries."""
        engine = create_async_engine("sqlite+aiosqlite:///:memory:")
        factory = create_session_factory(engine)
        try:
            gen = get_session(factory)
            session = await gen.__anext__()
            result = await session.execute(text("SELECT 1"))
            assert result.scalar() == 1
            with pytest.raises(StopAsyncIteration):
                await gen.__anext__()
        finally:
            await engine.dispose()

    @pytest.mark.asyncio
    async def test_exception_type_preserved_after_rollback(self) -> None:
        """Test that the original exception type re-raises after rollback."""
        engine = create_async_engine("sqlite+aiosqlite:///:memory:")
        factory = create_session_factory(engine)
        try:
            gen = get_session(factory)
            _session = await gen.__anext__()
            with pytest.raises(RuntimeError, match="async failure"):
                await gen.athrow(RuntimeError("async failure"))
        finally:
            await engine.dispose()


# ---------------------------------------------------------------------------
# Tests for create_sync_engine
# ---------------------------------------------------------------------------


class TestCreateSyncEngine:
    """Test suite for create_sync_engine."""

    @patch("energy_forecast.db.engine.create_engine")
    def test_returns_engine(self, mock_create: MagicMock) -> None:
        """Test create_sync_engine calls create_engine and returns result."""
        sentinel = MagicMock(spec=Engine)
        mock_create.return_value = sentinel

        result = create_sync_engine("postgresql+psycopg2://user:pw@host/db")

        assert result is sentinel
        mock_create.assert_called_once_with(
            "postgresql+psycopg2://user:pw@host/db",
            pool_size=1,
            max_overflow=0,
            pool_pre_ping=True,
        )

    @patch("energy_forecast.db.engine.create_engine")
    def test_hardcoded_pool_params(self, mock_create: MagicMock) -> None:
        """Test that pool_size=1 and max_overflow=0 are hardcoded."""
        create_sync_engine("postgresql+psycopg2://localhost/test")

        call_kwargs = mock_create.call_args.kwargs
        assert call_kwargs["pool_size"] == 1
        assert call_kwargs["max_overflow"] == 0
        assert call_kwargs["pool_pre_ping"] is True


# ---------------------------------------------------------------------------
# Tests for create_sync_session_factory
# ---------------------------------------------------------------------------


class TestCreateSyncSessionFactory:
    """Test suite for create_sync_session_factory."""

    def test_returns_sessionmaker(self) -> None:
        """Test create_sync_session_factory returns a sessionmaker."""
        engine = _sqlite_sync_engine()
        try:
            factory = create_sync_session_factory(engine)
            assert isinstance(factory, sessionmaker)
        finally:
            engine.dispose()

    def test_factory_creates_sync_session(self) -> None:
        """Test the factory produces Session instances."""
        engine = _sqlite_sync_engine()
        try:
            factory = create_sync_session_factory(engine)
            with factory() as session:
                assert isinstance(session, Session)
        finally:
            engine.dispose()


# ---------------------------------------------------------------------------
# Tests for get_sync_session (sync generator)
# ---------------------------------------------------------------------------


class TestGetSyncSession:
    """Test suite for get_sync_session sync generator."""

    def test_yields_session_and_commits(self) -> None:
        """Test get_sync_session yields a session and commits on normal exit."""
        engine = _sqlite_sync_engine()
        factory = create_sync_session_factory(engine)
        try:
            gen = get_sync_session(factory)
            session = next(gen)
            assert isinstance(session, Session)
            # Normal close triggers commit path (line 107)
            with pytest.raises(StopIteration):
                next(gen)
        finally:
            engine.dispose()

    def test_rolls_back_on_exception(self) -> None:
        """Test get_sync_session rolls back when an exception is raised."""
        engine = _sqlite_sync_engine()
        factory = create_sync_session_factory(engine)
        try:
            gen = get_sync_session(factory)
            session = next(gen)
            assert isinstance(session, Session)
            # Throw an exception to trigger rollback path (lines 108-110)
            with pytest.raises(ValueError, match="test error"):
                gen.throw(ValueError("test error"))
        finally:
            engine.dispose()

    def test_session_executes_query(self) -> None:
        """Test the yielded session can execute SQL queries."""
        engine = _sqlite_sync_engine()
        factory = create_sync_session_factory(engine)
        try:
            gen = get_sync_session(factory)
            session = next(gen)
            result = session.execute(text("SELECT 1"))
            assert result.scalar() == 1
            with pytest.raises(StopIteration):
                next(gen)
        finally:
            engine.dispose()

    def test_exception_type_preserved_after_rollback(self) -> None:
        """Test that the original exception type re-raises after rollback."""
        engine = _sqlite_sync_engine()
        factory = create_sync_session_factory(engine)
        try:
            gen = get_sync_session(factory)
            _session = next(gen)
            with pytest.raises(RuntimeError, match="sync failure"):
                gen.throw(RuntimeError("sync failure"))
        finally:
            engine.dispose()
