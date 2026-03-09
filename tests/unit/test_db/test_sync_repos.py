"""Tests for SyncDataAccess layer.

Since SyncDataAccess uses pg_insert (PostgreSQL-only ON CONFLICT DO UPDATE),
write/upsert tests use MagicMock while read tests use sync SQLite engine.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from energy_forecast.db.base import Base
import energy_forecast.db.models  # noqa: F401
from energy_forecast.db.models import (
    EpiasMarketModel,
    TurkishHolidayModel,
)
from energy_forecast.db.sync_repos import SyncDataAccess, _rows_to_df


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sync_engine():
    """Create a sync SQLite in-memory engine with schema."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()


@pytest.fixture()
def sync_session(sync_engine):  # type: ignore[no-untyped-def]
    """Yield a sync Session for testing."""
    factory = sessionmaker(bind=sync_engine, expire_on_commit=False)
    with factory() as session:
        yield session


@pytest.fixture()
def dao(sync_session: Session) -> SyncDataAccess:
    """Return a SyncDataAccess instance backed by SQLite."""
    return SyncDataAccess(sync_session)


# ---------------------------------------------------------------------------
# _rows_to_df helper
# ---------------------------------------------------------------------------


class TestRowsToDf:
    """Test the _rows_to_df utility function."""

    def test_empty_result(self, sync_session: Session) -> None:
        """Empty query result returns empty DataFrame."""
        result = sync_session.execute(
            EpiasMarketModel.__table__.select()
        )
        df = _rows_to_df(result)
        assert df.empty

    def test_with_data(self, sync_session: Session) -> None:
        """Rows are converted to DataFrame with datetime index."""
        now = datetime(2024, 1, 1, 0, 0)
        sync_session.add(EpiasMarketModel(dt=now, rtc=100.0))
        sync_session.flush()

        from sqlalchemy import select as sa_select
        result = sync_session.execute(sa_select(EpiasMarketModel))
        df = _rows_to_df(result)
        assert len(df) == 1
        assert "rtc" in df.columns


# ---------------------------------------------------------------------------
# EPIAS Market reads (SQLite)
# ---------------------------------------------------------------------------


class TestEpiasMarketRead:
    """Read operations for EPIAS market — real SQLite engine."""

    def _seed_market(
        self, session: Session, hours: int = 24, start: str = "2024-01-01",
    ) -> None:
        """Insert sample market rows."""
        rng = np.random.default_rng(42)
        base = datetime.fromisoformat(start)
        for h in range(hours):
            dt = base + timedelta(hours=h)
            session.add(EpiasMarketModel(
                dt=dt,
                rtc=float(rng.uniform(500, 1500)),
                dam_purchase=float(rng.uniform(400, 1200)),
                load_forecast=float(rng.uniform(600, 1600)),
            ))
        session.flush()

    def test_get_range_empty(self, dao: SyncDataAccess) -> None:
        """Empty DB returns empty DataFrame."""
        df = dao.get_epias_market_range(
            datetime(2024, 1, 1), datetime(2024, 1, 2),
        )
        assert df.empty

    def test_get_range_with_data(
        self, sync_session: Session, dao: SyncDataAccess,
    ) -> None:
        """Returns data within requested range."""
        self._seed_market(sync_session, hours=48)
        df = dao.get_epias_market_range(
            datetime(2024, 1, 1, 0), datetime(2024, 1, 1, 23),
        )
        assert len(df) == 24
        assert "rtc" in df.columns

    def test_get_year_with_data(
        self, sync_session: Session, dao: SyncDataAccess,
    ) -> None:
        """Year filter returns correct year data."""
        self._seed_market(sync_session, hours=48)
        df = dao.get_epias_market_year(2024)
        assert len(df) == 48

    def test_get_year_no_match(self, dao: SyncDataAccess) -> None:
        """Nonexistent year returns empty DataFrame."""
        df = dao.get_epias_market_year(1999)
        assert df.empty


# ---------------------------------------------------------------------------
# Holiday reads (SQLite)
# ---------------------------------------------------------------------------


class TestHolidayRead:
    """Read operations for holidays — real SQLite engine."""

    def _seed_holidays(self, session: Session) -> None:
        """Insert sample holidays."""
        session.add(TurkishHolidayModel(
            date=date(2024, 1, 1),
            holiday_name="Yilbasi",
            raw_holiday_name="Yılbaşı",
            is_ramadan=False,
        ))
        session.add(TurkishHolidayModel(
            date=date(2024, 4, 10),
            holiday_name="Ramazan_Bayrami_1",
            raw_holiday_name="Ramazan Bayramı 1. Gün",
            is_ramadan=True,
            bayram_gun_no=1,
        ))
        session.flush()

    def test_get_holidays_empty(self, dao: SyncDataAccess) -> None:
        """Empty table returns empty DataFrame."""
        df = dao.get_holidays()
        assert df.empty

    def test_get_holidays(
        self, sync_session: Session, dao: SyncDataAccess,
    ) -> None:
        """Returns all holidays with date index."""
        self._seed_holidays(sync_session)
        df = dao.get_holidays()
        assert len(df) == 2
        assert "holiday_name" in df.columns

    def test_get_holidays_range(
        self, sync_session: Session, dao: SyncDataAccess,
    ) -> None:
        """Date range filter works correctly."""
        self._seed_holidays(sync_session)
        df = dao.get_holidays_range(date(2024, 4, 1), date(2024, 4, 30))
        assert len(df) == 1
        assert df.iloc[0]["holiday_name"] == "Ramazan_Bayrami_1"


# ---------------------------------------------------------------------------
# Upsert operations (mock — pg_insert is PostgreSQL-only)
# ---------------------------------------------------------------------------


class TestUpsertMocked:
    """Upsert methods are PostgreSQL-specific; verify argument flow via mock."""

    def test_upsert_market_empty(self, dao: SyncDataAccess) -> None:
        """Empty rows list returns 0 without hitting DB."""
        assert dao.upsert_epias_market([]) == 0

    def test_upsert_generation_empty(self, dao: SyncDataAccess) -> None:
        """Empty rows list returns 0."""
        assert dao.upsert_epias_generation([]) == 0

    def test_upsert_weather_empty(self, dao: SyncDataAccess) -> None:
        """Empty rows list returns 0."""
        assert dao.upsert_weather([]) == 0

    @patch("energy_forecast.db.sync_repos.pg_insert")
    def test_upsert_market_calls_execute(
        self, mock_pg: MagicMock, dao: SyncDataAccess,
    ) -> None:
        """Non-empty rows trigger session.execute + flush."""
        mock_stmt = MagicMock()
        mock_pg.return_value = mock_stmt
        mock_stmt.on_conflict_do_update.return_value = mock_stmt
        # Mock excluded attribute for set_ dict
        mock_stmt.excluded = {col: col for col in [
            "fdpp", "rtc", "dam_purchase", "bilateral",
            "load_forecast", "fetched_at",
        ]}

        dao._session = MagicMock()
        rows: list[dict[str, Any]] = [{"datetime": datetime(2024, 1, 1), "rtc": 100.0}]
        count = dao.upsert_epias_market(rows)

        assert count == 1
        dao._session.execute.assert_called_once()
        dao._session.flush.assert_called_once()
