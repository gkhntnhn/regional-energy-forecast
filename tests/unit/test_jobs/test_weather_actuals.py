"""Tests for src/energy_forecast/jobs/weather_actuals.py."""

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker

from energy_forecast.config.settings import Settings
from energy_forecast.jobs.weather_actuals import (
    fetch_and_store_actuals,
    main,
    run_scheduler,
)

TZ = timezone(timedelta(hours=3))

# Patch targets — lazy imports inside fetch_and_store_actuals resolve from source modules
_PATCH_REPO = "energy_forecast.db.repositories.weather_repo.WeatherSnapshotRepository"
_PATCH_CLIENT = "energy_forecast.data.openmeteo_client.OpenMeteoClient"
_PATCH_LOAD_CONFIG = "energy_forecast.config.settings.load_config"
_PATCH_CREATE_ENGINE = "energy_forecast.db.create_db_engine"
_PATCH_CREATE_SF = "energy_forecast.db.create_session_factory"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_weather_df(start: str, hours: int = 24) -> pd.DataFrame:
    """Create a sample weather DataFrame with DatetimeIndex."""
    rng = np.random.default_rng(42)
    idx = pd.date_range(start, periods=hours, freq="h", tz=TZ)
    return pd.DataFrame(
        {
            "temperature_2m": rng.uniform(5, 30, hours),
            "apparent_temperature": rng.uniform(3, 28, hours),
            "relative_humidity_2m": rng.uniform(30, 90, hours),
            "wind_speed_10m": rng.uniform(0, 20, hours),
            "weather_code": rng.choice([0, 1, 2, 3, 61], hours),
        },
        index=idx,
    )


def _make_mock_session_factory() -> MagicMock:
    """Create a mock async_sessionmaker that passes isinstance check."""
    sf = MagicMock(spec=async_sessionmaker)
    mock_session = AsyncMock()
    # Make sf() return an async context manager yielding mock_session
    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=mock_session)
    ctx.__aexit__ = AsyncMock(return_value=False)
    sf.return_value = ctx
    return sf


# ---------------------------------------------------------------------------
# TestFetchAndStoreActuals
# ---------------------------------------------------------------------------


class TestFetchAndStoreActuals:
    """Test suite for fetch_and_store_actuals."""

    @pytest.mark.asyncio
    async def test_invalid_session_factory_returns_zero(self, settings: Settings) -> None:
        """Test fetch_and_store_actuals when session_factory is not async_sessionmaker returns 0."""
        # Arrange
        invalid_sf = "not_a_session_factory"

        # Act
        result = await fetch_and_store_actuals(invalid_sf, settings)

        # Assert
        assert result == 0

    @pytest.mark.asyncio
    async def test_invalid_settings_returns_zero(self) -> None:
        """Test fetch_and_store_actuals when settings is not Settings returns 0."""
        # Arrange
        sf = _make_mock_session_factory()
        invalid_settings = {"not": "settings"}

        # Act
        result = await fetch_and_store_actuals(sf, invalid_settings)

        # Assert
        assert result == 0

    @pytest.mark.asyncio
    async def test_none_session_factory_returns_zero(self, settings: Settings) -> None:
        """Test fetch_and_store_actuals when session_factory is None returns 0."""
        result = await fetch_and_store_actuals(None, settings)
        assert result == 0

    @pytest.mark.asyncio
    async def test_none_settings_returns_zero(self) -> None:
        """Test fetch_and_store_actuals when settings is None returns 0."""
        sf = _make_mock_session_factory()
        result = await fetch_and_store_actuals(sf, None)
        assert result == 0

    @pytest.mark.asyncio
    async def test_idempotent_skip_when_actuals_exist(self, settings: Settings) -> None:
        """Test fetch_and_store_actuals skips when actuals already exist in DB."""
        # Arrange
        mock_repo = AsyncMock()
        mock_repo.has_actuals_for_date = AsyncMock(return_value=True)

        mock_session = AsyncMock()
        sf = _make_mock_session_factory()
        ctx = AsyncMock()
        ctx.__aenter__ = AsyncMock(return_value=mock_session)
        ctx.__aexit__ = AsyncMock(return_value=False)
        sf.return_value = ctx

        with patch(
            _PATCH_REPO,
            return_value=mock_repo,
        ):
            # Act
            result = await fetch_and_store_actuals(sf, settings)

        # Assert
        assert result == 0
        mock_repo.has_actuals_for_date.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_fetches_and_stores_actuals_successfully(
        self, settings: Settings
    ) -> None:
        """Test fetch_and_store_actuals fetches weather data and stores in DB."""
        # Arrange
        weather_df = _make_weather_df("2026-03-05")

        mock_repo_check = AsyncMock()
        mock_repo_check.has_actuals_for_date = AsyncMock(return_value=False)

        mock_repo_store = AsyncMock()
        mock_repo_store.bulk_create_actuals = AsyncMock(return_value=24)

        # Two calls to sf() — first for idempotent check, second for storing
        call_count = 0
        repos = [mock_repo_check, mock_repo_store]

        def make_repo(session: object) -> AsyncMock:
            nonlocal call_count
            repo = repos[call_count]
            call_count += 1
            return repo

        mock_session_1 = AsyncMock()
        mock_session_2 = AsyncMock()

        ctx_1 = AsyncMock()
        ctx_1.__aenter__ = AsyncMock(return_value=mock_session_1)
        ctx_1.__aexit__ = AsyncMock(return_value=False)

        ctx_2 = AsyncMock()
        ctx_2.__aenter__ = AsyncMock(return_value=mock_session_2)
        ctx_2.__aexit__ = AsyncMock(return_value=False)

        sf = _make_mock_session_factory()
        sf.side_effect = [ctx_1, ctx_2]

        mock_client = MagicMock()
        mock_client.fetch_historical = MagicMock(return_value=weather_df)
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)

        with (
            patch(
                _PATCH_REPO,
                side_effect=make_repo,
            ),
            patch(
                _PATCH_CLIENT,
                return_value=mock_client,
            ),
        ):
            # Act
            result = await fetch_and_store_actuals(sf, settings)

        # Assert
        assert result == 24
        mock_repo_check.has_actuals_for_date.assert_awaited_once()
        mock_client.fetch_historical.assert_called_once()
        mock_repo_store.bulk_create_actuals.assert_awaited_once()
        mock_session_2.commit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_empty_dataframe_returns_zero(self, settings: Settings) -> None:
        """Test fetch_and_store_actuals returns 0 when OpenMeteo returns empty DataFrame."""
        # Arrange
        empty_df = pd.DataFrame()

        mock_repo = AsyncMock()
        mock_repo.has_actuals_for_date = AsyncMock(return_value=False)

        mock_session = AsyncMock()

        ctx = AsyncMock()
        ctx.__aenter__ = AsyncMock(return_value=mock_session)
        ctx.__aexit__ = AsyncMock(return_value=False)

        sf = _make_mock_session_factory()
        sf.return_value = ctx

        mock_client = MagicMock()
        mock_client.fetch_historical = MagicMock(return_value=empty_df)
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)

        with (
            patch(
                _PATCH_REPO,
                return_value=mock_repo,
            ),
            patch(
                _PATCH_CLIENT,
                return_value=mock_client,
            ),
        ):
            # Act
            result = await fetch_and_store_actuals(sf, settings)

        # Assert
        assert result == 0
        mock_client.fetch_historical.assert_called_once()

    @pytest.mark.asyncio
    async def test_openmeteo_client_receives_correct_dates(
        self, settings: Settings
    ) -> None:
        """Test fetch_and_store_actuals passes T-2 date range to OpenMeteo client."""
        # Arrange
        mock_repo_check = AsyncMock()
        mock_repo_check.has_actuals_for_date = AsyncMock(return_value=False)

        mock_repo_store = AsyncMock()
        mock_repo_store.bulk_create_actuals = AsyncMock(return_value=24)

        call_count = 0
        repos = [mock_repo_check, mock_repo_store]

        def make_repo(session: object) -> AsyncMock:
            nonlocal call_count
            repo = repos[call_count]
            call_count += 1
            return repo

        mock_session = AsyncMock()
        ctx_1 = AsyncMock()
        ctx_1.__aenter__ = AsyncMock(return_value=mock_session)
        ctx_1.__aexit__ = AsyncMock(return_value=False)
        ctx_2 = AsyncMock()
        ctx_2.__aenter__ = AsyncMock(return_value=mock_session)
        ctx_2.__aexit__ = AsyncMock(return_value=False)

        sf = _make_mock_session_factory()
        sf.side_effect = [ctx_1, ctx_2]

        weather_df = _make_weather_df("2026-03-05")
        mock_client = MagicMock()
        mock_client.fetch_historical = MagicMock(return_value=weather_df)
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)

        fixed_now = datetime(2026, 3, 7, 10, 0, 0, tzinfo=TZ)

        with (
            patch(
                _PATCH_REPO,
                side_effect=make_repo,
            ),
            patch(
                _PATCH_CLIENT,
                return_value=mock_client,
            ),
            patch(
                "energy_forecast.jobs.weather_actuals.datetime",
            ) as mock_dt,
        ):
            mock_dt.now.return_value = fixed_now
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            # Act
            result = await fetch_and_store_actuals(sf, settings)

        # Assert — T-2 from 2026-03-07 is 2026-03-05
        expected_start = "2026-03-05"
        expected_end = "2026-03-06"
        mock_client.fetch_historical.assert_called_once_with(
            start_date=expected_start,
            end_date=expected_end,
        )

    @pytest.mark.asyncio
    async def test_openmeteo_client_created_with_correct_config(
        self, settings: Settings
    ) -> None:
        """Test that OpenMeteoClient is initialized with settings.openmeteo, .region, .project.timezone."""
        # Arrange
        mock_repo = AsyncMock()
        mock_repo.has_actuals_for_date = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        ctx_1 = AsyncMock()
        ctx_1.__aenter__ = AsyncMock(return_value=mock_session)
        ctx_1.__aexit__ = AsyncMock(return_value=False)
        ctx_2 = AsyncMock()
        ctx_2.__aenter__ = AsyncMock(return_value=mock_session)
        ctx_2.__aexit__ = AsyncMock(return_value=False)

        sf = _make_mock_session_factory()
        sf.side_effect = [ctx_1, ctx_2]

        call_count = 0

        def make_repo(session: object) -> AsyncMock:
            nonlocal call_count
            r = mock_repo if call_count == 0 else AsyncMock(
                bulk_create_actuals=AsyncMock(return_value=24)
            )
            call_count += 1
            return r

        weather_df = _make_weather_df("2026-03-05")
        mock_client_cls = MagicMock()
        mock_client_instance = MagicMock()
        mock_client_instance.fetch_historical = MagicMock(return_value=weather_df)
        mock_client_instance.__enter__ = MagicMock(return_value=mock_client_instance)
        mock_client_instance.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client_instance

        with (
            patch(
                _PATCH_REPO,
                side_effect=make_repo,
            ),
            patch(
                _PATCH_CLIENT,
                mock_client_cls,
            ),
        ):
            await fetch_and_store_actuals(sf, settings)

        # Assert
        mock_client_cls.assert_called_once_with(
            config=settings.openmeteo,
            region=settings.region,
            timezone=settings.project.timezone,
        )


# ---------------------------------------------------------------------------
# TestRunScheduler
# ---------------------------------------------------------------------------


class TestRunScheduler:
    """Test suite for run_scheduler."""

    @pytest.mark.asyncio
    async def test_calls_fetch_at_correct_hour(self) -> None:
        """Test run_scheduler calls fetch_and_store_actuals when hour matches run_hour."""
        # Arrange
        sf = _make_mock_session_factory()
        mock_settings = MagicMock(spec=Settings)
        run_hour = 4

        # Simulate: first sleep wakes up, hour matches, fetch runs, then cancel
        call_count = 0

        async def fake_sleep(seconds: float) -> None:
            nonlocal call_count
            call_count += 1
            if call_count > 1:
                raise asyncio.CancelledError

        fixed_now = datetime(2026, 3, 7, run_hour, 30, 0, tzinfo=TZ)

        with (
            patch("energy_forecast.jobs.weather_actuals.asyncio.sleep", side_effect=fake_sleep),
            patch(
                "energy_forecast.jobs.weather_actuals.datetime",
            ) as mock_dt,
            patch(
                "energy_forecast.jobs.weather_actuals.fetch_and_store_actuals",
                new_callable=AsyncMock,
                return_value=24,
            ) as mock_fetch,
        ):
            mock_dt.now.return_value = fixed_now

            with pytest.raises(asyncio.CancelledError):
                await run_scheduler(sf, mock_settings, run_hour=run_hour)

        # Assert
        mock_fetch.assert_awaited_once_with(sf, mock_settings)

    @pytest.mark.asyncio
    async def test_skips_when_hour_does_not_match(self) -> None:
        """Test run_scheduler skips fetch when current hour does not match run_hour."""
        # Arrange
        sf = _make_mock_session_factory()
        mock_settings = MagicMock(spec=Settings)

        call_count = 0

        async def fake_sleep(seconds: float) -> None:
            nonlocal call_count
            call_count += 1
            if call_count > 1:
                raise asyncio.CancelledError

        # Hour is 10, but run_hour is 4 => should skip
        wrong_hour_now = datetime(2026, 3, 7, 10, 0, 0, tzinfo=TZ)

        with (
            patch("energy_forecast.jobs.weather_actuals.asyncio.sleep", side_effect=fake_sleep),
            patch(
                "energy_forecast.jobs.weather_actuals.datetime",
            ) as mock_dt,
            patch(
                "energy_forecast.jobs.weather_actuals.fetch_and_store_actuals",
                new_callable=AsyncMock,
            ) as mock_fetch,
        ):
            mock_dt.now.return_value = wrong_hour_now

            with pytest.raises(asyncio.CancelledError):
                await run_scheduler(sf, mock_settings, run_hour=4)

        # Assert — should not have been called since hour != 4
        mock_fetch.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_skips_if_already_ran_today(self) -> None:
        """Test run_scheduler skips second run on the same day."""
        # Arrange
        sf = _make_mock_session_factory()
        mock_settings = MagicMock(spec=Settings)
        run_hour = 4

        iteration = 0

        async def fake_sleep(seconds: float) -> None:
            nonlocal iteration
            iteration += 1
            if iteration > 2:
                raise asyncio.CancelledError

        # Both iterations return the same day + correct hour
        fixed_now = datetime(2026, 3, 7, run_hour, 0, 0, tzinfo=TZ)

        with (
            patch("energy_forecast.jobs.weather_actuals.asyncio.sleep", side_effect=fake_sleep),
            patch(
                "energy_forecast.jobs.weather_actuals.datetime",
            ) as mock_dt,
            patch(
                "energy_forecast.jobs.weather_actuals.fetch_and_store_actuals",
                new_callable=AsyncMock,
                return_value=24,
            ) as mock_fetch,
        ):
            mock_dt.now.return_value = fixed_now

            with pytest.raises(asyncio.CancelledError):
                await run_scheduler(sf, mock_settings, run_hour=run_hour)

        # Assert — only called once even though hour matched twice (same day guard)
        assert mock_fetch.await_count == 1

    @pytest.mark.asyncio
    async def test_exception_does_not_crash_loop(self) -> None:
        """Test run_scheduler continues after fetch_and_store_actuals raises."""
        # Arrange
        sf = _make_mock_session_factory()
        mock_settings = MagicMock(spec=Settings)
        run_hour = 4

        iteration = 0

        async def fake_sleep(seconds: float) -> None:
            nonlocal iteration
            iteration += 1
            if iteration > 2:
                raise asyncio.CancelledError

        # Two different days so the second attempt is not skipped by same-day guard
        day1 = datetime(2026, 3, 7, run_hour, 0, 0, tzinfo=TZ)
        day2 = datetime(2026, 3, 8, run_hour, 0, 0, tzinfo=TZ)

        with (
            patch("energy_forecast.jobs.weather_actuals.asyncio.sleep", side_effect=fake_sleep),
            patch(
                "energy_forecast.jobs.weather_actuals.datetime",
            ) as mock_dt,
            patch(
                "energy_forecast.jobs.weather_actuals.fetch_and_store_actuals",
                new_callable=AsyncMock,
                side_effect=[RuntimeError("DB down"), 24],
            ) as mock_fetch,
        ):
            mock_dt.now.side_effect = [day1, day2]

            with pytest.raises(asyncio.CancelledError):
                await run_scheduler(sf, mock_settings, run_hour=run_hour)

        # Assert — called twice: first fails, second succeeds, loop survived
        assert mock_fetch.await_count == 2

    @pytest.mark.asyncio
    async def test_sleep_interval_is_3600(self) -> None:
        """Test run_scheduler sleeps for 3600 seconds (1 hour) between checks."""
        sf = _make_mock_session_factory()
        mock_settings = MagicMock(spec=Settings)

        async def fake_sleep(seconds: float) -> None:
            assert seconds == 3600, f"Expected 3600s sleep, got {seconds}"
            raise asyncio.CancelledError

        with (
            patch("energy_forecast.jobs.weather_actuals.asyncio.sleep", side_effect=fake_sleep),
        ):
            with pytest.raises(asyncio.CancelledError):
                await run_scheduler(sf, mock_settings)


# ---------------------------------------------------------------------------
# TestMain
# ---------------------------------------------------------------------------


class TestMain:
    """Test suite for main() CLI entry point."""

    @staticmethod
    def _make_mock_settings(database_url: str = "") -> MagicMock:
        """Create a mock Settings with env.database_url set."""
        mock_settings = MagicMock()
        mock_settings.env.database_url = database_url
        mock_settings.database = MagicMock()
        return mock_settings

    @staticmethod
    def _close_coro(coro: object) -> None:
        """Close a coroutine to suppress 'was never awaited' warning."""
        if asyncio.iscoroutine(coro):
            coro.close()  # type: ignore[union-attr]

    def test_no_database_url_returns_early(self) -> None:
        """Test main() logs error and returns when DATABASE_URL is not set."""
        # Arrange
        mock_settings = self._make_mock_settings(database_url="")

        with (
            patch(
                _PATCH_LOAD_CONFIG,
                return_value=mock_settings,
            ),
            patch(
                _PATCH_CREATE_ENGINE,
            ) as mock_engine,
            patch(
                _PATCH_CREATE_SF,
            ) as mock_sf,
        ):
            # Act
            main()

        # Assert — engine and session factory should never be called
        mock_engine.assert_not_called()
        mock_sf.assert_not_called()

    def test_with_database_url_runs_async(self) -> None:
        """Test main() creates engine, session factory, and runs async fetch."""
        # Arrange
        mock_settings = self._make_mock_settings(
            database_url="postgresql+asyncpg://user:pass@host/db"
        )

        mock_engine = AsyncMock()
        mock_engine.dispose = AsyncMock()

        mock_sf = MagicMock(spec=async_sessionmaker)

        with (
            patch(
                _PATCH_LOAD_CONFIG,
                return_value=mock_settings,
            ),
            patch(
                _PATCH_CREATE_ENGINE,
                return_value=mock_engine,
            ),
            patch(
                _PATCH_CREATE_SF,
                return_value=mock_sf,
            ),
            patch(
                "energy_forecast.jobs.weather_actuals.asyncio.run",
                side_effect=self._close_coro,
            ) as mock_run,
        ):
            # Act
            main()

        # Assert
        mock_run.assert_called_once()
        # Verify the coroutine argument is a coroutine
        call_args = mock_run.call_args
        assert call_args is not None

    def test_create_db_engine_called_with_settings(self) -> None:
        """Test main() passes database_url and database config to create_db_engine."""
        # Arrange
        mock_settings = self._make_mock_settings(
            database_url="postgresql+asyncpg://user:pass@host/db"
        )

        mock_engine = AsyncMock()
        mock_engine.dispose = AsyncMock()

        with (
            patch(
                _PATCH_LOAD_CONFIG,
                return_value=mock_settings,
            ),
            patch(
                _PATCH_CREATE_ENGINE,
                return_value=mock_engine,
            ) as mock_create_engine,
            patch(
                _PATCH_CREATE_SF,
            ),
            patch(
                "energy_forecast.jobs.weather_actuals.asyncio.run",
                side_effect=self._close_coro,
            ),
        ):
            main()

        # Assert
        mock_create_engine.assert_called_once_with(
            mock_settings.env.database_url,
            mock_settings.database,
        )

    @pytest.mark.asyncio
    async def test_inner_run_coroutine_calls_fetch_and_dispose(self) -> None:
        """Test the _run() coroutine inside main() calls fetch and disposes engine."""
        # Arrange
        mock_settings = self._make_mock_settings(
            database_url="postgresql+asyncpg://user:pass@host/db"
        )

        mock_engine = AsyncMock()
        mock_engine.dispose = AsyncMock()

        mock_sf = MagicMock(spec=async_sessionmaker)

        captured_coro = None

        def capture_coro(coro: object) -> None:
            nonlocal captured_coro
            captured_coro = coro

        with (
            patch(
                _PATCH_LOAD_CONFIG,
                return_value=mock_settings,
            ),
            patch(
                _PATCH_CREATE_ENGINE,
                return_value=mock_engine,
            ),
            patch(
                _PATCH_CREATE_SF,
                return_value=mock_sf,
            ),
            patch(
                "energy_forecast.jobs.weather_actuals.asyncio.run",
                side_effect=capture_coro,
            ),
        ):
            main()

        # Now actually run the captured coroutine
        assert captured_coro is not None

        with patch(
            "energy_forecast.jobs.weather_actuals.fetch_and_store_actuals",
            new_callable=AsyncMock,
            return_value=10,
        ) as mock_fetch:
            await captured_coro

        # Assert
        mock_fetch.assert_awaited_once_with(mock_sf, mock_settings)
        mock_engine.dispose.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_inner_run_disposes_engine_on_exception(self) -> None:
        """Test the _run() coroutine disposes engine even when fetch raises."""
        # Arrange
        mock_settings = self._make_mock_settings(
            database_url="postgresql+asyncpg://user:pass@host/db"
        )

        mock_engine = AsyncMock()
        mock_engine.dispose = AsyncMock()

        mock_sf = MagicMock(spec=async_sessionmaker)

        captured_coro = None

        def capture_coro(coro: object) -> None:
            nonlocal captured_coro
            captured_coro = coro

        with (
            patch(
                _PATCH_LOAD_CONFIG,
                return_value=mock_settings,
            ),
            patch(
                _PATCH_CREATE_ENGINE,
                return_value=mock_engine,
            ),
            patch(
                _PATCH_CREATE_SF,
                return_value=mock_sf,
            ),
            patch(
                "energy_forecast.jobs.weather_actuals.asyncio.run",
                side_effect=capture_coro,
            ),
        ):
            main()

        assert captured_coro is not None

        with (
            patch(
                "energy_forecast.jobs.weather_actuals.fetch_and_store_actuals",
                new_callable=AsyncMock,
                side_effect=RuntimeError("fetch failed"),
            ),
            pytest.raises(RuntimeError, match="fetch failed"),
        ):
            await captured_coro

        # Assert — engine.dispose() called even on exception (finally block)
        mock_engine.dispose.assert_awaited_once()


# ---------------------------------------------------------------------------
# Integration-style: fetch_and_store_actuals with real DB fixtures
# ---------------------------------------------------------------------------


class TestFetchAndStoreActualsIntegration:
    """Integration-style tests using conftest DB fixtures (aiosqlite)."""

    @pytest.mark.asyncio
    async def test_stores_actuals_in_db(
        self, db_session_factory: async_sessionmaker, settings: Settings
    ) -> None:
        """Test full flow with real DB: mock OpenMeteo, verify rows in DB."""
        # Arrange
        weather_df = _make_weather_df("2026-03-05")

        mock_client = MagicMock()
        mock_client.fetch_historical = MagicMock(return_value=weather_df)
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)

        with patch(
            _PATCH_CLIENT,
            return_value=mock_client,
        ):
            # Act
            result = await fetch_and_store_actuals(db_session_factory, settings)

        # Assert
        assert result == 24

    @pytest.mark.asyncio
    async def test_idempotent_second_call_returns_zero(
        self, db_session_factory: async_sessionmaker, settings: Settings
    ) -> None:
        """Test calling fetch_and_store_actuals twice returns 0 on second call."""
        # Arrange
        weather_df = _make_weather_df("2026-03-05")

        mock_client = MagicMock()
        mock_client.fetch_historical = MagicMock(return_value=weather_df)
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)

        with patch(
            _PATCH_CLIENT,
            return_value=mock_client,
        ):
            # First call stores data
            first_result = await fetch_and_store_actuals(db_session_factory, settings)
            # Second call should find actuals exist and skip
            second_result = await fetch_and_store_actuals(db_session_factory, settings)

        # Assert
        assert first_result == 24
        assert second_result == 0

    @pytest.mark.asyncio
    async def test_empty_df_does_not_store(
        self, db_session_factory: async_sessionmaker, settings: Settings
    ) -> None:
        """Test empty DataFrame from OpenMeteo results in 0 rows stored."""
        # Arrange
        empty_df = pd.DataFrame()

        mock_client = MagicMock()
        mock_client.fetch_historical = MagicMock(return_value=empty_df)
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)

        with patch(
            _PATCH_CLIENT,
            return_value=mock_client,
        ):
            result = await fetch_and_store_actuals(db_session_factory, settings)

        assert result == 0
