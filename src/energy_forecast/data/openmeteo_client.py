"""Open-Meteo weather API client."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
from loguru import logger
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from energy_forecast.config.settings import CityConfig, OpenMeteoConfig, RegionConfig
from energy_forecast.data.exceptions import OpenMeteoApiError


class OpenMeteoClient:
    """Fetches weather data from Open-Meteo API with caching.

    Computes weighted average across multiple city locations
    as defined in RegionConfig.

    Args:
        config: OpenMeteo configuration (URLs, variables, cache).
        region: Region configuration with city weights.
    """

    def __init__(self, config: OpenMeteoConfig, region: RegionConfig) -> None:
        self.config = config
        self.region = region
        self._client = httpx.Client(timeout=config.api.timeout)
        self._cache_db: sqlite3.Connection | None = None

    def close(self) -> None:
        """Close the HTTP client and cache connection."""
        self._client.close()
        if self._cache_db is not None:
            self._cache_db.close()
            self._cache_db = None

    def __enter__(self) -> OpenMeteoClient:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_historical(
        self,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Fetch historical weather for all cities, return weighted average.

        Args:
            start_date: Start date (YYYY-MM-DD).
            end_date: End date (YYYY-MM-DD).

        Returns:
            DataFrame with hourly DatetimeIndex and 11 weather columns,
            weighted-averaged across cities.
        """
        city_dfs: list[tuple[CityConfig, pd.DataFrame]] = []
        for city in self.region.cities:
            df = self._fetch_single_location(
                base_url=self.config.api.base_url_historical,
                latitude=city.latitude,
                longitude=city.longitude,
                start_date=start_date,
                end_date=end_date,
            )
            city_dfs.append((city, df))

        result = self._compute_weighted_average(city_dfs)
        logger.info(
            "Fetched historical weather: {} rows ({} to {})",
            len(result),
            start_date,
            end_date,
        )
        return result

    def fetch_forecast(self, forecast_days: int = 2) -> pd.DataFrame:
        """Fetch weather forecast for T and T+1.

        Args:
            forecast_days: Number of days to forecast (default 2).

        Returns:
            Weighted-average weather forecast DataFrame.
        """
        city_dfs: list[tuple[CityConfig, pd.DataFrame]] = []
        for city in self.region.cities:
            df = self._fetch_single_location(
                base_url=self.config.api.base_url_forecast,
                latitude=city.latitude,
                longitude=city.longitude,
                forecast_days=forecast_days,
            )
            city_dfs.append((city, df))

        result = self._compute_weighted_average(city_dfs)
        logger.info("Fetched weather forecast: {} rows", len(result))
        return result

    # ------------------------------------------------------------------
    # Internal: fetch + parse
    # ------------------------------------------------------------------

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(OpenMeteoApiError),
        reraise=True,
    )
    def _fetch_single_location(
        self,
        base_url: str,
        latitude: float,
        longitude: float,
        start_date: str | None = None,
        end_date: str | None = None,
        forecast_days: int | None = None,
    ) -> pd.DataFrame:
        """Fetch weather for a single location.

        Args:
            base_url: API base URL (historical or forecast).
            latitude: Location latitude.
            longitude: Location longitude.
            start_date: Start date for historical (YYYY-MM-DD).
            end_date: End date for historical (YYYY-MM-DD).
            forecast_days: Number of forecast days.

        Returns:
            DataFrame with hourly DatetimeIndex and weather columns.
        """
        params: dict[str, Any] = {
            "latitude": latitude,
            "longitude": longitude,
            "hourly": ",".join(self.config.variables),
            "timezone": "Europe/Istanbul",
        }
        if start_date and end_date:
            params["start_date"] = start_date
            params["end_date"] = end_date
        if forecast_days is not None:
            params["forecast_days"] = forecast_days

        # Check cache
        cache_key = _make_cache_key(params)
        cached = self._load_from_cache(cache_key)
        if cached is not None:
            return cached

        logger.debug(
            "Fetching weather for ({:.3f}, {:.3f})",
            latitude,
            longitude,
        )
        try:
            response = self._client.get(base_url, params=params)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            msg = f"OpenMeteo API error: {exc.response.status_code}"
            raise OpenMeteoApiError(msg) from exc
        except httpx.HTTPError as exc:
            msg = f"OpenMeteo request failed: {exc}"
            raise OpenMeteoApiError(msg) from exc

        data = response.json()
        if data.get("error"):
            msg = f"OpenMeteo API error: {data.get('reason', 'unknown')}"
            raise OpenMeteoApiError(msg)

        df = self._parse_response(data)
        self._save_to_cache(cache_key, df)
        return df

    def _parse_response(self, data: dict[str, Any]) -> pd.DataFrame:
        """Parse Open-Meteo JSON response to DataFrame.

        Args:
            data: Raw JSON response from Open-Meteo API.

        Returns:
            DataFrame with DatetimeIndex and weather variable columns.
        """
        hourly = data.get("hourly")
        if not hourly or "time" not in hourly:
            msg = "Invalid OpenMeteo response: missing 'hourly' or 'time'"
            raise OpenMeteoApiError(msg)

        times = pd.to_datetime(hourly["time"])
        columns: dict[str, list[Any]] = {}
        for var in self.config.variables:
            if var in hourly:
                columns[var] = hourly[var]

        df = pd.DataFrame(columns, index=times)
        df.index.name = "datetime"
        return df

    # ------------------------------------------------------------------
    # Weighted average
    # ------------------------------------------------------------------

    def _compute_weighted_average(
        self,
        city_dfs: list[tuple[CityConfig, pd.DataFrame]],
    ) -> pd.DataFrame:
        """Compute weighted average across city DataFrames.

        Args:
            city_dfs: List of (CityConfig, DataFrame) pairs.

        Returns:
            Single DataFrame with weighted-average values.
        """
        if not city_dfs:
            msg = "No city DataFrames to average"
            raise OpenMeteoApiError(msg)

        # Use the first city's index as base
        base_index = city_dfs[0][1].index
        variables = list(city_dfs[0][1].columns)

        result = pd.DataFrame(0.0, index=base_index, columns=variables)
        result.index.name = "datetime"

        for city, df in city_dfs:
            aligned = df.reindex(base_index)
            for var in variables:
                if var in aligned.columns:
                    result[var] = result[var] + city.weight * aligned[var].fillna(0.0)

        return result

    # ------------------------------------------------------------------
    # SQLite cache
    # ------------------------------------------------------------------

    def _get_cache_db(self) -> sqlite3.Connection:
        """Get or create SQLite cache connection."""
        if self._cache_db is not None:
            return self._cache_db

        cache_path = Path(self.config.cache.path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache_db = sqlite3.connect(str(cache_path))
        self._cache_db.execute(
            "CREATE TABLE IF NOT EXISTS weather_cache ("
            "  key TEXT PRIMARY KEY,"
            "  data TEXT NOT NULL,"
            "  created_at REAL NOT NULL"
            ")"
        )
        self._cache_db.commit()
        return self._cache_db

    def _load_from_cache(self, key: str) -> pd.DataFrame | None:
        """Load DataFrame from SQLite cache if TTL is valid."""
        ttl = self.config.cache.ttl_hours * 3600
        if ttl <= 0:
            return None

        db = self._get_cache_db()
        row = db.execute(
            "SELECT data, created_at FROM weather_cache WHERE key = ?",
            (key,),
        ).fetchone()

        if row is None:
            return None

        data_json, created_at = row
        if (time.time() - created_at) > ttl:
            db.execute("DELETE FROM weather_cache WHERE key = ?", (key,))
            db.commit()
            return None

        df: pd.DataFrame = pd.read_json(data_json, orient="split")
        df.index.name = "datetime"
        return df

    def _save_to_cache(self, key: str, df: pd.DataFrame) -> None:
        """Save DataFrame to SQLite cache."""
        ttl = self.config.cache.ttl_hours * 3600
        if ttl <= 0:
            return

        db = self._get_cache_db()
        data_json = df.to_json(orient="split", date_format="iso")
        db.execute(
            "INSERT OR REPLACE INTO weather_cache (key, data, created_at) VALUES (?, ?, ?)",
            (key, data_json, time.time()),
        )
        db.commit()


def _make_cache_key(params: dict[str, Any]) -> str:
    """Create a deterministic cache key from request parameters."""
    raw = json.dumps(params, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()
