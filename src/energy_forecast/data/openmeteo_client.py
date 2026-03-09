"""Open-Meteo weather API client using the official SDK."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import openmeteo_requests
import pandas as pd
import requests
import requests_cache
from loguru import logger
from retry_requests import retry

from energy_forecast.config import CityConfig, OpenMeteoConfig, RegionConfig
from energy_forecast.data.exceptions import OpenMeteoApiError

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

    from energy_forecast.db.sync_repos import SyncDataAccess


class OpenMeteoClient:
    """Fetches weather data from Open-Meteo API using the official SDK.

    Uses ``openmeteo-requests`` for HTTP + FlatBuffers decoding,
    ``requests-cache`` for transparent SQLite caching, and
    ``retry-requests`` for exponential-backoff retries.

    Computes weighted average across multiple city locations
    as defined in RegionConfig.

    Args:
        config: OpenMeteo configuration (URLs, variables, cache).
        region: Region configuration with city weights.
    """

    def __init__(
        self,
        config: OpenMeteoConfig,
        region: RegionConfig,
        timezone: str = "Europe/Istanbul",
        db_session: Session | None = None,
    ) -> None:
        self.config = config
        self.region = region
        self.timezone = timezone

        self._db: SyncDataAccess | None = None
        if db_session is not None:
            from energy_forecast.db.sync_repos import SyncDataAccess as SyncDA

            self._db = SyncDA(db_session)

        cache_path = Path(config.cache.path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        # Strip .db suffix — requests-cache adds its own extension
        cache_name = str(cache_path.with_suffix(""))

        cache_session = requests_cache.CachedSession(
            cache_name,
            backend=config.cache.backend,
            expire_after=config.cache.ttl_hours * 3600,
        )
        retry_session = retry(
            cache_session,
            retries=config.api.retry_attempts,
            backoff_factor=config.api.backoff_factor,
        )
        self._client = openmeteo_requests.Client(session=retry_session)

    def close(self) -> None:
        """Close the underlying session."""
        if hasattr(self._client, "_session"):
            session = self._client._session
            if hasattr(session, "close"):
                session.close()

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

        DB-first: if DB has complete data for all cities, use it.
        Otherwise fetch from API and save per-city data to DB.

        Args:
            start_date: Start date (YYYY-MM-DD).
            end_date: End date (YYYY-MM-DD).

        Returns:
            DataFrame with hourly DatetimeIndex and weather columns,
            weighted-averaged across cities.
        """
        # 1. Try DB-first
        if self._db is not None:
            city_dfs = self._load_cities_from_db(start_date, end_date, source="historical")
            if city_dfs is not None:
                logger.info("Loaded historical weather from DB ({} to {})", start_date, end_date)
                return self._compute_weighted_average(city_dfs)

        # 2. Fetch from API
        city_dfs = self._fetch_all_cities(
            url=self.config.api.base_url_historical,
            start_date=start_date,
            end_date=end_date,
        )

        # 3. Save per-city data to DB
        if self._db is not None:
            self._save_cities_to_db(city_dfs, source="historical")

        result = self._compute_weighted_average(city_dfs)
        logger.info(
            "Fetched historical weather: {} rows ({} to {})",
            len(result),
            start_date,
            end_date,
        )
        return result

    def fetch_historical_forecast(
        self,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Fetch historical forecast (reanalysis + forecast blend).

        Useful for bridging the ~5 day gap in the Historical API.

        Args:
            start_date: Start date (YYYY-MM-DD).
            end_date: End date (YYYY-MM-DD).

        Returns:
            Weighted-average weather DataFrame.
        """
        city_dfs = self._fetch_all_cities(
            url=self.config.api.base_url_historical_forecast,
            start_date=start_date,
            end_date=end_date,
        )

        # Save to DB (source=historical — same data type)
        if self._db is not None:
            self._save_cities_to_db(city_dfs, source="historical")

        result = self._compute_weighted_average(city_dfs)
        logger.info(
            "Fetched historical forecast weather: {} rows ({} to {})",
            len(result),
            start_date,
            end_date,
        )
        return result

    def fetch_forecast(self, forecast_days: int = 2) -> pd.DataFrame:
        """Fetch weather forecast for T and T+1.

        Forecast always comes from API (must be current).
        Saves per-city data to DB for audit trail.

        Args:
            forecast_days: Number of days to forecast (default 2).

        Returns:
            Weighted-average weather forecast DataFrame.
        """
        city_dfs = self._fetch_all_cities(
            url=self.config.api.base_url_forecast,
            forecast_days=forecast_days,
        )

        # Save to DB
        if self._db is not None:
            self._save_cities_to_db(city_dfs, source="forecast")

        result = self._compute_weighted_average(city_dfs)
        logger.info("Fetched weather forecast: {} rows", len(result))
        return result

    def resolve_coordinates(self, city_name: str) -> dict[str, float]:
        """Resolve city name to coordinates via Geocoding API.

        Args:
            city_name: City name to search for.

        Returns:
            Dict with ``latitude``, ``longitude``, ``elevation`` keys.

        Raises:
            OpenMeteoApiError: If geocoding fails or no results found.
        """
        geo_cfg = self.config.geocoding
        if not geo_cfg.enabled:
            msg = "Geocoding is disabled in config"
            raise OpenMeteoApiError(msg)

        params: dict[str, Any] = {
            "name": city_name,
            "count": geo_cfg.count,
            "language": geo_cfg.language,
            "format": "json",
        }

        try:
            session = self._client._session
            response = session.get(geo_cfg.api_url, params=params)
            response.raise_for_status()
            data = response.json()
        except (requests.RequestException, json.JSONDecodeError, KeyError, OSError) as exc:
            msg = f"Geocoding request failed for '{city_name}': {exc}"
            raise OpenMeteoApiError(msg) from exc

        results = data.get("results")
        if not results:
            msg = f"No geocoding results for '{city_name}'"
            raise OpenMeteoApiError(msg)

        hit = results[0]
        return {
            "latitude": float(hit["latitude"]),
            "longitude": float(hit["longitude"]),
            "elevation": float(hit.get("elevation", 0.0)),
        }

    # ------------------------------------------------------------------
    # Internal: DB + multi-city helpers
    # ------------------------------------------------------------------

    def _fetch_all_cities(
        self,
        url: str,
        start_date: str | None = None,
        end_date: str | None = None,
        forecast_days: int | None = None,
    ) -> list[tuple[CityConfig, pd.DataFrame]]:
        """Fetch weather for all cities from API."""
        city_dfs: list[tuple[CityConfig, pd.DataFrame]] = []
        for city in self.region.cities:
            df = self._fetch_single_location(
                url=url,
                latitude=city.latitude,
                longitude=city.longitude,
                start_date=start_date,
                end_date=end_date,
                forecast_days=forecast_days,
            )
            city_dfs.append((city, df))
        return city_dfs

    def _load_cities_from_db(
        self,
        start_date: str,
        end_date: str,
        source: str,
    ) -> list[tuple[CityConfig, pd.DataFrame]] | None:
        """Load per-city weather from DB. Returns None if any city has gaps."""
        if self._db is None:
            return None

        start = pd.Timestamp(start_date, tz=self.timezone)
        end = pd.Timestamp(end_date, tz=self.timezone) + pd.Timedelta(hours=23)
        expected_hours = (end - start).total_seconds() / 3600 + 1

        city_dfs: list[tuple[CityConfig, pd.DataFrame]] = []
        for city in self.region.cities:
            df = self._db.get_weather_range(start, end, city=city.name, source=source)
            if df.empty or len(df) < expected_hours * 0.9:
                return None  # Incomplete data — fall back to API
            # Set datetime index and drop non-weather columns
            if "datetime" in df.columns:
                df = df.set_index("datetime")
            df = df.drop(columns=["city", "source", "fetched_at"], errors="ignore")
            idx = pd.DatetimeIndex(df.index)
            if idx.tz is not None:
                df.index = idx.tz_localize(None)
            df.index.name = "datetime"
            city_dfs.append((city, df))
        return city_dfs

    def _save_cities_to_db(
        self,
        city_dfs: list[tuple[CityConfig, pd.DataFrame]],
        source: str,
    ) -> None:
        """Save per-city weather data to weather_cache table (non-fatal)."""
        if self._db is None:
            return
        try:
            total = 0
            for city, df in city_dfs:
                rows: list[dict[str, Any]] = []
                for idx, row in df.iterrows():
                    d: dict[str, Any] = {
                        "datetime": idx,
                        "city": city.name,
                        "source": source,
                    }
                    for col in df.columns:
                        val = row[col]
                        d[col] = float(val) if pd.notna(val) else None
                    rows.append(d)
                count = self._db.upsert_weather(rows)
                total += count
            logger.debug("Saved {} weather rows to DB (source={})", total, source)
        except Exception as e:
            logger.warning("Weather DB write failed (continuing): {}", e)

    # ------------------------------------------------------------------
    # Internal: fetch + parse
    # ------------------------------------------------------------------

    def _fetch_single_location(
        self,
        url: str,
        latitude: float,
        longitude: float,
        start_date: str | None = None,
        end_date: str | None = None,
        forecast_days: int | None = None,
    ) -> pd.DataFrame:
        """Fetch weather for a single location via the SDK.

        Args:
            url: API endpoint URL.
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
            "hourly": self.config.variables,
            "timezone": self.timezone,
        }
        if start_date and end_date:
            params["start_date"] = start_date
            params["end_date"] = end_date
        if forecast_days is not None:
            params["forecast_days"] = forecast_days

        logger.debug(
            "Fetching weather for ({:.3f}, {:.3f})",
            latitude,
            longitude,
        )
        try:
            responses = self._client.weather_api(url, params=params)
        except openmeteo_requests.OpenMeteoRequestsError as exc:
            msg = f"OpenMeteo API error: {exc}"
            raise OpenMeteoApiError(msg) from exc
        except (requests.RequestException, json.JSONDecodeError, KeyError, OSError) as exc:
            msg = f"OpenMeteo request failed: {exc}"
            raise OpenMeteoApiError(msg) from exc

        return self._parse_sdk_response(responses[0])

    def _parse_sdk_response(self, response: Any) -> pd.DataFrame:
        """Parse SDK FlatBuffers response to DataFrame.

        Args:
            response: ``WeatherApiResponse`` from the SDK.

        Returns:
            DataFrame with DatetimeIndex and weather variable columns.
        """
        hourly = response.Hourly()
        if hourly is None:
            msg = "Invalid OpenMeteo response: missing hourly data"
            raise OpenMeteoApiError(msg)

        # Build hourly time index from epoch timestamps
        time_start = hourly.Time()
        time_end = hourly.TimeEnd()
        interval = hourly.Interval()

        utc_offset = response.UtcOffsetSeconds()
        times = pd.date_range(
            start=pd.to_datetime(time_start + utc_offset, unit="s"),
            end=pd.to_datetime(time_end + utc_offset, unit="s"),
            freq=pd.Timedelta(seconds=interval),
            inclusive="left",
        )

        # Extract each variable by index (order matches config.variables)
        columns: dict[str, np.ndarray[Any, Any]] = {}
        for i, var_name in enumerate(self.config.variables):
            variable = hourly.Variables(i)
            if variable is not None:
                columns[var_name] = variable.ValuesAsNumpy()

        df = pd.DataFrame(columns, index=times)
        df.index.name = "datetime"
        return df

    # ------------------------------------------------------------------
    # Weighted average
    # ------------------------------------------------------------------

    # Columns that are categorical WMO codes — must NOT be averaged numerically.
    _CATEGORICAL_WEATHER_COLS: frozenset[str] = frozenset({"weather_code"})

    def _compute_weighted_average(
        self,
        city_dfs: list[tuple[CityConfig, pd.DataFrame]],
    ) -> pd.DataFrame:
        """Compute weighted average across city DataFrames.

        NaN-safe: when a city has missing data for a timestep, its weight
        is excluded and the remaining weights are re-normalized.  If all
        cities are NaN for a timestep, the result stays NaN.

        Categorical weather columns (e.g. ``weather_code``) are handled
        separately using the dominant-city strategy instead of averaging.

        Args:
            city_dfs: List of (CityConfig, DataFrame) pairs.

        Returns:
            Single DataFrame with weighted-average values.
        """
        if not city_dfs:
            msg = "No city DataFrames to average"
            raise OpenMeteoApiError(msg)

        base_index = city_dfs[0][1].index
        variables = list(city_dfs[0][1].columns)

        numeric_vars = [v for v in variables if v not in self._CATEGORICAL_WEATHER_COLS]
        categorical_vars = [v for v in variables if v in self._CATEGORICAL_WEATHER_COLS]

        result = pd.DataFrame(np.nan, index=base_index, columns=variables)
        result.index.name = "datetime"

        # --- Numeric variables: NaN-safe weighted average ---
        for var in numeric_vars:
            city_values = pd.DataFrame(index=base_index)
            city_weights = pd.DataFrame(index=base_index)

            for city, df in city_dfs:
                aligned = df.reindex(base_index)
                if var in aligned.columns:
                    city_values[city.name] = aligned[var]
                    city_weights[city.name] = city.weight

            valid_mask = city_values.notna()
            adj_weights = city_weights * valid_mask

            weight_sum = adj_weights.sum(axis=1).replace(0.0, np.nan)
            weighted_vals = (city_values.fillna(0.0) * adj_weights).sum(axis=1)
            result[var] = weighted_vals / weight_sum

        # --- Categorical variables: dominant city (highest weight, NaN fallback) ---
        for var in categorical_vars:
            result[var] = self._dominant_city_value(
                city_dfs, cast(pd.DatetimeIndex, base_index), var,
            )

        return result

    @staticmethod
    def _dominant_city_value(
        city_dfs: list[tuple[CityConfig, pd.DataFrame]],
        base_index: pd.DatetimeIndex,
        var: str,
    ) -> pd.Series:
        """Pick the value from the highest-weight city, with NaN fallback.

        Iterates cities in descending weight order.  For each timestep the
        first non-NaN value is used.  Result dtype is float (WMO codes are
        stored as float by OpenMeteo SDK).

        Args:
            city_dfs: City config + DataFrame pairs.
            base_index: Common datetime index.
            var: Column name to extract.

        Returns:
            Series with dominant-city values.
        """
        # Sort cities by weight descending
        sorted_pairs = sorted(city_dfs, key=lambda pair: pair[0].weight, reverse=True)

        result = pd.Series(np.nan, index=base_index, name=var)
        for _city, df in sorted_pairs:
            aligned = df.reindex(base_index)
            if var in aligned.columns:
                result = result.fillna(aligned[var])
        return result
