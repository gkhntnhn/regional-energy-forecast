"""EPIAS Transparency Platform API client."""

from __future__ import annotations

import contextlib
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import httpx
import pandas as pd
from loguru import logger
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from energy_forecast.config import EpiasApiConfig
from energy_forecast.data.exceptions import EpiasApiError, EpiasAuthError

# ---------------------------------------------------------------------------
# EPIAS variable definitions
# ---------------------------------------------------------------------------

# Real-time generation: API field name → DataFrame column name
_GENERATION_FUEL_MAP: dict[str, str] = {
    "asphaltiteCoal": "gen_asphaltite_coal",
    "biomass": "gen_biomass",
    "blackCoal": "gen_black_coal",
    "dammedHydro": "gen_dammed_hydro",
    "fueloil": "gen_fueloil",
    "geothermal": "gen_geothermal",
    "importCoal": "gen_import_coal",
    "importExport": "gen_import_export",
    "lignite": "gen_lignite",
    "lng": "gen_lng",
    "naphta": "gen_naphta",
    "naturalGas": "gen_natural_gas",
    "river": "gen_river",
    "sun": "gen_sun",
    "total": "gen_total",
    "wasteheat": "gen_wasteheat",
    "wind": "gen_wind",
}

_VARIABLE_MAP: dict[str, dict[str, Any]] = {
    "FDPP": {
        "endpoint": "/generation/data/dpp",
        "response_key": "toplam",
        "extra_params": {"region": "TR1"},
        "quarterly": True,
    },
    "Real_Time_Consumption": {
        "endpoint": "/consumption/data/realtime-consumption",
        "response_key": "consumption",
    },
    "DAM_Purchase": {
        "endpoint": "/markets/dam/data/clearing-quantity",
        "response_key": "matchedBids",
    },
    "Bilateral_Agreement_Purchase": {
        "endpoint": "/markets/bilateral-contracts/data/bilateral-contracts-bid-quantity",
        "response_key": "quantity",
    },
    "Load_Forecast": {
        "endpoint": "/consumption/data/load-estimation-plan",
        "response_key": "lep",
    },
}


class EpiasClient:
    """Fetches market data from EPIAS REST API with caching.

    Uses year-based parquet caching to avoid re-fetching historical data.
    Implements rate limiting and retry with exponential backoff.

    Args:
        username: EPIAS account username.
        password: EPIAS account password.
        config: EPIAS API configuration (URLs, timeouts, rate limits).
        variables: List of EPIAS variables to fetch. Defaults to all 5.
    """

    def __init__(
        self,
        username: str,
        password: str,
        config: EpiasApiConfig | None = None,
        variables: list[str] | None = None,
    ) -> None:
        self.username = username
        self.password = password
        self.config = config or EpiasApiConfig()
        self.cache_dir = Path(self.config.cache_dir)
        self.file_pattern = self.config.file_pattern
        self.rate_limit_seconds = self.config.rate_limit_seconds
        self.variables = variables or list(_VARIABLE_MAP.keys())
        self._token: str | None = None
        self._token_time: float = 0.0
        self._token_ttl: float = self.config.token_ttl_seconds
        self._last_request_time: float = 0.0
        self._client = httpx.Client(timeout=self.config.timeout_seconds)

    def close(self) -> None:
        """Close the HTTP client."""
        self._client.close()

    def __enter__(self) -> EpiasClient:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Authentication
    # ------------------------------------------------------------------

    def authenticate(self) -> str:
        """Get TGT token from EPIAS CAS.

        Returns cached token if still valid (< 1 hour old).

        Returns:
            TGT token string.

        Raises:
            EpiasAuthError: If authentication fails.
        """
        now = time.monotonic()
        if self._token and (now - self._token_time) < self._token_ttl:
            return self._token

        logger.debug("Authenticating with EPIAS CAS")
        try:
            response = self._client.post(
                self.config.auth_url,
                data={"username": self.username, "password": self.password},
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            msg = f"EPIAS authentication failed: {exc.response.status_code}"
            raise EpiasAuthError(msg) from exc
        except httpx.HTTPError as exc:
            msg = f"EPIAS authentication request failed: {exc}"
            raise EpiasAuthError(msg) from exc

        # TGT token is in Location header, not body
        location = response.headers.get("location", "")
        self._token = location.split("/")[-1] if location else response.text.strip()
        self._token_time = now
        logger.debug("EPIAS authentication successful")
        return self._token

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch all EPIAS variables for date range.

        Uses year-based caching: loads from parquet if available,
        otherwise fetches from API and caches.

        Args:
            start_date: Start date (YYYY-MM-DD).
            end_date: End date (YYYY-MM-DD).

        Returns:
            DataFrame with DatetimeIndex and EPIAS variable columns.
        """
        start_dt = pd.Timestamp(start_date)
        end_dt = pd.Timestamp(end_date)
        start_year = start_dt.year
        end_year = end_dt.year

        dfs: list[pd.DataFrame] = []
        for year in range(start_year, end_year + 1):
            cached = self.load_cache(year)
            if cached is not None:
                logger.debug("Loaded EPIAS cache for year {}", year)
                dfs.append(cached)
            else:
                logger.info("Fetching EPIAS data for year {} from API", year)
                year_df = self.fetch_year(year)
                dfs.append(year_df)

        if not dfs:
            msg = f"No EPIAS data available for {start_date} to {end_date}"
            raise EpiasApiError(msg)

        combined = pd.concat(dfs)
        combined = combined.sort_index()
        combined = combined[~combined.index.duplicated(keep="first")]

        # Ensure DatetimeIndex and normalize to tz-naive
        if not isinstance(combined.index, pd.DatetimeIndex):
            combined.index = pd.to_datetime(combined.index)
        if combined.index.tz is not None:
            combined.index = combined.index.tz_localize(None)

        # Filter to requested range
        mask = (combined.index >= start_dt) & (combined.index <= end_dt + pd.Timedelta(hours=23))
        return combined.loc[mask]

    def fetch_year(self, year: int) -> pd.DataFrame:
        """Fetch all variables for a full year from API and cache result.

        Args:
            year: Year to fetch (e.g. 2024).

        Returns:
            DataFrame with DatetimeIndex and EPIAS variable columns.
        """
        start = f"{year}-01-01"
        end = f"{year}-12-31"

        var_dfs: dict[str, pd.DataFrame] = {}
        for var_name in self.variables:
            if var_name not in _VARIABLE_MAP:
                logger.warning("Unknown EPIAS variable: {}", var_name)
                continue
            var_info = _VARIABLE_MAP[var_name]
            try:
                if var_info.get("quarterly"):
                    var_df = self._fetch_variable_quarterly(
                        var_info, var_name, year
                    )
                else:
                    var_df = self._fetch_variable(
                        endpoint=var_info["endpoint"],
                        response_key=var_info["response_key"],
                        column_name=var_name,
                        start_date=start,
                        end_date=end,
                        extra_params=var_info.get("extra_params"),
                    )
                var_dfs[var_name] = var_df
            except EpiasApiError:
                logger.warning("Failed to fetch EPIAS variable: {}", var_name)

        if not var_dfs:
            msg = f"Failed to fetch any EPIAS variables for year {year}"
            raise EpiasApiError(msg)

        # Filter out empty DataFrames (no rows parsed from response)
        non_empty = {k: v for k, v in var_dfs.items() if len(v) > 0}
        if not non_empty:
            msg = f"All EPIAS variables returned empty data for year {year}"
            raise EpiasApiError(msg)

        # Inner merge all variables on datetime index
        frames = list(non_empty.values())
        result = frames[0]
        for var_df in frames[1:]:
            result = result.join(var_df, how="inner")
        self.save_cache(year, result)
        return result

    # ------------------------------------------------------------------
    # Cache
    # ------------------------------------------------------------------

    def load_cache(
        self,
        year: int,
        file_pattern: str | None = None,
    ) -> pd.DataFrame | None:
        """Load cached parquet for a year.

        Args:
            year: Year to load.
            file_pattern: Override file pattern (default: market pattern).

        Returns:
            DataFrame if cache exists, None otherwise.
        """
        pattern = file_pattern or self.file_pattern
        path = self.cache_dir / pattern.format(year=year)
        if not path.exists():
            return None
        df = pd.read_parquet(path)
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"])
            df = df.set_index("datetime")
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        df.index.name = "datetime"
        return df

    def save_cache(
        self,
        year: int,
        df: pd.DataFrame,
        file_pattern: str | None = None,
    ) -> None:
        """Save year data as parquet file.

        Args:
            year: Year being cached.
            df: DataFrame to save.
            file_pattern: Override file pattern (default: market pattern).
        """
        pattern = file_pattern or self.file_pattern
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self.cache_dir / pattern.format(year=year)
        save_df = df.copy()
        save_df.index.name = "datetime"
        save_df = save_df.reset_index()
        save_df.to_parquet(path, engine="pyarrow", compression="snappy")
        logger.info("Cached EPIAS data for year {} → {}", year, path)

    # ------------------------------------------------------------------
    # Generation data (supply-side)
    # ------------------------------------------------------------------

    def fetch_generation(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch real-time generation data for date range.

        Uses year-based caching with a separate file pattern from market data.

        Args:
            start_date: Start date (YYYY-MM-DD).
            end_date: End date (YYYY-MM-DD).

        Returns:
            DataFrame with DatetimeIndex and gen_* fuel-type columns.
        """
        gen_pattern = self.config.generation_file_pattern
        start_dt = pd.Timestamp(start_date)
        end_dt = pd.Timestamp(end_date)
        start_year = start_dt.year
        end_year = end_dt.year

        dfs: list[pd.DataFrame] = []
        for year in range(start_year, end_year + 1):
            cached = self.load_cache(year, file_pattern=gen_pattern)
            if cached is not None:
                logger.debug("Loaded generation cache for year {}", year)
                dfs.append(cached)
            else:
                logger.info("Fetching generation data for year {} from API", year)
                year_df = self.fetch_generation_year(year)
                dfs.append(year_df)

        if not dfs:
            msg = f"No generation data available for {start_date} to {end_date}"
            raise EpiasApiError(msg)

        combined = pd.concat(dfs)
        combined = combined.sort_index()
        combined = combined[~combined.index.duplicated(keep="first")]

        if not isinstance(combined.index, pd.DatetimeIndex):
            combined.index = pd.to_datetime(combined.index)
        if combined.index.tz is not None:
            combined.index = combined.index.tz_localize(None)

        mask = (combined.index >= start_dt) & (
            combined.index <= end_dt + pd.Timedelta(hours=23)
        )
        return combined.loc[mask]

    def fetch_generation_year(self, year: int) -> pd.DataFrame:
        """Fetch generation data for a full year from API and cache.

        EPIAS generation endpoint limits response size, so the year
        is fetched in quarterly chunks and concatenated.

        Args:
            year: Year to fetch (e.g. 2024).

        Returns:
            DataFrame with DatetimeIndex and gen_* fuel-type columns.
        """
        quarters = [
            (f"{year}-01-01", f"{year}-03-31"),
            (f"{year}-04-01", f"{year}-06-30"),
            (f"{year}-07-01", f"{year}-09-30"),
            (f"{year}-10-01", f"{year}-12-31"),
        ]

        dfs: list[pd.DataFrame] = []
        for q_start, q_end in quarters:
            logger.info("Fetching generation Q{} {}", quarters.index((q_start, q_end)) + 1, year)
            try:
                q_df = self._fetch_generation_data(q_start, q_end)
                if not q_df.empty:
                    dfs.append(q_df)
            except EpiasApiError as exc:
                logger.warning("Generation Q fetch failed ({} to {}): {}", q_start, q_end, exc)

        if not dfs:
            msg = f"Failed to fetch any generation data for year {year}"
            raise EpiasApiError(msg)

        result = pd.concat(dfs).sort_index()
        result = result[~result.index.duplicated(keep="first")]

        gen_pattern = self.config.generation_file_pattern
        self.save_cache(year, result, file_pattern=gen_pattern)
        return result

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        retry=retry_if_exception_type(EpiasApiError),
        reraise=True,
    )
    def _fetch_generation_data(
        self,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Fetch real-time generation from EPIAS API.

        Args:
            start_date: Start date (YYYY-MM-DD).
            end_date: End date (YYYY-MM-DD).

        Returns:
            DataFrame with DatetimeIndex and gen_* columns.
        """
        token = self.authenticate()
        self._wait_rate_limit()

        url = f"{self.config.base_url}/generation/data/realtime-generation"
        start_ts = _to_epias_timestamp(start_date)
        end_ts = _to_epias_timestamp(end_date, end_of_day=True)

        body: dict[str, Any] = {"startDate": start_ts, "endDate": end_ts}
        headers = {"TGT": token, "Content-Type": "application/json"}

        logger.debug("Fetching generation data from {} to {}", start_date, end_date)

        try:
            response = self._client.post(url, json=body, headers=headers)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = ""
            with contextlib.suppress(Exception):
                detail = f" — {exc.response.text[:500]}"
            msg = f"EPIAS generation API error: {exc.response.status_code}{detail}"
            raise EpiasApiError(msg) from exc
        except httpx.HTTPError as exc:
            msg = f"EPIAS generation request failed: {exc}"
            raise EpiasApiError(msg) from exc

        data = response.json()
        items = data.get("body", {}).get("content", data.get("items", []))
        if not items:
            items = data if isinstance(data, list) else []

        if not items:
            logger.warning("No generation data returned")
            return pd.DataFrame(columns=list(_GENERATION_FUEL_MAP.values()))

        rows: list[dict[str, Any]] = []
        for item in items:
            date_val = item.get("date") or item.get("period") or item.get("time")
            if date_val is None:
                continue
            row: dict[str, Any] = {"datetime": date_val}
            for api_key, col_name in _GENERATION_FUEL_MAP.items():
                value = item.get(api_key)
                if value is not None:
                    row[col_name] = float(value)
            rows.append(row)

        if not rows:
            return pd.DataFrame(columns=list(_GENERATION_FUEL_MAP.values()))

        df = pd.DataFrame(rows)
        df["datetime"] = pd.to_datetime(df["datetime"]).dt.tz_localize(None)
        df = df.set_index("datetime").sort_index()
        df = df[~df.index.duplicated(keep="first")]
        return df

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _wait_rate_limit(self) -> None:
        """Enforce rate limiting between API requests."""
        now = time.monotonic()
        elapsed = now - self._last_request_time
        if elapsed < self.rate_limit_seconds:
            sleep_time = self.rate_limit_seconds - elapsed
            logger.debug("Rate limit: sleeping {:.1f}s", sleep_time)
            time.sleep(sleep_time)
        self._last_request_time = time.monotonic()

    def _fetch_variable_quarterly(
        self,
        var_info: dict[str, Any],
        column_name: str,
        year: int,
    ) -> pd.DataFrame:
        """Fetch a variable in quarterly chunks for endpoints with date range limits.

        Args:
            var_info: Variable mapping entry from _VARIABLE_MAP.
            column_name: Output column name.
            year: Year to fetch.

        Returns:
            DataFrame with DatetimeIndex and single value column.
        """
        quarters = [
            (f"{year}-01-01", f"{year}-03-31"),
            (f"{year}-04-01", f"{year}-06-30"),
            (f"{year}-07-01", f"{year}-09-30"),
            (f"{year}-10-01", f"{year}-12-31"),
        ]

        dfs: list[pd.DataFrame] = []
        for i, (q_start, q_end) in enumerate(quarters, 1):
            logger.info("Fetching {} Q{} {}", column_name, i, year)
            try:
                q_df = self._fetch_variable(
                    endpoint=var_info["endpoint"],
                    response_key=var_info["response_key"],
                    column_name=column_name,
                    start_date=q_start,
                    end_date=q_end,
                    extra_params=var_info.get("extra_params"),
                )
                if not q_df.empty:
                    dfs.append(q_df)
            except EpiasApiError as exc:
                logger.warning("{} Q{} fetch failed: {}", column_name, i, exc)

        if not dfs:
            msg = f"Failed to fetch any {column_name} data for year {year}"
            raise EpiasApiError(msg)

        result = pd.concat(dfs).sort_index()
        return result[~result.index.duplicated(keep="first")]

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        retry=retry_if_exception_type(EpiasApiError),
        reraise=True,
    )
    def _fetch_variable(
        self,
        endpoint: str,
        response_key: str,
        column_name: str,
        start_date: str,
        end_date: str,
        extra_params: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        """Fetch a single EPIAS variable with retry and rate limiting.

        Args:
            endpoint: API endpoint path.
            response_key: Key in response items containing the value.
            column_name: Output column name.
            start_date: Start date (YYYY-MM-DD).
            end_date: End date (YYYY-MM-DD).
            extra_params: Additional request body params (e.g. region for DPP).

        Returns:
            DataFrame with DatetimeIndex and single value column.
        """
        token = self.authenticate()
        self._wait_rate_limit()

        url = f"{self.config.base_url}{endpoint}"
        start_ts = _to_epias_timestamp(start_date)
        end_ts = _to_epias_timestamp(end_date, end_of_day=True)

        body: dict[str, Any] = {
            "startDate": start_ts,
            "endDate": end_ts,
        }
        if extra_params:
            body.update(extra_params)
        headers = {"TGT": token, "Content-Type": "application/json"}

        logger.debug("Fetching EPIAS {} from {} to {}", column_name, start_date, end_date)

        try:
            response = self._client.post(url, json=body, headers=headers)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            msg = f"EPIAS API error for {column_name}: {exc.response.status_code}"
            raise EpiasApiError(msg) from exc
        except httpx.HTTPError as exc:
            msg = f"EPIAS request failed for {column_name}: {exc}"
            raise EpiasApiError(msg) from exc

        data = response.json()
        items = data.get("body", {}).get("content", data.get("items", []))
        if not items:
            items = data if isinstance(data, list) else []

        if not items:
            logger.warning("No data returned for EPIAS {}", column_name)
            return pd.DataFrame(columns=[column_name])

        rows: list[dict[str, Any]] = []
        for item in items:
            date_val = item.get("date") or item.get("period") or item.get("time")
            value = item.get(response_key)
            if date_val is not None and value is not None:
                rows.append({"datetime": date_val, column_name: float(value)})

        if not rows:
            return pd.DataFrame(columns=[column_name])

        df = pd.DataFrame(rows)
        df["datetime"] = pd.to_datetime(df["datetime"]).dt.tz_localize(None)
        df = df.set_index("datetime").sort_index()
        df = df[~df.index.duplicated(keep="first")]
        return df


def _to_epias_timestamp(date_str: str, *, end_of_day: bool = False) -> str:
    """Convert YYYY-MM-DD to EPIAS API timestamp format.

    Args:
        date_str: Date string in YYYY-MM-DD format.
        end_of_day: If True, set time to 23:00.

    Returns:
        ISO-8601 timestamp string with +03:00 timezone.
    """
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(
        tzinfo=ZoneInfo("Europe/Istanbul"),
    )
    if end_of_day:
        dt = dt.replace(hour=23, minute=0, second=0)
    return dt.strftime("%Y-%m-%dT%H:%M:%S+03:00")
