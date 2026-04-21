"""Backfill weather data from ECMWF IFS for all grid points.

Fetches historical-forecast data for all 15 grid points, computes
weighted average, and saves per-year aggregated parquet files.

Output: data/external/weather_grid/ecmwf_ifs_{year}.parquet
Resume-safe: completed years are skipped via marker files.

Usage:
    uv run python scripts/backfill_weather_grid.py
    uv run python scripts/backfill_weather_grid.py --start 2024-01-01 --end 2024-12-31
    uv run python scripts/backfill_weather_grid.py --resume
    uv run python scripts/backfill_weather_grid.py --chunk-days 60
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import openmeteo_requests
import pandas as pd
import requests_cache
import yaml
from loguru import logger
from retry_requests import retry

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
GRID_POINTS_FILE = PROJECT_ROOT / "data" / "static" / "grid_points.yaml"
OUTPUT_DIR = PROJECT_ROOT / "data" / "external" / "weather_grid"

HISTORICAL_FORECAST_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
MODEL = "ecmwf_ifs"

VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "apparent_temperature",
    "precipitation",
    "snow_depth",
    "weather_code",
    "surface_pressure",
    "wind_speed_10m",
    "wind_direction_10m",
    "shortwave_radiation",
]

# Province consumption weights (must match settings.yaml)
CONSUMPTION_WEIGHTS: dict[str, float] = {
    "Bursa": 0.60,
    "Balikesir": 0.24,
    "Yalova": 0.10,
    "Canakkale": 0.06,
}


def _load_grid_points(path: Path) -> list[dict]:
    """Load grid points from YAML."""
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data["grid_points"]


def _compute_final_weights(grid_points: list[dict]) -> list[float]:
    """Compute final weight for each grid point (consumption * population)."""
    weights = []
    for gp in grid_points:
        cw = CONSUMPTION_WEIGHTS[gp["province"]]
        pw = gp["population_weight"]
        weights.append(cw * pw)
    return weights


def _create_client() -> openmeteo_requests.Client:
    """Create an OpenMeteo SDK client with cache and retry."""
    cache_path = OUTPUT_DIR / ".backfill_cache"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_session = requests_cache.CachedSession(
        str(cache_path),
        backend="sqlite",
        expire_after=86400 * 7,  # 7 day cache for backfill
    )
    retry_session = retry(cache_session, retries=3, backoff_factor=0.5)
    return openmeteo_requests.Client(session=retry_session)


def _fetch_chunk(
    client: openmeteo_requests.Client,
    grid_points: list[dict],
    start_date: str,
    end_date: str,
) -> list[pd.DataFrame]:
    """Fetch one time chunk for all grid points (batch request).

    Returns list of DataFrames, one per grid point, same order as grid_points.
    """
    lats = [gp["latitude"] for gp in grid_points]
    lons = [gp["longitude"] for gp in grid_points]

    params = {
        "latitude": lats,
        "longitude": lons,
        "hourly": VARIABLES,
        "models": MODEL,
        "timezone": "Europe/Istanbul",
        "start_date": start_date,
        "end_date": end_date,
    }

    responses = client.weather_api(HISTORICAL_FORECAST_URL, params=params)

    dfs: list[pd.DataFrame] = []
    for resp in responses:
        hourly = resp.Hourly()
        if hourly is None:
            dfs.append(pd.DataFrame())
            continue

        utc_offset = resp.UtcOffsetSeconds()
        times = pd.date_range(
            start=pd.to_datetime(hourly.Time() + utc_offset, unit="s"),
            end=pd.to_datetime(hourly.TimeEnd() + utc_offset, unit="s"),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left",
        )

        columns: dict[str, np.ndarray] = {}
        for i, var_name in enumerate(VARIABLES):
            variable = hourly.Variables(i)
            if variable is not None:
                columns[var_name] = variable.ValuesAsNumpy()

        df = pd.DataFrame(columns, index=times)
        df.index.name = "datetime"
        dfs.append(df)

    return dfs


def _weighted_average(
    point_dfs: list[pd.DataFrame],
    weights: list[float],
) -> pd.DataFrame:
    """Compute weighted average across grid point DataFrames."""
    if not point_dfs or all(df.empty for df in point_dfs):
        return pd.DataFrame()

    base_index = point_dfs[0].index
    numeric_vars = [v for v in VARIABLES if v != "weather_code"]
    result = pd.DataFrame(np.nan, index=base_index, columns=VARIABLES)
    result.index.name = "datetime"

    # Numeric: NaN-safe weighted average
    for var in numeric_vars:
        values = pd.DataFrame(index=base_index)
        w_df = pd.DataFrame(index=base_index)
        for i, (df, w) in enumerate(zip(point_dfs, weights, strict=True)):
            if var in df.columns:
                aligned = df.reindex(base_index)
                values[str(i)] = aligned[var]
                w_df[str(i)] = w

        valid = values.notna()
        adj_w = w_df * valid
        w_sum = adj_w.sum(axis=1).replace(0.0, np.nan)
        result[var] = (values.fillna(0.0) * adj_w).sum(axis=1) / w_sum

    # weather_code: dominant (highest weight, NaN fallback)
    sorted_indices = sorted(range(len(weights)), key=lambda i: weights[i], reverse=True)
    wc = pd.Series(np.nan, index=base_index, name="weather_code")
    for i in sorted_indices:
        if "weather_code" in point_dfs[i].columns:
            aligned = point_dfs[i].reindex(base_index)["weather_code"]
            wc = wc.fillna(aligned)
    result["weather_code"] = wc

    return result


def _generate_chunks(
    start: datetime, end: datetime, chunk_days: int
) -> list[tuple[str, str]]:
    """Generate (start_date, end_date) string pairs for chunked fetching."""
    chunks: list[tuple[str, str]] = []
    current = start
    while current <= end:
        chunk_end = min(current + timedelta(days=chunk_days - 1), end)
        chunks.append((current.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")))
        current = chunk_end + timedelta(days=1)
    return chunks


def main() -> None:
    """Run the backfill."""
    parser = argparse.ArgumentParser(description="Backfill ECMWF IFS weather grid data")
    parser.add_argument("--start", default="2020-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="End date (default: yesterday)")
    parser.add_argument("--chunk-days", type=int, default=90, help="Days per API chunk")
    parser.add_argument("--resume", action="store_true", help="Skip completed years")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between API calls (s)")
    args = parser.parse_args()

    if args.end is None:
        args.end = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    start = datetime.strptime(args.start, "%Y-%m-%d")
    end = datetime.strptime(args.end, "%Y-%m-%d")

    # Load grid points
    grid_points = _load_grid_points(GRID_POINTS_FILE)
    weights = _compute_final_weights(grid_points)
    logger.info("Loaded {} grid points, total weight sum: {:.4f}",
                len(grid_points), sum(weights))

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Generate time chunks
    chunks = _generate_chunks(start, end, args.chunk_days)
    logger.info("Fetching {} chunks ({} to {})", len(chunks), args.start, args.end)

    # Create client
    client = _create_client()

    # Collect data by year (aggregated + raw)
    year_agg: dict[int, list[pd.DataFrame]] = {}
    year_raw: dict[int, list[pd.DataFrame]] = {}

    for i, (chunk_start, chunk_end) in enumerate(chunks):
        # Resume: check if all years in this chunk are already complete
        chunk_start_year = int(chunk_start[:4])
        chunk_end_year = int(chunk_end[:4])
        if args.resume:
            all_done = all(
                (OUTPUT_DIR / f".completed_{y}").exists()
                for y in range(chunk_start_year, chunk_end_year + 1)
            )
            if all_done:
                logger.info("[{}/{}] Skipping {} to {} (resume)",
                            i + 1, len(chunks), chunk_start, chunk_end)
                continue

        logger.info("[{}/{}] Fetching {} to {}...",
                    i + 1, len(chunks), chunk_start, chunk_end)

        max_retries = 3
        for attempt in range(max_retries):
            try:
                point_dfs = _fetch_chunk(client, grid_points, chunk_start, chunk_end)
                agg_df = _weighted_average(point_dfs, weights)

                if agg_df.empty:
                    logger.warning("Empty result for {} to {}", chunk_start, chunk_end)
                    break

                # Build raw long-format DataFrame (all grid points)
                raw_parts: list[pd.DataFrame] = []
                for gp, pdf in zip(grid_points, point_dfs, strict=True):
                    if pdf.empty:
                        continue
                    raw = pdf.copy()
                    raw["latitude"] = gp["latitude"]
                    raw["longitude"] = gp["longitude"]
                    raw["province"] = gp["province"]
                    raw_parts.append(raw)
                raw_df = pd.concat(raw_parts) if raw_parts else pd.DataFrame()

                # Split by year and accumulate
                for year in agg_df.index.year.unique():
                    agg_slice = agg_df[agg_df.index.year == year]
                    if year not in year_agg:
                        year_agg[year] = []
                    year_agg[year].append(agg_slice)

                    if not raw_df.empty:
                        raw_slice = raw_df[raw_df.index.year == year]
                        if year not in year_raw:
                            year_raw[year] = []
                        year_raw[year].append(raw_slice)

                logger.info("  {} rows fetched, {} non-null",
                            len(agg_df), agg_df["temperature_2m"].notna().sum())
                break  # Success

            except Exception as e:
                err_str = str(e)
                if "limit exceeded" in err_str.lower():
                    wait = 65 * (attempt + 1)
                    logger.warning("Rate limited, waiting {}s (attempt {}/{})...",
                                   wait, attempt + 1, max_retries)
                    time.sleep(wait)
                else:
                    logger.error("Failed chunk {} to {}: {}",
                                 chunk_start, chunk_end, e)
                    break

        # Rate limit delay between chunks
        if i < len(chunks) - 1:
            time.sleep(args.delay)

    # Save per-year parquet files (merge with existing data)
    raw_dir = OUTPUT_DIR / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    for year in sorted(set(year_agg) | set(year_raw)):
        # --- Aggregated ---
        if year in year_agg:
            combined = pd.concat(year_agg[year]).sort_index()
            agg_path = OUTPUT_DIR / f"ecmwf_ifs_{year}.parquet"
            if agg_path.exists():
                existing = pd.read_parquet(agg_path)
                combined = pd.concat([existing, combined]).sort_index()
            combined = combined[~combined.index.duplicated(keep="last")]
            combined.to_parquet(agg_path, engine="pyarrow")
            logger.info("Saved {} ({} rows)", agg_path.name, len(combined))

        # --- Raw per-point ---
        if year in year_raw:
            raw_combined = pd.concat(year_raw[year]).sort_index()
            raw_path = raw_dir / f"ecmwf_ifs_{year}.parquet"
            if raw_path.exists():
                existing_raw = pd.read_parquet(raw_path)
                raw_combined = pd.concat([existing_raw, raw_combined]).sort_index()
            # Deduplicate on (datetime, latitude, longitude)
            raw_combined = raw_combined.reset_index()
            raw_combined = raw_combined.drop_duplicates(
                subset=["datetime", "latitude", "longitude"], keep="last",
            ).set_index("datetime").sort_index()
            raw_combined.to_parquet(raw_path, engine="pyarrow")
            logger.info("Saved raw/{} ({} rows, {} points)",
                        raw_path.name, len(raw_combined),
                        raw_combined[["latitude", "longitude"]].drop_duplicates().shape[0])

        # Write completion marker
        marker = OUTPUT_DIR / f".completed_{year}"
        marker.touch()

    # Summary
    agg_total = sum(len(pd.concat(dfs)) for dfs in year_agg.values()) if year_agg else 0
    raw_total = sum(len(pd.concat(dfs)) for dfs in year_raw.values()) if year_raw else 0
    logger.info("Backfill complete: {} years, {} agg rows, {} raw rows",
                len(year_agg), agg_total, raw_total)


if __name__ == "__main__":
    main()
