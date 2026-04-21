"""Seed weather_cache table from parquet files or OpenMeteo API.

Priority: parquet file (fast) -> OpenMeteo API (slow, dual-write).
Only seeds source='historical' data.

Usage:
    python scripts/seed_weather.py                            # Last 6 years
    python scripts/seed_weather.py --start 2020-01-01         # From date to today
    python scripts/seed_weather.py --start 2024-01-01 --end 2024-12-31
    python scripts/seed_weather.py --force-api                # Skip parquet, fetch from API
    python scripts/seed_weather.py --dry-run                  # Show plan, don't write
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
from dotenv import load_dotenv
from loguru import logger

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

load_dotenv()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Seed weather_cache table from parquet or OpenMeteo API.",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Start date YYYY-MM-DD (default: 2020-01-01).",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="End date YYYY-MM-DD (default: yesterday).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show plan without writing to DB.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Seconds between API requests (default: 2.0).",
    )
    parser.add_argument(
        "--force-api",
        action="store_true",
        help="Skip parquet cache, always fetch from API.",
    )
    return parser.parse_args()


def generate_monthly_chunks(
    start_date: str, end_date: str,
) -> list[tuple[str, str]]:
    """Generate monthly (start, end) date pairs."""
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    chunks: list[tuple[str, str]] = []

    current = start
    while current <= end:
        month_end = (current + pd.offsets.MonthEnd(1))
        chunk_end = min(month_end, end)
        chunks.append((
            current.strftime("%Y-%m-%d"),
            chunk_end.strftime("%Y-%m-%d"),
        ))
        current = chunk_end + timedelta(days=1)
    return chunks


def _load_year_from_parquet(
    parquet_dir: Path,
    file_pattern: str,
    year: int,
) -> pd.DataFrame | None:
    """Load a year's weather data from parquet file.

    Returns DataFrame with columns: datetime, city, source, + weather vars.
    Returns None if file doesn't exist.
    """
    path = parquet_dir / file_pattern.format(year=year)
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    logger.info("[PARQUET READ] Weather {} from {} ({} rows)", year, path, len(df))
    return df


def _parquet_to_db_rows(df: pd.DataFrame) -> list[dict[str, object]]:
    """Convert parquet DataFrame to list of dicts for DB upsert."""
    rows: list[dict[str, object]] = []
    for _, row in df.iterrows():
        d: dict[str, object] = {}
        for col in df.columns:
            val = row[col]
            if col == "datetime":
                d[col] = val
            elif col in ("city", "source"):
                d[col] = str(val)
            elif pd.notna(val):
                d[col] = float(val)
            else:
                d[col] = None
        rows.append(d)
    return rows


def _save_year_to_parquet(
    city_dfs: list[tuple[object, pd.DataFrame]],
    year: int,
    parquet_dir: Path,
    file_pattern: str,
) -> None:
    """Save per-city API data to yearly parquet file."""
    all_rows: list[dict[str, object]] = []
    for city, df in city_dfs:
        for idx, row in df.iterrows():
            d: dict[str, object] = {
                "datetime": idx,
                "city": city.name,  # type: ignore[union-attr]
                "source": "historical",
            }
            for col in df.columns:
                val = row[col]
                d[col] = float(val) if pd.notna(val) else None
            all_rows.append(d)

    if not all_rows:
        return

    out_df = pd.DataFrame(all_rows)
    if "datetime" in out_df.columns:
        out_df["datetime"] = pd.to_datetime(out_df["datetime"]).dt.tz_localize(None)

    parquet_dir.mkdir(parents=True, exist_ok=True)
    path = parquet_dir / file_pattern.format(year=year)
    out_df.to_parquet(path, engine="pyarrow", compression="snappy", index=False)
    logger.info("[PARQUET WRITE] Weather {} -> {} ({} rows)", year, path, len(out_df))


def main() -> int:
    """Main entry point."""
    args = parse_args()

    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level:7}</level> | {message}",
    )

    # Default date range
    if args.start is None:
        start_date = "2020-01-01"
    else:
        start_date = args.start

    if args.end is None:
        end_date = (
            datetime.now(tz=ZoneInfo("Europe/Istanbul")) - timedelta(days=1)
        ).strftime("%Y-%m-%d")
    else:
        end_date = args.end

    logger.info("Weather seed: {} to {}", start_date, end_date)

    # Determine years in range
    start_year = int(start_date[:4])
    end_year = int(end_date[:4])
    years = list(range(start_year, end_year + 1))
    logger.info("Year range: {} ({} years)", years, len(years))

    # Connect to DB
    db_url = os.getenv("DATABASE_URL_SYNC")
    if not db_url:
        logger.error("DATABASE_URL_SYNC not set. Cannot seed without database.")
        return 1

    from energy_forecast.config import load_config
    from energy_forecast.data.openmeteo_client import OpenMeteoClient
    from energy_forecast.db.engine import create_sync_engine, create_sync_session_factory
    from energy_forecast.db.sync_repos import SyncDataAccess

    configs_dir = PROJECT_ROOT / "configs"
    settings = load_config(configs_dir)

    parquet_dir = Path(settings.openmeteo.cache.parquet_dir)
    file_pattern = settings.openmeteo.cache.file_pattern

    if args.dry_run:
        for yr in years:
            pq_path = parquet_dir / file_pattern.format(year=yr)
            source = "parquet" if pq_path.exists() and not args.force_api else "API"
            logger.info("  {} -> source: {} ({})", yr, source, pq_path)
        logger.info("Dry run -- no DB writes.")
        return 0

    engine = create_sync_engine(db_url)
    factory = create_sync_session_factory(engine)
    session = factory()
    dao = SyncDataAccess(session)

    total_rows = 0
    start_time = time.monotonic()
    parquet_loaded = 0
    api_fetched = 0

    try:
        for yr in years:
            # --- Strategy 1: Load from parquet ---
            if not args.force_api:
                df = _load_year_from_parquet(parquet_dir, file_pattern, yr)
                if df is not None:
                    rows = _parquet_to_db_rows(df)
                    # Filter rows within requested date range
                    rows = [
                        r for r in rows
                        if start_date <= pd.Timestamp(r["datetime"]).strftime("%Y-%m-%d") <= end_date
                    ]
                    if rows:
                        count = dao.upsert_weather(rows)
                        session.commit()
                        total_rows += count
                        parquet_loaded += 1
                        logger.info("  {} -> DB from parquet ({} rows)", yr, count)
                    continue

            # --- Strategy 2: Fetch from API (monthly chunks) ---
            logger.info("  {} -> Fetching from API...", yr)
            yr_start = max(start_date, f"{yr}-01-01")
            yr_end = min(end_date, f"{yr}-12-31")
            chunks = generate_monthly_chunks(yr_start, yr_end)

            # Collect all city_dfs for this year (for parquet dual-write)
            year_city_data: dict[str, list[tuple[object, pd.DataFrame]]] = {}

            for j, (chunk_start, chunk_end) in enumerate(chunks, 1):
                logger.info("    [{}/{}] {} to {}", j, len(chunks), chunk_start, chunk_end)

                with OpenMeteoClient(
                    config=settings.openmeteo,
                    region=settings.region,
                    timezone=settings.project.timezone,
                ) as client:
                    city_dfs = client.fetch_all_cities(
                        url=settings.openmeteo.api.base_url_historical,
                        start_date=chunk_start,
                        end_date=chunk_end,
                    )

                # Write per-city data to DB
                chunk_rows = 0
                for city, df in city_dfs:
                    rows_list: list[dict[str, object]] = []
                    for idx, row in df.iterrows():
                        d: dict[str, object] = {
                            "datetime": idx,
                            "city": city.name,
                            "source": "historical",
                        }
                        for col in df.columns:
                            val = row[col]
                            d[col] = float(val) if pd.notna(val) else None
                        rows_list.append(d)
                    count = dao.upsert_weather(rows_list)
                    chunk_rows += count

                    # Accumulate for parquet write
                    if city.name not in year_city_data:
                        year_city_data[city.name] = []
                    year_city_data[city.name].append((city, df))

                session.commit()
                total_rows += chunk_rows
                logger.info("    Written {} rows", chunk_rows)

                # Rate limit
                if j < len(chunks):
                    time.sleep(args.delay)

            # Dual-write: save this year to parquet
            all_city_dfs: list[tuple[object, pd.DataFrame]] = []
            for city_name, pairs in year_city_data.items():
                for pair in pairs:
                    all_city_dfs.append(pair)
            if all_city_dfs:
                _save_year_to_parquet(all_city_dfs, yr, parquet_dir, file_pattern)

            api_fetched += 1

    except Exception as e:
        logger.error("Seed failed: {}", e)
        session.rollback()
        return 1

    # --- Grid mode: seed per-point raw data from backfill parquets ---
    grid_raw_dir = PROJECT_ROOT / "data" / "external" / "weather_grid" / "raw"
    if grid_raw_dir.exists() and settings.region.mode == "grid":
        logger.info("Seeding grid weather from backfill parquets...")
        grid_rows = 0
        for yr in years:
            grid_path = grid_raw_dir / f"ecmwf_ifs_{yr}.parquet"
            if not grid_path.exists():
                continue
            gdf = pd.read_parquet(grid_path)
            # Raw parquet has: datetime(index), lat, lon, province, + weather vars
            weather_vars = [
                c for c in gdf.columns
                if c not in ("latitude", "longitude", "province")
            ]
            rows_list: list[dict[str, object]] = []
            for idx, row in gdf.iterrows():
                loc_name = f"{row['province']}_{row['latitude']}_{row['longitude']}"
                d: dict[str, object] = {
                    "datetime": idx,
                    "city": loc_name,
                    "source": "historical",
                }
                for col in weather_vars:
                    val = row[col]
                    d[col] = float(val) if pd.notna(val) else None
                rows_list.append(d)

            if rows_list:
                count = dao.upsert_weather(rows_list)
                session.commit()
                grid_rows += count
                logger.info("  {} -> grid DB ({} rows, {} points)",
                            yr, count,
                            gdf[["latitude", "longitude"]].drop_duplicates().shape[0])

        total_rows += grid_rows
        logger.info("Grid weather seed: {} rows", grid_rows)

    session.close()

    elapsed = time.monotonic() - start_time
    logger.info(
        "Weather seed complete: {} rows in {:.1f}s "
        "(parquet: {} years, API: {} years)",
        total_rows, elapsed, parquet_loaded, api_fetched,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
