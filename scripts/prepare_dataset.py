"""Prepare feature-engineered datasets for training and prediction.

This script orchestrates the complete data preparation workflow:
1. Load consumption data from Excel
2. Extend with 48 empty forecast rows (T + T+1)
3. Fetch EPIAS market data (cache or API)
4. Fetch OpenMeteo weather data (historical + forecast)
5. Merge all data sources
6. Run feature pipeline ONCE on entire dataset
7. Split and save: features_historical.parquet + features_forecast.parquet

The key insight is that feature pipeline runs on the COMBINED dataset,
so lag/rolling/expanding features are correctly calculated even for
forecast rows (they use historical data for context).

Usage:
    python scripts/prepare_dataset.py
    python scripts/prepare_dataset.py --excel data/raw/custom.xlsx
    python scripts/prepare_dataset.py --skip-epias --skip-weather
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import timedelta
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from loguru import logger

# Load .env for credentials
load_dotenv()

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from energy_forecast.config import Settings, load_config  # noqa: E402
from energy_forecast.data.epias_client import EpiasClient  # noqa: E402
from energy_forecast.data.loader import DataLoader  # noqa: E402
from energy_forecast.data.openmeteo_client import OpenMeteoClient  # noqa: E402
from energy_forecast.features.pipeline import FeaturePipeline  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Prepare feature-engineered datasets for training and prediction.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--excel",
        type=Path,
        default=None,
        help="Path to consumption Excel file (default: from config).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Output directory for parquet files (default: data/processed).",
    )
    parser.add_argument(
        "--forecast-hours",
        type=int,
        default=48,
        help="Forecast horizon in hours (default: 48).",
    )
    parser.add_argument(
        "--skip-epias",
        action="store_true",
        help="Skip EPIAS data fetching (use cache only).",
    )
    parser.add_argument(
        "--skip-weather",
        action="store_true",
        help="Skip weather data fetching.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging.",
    )
    return parser.parse_args()


def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    logger.remove()
    level = "DEBUG" if verbose else "INFO"
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level:7}</level> | {message}",
    )


def load_consumption_data(settings: Settings, excel_path: Path | None) -> pd.DataFrame:
    """Load consumption data from Excel.

    Args:
        settings: Application settings.
        excel_path: Optional override path to Excel file.

    Returns:
        DataFrame with DatetimeIndex and consumption column.
    """
    path = excel_path or Path(settings.paths.raw_excel)
    logger.info("Loading Excel: {}", path)

    loader = DataLoader(settings.data_loader)
    df = loader.load_excel(path)

    logger.info(
        "Loaded {} rows, range {} to {}",
        len(df),
        df.index.min(),
        df.index.max(),
    )
    return df


def extend_with_forecast_rows(
    df: pd.DataFrame,
    forecast_hours: int = 48,
) -> pd.DataFrame:
    """Extend DataFrame with empty forecast rows.

    Creates rows for T (today) and T+1 (tomorrow) starting from the hour
    after the last data point. Consumption values are NaN for these rows.

    Args:
        df: DataFrame with DatetimeIndex.
        forecast_hours: Number of forecast hours (default: 48).

    Returns:
        Extended DataFrame with forecast rows appended.
    """
    last_timestamp = df.index.max()
    logger.info("Last data point: {}", last_timestamp)

    # Create forecast timestamps starting from next hour
    forecast_start = last_timestamp + timedelta(hours=1)
    forecast_index = pd.date_range(
        start=forecast_start,
        periods=forecast_hours,
        freq="h",
    )
    logger.info(
        "Creating {} forecast rows: {} to {}",
        forecast_hours,
        forecast_index.min(),
        forecast_index.max(),
    )

    # Create empty DataFrame with same columns
    forecast_df = pd.DataFrame(index=forecast_index, columns=df.columns)
    forecast_df.index.name = df.index.name

    # Concatenate
    extended = pd.concat([df, forecast_df])
    logger.info("Extended dataset: {} rows total", len(extended))

    return extended


def fetch_epias_data(
    settings: Settings,
    start_date: str,
    end_date: str,
    *,
    skip_api: bool = False,
) -> pd.DataFrame | None:
    """Fetch EPIAS market data.

    Args:
        settings: Application settings.
        start_date: Start date (YYYY-MM-DD).
        end_date: End date (YYYY-MM-DD).
        skip_api: If True, only use cache (no API calls).

    Returns:
        DataFrame with EPIAS data, or None if unavailable.
    """
    username = os.getenv("EPIAS_USERNAME", "")
    password = os.getenv("EPIAS_PASSWORD", "")

    if skip_api or not username or not password:
        logger.warning("EPIAS API skipped, using cache only")
        cache_dir = Path(settings.epias_api.cache_dir)
        start_year = int(start_date[:4])
        end_year = int(end_date[:4])

        dfs = []
        for year in range(start_year, end_year + 1):
            cache_path = cache_dir / f"{year}.parquet"
            if cache_path.exists():
                dfs.append(pd.read_parquet(cache_path))
                logger.debug("Loaded cache: {}", cache_path.name)

        if dfs:
            epias_df = pd.concat(dfs)
            if "date" in epias_df.columns:
                epias_df["date"] = pd.to_datetime(epias_df["date"])
                epias_df = epias_df.set_index("date").sort_index()
            logger.info("EPIAS cache: {} rows", len(epias_df))
            return epias_df

        logger.warning("No EPIAS cache found")
        return None

    logger.info("Fetching EPIAS data from API...")
    with EpiasClient(username, password, settings.epias_api) as client:
        epias_df = client.fetch(start_date, end_date)

    if epias_df is not None and not epias_df.empty:
        if "date" in epias_df.columns:
            epias_df = epias_df.set_index("date").sort_index()
        logger.info("EPIAS API: {} rows", len(epias_df))
        return epias_df

    logger.warning("EPIAS API returned no data")
    return None


def fetch_weather_data(
    settings: Settings,
    start_date: str,
    end_date: str,
) -> pd.DataFrame | None:
    """Fetch OpenMeteo weather data (historical + forecast).

    Args:
        settings: Application settings.
        start_date: Start date (YYYY-MM-DD).
        end_date: End date (YYYY-MM-DD).

    Returns:
        DataFrame with weather data, or None if unavailable.
    """
    logger.info("Fetching weather data...")
    try:
        with OpenMeteoClient(
            settings.openmeteo,
            settings.region,
            settings.project.timezone,
        ) as client:
            # Fetch historical
            weather_df = client.fetch_historical(start_date, end_date)

            if weather_df is not None and not weather_df.empty:
                if "date" in weather_df.columns:
                    weather_df = weather_df.set_index("date").sort_index()
                logger.info("Weather data: {} rows", len(weather_df))
                return weather_df

    except Exception as e:
        logger.error("Weather fetch failed: {}", e)

    logger.warning("Weather data unavailable")
    return None


def merge_data_sources(
    consumption_df: pd.DataFrame,
    epias_df: pd.DataFrame | None,
    weather_df: pd.DataFrame | None,
) -> pd.DataFrame:
    """Merge all data sources on DatetimeIndex.

    Args:
        consumption_df: Consumption data with DatetimeIndex.
        epias_df: Optional EPIAS data.
        weather_df: Optional weather data.

    Returns:
        Merged DataFrame.
    """
    merged = consumption_df.copy()
    logger.info("Base: {} columns", len(merged.columns))

    if epias_df is not None:
        # Remove duplicates from EPIAS before reindex
        if epias_df.index.duplicated().any():
            n_dups = epias_df.index.duplicated().sum()
            logger.warning("EPIAS has {} duplicate timestamps, keeping first", n_dups)
            epias_df = epias_df[~epias_df.index.duplicated(keep="first")]

        # Align and merge EPIAS
        epias_aligned = epias_df.reindex(merged.index)
        for col in epias_df.columns:
            if col not in merged.columns:
                merged[col] = epias_aligned[col]
        logger.info("After EPIAS merge: {} columns", len(merged.columns))

    if weather_df is not None:
        # Remove duplicates from weather before reindex
        if weather_df.index.duplicated().any():
            n_dups = weather_df.index.duplicated().sum()
            logger.warning("Weather has {} duplicate timestamps, keeping first", n_dups)
            weather_df = weather_df[~weather_df.index.duplicated(keep="first")]

        # Align and merge weather
        weather_aligned = weather_df.reindex(merged.index)
        for col in weather_df.columns:
            if col not in merged.columns:
                merged[col] = weather_aligned[col]
        logger.info("After weather merge: {} columns", len(merged.columns))

    return merged


def run_feature_pipeline(settings: Settings, df: pd.DataFrame) -> pd.DataFrame:
    """Run feature engineering pipeline.

    Args:
        settings: Application settings.
        df: Merged raw data DataFrame.

    Returns:
        Feature-engineered DataFrame.
    """
    logger.info("Running feature pipeline...")
    pipeline = FeaturePipeline(settings)
    features_df = pipeline.run(df)
    logger.info("Features: {} rows, {} columns", len(features_df), len(features_df.columns))
    return features_df


def split_and_save(
    df: pd.DataFrame,
    output_dir: Path,
    forecast_hours: int = 48,
) -> tuple[Path, Path]:
    """Split into historical and forecast, save as parquet.

    Args:
        df: Feature-engineered DataFrame.
        output_dir: Directory for output files.
        forecast_hours: Number of forecast rows at the end.

    Returns:
        Tuple of (historical_path, forecast_path).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Split
    historical_df = df.iloc[:-forecast_hours].copy()
    forecast_df = df.iloc[-forecast_hours:].copy()

    logger.info(
        "Split: historical={} rows, forecast={} rows",
        len(historical_df),
        len(forecast_df),
    )

    # Save
    historical_path = output_dir / "features_historical.parquet"
    forecast_path = output_dir / "features_forecast.parquet"

    historical_df.to_parquet(historical_path, compression="snappy")
    forecast_df.to_parquet(forecast_path, compression="snappy")

    logger.info("Saved: {}", historical_path)
    logger.info("Saved: {}", forecast_path)

    return historical_path, forecast_path


def main() -> int:
    """Main entry point."""
    args = parse_args()
    setup_logging(args.verbose)

    start_time = time.monotonic()
    logger.info("=" * 60)
    logger.info("PREPARE DATASET")
    logger.info("=" * 60)

    # Load config
    configs_dir = PROJECT_ROOT / "configs"
    settings = load_config(configs_dir)

    # Step 1: Load consumption data
    logger.info("[1/6] Loading consumption data...")
    consumption_df = load_consumption_data(settings, args.excel)

    # Step 2: Extend with forecast rows
    logger.info("[2/6] Extending with forecast rows...")
    extended_df = extend_with_forecast_rows(consumption_df, args.forecast_hours)

    # Get date range for external data
    start_date = extended_df.index.min().strftime("%Y-%m-%d")
    end_date = extended_df.index.max().strftime("%Y-%m-%d")

    # Step 3: Fetch EPIAS data
    logger.info("[3/6] Fetching EPIAS data...")
    epias_df = fetch_epias_data(
        settings, start_date, end_date, skip_api=args.skip_epias
    )

    # Step 4: Fetch weather data
    if args.skip_weather:
        logger.info("[4/6] Weather fetch skipped")
        weather_df = None
    else:
        logger.info("[4/6] Fetching weather data...")
        weather_df = fetch_weather_data(settings, start_date, end_date)

    # Step 5: Merge all data sources
    logger.info("[5/6] Merging data sources...")
    merged_df = merge_data_sources(extended_df, epias_df, weather_df)

    # Fill missing values for external data columns (forecast rows)
    # Use forward fill then backward fill
    merged_df = merged_df.ffill().bfill()

    # Step 6: Run feature pipeline
    logger.info("[6/6] Running feature pipeline...")
    features_df = run_feature_pipeline(settings, merged_df)

    # Split and save
    logger.info("=" * 60)
    logger.info("SAVING DATASETS")
    logger.info("=" * 60)
    historical_path, forecast_path = split_and_save(
        features_df, args.output_dir, args.forecast_hours
    )

    elapsed = time.monotonic() - start_time
    logger.info("=" * 60)
    logger.info("COMPLETED in {:.1f}s", elapsed)
    logger.info("  Historical: {} ({} rows)", historical_path, len(features_df) - args.forecast_hours)
    logger.info("  Forecast:   {} ({} rows)", forecast_path, args.forecast_hours)
    logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
