"""A/B test: compare weather grid variants on CatBoost performance.

Variants:
  A: legacy 4 cities + ERA5/best_match (baseline)
  B: legacy 4 cities + ECMWF IFS (model consistency)
  C: grid 15 points + ECMWF IFS, no spatial features (spatial coverage)
  D: grid 15 points + ECMWF IFS + spatial features (full grid)

All variants use the same:
  - Consumption data (Excel)
  - EPIAS market data (cache)
  - CatBoost R6 best params (fixed, no HPO)
  - 12-fold TSCV, 1000 iterations, early stopping

Usage:
    uv run python scripts/ab_test_weather_grid.py
    uv run python scripts/ab_test_weather_grid.py --variants A D
    uv run python scripts/ab_test_weather_grid.py --iterations 500
"""

from __future__ import annotations

import argparse
import sys
import time
from copy import deepcopy
from pathlib import Path

import pandas as pd
from loguru import logger

# ---------------------------------------------------------------------------
# Project setup
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from energy_forecast.config import (  # noqa: E402
    OpenMeteoApiConfig,
    OpenMeteoConfig,
    RegionConfig,
    Settings,
    WeatherCacheConfig,
    load_config,
)
from energy_forecast.data.openmeteo_client import OpenMeteoClient  # noqa: E402

OUTPUT_DIR = PROJECT_ROOT / "data" / "ab_test"

# CatBoost R6 best params (fixed for all variants)
BEST_PARAMS = {
    "learning_rate": 0.11905749590096461,
    "depth": 6,
    "l2_leaf_reg": 2.0915938573471005,
    "min_child_samples": 82,
    "subsample": 0.7146458682627068,
    "loss_function": "RMSE",
}

# Spatial feature columns (dropped for variant C)
SPATIAL_COLS = [
    "wth_temp_spread",
    "wth_precip_coverage",
    "wth_temp_gradient_ns",
    "wth_pressure_spread",
]


# ---------------------------------------------------------------------------
# Variant config builders
# ---------------------------------------------------------------------------


def _build_variant_configs(
    base_settings: Settings,
) -> dict[str, tuple[OpenMeteoConfig, RegionConfig]]:
    """Build (OpenMeteoConfig, RegionConfig) pairs for each variant."""
    # Variant A: legacy + no explicit model (archive-api / best_match)
    region_legacy = RegionConfig(
        name="Uludag",
        mode="legacy",
        cities=list(base_settings.region.cities),
    )
    openmeteo_a = OpenMeteoConfig(
        api=OpenMeteoApiConfig(model=None),
        variables=list(base_settings.openmeteo.variables),
        cache=WeatherCacheConfig(
            path=str(OUTPUT_DIR / "cache_a.db"), ttl_hours=168,
        ),
    )

    # Variant B: legacy + ECMWF IFS
    openmeteo_b = OpenMeteoConfig(
        api=OpenMeteoApiConfig(model="ecmwf_ifs"),
        variables=list(base_settings.openmeteo.variables),
        cache=WeatherCacheConfig(
            path=str(OUTPUT_DIR / "cache_b.db"), ttl_hours=168,
        ),
    )

    # Variant C/D: grid + ECMWF IFS (same fetch, C drops spatial post-hoc)
    region_grid = deepcopy(base_settings.region)
    openmeteo_cd = OpenMeteoConfig(
        api=OpenMeteoApiConfig(model="ecmwf_ifs"),
        variables=list(base_settings.openmeteo.variables),
        cache=WeatherCacheConfig(
            path=str(OUTPUT_DIR / "cache_cd.db"), ttl_hours=168,
        ),
    )

    return {
        "A": (openmeteo_a, region_legacy),
        "B": (openmeteo_b, region_legacy),
        "C": (openmeteo_cd, region_grid),
        "D": (openmeteo_cd, region_grid),
    }


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------


def _load_base_data(
    settings: Settings,
) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None]:
    """Load consumption + EPIAS + generation (shared across variants)."""
    from scripts.prepare_dataset import (
        extend_with_forecast_rows,
        fetch_epias_data,
        fetch_generation_data,
        load_consumption_data,
    )

    consumption_df = load_consumption_data(settings, None)
    extended_df = extend_with_forecast_rows(consumption_df, 48)

    start_date = extended_df.index.min().strftime("%Y-%m-%d")
    end_date = extended_df.index.max().strftime("%Y-%m-%d")

    epias_df = fetch_epias_data(settings, start_date, end_date, skip_api=True)
    generation_df = fetch_generation_data(settings, start_date, end_date, skip_api=True)

    return extended_df, epias_df, generation_df


def _fetch_weather(
    openmeteo_config: OpenMeteoConfig,
    region_config: RegionConfig,
    start_date: str,
    end_date: str,
    timezone: str,
    max_retries: int = 3,
) -> pd.DataFrame | None:
    """Fetch weather data for a variant with rate limit retry."""
    for attempt in range(max_retries):
        try:
            with OpenMeteoClient(openmeteo_config, region_config, timezone) as client:
                historical_df = client.fetch_historical(start_date, end_date)
                forecast_df = client.fetch_forecast(forecast_days=3)

                if historical_df is not None and forecast_df is not None:
                    weather_df = pd.concat([historical_df, forecast_df])
                    weather_df = weather_df[~weather_df.index.duplicated(keep="last")]
                    return weather_df.sort_index()
                return historical_df if historical_df is not None else forecast_df
        except Exception as e:
            err_str = str(e)
            if "limit exceeded" in err_str.lower() and attempt < max_retries - 1:
                wait = 65 * (attempt + 1)
                logger.warning("Rate limited, waiting {}s (attempt {}/{})...",
                               wait, attempt + 1, max_retries)
                time.sleep(wait)
            else:
                logger.error("Weather fetch failed: {}", e)
                return None
    return None


def _build_features(
    settings: Settings,
    extended_df: pd.DataFrame,
    epias_df: pd.DataFrame | None,
    generation_df: pd.DataFrame | None,
    weather_df: pd.DataFrame | None,
) -> pd.DataFrame:
    """Merge data sources and run feature pipeline."""
    from energy_forecast.utils import WEATHER_FILL_PREFIXES
    from scripts.prepare_dataset import merge_data_sources, run_feature_pipeline

    merged_df = merge_data_sources(extended_df, epias_df, weather_df, generation_df)

    # Forward/back-fill weather columns only
    weather_cols = [
        c for c in merged_df.columns if c.startswith(WEATHER_FILL_PREFIXES)
    ]
    if weather_cols:
        merged_df[weather_cols] = merged_df[weather_cols].ffill().bfill()

    cat_prefixes = ("weather_code", "weather_group")
    cat_cols = [c for c in merged_df.columns if c.startswith(cat_prefixes)]
    if cat_cols:
        merged_df[cat_cols] = merged_df[cat_cols].ffill()

    features_df = run_feature_pipeline(settings, merged_df)

    # Return only historical rows (drop forecast rows)
    return features_df.iloc[:-48]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def _train_catboost(
    features_df: pd.DataFrame,
    settings: Settings,
    iterations: int = 1000,
) -> dict[str, float]:
    """Train CatBoost with fixed params, 12-fold TSCV, return avg metrics."""
    import json

    from catboost import CatBoostRegressor, Pool

    from energy_forecast.training.splitter import TimeSeriesSplitter

    target_col = "consumption"
    cat_features = list(settings.catboost.categorical_features)

    # Load selected features (229) as base set
    selected_path = PROJECT_ROOT / settings.catboost.selected_features_path
    if selected_path.exists():
        with open(selected_path) as f:
            selected_data = json.load(f)
        if isinstance(selected_data, dict) and "features" in selected_data:
            selected = selected_data["features"]
        elif isinstance(selected_data, list):
            selected = selected_data
        else:
            selected = list(selected_data.keys())
        feature_cols = [c for c in selected if c in features_df.columns]
    else:
        drop_cols = {target_col, "datetime", "date", "time"}
        feature_cols = [c for c in features_df.columns if c not in drop_cols]

    # Add spatial features if present (variant D) — append to selected set
    for sp_col in SPATIAL_COLS:
        if sp_col in features_df.columns and sp_col not in feature_cols:
            feature_cols.append(sp_col)

    logger.info("  Using {} features", len(feature_cols))

    # Filter categorical to only those in feature_cols
    cat_in_features = [c for c in cat_features if c in feature_cols]

    # Ensure categoricals are string type
    for col in cat_in_features:
        if col in features_df.columns:
            features_df[col] = features_df[col].astype(str)

    # TSCV splitter
    splitter = TimeSeriesSplitter(
        n_splits=12,
        val_months=1,
        test_months=1,
        gap_hours=0,
    )

    all_val_mape: list[float] = []
    all_test_mape: list[float] = []
    all_val_mae: list[float] = []
    all_test_mae: list[float] = []

    splits = splitter.split(features_df)
    for split_info in splits:
        i = split_info.split_idx
        train_df = features_df[split_info.train_start:split_info.train_end]
        val_df = features_df[split_info.val_start:split_info.val_end]
        test_df = features_df[split_info.test_start:split_info.test_end]

        x_train = train_df[feature_cols]
        y_train = train_df[target_col]
        x_val = val_df[feature_cols]
        y_val = val_df[target_col]
        x_test = test_df[feature_cols]
        y_test = test_df[target_col]

        model = CatBoostRegressor(
            iterations=iterations,
            **BEST_PARAMS,
            eval_metric="MAPE",
            early_stopping_rounds=100,
            has_time=True,
            random_seed=42,
            verbose=0,
            cat_features=cat_in_features,
            bootstrap_type="MVS",
        )

        train_pool = Pool(x_train, y_train, cat_features=cat_in_features)
        val_pool = Pool(x_val, y_val, cat_features=cat_in_features)

        model.fit(train_pool, eval_set=val_pool, verbose=0)

        # Metrics
        val_pred = model.predict(x_val)
        test_pred = model.predict(x_test)

        val_mape = (abs(y_val - val_pred) / y_val).mean() * 100
        test_mape = (abs(y_test - test_pred) / y_test).mean() * 100
        val_mae = abs(y_val - val_pred).mean()
        test_mae = abs(y_test - test_pred).mean()

        all_val_mape.append(val_mape)
        all_test_mape.append(test_mape)
        all_val_mae.append(val_mae)
        all_test_mae.append(test_mae)

        best_iter = model.get_best_iteration() or iterations
        logger.info(
            "  Split {:>2}/{}: val={:.2f}% test={:.2f}% iter={}",
            i, len(splits), val_mape, test_mape, best_iter,
        )

    return {
        "val_mape": sum(all_val_mape) / len(all_val_mape),
        "test_mape": sum(all_test_mape) / len(all_test_mape),
        "val_mae": sum(all_val_mae) / len(all_val_mae),
        "test_mae": sum(all_test_mae) / len(all_test_mae),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    """Run A/B test."""
    parser = argparse.ArgumentParser(description="Weather grid A/B test")
    parser.add_argument(
        "--variants", nargs="+", default=["A", "B", "C", "D"],
        choices=["A", "B", "C", "D"],
        help="Variants to test (default: all)",
    )
    parser.add_argument(
        "--iterations", type=int, default=1000,
        help="CatBoost max iterations (default: 1000)",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("WEATHER GRID A/B TEST")
    logger.info("Variants: {} | Iterations: {}", args.variants, args.iterations)
    logger.info("=" * 60)

    start_time = time.monotonic()

    # Load config and build variant configs
    settings = load_config(PROJECT_ROOT / "configs")
    variant_configs = _build_variant_configs(settings)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load base data (consumption + EPIAS) — shared across variants
    logger.info("[SHARED] Loading consumption + EPIAS data...")
    extended_df, epias_df, generation_df = _load_base_data(settings)
    start_date = extended_df.index.min().strftime("%Y-%m-%d")
    end_date = extended_df.index.max().strftime("%Y-%m-%d")
    logger.info("[SHARED] Date range: {} to {}", start_date, end_date)

    # Run each variant
    results: dict[str, dict[str, float]] = {}

    for variant in args.variants:
        logger.info("")
        logger.info("=" * 60)
        logger.info("VARIANT {} — {}", variant, {
            "A": "legacy + ERA5/best_match",
            "B": "legacy + ECMWF IFS",
            "C": "grid + ECMWF IFS (no spatial)",
            "D": "grid + ECMWF IFS + spatial",
        }[variant])
        logger.info("=" * 60)

        openmeteo_cfg, region_cfg = variant_configs[variant]

        # Fetch weather
        logger.info("[{}] Fetching weather...", variant)
        weather_df = _fetch_weather(
            openmeteo_cfg, region_cfg, start_date, end_date,
            settings.project.timezone,
        )

        if weather_df is None:
            logger.error("[{}] Weather fetch failed, skipping", variant)
            continue

        # For variant C: drop spatial columns from weather
        if variant == "C":
            spatial_in_weather = [c for c in SPATIAL_COLS if c in weather_df.columns]
            if spatial_in_weather:
                weather_df = weather_df.drop(columns=spatial_in_weather)
                logger.info("[C] Dropped {} spatial columns", len(spatial_in_weather))

        logger.info("[{}] Weather: {} rows, {} cols", variant,
                    len(weather_df), len(weather_df.columns))

        # Build features
        logger.info("[{}] Building features...", variant)
        features_df = _build_features(
            settings, extended_df.copy(), epias_df, generation_df, weather_df,
        )
        logger.info("[{}] Features: {} rows x {} cols", variant,
                    len(features_df), len(features_df.columns))

        # Save variant dataset
        variant_path = OUTPUT_DIR / f"features_variant_{variant}.parquet"
        features_df.to_parquet(variant_path, engine="pyarrow")
        logger.info("[{}] Saved: {}", variant, variant_path.name)

        # Train CatBoost
        logger.info("[{}] Training CatBoost (12-fold, {} iter)...",
                    variant, args.iterations)
        metrics = _train_catboost(features_df, settings, args.iterations)
        results[variant] = metrics

        logger.info(
            "[{}] RESULT: Val MAPE={:.2f}%  Test MAPE={:.2f}%",
            variant, metrics["val_mape"], metrics["test_mape"],
        )

    # Summary table
    elapsed = time.monotonic() - start_time
    logger.info("")
    logger.info("=" * 60)
    logger.info("A/B TEST RESULTS (elapsed: {:.0f}s)", elapsed)
    logger.info("=" * 60)
    logger.info("{:<10} {:>10} {:>10} {:>10} {:>10}",
                "Variant", "Val MAPE", "Test MAPE", "Val MAE", "Test MAE")
    logger.info("-" * 55)
    for v, m in sorted(results.items()):
        logger.info("{:<10} {:>9.2f}% {:>9.2f}% {:>10.1f} {:>10.1f}",
                    v, m["val_mape"], m["test_mape"], m["val_mae"], m["test_mae"])

    # Save results
    results_df = pd.DataFrame(results).T
    results_path = OUTPUT_DIR / "ab_test_results.csv"
    results_df.to_csv(results_path)
    logger.info("\nResults saved: {}", results_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
