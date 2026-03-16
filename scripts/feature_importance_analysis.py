"""CatBoost per-split feature importance analysis for pruning.

Trains CatBoost on each TSCV split (12-fold, 1000 iterations) and extracts
feature importance from every split's model. Features that are consistently
unimportant across ALL splits are safe to prune.

Usage:
    uv run python scripts/feature_importance_analysis.py
    uv run python scripts/feature_importance_analysis.py --threshold 0.05
    uv run python scripts/feature_importance_analysis.py --top 50
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from loguru import logger

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from energy_forecast.config import Settings, load_config
from energy_forecast.training.metrics import compute_all

# Log to file + stderr
_LOG_PATH = Path("data/analysis/importance_run.log")
_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
logger.add(str(_LOG_PATH), mode="w", format="{time:HH:mm:ss} | {message}")
from energy_forecast.training.splitter import TimeSeriesSplitter


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="CatBoost feature importance analysis")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help="Importance threshold. Features below this in ALL splits pruned (default: 0.05)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=50,
        help="Show top N features in summary (default: 50)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1000,
        help="CatBoost iterations per split (default: 1000)",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=12,
        help="Number of TSCV splits (default: 12)",
    )
    return parser.parse_args()


def load_data(settings: Settings) -> pd.DataFrame:
    """Load feature-engineered historical data."""
    path = Path(settings.paths.features_data)
    logger.info("Loading data from {}", path)
    df = pd.read_parquet(path)
    logger.info("Loaded {} rows × {} columns", len(df), len(df.columns))
    return df


def train_split(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    target_col: str,
    cat_cols: list[str],
    params: dict[str, object],
    early_stopping: int,
) -> tuple[CatBoostRegressor, dict[str, float]]:
    """Train CatBoost on a single split, return model and val metrics."""
    # Split X/y
    y_train = train_df[target_col]
    x_train = train_df.drop(columns=[target_col])
    y_val = val_df[target_col]
    x_val = val_df.drop(columns=[target_col])

    # Prepare categoricals
    valid_cats = [c for c in cat_cols if c in x_train.columns]
    x_train = x_train.copy()
    x_val = x_val.copy()
    for col in valid_cats:
        x_train[col] = x_train[col].fillna("missing").astype(str)
        x_val[col] = x_val[col].fillna("missing").astype(str)
    cat_indices = [x_train.columns.get_loc(c) for c in valid_cats]

    # Build pools and train
    train_pool = Pool(x_train, label=y_train, cat_features=cat_indices)
    val_pool = Pool(x_val, label=y_val, cat_features=cat_indices)

    model = CatBoostRegressor(**params, allow_writing_files=False)
    model.fit(
        train_pool,
        eval_set=val_pool,
        early_stopping_rounds=early_stopping,
        verbose=200,
    )

    # Compute metrics
    val_pred = model.predict(x_val)
    metrics = compute_all(y_val.to_numpy(), val_pred)

    return model, {
        "mape": metrics.mape,
        "rmse": metrics.rmse,
        "mae": metrics.mae,
        "best_iteration": int(model.best_iteration_),
    }


def run_analysis(args: argparse.Namespace) -> None:
    """Run full feature importance analysis across all TSCV splits."""
    settings = load_config()
    df = load_data(settings)

    target_col = settings.hyperparameters.target_col
    cat_cols = list(settings.catboost.categorical_features)

    # Fixed params for importance analysis (R1-best based)
    params: dict[str, object] = {
        "task_type": "CPU",
        "iterations": args.iterations,
        "learning_rate": 0.05,
        "depth": 6,
        "loss_function": "RMSE",
        "eval_metric": "MAPE",
        "random_seed": 42,
        "has_time": True,
        "use_best_model": True,
        "bootstrap_type": "MVS",
        "subsample": 0.75,
        "l2_leaf_reg": 1.57,
        "min_child_samples": 75,
    }

    # Create splitter with requested n_splits
    splitter = TimeSeriesSplitter(
        n_splits=args.n_splits,
        val_months=1,
        test_months=1,
    )

    # Feature names (excluding target)
    feature_names = [c for c in df.columns if c != target_col]
    n_features = len(feature_names)
    logger.info(
        "Features: {}, Splits: {}, Iterations: {}", n_features, args.n_splits, args.iterations
    )

    # Collect per-split importances
    all_importances: list[np.ndarray] = []
    split_metrics: list[dict[str, float]] = []

    start = time.monotonic()

    for fold_idx, (info, train_df, val_df, _test_df) in enumerate(splitter.iter_splits(df)):
        fold_start = time.monotonic()

        model, metrics = train_split(
            train_df, val_df, target_col, cat_cols, params, early_stopping=100
        )

        # Extract importance (PredictionValuesChange — CatBoost default)
        importance = model.get_feature_importance()
        all_importances.append(importance)
        split_metrics.append(metrics)

        fold_elapsed = time.monotonic() - fold_start
        logger.info(
            "Split {:2d}/{} | val MAPE={:.2f}% | best_iter={:4d} | {:.1f}s | train={}..{}",
            fold_idx + 1,
            args.n_splits,
            metrics["mape"],
            int(metrics["best_iteration"]),
            fold_elapsed,
            info.train_start.strftime("%Y-%m"),
            info.train_end.strftime("%Y-%m"),
        )

    total_elapsed = time.monotonic() - start
    logger.info("All {} splits done in {:.1f}s", args.n_splits, total_elapsed)

    # --- Aggregate importances ---
    imp_matrix = np.array(all_importances)  # shape: (n_splits, n_features)

    # Normalize each split to percentages (sum=100)
    imp_pct = imp_matrix / imp_matrix.sum(axis=1, keepdims=True) * 100

    # Statistics across splits
    mean_imp = imp_pct.mean(axis=0)
    std_imp = imp_pct.std(axis=0)
    min_imp = imp_pct.min(axis=0)
    max_imp = imp_pct.max(axis=0)
    median_imp = np.median(imp_pct, axis=0)

    # Build results DataFrame
    results = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_pct": mean_imp,
            "std_pct": std_imp,
            "min_pct": min_imp,
            "max_pct": max_imp,
            "median_pct": median_imp,
        }
    )
    results = results.sort_values("mean_pct", ascending=False).reset_index(drop=True)
    results["rank"] = range(1, len(results) + 1)
    results["cumulative_pct"] = results["mean_pct"].cumsum()

    # --- Classification ---
    threshold = args.threshold
    # A feature is "prunable" if its MAX importance across ALL splits < threshold
    results["prunable"] = results["max_pct"] < threshold

    n_keep = (~results["prunable"]).sum()
    n_prune = results["prunable"].sum()
    cumulative_pruned = results.loc[results["prunable"], "mean_pct"].sum()

    # --- Output ---
    logger.info("")
    logger.info("=" * 80)
    logger.info("FEATURE IMPORTANCE ANALYSIS RESULTS")
    logger.info("=" * 80)
    logger.info(
        "Threshold: {:.2f}% (features with MAX importance < {:.2f}% across ALL {} splits)",
        threshold,
        threshold,
        args.n_splits,
    )
    logger.info("KEEP: {} features  |  PRUNE: {} features", n_keep, n_prune)
    logger.info("Pruned features total importance: {:.2f}%", cumulative_pruned)
    logger.info("")

    # Cross-validation summary
    val_mapes = [m["mape"] for m in split_metrics]
    logger.info(
        "CV Summary: avg MAPE={:.2f}% ± {:.2f}% (min={:.2f}%, max={:.2f}%)",
        np.mean(val_mapes),
        np.std(val_mapes),
        np.min(val_mapes),
        np.max(val_mapes),
    )
    logger.info("")

    # Top features
    logger.info("--- TOP {} FEATURES ---", args.top)
    logger.info(
        "{:>4s}  {:<45s}  {:>8s}  {:>8s}  {:>8s}  {:>8s}  {:>8s}",
        "Rank",
        "Feature",
        "Mean%",
        "Std%",
        "Min%",
        "Max%",
        "Cum%",
    )
    for _, row in results.head(args.top).iterrows():
        logger.info(
            "{:4d}  {:<45s}  {:8.3f}  {:8.3f}  {:8.3f}  {:8.3f}  {:8.2f}",
            int(row["rank"]),
            str(row["feature"]),
            row["mean_pct"],
            row["std_pct"],
            row["min_pct"],
            row["max_pct"],
            row["cumulative_pct"],
        )

    logger.info("")

    # Prunable features
    prunable_df = results[results["prunable"]].sort_values("mean_pct", ascending=True)
    logger.info("--- PRUNABLE FEATURES ({}, max < {:.2f}%) ---", n_prune, threshold)

    # Group prunable by category
    categories = {
        "EPIAS Generation": [f for f in prunable_df["feature"] if str(f).startswith("gen_")],
        "EPIAS Market": [
            f
            for f in prunable_df["feature"]
            if any(
                str(f).startswith(p)
                for p in ["FDPP", "Real_Time", "DAM_", "Load_Forecast", "Bilateral"]
            )
        ],
        "Solar": [f for f in prunable_df["feature"] if str(f).startswith("sol_")],
        "Weather": [
            f
            for f in prunable_df["feature"]
            if any(
                str(f).startswith(p)
                for p in [
                    "wth_",
                    "temperature_",
                    "relative_",
                    "apparent_",
                    "dew_",
                    "wind_",
                    "shortwave_",
                    "surface_",
                    "snow_",
                    "precipitation",
                    "weather_",
                    "heat_index",
                    "temp_",
                ]
            )
        ],
        "Consumption": [f for f in prunable_df["feature"] if str(f).startswith("consumption_")],
        "Calendar": [
            f
            for f in prunable_df["feature"]
            if any(
                str(f).startswith(p)
                for p in [
                    "hour",
                    "day_",
                    "week_",
                    "month",
                    "quarter",
                    "year",
                    "season",
                    "is_",
                    "tatil",
                    "bayram",
                    "holiday",
                    "days_",
                    "spline_",
                ]
            )
        ],
        "Other": [],
    }
    # Collect uncategorized
    all_categorized = set()
    for feats in categories.values():
        all_categorized.update(feats)
    categories["Other"] = [f for f in prunable_df["feature"] if f not in all_categorized]

    for cat_name, feats in categories.items():
        if feats:
            logger.info(
                "  {} ({}): {}", cat_name, len(feats), ", ".join(str(f) for f in feats[:10])
            )
            if len(feats) > 10:
                logger.info("    ... and {} more", len(feats) - 10)

    # --- Save results ---
    output_dir = Path("data/analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Full results CSV
    csv_path = output_dir / "feature_importance_12fold.csv"
    results.to_csv(csv_path, index=False, float_format="%.6f")
    logger.info("")
    logger.info("Full results saved to: {}", csv_path)

    # Per-split raw importance matrix
    matrix_path = output_dir / "feature_importance_matrix.csv"
    matrix_df = pd.DataFrame(imp_pct, columns=feature_names)
    matrix_df.index.name = "split"
    matrix_df.to_csv(matrix_path, float_format="%.6f")
    logger.info("Per-split matrix saved to: {}", matrix_path)

    # Keep/prune lists as JSON
    keep_features = results[~results["prunable"]]["feature"].tolist()
    prune_features = results[results["prunable"]]["feature"].tolist()

    lists_path = output_dir / "feature_selection.json"
    lists_path.write_text(
        json.dumps(
            {
                "threshold_pct": threshold,
                "n_splits": args.n_splits,
                "iterations": args.iterations,
                "n_keep": len(keep_features),
                "n_prune": len(prune_features),
                "cv_avg_mape": float(np.mean(val_mapes)),
                "cv_std_mape": float(np.std(val_mapes)),
                "keep": keep_features,
                "prune": prune_features,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("Feature lists saved to: {}", lists_path)

    # Split metrics
    metrics_path = output_dir / "split_metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "splits": split_metrics,
                "summary": {
                    "avg_mape": float(np.mean(val_mapes)),
                    "std_mape": float(np.std(val_mapes)),
                    "min_mape": float(np.min(val_mapes)),
                    "max_mape": float(np.max(val_mapes)),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("Split metrics saved to: {}", metrics_path)

    logger.info("")
    logger.info("=" * 80)
    logger.info(
        "SUMMARY: Keep {} / Prune {} features (threshold: max < {:.2f}%)",
        n_keep,
        n_prune,
        threshold,
    )
    logger.info("=" * 80)


if __name__ == "__main__":
    run_analysis(parse_args())
