"""Comprehensive ensemble model debug analysis."""

import json
import os
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Suppress loguru
from loguru import logger

logger.remove()

BASE = Path(r"C:/Users/pc/Desktop/Python/Projects/regional-energy-forecast")

print("=" * 80)
print("   ENSEMBLE MODEL DEBUG ANALYSIS REPORT")
print("   Project: regional-energy-forecast")
print("=" * 80)

# ========================================================================
# SECTION 1: ARTIFACT CHECK
# ========================================================================
print("\n" + "=" * 80)
print("  SECTION 1: ARTIFACT CHECK")
print("=" * 80)

weights_path = BASE / "models" / "ensemble_weights.json"
cb_path = BASE / "models" / "catboost" / "model.cbm"
prophet_path = BASE / "models" / "prophet" / "model.pkl"
tft_path = BASE / "models" / "tft" / "tft_model.ckpt"
tft_meta_path = BASE / "models" / "tft" / "metadata.json"
tft_dataset_path = BASE / "models" / "tft" / "dataset_params.json"

artifacts = {
    "ensemble_weights.json": weights_path,
    "catboost/model.cbm": cb_path,
    "prophet/model.pkl": prophet_path,
    "tft/tft_model.ckpt": tft_path,
    "tft/metadata.json": tft_meta_path,
    "tft/dataset_params.json": tft_dataset_path,
}

for name, path in artifacts.items():
    exists = path.exists()
    size = path.stat().st_size if exists else 0
    size_str = (
        f"{size / 1024:.1f} KB" if size < 1024 * 1024 else f"{size / (1024*1024):.1f} MB"
    )
    status = f"OK ({size_str})" if exists else "MISSING"
    mark = "v" if exists else "x"
    print(f"  [{mark}] {name:35s} {status}")

# Load weights
with open(weights_path) as f:
    weights = json.load(f)
print(f"\n  Ensemble Weights:")
for model_name, w in weights.items():
    print(f"    {model_name:12s}: {w:.4f} ({w*100:.1f}%)")
print(f"    {'Sum':12s}: {sum(weights.values()):.4f}")

# ========================================================================
# SECTION 2: LOAD DATA AND MODELS
# ========================================================================
print("\n" + "=" * 80)
print("  SECTION 2: LOAD DATA AND MODELS")
print("=" * 80)

# Load historical data
hist_path = BASE / "data" / "processed" / "features_historical.parquet"
df = pd.read_parquet(hist_path)
print(f"\n  Historical data: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"  Date range: {df.index.min()} to {df.index.max()}")
print(
    f"  Target (consumption): mean={df['consumption'].mean():.1f}, std={df['consumption'].std():.1f}"
)
print(f"  Missing consumption: {df['consumption'].isna().sum()} rows")

# Split: last 2 months as test
test_start = df.index.max() - pd.Timedelta(days=60)
train_df = df[df.index < test_start].copy()
test_df = df[df.index >= test_start].copy()

# Remove rows with NaN consumption from test set
test_df = test_df.dropna(subset=["consumption"])
train_df_clean = train_df.dropna(subset=["consumption"])

print(
    f"\n  Train set: {train_df_clean.shape[0]} rows ({train_df_clean.index.min()} to {train_df_clean.index.max()})"
)
print(
    f"  Test set:  {test_df.shape[0]} rows ({test_df.index.min()} to {test_df.index.max()})"
)

y_test = test_df["consumption"].values.astype(np.float64)
y_train = train_df_clean["consumption"].values.astype(np.float64)

# ----- CatBoost -----
print("\n  --- Loading CatBoost ---")
cb_predictions_test = None
cb_predictions_train = None
cb_model = None
cb_feature_names = None
try:
    from catboost import CatBoostRegressor

    cb_model = CatBoostRegressor()
    cb_model.load_model(str(cb_path))
    cb_feature_names = cb_model.feature_names_
    print(f"  CatBoost loaded. Features: {len(cb_feature_names)}")

    # Prepare features (drop target)
    cat_cols = [
        "hour",
        "day_of_week",
        "month",
        "is_holiday",
        "is_weekend",
        "is_ramadan",
        "bayram_gun_no",
        "weather_code",
        "season",
    ]

    X_test = test_df.drop(columns=["consumption"])
    X_train = train_df_clean.drop(columns=["consumption"])

    # Align columns to model features
    missing_cols = [c for c in cb_feature_names if c not in X_test.columns]
    if missing_cols:
        print(f"  WARNING: Missing features in data: {missing_cols[:10]}...")
    extra_cols = [c for c in X_test.columns if c not in cb_feature_names]
    if extra_cols:
        print(f"  Extra features in data (not in model): {len(extra_cols)}")

    X_test = X_test[[c for c in cb_feature_names if c in X_test.columns]]
    X_train = X_train[[c for c in cb_feature_names if c in X_train.columns]]

    # Handle categoricals
    for col in cat_cols:
        if col in X_test.columns:
            X_test[col] = X_test[col].fillna("missing").astype(str)
            X_train[col] = X_train[col].fillna("missing").astype(str)

    cb_predictions_test = cb_model.predict(X_test)
    cb_predictions_train = cb_model.predict(X_train)
    print(
        f"  CatBoost predictions: test={len(cb_predictions_test)}, train={len(cb_predictions_train)}"
    )
except Exception as e:
    print(f"  CatBoost FAILED: {e}")

# ----- Prophet -----
print("\n  --- Loading Prophet ---")
prophet_predictions_test = None
prophet_predictions_train = None
try:
    with open(prophet_path, "rb") as f:
        prophet_model = pickle.load(f)

    regressor_names = ["temperature_2m", "relative_humidity_2m"]

    def make_prophet_df(data_df: pd.DataFrame) -> pd.DataFrame:
        pdf = pd.DataFrame()
        pdf["ds"] = data_df.index
        for reg in regressor_names:
            if reg in data_df.columns:
                pdf[reg] = data_df[reg].values
            else:
                pdf[reg] = 0.0
        return pdf

    test_prophet = make_prophet_df(test_df)
    train_prophet = make_prophet_df(train_df_clean)

    # Fill NaN in regressors
    for reg in regressor_names:
        test_prophet[reg] = test_prophet[reg].ffill().fillna(0)
        train_prophet[reg] = train_prophet[reg].ffill().fillna(0)

    forecast_test = prophet_model.predict(test_prophet)
    forecast_train = prophet_model.predict(train_prophet)

    prophet_predictions_test = forecast_test["yhat"].values.astype(np.float64)
    prophet_predictions_train = forecast_train["yhat"].values.astype(np.float64)
    print(
        f"  Prophet predictions: test={len(prophet_predictions_test)}, train={len(prophet_predictions_train)}"
    )
except Exception as e:
    print(f"  Prophet FAILED: {e}")

# ----- TFT -----
print("\n  --- Loading TFT ---")
tft_predictions_test = None
tft_loaded = False
try:
    with open(tft_dataset_path) as f:
        ds_params = json.load(f)
    with open(tft_meta_path) as f:
        meta = json.load(f)

    enc_len = meta["encoder_length"]  # 168
    pred_len = meta["prediction_length"]  # 48

    print(
        f"  TFT metadata: hidden_size={meta['hidden_size']}, encoder={enc_len}, pred={pred_len}"
    )
    print(f"  TFT known reals: {ds_params['time_varying_known_reals']}")

    # TFT requires special dataset setup - attempt loading
    try:
        from energy_forecast.models.tft import TFTForecaster

        forecaster = TFTForecaster.from_checkpoint(BASE / "models" / "tft")

        # Rolling prediction: prepend encoder context from training data
        context_start = test_start - pd.Timedelta(hours=enc_len)
        context_df = df[(df.index >= context_start) & (df.index < test_start)].copy()
        full_df = pd.concat([context_df, test_df]).sort_index()
        full_df = full_df[~full_df.index.duplicated(keep="last")]

        print(
            f"  Rolling prediction: context={len(context_df)} + test={len(test_df)} = {len(full_df)} rows"
        )

        tft_pred_df = forecaster.predict_rolling(full_df, target_col="consumption")
        # Align to test set index
        tft_pred_df = tft_pred_df.reindex(test_df.index)
        tft_predictions_test = tft_pred_df["yhat"].values.astype(np.float64)
        tft_loaded = True
        print(f"  TFT predictions: test={len(tft_predictions_test)}")
    except Exception as e2:
        print(
            f"  TFT predict FAILED: {type(e2).__name__}: {e2}"
        )
        print(f"  TFT will be skipped in ensemble analysis.")
except Exception as e:
    print(f"  TFT FAILED: {type(e).__name__}: {e}")

# ========================================================================
# SECTION 3: PER-MODEL METRICS
# ========================================================================
print("\n" + "=" * 80)
print("  SECTION 3: PER-MODEL METRICS (Test Set - Last 2 Months)")
print("=" * 80)


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> dict[str, float]:
    mask = y_true != 0
    mape_val = float(
        np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    )
    mae_val = float(np.mean(np.abs(y_true - y_pred)))
    rmse_val = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2_val = 1.0 - ss_res / ss_tot if ss_tot != 0 else 0.0
    mbe_val = float(np.mean(y_pred - y_true))
    return {
        "MAPE": mape_val,
        "MAE": mae_val,
        "RMSE": rmse_val,
        "R2": r2_val,
        "MBE": mbe_val,
    }


header = f"\n  {'Model':15s} {'MAPE%':>8s} {'MAE':>10s} {'RMSE':>10s} {'R2':>8s} {'MBE':>10s}"
print(header)
print("  " + "-" * 63)

model_metrics: dict[str, dict[str, float]] = {}
predictions: dict[str, np.ndarray] = {}

if cb_predictions_test is not None:
    m = compute_metrics(y_test, cb_predictions_test)
    model_metrics["catboost"] = m
    predictions["catboost"] = cb_predictions_test
    print(
        f"  {'CatBoost':15s} {m['MAPE']:8.2f} {m['MAE']:10.1f} {m['RMSE']:10.1f} {m['R2']:8.4f} {m['MBE']:10.1f}"
    )

if prophet_predictions_test is not None:
    m = compute_metrics(y_test, prophet_predictions_test)
    model_metrics["prophet"] = m
    predictions["prophet"] = prophet_predictions_test
    print(
        f"  {'Prophet':15s} {m['MAPE']:8.2f} {m['MAE']:10.1f} {m['RMSE']:10.1f} {m['R2']:8.4f} {m['MBE']:10.1f}"
    )

if tft_predictions_test is not None:
    # Handle any NaN values from reindexing (edges of rolling window)
    tft_valid_mask = ~np.isnan(tft_predictions_test)
    n_valid = int(tft_valid_mask.sum())

    if n_valid < len(y_test):
        # Partial coverage: evaluate only valid predictions
        y_test_tft = y_test[tft_valid_mask]
        tft_preds_valid = tft_predictions_test[tft_valid_mask]
        m = compute_metrics(y_test_tft, tft_preds_valid)
        model_metrics["tft"] = m
        # Fill NaN with mean of valid predictions for ensemble participation
        tft_filled = tft_predictions_test.copy()
        tft_filled[~tft_valid_mask] = np.nanmean(tft_predictions_test)
        predictions["tft"] = tft_filled
        print(
            f"  {'TFT':15s} {m['MAPE']:8.2f} {m['MAE']:10.1f} {m['RMSE']:10.1f} {m['R2']:8.4f} {m['MBE']:10.1f}  ({n_valid}/{len(y_test)} valid)"
        )
    else:
        m = compute_metrics(y_test, tft_predictions_test)
        model_metrics["tft"] = m
        predictions["tft"] = tft_predictions_test
        print(
            f"  {'TFT':15s} {m['MAPE']:8.2f} {m['MAE']:10.1f} {m['RMSE']:10.1f} {m['R2']:8.4f} {m['MBE']:10.1f}"
        )

# Ensemble prediction (using available models with matching length)
ensemble_pred = np.zeros(len(y_test))
total_weight = 0.0
for mn, preds in predictions.items():
    w = weights.get(mn, 0.0)
    ensemble_pred += w * preds
    total_weight += w

if total_weight > 0:
    ensemble_pred /= total_weight  # Renormalize
    m = compute_metrics(y_test, ensemble_pred)
    model_metrics["ensemble"] = m
    predictions["ensemble"] = ensemble_pred
    print("  " + "-" * 63)

    active_models = [k for k in predictions.keys() if k != "ensemble"]
    active_weights = {k: weights.get(k, 0.0) / total_weight for k in active_models}
    print(
        f"  {'ENSEMBLE':15s} {m['MAPE']:8.2f} {m['MAE']:10.1f} {m['RMSE']:10.1f} {m['R2']:8.4f} {m['MBE']:10.1f}"
    )
    print(
        f"  (Active models: {active_models}, renormalized weights: { {k: round(v, 3) for k, v in active_weights.items()} })"
    )

# ========================================================================
# SECTION 4: RESIDUAL ANALYSIS (Ensemble)
# ========================================================================
print("\n" + "=" * 80)
print("  SECTION 4: RESIDUAL ANALYSIS (Ensemble)")
print("=" * 80)

worst_idx = np.array([], dtype=int)  # fallback

if "ensemble" in predictions:
    residuals = ensemble_pred - y_test
    test_index = test_df.index

    bias_dir = "OVER" if np.mean(residuals) > 0 else "UNDER"
    print(f"\n  Overall Residual Statistics:")
    print(f"    Mean residual (bias):   {np.mean(residuals):+.2f} MWh ({bias_dir}-predicting)")
    print(f"    Std of residuals:       {np.std(residuals):.2f} MWh")
    print(f"    Median residual:        {np.median(residuals):+.2f} MWh")
    print(f"    Min residual:           {np.min(residuals):+.2f} MWh")
    print(f"    Max residual:           {np.max(residuals):+.2f} MWh")

    # Hourly pattern
    print(f"\n  Hourly Residual Pattern (Mean Absolute Error by Hour):")
    hourly_df = pd.DataFrame(
        {
            "residual": residuals,
            "abs_residual": np.abs(residuals),
            "hour": test_index.hour,
        },
        index=test_index,
    )
    hourly_stats = hourly_df.groupby("hour").agg(
        mean_residual=("residual", "mean"),
        mae=("abs_residual", "mean"),
        count=("residual", "count"),
    )
    print(f"    {'Hour':>4s}  {'Mean Res':>10s}  {'MAE':>10s}  {'Count':>5s}")
    for h, row in hourly_stats.iterrows():
        bar = "#" * int(row["mae"] / hourly_stats["mae"].max() * 30)
        print(
            f"    {h:4d}  {row['mean_residual']:+10.1f}  {row['mae']:10.1f}  {row['count']:5.0f}  {bar}"
        )

    worst_hours = hourly_stats.nlargest(3, "mae")
    print(f"\n  Worst 3 hours by MAE: {list(worst_hours.index)}")

    # Weekly pattern
    print(f"\n  Day of Week Residual Pattern:")
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    weekly_df = pd.DataFrame(
        {
            "residual": residuals,
            "abs_residual": np.abs(residuals),
            "dow": test_index.dayofweek,
        },
        index=test_index,
    )
    weekly_stats = weekly_df.groupby("dow").agg(
        mean_residual=("residual", "mean"),
        mae=("abs_residual", "mean"),
        count=("residual", "count"),
    )
    print(f"    {'Day':>4s}  {'Mean Res':>10s}  {'MAE':>10s}  {'Count':>5s}")
    for d, row in weekly_stats.iterrows():
        bar = "#" * int(row["mae"] / weekly_stats["mae"].max() * 30)
        print(
            f"    {day_names[d]:4s}  {row['mean_residual']:+10.1f}  {row['mae']:10.1f}  {row['count']:5.0f}  {bar}"
        )

    worst_days = weekly_stats.nlargest(2, "mae")
    print(f"\n  Worst 2 days by MAE: {[day_names[d] for d in worst_days.index]}")

    # Monthly pattern
    print(f"\n  Monthly Residual Pattern:")
    monthly_df = pd.DataFrame(
        {
            "residual": residuals,
            "abs_residual": np.abs(residuals),
            "month": test_index.month,
        },
        index=test_index,
    )
    monthly_stats = monthly_df.groupby("month").agg(
        mean_residual=("residual", "mean"),
        mae=("abs_residual", "mean"),
        count=("residual", "count"),
    )
    month_names_map = {
        1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
        7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
    }
    print(f"    {'Month':>5s}  {'Mean Res':>10s}  {'MAE':>10s}  {'Count':>5s}")
    for m_idx, row in monthly_stats.iterrows():
        print(
            f"    {month_names_map[m_idx]:5s}  {row['mean_residual']:+10.1f}  {row['mae']:10.1f}  {row['count']:5.0f}"
        )

    # Worst 10 timestamps
    print(f"\n  Worst 10 Prediction Timestamps (by absolute error):")
    abs_errors = np.abs(residuals)
    worst_idx = np.argsort(abs_errors)[-10:][::-1]
    print(
        f"    {'Timestamp':>22s}  {'Actual':>10s}  {'Predicted':>10s}  {'Error':>10s}  {'Error%':>8s}"
    )
    for idx in worst_idx:
        ts = test_index[idx]
        actual = y_test[idx]
        pred = ensemble_pred[idx]
        err = residuals[idx]
        pct = abs(err / actual) * 100 if actual != 0 else 0
        dow = day_names[ts.dayofweek]
        print(
            f"    {str(ts):>22s}  {actual:10.1f}  {pred:10.1f}  {err:+10.1f}  {pct:7.1f}%  ({dow} h{ts.hour:02d})"
        )
else:
    print("  No ensemble predictions available.")

# ========================================================================
# SECTION 5: CATBOOST FEATURE IMPORTANCE
# ========================================================================
print("\n" + "=" * 80)
print("  SECTION 5: CATBOOST FEATURE IMPORTANCE")
print("=" * 80)

if cb_model is not None and cb_feature_names is not None:
    importances = cb_model.get_feature_importance()
    feature_imp = sorted(
        zip(cb_feature_names, importances), key=lambda x: x[1], reverse=True
    )

    print(f"\n  Top 20 Features:")
    print(f"    {'#':>3s}  {'Feature':40s}  {'Importance':>12s}")
    for i, (feat, imp) in enumerate(feature_imp[:20]):
        bar = "#" * int(imp / feature_imp[0][1] * 30)
        print(f"    {i+1:3d}  {feat:40s}  {imp:12.2f}  {bar}")

    # Near-zero importance
    zero_features = [(f, i) for f, i in feature_imp if i < 0.01]
    print(
        f"\n  Near-Zero Importance Features (<0.01): {len(zero_features)} / {len(feature_imp)}"
    )
    if zero_features:
        for f_name, f_imp in zero_features[:10]:
            print(f"    - {f_name} ({f_imp:.4f})")
        if len(zero_features) > 10:
            print(f"    ... and {len(zero_features) - 10} more")

    # Feature group analysis
    print(f"\n  Feature Group Importance:")
    groups = {
        "consumption_lag": [
            f
            for f, _ in feature_imp
            if "consumption_lag" in f
            or "consumption_rolling" in f
            or "consumption_ewma" in f
        ],
        "calendar": [
            f
            for f, _ in feature_imp
            if any(
                x in f
                for x in [
                    "hour",
                    "day_of_week",
                    "month",
                    "is_holiday",
                    "is_weekend",
                    "is_ramadan",
                    "season",
                    "bayram",
                ]
            )
        ],
        "weather": [
            f
            for f, _ in feature_imp
            if any(
                x in f
                for x in [
                    "temperature",
                    "humidity",
                    "wind",
                    "pressure",
                    "precipitation",
                    "hdd",
                    "cdd",
                    "comfort",
                    "weather_code",
                    "dew_point",
                    "apparent_temp",
                    "snow",
                ]
            )
        ],
        "solar": [
            f
            for f, _ in feature_imp
            if any(
                x in f
                for x in [
                    "sol_",
                    "ghi",
                    "dni",
                    "dhi",
                    "poa",
                    "clearness",
                    "shortwave",
                    "cloud_proxy",
                ]
            )
        ],
        "epias": [
            f
            for f, _ in feature_imp
            if any(
                x in f
                for x in [
                    "epias",
                    "dam_",
                    "bilateral",
                    "load_forecast",
                    "rtc_",
                    "real_time",
                ]
            )
        ],
    }

    imp_dict = dict(feature_imp)
    for group_name, features in groups.items():
        total_imp = sum(imp_dict.get(f, 0) for f in features)
        print(f"    {group_name:25s}: {total_imp:8.2f} ({len(features)} features)")
else:
    print("  CatBoost model not loaded.")

# ========================================================================
# SECTION 6: ENSEMBLE WEIGHT ANALYSIS
# ========================================================================
print("\n" + "=" * 80)
print("  SECTION 6: ENSEMBLE WEIGHT ANALYSIS")
print("=" * 80)

print(f"\n  Configured Weights:")
for model_name, w in weights.items():
    bar = "#" * int(w * 50)
    print(f"    {model_name:12s}: {w:.3f}  {bar}")

max_w = max(weights.values())
min_w = min(weights.values())
ratio = max_w / min_w if min_w > 0 else float("inf")
print(f"\n  Weight Ratio (max/min): {ratio:.1f}x")
if ratio > 5:
    dominant = max(weights, key=weights.get)  # type: ignore
    print(
        f"  WARNING: Weights are heavily skewed! The ensemble is dominated by {dominant}"
    )
elif ratio > 2:
    dominant = max(weights, key=weights.get)  # type: ignore
    print(f"  NOTE: Moderate weight imbalance. {dominant} is dominant.")
else:
    print(f"  Weights are reasonably balanced.")

# Correlation of weights with performance
print(f"\n  Weight vs Performance (Test MAPE):")
print(
    f"    {'Model':12s}  {'Weight':>8s}  {'MAPE%':>8s}  {'Weight Justified?':<25s}"
)
for mn in weights:
    w = weights[mn]
    if mn in model_metrics:
        mape_val = model_metrics[mn]["MAPE"]
        if (w > 0.3 and mape_val < 10) or (w < 0.3 and mape_val > 10):
            justified = "YES (low MAPE, high weight)"
        else:
            justified = "CHECK"
        print(f"    {mn:12s}  {w:8.3f}  {mape_val:8.2f}  {justified}")
    else:
        print(f"    {mn:12s}  {w:8.3f}  {'N/A':>8s}  (model not loaded)")

# ========================================================================
# SECTION 7: OVERFITTING CHECK
# ========================================================================
print("\n" + "=" * 80)
print("  SECTION 7: OVERFITTING CHECK (CatBoost)")
print("=" * 80)

mape_gap = 0.0  # default

if cb_predictions_train is not None and cb_predictions_test is not None:
    train_metrics_cb = compute_metrics(y_train, cb_predictions_train)
    test_metrics_cb = compute_metrics(y_test, cb_predictions_test)

    print(
        f"\n  {'Metric':10s}  {'Train':>10s}  {'Test':>10s}  {'Ratio':>8s}  {'Status':<15s}"
    )
    print("  " + "-" * 60)

    for metric_name in ["MAPE", "MAE", "RMSE", "R2"]:
        t_val = train_metrics_cb[metric_name]
        s_val = test_metrics_cb[metric_name]

        if metric_name == "R2":
            diff = t_val - s_val
            status = "OVERFIT" if diff > 0.05 else "OK"
            print(
                f"  {metric_name:10s}  {t_val:10.4f}  {s_val:10.4f}  {diff:+8.4f}  {status}"
            )
        else:
            ratio_val = s_val / t_val if t_val > 0 else float("inf")
            status = (
                "OVERFIT" if ratio_val > 2.0 else ("MILD" if ratio_val > 1.5 else "OK")
            )
            print(
                f"  {metric_name:10s}  {t_val:10.2f}  {s_val:10.2f}  {ratio_val:8.2f}x  {status}"
            )

    mape_gap = test_metrics_cb["MAPE"] - train_metrics_cb["MAPE"]
    print(f"\n  MAPE Gap (Test - Train): {mape_gap:+.2f}%")
    if mape_gap > 5:
        print(
            "  VERDICT: SIGNIFICANT OVERFITTING detected. Train MAPE much lower than test."
        )
    elif mape_gap > 2:
        print("  VERDICT: MILD OVERFITTING. Some generalization gap but acceptable.")
    else:
        print("  VERDICT: Good generalization. Minimal overfitting.")
else:
    print("  CatBoost predictions not available for overfitting check.")

# Also check Prophet
if prophet_predictions_train is not None and prophet_predictions_test is not None:
    print(f"\n  Prophet Overfitting Check:")
    train_m_p = compute_metrics(y_train, prophet_predictions_train)
    test_m_p = compute_metrics(y_test, prophet_predictions_test)
    print(f"    Train MAPE: {train_m_p['MAPE']:.2f}%")
    print(f"    Test MAPE:  {test_m_p['MAPE']:.2f}%")
    gap_p = test_m_p["MAPE"] - train_m_p["MAPE"]
    print(f"    Gap:        {gap_p:+.2f}%")

# ========================================================================
# SECTION 8: HOLIDAY / RAMADAN IMPACT
# ========================================================================
print("\n" + "=" * 80)
print("  SECTION 8: HOLIDAY / RAMADAN IMPACT")
print("=" * 80)

holidays_path = BASE / "data" / "static" / "turkish_holidays.parquet"
if holidays_path.exists() and "ensemble" in predictions:
    hol_df = pd.read_parquet(holidays_path)
    print(f"\n  Holidays data: {len(hol_df)} entries")
    print(f"  Columns: {list(hol_df.columns)}")

    # Show sample
    print(f"\n  Holiday data sample (first 5):")
    print(hol_df.head().to_string())

    # Check which holidays fall in test period
    if "date" in hol_df.columns:
        hol_dates = pd.to_datetime(hol_df["date"])
    elif "ds" in hol_df.columns:
        hol_dates = pd.to_datetime(hol_df["ds"])
    else:
        hol_dates = pd.to_datetime(hol_df.iloc[:, 0])

    test_start_date = test_df.index.min().date()
    test_end_date = test_df.index.max().date()

    test_holidays = hol_df[
        (hol_dates.dt.date >= test_start_date) & (hol_dates.dt.date <= test_end_date)
    ]
    print(
        f"\n  Holidays in test period ({test_start_date} to {test_end_date}):"
    )
    if len(test_holidays) > 0:
        print(test_holidays.to_string())
    else:
        print("    No holidays in test period.")

    # Check if is_holiday and is_ramadan columns exist in test data
    if "is_holiday" in test_df.columns:
        holiday_mask = test_df["is_holiday"] == 1
        ramadan_mask = (
            test_df["is_ramadan"] == 1
            if "is_ramadan" in test_df.columns
            else pd.Series(False, index=test_df.index)
        )

        n_holiday = holiday_mask.sum()
        n_ramadan = ramadan_mask.sum()
        n_normal = (~holiday_mask & ~ramadan_mask).sum()

        print(f"\n  Test set composition:")
        print(f"    Holiday hours:  {n_holiday}")
        print(f"    Ramadan hours:  {n_ramadan}")
        print(f"    Normal hours:   {n_normal}")

        if n_holiday > 0:
            holiday_errors = np.abs(residuals[holiday_mask.values])
            normal_errors = np.abs(residuals[(~holiday_mask & ~ramadan_mask).values])
            print(f"\n  Error comparison:")
            print(f"    Holiday MAE:  {np.mean(holiday_errors):.1f} MWh")
            print(f"    Normal MAE:   {np.mean(normal_errors):.1f} MWh")
            ratio_h = np.mean(holiday_errors) / np.mean(normal_errors)
            print(f"    Ratio:        {ratio_h:.2f}x")

            if ratio_h > 1.5:
                print("    WARNING: Holiday errors are significantly higher!")

        if n_ramadan > 0:
            ramadan_errors = np.abs(residuals[ramadan_mask.values])
            normal_errors_nr = np.abs(
                residuals[(~holiday_mask & ~ramadan_mask).values]
            )
            print(f"\n    Ramadan MAE:  {np.mean(ramadan_errors):.1f} MWh")
            print(f"    Normal MAE:   {np.mean(normal_errors_nr):.1f} MWh")
            ratio_r = np.mean(ramadan_errors) / np.mean(normal_errors_nr)
            print(f"    Ratio:        {ratio_r:.2f}x")

    # Check if worst predictions cluster around holidays
    if len(worst_idx) > 0:
        day_names_arr = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        print(f"\n  Worst 10 predictions - Holiday status:")
        for idx in worst_idx:
            ts = test_df.index[idx]
            is_hol = test_df.iloc[idx].get("is_holiday", 0)
            is_ram = test_df.iloc[idx].get("is_ramadan", 0)
            err = abs(residuals[idx])
            labels = []
            if is_hol:
                labels.append("HOLIDAY")
            if is_ram:
                labels.append("RAMADAN")
            if not labels:
                labels.append("normal")
            print(
                f"    {str(ts):>22s}  error={err:.1f} MWh  [{' + '.join(labels)}]"
            )
else:
    print("  Holidays data or ensemble predictions not available.")

# ========================================================================
# SUMMARY
# ========================================================================
print("\n" + "=" * 80)
print("  SUMMARY & KEY FINDINGS")
print("=" * 80)

findings = []

# Model performance
if "catboost" in model_metrics:
    findings.append(
        f"CatBoost test MAPE: {model_metrics['catboost']['MAPE']:.2f}%"
    )
if "prophet" in model_metrics:
    findings.append(
        f"Prophet test MAPE: {model_metrics['prophet']['MAPE']:.2f}%"
    )
if "tft" in model_metrics:
    findings.append(f"TFT test MAPE: {model_metrics['tft']['MAPE']:.2f}%")
if "ensemble" in model_metrics:
    findings.append(
        f"Ensemble test MAPE: {model_metrics['ensemble']['MAPE']:.2f}%"
    )

# Weight analysis
findings.append(
    f"Ensemble weights: CatBoost={weights['catboost']:.0%}, Prophet={weights['prophet']:.0%}, TFT={weights['tft']:.0%}"
)
if ratio > 5:
    dominant = max(weights, key=weights.get)  # type: ignore
    findings.append(
        f"ISSUE: Ensemble is heavily dominated by {dominant} ({max_w:.0%})"
    )

# Overfitting
if cb_predictions_train is not None:
    if mape_gap > 5:
        findings.append(
            f"ISSUE: CatBoost shows significant overfitting (gap={mape_gap:.1f}%)"
        )
    elif mape_gap > 2:
        findings.append(
            f"NOTE: CatBoost shows mild overfitting (gap={mape_gap:.1f}%)"
        )
    else:
        findings.append(
            f"CatBoost generalizes well (gap={mape_gap:.1f}%)"
        )

# TFT status
if not tft_loaded:
    findings.append(
        "NOTE: TFT model could not be loaded for inference in this analysis"
    )

print()
for i, f in enumerate(findings, 1):
    print(f"  {i}. {f}")

print("\n" + "=" * 80)
print("  END OF REPORT")
print("=" * 80)
