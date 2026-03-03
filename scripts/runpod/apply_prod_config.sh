#!/usr/bin/env bash
# apply_prod_config.sh — Switch configs to production HPO values
# Target: RunPod RTX PRO 4500 (32 GB VRAM, 62 GB RAM, 32 vCPU)
# Run from project root: bash scripts/runpod/apply_prod_config.sh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

echo "=== Applying Production HPO Config ==="
echo ""

# --- Backup current configs ---
BACKUP_DIR="configs/models/.dev_backup"
mkdir -p "$BACKUP_DIR"
cp configs/models/hyperparameters.yaml "$BACKUP_DIR/hyperparameters.yaml"
cp configs/models/catboost.yaml "$BACKUP_DIR/catboost.yaml"
cp configs/models/tft.yaml "$BACKUP_DIR/tft.yaml"
echo "[OK] Dev configs backed up to $BACKUP_DIR/"

# --- hyperparameters.yaml ---
cat > configs/models/hyperparameters.yaml << 'EOF'
# Optuna hyperparameter search spaces — PRODUCTION HPO
# Each param: type (int/float/categorical), low, high, step?, log?, choices?
# Adding a new parameter to YAML requires NO code change

target_col: consumption

catboost:
  n_trials: 50
  search_space:
    # iterations NOT in search space — fixed at 5000 in catboost.yaml
    # early_stopping_rounds=100 handles optimal tree count automatically
    learning_rate:
      type: float
      low: 0.01
      high: 0.1
      log: true
    depth:
      type: int
      low: 4
      high: 7
    l2_leaf_reg:
      type: float
      low: 1.0
      high: 10.0
    min_child_samples:
      type: int
      low: 5
      high: 100
    subsample:
      type: float
      low: 0.6
      high: 1.0
    bootstrap_type:
      type: categorical
      choices: ["MVS"]
    loss_function:
      type: categorical
      choices: ["RMSE"]   # Fixed — RMSE beat MAE by 0.16% in R1 (2.42% vs 2.58%)

prophet:
  n_trials: 30
  search_space:
    changepoint_prior_scale:
      type: float
      low: 0.001
      high: 1.0
      log: true
    seasonality_prior_scale:
      type: float
      low: 0.01
      high: 50.0
      log: true
    holidays_prior_scale:
      type: float
      low: 0.01
      high: 50.0
      log: true
    n_changepoints:
      type: int
      low: 15
      high: 50
    # seasonality_mode removed from search space — energy consumption is
    # inherently multiplicative (seasonal effects scale with load level).
    # Fixed in configs/models/prophet.yaml → seasonality.mode: "multiplicative"

tft:
  n_trials: 20
  search_space:
    hidden_size:
      type: int
      low: 32
      high: 128
      step: 32
    attention_head_size:
      type: categorical
      choices: [1, 2, 4]
    dropout:
      type: float
      low: 0.05
      high: 0.3
    hidden_continuous_size:
      type: int
      low: 8
      high: 64
      step: 8
    lstm_layers:
      type: categorical
      choices: [1, 2]
    learning_rate:
      type: float
      low: 0.0001
      high: 0.01
      log: true
    batch_size:
      type: categorical
      choices: [256, 512]         # 512 → ~88 steps/epoch (sweet spot), 256 → finer gradients

cross_validation:
  n_splits: 12
  val_months: 1
  test_months: 1
  gap_hours: 0
  shuffle: false
EOF
echo "[OK] hyperparameters.yaml → Production (CB:50, Prophet:30, TFT:20 trials, 12-fold CV)"

# --- catboost.yaml ---
cat > configs/models/catboost.yaml << 'EOF'
# CatBoost model configuration — PRODUCTION
training:
  task_type: "CPU"
  iterations: 5000
  learning_rate: 0.05
  depth: 6
  loss_function: "RMSE"       # R1: RMSE beat MAE by 0.16% (2.42% vs 2.58%)
  eval_metric: "MAPE"
  early_stopping_rounds: 100
  bootstrap_type: "MVS"       # R1 lesson: GPU compat, subsample requires MVS
  has_time: true
  random_seed: 42
  verbose: 500

# R2: 36→28 categorical features (pruned zero-importance + misclassified)
categorical_features:
  # Time
  - hour
  - day_of_week
  - day_of_month
  - week_of_year
  - month
  - quarter
  - season
  - year
  # Holiday / special days
  - is_holiday
  - is_weekend
  - is_ramadan
  - is_bridge_day
  - tatil_tipi
  - bayram_gun_no
  - holiday_duration
  # Interaction (flag x hour)
  - is_holiday_x_hour
  - is_ramadan_x_hour
  - is_weekend_x_hour
  # Time-period flags
  - is_business_hours
  - is_peak
  - is_ramp_morning
  # DROPPED R2: is_ramp_evening (zero importance, removed from pipeline)
  - is_friday
  - is_monday
  - is_sunday
  # Weather
  - weather_code
  - weather_group
  - wth_extreme_cold
  # DROPPED R2: wth_extreme_hot, wth_extreme_wind, wth_heavy_precip (zero importance, disabled)
  # DROPPED R2: wth_is_severe, wth_severity (severity disabled in pipeline)
  # Season / solar
  - is_cooling_season
  # DROPPED R2: is_heating_season (zero importance, removed from pipeline)
  # DROPPED R2: sol_is_daylight (zero importance, removed from pipeline)
  # DROPPED R2: sol_daylight_hours (continuous value, was misclassified as categorical)

nan_handling:
  categorical: "missing"
EOF
echo "[OK] catboost.yaml → CPU mode, RMSE loss, MVS bootstrap, 28 R2-pruned categoricals"

# --- tft.yaml ---
cat > configs/models/tft.yaml << 'EOF'
# Temporal Fusion Transformer configuration — PRODUCTION
# Target: RTX PRO 4500 (32 GB VRAM, 32 vCPU)
# R2: Architecture defaults updated from smoke (8) to HPO midpoint (64)
architecture:
  hidden_size: 64             # 8→64 (HPO midpoint, search [32, 128])
  attention_head_size: 2      # 1→2 (multi-head exploration)
  lstm_layers: 1
  dropout: 0.1
  hidden_continuous_size: 32  # 8→32 (larger continuous embedding)

training:
  encoder_length: 168  # 7 days
  prediction_length: 48
  batch_size: 512             # RTX PRO 4500 32GB — ~88 steps/epoch, VRAM ~2GB worst case
  max_epochs: 50              # PRODUCTION (dev: 2)
  learning_rate: 0.001
  early_stop_patience: 7      # RTX slightly slower per epoch → more patience for convergence
  gradient_clip_val: 0.1
  random_seed: 42
  accelerator: "gpu"          # PRODUCTION (dev: auto)
  num_workers: 8              # n_jobs=1 × 8 workers = 8 (32 vCPU has plenty of headroom)
  enable_progress_bar: false  # suppress verbose progress bars in log files
  precision: "16-mixed"       # Mixed precision for ~30-40% GPU speedup

optimization:
  # fast_epochs: no longer used — epoch-level Optuna pruning replaces the
  # fast_epochs/retrain pattern.  All trials now train at max_epochs; bad
  # trials are pruned early by MedianPruner via PyTorchLightningPruningCallback.
  optuna_splits: 6            # PRODUCTION (dev: 3) — 6 months seasonality coverage
  n_jobs: 1                   # MUST be 1 — parallel trials on single GPU = 100x slowdown
  val_size_hours: 720  # ~1 month (24 * 30) for final model validation

# R2: Covariates pruned via CatBoost feature importance (top-75, >= 0.1%)
# FIXES: hdd_x_hour moved unknown→known (forecast-derivable)
# ADDS: wth_cdd, wth_hdd, temp_x_hour, cdd_x_hour to known (all forecast-derivable)
# REMOVES: 3 low-importance unknowns with high NaN rates
covariates:
  # Deterministic or forecast-available — enters both encoder AND decoder
  time_varying_known:
    # Cyclical time encodings
    - hour_cos              # 0.23% importance
    - day_of_week_sin       # 0.94%
    - day_of_week_cos       # 0.22%
    - month_sin             # 0.17%
    - day_of_year_sin       # 0.16%
    - day_of_year_cos       # 0.30%
    # Weekend/holiday signals
    - is_weekend            # 0.30%
    - is_sunday             # 0.36%
    - is_bridge_day         # 0.13%
    - tatil_tipi            # 1.44% — holiday type (0/1/2)
    - holiday_duration      # 2.22% — consecutive holiday length
    - bayrama_kalan_gun     # 0.62% — days until bayram
    - bayram_gun_no         # 0.45% — bayram day number
    - days_since_holiday    # 0.36%
    - days_until_holiday    # 0.17%
    # Weather (forecast available at prediction time)
    - temperature_2m        # 1.10%
    - apparent_temperature  # 2.24%
    - shortwave_radiation   # 0.23%
    # Solar (deterministic, astronomic)
    - sol_elevation         # 0.11%
    # R2 NEW: Weather-derived features (forecast-derivable, all known at prediction time)
    - wth_cdd              # cooling degree days, forecast-derived
    - wth_hdd              # heating degree days, forecast-derived
    - hdd_x_hour           # FIX: was in unknown, but wth_hdd*hour = forecast*deterministic = KNOWN
    - temp_x_hour          # temperature*hour interaction, forecast-derived
    - cdd_x_hour           # CDD*hour interaction, forecast-derived

  # Observed only in past (lagged >= 48h) — enters ONLY encoder
  time_varying_unknown:
    # Consumption lags (core autoregressive signal, 42.5% total)
    - consumption_lag_48          # 6.09%
    - consumption_lag_168         # 27.04% — strongest single feature
    - consumption_lag_336         # 9.24%
    - consumption_lag_720         # long-term baseline
    # Consumption derived (momentum, ratios, profile — 22.7% total)
    - consumption_week_ratio      # 8.02%
    - consumption_hourly_profile  # 5.59%
    - consumption_momentum_168    # 2.46%
    - consumption_pct_change_168  # 1.96%
    - consumption_trend_ratio_168_336  # 0.80%
    - consumption_trend_ratio_48_168   # 0.48%
    - consumption_window_720_std  # 1.15%
    - consumption_window_48_max   # 0.38%
    - consumption_window_336_max  # 0.37%
    # DROPPED R2: consumption_window_720_mean (0.13%, high NaN from 720h window)
    # DROPPED R2: consumption_window_336_min (0.11%, moderate NaN)
    # DROPPED R2: consumption_q75_168 (0.14%, single quantile not informative alone)
    # Weather windows (observed, not forecast)
    - temperature_2m_window_24_max   # 2.24%
    - temperature_2m_window_12_max   # 1.49%
    - temperature_2m_window_6_mean   # 1.48%
    # MOVED R2: hdd_x_hour → time_varying_known (forecast-derivable)

quantiles:
  - 0.02
  - 0.10
  - 0.25
  - 0.50
  - 0.75
  - 0.90
  - 0.98

loss: "quantile"
EOF
echo "[OK] tft.yaml → Production RTX PRO 4500 (epochs=50, gpu, workers=8, n_jobs=1, epoch-level pruning)"

echo ""
echo "=== Production Config Applied (RTX PRO 4500) ==="
echo ""
echo "Target hardware: RTX PRO 4500 (32 GB VRAM, 62 GB RAM, 32 vCPU)"
echo ""
echo "Summary of changes:"
echo "  hyperparameters.yaml:"
echo "    - CatBoost: 50 trials, RMSE-only, bootstrap MVS"
echo "    - Prophet:  30 trials"
echo "    - TFT:      20 trials, batch search [256, 512]"
echo "    - CV:       12 splits (was 2)"
echo "  catboost.yaml:"
echo "    - CPU 32-core, RMSE loss, MVS bootstrap"
echo "    - iterations: 5000, early_stopping: 100"
echo "    - R2 pruned: 28 categoricals (was 36)"
echo "  tft.yaml:"
echo "    - Architecture: hidden=64, heads=2, lstm=1 (HPO midpoints)"
echo "    - Training: max_epochs=50, patience=7, batch=512, gpu, 16-mixed precision"
echo "    - DataLoader: num_workers=8 (n_jobs=1 × 8 = 8, 32 vCPU has headroom)"
echo "    - Optuna: epoch-level pruning (MedianPruner), n_jobs=1 (MUST be 1 on single GPU)"
echo "    - Optuna: optuna_splits=6 (6 months seasonality coverage)"
echo "    - R2 covariates: 24 known + 16 unknown"
echo ""
echo "To revert: bash scripts/runpod/restore_dev_config.sh"
