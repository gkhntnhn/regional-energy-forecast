# M9: 3-Model Ensemble (CatBoost + Prophet + TFT)

> Status: ✅ Completed
> Date: 2026-02-08

## Overview

Extended M7's 2-model ensemble (CatBoost + Prophet) to include TFT as the third model.
Added dynamic `active_models` system with CLI override support.

## Key Changes

### 1. Configuration (`configs/models/ensemble.yaml`)
- Added `active_models` list: `["catboost", "prophet", "tft"]`
- Extended weights to 3 models: `catboost: 0.45, prophet: 0.30, tft: 0.25`
- Added per-model optimization bounds

### 2. Settings (`src/energy_forecast/config/settings.py`)
- `EnsembleWeightsConfig`: Added TFT weight, `get_normalized()` method for auto-normalization
- `EnsembleWeightBoundsConfig`: Per-model weight bounds for optimization
- `EnsembleConfig`: Added `active_models` field with validation

### 3. Trainer (`src/energy_forecast/training/ensemble_trainer.py`)
- `EnsembleTrainer.__init__`: Added `active_models_override` parameter (no config mutation)
- Dynamic trainer creation based on active models
- N-dimensional weight optimization using `scipy.optimize.minimize` with SLSQP
- Updated result dataclasses for N-model support

### 4. Forecaster (`src/energy_forecast/models/ensemble.py`)
- `EnsembleForecaster`: Added TFT support, `active_models` property
- `predict()`: Auto-normalizes weights for active models with predictions
- `set_models()`: Optional parameters for each model

### 5. CLI (`src/energy_forecast/training/run.py`)
- Added `--models` argument: `--models catboost,prophet` to override active models

## Usage

```bash
# Train all 3 models (default)
make train-ensemble

# Train only CatBoost + Prophet
python -m energy_forecast.training.run --model ensemble --models catboost,prophet

# Train only CatBoost + TFT
python -m energy_forecast.training.run --model ensemble --models catboost,tft
```

## Weight Normalization

When subset of models is active, weights are automatically normalized to sum=1:

```python
# Config: catboost=0.45, prophet=0.30, tft=0.25
# Active: ["catboost", "prophet"]
# Normalized: catboost=0.6 (0.45/0.75), prophet=0.4 (0.30/0.75)
```

## TFT Median Quantile

TFT predictions use the median (0.50) quantile for ensemble, as implemented in M8's `TFTForecaster.predict()`.

## Testing

- 48 unit tests covering:
  - Weight normalization for 2 and 3 model subsets
  - Active models override via `__init__` parameter
  - N-dimensional optimization with SLSQP
  - Comparison DataFrame generation for N models
