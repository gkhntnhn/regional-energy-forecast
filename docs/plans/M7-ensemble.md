# M7: 2-Model Ensemble (CatBoost + Prophet)

## Status: COMPLETE

## Summary

CatBoost ve Prophet tahminlerini birleştiren weighted-average ensemble sistemi.
Ağırlıklar config'den gelir, validation seti üzerinde scipy.minimize_scalar ile optimize edilir.

## Implemented Components

### 1. Config
- `configs/models/ensemble.yaml`: Default weights, optimization settings, fallback config

### 2. Config Models (settings.py)
- `EnsembleWeightsConfig`: catboost + prophet weights (sum to 1.0)
- `EnsembleOptimizationConfig`: enabled, metric, prophet_weight_min/max
- `EnsembleFallbackConfig`: enabled flag for graceful degradation
- `EnsembleConfig`: Combined config

### 3. EnsembleTrainer (training/ensemble_trainer.py)
- Orchestrates CatBoost and Prophet training
- Collects validation predictions from TSCV splits
- Optimizes weights using scipy.minimize_scalar (bounded)
- Generates comparison DataFrame (Model | Val MAPE | Test MAPE | Improvement)
- Prints formatted summary table to terminal

### 4. EnsembleForecaster (models/ensemble.py)
- Loads pre-trained CatBoost and Prophet models
- Weighted-average prediction: cb_weight * cb_pred + pr_weight * pr_pred
- Save/load weights to JSON
- Load models from .cbm and .pkl files

### 5. CLI Support (training/run.py)
- `--model ensemble` option
- `run_ensemble()` function
- Saves weights to models/ensemble_weights.json

### 6. MLflow Logging (training/experiment.py)
- `log_ensemble_weights()`: Per-model weights
- `log_comparison_metrics()`: CatBoost vs Prophet vs Ensemble MAPE

## Output Format

```
======================================================================
                    ENSEMBLE TRAINING REPORT
======================================================================

Model           Val MAPE     Test MAPE    Improvement
------------------------------------------------------
CatBoost           5.23%        5.45%    baseline
Prophet            7.12%        7.34%    +36.1%
Ensemble           4.87%        5.02%    -7.9%
------------------------------------------------------
Optimized Weights: CatBoost=0.650, Prophet=0.350
======================================================================
```

## Files Changed

| File | Change |
|------|--------|
| configs/models/ensemble.yaml | CREATE |
| src/energy_forecast/config/settings.py | EDIT - Add EnsembleConfig |
| src/energy_forecast/training/ensemble_trainer.py | CREATE |
| src/energy_forecast/models/ensemble.py | EDIT - Full implementation |
| src/energy_forecast/training/run.py | EDIT - Add ensemble CLI |
| src/energy_forecast/training/experiment.py | EDIT - Ensemble logging |
| src/energy_forecast/training/__init__.py | EDIT - Exports |
| tests/unit/test_training/test_ensemble_trainer.py | CREATE |
| tests/unit/test_models/test_ensemble.py | CREATE |
| pyproject.toml | EDIT - Add scipy to mypy overrides |

## Test Coverage

- 29 new tests for ensemble functionality
- All 321 project tests passing
