# M8: TFT Training Implementation

## Completed

✅ TFT (Temporal Fusion Transformer) training pipeline implemented using M5/M6 shared infrastructure.

## Changes

### Config Updates
- `configs/models/hyperparameters.yaml`: TFT Optuna search space (hidden_size, attention_head_size, dropout, etc.)
- `configs/models/tft.yaml`: Added accelerator, num_workers, hidden_continuous_size, covariates section

### Settings
- `src/energy_forecast/config/settings.py`:
  - TFTArchitectureConfig: Added hidden_continuous_size
  - TFTTrainingConfig: Added accelerator ("cpu"/"gpu"/"auto"), num_workers
  - TFTCovariatesConfig: New class for time_varying_known features
  - TFTConfig: Added covariates field

### Model Wrapper
- `src/energy_forecast/models/tft.py`: Full TFTForecaster implementation
  - TimeSeriesDataSet conversion (_prepare_dataframe, _create_dataset)
  - pytorch-forecasting integration
  - Quantile predictions (median as main output)
  - `get_quantile_predictions()` for uncertainty access
  - Save/load with PyTorch checkpoint

### Trainer
- `src/energy_forecast/training/tft_trainer.py`: TFTTrainer with TSCV + Optuna + MLflow
  - Same pattern as CatBoostTrainer/ProphetTrainer
  - Uses shared TimeSeriesSplitter
  - Uses shared suggest_params for dynamic search space
  - Fast optimization with reduced epochs on first split

### Experiment Tracker
- `src/energy_forecast/training/experiment.py`: Added log_tft_model() method

### CLI
- `src/energy_forecast/training/run.py`: Added `--model tft` option

### Tests
- `tests/unit/test_models/test_tft_forecaster.py`: 12 tests
- `tests/unit/test_training/test_tft_trainer.py`: 8 tests

## Key Design Decisions

1. **Predict output is median (0.50 quantile)**: All quantiles stored internally, accessible via `get_quantile_predictions()`

2. **Accelerator from config**: Default "cpu", can be "gpu" or "auto" in tft.yaml

3. **TimeSeriesDataSet conversion in trainer**: Feature-engineered DataFrame is converted to pytorch-forecasting format automatically

4. **Fast optimization**: During Optuna trials, uses reduced epochs (10) on first split only for speed

## Usage

```bash
# Full training
make train-tft

# Quick test (3 trials)
python -m energy_forecast.training.run --model tft --n-trials 3 --no-mlflow
```
