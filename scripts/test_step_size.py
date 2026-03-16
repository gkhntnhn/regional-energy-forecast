"""TFT step_size comparison test.

Runs TFT training with step_size=1, 12, 24 and compares
epoch/step ratio, training speed, and loss.

Usage:
    uv run python scripts/test_step_size.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from energy_forecast.config import load_config
from energy_forecast.models.tft import TFTForecaster

# Smoke config overrides
MAX_STEPS = 50
HIDDEN_SIZE = 8
BATCH_SIZE = 128
STEP_SIZES = [1, 12, 24]


def run_test(step_size: int, df: pd.DataFrame, settings: object) -> dict:
    """Train TFT with given step_size and return timing info."""
    from energy_forecast.config.models import (
        TFTArchitectureConfig,
        TFTConfig,
        TFTCovariatesConfig,
        TFTOptimizationConfig,
        TFTTrainingConfig,
    )

    cfg = TFTConfig(
        architecture=TFTArchitectureConfig(
            hidden_size=HIDDEN_SIZE,
            n_head=2,
            n_rnn_layers=1,
            dropout=0.1,
        ),
        training=TFTTrainingConfig(
            encoder_length=168,
            prediction_length=48,
            max_steps=MAX_STEPS,
            windows_batch_size=BATCH_SIZE,
            learning_rate=0.001,
            early_stop_patience_steps=-1,
            val_check_steps=MAX_STEPS + 1,
            gradient_clip_val=0.1,
            random_seed=42,
            accelerator="cpu",
            num_workers=0,
            enable_progress_bar=False,
            precision="32-true",
            scaler_type="robust",
            rnn_type="lstm",
        ),
        covariates=TFTCovariatesConfig(),
        optimization=TFTOptimizationConfig(),
    )

    forecaster = TFTForecaster(cfg)

    # Monkey-patch step_size into the model build
    original_build = forecaster._build_nf_model

    def patched_build(callbacks=None, *, max_steps=None):
        nf = original_build(callbacks=callbacks, max_steps=max_steps)
        nf.models[0].step_size = step_size
        logger.info(
            "step_size={} set on TFT model", step_size
        )
        return nf

    forecaster._build_nf_model = patched_build  # type: ignore[assignment]

    # Split data
    target_col = "consumption"
    val_size = 720
    train_df = df.iloc[:-val_size]
    val_df = df.iloc[-val_size:]

    start = time.monotonic()
    forecaster.train(train_df, val_df, target_col=target_col)
    elapsed = time.monotonic() - start

    return {
        "step_size": step_size,
        "elapsed_s": round(elapsed, 1),
        "max_steps": MAX_STEPS,
    }


def main() -> None:
    """Run step_size comparison."""
    settings = load_config()
    df = pd.read_parquet("data/processed/features_historical.parquet")
    logger.info(
        "Data: {} rows, testing step_size={} with {} steps",
        len(df), STEP_SIZES, MAX_STEPS,
    )

    results = []
    for ss in STEP_SIZES:
        logger.info("=" * 50)
        logger.info("Testing step_size={}", ss)
        logger.info("=" * 50)
        r = run_test(ss, df, settings)
        results.append(r)
        logger.info(
            "step_size={} done in {:.1f}s",
            ss, r["elapsed_s"],
        )

    logger.info("")
    logger.info("=" * 60)
    logger.info("RESULTS")
    logger.info("=" * 60)
    logger.info(
        "{:>10s}  {:>10s}  {:>10s}",
        "step_size", "time (s)", "speedup",
    )
    baseline = results[0]["elapsed_s"]
    for r in results:
        speedup = baseline / r["elapsed_s"] if r["elapsed_s"] > 0 else 0
        logger.info(
            "{:>10d}  {:>10.1f}  {:>9.1f}x",
            r["step_size"], r["elapsed_s"], speedup,
        )


if __name__ == "__main__":
    main()
