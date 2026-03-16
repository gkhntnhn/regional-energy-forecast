"""TFT speed benchmark: step_size × windows_batch_size combinations.

Tests GPU utilization with different batch and step configurations.
Run on RunPod or any GPU machine.

Usage:
    uv run python scripts/test_tft_speed.py
    uv run python scripts/test_tft_speed.py --gpu     # Force GPU
    uv run python scripts/test_tft_speed.py --cpu     # Force CPU
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from energy_forecast.config.models import (
    TFTArchitectureConfig,
    TFTConfig,
    TFTCovariatesConfig,
    TFTOptimizationConfig,
    TFTTrainingConfig,
)
from energy_forecast.models.tft import TFTForecaster

# Test configurations: (step_size, windows_batch_size, label)
CONFIGS = [
    (1, 1024, "ss=1  bs=1024"),
    (1, 2048, "ss=1  bs=2048"),
    (1, 4096, "ss=1  bs=4096"),
    (12, 1024, "ss=12 bs=1024"),
    (12, 2048, "ss=12 bs=2048"),
    (12, 4096, "ss=12 bs=4096"),
    (24, 1024, "ss=24 bs=1024"),
    (24, 2048, "ss=24 bs=2048"),
    (24, 4096, "ss=24 bs=4096"),
]

MAX_STEPS = 100
HIDDEN_SIZE = 128  # Production size


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    p = argparse.ArgumentParser(description="TFT speed benchmark")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--gpu", action="store_true", help="Force GPU")
    g.add_argument("--cpu", action="store_true", help="Force CPU")
    p.add_argument(
        "--steps", type=int, default=MAX_STEPS, help="Max steps (default: 100)"
    )
    return p.parse_args()


def run_config(
    step_size: int,
    batch_size: int,
    df: pd.DataFrame,
    accelerator: str,
    max_steps: int,
) -> dict[str, object]:
    """Train TFT with given config, return timing."""
    precision = "bf16-mixed" if accelerator == "gpu" else "32-true"
    num_workers = 4 if accelerator == "gpu" else 0

    cfg = TFTConfig(
        architecture=TFTArchitectureConfig(
            hidden_size=HIDDEN_SIZE, n_head=2, n_rnn_layers=1, dropout=0.1
        ),
        training=TFTTrainingConfig(
            encoder_length=168,
            prediction_length=48,
            max_steps=max_steps,
            windows_batch_size=batch_size,
            learning_rate=0.001,
            early_stop_patience_steps=-1,
            val_check_steps=max_steps + 1,
            gradient_clip_val=0.1,
            random_seed=42,
            accelerator=accelerator,
            num_workers=num_workers,
            enable_progress_bar=False,
            precision=precision,
            scaler_type="robust",
            rnn_type="lstm",
        ),
        covariates=TFTCovariatesConfig(),
        optimization=TFTOptimizationConfig(),
    )

    forecaster = TFTForecaster(cfg)

    # Patch step_size
    original_build = forecaster._build_nf_model

    def patched_build(callbacks=None, *, max_steps=None):
        nf = original_build(callbacks=callbacks, max_steps=max_steps)
        nf.models[0].step_size = step_size
        return nf

    forecaster._build_nf_model = patched_build  # type: ignore[assignment]

    val_size = 720
    train_df = df.iloc[:-val_size]
    val_df = df.iloc[-val_size:]

    start = time.monotonic()
    try:
        forecaster.train(train_df, val_df, target_col="consumption")
        elapsed = time.monotonic() - start
        status = "OK"
    except Exception as e:
        elapsed = time.monotonic() - start
        status = f"FAIL: {e!s:.50}"
    finally:
        del forecaster
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {
        "step_size": step_size,
        "batch_size": batch_size,
        "elapsed_s": round(elapsed, 1),
        "status": status,
        "steps_per_sec": round(max_steps / elapsed, 1) if "OK" in str(status) else 0,
    }


def main() -> None:
    """Run benchmark."""
    args = parse_args()

    if args.gpu:
        accel = "gpu"
    elif args.cpu:
        accel = "cpu"
    else:
        accel = "gpu" if torch.cuda.is_available() else "cpu"

    df = pd.read_parquet("data/processed/features_historical.parquet")

    if accel == "gpu" and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info("GPU: {} ({:.0f}GB)", gpu_name, vram)
    else:
        logger.info("Running on CPU")

    logger.info(
        "Data: {} rows | hidden={} | steps={} | configs={}",
        len(df), HIDDEN_SIZE, args.steps, len(CONFIGS),
    )
    logger.info("")

    results = []
    for ss, bs, label in CONFIGS:
        logger.info("Testing {} ...", label)
        r = run_config(ss, bs, df, accel, args.steps)
        results.append(r)
        logger.info(
            "  {} → {:.1f}s ({:.1f} steps/s) [{}]",
            label, r["elapsed_s"], r["steps_per_sec"], r["status"],
        )

    # Results table
    logger.info("")
    logger.info("=" * 70)
    logger.info("RESULTS — hidden={}, steps={}, {}", HIDDEN_SIZE, args.steps, accel)
    logger.info("=" * 70)
    logger.info(
        "{:>10s}  {:>10s}  {:>10s}  {:>12s}  {:>8s}  {}",
        "step_size", "batch_size", "time (s)", "steps/sec", "speedup", "status",
    )
    baseline = results[0]["elapsed_s"] if results[0]["elapsed_s"] > 0 else 1
    for r in results:
        speedup = baseline / r["elapsed_s"] if r["elapsed_s"] > 0 else 0
        logger.info(
            "{:>10d}  {:>10d}  {:>10.1f}  {:>12.1f}  {:>7.1f}x  {}",
            r["step_size"],
            r["batch_size"],
            r["elapsed_s"],
            r["steps_per_sec"],
            speedup,
            r["status"],
        )


if __name__ == "__main__":
    main()
