"""R12 FAZ 6 — Multi-seed lokal runner.

R12 HPO winner config (n_block=4, ff_dim=96, dropout=0.201, revin=False,
scaler=standard, loss=huber_0.5, ...) ile 5 seed × 12-fold TSCV training.
Sonuc: per-seed MAPE + ensemble (Jensen) MAPE + variance.

Usage:
  uv run python scripts/run_multi_seed.py --wbs 256          # Lokal 4060 Ti (OOM marji)
  uv run python scripts/run_multi_seed.py --wbs 512          # Pod RTX 3090
  uv run python scripts/run_multi_seed.py --seeds 42         # Tek seed (OOM check)
  uv run python scripts/run_multi_seed.py                    # Default 5 seed wbs=512
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Ensure local src/ is importable when run from project root
sys.path.insert(0, str(Path("src").resolve()))

import numpy as np
import pandas as pd
from loguru import logger

from energy_forecast.config import load_config
from energy_forecast.training.multi_seed_trainer import (
    DEFAULT_SEEDS,
    MultiSeedTSMixerxTrainer,
)

# R12 HPO winner (RTX 3090 pod, 2026-04-19, val 1.77% test 1.83%)
R12_BEST_PARAMS: dict = {
    "n_block": 4,
    "ff_dim": 96,
    "dropout": 0.20132633302063577,
    "input_size": 168,
    "revin": False,
    "scaler_type": "standard",
    "learning_rate": 0.002534832369549769,
    "weight_decay": 0.00018468390311739306,
    "windows_batch_size": 512,
    "early_stop_patience_steps": 325,
    "step_size": 24,
    "loss": "huber_0.5",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="R12 FAZ 6 multi-seed runner")
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=DEFAULT_SEEDS,
        help=f"Seeds to train (default: {DEFAULT_SEEDS})",
    )
    parser.add_argument(
        "--wbs", type=int, default=None,
        help="Override windows_batch_size (4060 Ti: 256, pod: 512)",
    )
    parser.add_argument(
        "--no-deterministic", action="store_true",
        help="Disable deterministic algorithms (faster, less reproducible)",
    )
    parser.add_argument(
        "--report", type=Path, default=Path("debug/r12_research/06_multi_seed_results.json"),
        help="Output report path",
    )
    parser.add_argument(
        "--preds-dir",
        type=Path,
        default=Path("debug/r12_research/06_multi_seed_preds"),
        help=(
            "Per-seed OOF predictions output directory (Lesson 166 fix). "
            "Each seed saved as seed_{N}.npz with val_pred/val_actual/"
            "test_pred/test_actual arrays. Enables post-mortem bootstrap "
            "and per-split Jensen analysis without re-training."
        ),
    )
    parser.add_argument(
        "--no-save-preds",
        action="store_true",
        help="Skip OOF prediction persistence (save disk, lose retrospective analysis).",
    )
    args = parser.parse_args()

    logger.info("=== R12 FAZ 6 Multi-Seed Runner ===")
    settings = load_config(Path("configs"))
    df = pd.read_parquet("data/processed/features_historical.parquet")
    logger.info("Data loaded: {} rows, {} cols", *df.shape)

    params = dict(R12_BEST_PARAMS)
    if args.wbs is not None:
        original = params["windows_batch_size"]
        params["windows_batch_size"] = args.wbs
        logger.warning("OVERRIDE windows_batch_size: {} -> {}", original, args.wbs)

    logger.info("Seeds: {}", args.seeds)
    logger.info("Deterministic: {}", not args.no_deterministic)
    logger.info("Best params: {}", params)

    start = time.monotonic()
    trainer = MultiSeedTSMixerxTrainer(
        settings, seeds=args.seeds, deterministic=not args.no_deterministic,
    )
    result = trainer.run(df, params)
    elapsed = time.monotonic() - start

    # Report
    print("\n" + "=" * 60)
    print("=== R12 FAZ 6 — Multi-Seed Sonuc ===")
    print("=" * 60)
    print(f"Seeds:                  {args.seeds}")
    print(f"Per-seed val MAPE:      {[f'{m:.3f}%' for m in result.seed_val_mapes]}")
    print(f"Per-seed test MAPE:     {[f'{m:.3f}%' for m in result.seed_test_mapes]}")
    print(f"Variance sigma val:     {np.std(result.seed_val_mapes):.4f}%")
    print(f"Variance sigma test:    {np.std(result.seed_test_mapes):.4f}%")
    print(f"Naive avg val/test:     {result.naive_avg_val_mape:.3f}% / "
          f"{result.naive_avg_test_mape:.3f}%")
    print(f"Ensemble val (Jensen):  {result.ensemble_val_mape:.3f}%")
    print(f"Ensemble test (Jensen): {result.ensemble_test_mape:.3f}%")
    print(f"Total time:             {elapsed/60:.1f} min")
    print(f"Per-seed avg time:      {elapsed/(60*len(args.seeds)):.1f} min")
    print(f"Models saved to:        {result.seed_models_dir}")
    print("=" * 60)

    # Save JSON report
    report = {
        "seeds": list(args.seeds),
        "best_params": params,
        "deterministic": not args.no_deterministic,
        "per_seed_val_mape": result.seed_val_mapes,
        "per_seed_test_mape": result.seed_test_mapes,
        "std_val_mape": float(np.std(result.seed_val_mapes)),
        "std_test_mape": float(np.std(result.seed_test_mapes)),
        "naive_avg_val_mape": result.naive_avg_val_mape,
        "naive_avg_test_mape": result.naive_avg_test_mape,
        "ensemble_val_mape": result.ensemble_val_mape,
        "ensemble_test_mape": result.ensemble_test_mape,
        "training_time_seconds": elapsed,
        "training_time_minutes": elapsed / 60,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, default=str))
    logger.info("Report saved to {}", args.report)

    # Persist per-seed OOF predictions (Lesson 166 fix).
    # Previously these lived only in memory and were lost after exit —
    # forcing 108-min retrain for post-mortem bootstrap/per-split analyses.
    if not args.no_save_preds and result.seed_predictions:
        args.preds_dir.mkdir(parents=True, exist_ok=True)
        for seed, preds in result.seed_predictions.items():
            np.savez_compressed(
                args.preds_dir / f"seed_{seed}.npz",
                val_pred=preds["val_pred"],
                val_actual=preds["val_actual"],
                test_pred=preds["test_pred"],
                test_actual=preds["test_actual"],
            )
        logger.info(
            "Persisted per-seed OOF predictions ({} seeds) -> {}",
            len(result.seed_predictions),
            args.preds_dir,
        )


if __name__ == "__main__":
    main()
