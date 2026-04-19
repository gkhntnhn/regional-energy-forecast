"""Multi-seed TSMixerx ensemble trainer for R12 FAZ 6.

Wraps TSMixerxTrainer to train N models with different seeds and average
their predictions. Reduces seed-to-seed variance (~+0.05-0.10% MAPE).

Pattern: Lakshminarayanan et al. 2017 "Deep Ensembles" — independent random
init + SGD trajectory diversity → variance reduction by sigma/sqrt(k_eff).

Usage (FAZ 6):
    from energy_forecast.training.multi_seed_trainer import (
        MultiSeedTSMixerxTrainer, DEFAULT_SEEDS,
    )
    trainer = MultiSeedTSMixerxTrainer(settings, seeds=DEFAULT_SEEDS)
    result = trainer.run(df, best_params=r12_best_params)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from loguru import logger
from numpy.typing import NDArray

from energy_forecast.config import Settings
from energy_forecast.training.experiment import ExperimentTracker
from energy_forecast.training.metrics import compute_all
from energy_forecast.training.tsmixerx_trainer import TSMixerxTrainer

# R12 default seeds — diverse selection (mix of standard 42 + arbitrary primes)
DEFAULT_SEEDS: list[int] = [42, 123, 456, 789, 2026]


@dataclass(frozen=True)
class MultiSeedResult:
    """Per-seed metrics + ensemble (averaged-prediction) metrics."""

    seeds: list[int]
    seed_val_mapes: list[float]
    seed_test_mapes: list[float]
    ensemble_val_mape: float          # MAPE on averaged predictions (Jensen)
    ensemble_test_mape: float
    naive_avg_val_mape: float          # mean of per-seed MAPE (for comparison)
    naive_avg_test_mape: float
    seed_models_dir: Path              # Per-seed checkpoints saved here
    training_time_seconds: float
    seed_predictions: dict[int, dict[str, NDArray[Any]]] = field(default_factory=dict)


class MultiSeedTSMixerxTrainer:
    """Train N TSMixerx models with different seeds, average predictions.

    Args:
        settings: Full application settings.
        seeds: List of random seeds (default: [42, 123, 456, 789, 2026]).
        tracker: MLflow experiment tracker (disabled by default).
        deterministic: Enable PyTorch deterministic algorithms (5-10% slowdown).

    R12 FAZ 6 implementation note:
        Per-seed predictions stored in MultiSeedResult.seed_predictions for
        offline analysis. Ensemble MAPE computed on averaged predictions
        (NOT mean of per-seed MAPE — see Jensen's inequality).
    """

    def __init__(
        self,
        settings: Settings,
        seeds: list[int] | None = None,
        tracker: ExperimentTracker | None = None,
        *,
        deterministic: bool = True,
    ) -> None:
        self._settings = settings
        self._seeds = seeds if seeds is not None else DEFAULT_SEEDS
        self._tracker = tracker or ExperimentTracker(enabled=False)
        self._deterministic = deterministic
        self._target_col = settings.hyperparameters.target_col

    def _enable_determinism(self) -> None:
        """Configure PyTorch for reproducible runs (R12 FAZ 6)."""
        if not self._deterministic:
            return
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
            # CUBLAS workspace required for deterministic CUDA matmul
            import os
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            logger.info("Determinism enabled (5-10%% slowdown expected)")
        except Exception as e:  # pragma: no cover — environment-specific
            logger.warning("Could not enable full determinism: {}", e)

    def run(
        self,
        df: pd.DataFrame,
        best_params: dict[str, Any],
    ) -> MultiSeedResult:
        """Train N seed models, save per-seed checkpoints, compute ensemble metrics.

        Args:
            df: Feature-engineered DataFrame.
            best_params: HPO winner params (from FAZ 5). ``random_seed`` will
                be overridden per iteration.

        Returns:
            MultiSeedResult with per-seed + ensemble metrics.
        """
        start = time.monotonic()
        self._enable_determinism()

        models_dir = Path(self._settings.paths.models_dir)
        seed_models_dir = models_dir / "tsmixerx_multi_seed"
        seed_models_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "Multi-seed training: {} seeds (deterministic={})",
            len(self._seeds), self._deterministic,
        )

        seed_val_mapes: list[float] = []
        seed_test_mapes: list[float] = []
        seed_predictions: dict[int, dict[str, NDArray[Any]]] = {}

        for i, seed in enumerate(self._seeds, start=1):
            logger.info("[{}/{}] Training seed={}", i, len(self._seeds), seed)
            params = {**best_params, "random_seed": seed}

            # Re-instantiate trainer per seed (clean state)
            trainer = TSMixerxTrainer(self._settings, self._tracker)
            cv_result = trainer._train_all_splits(df, params)

            seed_val_mapes.append(cv_result.avg_val_mape)
            seed_test_mapes.append(cv_result.avg_test_mape)

            # Collect per-split predictions (for true ensemble MAPE)
            val_preds = np.concatenate([sr.val_predictions for sr in cv_result.split_results])
            val_actuals = np.concatenate([sr.val_actuals for sr in cv_result.split_results])
            test_preds = np.concatenate([sr.test_predictions for sr in cv_result.split_results])
            test_actuals = np.concatenate([sr.test_actuals for sr in cv_result.split_results])
            seed_predictions[seed] = {
                "val_pred": val_preds, "val_actual": val_actuals,
                "test_pred": test_preds, "test_actual": test_actuals,
            }

            # Save final model per seed (for production ensemble inference)
            final_model = trainer.train_final(df, params)
            seed_dir = seed_models_dir / f"seed_{seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)
            final_model.save(seed_dir)
            logger.info("Seed {} complete: val MAPE={:.3f}% test MAPE={:.3f}%",
                        seed, cv_result.avg_val_mape, cv_result.avg_test_mape)

        # Ensemble MAPE — average predictions, then compute MAPE (Jensen)
        all_val_preds = np.stack([seed_predictions[s]["val_pred"] for s in self._seeds])
        all_test_preds = np.stack([seed_predictions[s]["test_pred"] for s in self._seeds])
        ens_val_pred = np.mean(all_val_preds, axis=0)
        ens_test_pred = np.mean(all_test_preds, axis=0)

        val_actuals = seed_predictions[self._seeds[0]]["val_actual"]
        test_actuals = seed_predictions[self._seeds[0]]["test_actual"]

        ens_val_metrics = compute_all(val_actuals, ens_val_pred)
        ens_test_metrics = compute_all(test_actuals, ens_test_pred)

        elapsed = time.monotonic() - start
        result = MultiSeedResult(
            seeds=list(self._seeds),
            seed_val_mapes=seed_val_mapes,
            seed_test_mapes=seed_test_mapes,
            ensemble_val_mape=ens_val_metrics.mape,
            ensemble_test_mape=ens_test_metrics.mape,
            naive_avg_val_mape=float(np.mean(seed_val_mapes)),
            naive_avg_test_mape=float(np.mean(seed_test_mapes)),
            seed_models_dir=seed_models_dir,
            training_time_seconds=elapsed,
            seed_predictions=seed_predictions,
        )

        logger.info("Multi-seed complete in {:.1f}s", elapsed)
        logger.info("Per-seed val MAPE: {}", [f"{m:.3f}%" for m in seed_val_mapes])
        logger.info("Naive avg val MAPE:    {:.3f}%", result.naive_avg_val_mape)
        logger.info("Ensemble val MAPE:     {:.3f}%  (delta vs naive: {:+.3f}%)",
                    result.ensemble_val_mape,
                    result.ensemble_val_mape - result.naive_avg_val_mape)
        logger.info("Ensemble test MAPE:    {:.3f}%  (delta vs naive: {:+.3f}%)",
                    result.ensemble_test_mape,
                    result.ensemble_test_mape - result.naive_avg_test_mape)
        return result
