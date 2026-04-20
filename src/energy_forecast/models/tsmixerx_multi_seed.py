"""Multi-seed TSMixerx ensemble forecaster (R12 FAZ 7).

Wraps N independent TSMixerxForecaster instances trained with different random
seeds. At inference time, runs all seeds and averages predictions (Jensen
ensemble). Target test MAPE 1.649% — R7 ceiling 1.82% minus -0.171%.

Pattern: Lakshminarayanan et al. 2017 "Deep Ensembles" — independent init +
SGD trajectory diversity -> variance reduction by sigma/sqrt(k_eff).

Directory convention::

    {path}/
        metadata.json          # optional top-level summary
        seed_{N}/              # standard NF checkpoint dir per seed
            TSMixerx_0.ckpt
            metadata.json
            configuration.pkl
            dataset.pkl
            alias_to_model.pkl

Graceful degrade: If individual seeds fail to load, continue with remaining
(requires >= 1 loaded). Warnings logged and exposed via get_seed_info().
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from numpy.typing import NDArray

from energy_forecast.config import TSMixerxConfig
from energy_forecast.models.base import PREDICTION_COL, BaseForecaster
from energy_forecast.models.tsmixerx import TSMixerxForecaster


class MultiSeedTSMixerxForecaster(BaseForecaster):
    """N-seed TSMixerx ensemble with Jensen averaging.

    Loads multiple TSMixerxForecaster instances from ``seed_*/`` subdirectories
    and averages their predictions at inference time.

    Args:
        config: TSMixerxConfig from the canonical (first-loaded) seed.

    Attributes:
        _tsmixerx_config: Reference config (proxied from first loaded seed).
        _forecasters: List of loaded TSMixerxForecaster instances.
        _seed_paths: Directory paths of successfully loaded seeds.
        _seeds_requested: Names of all ``seed_*/`` dirs discovered.
        _seeds_loaded: Names of seeds that loaded successfully.
        _top_metadata: Contents of top-level metadata.json (if present).
    """

    METADATA_FILENAME = "metadata.json"

    def __init__(self, config: TSMixerxConfig) -> None:
        super().__init__(config.model_dump())
        self._tsmixerx_config = config
        self._forecasters: list[TSMixerxForecaster] = []
        self._seed_paths: list[Path] = []
        self._seeds_requested: list[str] = []
        self._seeds_loaded: list[str] = []
        self._top_metadata: dict[str, Any] = {}
        # Predict-time telemetry — populated by each predict() call so
        # get_seed_info() can report the actual request outcome, not just
        # the stale load-time count (audit finding K1).
        self._last_request_n_succeeded: int | None = None
        self._last_request_failed_seeds: list[str] = []

    # ------------------------------------------------------------------
    # Status / observability
    # ------------------------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        """At least one seed must be loaded."""
        return len(self._forecasters) >= 1

    @property
    def n_seeds_loaded(self) -> int:
        """Count of seeds currently loaded and usable for prediction."""
        return len(self._forecasters)

    @property
    def n_seeds_requested(self) -> int:
        """Count of ``seed_*/`` subdirectories discovered at load time."""
        return len(self._seeds_requested)

    @property
    def is_degraded(self) -> bool:
        """True if fewer seeds loaded than discovered (graceful degrade occurred)."""
        return self.n_seeds_loaded < self.n_seeds_requested

    def get_seed_info(self) -> dict[str, Any]:
        """Return observability metadata about loaded seeds.

        Includes last-request runtime telemetry so lineage records reflect
        predict-time degradation, not just load-time status (audit K1).

        Returns:
            Dict with keys: ensemble_type, n_requested, n_loaded,
            is_degraded, seeds_requested, seeds_loaded,
            last_request_n_succeeded (None if no predict() yet),
            last_request_failed_seeds.
        """
        return {
            "ensemble_type": "multi_seed_jensen",
            "n_requested": self.n_seeds_requested,
            "n_loaded": self.n_seeds_loaded,
            "is_degraded": self.is_degraded,
            "seeds_requested": list(self._seeds_requested),
            "seeds_loaded": list(self._seeds_loaded),
            "last_request_n_succeeded": self._last_request_n_succeeded,
            "last_request_failed_seeds": list(self._last_request_failed_seeds),
        }

    # ------------------------------------------------------------------
    # Load / from_checkpoint
    # ------------------------------------------------------------------

    @classmethod
    def from_checkpoint(cls, path: Path | str) -> MultiSeedTSMixerxForecaster:
        """Load all ``seed_*/`` subdirectories from path.

        Graceful degrade: individual seed load failures are logged as warnings
        but do not abort. Only fails if zero seeds load successfully.

        Args:
            path: Directory containing ``seed_*/`` subdirectories.

        Returns:
            Initialized MultiSeedTSMixerxForecaster.

        Raises:
            FileNotFoundError: If no ``seed_*/`` subdirectories are found.
            RuntimeError: If every discovered seed failed to load, or if
                configs disagree across seeds (e.g. different input_size).
        """
        path = Path(path)
        seed_dirs = sorted(path.glob("seed_*"))
        if not seed_dirs:
            msg = (
                f"No seed_*/ subdirectories found in {path}. "
                "Expected structure: {path}/seed_42/, {path}/seed_123/, ..."
            )
            raise FileNotFoundError(msg)

        logger.info(
            "Discovering multi-seed TSMixerx in {} ({} seed dirs found)",
            path,
            len(seed_dirs),
        )

        forecasters: list[TSMixerxForecaster] = []
        loaded_paths: list[Path] = []
        loaded_names: list[str] = []
        canonical_config: TSMixerxConfig | None = None

        for seed_dir in seed_dirs:
            seed_name = seed_dir.name
            try:
                forecaster = TSMixerxForecaster.from_checkpoint(seed_dir)
                this_config = forecaster._tsmixerx_config

                # Config consistency check across seeds
                if canonical_config is None:
                    canonical_config = this_config
                else:
                    cls._verify_config_match(canonical_config, this_config, seed_name)

                forecasters.append(forecaster)
                loaded_paths.append(seed_dir)
                loaded_names.append(seed_name)
                logger.info(
                    "Loaded {} ({}/{})",
                    seed_name,
                    len(forecasters),
                    len(seed_dirs),
                )
            except Exception as e:
                logger.warning(
                    "Failed to load {}: {} (continuing, graceful degrade)",
                    seed_name,
                    e,
                )

        if not forecasters:
            msg = (
                f"All {len(seed_dirs)} seeds failed to load from {path}. "
                "Cannot initialize ensemble — check logs for per-seed errors."
            )
            raise RuntimeError(msg)

        n_loaded = len(forecasters)
        n_requested = len(seed_dirs)
        if n_loaded < n_requested:
            logger.warning(
                "MULTI-SEED DEGRADED: loaded {}/{} seeds. "
                "Jensen averaging will use {} models (expected {}).",
                n_loaded,
                n_requested,
                n_loaded,
                n_requested,
            )

        top_metadata: dict[str, Any] = {}
        top_meta_path = path / cls.METADATA_FILENAME
        if top_meta_path.exists():
            try:
                top_metadata = json.loads(top_meta_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as e:
                logger.debug("Top-level metadata load skipped: {}", e)

        assert canonical_config is not None
        instance = cls(canonical_config)
        instance._forecasters = forecasters
        instance._seed_paths = loaded_paths
        instance._seeds_requested = [d.name for d in seed_dirs]
        instance._seeds_loaded = loaded_names
        instance._top_metadata = top_metadata

        logger.info(
            "Multi-seed TSMixerx initialized: {} seeds active (degraded={})",
            n_loaded,
            instance.is_degraded,
        )
        return instance

    @staticmethod
    def _verify_config_match(
        canonical: TSMixerxConfig,
        other: TSMixerxConfig,
        seed_name: str,
    ) -> None:
        """Reject seeds whose architecture/covariates disagree with canonical.

        Different input_size or covariate set across seeds breaks Jensen
        averaging (predictions would align on different input windows).

        Raises:
            RuntimeError: If architecture or covariates differ.
        """
        if canonical.architecture.input_size != other.architecture.input_size:
            msg = (
                f"Config mismatch in {seed_name}: "
                f"input_size {other.architecture.input_size} "
                f"vs canonical {canonical.architecture.input_size}"
            )
            raise RuntimeError(msg)
        if list(canonical.covariates.futr_exog) != list(other.covariates.futr_exog):
            msg = f"Config mismatch in {seed_name}: futr_exog list differs"
            raise RuntimeError(msg)
        if list(canonical.covariates.hist_exog) != list(other.covariates.hist_exog):
            msg = f"Config mismatch in {seed_name}: hist_exog list differs"
            raise RuntimeError(msg)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, X: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        """Jensen ensemble prediction — mean of per-seed predictions.

        Jensen's inequality applies: mean(MAPE(preds_i, y)) >= MAPE(mean(preds_i), y)
        for convex metrics. FAZ 6 empirically confirmed -0.15% gain over
        per-seed mean on test set.

        Args:
            X: Feature DataFrame with DatetimeIndex (context + forecast window).
            **kwargs: Forwarded to each TSMixerxForecaster.predict().

        Returns:
            DataFrame with PREDICTION_COL (mean across loaded seeds).

        Raises:
            RuntimeError: If no seeds loaded OR every per-seed predict fails.
        """
        if not self.is_fitted:
            msg = "Multi-seed forecaster has no loaded seeds. Call from_checkpoint() first."
            raise RuntimeError(msg)

        per_seed_preds, output_index = self._predict_per_seed(X, **kwargs)

        # Jensen averaging: average PREDICTIONS, then derive metric upstream.
        stacked = np.stack(per_seed_preds, axis=0)
        mean_pred = np.mean(stacked, axis=0)

        result = pd.DataFrame(
            {PREDICTION_COL: mean_pred},
            index=output_index,
        )
        return result

    def _predict_per_seed(
        self,
        X: pd.DataFrame,
        **kwargs: Any,
    ) -> tuple[list[NDArray[np.floating[Any]]], pd.Index]:
        """Run predict on each loaded seed, collect prediction arrays.

        Graceful degrade at predict-time: a runtime failure in one seed is
        logged and skipped. If every seed fails, raises RuntimeError.

        Telemetry: updates ``_last_request_n_succeeded`` and
        ``_last_request_failed_seeds`` so ``get_seed_info()`` reports the
        actual request outcome (audit K1 fix).

        Returns:
            (predictions_list, reference_index) tuple. reference_index comes
            from the first successful prediction for output DataFrame shape.
        """
        predictions: list[NDArray[np.floating[Any]]] = []
        reference_index: pd.Index | None = None
        failed_seeds: list[str] = []

        for name, forecaster in zip(
            self._seeds_loaded, self._forecasters, strict=True
        ):
            try:
                pred_df = forecaster.predict(X, **kwargs)
                if reference_index is None:
                    reference_index = pred_df.index
                predictions.append(
                    np.asarray(pred_df[PREDICTION_COL].values, dtype=np.float64)
                )
            except Exception as e:
                failed_seeds.append(name)
                logger.warning(
                    "Predict-time failure for {}: {} (skipping)", name, e
                )

        # Update telemetry BEFORE raising so observers can still query
        # get_seed_info() to see what happened on a total failure.
        self._last_request_n_succeeded = len(predictions)
        self._last_request_failed_seeds = failed_seeds

        if not predictions or reference_index is None:
            msg = "All per-seed predictions failed at runtime."
            raise RuntimeError(msg)

        if failed_seeds:
            logger.warning(
                "Predict-time graceful degrade: {}/{} seeds succeeded "
                "(failed: {})",
                len(predictions),
                len(self._seeds_loaded),
                failed_seeds,
            )

        return predictions, reference_index

    def get_per_seed_predictions(
        self,
        X: pd.DataFrame,
        **kwargs: Any,
    ) -> dict[str, NDArray[np.floating[Any]]]:
        """Return per-seed predictions as a dict for debugging / analytics.

        Unlike predict(), this does NOT average — callers get raw per-seed
        arrays to inspect seed agreement, compute custom aggregations, etc.

        Args:
            X: Feature DataFrame.

        Returns:
            Dict mapping seed_name -> prediction array.
        """
        result: dict[str, NDArray[np.floating[Any]]] = {}
        for name, forecaster in zip(
            self._seeds_loaded, self._forecasters, strict=True
        ):
            pred_df = forecaster.predict(X, **kwargs)
            result[name] = np.asarray(
                pred_df[PREDICTION_COL].values, dtype=np.float64
            )
        return result

    # ------------------------------------------------------------------
    # Unsupported abstract methods (managed via trainer / from_checkpoint)
    # ------------------------------------------------------------------

    def train(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame | None = None,
        **kwargs: Any,
    ) -> dict[str, float] | None:
        """Training is managed by MultiSeedTSMixerxTrainer, not this class."""
        msg = (
            "MultiSeedTSMixerxForecaster.train() is not supported. "
            "Use MultiSeedTSMixerxTrainer to train seed models, then load "
            "the ensemble via MultiSeedTSMixerxForecaster.from_checkpoint()."
        )
        raise NotImplementedError(msg)

    def save(self, path: Path) -> None:
        """Saving individual seeds is handled by each TSMixerxForecaster.save()."""
        msg = (
            "MultiSeedTSMixerxForecaster.save() is not supported. "
            "Per-seed checkpoints are saved by MultiSeedTSMixerxTrainer during "
            "training; this class only loads them via from_checkpoint()."
        )
        raise NotImplementedError(msg)

    def load(self, path: Path) -> None:
        """Use the ``from_checkpoint`` classmethod to load the ensemble."""
        msg = (
            "MultiSeedTSMixerxForecaster.load() is not supported — "
            "use MultiSeedTSMixerxForecaster.from_checkpoint(path) instead."
        )
        raise NotImplementedError(msg)
