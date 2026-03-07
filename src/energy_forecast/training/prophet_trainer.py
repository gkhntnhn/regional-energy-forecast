"""Prophet training pipeline: TSCV + Optuna + MLflow.

Orchestrates hyperparameter optimization via Optuna, cross-validated
training on calendar-month splits, and final model training on all data.

Uses the same shared infrastructure as CatBoostTrainer:
- TimeSeriesSplitter for calendar-month TSCV
- suggest_params for dynamic Optuna search space
- compute_all / MetricsResult for metrics
- ExperimentTracker for MLflow logging
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
from loguru import logger
from optuna import Study, Trial, TrialPruned, create_study
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from prophet import Prophet

from energy_forecast.config.settings import Settings
from energy_forecast.models.prophet import ProphetForecaster
from energy_forecast.training.experiment import ExperimentTracker
from energy_forecast.training.metrics import MetricsResult, compute_all
from energy_forecast.training.search import suggest_params
from energy_forecast.training.splitter import SplitInfo, TimeSeriesSplitter

# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProphetSplitResult:
    """Result from a single CV split."""

    split_idx: int
    train_metrics: MetricsResult
    val_metrics: MetricsResult
    test_metrics: MetricsResult
    val_month: str
    test_month: str
    val_predictions: np.ndarray[Any, np.dtype[np.floating[Any]]] | None = None
    val_actuals: np.ndarray[Any, np.dtype[np.floating[Any]]] | None = None
    test_predictions: np.ndarray[Any, np.dtype[np.floating[Any]]] | None = None
    test_actuals: np.ndarray[Any, np.dtype[np.floating[Any]]] | None = None


@dataclass(frozen=True)
class ProphetTrainingResult:
    """Aggregated result across all CV splits."""

    split_results: list[ProphetSplitResult]
    avg_val_mape: float
    avg_test_mape: float
    std_val_mape: float
    regressor_names: list[str]


@dataclass(frozen=True)
class ProphetPipelineResult:
    """Full training pipeline result."""

    study: Study
    best_params: dict[str, Any]
    training_result: ProphetTrainingResult
    final_model: Prophet
    training_time_seconds: float


# ---------------------------------------------------------------------------
# ProphetTrainer
# ---------------------------------------------------------------------------


class ProphetTrainer:
    """Prophet training pipeline with TSCV, Optuna, and MLflow.

    Follows the same pattern as CatBoostTrainer, using shared M5 infrastructure.

    Args:
        settings: Full application settings.
        tracker: MLflow experiment tracker (disabled by default).
    """

    def __init__(
        self,
        settings: Settings,
        tracker: ExperimentTracker | None = None,
    ) -> None:
        self._settings = settings
        self._prophet_config = settings.prophet
        self._hp_config = settings.hyperparameters
        self._search_config = settings.hyperparameters.prophet
        self._tracker = tracker or ExperimentTracker(enabled=False)
        self._splitter = TimeSeriesSplitter.from_config(settings.hyperparameters.cross_validation)
        self._target_col = settings.hyperparameters.target_col
        self._skip_validation = settings.hyperparameters.skip_validation_after_optuna
        self._regressor_names: list[str] = []
        self._holidays_df = self._load_holidays()  # load once, reuse across splits

    # -- Optuna storage --

    def _optuna_storage(self, model_name: str) -> optuna.storages.RDBStorage | str | None:
        """Return Optuna storage: PostgreSQL if available, else SQLite."""
        if self._search_config.n_trials <= 3:
            return None
        db_url = os.environ.get("DATABASE_URL_SYNC", "")
        if db_url:
            return optuna.storages.RDBStorage(
                url=db_url,
                engine_kwargs={"pool_size": 1, "max_overflow": 0},
            )
        studies_dir = Path(self._settings.paths.models_dir) / "optuna_studies"
        studies_dir.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{studies_dir / model_name}.db"

    # -- Prophet format conversion --

    def _to_prophet_format(
        self,
        df: pd.DataFrame,
        include_target: bool = True,
    ) -> pd.DataFrame:
        """Convert feature DataFrame to Prophet ds+y+regressors format.

        Args:
            df: DataFrame with DatetimeIndex.
            include_target: If True, include y column (for training).

        Returns:
            Prophet-formatted DataFrame.
        """
        prophet_df = pd.DataFrame()
        prophet_df["ds"] = df.index

        if include_target:
            prophet_df["y"] = df[self._target_col].values

        # Add regressors from config
        for reg in self._prophet_config.regressors:
            if reg.name in df.columns:
                prophet_df[reg.name] = df[reg.name].values
                if reg.name not in self._regressor_names:
                    self._regressor_names.append(reg.name)
            else:
                logger.warning(
                    "Regressor '{}' configured but not in DataFrame columns", reg.name
                )

        # Drop rows with NaN in regressor columns (e.g. consumption_lag_168
        # has 168 NaN rows at the dataset start where the lag window exceeds
        # available history).  Prophet raises ValueError on NaN regressors.
        n_before = len(prophet_df)
        prophet_df = prophet_df.dropna(subset=self._regressor_names).reset_index(drop=True)
        n_dropped = n_before - len(prophet_df)
        if n_dropped > 0:
            logger.info(
                "Dropped {} rows with NaN regressors ({:.1f}% of {})",
                n_dropped, n_dropped / n_before * 100, n_before,
            )

        return prophet_df

    # -- Prophet model creation --

    def _create_prophet(self, params: dict[str, Any]) -> Prophet:
        """Create Prophet model with given parameters.

        Only adds regressors that were found in the data (populated by
        ``_to_prophet_format``).  This prevents Prophet from raising
        ``ValueError`` when a configured regressor is missing from the
        DataFrame.

        Args:
            params: Hyperparameters (from Optuna or defaults).

        Returns:
            Configured Prophet model (unfitted).
        """
        cfg = self._prophet_config

        # Fixed parameters from config
        # Disable auto-detect seasonality — we add them manually with config
        # fourier_order values (daily=12, weekly=6, yearly=10 by default)
        model_params: dict[str, Any] = {
            "daily_seasonality": False,
            "weekly_seasonality": False,
            "yearly_seasonality": False,
            "uncertainty_samples": cfg.uncertainty.mcmc_samples,
            "interval_width": cfg.uncertainty.interval_width,
            "n_changepoints": cfg.changepoint.n_changepoints,
        }

        # Override with Optuna-suggested params
        model_params.update(params)

        model = Prophet(**model_params)

        # Seasonality mode fixed from config (energy data is multiplicative).
        # Not part of Optuna search space, but kept defensive for manual overrides.
        season_mode: str = params.get("seasonality_mode", cfg.seasonality.mode)

        # Add seasonalities with config-specified Fourier orders
        model.add_seasonality(
            name="daily",
            period=1,
            fourier_order=cfg.seasonality.daily.fourier_order,
            mode=season_mode,
        )
        model.add_seasonality(
            name="weekly",
            period=7,
            fourier_order=cfg.seasonality.weekly.fourier_order,
            mode=season_mode,
        )
        model.add_seasonality(
            name="yearly",
            period=365.25,
            fourier_order=cfg.seasonality.yearly.fourier_order,
            mode=season_mode,
        )

        # Add only regressors confirmed present in data
        for reg in cfg.regressors:
            if reg.name in self._regressor_names:
                model.add_regressor(reg.name, mode=reg.mode)
            else:
                logger.warning("Regressor '{}' not found in data, skipping", reg.name)

        return model

    # -- Holidays loading --

    def _load_holidays(self) -> pd.DataFrame | None:
        """Load holidays from parquet file with window parameters.

        Bayram (religious holiday) days get lower/upper windows to capture
        the eve and day-after effects on electricity consumption.

        Returns:
            DataFrame with 'ds', 'holiday', 'lower_window', 'upper_window'
            columns, or None if not found.
        """
        holidays_path = Path(self._settings.data_loader.paths.holidays)
        if not holidays_path.exists():
            logger.warning("Holidays file not found: {}", holidays_path)
            return None

        df = pd.read_parquet(holidays_path)

        # Handle column naming variations
        rename_map: dict[str, str] = {}
        if "date" in df.columns:
            rename_map["date"] = "ds"
        if "holiday_name" in df.columns:
            rename_map["holiday_name"] = "holiday"

        if rename_map:
            df = df.rename(columns=rename_map)

        df["ds"] = pd.to_datetime(df["ds"])

        # Holiday windows: different holidays have different spill-over effects
        # on electricity consumption patterns.
        is_bayram = df["holiday"].str.contains("Bayrami", na=False)
        is_ramazan = df["holiday"] == "Ramazan"
        # Resmi tatiller: not bayram, not fasting days
        is_resmi = ~is_bayram & ~is_ramazan

        # Bayram: eve effect (lower=-1) + day-after (upper=1)
        # Resmi tatiller: day-after effect only (upper=1)
        # Ramazan fasting: no spill-over (0, 0)
        df["lower_window"] = np.where(is_bayram, -1, 0)
        df["upper_window"] = np.where(is_bayram, 1, np.where(is_resmi, 1, 0))

        cols = ["ds", "holiday", "lower_window", "upper_window"]
        available = [c for c in cols if c in df.columns]
        return df[available]

    # -- Single split training --

    def _train_split(
        self,
        split_info: SplitInfo,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        params: dict[str, Any],
    ) -> ProphetSplitResult:
        """Train Prophet on a single CV split.

        Args:
            split_info: Split boundary information.
            train_df: Training data.
            val_df: Validation data.
            test_df: Test data.
            params: Hyperparameters.

        Returns:
            ProphetSplitResult with metrics.
        """
        # Convert to Prophet format
        train_prophet = self._to_prophet_format(train_df, include_target=True)
        val_prophet = self._to_prophet_format(val_df, include_target=True)
        test_prophet = self._to_prophet_format(test_df, include_target=True)

        # Create and fit model
        model = self._create_prophet(params)

        if self._holidays_df is not None:
            model.holidays = self._holidays_df

        model.fit(train_prophet)

        # Predictions
        train_forecast = model.predict(train_prophet)
        val_forecast = model.predict(val_prophet)
        test_forecast = model.predict(test_prophet)

        train_pred = np.asarray(train_forecast["yhat"].values, dtype=np.float64)
        val_pred = np.asarray(val_forecast["yhat"].values, dtype=np.float64)
        test_pred = np.asarray(test_forecast["yhat"].values, dtype=np.float64)

        # Metrics
        y_train = np.asarray(train_prophet["y"].values, dtype=np.float64)
        y_val = np.asarray(val_prophet["y"].values, dtype=np.float64)
        y_test = np.asarray(test_prophet["y"].values, dtype=np.float64)

        return ProphetSplitResult(
            split_idx=split_info.split_idx,
            train_metrics=compute_all(y_train, train_pred),
            val_metrics=compute_all(y_val, val_pred),
            test_metrics=compute_all(y_test, test_pred),
            val_month=split_info.val_start.strftime("%Y-%m"),
            test_month=split_info.test_start.strftime("%Y-%m"),
            val_predictions=val_pred,
            val_actuals=y_val,
            test_predictions=test_pred,
            test_actuals=y_test,
        )

    # -- All splits training --

    def _train_all_splits(
        self,
        df: pd.DataFrame,
        params: dict[str, Any],
        trial: Trial | None = None,
    ) -> ProphetTrainingResult:
        """Train on all TSCV splits and aggregate results.

        Args:
            df: Full feature-engineered DataFrame.
            params: Hyperparameters.
            trial: Optuna trial for pruning (None during final validation).

        Returns:
            ProphetTrainingResult with aggregated metrics.
        """
        results: list[ProphetSplitResult] = []

        for fold_idx, (info, train_df, val_df, test_df) in enumerate(
            self._splitter.iter_splits(df)
        ):
            result = self._train_split(info, train_df, val_df, test_df, params)
            results.append(result)
            logger.info(
                "Split {} | val={} MAPE={:.2f}% | test={} MAPE={:.2f}%",
                result.split_idx,
                result.val_month,
                result.val_metrics.mape,
                result.test_month,
                result.test_metrics.mape,
            )

            if trial is not None:
                trial.report(result.val_metrics.mape, fold_idx)
                if trial.should_prune():
                    raise TrialPruned()

        val_mapes = [r.val_metrics.mape for r in results]
        test_mapes = [r.test_metrics.mape for r in results]

        return ProphetTrainingResult(
            split_results=results,
            avg_val_mape=float(np.mean(val_mapes)),
            avg_test_mape=float(np.mean(test_mapes)),
            std_val_mape=float(np.std(val_mapes)),
            regressor_names=list(self._regressor_names),
        )

    # -- Optuna objective (dynamic from YAML) --

    def _create_objective(
        self, df: pd.DataFrame
    ) -> tuple[Callable[[Trial], float], dict[int, ProphetTrainingResult]]:
        """Create Optuna objective using dynamic YAML search space.

        Returns:
            Tuple of (objective function, trial results cache).
        """
        search_space = self._search_config.search_space
        trial_results: dict[int, ProphetTrainingResult] = {}

        def objective(trial: Trial) -> float:
            suggested = suggest_params(trial, search_space)
            result = self._train_all_splits(df, suggested, trial=trial)
            trial.set_user_attr("avg_test_mape", result.avg_test_mape)
            trial_results[trial.number] = result
            return result.avg_val_mape

        return objective, trial_results

    # -- Optimize --

    def optimize(self, df: pd.DataFrame) -> tuple[Study, ProphetTrainingResult]:
        """Run Optuna hyperparameter optimization.

        Args:
            df: Feature-engineered DataFrame.

        Returns:
            Tuple of (study, best_trial_result).
        """
        storage = self._optuna_storage("prophet")
        study = create_study(
            study_name="prophet",
            direction="minimize",
            storage=storage,
            load_if_exists=True,
            sampler=TPESampler(seed=self._prophet_config.optimization.random_seed),
            pruner=MedianPruner(n_startup_trials=3, n_warmup_steps=2),
        )

        objective, trial_results = self._create_objective(df)
        study.optimize(objective, n_trials=self._search_config.n_trials)

        logger.info(
            "Optimization done — best val MAPE: {:.2f}%, params: {}",
            study.best_value,
            study.best_params,
        )

        best_trial_num = study.best_trial.number

        if best_trial_num in trial_results:
            best_result = trial_results[best_trial_num]
            logger.info("Using cached predictions from trial {}", best_trial_num)
        elif self._skip_validation:
            logger.info("Skipping post-Optuna validation (skip_validation_after_optuna=true)")
            best_result = ProphetTrainingResult(
                split_results=[],
                avg_val_mape=study.best_value,
                avg_test_mape=float(
                    study.best_trial.user_attrs.get("avg_test_mape", float("nan"))
                ),
                std_val_mape=0.0,
                regressor_names=list(self._regressor_names),
            )
        else:
            logger.warning("Cache miss for trial {}, retraining", best_trial_num)
            best_result = self._train_all_splits(df, study.best_params)

        return study, best_result

    # -- Final model --

    def train_final(self, df: pd.DataFrame, params: dict[str, Any]) -> Prophet:
        """Train final model on all data with best params.

        Args:
            df: Full dataset.
            params: Best hyperparameters from optimization.

        Returns:
            Trained Prophet model.
        """
        train_prophet = self._to_prophet_format(df, include_target=True)
        model = self._create_prophet(params)

        if self._holidays_df is not None:
            model.holidays = self._holidays_df

        model.fit(train_prophet)
        logger.info("Final Prophet model trained on {} samples", len(df))
        return model

    # -- Full pipeline --

    def run(self, df: pd.DataFrame) -> ProphetPipelineResult:
        """Execute full training pipeline: optimize + final model + MLflow.

        Args:
            df: Feature-engineered DataFrame (pipeline output).

        Returns:
            ProphetPipelineResult with study, final model, and metrics.
        """
        start = time.monotonic()

        with self._tracker.start_run("prophet_optimization"):
            study, best_result = self.optimize(df)
            self._tracker.log_params(study.best_params)
            self._tracker.log_metrics(
                {
                    "avg_val_mape": best_result.avg_val_mape,
                    "avg_test_mape": best_result.avg_test_mape,
                    "std_val_mape": best_result.std_val_mape,
                }
            )
            for sr in best_result.split_results:
                self._tracker.log_split_metrics(
                    sr.split_idx, sr.train_metrics, sr.val_metrics, sr.test_metrics
                )

        with self._tracker.start_run("prophet_final"):
            final_model = self.train_final(df, study.best_params)

            # Save model to timestamped subdirectory via ProphetForecaster.save()
            # (consistent naming, metadata with regressor_names, config, SHA256 hash).
            from datetime import datetime

            from energy_forecast.utils import TZ_ISTANBUL

            run_ts = datetime.now(tz=TZ_ISTANBUL).strftime("%Y-%m-%d_%H-%M")
            model_dir = (
                Path(self._settings.paths.models_dir)
                / "prophet"
                / f"prophet_{run_ts}"
            )
            model_dir.mkdir(parents=True, exist_ok=True)
            forecaster = ProphetForecaster(self._prophet_config.model_dump())
            forecaster.set_model(final_model, regressor_names=list(self._regressor_names))
            forecaster.save(model_dir)

            logger.info("Model saved to {}", model_dir)

            self._tracker.log_prophet_model(final_model, "prophet_model")

        elapsed = time.monotonic() - start
        logger.info("Prophet pipeline complete in {:.1f}s", elapsed)

        return ProphetPipelineResult(
            study=study,
            best_params=study.best_params,
            training_result=best_result,
            final_model=final_model,
            training_time_seconds=elapsed,
        )
