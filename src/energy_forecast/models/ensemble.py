"""Weighted ensemble of CatBoost, Prophet, and TFT forecasters."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from loguru import logger

from energy_forecast.config.settings import EnsembleConfig
from energy_forecast.models.base import BaseForecaster
from energy_forecast.models.tft import TFTForecaster

if TYPE_CHECKING:
    from prophet import Prophet


class EnsembleForecaster(BaseForecaster):
    """Weighted-average ensemble of CatBoost, Prophet, and TFT.

    Loads pre-trained models and uses optimized weights from training
    to produce ensemble predictions. Supports dynamic active model selection.

    Args:
        config: Ensemble configuration dictionary containing weights and active models.
            If not provided, defaults are loaded from EnsembleConfig.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        # Load defaults from EnsembleConfig if not provided
        default_cfg = EnsembleConfig()
        if config is None:
            config = {}

        # Merge with defaults
        merged_config = {
            "active_models": config.get("active_models", list(default_cfg.active_models)),
            "weights": config.get("weights", {
                "catboost": default_cfg.weights.catboost,
                "prophet": default_cfg.weights.prophet,
                "tft": default_cfg.weights.tft,
            }),
            "target_col": config.get("target_col", "consumption"),
            "prophet_regressors": config.get("prophet_regressors", []),
        }

        super().__init__(merged_config)
        self._active_models: list[str] = merged_config["active_models"]
        self._weights: dict[str, float] = merged_config["weights"]
        self._catboost_model: CatBoostRegressor | None = None
        self._prophet_model: Prophet | None = None
        self._tft_model: TFTForecaster | None = None
        self._prophet_regressors: list[str] = merged_config["prophet_regressors"]

    @property
    def weights(self) -> dict[str, float]:
        """Get current ensemble weights."""
        return self._weights.copy()

    @property
    def active_models(self) -> list[str]:
        """Get current active models."""
        return self._active_models.copy()

    def set_models(
        self,
        catboost_model: CatBoostRegressor | None = None,
        prophet_model: Prophet | None = None,
        tft_model: TFTForecaster | None = None,
    ) -> None:
        """Set pre-trained models for prediction.

        Args:
            catboost_model: Trained CatBoost model.
            prophet_model: Trained Prophet model.
            tft_model: Trained TFT model.
        """
        if catboost_model is not None:
            self._catboost_model = catboost_model
        if prophet_model is not None:
            self._prophet_model = prophet_model
        if tft_model is not None:
            self._tft_model = tft_model
        logger.info(
            "Ensemble models set: catboost={}, prophet={}, tft={}",
            self._catboost_model is not None,
            self._prophet_model is not None,
            self._tft_model is not None,
        )

    def set_weights(self, weights: dict[str, float]) -> None:
        """Set ensemble weights.

        Args:
            weights: Dictionary with model weights.

        Raises:
            ValueError: If weights don't sum to 1.0.
        """
        total = sum(weights.values())
        if abs(total - 1.0) > 1e-6:
            msg = f"Weights must sum to 1.0, got {total}"
            raise ValueError(msg)
        self._weights = weights.copy()
        logger.info("Ensemble weights updated: {}", self._weights)

    def set_active_models(self, models: list[str]) -> None:
        """Set active models for prediction.

        Args:
            models: List of model names to use.

        Raises:
            ValueError: If unknown model name provided.
        """
        valid = {"catboost", "prophet", "tft"}
        for m in models:
            if m not in valid:
                msg = f"Unknown model: {m}. Valid: {valid}"
                raise ValueError(msg)
        self._active_models = models
        logger.info("Active models set: {}", self._active_models)

    def train(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame | None = None,
        **kwargs: Any,
    ) -> None:
        """Train is not supported — use EnsembleTrainer instead.

        The ensemble training is orchestrated by EnsembleTrainer which
        trains CatBoost, Prophet, and TFT separately, then optimizes weights.

        Args:
            train_df: Training data (unused).
            val_df: Validation data (unused).
            **kwargs: Additional arguments (unused).

        Raises:
            NotImplementedError: Always, use EnsembleTrainer.run() instead.
        """
        msg = (
            "EnsembleForecaster.train() is not supported. "
            "Use EnsembleTrainer.run() to train the ensemble, "
            "then load the trained models with set_models()."
        )
        raise NotImplementedError(msg)

    def predict(
        self,
        X: pd.DataFrame,
        *,
        history: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Generate weighted-average ensemble prediction.

        Uses only active models that have been loaded.

        Args:
            X: Feature DataFrame with DatetimeIndex.
            history: Optional historical data for TFT rolling prediction.
                When provided and len(X) > prediction_length, TFT uses
                predict_rolling() with encoder context from history.
                When None, TFT uses standard single-window predict().

        Returns:
            DataFrame with 'prediction' column and individual model predictions.

        Raises:
            RuntimeError: If no active models are loaded.
        """
        predictions: dict[str, np.ndarray[Any, np.dtype[np.floating[Any]]]] = {}

        # CatBoost prediction
        if "catboost" in self._active_models and self._catboost_model is not None:
            features = X.drop(columns=[self._target_col], errors="ignore")
            predictions["catboost"] = np.asarray(
                self._catboost_model.predict(features), dtype=np.float64
            )

        # Prophet prediction
        if "prophet" in self._active_models and self._prophet_model is not None:
            prophet_df = self._to_prophet_format(X)
            prophet_forecast = self._prophet_model.predict(prophet_df)
            predictions["prophet"] = np.asarray(
                prophet_forecast["yhat"].values, dtype=np.float64
            )

        # TFT prediction (uses median quantile)
        if "tft" in self._active_models and self._tft_model is not None:
            pred_len = self._tft_model._tft_config.training.prediction_length
            enc_len = self._tft_model._tft_config.training.encoder_length

            if history is not None and len(X) > pred_len:
                # Rolling prediction: prepend encoder context from history
                context = history.iloc[-enc_len:]
                full_df = pd.concat([context, X])
                full_df = full_df[~full_df.index.duplicated(keep="last")].sort_index()
                tft_result = self._tft_model.predict_rolling(
                    full_df, target_col=self._target_col
                )
                # Reindex to X's index (drop encoder context timestamps)
                tft_aligned = tft_result.reindex(X.index)
                predictions["tft"] = np.asarray(
                    tft_aligned["yhat"].values, dtype=np.float64
                )
            else:
                # Standard single-window prediction (serving case, ~48 rows)
                tft_result = self._tft_model.predict(X, target_col=self._target_col)
                predictions["tft"] = np.asarray(
                    tft_result["yhat"].values, dtype=np.float64
                )

        if not predictions:
            msg = "No active models loaded. Call set_models() first."
            raise RuntimeError(msg)

        # Normalize weights to active models with predictions
        active_with_preds = list(predictions.keys())
        active_weights = {m: self._weights.get(m, 0.0) for m in active_with_preds}
        weight_sum = sum(active_weights.values())
        if weight_sum < 1e-6:
            # Equal weights if all are zero
            n = len(active_with_preds)
            normalized_weights = {m: 1.0 / n for m in active_with_preds}
        else:
            normalized_weights = {m: w / weight_sum for m, w in active_weights.items()}

        # Weighted average
        ensemble_pred = sum(
            normalized_weights[m] * predictions[m] for m in predictions
        )

        # Build output DataFrame
        result = pd.DataFrame(index=X.index)
        result["prediction"] = ensemble_pred
        for model_name, pred in predictions.items():
            result[f"{model_name}_prediction"] = pred

        return result

    def _to_prophet_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert feature DataFrame to Prophet ds+regressors format.

        Args:
            df: DataFrame with DatetimeIndex.

        Returns:
            Prophet-formatted DataFrame.
        """
        prophet_df = pd.DataFrame()
        prophet_df["ds"] = df.index

        # Add regressors
        for reg in self._prophet_regressors:
            if reg in df.columns:
                prophet_df[reg] = df[reg].values

        return prophet_df

    def save(self, path: Path) -> None:
        """Save ensemble configuration and model paths.

        Saves weights to JSON. Models are saved separately by their
        respective trainers.

        Args:
            path: Directory to save ensemble config.
        """
        path.mkdir(parents=True, exist_ok=True)

        # Save weights
        weights_path = path / "ensemble_weights.json"
        with open(weights_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "weights": self._weights,
                    "active_models": self._active_models,
                    "target_col": self._target_col,
                    "prophet_regressors": self._prophet_regressors,
                },
                f,
                indent=2,
            )

        logger.info("Saved ensemble config to {}", path)

    def load(self, path: Path) -> None:
        """Load ensemble configuration from disk.

        Loads weights and config. Models must be loaded separately
        using set_models() with CatBoost, Prophet, and TFT model files.

        Args:
            path: Directory containing ensemble config.
        """
        weights_path = path / "ensemble_weights.json"

        with open(weights_path, encoding="utf-8") as f:
            config = json.load(f)

        self._weights = config["weights"]
        self._active_models = config.get("active_models", ["catboost", "prophet", "tft"])
        self._target_col = config.get("target_col", "consumption")
        self._prophet_regressors = config.get("prophet_regressors", [])

        logger.info("Loaded ensemble config from {}", path)

    def load_models(
        self,
        catboost_path: Path | None = None,
        prophet_path: Path | None = None,
        tft_path: Path | None = None,
    ) -> None:
        """Load pre-trained models from disk.

        Args:
            catboost_path: Path to CatBoost .cbm file.
            prophet_path: Path to Prophet .pkl file.
            tft_path: Path to TFT model directory.

        Raises:
            RuntimeError: If model file is corrupted.
        """
        # Load CatBoost
        if catboost_path is not None and catboost_path.exists():
            self._catboost_model = CatBoostRegressor()
            self._catboost_model.load_model(str(catboost_path))
            logger.info("Loaded CatBoost model from {}", catboost_path)

        # Load Prophet
        if prophet_path is not None and prophet_path.exists():
            try:
                with open(prophet_path, "rb") as f:
                    self._prophet_model = pickle.load(f)
                logger.info("Loaded Prophet model from {}", prophet_path)
            except (pickle.UnpicklingError, EOFError, AttributeError) as e:
                msg = f"Failed to load Prophet model (corrupted file?): {e}"
                raise RuntimeError(msg) from e

        # Load TFT
        if tft_path is not None and tft_path.exists():
            try:
                self._tft_model = TFTForecaster.from_checkpoint(tft_path)
                logger.info("Loaded TFT model from {}", tft_path)
            except FileNotFoundError as e:
                logger.warning("TFT model incomplete, skipping: {}", e)
