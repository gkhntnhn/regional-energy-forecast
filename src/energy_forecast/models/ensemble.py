"""Weighted ensemble of CatBoost and Prophet forecasters."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from loguru import logger
from prophet import Prophet

from energy_forecast.models.base import BaseForecaster


class EnsembleForecaster(BaseForecaster):
    """Weighted-average ensemble of CatBoost and Prophet.

    Loads pre-trained models and uses optimized weights from training
    to produce ensemble predictions.

    Args:
        config: Ensemble configuration dictionary containing weights.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self._weights: dict[str, float] = config.get(
            "weights", {"catboost": 0.6, "prophet": 0.4}
        )
        self._catboost_model: CatBoostRegressor | None = None
        self._prophet_model: Prophet | None = None
        self._target_col: str = config.get("target_col", "consumption")
        self._prophet_regressors: list[str] = config.get("prophet_regressors", [])

    @property
    def weights(self) -> dict[str, float]:
        """Get current ensemble weights."""
        return self._weights.copy()

    def set_models(
        self,
        catboost_model: CatBoostRegressor,
        prophet_model: Prophet,
    ) -> None:
        """Set pre-trained models for prediction.

        Args:
            catboost_model: Trained CatBoost model.
            prophet_model: Trained Prophet model.
        """
        self._catboost_model = catboost_model
        self._prophet_model = prophet_model
        logger.info("Ensemble models set")

    def set_weights(self, weights: dict[str, float]) -> None:
        """Set ensemble weights.

        Args:
            weights: Dictionary with 'catboost' and 'prophet' weights.

        Raises:
            ValueError: If weights don't sum to 1.0.
        """
        total = weights.get("catboost", 0) + weights.get("prophet", 0)
        if abs(total - 1.0) > 1e-6:
            msg = f"Weights must sum to 1.0, got {total}"
            raise ValueError(msg)
        self._weights = weights.copy()
        logger.info("Ensemble weights updated: {}", self._weights)

    def train(self, train_df: pd.DataFrame, val_df: pd.DataFrame | None = None) -> None:
        """Train is not supported — use EnsembleTrainer instead.

        The ensemble training is orchestrated by EnsembleTrainer which
        trains CatBoost and Prophet separately, then optimizes weights.

        Raises:
            NotImplementedError: Always, use EnsembleTrainer.run() instead.
        """
        msg = (
            "EnsembleForecaster.train() is not supported. "
            "Use EnsembleTrainer.run() to train the ensemble, "
            "then load the trained models with set_models()."
        )
        raise NotImplementedError(msg)

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Generate weighted-average ensemble prediction.

        Args:
            X: Feature DataFrame with DatetimeIndex.

        Returns:
            DataFrame with 'prediction' column and individual model predictions.

        Raises:
            RuntimeError: If models are not loaded.
        """
        if self._catboost_model is None or self._prophet_model is None:
            msg = "Models not loaded. Call set_models() or load() first."
            raise RuntimeError(msg)

        # CatBoost prediction
        features = X.drop(columns=[self._target_col], errors="ignore")
        catboost_pred: np.ndarray[Any, np.dtype[np.floating[Any]]] = (
            self._catboost_model.predict(features)
        )

        # Prophet prediction
        prophet_df = self._to_prophet_format(X)
        prophet_forecast = self._prophet_model.predict(prophet_df)
        prophet_pred: np.ndarray[Any, np.dtype[np.floating[Any]]] = np.asarray(
            prophet_forecast["yhat"].values, dtype=np.float64
        )

        # Weighted average
        cb_weight = self._weights["catboost"]
        pr_weight = self._weights["prophet"]
        ensemble_pred = cb_weight * catboost_pred + pr_weight * prophet_pred

        # Build output DataFrame
        result = pd.DataFrame(index=X.index)
        result["prediction"] = ensemble_pred
        result["catboost_prediction"] = catboost_pred
        result["prophet_prediction"] = prophet_pred

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
        using set_models() with CatBoost and Prophet model files.

        Args:
            path: Directory containing ensemble config.
        """
        weights_path = path / "ensemble_weights.json"

        with open(weights_path, encoding="utf-8") as f:
            config = json.load(f)

        self._weights = config["weights"]
        self._target_col = config.get("target_col", "consumption")
        self._prophet_regressors = config.get("prophet_regressors", [])

        logger.info("Loaded ensemble config from {}", path)

    def load_models(
        self,
        catboost_path: Path,
        prophet_path: Path,
    ) -> None:
        """Load pre-trained models from disk.

        Args:
            catboost_path: Path to CatBoost .cbm file.
            prophet_path: Path to Prophet .pkl file.
        """
        import pickle

        # Load CatBoost
        self._catboost_model = CatBoostRegressor()
        self._catboost_model.load_model(str(catboost_path))

        # Load Prophet
        with open(prophet_path, "rb") as f:
            self._prophet_model = pickle.load(f)

        logger.info(
            "Loaded models: CatBoost={}, Prophet={}",
            catboost_path,
            prophet_path,
        )
