"""Prophet trend and seasonality forecaster."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

from energy_forecast.models.base import BaseForecaster


class ProphetForecaster(BaseForecaster):
    """Prophet-based hourly consumption forecaster.

    Training is done by ProphetTrainer.
    This class is used for loading trained models and prediction.

    Args:
        config: Prophet configuration dict.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self._model: Any = None  # Prophet instance
        self._regressor_names: list[str] = []

    def train(self, train_df: pd.DataFrame, val_df: pd.DataFrame | None = None) -> None:
        """Simple training (without Trainer). For test convenience.

        Args:
            train_df: Training DataFrame with DatetimeIndex.
            val_df: Validation DataFrame (unused for Prophet).
        """
        from prophet import Prophet

        prophet_df = self._to_prophet_format(train_df, include_target=True)
        self._model = Prophet()
        self._model.fit(prophet_df)

    def _to_prophet_format(
        self,
        df: pd.DataFrame,
        include_target: bool = True,
    ) -> pd.DataFrame:
        """Convert DataFrame to Prophet format.

        Args:
            df: DataFrame with DatetimeIndex.
            include_target: If True, include y column.

        Returns:
            Prophet-formatted DataFrame with ds, y, and regressors.
        """
        prophet_df = pd.DataFrame()
        prophet_df["ds"] = df.index

        target_col = self.config.get("target_col", "consumption")
        if include_target and target_col in df.columns:
            prophet_df["y"] = df[target_col].values

        for col in self._regressor_names:
            if col in df.columns:
                prophet_df[col] = df[col].values

        return prophet_df

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Generate predictions using trained Prophet model.

        Args:
            X: Feature DataFrame with DatetimeIndex.

        Returns:
            DataFrame with consumption_mwh predictions.

        Raises:
            RuntimeError: If model not loaded.
        """
        if self._model is None:
            msg = "Model not loaded. Call load() first."
            raise RuntimeError(msg)

        future_df = self._to_prophet_format(X, include_target=False)
        forecast = self._model.predict(future_df)

        return pd.DataFrame(
            {"consumption_mwh": forecast["yhat"].values},
            index=X.index,
        )

    def save(self, path: Path) -> None:
        """Save Prophet model (.pkl format).

        Args:
            path: Directory path to save model files.

        Raises:
            RuntimeError: If no model to save.
        """
        if self._model is None:
            msg = "No model to save."
            raise RuntimeError(msg)

        path.mkdir(parents=True, exist_ok=True)

        # Model pickle
        with open(path / "prophet_model.pkl", "wb") as f:
            pickle.dump(self._model, f)

        # Metadata
        with open(path / "metadata.json", "w") as f:
            json.dump(
                {
                    "regressor_names": self._regressor_names,
                    "config": self.config,
                },
                f,
                indent=2,
            )

        logger.info("Prophet model saved to {}", path)

    def load(self, path: Path) -> None:
        """Load Prophet model from .pkl file.

        Args:
            path: Directory path containing model files.

        Raises:
            FileNotFoundError: If model file not found.
        """
        model_path = path / "prophet_model.pkl"
        if not model_path.exists():
            msg = f"Model not found: {model_path}"
            raise FileNotFoundError(msg)

        with open(model_path, "rb") as f:
            self._model = pickle.load(f)

        # Load metadata
        metadata_path = path / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                meta = json.load(f)
                self._regressor_names = meta.get("regressor_names", [])
                self.config.update(meta.get("config", {}))

        logger.info("Prophet model loaded from {}", path)

    def set_model(self, model: Any, regressor_names: list[str] | None = None) -> None:
        """Set pre-trained Prophet model (from ProphetTrainer).

        Args:
            model: Trained Prophet model.
            regressor_names: List of regressor column names.
        """
        self._model = model
        if regressor_names:
            self._regressor_names = regressor_names
