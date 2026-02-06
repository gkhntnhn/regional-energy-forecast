"""CatBoost gradient boosting forecaster."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from catboost import CatBoostRegressor
from loguru import logger

from energy_forecast.models.base import BaseForecaster


class CatBoostForecaster(BaseForecaster):
    """CatBoost-based hourly consumption forecaster.

    Training is done via ``CatBoostTrainer``.
    This class wraps a trained model for save/load/predict.

    Args:
        config: CatBoost configuration from catboost.yaml.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self._model: CatBoostRegressor | None = None

    @property
    def model(self) -> CatBoostRegressor:
        """Return the underlying CatBoost model."""
        if self._model is None:
            msg = "Model not loaded. Call load() or set_model() first."
            raise RuntimeError(msg)
        return self._model

    def set_model(self, model: CatBoostRegressor) -> None:
        """Attach a trained model from CatBoostTrainer.

        Args:
            model: Trained CatBoostRegressor instance.
        """
        self._model = model
        logger.info("CatBoost model attached to forecaster")

    def train(self, train_df: pd.DataFrame, val_df: pd.DataFrame | None = None) -> None:
        """Train CatBoost model (simple, without Optuna).

        For full pipeline with Optuna + TSCV, use ``CatBoostTrainer.run()``.

        Args:
            train_df: Training data with features and target.
            val_df: Optional validation data for early stopping.
        """
        target_col = self.config.get("target_col", "consumption")
        y_train = train_df[target_col]
        x_train = train_df.drop(columns=[target_col])

        self._model = CatBoostRegressor(**self.config.get("params", {}))

        if val_df is not None:
            y_val = val_df[target_col]
            x_val = val_df.drop(columns=[target_col])
            self._model.fit(x_train, y_train, eval_set=(x_val, y_val), verbose=0)
        else:
            self._model.fit(x_train, y_train, verbose=0)

        logger.info("CatBoost simple training complete")

    def predict(self, x: pd.DataFrame) -> pd.DataFrame:
        """Generate predictions using trained CatBoost model.

        Args:
            x: Feature DataFrame for prediction period.

        Returns:
            DataFrame with ``consumption_mwh`` column and same index as input.
        """
        predictions: Any = self.model.predict(x)
        return pd.DataFrame({"consumption_mwh": predictions}, index=x.index)

    def save(self, path: Path) -> None:
        """Save CatBoost model (.cbm format).

        Args:
            path: Directory to save model files.
        """
        path.mkdir(parents=True, exist_ok=True)
        model_path = path / "model.cbm"
        self.model.save_model(str(model_path))
        logger.info("CatBoost model saved to {}", model_path)

    def load(self, path: Path) -> None:
        """Load CatBoost model from .cbm file.

        Args:
            path: Directory containing model files.
        """
        model_path = path / "model.cbm"
        self._model = CatBoostRegressor()
        self._model.load_model(str(model_path))
        logger.info("CatBoost model loaded from {}", model_path)
