"""Prophet trend and seasonality forecaster."""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
from loguru import logger

from energy_forecast.models.base import BaseForecaster
from energy_forecast.utils.prophet_utils import to_prophet_format

if TYPE_CHECKING:
    from prophet import Prophet


class ModelIntegrityError(RuntimeError):
    """Raised when model file hash verification fails (CWE-502 mitigation)."""


class ProphetForecaster(BaseForecaster):
    """Prophet-based hourly consumption forecaster.

    Training is done by ProphetTrainer.
    This class is used for loading trained models and prediction.

    Args:
        config: Prophet configuration dict.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self._model: Prophet | None = None
        self._regressor_names: list[str] = []

    def train(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame | None = None,
        **kwargs: Any,
    ) -> None:
        """Simple training (without Trainer). For test convenience.

        Args:
            train_df: Training DataFrame with DatetimeIndex.
            val_df: Validation DataFrame (unused for Prophet).
            **kwargs: Additional arguments (unused, for base class compatibility).

        Returns:
            None (metrics not tracked in simple training).
        """
        from prophet import Prophet as ProphetModel

        prophet_df = self._to_prophet_format(train_df, include_target=True)
        self._model = ProphetModel()
        self._model.fit(prophet_df)
        return None

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
        target = self._target_col if include_target else None
        return to_prophet_format(df, self._regressor_names, target_col=target)

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
        pkl_path = path / "prophet_model.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(self._model, f)

        # Compute SHA256 hash for integrity verification (CWE-502 mitigation)
        model_hash = "sha256:" + hashlib.sha256(pkl_path.read_bytes()).hexdigest()

        # Metadata (includes model hash)
        with open(path / "metadata.json", "w") as f:
            json.dump(
                {
                    "regressor_names": self._regressor_names,
                    "config": self.config,
                    "model_hash": model_hash,
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

        # Verify integrity via SHA256 hash if metadata exists
        metadata_path = path / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, encoding="utf-8") as f:
                metadata = json.load(f)
            expected_hash = metadata.get("model_hash")
            if expected_hash:
                actual_hash = "sha256:" + hashlib.sha256(model_path.read_bytes()).hexdigest()
                if actual_hash != expected_hash:
                    msg = f"Model file integrity check failed: {model_path}"
                    raise ModelIntegrityError(msg)
            else:
                logger.warning("No model_hash in metadata — skipping integrity check")
        else:
            logger.warning("No metadata.json found — skipping integrity check for {}", path)

        try:
            with open(model_path, "rb") as f:
                self._model = pickle.load(f)
        except (pickle.UnpicklingError, EOFError, AttributeError) as e:
            msg = f"Failed to load Prophet model (corrupted file?): {e}"
            raise RuntimeError(msg) from e

        # Load metadata
        metadata_path = path / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                meta = json.load(f)
                self._regressor_names = meta.get("regressor_names", [])
                self.config = {**self.config, **meta.get("config", {})}

        logger.info("Prophet model loaded from {}", path)

    def set_model(self, model: Prophet, regressor_names: list[str] | None = None) -> None:
        """Set pre-trained Prophet model (from ProphetTrainer).

        Args:
            model: Trained Prophet model.
            regressor_names: List of regressor column names.
        """
        self._model = model
        if regressor_names:
            self._regressor_names = regressor_names
