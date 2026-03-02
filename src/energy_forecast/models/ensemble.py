"""Ensemble of CatBoost, Prophet, and TFT forecasters.

Supports two modes:
- stacking: CatBoost meta-learner combines base predictions with context features
- weighted_average: Static weight blending (fallback)
"""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from loguru import logger

from energy_forecast.config.settings import EnsembleConfig
from energy_forecast.models.base import PREDICTION_COL, BaseForecaster
from energy_forecast.models.tft import TFTForecaster
from energy_forecast.utils.prophet_utils import to_prophet_format

if TYPE_CHECKING:
    from prophet import Prophet


class EnsembleForecaster(BaseForecaster):
    """Ensemble of CatBoost, Prophet, and TFT.

    Loads pre-trained models and uses either a stacking meta-learner or
    optimized weights to produce ensemble predictions.

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
            "mode": config.get("mode", default_cfg.mode),
            "context_features": config.get(
                "context_features",
                list(default_cfg.stacking.context_features),
            ),
        }

        super().__init__(merged_config)
        self._active_models: list[str] = merged_config["active_models"]
        self._weights: dict[str, float] = merged_config["weights"]
        self._mode: str = merged_config["mode"]
        self._context_features: list[str] = merged_config["context_features"]
        self._catboost_model: CatBoostRegressor | None = None
        self._prophet_model: Prophet | None = None
        self._tft_model: TFTForecaster | None = None
        self._meta_model: CatBoostRegressor | None = None
        self._prophet_regressors: list[str] = merged_config["prophet_regressors"]

    @property
    def weights(self) -> dict[str, float]:
        """Get current ensemble weights."""
        return self._weights.copy()

    @property
    def active_models(self) -> list[str]:
        """Get current active models."""
        return self._active_models.copy()

    @property
    def mode(self) -> str:
        """Get ensemble mode."""
        return self._mode

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

    def set_meta_model(self, meta_model: CatBoostRegressor) -> None:
        """Set stacking meta-learner model.

        Args:
            meta_model: Trained CatBoost meta-learner.
        """
        self._meta_model = meta_model
        logger.info("Meta-learner set for stacking ensemble")

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
        """Generate ensemble prediction.

        Routes to stacking (meta-learner) or weighted_average based on mode.

        Args:
            X: Feature DataFrame with DatetimeIndex.
            history: Optional historical data for TFT rolling prediction.

        Returns:
            DataFrame with PREDICTION_COL and individual model predictions.

        Raises:
            RuntimeError: If no active models are loaded.
        """
        predictions = self._get_base_predictions(X, history=history)

        if not predictions:
            msg = "No active models loaded. Call set_models() first."
            raise RuntimeError(msg)

        if self._mode == "stacking" and self._meta_model is not None:
            ensemble_pred = self._predict_stacking(X, predictions)
        else:
            ensemble_pred = self._predict_weighted_average(predictions)

        # Build output DataFrame
        result = pd.DataFrame(index=X.index)
        result[PREDICTION_COL] = ensemble_pred
        for model_name, pred in predictions.items():
            result[f"{model_name}_prediction"] = pred

        return result

    def _get_base_predictions(
        self,
        X: pd.DataFrame,
        *,
        history: pd.DataFrame | None = None,
    ) -> dict[str, np.ndarray[Any, np.dtype[np.floating[Any]]]]:
        """Get predictions from all active base models.

        Args:
            X: Feature DataFrame.
            history: Historical data for TFT encoder context.

        Returns:
            Dict of model_name -> prediction array.
        """
        predictions: dict[str, np.ndarray[Any, np.dtype[np.floating[Any]]]] = {}

        # CatBoost prediction
        if "catboost" in self._active_models and self._catboost_model is not None:
            features = X.drop(columns=[self._target_col], errors="ignore").copy()
            cat_indices = self._catboost_model.get_cat_feature_indices()
            if cat_indices:
                cat_cols = [features.columns[i] for i in cat_indices if i < len(features.columns)]
                for col in cat_cols:
                    features[col] = features[col].fillna("missing").astype(str)
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
            enc_len = self._tft_model._tft_config.training.encoder_length

            if history is not None:
                # TFT always needs encoder context — prepend from history
                context = history.iloc[-enc_len:]
                full_df = pd.concat([context, X])
                full_df = full_df[~full_df.index.duplicated(keep="last")].sort_index()
                # predict_rolling delegates to predict() for short inputs
                tft_result = self._tft_model.predict_rolling(
                    full_df, target_col=self._target_col
                )
                # Reindex to X's index (drop encoder context timestamps)
                tft_aligned = tft_result.reindex(X.index)
                predictions["tft"] = np.asarray(
                    tft_aligned[PREDICTION_COL].values, dtype=np.float64
                )
            else:
                # No history — direct prediction (needs sufficient rows)
                tft_result = self._tft_model.predict(X, target_col=self._target_col)
                predictions["tft"] = np.asarray(
                    tft_result[PREDICTION_COL].values, dtype=np.float64
                )

        return predictions

    def _predict_stacking(
        self,
        X: pd.DataFrame,
        predictions: dict[str, np.ndarray[Any, np.dtype[np.floating[Any]]]],
    ) -> np.ndarray[Any, np.dtype[np.floating[Any]]]:
        """Apply meta-learner to base predictions + context features.

        Args:
            X: Feature DataFrame with DatetimeIndex (for context extraction).
            predictions: Base model predictions.

        Returns:
            Meta-learner ensemble prediction array.
        """
        meta_features = pd.DataFrame(index=X.index)
        for model_name in self._active_models:
            if model_name in predictions:
                meta_features[f"pred_{model_name}"] = predictions[model_name]

        # Context features from DatetimeIndex
        dt_idx = pd.DatetimeIndex(X.index)
        if "hour" in self._context_features:
            meta_features["hour"] = dt_idx.hour
        if "day_of_week" in self._context_features:
            meta_features["day_of_week"] = dt_idx.dayofweek
        if "is_weekend" in self._context_features:
            meta_features["is_weekend"] = (dt_idx.dayofweek >= 5).astype(int)
        if "month" in self._context_features:
            meta_features["month"] = dt_idx.month
        if "is_holiday" in self._context_features:
            if "is_holiday" in X.columns:
                meta_features["is_holiday"] = X["is_holiday"].values
            else:
                meta_features["is_holiday"] = 0

        # Categorical columns to string (matching training format)
        for col in ["hour", "day_of_week", "month"]:
            if col in meta_features.columns:
                meta_features[col] = meta_features[col].astype(str)

        assert self._meta_model is not None
        return np.asarray(self._meta_model.predict(meta_features), dtype=np.float64)

    def _predict_weighted_average(
        self,
        predictions: dict[str, np.ndarray[Any, np.dtype[np.floating[Any]]]],
    ) -> np.ndarray[Any, np.dtype[np.floating[Any]]]:
        """Apply static weight blending to base predictions.

        Args:
            predictions: Base model predictions.

        Returns:
            Weighted average ensemble prediction array.
        """
        active_with_preds = list(predictions.keys())
        active_weights = {m: self._weights.get(m, 0.0) for m in active_with_preds}
        weight_sum = sum(active_weights.values())
        if weight_sum < 1e-6:
            n = len(active_with_preds)
            normalized_weights = {m: 1.0 / n for m in active_with_preds}
        else:
            normalized_weights = {m: w / weight_sum for m, w in active_weights.items()}

        ensemble_pred = sum(
            normalized_weights[m] * predictions[m] for m in predictions
        )
        return np.asarray(ensemble_pred, dtype=np.float64)

    def _to_prophet_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert feature DataFrame to Prophet ds+regressors format.

        Args:
            df: DataFrame with DatetimeIndex.

        Returns:
            Prophet-formatted DataFrame.
        """
        return to_prophet_format(df, self._prophet_regressors)

    def save(self, path: Path) -> None:
        """Save ensemble configuration, weights, and meta-learner.

        Args:
            path: Directory to save ensemble artifacts.
        """
        path.mkdir(parents=True, exist_ok=True)

        config: dict[str, Any] = {
            "mode": self._mode,
            "weights": self._weights,
            "active_models": self._active_models,
            "target_col": self._target_col,
            "prophet_regressors": self._prophet_regressors,
            "context_features": self._context_features,
        }

        # Save config
        weights_path = path / "ensemble_weights.json"
        with open(weights_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

        # Save meta-learner if stacking
        if self._meta_model is not None:
            meta_path = path / "meta_model.cbm"
            self._meta_model.save_model(str(meta_path))
            logger.info("Saved meta-learner to {}", meta_path)

        logger.info("Saved ensemble config to {}", path)

    def load(self, path: Path) -> None:
        """Load ensemble configuration from disk.

        Loads weights, config, and meta-learner if stacking mode.
        Models must be loaded separately using set_models().

        Args:
            path: Directory containing ensemble config.
        """
        weights_path = path / "ensemble_weights.json"

        with open(weights_path, encoding="utf-8") as f:
            config = json.load(f)

        # Support both nested {"weights": {...}} and flat {"catboost": 0.3, ...} formats
        if "weights" in config and isinstance(config["weights"], dict):
            self._weights = config["weights"]
        else:
            # Flat format from ensemble_trainer — all keys are model weights
            self._weights = {
                k: v for k, v in config.items()
                if k not in (
                    "active_models", "target_col", "prophet_regressors",
                    "mode", "context_features",
                )
            }
        self._active_models = config.get("active_models", self._active_models)
        self._target_col = config.get("target_col", self._target_col)
        self._mode = config.get("mode", "weighted_average")
        self._context_features = config.get(
            "context_features",
            ["hour", "day_of_week", "is_weekend", "is_holiday", "month"],
        )
        if "prophet_regressors" in config:
            self._prophet_regressors = config["prophet_regressors"]

        # Load meta-learner if stacking
        meta_path = path / "meta_model.cbm"
        if meta_path.exists():
            self._meta_model = CatBoostRegressor()
            self._meta_model.load_model(str(meta_path))
            logger.info("Loaded meta-learner from {}", meta_path)
            if self._mode != "stacking":
                self._mode = "stacking"
                logger.info("Auto-switched to stacking mode (meta_model.cbm found)")

        logger.info("Loaded ensemble config from {} (mode={})", path, self._mode)

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
            try:
                self._catboost_model = CatBoostRegressor()
                self._catboost_model.load_model(str(catboost_path))
                logger.info("Loaded CatBoost model from {}", catboost_path)
            except Exception as e:
                logger.warning("Failed to load CatBoost model, skipping: {}", e)

        # Load Prophet (with hash integrity check)
        if prophet_path is not None and prophet_path.exists():
            # Verify hash if metadata exists
            metadata_path = prophet_path.parent / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, encoding="utf-8") as f:
                    metadata = json.load(f)
                expected_hash = metadata.get("model_hash")
                if expected_hash:
                    actual = "sha256:" + hashlib.sha256(
                        prophet_path.read_bytes()
                    ).hexdigest()
                    if actual != expected_hash:
                        msg = f"Prophet model integrity check failed: {prophet_path}"
                        raise RuntimeError(msg)
                else:
                    logger.warning("No model_hash in metadata — skipping integrity check")
            else:
                logger.warning("No metadata.json — skipping Prophet integrity check")

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
