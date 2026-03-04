"""Temporal Fusion Transformer forecaster with uncertainty quantification.

Wraps NeuralForecast's TFT model to conform to BaseForecaster interface.
Handles long-format conversion and quantile prediction output.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from numpy.typing import NDArray

from energy_forecast.config.settings import TFTConfig
from energy_forecast.models.base import PREDICTION_COL, BaseForecaster

# NeuralForecast long-format constants
NF_UNIQUE_ID = "uludag"


class TFTForecaster(BaseForecaster):
    """TFT-based hourly consumption forecaster with uncertainty quantification.

    Uses NeuralForecast's TFT implementation for efficient GPU-accelerated training
    with pre-computed tensor windowing.

    Args:
        config: TFT configuration from settings.
    """

    METADATA_FILENAME = "metadata.json"

    def __init__(self, config: TFTConfig) -> None:
        super().__init__(config.model_dump())
        self._tft_config = config
        self._nf: Any | None = None  # NeuralForecast instance
        self._quantiles: list[float] = list(config.quantiles)
        self._all_quantile_predictions: dict[float, NDArray[np.floating[Any]]] | None = None
        self._last_train_df: pd.DataFrame | None = None  # for predict() context

    @property
    def is_fitted(self) -> bool:
        """Check if the model has been trained."""
        return self._nf is not None

    def _to_nf_format(
        self,
        df: pd.DataFrame,
        target_col: str,
        *,
        drop_target_nan: bool = True,
    ) -> pd.DataFrame:
        """Convert DatetimeIndex DataFrame to NeuralForecast long format.

        Args:
            df: DataFrame with DatetimeIndex.
            target_col: Target column name.
            drop_target_nan: Drop rows where target is NaN.

        Returns:
            NF-formatted DataFrame with unique_id, ds, y columns.
        """
        nf_df = df.reset_index()

        # Detect datetime column name after reset_index
        dt_col = "date" if "date" in nf_df.columns else "index"
        if dt_col not in nf_df.columns:
            # Try the first column
            dt_col = nf_df.columns[0]
        nf_df = nf_df.rename(columns={dt_col: "ds"})

        nf_df["unique_id"] = NF_UNIQUE_ID
        nf_df = nf_df.rename(columns={target_col: "y"})

        # NeuralForecast converts ALL columns to float32, so we must filter
        # to only keep the required columns + specified covariates.
        cfg = self._tft_config.covariates
        covariate_cols = [
            c for c in list(cfg.time_varying_known) + list(cfg.time_varying_unknown)
            if c in nf_df.columns
        ]
        keep_cols = ["unique_id", "ds", "y"] + covariate_cols
        nf_df = nf_df[keep_cols]

        # Drop NaN covariates (lag features have NaN at start)
        n_before = len(nf_df)
        nf_df = nf_df.dropna(subset=covariate_cols)
        n_dropped = n_before - len(nf_df)
        if n_dropped > 0:
            logger.info(
                "Dropped {} rows with NaN in covariates ({:.1f}%)",
                n_dropped,
                100.0 * n_dropped / n_before,
            )

        if drop_target_nan:
            nf_df = nf_df.dropna(subset=["y"])

        return nf_df

    def _build_nf_model(
        self,
        callbacks: list[Any] | None = None,
        *,
        max_steps: int | None = None,
    ) -> Any:
        """Build NeuralForecast TFT model from config.

        Args:
            callbacks: Extra Lightning callbacks (e.g. Optuna pruning).
            max_steps: Override max_steps (for HPO).

        Returns:
            NeuralForecast instance wrapping a TFT model.
        """
        from neuralforecast import NeuralForecast
        from neuralforecast.losses.pytorch import MQLoss
        from neuralforecast.models import TFT

        cfg = self._tft_config
        arch = cfg.architecture
        train_cfg = cfg.training

        steps = max_steps if max_steps is not None else train_cfg.max_steps

        # NeuralForecast uses **kwargs to capture extra arguments as trainer_kwargs.
        # Pass precision/gradient_clip/callbacks as flat kwargs, NOT in a dict.
        extra_trainer_kwargs: dict[str, Any] = {
            "accelerator": train_cfg.accelerator,
            "precision": train_cfg.precision,
            "gradient_clip_val": train_cfg.gradient_clip_val,
            "enable_progress_bar": train_cfg.enable_progress_bar,
            "log_every_n_steps": 100,
        }
        # NF defaults devices=-1 (all GPUs); CPU requires devices=1
        if train_cfg.accelerator == "cpu":
            extra_trainer_kwargs["devices"] = 1

        # Progress bar: refresh every 100 steps instead of every step
        if train_cfg.enable_progress_bar:
            from pytorch_lightning.callbacks import TQDMProgressBar

            progress_callbacks = [TQDMProgressBar(refresh_rate=100)]
            if callbacks:
                callbacks = list(callbacks) + progress_callbacks
            else:
                callbacks = progress_callbacks
        if callbacks:
            extra_trainer_kwargs["callbacks"] = callbacks

        model = TFT(
            h=train_cfg.prediction_length,
            input_size=train_cfg.encoder_length,
            hidden_size=arch.hidden_size,
            n_head=arch.n_head,
            n_rnn_layers=arch.n_rnn_layers,
            dropout=arch.dropout,
            rnn_type=train_cfg.rnn_type,
            loss=MQLoss(quantiles=cfg.quantiles),
            learning_rate=train_cfg.learning_rate,
            max_steps=steps,
            val_check_steps=train_cfg.val_check_steps,
            early_stop_patience_steps=train_cfg.early_stop_patience_steps,
            scaler_type=train_cfg.scaler_type,
            batch_size=1,  # Single time series
            windows_batch_size=train_cfg.windows_batch_size,
            futr_exog_list=list(cfg.covariates.time_varying_known),
            hist_exog_list=list(cfg.covariates.time_varying_unknown),
            num_lr_decays=-1,
            random_seed=train_cfg.random_seed,
            **extra_trainer_kwargs,
        )

        nf = NeuralForecast(models=[model], freq="h")

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.debug("TFT model built with {} trainable parameters", n_params)

        return nf

    def train(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame | None = None,
        **kwargs: Any,
    ) -> dict[str, float]:
        """Train TFT model.

        Args:
            train_df: Training DataFrame with DatetimeIndex.
            val_df: Optional validation DataFrame.
            **kwargs: Additional arguments:
                - target_col: Target column name.
                - max_steps: Override max_steps.
                - callbacks: Extra Lightning callbacks.

        Returns:
            Training metrics dict.
        """
        target_col: str = kwargs.get("target_col", self._target_col)
        max_steps: int | None = kwargs.get("max_steps")
        extra_callbacks: list[Any] | None = kwargs.get("callbacks")

        logger.info(
            "Starting TFT training | samples={} | val={}",
            len(train_df),
            len(val_df) if val_df is not None else 0,
        )

        # Build NF model
        nf = self._build_nf_model(callbacks=extra_callbacks, max_steps=max_steps)

        # Convert to NF format
        if val_df is not None:
            full_df = pd.concat([train_df, val_df])
            nf_df = self._to_nf_format(full_df, target_col)
            val_size = len(val_df)
        else:
            nf_df = self._to_nf_format(train_df, target_col)
            val_size = 0

        logger.info(
            "NF training data: {} rows, val_size={}",
            len(nf_df),
            val_size,
        )

        # Fit
        nf.fit(df=nf_df, val_size=val_size)

        self._nf = nf
        self._last_train_df = nf_df

        # Collect metrics from the underlying Lightning trainer
        metrics: dict[str, float] = {}
        try:
            tft_model = nf.models[0]
            if hasattr(tft_model, "trainer") and tft_model.trainer is not None:
                import torch

                for key, value in tft_model.trainer.callback_metrics.items():
                    if isinstance(value, torch.Tensor):
                        metrics[key] = float(value.item())
                    else:
                        metrics[key] = float(value)
        except Exception:
            logger.debug("Could not extract trainer metrics")

        logger.info("TFT training complete | metrics={}", metrics)
        return metrics

    def predict(
        self,
        X: pd.DataFrame,
        target_col: str | None = None,
    ) -> pd.DataFrame:
        """Generate predictions using median quantile.

        NeuralForecast predicts the next h steps from the end of the provided
        context DataFrame. The last prediction_length timestamps in X are used
        as the prediction target period.

        Args:
            X: Feature DataFrame with DatetimeIndex.
            target_col: Target column name.

        Returns:
            DataFrame with PREDICTION_COL (median prediction) column.
        """
        if target_col is None:
            target_col = self._target_col
        if not self.is_fitted:
            msg = "Model must be trained before prediction"
            raise RuntimeError(msg)

        pred_len = self._tft_config.training.prediction_length

        logger.debug("Generating TFT predictions for {} samples", len(X))

        # Build context DataFrame (everything up to the forecast period)
        # and future exogenous DataFrame (known covariates for forecast period)
        enc_len = self._tft_config.training.encoder_length

        # Determine context and forecast boundaries
        if len(X) > pred_len:
            context_end = len(X) - pred_len
            context_df = X.iloc[max(0, context_end - enc_len):context_end]
            forecast_df = X.iloc[context_end:]
        else:
            # Short input — use last_train_df as context
            context_df = None
            forecast_df = X.iloc[-pred_len:]

        # Prepare future exogenous DataFrame
        futr_cols = list(self._tft_config.covariates.time_varying_known)
        futr_data: dict[str, Any] = {
            "unique_id": NF_UNIQUE_ID,
            "ds": forecast_df.index,
        }
        for col in futr_cols:
            if col in forecast_df.columns:
                futr_data[col] = forecast_df[col].values
        futr_df = pd.DataFrame(futr_data)

        # Prepare context (if different from training data)
        nf_context = None
        if context_df is not None:
            nf_context = self._to_nf_format(
                context_df, target_col, drop_target_nan=False,
            )
            # Fill NaN target in context (forecast rows have NaN consumption)
            if nf_context["y"].isna().any():
                nf_context = nf_context.copy()
                nf_context["y"] = nf_context["y"].ffill().fillna(0)

        # Generate predictions
        preds = self._nf.predict(df=nf_context, futr_df=futr_df)

        # Extract median prediction
        median_col = "TFT-median"
        if median_col not in preds.columns:
            # Fallback: try TFT column (point forecast without quantiles)
            median_col = "TFT"
            if median_col not in preds.columns:
                # Use first available prediction column
                pred_cols = [c for c in preds.columns if c.startswith("TFT")]
                median_col = pred_cols[0] if pred_cols else preds.columns[-1]

        # Store all quantile predictions
        self._store_quantile_predictions(preds)

        # Build result DataFrame
        pred_values = preds[median_col].values
        n_preds = len(pred_values)
        result_index = forecast_df.index[-n_preds:]

        result = pd.DataFrame(
            {PREDICTION_COL: pred_values[-len(result_index):]},
            index=result_index,
        )

        return result

    def _store_quantile_predictions(self, preds: pd.DataFrame) -> None:
        """Extract and store quantile predictions from NF output.

        NF MQLoss output columns: TFT-median, TFT-lo-96.0, TFT-hi-96.0, etc.
        Maps back to our quantile format: {0.02: array, 0.10: array, ...}
        """
        result: dict[float, NDArray[np.floating[Any]]] = {}

        for q in self._quantiles:
            # NF naming convention for quantiles
            level = abs(q - 0.5) * 200  # e.g., q=0.02 → level=96.0
            if q == 0.5:
                col = "TFT-median"
            elif q < 0.5:
                col = f"TFT-lo-{level:.1f}"
            else:
                col = f"TFT-hi-{level:.1f}"

            if col in preds.columns:
                result[q] = preds[col].values.astype(np.float64)

        self._all_quantile_predictions = result if result else None

    def get_quantile_predictions(self) -> dict[float, NDArray[np.floating[Any]]]:
        """Get all quantile predictions from last predict() call.

        Returns:
            Dict mapping quantile value to predictions array.
        """
        if self._all_quantile_predictions is None:
            msg = "No predictions available. Call predict() first."
            raise RuntimeError(msg)

        return self._all_quantile_predictions

    def save(self, path: Path) -> None:
        """Save TFT model using NeuralForecast's built-in save.

        Args:
            path: Directory to save model files.
        """
        if not self.is_fitted:
            msg = "Cannot save unfitted model"
            raise ValueError(msg)

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        logger.info("Saving TFT model to {}", path)

        # NeuralForecast save (handles ckpt + config internally)
        self._nf.save(path=str(path), overwrite=True)

        # Save our metadata (quantiles, architecture, covariates)
        metadata = {
            "quantiles": self._quantiles,
            "architecture": self._tft_config.architecture.model_dump(),
            "training": {
                "encoder_length": self._tft_config.training.encoder_length,
                "prediction_length": self._tft_config.training.prediction_length,
                "num_workers": self._tft_config.training.num_workers,
                "scaler_type": self._tft_config.training.scaler_type,
                "rnn_type": self._tft_config.training.rnn_type,
            },
            "covariates": self._tft_config.covariates.model_dump(),
        }
        metadata_path = path / self.METADATA_FILENAME
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info("TFT model saved successfully")

    @classmethod
    def from_checkpoint(cls, path: Path | str) -> TFTForecaster:
        """Load a fully functional TFTForecaster from a saved checkpoint.

        Args:
            path: Directory containing saved model files.

        Returns:
            TFTForecaster ready for prediction.
        """
        from neuralforecast import NeuralForecast

        from energy_forecast.config.settings import (
            TFTArchitectureConfig,
            TFTConfig,
            TFTCovariatesConfig,
            TFTTrainingConfig,
        )

        path = Path(path)
        logger.info("Loading TFT model from checkpoint: {}", path)

        # Load our metadata
        metadata_path = path / cls.METADATA_FILENAME
        if not metadata_path.exists():
            msg = f"Metadata not found: {metadata_path}"
            raise FileNotFoundError(msg)
        with open(metadata_path) as f:
            metadata = json.load(f)

        # Load NeuralForecast model
        nf = NeuralForecast.load(path=str(path))

        # Reconstruct TFTConfig
        arch_data = metadata.get("architecture", {})
        train_data = metadata.get("training", {})
        cov_data = metadata.get("covariates", {})
        quantiles = metadata.get("quantiles", [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98])

        config = TFTConfig(
            architecture=TFTArchitectureConfig(**arch_data),
            training=TFTTrainingConfig(
                encoder_length=train_data.get("encoder_length", 168),
                prediction_length=train_data.get("prediction_length", 48),
                max_steps=1,  # Not used for inference
                num_workers=train_data.get("num_workers", 4),
                scaler_type=train_data.get("scaler_type", "robust"),
                rnn_type=train_data.get("rnn_type", "lstm"),
            ),
            covariates=TFTCovariatesConfig(**cov_data),
            quantiles=quantiles,
        )

        instance = cls(config)
        instance._nf = nf
        instance._quantiles = quantiles

        logger.info("TFT model loaded successfully — ready for prediction")
        return instance

    def load(self, path: Path) -> None:
        """Load TFT model from checkpoint directory.

        Args:
            path: Directory containing saved model files.
        """
        from neuralforecast import NeuralForecast

        path = Path(path)
        logger.info("Loading TFT model from {}", path)

        # Load NeuralForecast model
        self._nf = NeuralForecast.load(path=str(path))

        # Load metadata
        metadata_path = path / self.METADATA_FILENAME
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
                self._quantiles = metadata.get("quantiles", self._quantiles)

        logger.info("TFT model loaded successfully")
