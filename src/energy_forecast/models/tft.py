"""Temporal Fusion Transformer forecaster with uncertainty quantification.

Wraps pytorch-forecasting's TFT model to conform to BaseForecaster interface.
Handles TimeSeriesDataSet conversion and quantile prediction output.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import torch
from loguru import logger
from numpy.typing import NDArray
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss

from energy_forecast.config.settings import TFTConfig
from energy_forecast.models.base import BaseForecaster

if TYPE_CHECKING:
    import lightning.pytorch as pl

# Constants for TimeSeriesDataSet
GROUP_ID = "series_0"
GROUP_COL = "_group_id"
TIME_IDX_COL = "_time_idx"


class TFTForecaster(BaseForecaster):
    """TFT-based hourly consumption forecaster with uncertainty quantification.

    Converts feature-engineered DataFrame to pytorch-forecasting's TimeSeriesDataSet
    format and provides quantile predictions for uncertainty estimation.

    Args:
        config: TFT configuration from settings.
    """

    MODEL_FILENAME = "tft_model.ckpt"
    METADATA_FILENAME = "metadata.json"
    DATASET_PARAMS_FILENAME = "dataset_params.json"
    TRAINING_DATASET_FILENAME = "training_dataset.pkl"

    def __init__(self, config: TFTConfig) -> None:
        super().__init__(config.model_dump())  # Convert to dict for base class
        self._tft_config = config
        self._model: TemporalFusionTransformer | None = None
        self._training_dataset: TimeSeriesDataSet | None = None
        self._dataset_params: dict[str, Any] = {}
        self._quantiles: list[float] = list(config.quantiles)
        self._all_quantile_predictions: NDArray[np.floating[Any]] | None = None
        self._loaded_state_dict: dict[str, Any] | None = None

    @property
    def is_fitted(self) -> bool:
        """Check if the model has been trained."""
        return self._model is not None

    def _prepare_dataframe(
        self,
        df: pd.DataFrame,
        target_col: str,
        include_target: bool = True,
    ) -> pd.DataFrame:
        """Prepare DataFrame for TimeSeriesDataSet.

        Adds required _time_idx and _group_id columns.

        Args:
            df: DataFrame with DatetimeIndex.
            target_col: Target column name.
            include_target: Include target in output.

        Returns:
            DataFrame ready for TimeSeriesDataSet.
        """
        result = df.copy()

        # Ensure DatetimeIndex
        if not isinstance(result.index, pd.DatetimeIndex):
            msg = "DataFrame must have DatetimeIndex"
            raise ValueError(msg)

        result = result.sort_index()

        # Add time index (integer sequence)
        result[TIME_IDX_COL] = range(len(result))

        # Add group identifier (single series)
        result[GROUP_COL] = GROUP_ID

        # Reset index to make timestamp a column
        result = result.reset_index()
        if "index" in result.columns:
            result = result.rename(columns={"index": "timestamp"})

        if not include_target and target_col in result.columns:
            result = result.drop(columns=[target_col])

        return result

    def _create_dataset(
        self,
        data: pd.DataFrame,
        target_col: str,
        is_training: bool = True,
    ) -> TimeSeriesDataSet:
        """Create TimeSeriesDataSet from prepared DataFrame.

        Args:
            data: Prepared DataFrame with _time_idx and _group_id.
            target_col: Target column name.
            is_training: Whether this is for training.

        Returns:
            Configured TimeSeriesDataSet.
        """
        cfg = self._tft_config
        train_cfg = cfg.training
        cov_cfg = cfg.covariates

        # Filter known reals to only include available columns
        time_varying_known = [
            col for col in cov_cfg.time_varying_known if col in data.columns
        ]

        # Unknown reals: target + any lagged/rolling features
        time_varying_unknown = [target_col]

        # Store params for serialization
        self._dataset_params = {
            "time_idx": TIME_IDX_COL,
            "target": target_col,
            "group_ids": [GROUP_COL],
            "max_encoder_length": train_cfg.encoder_length,
            "max_prediction_length": train_cfg.prediction_length,
            "time_varying_known_reals": time_varying_known,
            "time_varying_unknown_reals": time_varying_unknown,
        }

        dataset = TimeSeriesDataSet(
            data,
            time_idx=TIME_IDX_COL,
            target=target_col,
            group_ids=[GROUP_COL],
            max_encoder_length=train_cfg.encoder_length,
            max_prediction_length=train_cfg.prediction_length,
            time_varying_known_reals=time_varying_known,
            time_varying_unknown_reals=time_varying_unknown,
            static_categoricals=[],
            static_reals=[],
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
            target_normalizer=GroupNormalizer(
                groups=[GROUP_COL],
                transformation="softplus",
            ),
            allow_missing_timesteps=True,
        )

        return dataset

    def _build_model(self, dataset: TimeSeriesDataSet) -> TemporalFusionTransformer:
        """Build TFT model from dataset.

        Args:
            dataset: TimeSeriesDataSet for model configuration.

        Returns:
            Configured TemporalFusionTransformer.
        """
        cfg = self._tft_config
        arch = cfg.architecture
        train_cfg = cfg.training

        model = TemporalFusionTransformer.from_dataset(
            dataset,
            hidden_size=arch.hidden_size,
            attention_head_size=arch.attention_head_size,
            dropout=arch.dropout,
            hidden_continuous_size=arch.hidden_continuous_size,
            lstm_layers=arch.lstm_layers,
            output_size=len(cfg.quantiles),
            loss=QuantileLoss(quantiles=cfg.quantiles),
            learning_rate=train_cfg.learning_rate,
            reduce_on_plateau_patience=train_cfg.early_stop_patience,
        )

        n_params = sum(p.numel() for p in model.parameters())
        logger.debug("TFT model built with {} parameters", n_params)

        return model

    def _create_trainer(self, max_epochs: int | None = None) -> pl.Trainer:
        """Create PyTorch Lightning trainer.

        Args:
            max_epochs: Override max epochs (for testing).

        Returns:
            Configured lightning Trainer.
        """
        import lightning.pytorch as pl_lib
        from lightning.pytorch.callbacks import Callback, EarlyStopping

        cfg = self._tft_config.training

        callbacks: list[Callback] = []

        if cfg.early_stop_patience > 0:
            callbacks.append(
                EarlyStopping(
                    monitor="val_loss",
                    min_delta=1e-4,
                    patience=cfg.early_stop_patience,
                    verbose=False,
                    mode="min",
                )
            )

        epochs = max_epochs if max_epochs is not None else cfg.max_epochs

        trainer = pl_lib.Trainer(
            max_epochs=epochs,
            accelerator=cfg.accelerator,
            devices=1,
            gradient_clip_val=cfg.gradient_clip_val,
            callbacks=callbacks,
            enable_progress_bar=True,
            enable_model_summary=False,
            logger=False,  # Disable Lightning's logger, we use MLflow
            default_root_dir=str(Path("models") / "tft"),
        )

        return trainer

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
                - target_col: Target column name (default: from config).
                - max_epochs: Override max epochs (for testing).

        Returns:
            Training metrics dict.
        """
        target_col = kwargs.get("target_col", self._target_col)
        max_epochs = kwargs.get("max_epochs")

        logger.info(
            "Starting TFT training | samples={} | val={}",
            len(train_df),
            len(val_df) if val_df is not None else 0,
        )

        # Prepare data
        train_prepared = self._prepare_dataframe(train_df, target_col)

        # Create training dataset
        training_dataset = self._create_dataset(train_prepared, target_col)
        self._training_dataset = training_dataset

        # Create validation dataset
        if val_df is not None:
            val_prepared = self._prepare_dataframe(val_df, target_col)
            validation_dataset = TimeSeriesDataSet.from_dataset(
                training_dataset,
                val_prepared,
                predict=False,
                stop_randomization=True,
            )
        else:
            validation_dataset = None

        # Create dataloaders
        cfg = self._tft_config.training
        train_dataloader = training_dataset.to_dataloader(
            train=True,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
        )

        val_dataloader = None
        if validation_dataset is not None:
            val_dataloader = validation_dataset.to_dataloader(
                train=False,
                batch_size=cfg.batch_size,
                num_workers=cfg.num_workers,
            )

        # Build model
        self._model = self._build_model(training_dataset)

        # Create trainer and fit
        trainer = self._create_trainer(max_epochs)
        trainer.fit(
            self._model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
        )

        # Collect metrics
        metrics: dict[str, float] = {}
        if hasattr(trainer, "callback_metrics"):
            for key, value in trainer.callback_metrics.items():
                if isinstance(value, torch.Tensor):
                    metrics[key] = float(value.item())
                else:
                    metrics[key] = float(value)

        logger.info("TFT training complete | metrics={}", metrics)
        return metrics

    def predict(
        self,
        X: pd.DataFrame,
        target_col: str | None = None,
    ) -> pd.DataFrame:
        """Generate predictions using median (0.50 quantile).

        All quantiles are stored in `self._all_quantile_predictions` for later use.

        Args:
            X: Feature DataFrame with DatetimeIndex.
            target_col: Target column name (default: from config).

        Returns:
            DataFrame with 'yhat' (median prediction) column.
        """
        if target_col is None:
            target_col = self._target_col
        if not self.is_fitted:
            msg = "Model must be trained before prediction"
            raise RuntimeError(msg)

        if self._training_dataset is None:
            msg = "Training dataset not available for prediction"
            raise RuntimeError(msg)

        logger.debug("Generating TFT predictions for {} samples", len(X))

        # Prepare data
        pred_df = self._prepare_dataframe(X, target_col, include_target=True)

        # Create prediction dataset
        prediction_dataset = TimeSeriesDataSet.from_dataset(
            self._training_dataset,
            pred_df,
            predict=True,
            stop_randomization=True,
        )

        cfg = self._tft_config.training
        prediction_dataloader = prediction_dataset.to_dataloader(
            train=False,
            batch_size=cfg.batch_size * 2,
            num_workers=cfg.num_workers,
        )

        # Generate predictions
        self._model.eval()  # type: ignore[union-attr]
        raw_predictions = self._model.predict(  # type: ignore[union-attr]
            prediction_dataloader,
            return_index=False,
            return_x=False,
            mode="prediction",
        )

        # Store all quantile predictions
        if isinstance(raw_predictions, torch.Tensor):
            self._all_quantile_predictions = raw_predictions.cpu().numpy()
        else:
            self._all_quantile_predictions = np.array(raw_predictions)

        # Extract median (0.50 quantile)
        if 0.50 in self._quantiles:
            median_idx = self._quantiles.index(0.50)
        else:
            median_idx = len(self._quantiles) // 2

        if len(self._all_quantile_predictions.shape) == 3:
            # Shape: (batch, prediction_length, n_quantiles)
            median_pred = self._all_quantile_predictions[:, :, median_idx].flatten()
        else:
            median_pred = self._all_quantile_predictions.flatten()

        # Align predictions with input index
        n_preds = len(median_pred)
        result_index = X.index[-n_preds:] if n_preds <= len(X) else X.index

        result = pd.DataFrame(
            {"yhat": median_pred[-len(result_index):]},
            index=result_index,
        )

        return result

    def predict_rolling(
        self,
        X: pd.DataFrame,
        target_col: str | None = None,
        *,
        step: int | None = None,
    ) -> pd.DataFrame:
        """Generate predictions using a sliding window across the full input.

        For inputs longer than encoder_length + prediction_length, slides a window
        of that size across the data, calling predict() per window. Overlapping
        predictions are averaged.

        For short inputs (<= encoder + prediction), delegates directly to predict().

        Args:
            X: Feature DataFrame with DatetimeIndex.
            target_col: Target column name (default: from config).
            step: Window step size in rows. Defaults to prediction_length
                  (non-overlapping predictions).

        Returns:
            DataFrame with 'yhat' column covering the prediction region of X.
        """
        if target_col is None:
            target_col = self._target_col

        enc_len = self._tft_config.training.encoder_length
        pred_len = self._tft_config.training.prediction_length
        window_size = enc_len + pred_len

        # Short input: delegate directly
        if len(X) <= window_size:
            return self.predict(X, target_col)

        if step is None:
            step = pred_len

        from collections import defaultdict

        all_preds: defaultdict[Any, list[float]] = defaultdict(list)
        n_windows = 0
        n_failed = 0
        pos = enc_len  # First decode position (encoder ends here)

        while pos < len(X):
            # Calculate window boundaries
            window_end = min(pos + pred_len, len(X))
            window_start = window_end - pred_len - enc_len

            # If last window is short, slide back to maintain full window
            if window_start < 0:
                window_start = 0
                window_end = min(window_size, len(X))

            window_df = X.iloc[window_start:window_end]

            if len(window_df) < window_size:
                # Skip windows that can't form a complete encoder+decoder
                pos += step
                continue

            try:
                preds = self.predict(window_df, target_col)
                for ts, row in preds.iterrows():
                    all_preds[ts].append(float(row["yhat"]))
                n_windows += 1
            except Exception:
                n_failed += 1
                logger.warning(
                    "Rolling window {} failed (start={}), skipping",
                    n_windows + n_failed,
                    window_df.index[0],
                )

            pos += step

        if n_windows == 0:
            msg = f"All {n_failed} rolling windows failed"
            raise RuntimeError(msg)

        if n_failed > 0:
            logger.warning(
                "Rolling prediction: {}/{} windows succeeded",
                n_windows,
                n_windows + n_failed,
            )

        # Build result: average overlapping predictions, sort by timestamp
        sorted_ts = sorted(all_preds.keys())
        averaged = [np.mean(all_preds[ts]) for ts in sorted_ts]

        result = pd.DataFrame(
            {"yhat": averaged},
            index=pd.DatetimeIndex(sorted_ts),
        )

        logger.debug(
            "Rolling prediction: {} windows, {} unique timestamps",
            n_windows,
            len(result),
        )

        return result

    def get_quantile_predictions(self) -> dict[float, NDArray[np.floating[Any]]]:
        """Get all quantile predictions from last predict() call.

        Returns:
            Dict mapping quantile value to predictions array.
        """
        if self._all_quantile_predictions is None:
            msg = "No predictions available. Call predict() first."
            raise RuntimeError(msg)

        result: dict[float, NDArray[np.floating[Any]]] = {}
        for i, q in enumerate(self._quantiles):
            if len(self._all_quantile_predictions.shape) == 3:
                result[q] = self._all_quantile_predictions[:, :, i].flatten()
            else:
                result[q] = self._all_quantile_predictions.flatten()

        return result

    def save(self, path: Path) -> None:
        """Save TFT model checkpoint, metadata, and training dataset.

        Saves all artifacts needed for full model reconstruction:
        - state_dict (model weights)
        - metadata (architecture params for rebuilding)
        - dataset_params (feature structure)
        - training_dataset (normalizers + structure for prediction)

        Args:
            path: Directory to save model files.
        """
        if not self.is_fitted:
            msg = "Cannot save unfitted model"
            raise ValueError(msg)

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        logger.info("Saving TFT model to {}", path)

        # Save model state dict
        model_path = path / self.MODEL_FILENAME
        torch.save(self._model.state_dict(), model_path)  # type: ignore[union-attr]

        # Save dataset params for reconstruction
        params_path = path / self.DATASET_PARAMS_FILENAME
        with open(params_path, "w") as f:
            json.dump(self._dataset_params, f, indent=2)

        # Save metadata with full architecture params for model rebuild
        metadata = {
            "quantiles": self._quantiles,
            "hidden_size": self._tft_config.architecture.hidden_size,
            "attention_head_size": self._tft_config.architecture.attention_head_size,
            "lstm_layers": self._tft_config.architecture.lstm_layers,
            "dropout": self._tft_config.architecture.dropout,
            "hidden_continuous_size": self._tft_config.architecture.hidden_continuous_size,
            "learning_rate": self._tft_config.training.learning_rate,
            "encoder_length": self._tft_config.training.encoder_length,
            "prediction_length": self._tft_config.training.prediction_length,
            "batch_size": self._tft_config.training.batch_size,
            "num_workers": self._tft_config.training.num_workers,
            "reduce_on_plateau_patience": self._tft_config.training.early_stop_patience,
        }
        metadata_path = path / self.METADATA_FILENAME
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        # Save training dataset (pickle) for prediction reconstruction
        if self._training_dataset is not None:
            ds_path = path / self.TRAINING_DATASET_FILENAME
            with open(ds_path, "wb") as f:
                pickle.dump(self._training_dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.debug("Saved training dataset to {}", ds_path)

        logger.info("TFT model saved successfully")

    @classmethod
    def from_checkpoint(cls, path: Path | str) -> TFTForecaster:
        """Load a fully functional TFTForecaster from a saved checkpoint directory.

        Reconstructs the model architecture from the training dataset and metadata,
        then loads the trained weights. The returned instance is ready for prediction.

        Args:
            path: Directory containing saved model files (ckpt, metadata, dataset).

        Returns:
            TFTForecaster ready for prediction.

        Raises:
            FileNotFoundError: If required files are missing.
        """
        path = Path(path)
        logger.info("Loading TFT model from checkpoint: {}", path)

        # Load metadata
        metadata_path = path / cls.METADATA_FILENAME
        if not metadata_path.exists():
            msg = f"Metadata not found: {metadata_path}"
            raise FileNotFoundError(msg)
        with open(metadata_path) as f:
            metadata = json.load(f)

        # Load dataset params
        params_path = path / cls.DATASET_PARAMS_FILENAME
        dataset_params: dict[str, Any] = {}
        if params_path.exists():
            with open(params_path) as f:
                dataset_params = json.load(f)

        # Load training dataset (required for model architecture reconstruction)
        ds_path = path / cls.TRAINING_DATASET_FILENAME
        if not ds_path.exists():
            msg = (
                f"Training dataset not found: {ds_path}. "
                "Model was saved with an older format. Please retrain."
            )
            raise FileNotFoundError(msg)
        with open(ds_path, "rb") as f:
            training_dataset: TimeSeriesDataSet = pickle.load(f)  # noqa: S301

        # Load state dict
        model_path = path / cls.MODEL_FILENAME
        if not model_path.exists():
            msg = f"Model checkpoint not found: {model_path}"
            raise FileNotFoundError(msg)
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)

        # Reconstruct model architecture from dataset + metadata
        quantiles = metadata.get("quantiles", [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98])
        tft_model = TemporalFusionTransformer.from_dataset(
            training_dataset,
            hidden_size=metadata.get("hidden_size", 64),
            attention_head_size=metadata.get("attention_head_size", 4),
            dropout=metadata.get("dropout", 0.1),
            hidden_continuous_size=metadata.get("hidden_continuous_size", 16),
            lstm_layers=metadata.get("lstm_layers", 2),
            output_size=len(quantiles),
            loss=QuantileLoss(quantiles=quantiles),
            learning_rate=metadata.get("learning_rate", 0.001),
            reduce_on_plateau_patience=metadata.get("reduce_on_plateau_patience", 4),
        )
        tft_model.load_state_dict(state_dict)
        tft_model.eval()

        # Build a minimal TFTConfig for the instance
        from energy_forecast.config.settings import (
            TFTArchitectureConfig,
            TFTConfig,
            TFTCovariatesConfig,
            TFTTrainingConfig,
        )

        config = TFTConfig(
            architecture=TFTArchitectureConfig(
                hidden_size=metadata.get("hidden_size", 64),
                attention_head_size=metadata.get("attention_head_size", 4),
                lstm_layers=metadata.get("lstm_layers", 2),
                dropout=metadata.get("dropout", 0.1),
                hidden_continuous_size=metadata.get("hidden_continuous_size", 16),
            ),
            training=TFTTrainingConfig(
                encoder_length=metadata.get("encoder_length", 168),
                prediction_length=metadata.get("prediction_length", 48),
                batch_size=metadata.get("batch_size", 64),
                max_epochs=1,  # Not used for inference
                learning_rate=metadata.get("learning_rate", 0.001),
                num_workers=metadata.get("num_workers", 0),
            ),
            covariates=TFTCovariatesConfig(
                time_varying_known=dataset_params.get("time_varying_known_reals", []),
            ),
            quantiles=quantiles,
        )

        # Construct the instance without calling __init__ training path
        instance = cls(config)
        instance._model = tft_model
        instance._training_dataset = training_dataset
        instance._dataset_params = dataset_params
        instance._quantiles = quantiles
        instance._loaded_state_dict = None

        logger.info("TFT model loaded successfully — ready for prediction")
        return instance

    def load(self, path: Path) -> None:
        """Load TFT model from checkpoint directory.

        If a training dataset pickle is available, the model is fully
        reconstructed and ready for prediction. Otherwise, only the state
        dict is loaded (backward compatibility).

        Args:
            path: Directory containing saved model files.
        """
        path = Path(path)
        logger.info("Loading TFT model from {}", path)

        # Load dataset params
        params_path = path / self.DATASET_PARAMS_FILENAME
        if params_path.exists():
            with open(params_path) as f:
                self._dataset_params = json.load(f)

        # Load metadata
        metadata_path = path / self.METADATA_FILENAME
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
                self._quantiles = metadata.get("quantiles", self._quantiles)

        # Model checkpoint
        model_path = path / self.MODEL_FILENAME
        if not model_path.exists():
            msg = f"Model checkpoint not found: {model_path}"
            raise FileNotFoundError(msg)

        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)

        # Try to load training dataset for full reconstruction
        ds_path = path / self.TRAINING_DATASET_FILENAME
        if ds_path.exists():
            with open(ds_path, "rb") as f:
                self._training_dataset = pickle.load(f)  # noqa: S301

            quantiles = self._quantiles
            tft_model = TemporalFusionTransformer.from_dataset(
                self._training_dataset,
                hidden_size=metadata.get("hidden_size", 64),
                attention_head_size=metadata.get("attention_head_size", 4),
                dropout=metadata.get("dropout", 0.1),
                hidden_continuous_size=metadata.get("hidden_continuous_size", 16),
                lstm_layers=metadata.get("lstm_layers", 2),
                output_size=len(quantiles),
                loss=QuantileLoss(quantiles=quantiles),
                learning_rate=metadata.get("learning_rate", 0.001),
                reduce_on_plateau_patience=metadata.get(
                    "reduce_on_plateau_patience", 4
                ),
            )
            tft_model.load_state_dict(state_dict)
            tft_model.eval()
            self._model = tft_model
            logger.info("TFT model fully reconstructed — ready for prediction")
        else:
            # Fallback: store state dict for later use (old format)
            self._loaded_state_dict = state_dict
            logger.warning(
                "Training dataset not found. State dict loaded but model "
                "needs retraining with new save format for full prediction."
            )
