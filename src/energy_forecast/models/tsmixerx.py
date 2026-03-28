"""TSMixerx forecaster for hourly consumption prediction.

Wraps NeuralForecast's TSMixerx model to conform to BaseForecaster interface.
Point forecast (MAE loss) — no quantile output unlike TFT.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

from energy_forecast.config import TSMixerxConfig
from energy_forecast.models.base import PREDICTION_COL, BaseForecaster

# NeuralForecast long-format constants
NF_UNIQUE_ID = "uludag"


class TSMixerxForecaster(BaseForecaster):
    """TSMixerx-based hourly consumption forecaster (point forecast).

    Uses NeuralForecast's TSMixerx (MLP-Mixer for time series) with
    time-mixing and feature-mixing blocks. Single-series mode (n_series=1).

    Args:
        config: TSMixerx configuration from settings.
    """

    METADATA_FILENAME = "metadata.json"

    def __init__(self, config: TSMixerxConfig) -> None:
        super().__init__(config.model_dump())
        self._tsmixerx_config = config
        self._nf: Any | None = None  # NeuralForecast instance
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
            dt_col = nf_df.columns[0]
        nf_df = nf_df.rename(columns={dt_col: "ds"})

        nf_df["unique_id"] = NF_UNIQUE_ID
        nf_df = nf_df.rename(columns={target_col: "y"})

        # Filter to only specified covariates (NF converts ALL columns to float32)
        cfg = self._tsmixerx_config.covariates
        covariate_cols = [
            c for c in list(cfg.futr_exog) + list(cfg.hist_exog) if c in nf_df.columns
        ]
        keep_cols = ["unique_id", "ds", "y", *covariate_cols]
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
        """Build NeuralForecast TSMixerx model from config.

        Args:
            callbacks: Extra Lightning callbacks (e.g. Optuna pruning).
            max_steps: Override max_steps (for HPO).

        Returns:
            NeuralForecast instance wrapping a TSMixerx model.
        """
        from neuralforecast import NeuralForecast
        from neuralforecast.losses.pytorch import MAE, HuberLoss, MSE, RMSE
        from neuralforecast.models import TSMixerx

        cfg = self._tsmixerx_config
        arch = cfg.architecture
        train_cfg = cfg.training

        # Config-driven loss selection
        loss_map: dict[str, Any] = {
            "mae": MAE(),
            "mse": MSE(),
            "rmse": RMSE(),
            "huber": HuberLoss(delta=1.0),
            "huber_0.5": HuberLoss(delta=0.5),
            "huber_2.0": HuberLoss(delta=2.0),
        }
        loss_fn = loss_map.get(cfg.loss, MAE())
        logger.info("TSMixerx loss function: {} (config={})", type(loss_fn).__name__, cfg.loss)

        steps = max_steps if max_steps is not None else train_cfg.max_steps

        # NeuralForecast uses **kwargs to capture extra arguments as trainer_kwargs.
        # Pass accelerator/progress bar/logger as flat kwargs, NOT in a dict.
        extra_trainer_kwargs: dict[str, Any] = {
            "accelerator": train_cfg.accelerator,
            "enable_progress_bar": train_cfg.enable_progress_bar,
            "logger": False,  # Windows tensorboard crash prevention
        }
        # NF defaults devices=-1 (all GPUs); CPU requires devices=1
        if train_cfg.accelerator == "cpu":
            extra_trainer_kwargs["devices"] = 1

        # Progress bar: refresh every 100 steps instead of every step
        if train_cfg.enable_progress_bar:
            from pytorch_lightning.callbacks import TQDMProgressBar

            progress_callbacks = [TQDMProgressBar(refresh_rate=100)]
            callbacks = list(callbacks) + progress_callbacks if callbacks else progress_callbacks
        if callbacks:
            extra_trainer_kwargs["callbacks"] = callbacks

        model = TSMixerx(
            h=train_cfg.prediction_length,
            input_size=arch.input_size,
            n_series=1,  # Single time series (Uludag region)
            n_block=arch.n_block,
            ff_dim=arch.ff_dim,
            dropout=arch.dropout,
            revin=arch.revin,
            loss=loss_fn,
            learning_rate=train_cfg.learning_rate,
            max_steps=steps,
            val_check_steps=train_cfg.val_check_steps,
            early_stop_patience_steps=train_cfg.early_stop_patience_steps,
            scaler_type=train_cfg.scaler_type,
            batch_size=1,  # Single time series
            windows_batch_size=train_cfg.windows_batch_size,
            step_size=train_cfg.step_size,
            futr_exog_list=list(cfg.covariates.futr_exog),
            hist_exog_list=list(cfg.covariates.hist_exog),
            num_lr_decays=-1,
            random_seed=train_cfg.random_seed,
            **extra_trainer_kwargs,
        )

        nf = NeuralForecast(models=[model], freq="h")

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.debug("TSMixerx model built with {} trainable parameters", n_params)

        return nf

    def train(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame | None = None,
        **kwargs: Any,
    ) -> dict[str, float]:
        """Train TSMixerx model.

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
            "Starting TSMixerx training | samples={} | val={}",
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
            tsmixerx_model = nf.models[0]
            if hasattr(tsmixerx_model, "trainer") and tsmixerx_model.trainer is not None:
                import torch

                for key, value in tsmixerx_model.trainer.callback_metrics.items():
                    if isinstance(value, torch.Tensor):
                        metrics[key] = float(value.item())
                    else:
                        metrics[key] = float(value)
        except Exception as e:
            logger.debug("Could not extract trainer metrics: {}", e)

        logger.info("TSMixerx training complete | metrics={}", metrics)
        return metrics

    def predict(
        self,
        X: pd.DataFrame,
        target_col: str | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Generate point predictions.

        NeuralForecast predicts the next h steps from the end of the provided
        context DataFrame. The last prediction_length timestamps in X are used
        as the prediction target period.

        Args:
            X: Feature DataFrame with DatetimeIndex.
            target_col: Target column name.

        Returns:
            DataFrame with PREDICTION_COL column.
        """
        if target_col is None:
            target_col = self._target_col
        if not self.is_fitted:
            msg = "Model must be trained before prediction"
            raise RuntimeError(msg)

        pred_len = self._tsmixerx_config.training.prediction_length

        logger.debug("Generating TSMixerx predictions for {} samples", len(X))

        enc_len = self._tsmixerx_config.architecture.input_size

        # Determine context and forecast boundaries
        if len(X) > pred_len:
            context_end = len(X) - pred_len
            context_df = X.iloc[max(0, context_end - enc_len) : context_end]
            forecast_df = X.iloc[context_end:]
        else:
            context_df = None
            forecast_df = X.iloc[-pred_len:]

        # Prepare future exogenous DataFrame
        futr_cols = list(self._tsmixerx_config.covariates.futr_exog)
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
                context_df,
                target_col,
                drop_target_nan=False,
            )
            if nf_context["y"].isna().any():
                nf_context = nf_context.copy()
                nf_context["y"] = nf_context["y"].ffill().bfill()

        # Generate predictions
        if self._nf is None:
            msg = "Model not fitted — call fit() or load() first"
            raise RuntimeError(msg)
        preds = self._nf.predict(df=nf_context, futr_df=futr_df)

        # Extract point prediction — TSMixerx outputs "TSMixerx" column
        pred_col = "TSMixerx"
        if pred_col not in preds.columns:
            # Fallback: use first available prediction column
            pred_cols = [c for c in preds.columns if "TSMixerx" in c]
            pred_col = pred_cols[0] if pred_cols else preds.columns[-1]

        # Build result DataFrame
        pred_values = preds[pred_col].values
        n_preds = len(pred_values)
        result_index = forecast_df.index[-n_preds:]

        result = pd.DataFrame(
            {PREDICTION_COL: pred_values[-len(result_index) :]},
            index=result_index,
        )

        return result

    def rolling_predict(
        self,
        full_df: pd.DataFrame,
        eval_start: pd.Timestamp,
        eval_end: pd.Timestamp,
        target_col: str | None = None,
        step_hours: int = 24,
    ) -> pd.DataFrame:
        """Predict full evaluation period using 48h rolling windows.

        Slides a prediction window across the evaluation period, producing
        predictions for every hour. For overlapping windows, the latest
        prediction wins (production-faithful).

        Args:
            full_df: Full DataFrame including history before eval_start.
            eval_start: Start of evaluation period (inclusive).
            eval_end: End of evaluation period (inclusive).
            target_col: Target column name.
            step_hours: Hours between consecutive prediction origins.

        Returns:
            DataFrame with PREDICTION_COL covering eval_start..eval_end.

        Raises:
            RuntimeError: If model is not fitted.
        """
        if target_col is None:
            target_col = self._target_col
        if not self.is_fitted:
            msg = "Model must be trained before prediction"
            raise RuntimeError(msg)
        if self._nf is None:
            msg = "Model not fitted — call fit() or load() first"
            raise RuntimeError(msg)

        enc_len = self._tsmixerx_config.architecture.input_size
        pred_len = self._tsmixerx_config.training.prediction_length
        futr_cols = list(self._tsmixerx_config.covariates.futr_exog)
        step_delta = pd.Timedelta(hours=step_hours)
        pred_delta = pd.Timedelta(hours=pred_len)

        # Collect predictions: later windows overwrite earlier (last wins)
        predictions: dict[pd.Timestamp, float] = {}

        origin = pd.Timestamp(eval_start)
        eval_end_ts = pd.Timestamp(eval_end)

        n_windows = 0
        while origin + pred_delta - pd.Timedelta(hours=1) <= eval_end_ts:
            # Context window: [origin - enc_len, origin)
            ctx_start = origin - pd.Timedelta(hours=enc_len)
            context_df = full_df.loc[ctx_start : origin - pd.Timedelta(hours=1)]

            if len(context_df) < enc_len // 2:
                origin += step_delta
                continue

            # Forecast window: [origin, origin + pred_len)
            fcast_end = origin + pred_delta - pd.Timedelta(hours=1)
            forecast_df = full_df.loc[origin:fcast_end]

            if forecast_df.empty:
                origin += step_delta
                continue

            # Build NF context
            nf_context = self._to_nf_format(context_df, target_col, drop_target_nan=False)
            if nf_context["y"].isna().any():
                nf_context = nf_context.copy()
                nf_context["y"] = nf_context["y"].ffill().bfill()

            # Build future exogenous DataFrame
            futr_data: dict[str, Any] = {
                "unique_id": NF_UNIQUE_ID,
                "ds": forecast_df.index,
            }
            for col in futr_cols:
                if col in forecast_df.columns:
                    futr_data[col] = forecast_df[col].values
            futr_df = pd.DataFrame(futr_data)

            # NF predict
            preds = self._nf.predict(df=nf_context, futr_df=futr_df)

            # Extract point prediction column
            pred_col = "TSMixerx"
            if pred_col not in preds.columns:
                pred_cols_avail = [c for c in preds.columns if "TSMixerx" in c]
                pred_col = pred_cols_avail[0] if pred_cols_avail else preds.columns[-1]

            pred_values = preds[pred_col].values
            for i, ts in enumerate(forecast_df.index[: len(pred_values)]):
                predictions[ts] = float(pred_values[i])

            n_windows += 1
            origin += step_delta

        logger.info(
            "Rolling predict: {} windows, {} unique hours predicted",
            n_windows,
            len(predictions),
        )

        if not predictions:
            return pd.DataFrame({PREDICTION_COL: []}, index=pd.DatetimeIndex([]))

        result_ts = sorted(predictions.keys())
        result = pd.DataFrame(
            {PREDICTION_COL: [predictions[ts] for ts in result_ts]},
            index=pd.DatetimeIndex(result_ts),
        )
        result = result.loc[eval_start:eval_end]

        return result

    def save(self, path: Path) -> None:
        """Save TSMixerx model using NeuralForecast's built-in save.

        Args:
            path: Directory to save model files.
        """
        if not self.is_fitted:
            msg = "Cannot save unfitted model"
            raise ValueError(msg)

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        logger.info("Saving TSMixerx model to {}", path)

        if self._nf is None:
            msg = "Model not fitted — call fit() or load() first"
            raise RuntimeError(msg)
        self._nf.save(path=str(path), overwrite=True)

        # Strip training callbacks from checkpoint — they hold dataset references
        # (inflating file size 100x) and cause duplicate EarlyStopping on load
        import torch

        ckpt_files = list(path.glob("*.ckpt"))
        for ckpt_file in ckpt_files:
            # weights_only=False needed to access hyper_parameters dict for callback strip.
            # Risk mitigated: only loads our own checkpoints, SHA256 verified in from_checkpoint()
            ckpt = torch.load(ckpt_file, map_location="cpu", weights_only=False)
            if "hyper_parameters" in ckpt and "callbacks" in ckpt["hyper_parameters"]:
                ckpt["hyper_parameters"]["callbacks"] = []
                torch.save(ckpt, ckpt_file)
                logger.debug("Stripped callbacks from {}", ckpt_file.name)

        # Compute checkpoint hashes for integrity verification
        ckpt_hashes: dict[str, str] = {}
        for ckpt_file in path.glob("*.ckpt"):
            ckpt_hashes[ckpt_file.name] = hashlib.sha256(ckpt_file.read_bytes()).hexdigest()

        # Save metadata (architecture, training, covariates, hashes)
        metadata = {
            "architecture": self._tsmixerx_config.architecture.model_dump(),
            "training": {
                "input_size": self._tsmixerx_config.architecture.input_size,
                "prediction_length": self._tsmixerx_config.training.prediction_length,
                "num_workers": self._tsmixerx_config.training.num_workers,
                "scaler_type": self._tsmixerx_config.training.scaler_type,
            },
            "covariates": self._tsmixerx_config.covariates.model_dump(),
            "ckpt_hashes": ckpt_hashes,
        }
        metadata_path = path / self.METADATA_FILENAME
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info("TSMixerx model saved successfully")

    @classmethod
    def from_checkpoint(cls, path: Path | str) -> TSMixerxForecaster:
        """Load a fully functional TSMixerxForecaster from a saved checkpoint.

        Args:
            path: Directory containing saved model files.

        Returns:
            TSMixerxForecaster ready for prediction.
        """
        from neuralforecast import NeuralForecast

        from energy_forecast.config import (
            TSMixerxArchitectureConfig,
            TSMixerxConfig,
            TSMixerxCovariatesConfig,
            TSMixerxTrainingConfig,
        )

        path = Path(path)
        logger.info("Loading TSMixerx model from checkpoint: {}", path)

        # Load metadata
        metadata_path = path / cls.METADATA_FILENAME
        if not metadata_path.exists():
            msg = f"Metadata not found: {metadata_path}"
            raise FileNotFoundError(msg)
        with open(metadata_path) as f:
            metadata = json.load(f)

        # Verify checkpoint integrity
        ckpt_hashes = metadata.get("ckpt_hashes", {})
        if ckpt_hashes:
            for ckpt_name, expected_hash in ckpt_hashes.items():
                ckpt_file = path / ckpt_name
                if ckpt_file.exists():
                    actual_hash = hashlib.sha256(ckpt_file.read_bytes()).hexdigest()
                    if actual_hash != expected_hash:
                        msg = (
                            f"TSMixerx checkpoint integrity check failed: {ckpt_name} "
                            f"(expected {expected_hash[:12]}..., "
                            f"got {actual_hash[:12]}...)"
                        )
                        raise RuntimeError(msg)
            logger.debug("TSMixerx checkpoint integrity verified ({} files)", len(ckpt_hashes))

        # Load NeuralForecast model
        nf = NeuralForecast.load(path=str(path))

        # Strip training callbacks from loaded checkpoint
        for model in nf.models:
            if hasattr(model, "hparams") and "callbacks" in model.hparams:
                model.hparams["callbacks"] = []
                logger.debug("Cleared training callbacks from loaded checkpoint")

        # Reconstruct TSMixerxConfig from metadata
        arch_data = metadata.get("architecture", {})
        train_data = metadata.get("training", {})
        cov_data = metadata.get("covariates", {})

        config = TSMixerxConfig(
            architecture=TSMixerxArchitectureConfig(**arch_data),
            training=TSMixerxTrainingConfig(
                prediction_length=train_data.get("prediction_length", 48),
                max_steps=1,  # Not used for inference
                num_workers=train_data.get("num_workers", 4),
                scaler_type=train_data.get("scaler_type", "robust"),
            ),
            covariates=TSMixerxCovariatesConfig(**cov_data),
        )

        instance = cls(config)
        instance._nf = nf

        logger.info("TSMixerx model loaded successfully — ready for prediction")
        return instance

    def load(self, path: Path) -> None:
        """Load TSMixerx model from checkpoint directory.

        Args:
            path: Directory containing saved model files.
        """
        from neuralforecast import NeuralForecast

        path = Path(path)
        logger.info("Loading TSMixerx model from {}", path)

        self._nf = NeuralForecast.load(path=str(path))

        # Strip training callbacks to prevent duplicate EarlyStopping crash
        for model in self._nf.models:
            if hasattr(model, "hparams") and "callbacks" in model.hparams:
                model.hparams["callbacks"] = []

        # Load metadata
        metadata_path = path / self.METADATA_FILENAME
        if metadata_path.exists():
            with open(metadata_path) as f:
                json.load(f)  # validate JSON, no state to restore

        logger.info("TSMixerx model loaded successfully")
