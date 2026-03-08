"""Prediction orchestration service."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger
from pydantic import BaseModel, Field

from energy_forecast.config import Settings
from energy_forecast.data.epias_client import EpiasClient
from energy_forecast.data.exceptions import EpiasAuthError
from energy_forecast.data.loader import DataLoader
from energy_forecast.data.openmeteo_client import OpenMeteoClient
from energy_forecast.features.pipeline import FeaturePipeline
from energy_forecast.models.ensemble import EnsembleForecaster
from energy_forecast.serving.exceptions import (
    FeaturePipelineError,
    ModelNotLoadedError,
    PredictionError,
)
from energy_forecast.utils import TZ_ISTANBUL


class PredictionServiceConfig(BaseModel, frozen=True):
    """Prediction service configuration."""

    models_dir: Path = Field(default=Path("models"))
    catboost_path: Path = Field(default=Path("models/catboost/model.cbm"))
    prophet_path: Path = Field(default=Path("models/prophet"))
    tft_path: Path = Field(default=Path("models/tft"))
    ensemble_dir: Path | None = Field(default=None)
    forecast_horizon: int = Field(default=48, ge=1)


class PredictionService:
    """Orchestrates prediction pipeline: data → features → ensemble.

    Maintains train-serve parity by using the same FeaturePipeline
    configuration as training.

    Args:
        config: Prediction service configuration.
        settings: Full application settings.
    """

    def __init__(
        self,
        config: PredictionServiceConfig,
        settings: Settings,
    ) -> None:
        self._config = config
        self._settings = settings
        self._ensemble: EnsembleForecaster | None = None
        self._feature_pipeline: FeaturePipeline | None = None
        self._data_loader: DataLoader | None = None
        self._models_loaded = False
        self._warnings: list[str] = []

    @property
    def is_ready(self) -> bool:
        """Check if models are loaded and service is ready."""
        return self._models_loaded and self._ensemble is not None

    @property
    def warnings(self) -> list[str]:
        """Warnings collected during the last prediction run."""
        return list(self._warnings)

    def load_models(self) -> None:
        """Load ensemble models and initialize pipeline.

        Should be called once at application startup.

        Raises:
            ModelNotLoadedError: If any model fails to load.
        """
        logger.info("Loading prediction models...")

        try:
            # Initialize data loader
            self._data_loader = DataLoader(self._settings.data_loader)

            # Initialize feature pipeline
            self._feature_pipeline = FeaturePipeline(self._settings)

            # Load ensemble with config
            ensemble_config = {
                "active_models": list(self._settings.ensemble.active_models),
                "weights": {
                    "catboost": self._settings.ensemble.weights.catboost,
                    "prophet": self._settings.ensemble.weights.prophet,
                    "tft": self._settings.ensemble.weights.tft,
                },
                "target_col": "consumption",
                "prophet_regressors": [
                    r.name for r in self._settings.prophet.regressors
                ],
            }
            self._ensemble = EnsembleForecaster(ensemble_config)

            # Try to load ensemble weights if available (timestamped subdir or legacy)
            ensemble_dir = self._config.ensemble_dir
            if ensemble_dir and (ensemble_dir / "ensemble_weights.json").exists():
                self._ensemble.load(ensemble_dir)
            else:
                # Legacy fallback: models/ensemble_weights.json
                legacy_path = self._config.models_dir / "ensemble_weights.json"
                if legacy_path.exists():
                    self._ensemble.load(self._config.models_dir)

            # Load individual models
            self._ensemble.load_models(
                catboost_path=self._config.catboost_path
                if self._config.catboost_path.exists()
                else None,
                prophet_path=self._config.prophet_path
                if self._config.prophet_path.exists()
                else None,
                tft_path=self._config.tft_path
                if self._config.tft_path.exists()
                else None,
            )

            self._models_loaded = True
            logger.info(
                "Models loaded successfully. Active: {}",
                self._ensemble.active_models,
            )

        except Exception as e:
            logger.error("Failed to load models: {}", e)
            raise ModelNotLoadedError(f"Model loading failed: {e}") from e

    def run_prediction(
        self,
        excel_path: Path,
        progress_callback: Callable[[str], None] | None = None,
    ) -> pd.DataFrame:
        """Run full prediction pipeline.

        1. Load consumption data from Excel
        2. Fetch EPIAS data (cached or API)
        3. Fetch weather forecast for T and T+1
        4. Run feature pipeline
        5. Generate ensemble predictions

        Args:
            excel_path: Path to uploaded Excel file.
            progress_callback: Optional callback for progress updates.

        Returns:
            DataFrame with 24-hour T+1 predictions.

        Raises:
            ModelNotLoadedError: If models not loaded.
            PredictionError: If prediction fails.
        """
        if not self.is_ready:
            raise ModelNotLoadedError("Models not loaded. Call load_models() first.")

        if not self._data_loader or not self._feature_pipeline or not self._ensemble:
            raise ModelNotLoadedError("Models not loaded. Call load_models() first.")

        self._warnings = []  # reset warnings for each prediction run
        start_time = time.perf_counter()

        def update_progress(msg: str) -> None:
            logger.info(msg)
            if progress_callback:
                progress_callback(msg)

        try:
            # Step 1: Load Excel data
            update_progress("Loading consumption data from Excel...")
            consumption_df = self._data_loader.load_excel(excel_path)
            last_timestamp = consumption_df.index.max()
            logger.info("Last data point: {}", last_timestamp)

            # Step 2: Extend for forecast period
            update_progress("Extending data for forecast horizon...")
            extended_df = self._data_loader.extend_for_forecast(
                consumption_df,
                horizon_hours=self._config.forecast_horizon,
            )

            # Step 3: Fetch EPIAS data
            update_progress("Fetching EPIAS market data...")
            epias_df = self._fetch_epias_data(extended_df)
            merged_df = extended_df.join(epias_df, how="left")

            # Step 3.5: Fetch generation data
            update_progress("Fetching generation data...")
            generation_df = self._fetch_generation_data(extended_df)
            if not generation_df.empty:
                merged_df = merged_df.join(generation_df, how="left")

            # Build EPIAS snapshot metadata
            epias_meta: dict[str, object] = {}
            if not epias_df.empty:
                epias_meta["data_range"] = {
                    "start": str(epias_df.index.min()),
                    "end": str(epias_df.index.max()),
                }
                epias_meta["row_count"] = len(epias_df)
                epias_meta["last_values"] = {
                    col: round(float(epias_df[col].iloc[-1]), 1)
                    for col in epias_df.columns
                    if pd.notna(epias_df[col].iloc[-1])
                }
                epias_meta["nan_summary"] = {
                    col: int(epias_df[col].isna().sum())
                    for col in epias_df.columns
                }

            # Step 4: Fetch weather data (historical + forecast)
            update_progress("Fetching weather data...")
            weather_df = self._fetch_weather_data(extended_df)
            merged_df = merged_df.join(weather_df, how="left")

            # Step 5: Run feature pipeline
            update_progress("Running feature engineering pipeline...")
            try:
                features_df = self._feature_pipeline.run(merged_df)
            except Exception as e:
                raise FeaturePipelineError(f"Feature pipeline failed: {e}") from e

            # Step 6: Extract forecast rows
            # (last 48 hours with NaN consumption, filtered to T+1 in output)
            forecast_mask = features_df.index > last_timestamp
            forecast_features = features_df.loc[forecast_mask].copy()

            if len(forecast_features) == 0:
                raise PredictionError("No forecast rows available after feature pipeline")

            # Step 7: Generate ensemble predictions
            update_progress("Generating ensemble predictions...")
            historical_features = features_df.loc[~forecast_mask].copy()
            predictions = self._ensemble.predict(
                forecast_features,
                history=historical_features,
            )

            # Step 8: Prepare output DataFrame
            # Keep raw predictions (with per-model columns) for DB storage
            raw_predictions = predictions.copy()
            result = self._prepare_output(predictions, last_timestamp)

            latency_ms = (time.perf_counter() - start_time) * 1000
            logger.info(
                "Prediction completed in {:.0f}ms — {} rows from {} to {}",
                latency_ms,
                len(result),
                result.index.min(),
                result.index.max(),
            )
            update_progress("Prediction complete!")

            # Attach metadata for API response and DB persistence
            result.attrs["latency_ms"] = round(latency_ms)
            result.attrs["weather_data"] = weather_df
            result.attrs["epias_snapshot"] = epias_meta
            result.attrs["features_df"] = features_df
            result.attrs["forecast_mask"] = forecast_mask
            result.attrs["raw_predictions"] = raw_predictions
            return result

        except (ModelNotLoadedError, PredictionError, FeaturePipelineError):
            raise
        except Exception as e:
            logger.error("Prediction failed: {}", e)
            raise PredictionError(f"Prediction failed: {e}") from e

    def _fetch_epias_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fetch EPIAS data for the date range in df.

        Raises:
            EpiasAuthError: If authentication fails (critical, not recoverable).
        """
        start_date = df.index.min().strftime("%Y-%m-%d")
        end_date = df.index.max().strftime("%Y-%m-%d")

        try:
            with EpiasClient(
                username=self._settings.env.epias_username,
                password=self._settings.env.epias_password,
                config=self._settings.epias_api,
            ) as client:
                return client.fetch(start_date, end_date)
        except EpiasAuthError:
            raise  # auth failures are critical — do not swallow
        except Exception as e:
            msg = f"EPIAS fetch failed, predictions will lack market features: {e}"
            logger.warning(msg)
            self._warnings.append(msg)
            return pd.DataFrame(index=df.index)

    def _fetch_generation_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fetch EPIAS generation data for the date range in df.

        Raises:
            EpiasAuthError: If authentication fails (critical, not recoverable).
        """
        start_date = df.index.min().strftime("%Y-%m-%d")
        end_date = df.index.max().strftime("%Y-%m-%d")

        try:
            with EpiasClient(
                username=self._settings.env.epias_username,
                password=self._settings.env.epias_password,
                config=self._settings.epias_api,
            ) as client:
                return client.fetch_generation(start_date, end_date)
        except EpiasAuthError:
            raise  # auth failures are critical — do not swallow
        except Exception as e:
            msg = f"Generation fetch failed, predictions will lack supply features: {e}"
            logger.warning(msg)
            self._warnings.append(msg)
            return pd.DataFrame(index=df.index)

    def _fetch_weather_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fetch weather data: historical for past, forecast for future."""
        with OpenMeteoClient(
            config=self._settings.openmeteo,
            region=self._settings.region,
            timezone=self._settings.project.timezone,
        ) as client:
            # Get date range
            start_date = df.index.min().strftime("%Y-%m-%d")
            end_date = df.index.max().strftime("%Y-%m-%d")

            try:
                # Try historical first (for training data portion)
                historical_df = client.fetch_historical(start_date, end_date)
            except Exception as e:
                logger.warning("Historical weather fetch failed: {}", e)
                historical_df = pd.DataFrame(index=df.index)

            try:
                # Get forecast for future portion
                forecast_df = client.fetch_forecast(forecast_days=3)
            except Exception as e:
                logger.warning("Weather forecast fetch failed: {}", e)
                forecast_df = pd.DataFrame(index=df.index)

            # Combine: historical for past, forecast for future
            if not historical_df.empty and not forecast_df.empty:
                combined = pd.concat([historical_df, forecast_df])
                combined = combined[~combined.index.duplicated(keep="last")]
                return combined.sort_index()
            elif not historical_df.empty:
                return historical_df
            elif not forecast_df.empty:
                return forecast_df
            else:
                return pd.DataFrame(index=df.index)

    def _prepare_output(
        self,
        predictions: pd.DataFrame,
        last_data_point: pd.Timestamp,
    ) -> pd.DataFrame:
        """Prepare final output DataFrame for customer delivery.

        Customer provides data up to T-1 23:00 and needs T+1 day forecast (GOP).
        Output contains only T+1 (24 rows) with datetime and prediction columns.
        """
        result = predictions[["consumption_mwh"]].copy()

        # Filter to T+1 only (day_ahead / GOP)
        # last_data_point = T-1 23:00, so T+1 starts 2 days later at 00:00
        t_plus_1_start = (last_data_point + pd.Timedelta(days=2)).normalize()
        result = result.loc[result.index >= t_plus_1_start]

        return result

    def get_model_info(self) -> dict[str, Any]:
        """Get information about loaded models."""
        if not self.is_ready or self._ensemble is None:
            return {"loaded": False}

        return {
            "loaded": True,
            "active_models": self._ensemble.active_models,
            "weights": self._ensemble.weights,
            "forecast_horizon": self._config.forecast_horizon,
        }

    def get_feature_importance_top(self, n: int = 15) -> list[dict[str, Any]] | None:
        """Get top-N CatBoost feature importance for analytics storage.

        Returns:
            List of {"feature": name, "importance": value} dicts, or None if unavailable.
        """
        if (
            self._ensemble is None
            or self._ensemble._catboost_model is None
        ):
            return None
        try:
            model = self._ensemble._catboost_model
            importances = model.get_feature_importance()
            feature_names = model.feature_names_
            pairs = sorted(
                zip(feature_names, importances, strict=True),
                key=lambda x: x[1],
                reverse=True,
            )[:n]
            return [{"feature": name, "importance": round(float(imp), 4)} for name, imp in pairs]
        except Exception:
            return None

    # ------------------------------------------------------------------
    # L3 Data Lineage helpers
    # ------------------------------------------------------------------

    def get_lineage_metadata(self) -> dict[str, Any]:
        """Return current config/model snapshot for DB storage."""
        weights: dict[str, float] = {}
        if self._ensemble:
            weights = self._ensemble.weights
        return {
            "config_snapshot": {
                "ensemble_method": "stacking",
                "ensemble_weights": weights,
                "feature_count": 153,
            },
            "model_versions": {
                "catboost": str(self._config.catboost_path),
                "prophet": str(self._config.prophet_path),
                "tft": str(self._config.tft_path),
            },
        }

    @staticmethod
    def compute_excel_hash(excel_path: Path) -> str:
        """Compute SHA256 hash of input Excel file."""
        sha = hashlib.sha256()
        with open(excel_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha.update(chunk)
        return sha.hexdigest()

    @staticmethod
    def archive_features(
        job_id: str,
        features_df: pd.DataFrame,
        forecast_mask: pd.Series,
    ) -> tuple[Path | None, Path | None]:
        """Save feature datasets to archive directory (non-fatal)."""
        try:
            archive_dir = Path("data/archive/jobs") / job_id
            archive_dir.mkdir(parents=True, exist_ok=True)

            hist_path = archive_dir / "features_historical.parquet"
            forecast_path = archive_dir / "features_forecast.parquet"

            features_df.loc[~forecast_mask].to_parquet(hist_path)
            features_df.loc[forecast_mask].to_parquet(forecast_path)

            return hist_path, forecast_path
        except Exception as e:
            logger.warning("Feature archival failed: {}", e)
            return None, None

    @staticmethod
    def write_metadata_json(
        job_id: str, lineage_data: dict[str, Any]
    ) -> Path | None:
        """Write job metadata JSON to archive directory."""
        try:
            archive_dir = Path("data/archive/jobs") / job_id
            archive_dir.mkdir(parents=True, exist_ok=True)

            metadata = {
                "job_id": job_id,
                "created_at": datetime.now(tz=TZ_ISTANBUL).isoformat(),
                "model_versions": lineage_data.get("model_versions", {}),
                "config_snapshot": lineage_data.get("config_snapshot", {}),
            }
            path = archive_dir / "metadata.json"
            path.write_text(
                json.dumps(metadata, indent=2, ensure_ascii=False)
            )
            return path
        except Exception as e:
            logger.warning("Metadata JSON write failed: {}", e)
            return None
