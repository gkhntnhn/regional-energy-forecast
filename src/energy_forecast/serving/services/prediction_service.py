"""Prediction orchestration service."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger
from pydantic import BaseModel, Field

from energy_forecast.config.settings import Settings
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


class PredictionServiceConfig(BaseModel, frozen=True):
    """Prediction service configuration."""

    models_dir: Path = Field(default=Path("models"))
    catboost_path: Path = Field(default=Path("models/catboost/model.cbm"))
    prophet_path: Path = Field(default=Path("models/prophet/model.pkl"))
    tft_path: Path = Field(default=Path("models/tft"))
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

            # Try to load ensemble weights if available
            ensemble_weights_path = self._config.models_dir / "ensemble_weights.json"
            if ensemble_weights_path.exists():
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
            DataFrame with 48-hour predictions.

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

            # Step 6: Extract forecast rows (last 48 hours with NaN consumption)
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

            # Attach latency metadata for API response
            result.attrs["latency_ms"] = round(latency_ms)
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
        """Prepare final output DataFrame with period labels."""
        result = predictions[["consumption_mwh"]].copy()

        # Add period labels (intraday = T, day_ahead = T+1)
        tomorrow_start = (last_data_point + pd.Timedelta(days=1)).normalize()

        result["period"] = result.index.map(
            lambda ts: "day_ahead" if ts >= tomorrow_start else "intraday"
        )

        # Add individual model predictions if available
        for col in predictions.columns:
            if col.endswith("_prediction"):
                model_name = col.replace("_prediction", "")
                result[f"{model_name}_mwh"] = predictions[col]

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
