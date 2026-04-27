"""Prediction orchestration service."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

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

if TYPE_CHECKING:
    from sqlalchemy.orm import Session, sessionmaker


class PredictionServiceConfig(BaseModel, frozen=True):
    """Prediction service configuration."""

    models_dir: Path = Field(default=Path("models"))
    catboost_path: Path = Field(default=Path("models/catboost/model.cbm"))
    tft_path: Path = Field(default=Path("models/tft"))
    tsmixerx_path: Path = Field(default=Path("models/tsmixerx"))
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
        sync_session_factory: sessionmaker[Session] | None = None,
    ) -> None:
        self._config = config
        self._settings = settings
        self._sync_session_factory = sync_session_factory
        self._ensemble: EnsembleForecaster | None = None
        self._feature_pipeline: FeaturePipeline | None = None
        self._data_loader: DataLoader | None = None
        self._models_loaded = False
        self._warnings: list[str] = []
        self._last_feature_count: int = 0

    def _get_sync_session(self) -> Session | None:
        """Get a sync session for data access (if configured)."""
        if self._sync_session_factory is None:
            return None
        return self._sync_session_factory()

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
                    "tft": self._settings.ensemble.weights.tft,
                    "tsmixerx": self._settings.ensemble.weights.tsmixerx,
                },
                "target_col": "consumption",
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
                tft_path=self._config.tft_path if self._config.tft_path.exists() else None,
                tsmixerx_path=self._config.tsmixerx_path
                if self._config.tsmixerx_path.exists()
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
        """Orchestrate the full prediction pipeline through phase helpers.

        Phases (each owned by a private helper):
            1. Load + extend (Excel -> DF + 48 forecast rows)
            2. Fetch market data (EPIAS + generation + meta)
            3. Fetch + apply weather (with selective ffill)
            4. Feature pipeline (DB holidays optional)
            5. Forecast/historical split
            6. Ensemble predict
            7. Build response (output formatting + metadata attrs)

        Args:
            excel_path: Path to uploaded Excel file.
            progress_callback: Optional callback for progress updates.

        Returns:
            DataFrame with 24-hour T+1 predictions and ``.attrs`` metadata.

        Raises:
            ModelNotLoadedError: If models not loaded.
            PredictionError: If any phase fails (wraps unexpected errors).
        """
        if not self.is_ready:
            raise ModelNotLoadedError("Models not loaded. Call load_models() first.")
        if not self._data_loader or not self._feature_pipeline or not self._ensemble:
            raise ModelNotLoadedError("Models not loaded. Call load_models() first.")

        self._warnings = []
        start_time = time.perf_counter()

        def update_progress(msg: str) -> None:
            logger.info(msg)
            if progress_callback:
                progress_callback(msg)

        try:
            update_progress("Loading consumption data from Excel...")
            _, last_timestamp, extended_df = self._load_consumption_and_extend(excel_path)

            update_progress("Fetching EPIAS market data...")
            merged_df, epias_meta = self._fetch_market_data(extended_df)

            update_progress("Fetching weather data...")
            merged_df, weather_df = self._fetch_and_apply_weather(merged_df, extended_df)

            update_progress("Running feature engineering pipeline...")
            features_df = self._run_feature_pipeline_with_holidays(merged_df)

            update_progress("Generating ensemble predictions...")
            forecast_features, historical_features, forecast_mask = (
                self._extract_forecast_split(features_df, last_timestamp)
            )
            predictions, raw_predictions = self._generate_predictions(
                forecast_features, historical_features
            )

            latency_ms = (time.perf_counter() - start_time) * 1000
            result = self._build_response(
                predictions, raw_predictions, last_timestamp, latency_ms,
                weather_df, epias_meta, features_df, forecast_mask,
            )
            update_progress("Prediction complete!")
            return result

        except (ModelNotLoadedError, PredictionError, FeaturePipelineError):
            raise
        except Exception as e:
            logger.error("Prediction failed: {}", e)
            raise PredictionError(f"Prediction failed: {e}") from e

    # ------------------------------------------------------------------
    # Phase helpers (called only by run_prediction; each is independently
    # unit-testable with focused mocks)
    # ------------------------------------------------------------------

    def _load_consumption_and_extend(
        self, excel_path: Path,
    ) -> tuple[pd.DataFrame, pd.Timestamp, pd.DataFrame]:
        """Load Excel and extend the dataframe with 48 empty forecast rows."""
        assert self._data_loader is not None
        consumption_df = self._data_loader.load_excel(excel_path)
        last_timestamp = consumption_df.index.max()
        logger.info("Last data point: {}", last_timestamp)
        extended_df = self._data_loader.extend_for_forecast(
            consumption_df,
            horizon_hours=self._config.forecast_horizon,
        )
        return consumption_df, last_timestamp, extended_df

    def _fetch_market_data(
        self, extended_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, dict[str, object]]:
        """Fetch EPIAS market + generation data and join onto the extended frame.

        Returns:
            ``(merged_df, epias_meta)`` — merged frame includes EPIAS and
            (when available) generation columns; meta is a structured
            snapshot intended for the API response payload.
        """
        epias_df = self._fetch_epias_data(extended_df)
        merged_df = extended_df.join(epias_df, how="left")

        generation_df = self._fetch_generation_data(extended_df)
        if not generation_df.empty:
            merged_df = merged_df.join(generation_df, how="left")

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
                col: int(epias_df[col].isna().sum()) for col in epias_df.columns
            }
        return merged_df, epias_meta

    def _fetch_and_apply_weather(
        self,
        merged_df: pd.DataFrame,
        extended_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Fetch weather, join onto merged frame, ffill weather columns only."""
        from energy_forecast.utils import WEATHER_FILL_PREFIXES

        weather_df = self._fetch_weather_data(extended_df)
        merged_df = merged_df.join(weather_df, how="left")

        weather_cols = [c for c in merged_df.columns if c.startswith(WEATHER_FILL_PREFIXES)]
        if weather_cols:
            merged_df[weather_cols] = merged_df[weather_cols].ffill()
        cat_weather_cols = [
            c for c in merged_df.columns if c.startswith(("weather_code", "weather_group"))
        ]
        if cat_weather_cols:
            merged_df[cat_weather_cols] = merged_df[cat_weather_cols].ffill()
        return merged_df, weather_df

    def _run_feature_pipeline_with_holidays(
        self, merged_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Run feature pipeline; rebuild with DB holidays if sync session set."""
        assert self._feature_pipeline is not None
        try:
            if self._sync_session_factory is not None:
                holidays_df = self._load_holidays_from_db()
                self._feature_pipeline = FeaturePipeline(
                    self._settings,
                    holidays_df=holidays_df,
                )
            features_df = self._feature_pipeline.run(merged_df)
            self._last_feature_count = len(features_df.columns)
            return features_df
        except Exception as e:
            raise FeaturePipelineError(f"Feature pipeline failed: {e}") from e

    def _extract_forecast_split(
        self,
        features_df: pd.DataFrame,
        last_timestamp: pd.Timestamp,
    ) -> tuple[pd.DataFrame, pd.DataFrame, Any]:
        """Split features into forecast vs historical rows by ``last_timestamp``.

        Raises:
            PredictionError: If no forecast rows result from the split.
        """
        forecast_mask = features_df.index > last_timestamp
        forecast_features = features_df.loc[forecast_mask].copy()
        if len(forecast_features) == 0:
            raise PredictionError("No forecast rows available after feature pipeline")
        if len(forecast_features) != self._config.forecast_horizon:
            logger.warning(
                "Expected {} forecast rows, got {}",
                self._config.forecast_horizon,
                len(forecast_features),
            )
        historical_features = features_df.loc[~forecast_mask].copy()
        return forecast_features, historical_features, forecast_mask

    def _generate_predictions(
        self,
        forecast_features: pd.DataFrame,
        historical_features: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Run ensemble prediction; return (display, raw) prediction frames."""
        assert self._ensemble is not None
        predictions = self._ensemble.predict(
            forecast_features,
            history=historical_features,
        )
        raw_predictions = predictions.copy()
        return predictions, raw_predictions

    def _build_response(
        self,
        predictions: pd.DataFrame,
        raw_predictions: pd.DataFrame,
        last_timestamp: pd.Timestamp,
        latency_ms: float,
        weather_df: pd.DataFrame,
        epias_meta: dict[str, object],
        features_df: pd.DataFrame,
        forecast_mask: Any,
    ) -> pd.DataFrame:
        """Format the output frame and attach metadata for API/DB consumers."""
        result = self._prepare_output(predictions, last_timestamp)
        logger.info(
            "Prediction completed in {:.0f}ms - {} rows from {} to {}",
            latency_ms, len(result), result.index.min(), result.index.max(),
        )
        result.attrs["latency_ms"] = round(latency_ms)
        result.attrs["weather_data"] = weather_df
        result.attrs["epias_snapshot"] = epias_meta
        result.attrs["features_df"] = features_df
        result.attrs["forecast_mask"] = forecast_mask
        result.attrs["raw_predictions"] = raw_predictions
        return result

    def _fetch_with_epias_client(
        self, df: pd.DataFrame, fetch_method: str, warn_msg: str
    ) -> pd.DataFrame:
        """Fetch data via EpiasClient with shared error handling.

        Args:
            df: DataFrame with DatetimeIndex for date range.
            fetch_method: Client method name ("fetch" or "fetch_generation").
            warn_msg: Warning prefix if fetch fails.

        Raises:
            EpiasAuthError: If authentication fails (critical, not recoverable).
        """
        start_date = df.index.min().strftime("%Y-%m-%d")
        end_date = df.index.max().strftime("%Y-%m-%d")

        session = self._get_sync_session()
        try:
            with EpiasClient(
                username=self._settings.env.epias_username,
                password=self._settings.env.epias_password,
                config=self._settings.epias_api,
                db_session=session,
            ) as client:
                result: pd.DataFrame = getattr(client, fetch_method)(start_date, end_date)
                return result
        except EpiasAuthError:
            raise
        except Exception as e:
            msg = f"{warn_msg}: {e}"
            logger.warning(msg)
            self._warnings.append(msg)
            return pd.DataFrame(index=df.index)
        finally:
            if session is not None:
                session.close()

    def _fetch_epias_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fetch EPIAS market data for the date range in df."""
        return self._fetch_with_epias_client(
            df, "fetch", "EPIAS fetch failed, predictions will lack market features"
        )

    def _fetch_generation_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fetch EPIAS generation data for the date range in df."""
        return self._fetch_with_epias_client(
            df, "fetch_generation", "Generation fetch failed, predictions will lack supply features"
        )

    def _fetch_weather_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fetch weather data: historical for past, forecast for future."""
        session = self._get_sync_session()
        try:
            with OpenMeteoClient(
                config=self._settings.openmeteo,
                region=self._settings.region,
                timezone=self._settings.project.timezone,
                db_session=session,
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
        finally:
            if session is not None:
                session.close()

    def _load_holidays_from_db(self) -> pd.DataFrame | None:
        """Load holidays from DB via sync session (if available)."""
        session = self._get_sync_session()
        if session is None:
            return None
        try:
            from energy_forecast.db.sync_repos import SyncDataAccess

            df = SyncDataAccess(session).get_holidays()
            return df if not df.empty else None
        except Exception as e:
            logger.warning("Holiday DB load failed: {}", e)
            return None
        finally:
            session.close()

    def _prepare_output(
        self,
        predictions: pd.DataFrame,
        last_data_point: pd.Timestamp,
    ) -> pd.DataFrame:
        """Prepare final output DataFrame with all 48 hours and period labels.

        Returns all 48 forecast rows (T + T+1) with a 'period' column:
        - T day (hours 0-23): 'intraday' (GIP)
        - T+1 day (hours 0-23): 'day_ahead' (GOP)

        Frontend toggles which rows to display.
        """
        result = predictions[["consumption_mwh"]].copy()

        # T+1 starts 2 days after last_data_point (T-1 23:00)
        t_plus_1_start = (last_data_point + pd.Timedelta(days=2)).normalize()
        result["period"] = "intraday"
        result.loc[result.index >= t_plus_1_start, "period"] = "day_ahead"

        return result

    def get_model_info(self) -> dict[str, Any]:
        """Get information about loaded models (including multi-seed status)."""
        if not self.is_ready or self._ensemble is None:
            return {"loaded": False}

        info: dict[str, Any] = {
            "loaded": True,
            "active_models": self._ensemble.active_models,
            "weights": self._ensemble.weights,
            "forecast_horizon": self._config.forecast_horizon,
        }

        # Multi-seed TSMixerx observability (R12). Only aggregate counts are
        # exposed on /models (which is an auth'd but non-admin endpoint);
        # explicit seed names live in admin-only paths. This reduces recon
        # surface if the API key is compromised (audit P1-2).
        tsmixerx = self._ensemble._tsmixerx_model
        if tsmixerx is not None and hasattr(tsmixerx, "get_seed_info"):
            seed_info = tsmixerx.get_seed_info()
            info["tsmixerx_ensemble"] = {
                "ensemble_type": seed_info["ensemble_type"],
                "n_requested": seed_info["n_requested"],
                "n_loaded": seed_info["n_loaded"],
                "is_degraded": seed_info["is_degraded"],
                "last_request_n_succeeded": seed_info.get("last_request_n_succeeded"),
            }

        return info

    def get_feature_importance_top(self, n: int = 15) -> list[dict[str, Any]] | None:
        """Get top-N CatBoost feature importance for analytics storage.

        Returns:
            List of {"feature": name, "importance": value} dicts, or None if unavailable.
        """
        if self._ensemble is None or self._ensemble.catboost_model is None:
            return None
        try:
            model = self._ensemble.catboost_model
            importances = model.get_feature_importance()
            feature_names = model.feature_names_
            pairs = sorted(
                zip(feature_names, importances, strict=True),
                key=lambda x: x[1],
                reverse=True,
            )[:n]
            return [{"feature": name, "importance": round(float(imp), 4)} for name, imp in pairs]
        except Exception as e:
            logger.debug("Feature importance extraction failed: {}", e)
            return None

    # ------------------------------------------------------------------
    # L3 Data Lineage helpers
    # ------------------------------------------------------------------

    def get_lineage_metadata(self) -> dict[str, Any]:
        """Return current config/model snapshot for DB storage."""
        weights: dict[str, float] = {}
        if self._ensemble:
            weights = self._ensemble.weights

        # TSMixerx version record — either flat path or dict for multi-seed
        tsmixerx_version: Any = str(self._config.tsmixerx_path)
        if self._ensemble is not None:
            tsmixerx = self._ensemble._tsmixerx_model
            if tsmixerx is not None and hasattr(tsmixerx, "get_seed_info"):
                tsmixerx_version = {
                    "path": str(self._config.tsmixerx_path),
                    **tsmixerx.get_seed_info(),
                }

        return {
            "config_snapshot": {
                "ensemble_method": "stacking",
                "ensemble_weights": weights,
                "feature_count": self._last_feature_count or 153,
            },
            "model_versions": {
                "catboost": str(self._config.catboost_path),
                "tft": str(self._config.tft_path),
                "tsmixerx": tsmixerx_version,
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
    def write_metadata_json(job_id: str, lineage_data: dict[str, Any]) -> Path | None:
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
            path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
            return path
        except Exception as e:
            logger.warning("Metadata JSON write failed: {}", e)
            return None
