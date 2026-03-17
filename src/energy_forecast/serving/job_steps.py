"""Extracted job pipeline step functions.

Each step is a standalone async function (no class dependency).
Called by JobManager.process_job_db orchestrator.
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
from loguru import logger

from energy_forecast.utils import TZ_ISTANBUL

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    from energy_forecast.serving.services.email_service import EmailService
    from energy_forecast.serving.services.file_service import FileService
    from energy_forecast.serving.services.prediction_service import PredictionService


# ------------------------------------------------------------------
# DB checkpoint helpers (DRY — eliminates repeated session boilerplate)
# ------------------------------------------------------------------


async def update_progress_db(
    session_factory: async_sessionmaker[AsyncSession],
    job_id: str,
    message: str,
) -> None:
    """Update job progress message via a short-lived DB session."""
    from energy_forecast.db.repositories.job_repo import JobRepository

    async with session_factory() as session:
        repo = JobRepository(session)
        await repo.update_progress(job_id, message)
        await session.commit()


async def update_status_db(
    session_factory: async_sessionmaker[AsyncSession],
    job_id: str,
    status: str,
    *,
    result_path: str | None = None,
    error: str | None = None,
) -> None:
    """Update job status via a short-lived DB session."""
    from energy_forecast.db.repositories.job_repo import JobRepository

    async with session_factory() as session:
        repo = JobRepository(session)
        await repo.update_status(
            job_id, status, result_path=result_path, error=error,
        )
        await session.commit()


# ------------------------------------------------------------------
# Pipeline steps
# ------------------------------------------------------------------


async def match_previous_predictions_step(
    job_id: str,
    excel_path: str,
    session_factory: async_sessionmaker[AsyncSession],
    prediction_service: PredictionService,
    email_service: EmailService,
) -> None:
    """Match previous predictions with actuals from new Excel (non-fatal)."""
    from energy_forecast.db.repositories.prediction_repo import (
        PredictionRepository,
    )

    try:
        if prediction_service._data_loader is not None:
            consumption_df = prediction_service._data_loader.load_excel(Path(excel_path))
            if not consumption_df.empty:
                async with session_factory() as session:
                    pred_repo = PredictionRepository(session)
                    matched = await pred_repo.match_predictions_with_actuals(consumption_df)
                    await session.commit()
                if matched > 0:
                    logger.info(
                        "Matched {} predictions with actuals",
                        matched,
                    )
                    await run_drift_check(
                        session_factory,
                        email_service,
                    )
    except Exception as e:
        logger.warning("Prediction matching failed (non-fatal): {}", e)


async def run_prediction_step(
    excel_path: str,
    prediction_service: PredictionService,
) -> pd.DataFrame:
    """Run model prediction pipeline. Raises on failure."""
    return await asyncio.to_thread(
        prediction_service.run_prediction,
        excel_path=Path(excel_path),
        progress_callback=lambda msg: None,
    )


async def store_predictions_step(
    job_id: str,
    predictions: pd.DataFrame,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """Store ensemble + per-model predictions in DB (non-fatal)."""
    from energy_forecast.db.repositories.job_repo import JobRepository
    from energy_forecast.db.repositories.prediction_repo import (
        PredictionRepository,
    )

    try:
        raw_preds = predictions.attrs.get("raw_predictions")
        async with session_factory() as session:
            pred_repo = PredictionRepository(session)
            pred_rows: list[dict[str, Any]] = []
            for _, row in predictions.iterrows():
                raw_dt = row.name if hasattr(row, "name") else row.get("datetime")
                dt = pd.Timestamp(raw_dt)  # type: ignore[arg-type]
                if dt.tzinfo is None:
                    dt = dt.tz_localize(TZ_ISTANBUL)
                mwh = float(row["consumption_mwh"])
                period = str(row.get("period", "day_ahead"))
                pred_rows.append(
                    {
                        "job_id": job_id,
                        "forecast_dt": dt,
                        "consumption_mwh": mwh,
                        "period": period,
                        "model_source": "ensemble",
                    }
                )

            # Per-model predictions for analytics (D1)
            if raw_preds is not None:
                ensemble_dts = {pd.Timestamp(r["forecast_dt"]) for r in pred_rows}
                model_col_map = {
                    "catboost": "catboost_prediction",
                    "prophet": "prophet_prediction",
                    "tft": "tft_prediction",
                }
                for model_name, col_name in model_col_map.items():
                    if col_name not in raw_preds.columns:
                        continue
                    for idx_val, raw_row in raw_preds.iterrows():
                        raw_dt = pd.Timestamp(idx_val)
                        if raw_dt.tzinfo is None:
                            raw_dt = raw_dt.tz_localize(TZ_ISTANBUL)
                        if raw_dt not in ensemble_dts:
                            continue
                        val = raw_row[col_name]
                        if pd.notna(val):
                            pred_rows.append(
                                {
                                    "job_id": job_id,
                                    "forecast_dt": raw_dt,
                                    "consumption_mwh": float(val),
                                    "period": "day_ahead",
                                    "model_source": model_name,
                                }
                            )

            await pred_repo.bulk_create(pred_rows)
            job_repo = JobRepository(session)
            await job_repo.update_progress(job_id, "Tahmin sonuclari kaydedildi")
            await session.commit()
    except Exception as e:
        logger.warning("DB snapshot failed (non-fatal): {}", e)


async def store_weather_step(
    job_id: str,
    predictions: pd.DataFrame,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """Store weather snapshot in DB (non-fatal)."""
    try:
        weather_df = predictions.attrs.get("weather_data")
        if weather_df is not None and not weather_df.empty:
            from energy_forecast.db.repositories.weather_repo import (
                WeatherSnapshotRepository,
            )

            async with session_factory() as session:
                weather_repo = WeatherSnapshotRepository(session)
                count = await weather_repo.bulk_create_forecast(
                    job_id=job_id,
                    weather_df=weather_df,
                    fetched_at=datetime.now(tz=TZ_ISTANBUL),
                )
                await session.commit()
            logger.info(
                "Stored {} weather snapshots for job {}",
                count,
                job_id,
            )
    except Exception as e:
        logger.warning("Weather snapshot failed (non-fatal): {}", e)


async def store_metadata_step(
    job_id: str,
    predictions: pd.DataFrame,
    prediction_service: PredictionService,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """Store EPIAS snapshot + feature importance metadata (non-fatal)."""
    from energy_forecast.db.repositories.job_repo import JobRepository

    try:
        epias_snap = predictions.attrs.get("epias_snapshot")
        fi_top = prediction_service.get_feature_importance_top(15)
        meta_update: dict[str, Any] = {}
        if epias_snap:
            meta_update["epias_snapshot"] = epias_snap
        if fi_top:
            meta_update["feature_importance_top15"] = fi_top
        if meta_update:
            async with session_factory() as session:
                job_repo = JobRepository(session)
                await job_repo.update_metadata(
                    job_id,
                    meta_update,
                )
                await session.commit()
    except Exception as e:
        logger.warning("Metadata snapshot failed (non-fatal): {}", e)


async def create_output_step(
    predictions: pd.DataFrame,
    file_stem: str,
    file_service: FileService,
) -> Path:
    """Create output Excel file. Raises on failure."""
    return await asyncio.to_thread(
        file_service.create_output_xlsx,
        predictions,
        file_stem,
    )


async def send_email_step(
    job_id: str,
    email: str,
    output_path: Path,
    created_at: datetime,
    session_factory: async_sessionmaker[AsyncSession],
    email_service: EmailService,
) -> bool:
    """Send prediction result email and update DB status (non-fatal).

    Returns:
        True if email was sent successfully, False otherwise.
        Email failure does NOT propagate -- the job completes regardless.
    """
    from energy_forecast.db.repositories.job_repo import JobRepository

    try:
        success, attempts, error_msg = email_service.send_with_retry(
            to_email=email,
            attachment_path=output_path,
            job_id=job_id,
            created_at=created_at.strftime("%Y-%m-%d %H:%M:%S"),
        )

        async with session_factory() as session:
            repo = JobRepository(session)
            if success:
                await repo.update_email_status(job_id, "sent", attempts=attempts)
                await repo.update_progress(job_id, "Sonuclar gonderildi")
            else:
                await repo.update_email_status(job_id, "failed", attempts=attempts)
                await repo.update_progress(job_id, f"E-posta gonderilemedi: {error_msg}")
            await session.commit()

        if not success:
            logger.warning(
                "Email delivery failed for job {} (non-fatal): {}",
                job_id,
                error_msg,
            )
        return success

    except Exception as e:
        logger.opt(exception=True).warning(
            "Email step failed for job {} (non-fatal): {}", job_id, e
        )
        try:
            async with session_factory() as session:
                repo = JobRepository(session)
                await repo.update_email_status(job_id, "failed", attempts=0)
                await session.commit()
        except Exception as exc:
            logger.debug("Email status DB update failed (non-fatal): {}", exc)
        return False


async def archive_step(
    job_id: str,
    file_stem: str,
    output_path: Path,
    created_at: datetime,
    predictions: pd.DataFrame,
    prediction_service: PredictionService,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """Archive features + upload to GDrive (non-fatal)."""
    from energy_forecast.db.repositories.job_repo import JobRepository

    try:
        features_df = predictions.attrs.get("features_df")
        forecast_mask = predictions.attrs.get("forecast_mask")
        if features_df is None or forecast_mask is None:
            return

        hist_path, fc_path = prediction_service.archive_features(
            job_id,
            features_df,
            forecast_mask,
        )
        meta_path = prediction_service.write_metadata_json(
            job_id,
            {
                "model_versions": (prediction_service.get_model_info()),
                "config_snapshot": (predictions.attrs.get("epias_snapshot", {})),
            },
        )

        # Upload to GDrive if configured
        creds = os.environ.get("GDRIVE_CREDENTIALS_PATH")
        folder_id = os.environ.get("GDRIVE_BACKUP_FOLDER_ID")
        if creds and folder_id:
            from energy_forecast.storage.gdrive import (
                GoogleDriveStorage,
            )

            files: dict[str, Path] = {}
            if hist_path:
                files["features_historical.parquet"] = hist_path
            if fc_path:
                files["features_forecast.parquet"] = fc_path
            if meta_path:
                files["metadata.json"] = meta_path
            files[f"{file_stem}_forecast.xlsx"] = output_path
            logger.info(
                "Uploading {} artifacts to GDrive...",
                len(files),
            )

            gdrive = GoogleDriveStorage(creds, folder_id)
            uploaded = await asyncio.to_thread(
                gdrive.upload_job_artifacts,
                job_id,
                files,
                created_at,
            )

            # Update DB with GDrive paths
            async with session_factory() as session:
                job_repo = JobRepository(session)
                path_meta: dict[str, str] = {}
                if hist_path:
                    path_meta["historical_path"] = str(hist_path)
                if fc_path:
                    path_meta["forecast_path"] = str(fc_path)
                if uploaded:
                    path_meta["archive_path"] = str(uploaded)
                await job_repo.update_metadata(job_id, path_meta)
                await session.commit()

            logger.info(
                "Archived {} files to GDrive for job {}",
                len(uploaded),
                job_id,
            )
        else:
            logger.debug("GDrive not configured -- skipping artifact upload")
    except Exception as e:
        logger.warning("Artifact archival failed (non-fatal): {}", e)


# ------------------------------------------------------------------
# Drift check (module-level helper)
# ------------------------------------------------------------------


async def run_drift_check(
    session_factory: async_sessionmaker[AsyncSession],
    email_service: EmailService | None,
) -> None:
    """Run drift detection after prediction matching (non-fatal).

    Checks model drift, logs warnings, and sends email alerts
    with cooldown to prevent spam.
    """
    import asyncio
    import os

    from energy_forecast.db.repositories.audit_repo import AuditRepository
    from energy_forecast.monitoring.drift_detector import (
        DriftConfig,
        check_model_drift,
    )

    try:
        # Load config
        monitoring_yaml = Path("configs/monitoring.yaml")
        if monitoring_yaml.exists():
            import yaml

            with open(monitoring_yaml, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            cfg = DriftConfig.from_dict(data.get("drift_detection", {}))
        else:
            cfg = DriftConfig()

        if not cfg.enabled:
            return

        async with session_factory() as session:
            alerts = await check_model_drift(session, config=cfg)

            if not alerts:
                return

            audit_repo = AuditRepository(session)

            for alert in alerts:
                logger.warning("Drift alert: {}", alert.message)

                # Determine if email should be sent
                should_email = alert.severity == "critical" or cfg.email_on_warning
                if not should_email or email_service is None:
                    continue

                # Cooldown check
                action_key = f"drift_alert_{alert.alert_type}"
                last_alert = await audit_repo.get_last_action(action=action_key)
                now = datetime.now(tz=TZ_ISTANBUL)

                if last_alert is not None and last_alert.created_at is not None:
                    elapsed = (now - last_alert.created_at).total_seconds()
                    if elapsed < cfg.cooldown_hours * 3600:
                        logger.info(
                            "Drift alert suppressed (cooldown): {}",
                            alert.alert_type,
                        )
                        continue

                # Send email in thread (sync SMTP)
                admin_email = cfg.admin_email or os.environ.get("SMTP_USERNAME", "")
                if admin_email:
                    sent = await asyncio.to_thread(
                        email_service.send_drift_alert,
                        admin_email,
                        alert,
                    )
                    if sent:
                        await audit_repo.log(
                            action=action_key,
                            details={
                                "severity": alert.severity,
                                "value": alert.current_value,
                                "threshold": alert.threshold,
                            },
                        )

            await session.commit()

    except Exception as e:
        logger.warning("Drift check failed (non-fatal): {}", e)
