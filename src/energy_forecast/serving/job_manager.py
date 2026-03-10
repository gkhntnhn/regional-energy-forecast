"""Job queue manager with single-worker guarantee.

Supports two modes:
- **Database mode** (DATABASE_URL set): Jobs persist in PostgreSQL via repositories.
- **In-memory mode** (DATABASE_URL empty): Jobs stored in a dict (dev/test only).
"""

from __future__ import annotations

import asyncio
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
from loguru import logger
from pydantic import BaseModel, Field

from energy_forecast.serving.exceptions import JobNotFoundError, JobQueueFullError
from energy_forecast.serving.schemas import JobStatus
from energy_forecast.utils import TZ_ISTANBUL

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    from energy_forecast.serving.services.email_service import EmailService
    from energy_forecast.serving.services.file_service import FileService
    from energy_forecast.serving.services.prediction_service import PredictionService


class Job(BaseModel):
    """In-memory job representation (used when DATABASE_URL is empty)."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    email: str
    excel_path: Path
    file_stem: str
    status: JobStatus = JobStatus.PENDING
    progress: str | None = None
    error: str | None = None
    result_path: Path | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(tz=TZ_ISTANBUL))
    completed_at: datetime | None = None

    model_config = {"arbitrary_types_allowed": True}


class JobManager:
    """Manages prediction jobs with single-worker queue.

    Uses asyncio.Lock to ensure only one prediction runs at a time.
    Operates in two modes depending on whether a session_factory is provided.
    """

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = asyncio.Lock()
        self._active_job_id: str | None = None

    # ------------------------------------------------------------------
    # In-memory helpers (dev mode fallback)
    # ------------------------------------------------------------------

    def has_active_job_in_memory(self) -> bool:
        """Check if a job is currently running (in-memory mode)."""
        return self._active_job_id is not None

    def get_active_job_in_memory(self) -> Job | None:
        """Get the currently running job (in-memory mode)."""
        if self._active_job_id:
            return self._jobs.get(self._active_job_id)
        return None

    def create_job_in_memory(
        self, email: str, excel_path: Path, file_stem: str
    ) -> Job:
        """Create a new pending job in memory."""
        if self.has_active_job_in_memory():
            active = self.get_active_job_in_memory()
            raise JobQueueFullError(
                f"A job is already running "
                f"(ID: {active.id if active else 'unknown'}). "
                "Please wait for it to complete."
            )
        job = Job(email=email, excel_path=excel_path, file_stem=file_stem)
        self._jobs[job.id] = job
        logger.info("Created job: {} for {}", job.id, email)
        return job

    def get_job_in_memory(self, job_id: str) -> Job:
        """Get job by ID (in-memory mode)."""
        job = self._jobs.get(job_id)
        if job is None:
            raise JobNotFoundError(f"Job not found: {job_id}")
        return job

    def get_all_jobs_in_memory(self) -> list[Job]:
        """Get all jobs (in-memory mode)."""
        return list(self._jobs.values())

    def get_stats_in_memory(self) -> dict[str, int]:
        """Get job statistics (in-memory mode)."""
        stats: dict[str, int] = {
            "total": len(self._jobs),
            "pending": 0,
            "running": 0,
            "completed": 0,
            "failed": 0,
        }
        for job in self._jobs.values():
            stats[job.status.value] += 1
        return stats

    # ------------------------------------------------------------------
    # DB helpers
    # ------------------------------------------------------------------

    async def has_active_job_db(self, session: AsyncSession) -> bool:
        """Check if a job is currently running (DB mode)."""
        from energy_forecast.db.repositories.job_repo import JobRepository

        repo = JobRepository(session)
        active = await repo.get_active_job()
        return active is not None

    async def get_active_job_db(
        self, session: AsyncSession
    ) -> Any:
        """Get the currently active job from DB."""
        from energy_forecast.db.repositories.job_repo import JobRepository

        repo = JobRepository(session)
        return await repo.get_active_job()

    async def create_job_db(
        self,
        session: AsyncSession,
        email: str,
        excel_path: Path,
        file_stem: str,
    ) -> Any:
        """Create a new pending job in DB."""
        from energy_forecast.db.repositories.job_repo import JobRepository

        repo = JobRepository(session)
        active = await repo.get_active_job()
        if active is not None:
            raise JobQueueFullError(
                f"A job is already running (ID: {active.id}). "
                "Please wait for it to complete."
            )

        job_id = uuid.uuid4().hex[:12]
        job_data = {
            "id": job_id,
            "email": email,
            "excel_path": str(excel_path),
            "file_stem": file_stem,
            "status": "pending",
            "email_status": "pending",
        }
        job = await repo.create(job_data)
        await session.commit()
        logger.info("Created job: {} for {}", job_id, email)
        return job

    async def get_job_db(
        self, session: AsyncSession, job_id: str
    ) -> Any:
        """Get job by ID from DB."""
        from energy_forecast.db.repositories.job_repo import JobRepository

        repo = JobRepository(session)
        job = await repo.get_by_id(job_id)
        if job is None:
            raise JobNotFoundError(f"Job not found: {job_id}")
        return job

    # ------------------------------------------------------------------
    # Process job (DB mode) — pipeline steps
    # ------------------------------------------------------------------

    async def _match_previous_predictions_step(
        self,
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
                consumption_df = (
                    prediction_service._data_loader.load_excel(
                        Path(excel_path)
                    )
                )
                if not consumption_df.empty:
                    async with session_factory() as session:
                        pred_repo = PredictionRepository(session)
                        matched = (
                            await pred_repo
                            .match_predictions_with_actuals(
                                consumption_df
                            )
                        )
                        await session.commit()
                    if matched > 0:
                        logger.info(
                            "Matched {} predictions with actuals",
                            matched,
                        )
                        await _run_drift_check(
                            session_factory,
                            email_service,
                        )
        except Exception as e:
            logger.warning(
                "Prediction matching failed (non-fatal): {}", e
            )

    async def _run_prediction_step(
        self,
        excel_path: str,
        prediction_service: PredictionService,
    ) -> pd.DataFrame:
        """Run model prediction pipeline. Raises on failure."""
        return prediction_service.run_prediction(
            excel_path=Path(excel_path),
            progress_callback=lambda msg: None,
        )

    async def _store_predictions_step(
        self,
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
                    raw_dt = row.name if hasattr(row, "name") else row.get(
                        "datetime"
                    )
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
                    ensemble_dts = {
                        pd.Timestamp(r["forecast_dt"])
                        for r in pred_rows
                    }
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
                await job_repo.update_progress(
                    job_id, "Tahmin sonuclari kaydedildi"
                )
                await session.commit()
        except Exception as e:
            logger.warning("DB snapshot failed (non-fatal): {}", e)

    async def _store_weather_step(
        self,
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
                    count, job_id,
                )
        except Exception as e:
            logger.warning(
                "Weather snapshot failed (non-fatal): {}", e
            )

    async def _store_metadata_step(
        self,
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
                        job_id, meta_update,
                    )
                    await session.commit()
        except Exception as e:
            logger.warning(
                "Metadata snapshot failed (non-fatal): {}", e
            )

    async def _create_output_step(
        self,
        predictions: pd.DataFrame,
        file_stem: str,
        file_service: FileService,
    ) -> Path:
        """Create output Excel file. Raises on failure."""
        return file_service.create_output_xlsx(predictions, file_stem)

    async def _send_email_step(
        self,
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
            Email failure does NOT propagate — the job completes regardless.
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
                    await repo.update_email_status(
                        job_id, "sent", attempts=attempts
                    )
                    await repo.update_progress(job_id, "Sonuclar gonderildi")
                else:
                    await repo.update_email_status(
                        job_id, "failed", attempts=attempts
                    )
                    await repo.update_progress(
                        job_id, f"E-posta gonderilemedi: {error_msg}"
                    )
                await session.commit()

            if not success:
                logger.warning(
                    "Email delivery failed for job {} (non-fatal): {}",
                    job_id, error_msg,
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
            except Exception:
                pass  # DB update failure is also non-fatal
            return False

    async def _archive_step(
        self,
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

            hist_path, fc_path = (
                prediction_service.archive_features(
                    job_id, features_df, forecast_mask,
                )
            )
            meta_path = (
                prediction_service.write_metadata_json(
                    job_id,
                    {
                        "model_versions": (
                            prediction_service.get_model_info()
                        ),
                        "config_snapshot": (
                            predictions.attrs.get(
                                "epias_snapshot", {}
                            )
                        ),
                    },
                )
            )

            # Upload to GDrive if configured
            creds = os.environ.get("GDRIVE_CREDENTIALS_PATH")
            folder_id = os.environ.get(
                "GDRIVE_BACKUP_FOLDER_ID"
            )
            if creds and folder_id:
                from energy_forecast.storage.gdrive import (
                    GoogleDriveStorage,
                )

                files: dict[str, Path] = {}
                if hist_path:
                    files["features_historical.parquet"] = (
                        hist_path
                    )
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
                        path_meta["historical_path"] = str(
                            hist_path
                        )
                    if fc_path:
                        path_meta["forecast_path"] = str(
                            fc_path
                        )
                    if uploaded:
                        path_meta["archive_path"] = str(
                            uploaded
                        )
                    await job_repo.update_metadata(
                        job_id, path_meta
                    )
                    await session.commit()

                logger.info(
                    "Archived {} files to GDrive for job {}",
                    len(uploaded),
                    job_id,
                )
            else:
                logger.debug(
                    "GDrive not configured — skipping artifact upload"
                )
        except Exception as e:
            logger.warning(
                "Artifact archival failed (non-fatal): {}", e
            )

    # ------------------------------------------------------------------
    # Process job (DB mode) — orchestrator
    # ------------------------------------------------------------------

    async def process_job_db(
        self,
        job_id: str,
        excel_path: str,
        email: str,
        file_stem: str,
        created_at: datetime,
        session_factory: async_sessionmaker[AsyncSession],
        prediction_service: PredictionService,
        file_service: FileService,
        email_service: EmailService,
    ) -> None:
        """Process a job using DB for persistence.

        Each checkpoint uses a separate session to avoid holding connections.
        Steps are delegated to private methods for clarity.
        """
        from energy_forecast.db.repositories.job_repo import JobRepository

        async with self._lock:
            # Mark running
            async with session_factory() as session:
                repo = JobRepository(session)
                await repo.update_status(job_id, "running")
                await session.commit()

            try:
                # Progress: data analysis
                async with session_factory() as session:
                    repo = JobRepository(session)
                    await repo.update_progress(
                        job_id, "Veri analizi yapiliyor..."
                    )
                    await session.commit()

                # Step 1: Match previous predictions with actuals
                logger.info("[Job {}] Step 1/8: Matching previous predictions", job_id)
                await self._match_previous_predictions_step(
                    job_id, excel_path, session_factory,
                    prediction_service, email_service,
                )

                # Step 2: Run prediction pipeline
                logger.info("[Job {}] Step 2/8: Running prediction pipeline", job_id)
                predictions = await self._run_prediction_step(
                    excel_path, prediction_service,
                )

                # Step 3: Store predictions in DB
                logger.info("[Job {}] Step 3/8: Storing predictions in DB", job_id)
                await self._store_predictions_step(
                    job_id, predictions, session_factory,
                )

                # Step 4: Store weather snapshot
                logger.info("[Job {}] Step 4/8: Storing weather snapshot", job_id)
                await self._store_weather_step(
                    job_id, predictions, session_factory,
                )

                # Step 5: Store EPIAS + feature importance metadata
                logger.info("[Job {}] Step 5/8: Storing metadata", job_id)
                await self._store_metadata_step(
                    job_id, predictions, prediction_service,
                    session_factory,
                )

                # Step 6: Create output file
                logger.info("[Job {}] Step 6/8: Creating output file", job_id)
                async with session_factory() as session:
                    repo = JobRepository(session)
                    await repo.update_progress(
                        job_id, "Rapor dosyasi olusturuluyor..."
                    )
                    await session.commit()

                output_path = await self._create_output_step(
                    predictions, file_stem, file_service,
                )

                # Step 7: Send email (NON-FATAL — prediction succeeded)
                logger.info("[Job {}] Step 7/8: Sending email", job_id)
                async with session_factory() as session:
                    repo = JobRepository(session)
                    await repo.update_progress(
                        job_id, "E-posta gonderiliyor..."
                    )
                    await session.commit()

                await self._send_email_step(
                    job_id, email, output_path, created_at,
                    session_factory, email_service,
                )

                # Step 8: Archive features + GDrive upload
                logger.info("[Job {}] Step 8/8: Archiving artifacts", job_id)
                await self._archive_step(
                    job_id, file_stem, output_path, created_at,
                    predictions, prediction_service, session_factory,
                )

                # Mark complete
                async with session_factory() as session:
                    repo = JobRepository(session)
                    await repo.update_status(
                        job_id, "completed", result_path=str(output_path)
                    )
                    await session.commit()

                logger.info("[Job {}] Completed successfully", job_id)

                # Audit: job_complete (non-fatal)
                try:
                    from energy_forecast.db.repositories.audit_repo import (
                        AuditRepository,
                    )

                    async with session_factory() as session:
                        audit = AuditRepository(session)
                        await audit.log(
                            action="job_complete",
                            user_email=email,
                            details={"job_id": job_id},
                        )
                        await session.commit()
                except Exception:
                    pass  # audit failure is silent

            except Exception as e:
                error_msg = str(e)
                logger.opt(exception=True).error("Job {} failed: {}", job_id, error_msg)
                async with session_factory() as session:
                    repo = JobRepository(session)
                    await repo.update_status(
                        job_id, "failed", error=error_msg
                    )
                    await session.commit()

                # Audit: job_failed (non-fatal)
                try:
                    from energy_forecast.db.repositories.audit_repo import (
                        AuditRepository,
                    )

                    async with session_factory() as session:
                        audit = AuditRepository(session)
                        await audit.log(
                            action="job_failed",
                            user_email=email,
                            details={
                                "job_id": job_id,
                                "error": error_msg[:500],
                            },
                        )
                        await session.commit()
                except Exception:
                    pass  # audit failure is silent

                try:
                    email_service.send_error_notification(
                        to_email=email,
                        job_id=job_id,
                        error_message=error_msg,
                    )
                except Exception as email_err:
                    logger.error(
                        "Failed to send error notification: {}", email_err
                    )

    # ------------------------------------------------------------------
    # Process job (in-memory mode)
    # ------------------------------------------------------------------

    async def process_job_in_memory(
        self,
        job: Job,
        prediction_service: PredictionService,
        file_service: FileService,
        email_service: EmailService,
    ) -> None:
        """Process a job with in-memory storage (original behavior)."""
        async with self._lock:
            job.status = JobStatus.RUNNING
            self._active_job_id = job.id
            logger.info("Job {} started", job.id)

            try:
                self._jobs[job.id].progress = "Veri analizi yapiliyor..."
                predictions = prediction_service.run_prediction(
                    excel_path=job.excel_path,
                    progress_callback=lambda msg: setattr(
                        self._jobs.get(job.id, job), "progress", msg
                    ),
                )

                self._jobs[job.id].progress = "Rapor dosyasi olusturuluyor..."
                output_path = file_service.create_output_xlsx(
                    predictions, job.file_stem
                )

                self._jobs[job.id].progress = "E-posta gonderiliyor..."
                try:
                    email_service.send_with_retry(
                        to_email=job.email,
                        attachment_path=output_path,
                        job_id=job.id,
                        created_at=job.created_at.strftime("%Y-%m-%d %H:%M:%S"),
                    )
                except Exception as email_err:
                    logger.opt(exception=True).warning(
                        "Email failed for job {} (non-fatal): {}",
                        job.id, email_err,
                    )

                job.status = JobStatus.COMPLETED
                job.result_path = output_path
                job.completed_at = datetime.now(tz=TZ_ISTANBUL)
                self._active_job_id = None
                logger.info("Job {} completed", job.id)

            except Exception as e:
                error_msg = str(e)
                job.status = JobStatus.FAILED
                job.error = error_msg
                job.completed_at = datetime.now(tz=TZ_ISTANBUL)
                self._active_job_id = None
                logger.error("Job {} failed: {}", job.id, error_msg)

                try:
                    email_service.send_error_notification(
                        to_email=job.email,
                        job_id=job.id,
                        error_message=error_msg,
                    )
                except Exception as email_err:
                    logger.error(
                        "Failed to send error notification: {}", email_err
                    )

    def cleanup_old_jobs(self, max_age_hours: int = 24) -> int:
        """Remove old completed/failed jobs from memory."""
        from datetime import timedelta

        threshold = datetime.now(tz=TZ_ISTANBUL) - timedelta(
            hours=max_age_hours
        )
        to_remove = []

        for job_id, job in self._jobs.items():
            is_finished = job.status in (
                JobStatus.COMPLETED,
                JobStatus.FAILED,
            )
            is_old = (
                job.completed_at is not None and job.completed_at < threshold
            )
            if is_finished and is_old:
                to_remove.append(job_id)

        for job_id in to_remove:
            del self._jobs[job_id]

        if to_remove:
            logger.info("Cleaned up {} old jobs", len(to_remove))
        return len(to_remove)


async def _run_drift_check(
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
                should_email = (
                    alert.severity == "critical" or cfg.email_on_warning
                )
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
                admin_email = cfg.admin_email or os.environ.get(
                    "SMTP_USERNAME", ""
                )
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
