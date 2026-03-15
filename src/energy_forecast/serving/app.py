"""FastAPI application with prediction endpoints."""

from __future__ import annotations

import os
import secrets
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any
from urllib.parse import urlparse, urlunparse

from fastapi import Depends, FastAPI, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles
from loguru import logger
from pydantic import EmailStr
from slowapi.errors import RateLimitExceeded
from starlette.middleware.base import BaseHTTPMiddleware

from energy_forecast import __version__
from energy_forecast.config import load_config
from energy_forecast.serving.exceptions import APIError, JobNotFoundError, JobQueueFullError
from energy_forecast.serving.job_manager import JobManager
from energy_forecast.serving.rate_limit import limiter
from energy_forecast.serving.schemas import (
    ErrorResponse,
    HealthResponse,
    JobResponse,
    JobStatus,
    JobStatusResponse,
)
from energy_forecast.serving.services.email_service import EmailService, EmailServiceConfig
from energy_forecast.serving.services.file_service import FileService, FileServiceConfig
from energy_forecast.serving.services.prediction_service import (
    PredictionService,
    PredictionServiceConfig,
)
from energy_forecast.serving.utils import mask_email
from energy_forecast.utils import TZ_ISTANBUL

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------

_bearer_scheme = HTTPBearer(auto_error=False)


async def verify_api_key(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),  # noqa: B008
) -> HTTPAuthorizationCredentials:
    """Validate Bearer token against configured API key.

    Raises:
        HTTPException: 401 if token is missing, empty, or invalid.
    """
    expected_key: str = getattr(request.app.state, "api_key", "")
    if not expected_key:
        raise HTTPException(status_code=401, detail="API key not configured on server")
    if credentials is None or not secrets.compare_digest(credentials.credentials, expected_key):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return credentials


# ---------------------------------------------------------------------------
# Rate Limiter
# ---------------------------------------------------------------------------


def _rate_limit_exceeded_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    """Custom handler for rate limit exceeded."""
    detail = getattr(exc, "detail", str(exc))
    return JSONResponse(
        status_code=429,
        content={"success": False, "error": "Rate limit exceeded", "detail": str(detail)},
    )


# ---------------------------------------------------------------------------
# Lifespan (startup/shutdown)
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan: load models on startup, cleanup on shutdown."""
    from dotenv import load_dotenv

    load_dotenv()  # populate os.environ from .env (needed for GDrive, etc.)

    # Configure file-based logging for persistent traceback capture
    from energy_forecast.utils.logging import setup_logger

    setup_logger(level="DEBUG", log_file="logs/api_server.log")

    logger.info("Starting Energy Forecast API...")

    # Load configuration
    try:
        settings = load_config()
    except FileNotFoundError:
        logger.warning("Config files not found, using defaults")
        from energy_forecast.config import get_default_config

        settings = get_default_config()

    # Initialize file service
    file_config = FileServiceConfig(
        upload_dir=Path("data/uploads"),
        output_dir=Path("data/outputs"),
    )
    app.state.file_service = FileService(file_config)

    # Initialize email service
    email_config = EmailServiceConfig(
        smtp_server=settings.env.smtp_server,
        smtp_port=settings.env.smtp_port,
        username=settings.env.smtp_username,
        password=settings.env.smtp_password,
        sender_email=settings.env.sender_email or settings.env.smtp_username,
        sender_name=settings.api.email.sender_name,
        subject_template=settings.api.email.subject_template,
        body_template=settings.api.email.body_template,
    )
    app.state.email_service = EmailService(email_config)

    # Initialize prediction service — prefer final_models/ (flat), fallback to timestamped
    models_dir = Path(settings.paths.models_dir)
    final_dir = Path("final_models")
    use_final = (final_dir / "catboost" / "model.cbm").exists()

    if use_final:
        catboost_path = final_dir / "catboost" / "model.cbm"
        prophet_path = final_dir / "prophet" / "prophet_model.pkl"
        tft_path = final_dir / "tft"
        ensemble_dir: Path | None = final_dir / "ensemble"
        logger.info("Serving from final_models/ directory")
    else:
        catboost_path = models_dir / "catboost" / "model.cbm"
        prophet_path = models_dir / "prophet" / "prophet_model.pkl"
        tft_path = models_dir / "tft"
        ensemble_dir = models_dir / "ensemble"
        logger.info("Serving from models/ directory")

    pred_config = PredictionServiceConfig(
        models_dir=models_dir,
        catboost_path=catboost_path,
        prophet_path=prophet_path,
        tft_path=tft_path,
        ensemble_dir=ensemble_dir,
    )

    # Create sync session factory for PredictionService (DB data access)
    sync_session_factory = None
    if settings.env.database_url_sync:
        from energy_forecast.db.engine import create_sync_engine, create_sync_session_factory

        sync_engine = create_sync_engine(settings.env.database_url_sync)
        sync_session_factory = create_sync_session_factory(sync_engine)
        logger.info("Sync DB session factory created for prediction service")

    app.state.prediction_service = PredictionService(
        pred_config,
        settings,
        sync_session_factory=sync_session_factory,
    )

    # Try to load models (warn if not available)
    try:
        app.state.prediction_service.load_models()
        logger.info("Models loaded successfully")
    except Exception as e:
        logger.warning("Failed to load models (API will reject predictions): {}", e)

    # Store API key for auth middleware
    app.state.api_key = settings.env.api_key
    if not settings.env.api_key:
        logger.warning("API_KEY is empty — all authenticated endpoints will reject requests")

    # Initialize job manager with queue
    app.state.job_manager = JobManager(max_queue_size=5)
    app.state.job_manager.start_worker()

    # Initialize database (if DATABASE_URL is set)
    if settings.env.database_url:
        from energy_forecast.db import create_db_engine, create_session_factory

        engine = create_db_engine(settings.env.database_url, settings.database)
        session_factory = create_session_factory(engine)
        app.state.db_engine = engine
        app.state.session_factory = session_factory
        app.state.use_db = True
        parsed = urlparse(settings.env.database_url)
        masked_url = urlunparse(
            parsed._replace(
                netloc=f"{parsed.username}:***@{parsed.hostname}:{parsed.port}"
            )
        )
        logger.info("Database connected: {}", masked_url)
    else:
        app.state.db_engine = None
        app.state.session_factory = None
        app.state.use_db = False
        logger.warning("DATABASE_URL not set — using in-memory job storage (dev mode)")

    # Start weather actuals scheduler (DB mode only)
    _scheduler_task = None
    if app.state.use_db:
        import asyncio

        from energy_forecast.jobs.weather_actuals import run_scheduler

        _scheduler_restart_count = 0
        _max_scheduler_restarts = 3

        def _on_scheduler_done(task: asyncio.Task[None]) -> None:
            nonlocal _scheduler_task, _scheduler_restart_count
            if task.cancelled():
                return
            exc = task.exception()
            if exc is not None:
                _scheduler_restart_count += 1
                if _scheduler_restart_count <= _max_scheduler_restarts:
                    logger.error(
                        "Weather scheduler crashed (restart {}/{}): {}",
                        _scheduler_restart_count,
                        _max_scheduler_restarts,
                        exc,
                    )
                    _scheduler_task = asyncio.create_task(
                        run_scheduler(app.state.session_factory, settings)
                    )
                    _scheduler_task.add_done_callback(_on_scheduler_done)
                else:
                    logger.critical(
                        "Weather scheduler crashed {} times — NOT restarting. "
                        "Manual intervention required.",
                        _scheduler_restart_count,
                    )

        _scheduler_task = asyncio.create_task(run_scheduler(app.state.session_factory, settings))
        _scheduler_task.add_done_callback(_on_scheduler_done)
        logger.info("Weather actuals scheduler started (daily at 04:00)")

    logger.info("Energy Forecast API started successfully")

    yield

    # Cleanup on shutdown
    logger.info("Shutting down Energy Forecast API...")
    if _scheduler_task is not None:
        _scheduler_task.cancel()
    await app.state.job_manager.stop_worker()
    app.state.file_service.cleanup_old_files()
    app.state.job_manager.cleanup_old_jobs()
    if app.state.db_engine:
        await app.state.db_engine.dispose()
    logger.info("Cleanup complete")


# ---------------------------------------------------------------------------
# FastAPI Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Energy Forecast API",
    description="Uludag region hourly electricity consumption forecasting",
    version=__version__,
    lifespan=lifespan,
)

# Config loaded at module level for CORS middleware; reloaded in lifespan for model paths
# Load CORS origins from config (fallback to ["*"] if config unavailable)
try:
    _cors_settings = load_config()
    _cors_origins = _cors_settings.api.cors_origins
    _rate_limit = _cors_settings.api.rate_limit
except Exception as e:
    logger.warning("Failed to load API config, falling back to safe defaults: {}", e)
    _cors_origins = ["http://localhost:8000", "http://127.0.0.1:8000"]
    _rate_limit = "10/minute"

# CORS spec: allow_credentials=True is incompatible with allow_origins=["*"]
_allow_credentials = "*" not in _cors_origins
if "*" in _cors_origins:
    logger.warning(
        "CORS allow_origins contains wildcard '*' — credentials disabled. "
        "Set specific origins in configs/api.yaml for production."
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=_allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Security headers middleware
class _SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Any) -> Any:
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
            "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net "
            "https://cdnjs.cloudflare.com https://fonts.googleapis.com; "
            "font-src 'self' https://cdnjs.cloudflare.com https://fonts.gstatic.com; "
            "img-src 'self' data:; "
            "connect-src 'self'"
        )
        if os.environ.get("APP_ENV") == "production":
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        return response


app.add_middleware(_SecurityHeadersMiddleware)

# Add rate limiter
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# Admin dashboard HTML — served without auth (JS handles its own token auth)
@app.get("/admin", include_in_schema=False)
@app.get("/admin/", include_in_schema=False)
async def admin_dashboard() -> FileResponse:
    """Serve the admin dashboard HTML (no auth — JS manages API key)."""
    return FileResponse(
        Path(__file__).parent / "static" / "admin.html",
        headers={"Cache-Control": "no-cache"},
    )


# Admin API router (analytics endpoints) — auth required
from energy_forecast.serving.routers.admin import admin_router  # noqa: E402

app.include_router(admin_router, dependencies=[Depends(verify_api_key)])

# Static files for dashboard
_static_dir = Path(__file__).parent / "static"
if _static_dir.exists():
    app.mount("/static", StaticFiles(directory=_static_dir), name="static")


# ---------------------------------------------------------------------------
# Exception Handlers
# ---------------------------------------------------------------------------


@app.exception_handler(APIError)
async def api_error_handler(request: Request, exc: APIError) -> JSONResponse:
    """Handle custom API exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(error=exc.__class__.__name__, detail=exc.detail).model_dump(),
    )


@app.exception_handler(JobQueueFullError)
async def job_queue_full_handler(request: Request, exc: JobQueueFullError) -> JSONResponse:
    """Handle job queue full (429)."""
    return JSONResponse(
        status_code=429,
        content=ErrorResponse(error="JobQueueFull", detail=exc.detail).model_dump(),
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        timestamp=datetime.now(tz=TZ_ISTANBUL),
        version=__version__,
    )


@app.post("/predict", response_model=JobResponse)
@limiter.limit(_rate_limit)
async def predict(
    request: Request,
    file: UploadFile,
    email: Annotated[EmailStr, Form()],
    _auth: HTTPAuthorizationCredentials = Depends(verify_api_key),  # noqa: B008
) -> JobResponse:
    """Submit a prediction job.

    Uploads Excel file, creates a job, and enqueues it for sequential processing.
    Returns 429 if the queue is full (max_queue_size exceeded).

    Args:
        request: FastAPI request object.
        file: Uploaded Excel file with consumption data.
        email: Email address to send results.

    Returns:
        Job creation response with job_id and queue position.
    """
    # TODO(future): forecast_type toggle (GOP-only vs GOP+GİP) — currently only T+1 (24h)
    # Get services from app state
    file_service: FileService = request.app.state.file_service
    job_manager: JobManager = request.app.state.job_manager
    prediction_service: PredictionService = request.app.state.prediction_service
    email_service: EmailService = request.app.state.email_service

    # Check if models are loaded
    if not prediction_service.is_ready:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Please contact administrator.",
        )

    # Save uploaded file
    excel_path, file_stem = file_service.save_upload(file)

    use_db: bool = getattr(request.app.state, "use_db", False)

    if use_db:
        session_factory = request.app.state.session_factory

        # Lock ensures create+enqueue is atomic — no interleaving under concurrency
        async with job_manager._enqueue_lock:
            async with session_factory() as session:
                job = await job_manager.create_job_db(
                    session, email=str(email), excel_path=excel_path, file_stem=file_stem
                )

            position = job_manager.enqueue(
                job_id=job.id,
                excel_path=str(excel_path),
                email=str(email),
                file_stem=file_stem,
                created_at=job.created_at,
                session_factory=session_factory,
                prediction_service=prediction_service,
                file_service=file_service,
                email_service=email_service,
                is_db_mode=True,
                job_ref=None,
            )

        # Audit log (non-fatal, outside lock)
        try:
            from energy_forecast.db.repositories.audit_repo import AuditRepository

            async with session_factory() as session:
                audit = AuditRepository(session)
                await audit.log(
                    action="predict_request",
                    user_email=str(email),
                    ip_address=(request.client.host if request.client else None),
                    details={
                        "job_id": job.id,
                        "file_name": file.filename,
                    },
                )
                await session.commit()
        except Exception as e:
            logger.warning("Audit log failed: {}", e)

        msg = "Job queued successfully. Results will be sent to your email."
        if position > 1:
            msg = f"Job queued at position {position}. Results will be sent to your email."

        return JobResponse(
            job_id=job.id,
            status=JobStatus(job.status),
            message=msg,
            created_at=job.created_at,
        )

    # In-memory fallback
    async with job_manager._enqueue_lock:
        job_mem = job_manager.create_job_in_memory(
            email=str(email), excel_path=excel_path, file_stem=file_stem
        )

        position = job_manager.enqueue(
            job_id=job_mem.id,
            excel_path=str(excel_path),
            email=str(email),
            file_stem=file_stem,
            created_at=job_mem.created_at,
            session_factory=None,
            prediction_service=prediction_service,
            file_service=file_service,
            email_service=email_service,
            is_db_mode=False,
            job_ref=job_mem,
        )

    msg = "Job queued successfully. Results will be sent to your email."
    if position > 1:
        msg = f"Job queued at position {position}. Results will be sent to your email."

    return JobResponse(
        job_id=job_mem.id,
        status=job_mem.status,
        message=msg,
        created_at=job_mem.created_at,
    )


@app.get("/status/active")
async def get_active_status(
    request: Request,
    _auth: HTTPAuthorizationCredentials = Depends(verify_api_key),  # noqa: B008
) -> dict[str, object]:
    """Check if a prediction job is currently running.

    Used by the frontend to disable the form when a job is in progress.
    Returns real queue_size from the asyncio.Queue.
    """
    job_manager: JobManager = request.app.state.job_manager
    qsize = job_manager.queue_size
    max_q = job_manager.max_queue_size

    use_db: bool = getattr(request.app.state, "use_db", False)
    if use_db:
        session_factory = request.app.state.session_factory
        async with session_factory() as session:
            active = await job_manager.get_active_job_db(session)
        return {
            "busy": active is not None,
            "queue_size": qsize,
            "max_queue_size": max_q,
            "queue_full": qsize >= max_q,
        }

    active_mem = job_manager.get_active_job_in_memory()
    return {
        "busy": active_mem is not None,
        "queue_size": qsize,
        "max_queue_size": max_q,
        "queue_full": qsize >= max_q,
    }


@app.get("/status/{job_id}", response_model=JobStatusResponse)
async def get_status(
    request: Request,
    job_id: str,
    _auth: HTTPAuthorizationCredentials = Depends(verify_api_key),  # noqa: B008
) -> JobStatusResponse:
    """Get job status by ID.

    Args:
        request: FastAPI request object.
        job_id: Job identifier from /predict response.

    Returns:
        Job status with progress information.
    """
    job_manager: JobManager = request.app.state.job_manager
    use_db: bool = getattr(request.app.state, "use_db", False)

    try:
        if use_db:
            session_factory = request.app.state.session_factory
            async with session_factory() as session:
                job = await job_manager.get_job_db(session, job_id)
            return JobStatusResponse(
                job_id=job.id,
                status=JobStatus(job.status),
                progress=job.progress,
                error=job.error,
                created_at=job.created_at,
                completed_at=job.completed_at,
            )
        job_mem = job_manager.get_job_in_memory(job_id)
        return JobStatusResponse(
            job_id=job_mem.id,
            status=job_mem.status,
            progress=job_mem.progress,
            error=job_mem.error,
            created_at=job_mem.created_at,
            completed_at=job_mem.completed_at,
        )
    except JobNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as e:
        logger.error("Unexpected error fetching job {}: {}", job_id, e)
        raise HTTPException(status_code=500, detail="Internal server error") from e


@app.delete("/status/{job_id}")
async def delete_job(
    request: Request,
    job_id: str,
    _auth: HTTPAuthorizationCredentials = Depends(verify_api_key),  # noqa: B008
) -> dict[str, str]:
    """Delete or cancel a job by ID."""
    job_manager: JobManager = request.app.state.job_manager
    use_db: bool = getattr(request.app.state, "use_db", False)

    try:
        if use_db:
            from energy_forecast.db.repositories.job_repo import JobRepository

            session_factory = request.app.state.session_factory
            async with session_factory() as session:
                repo = JobRepository(session)
                job = await repo.get_by_id(job_id)
                if job is None:
                    raise JobNotFoundError(f"Job not found: {job_id}")
                if job.status in ("running",):
                    raise HTTPException(status_code=409, detail="Cannot delete a running job")
                await repo.update_status(job_id, "archived")
                await session.commit()
            return {"detail": f"Job {job_id} archived"}

        job_mem = job_manager.get_job_in_memory(job_id)
        if job_mem.status == JobStatus.RUNNING:
            raise HTTPException(status_code=409, detail="Cannot delete a running job")
        job_mem.status = JobStatus.FAILED
        job_mem.error = "Cancelled by user"
        return {"detail": f"Job {job_id} cancelled"}
    except JobNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/models")
async def get_models(
    request: Request,
    _auth: HTTPAuthorizationCredentials = Depends(verify_api_key),  # noqa: B008
) -> dict[str, object]:
    """Get information about loaded models."""
    prediction_service: PredictionService = request.app.state.prediction_service
    return prediction_service.get_model_info()


@app.get("/jobs")
async def list_jobs(
    request: Request,
    _auth: HTTPAuthorizationCredentials = Depends(verify_api_key),  # noqa: B008
) -> dict[str, object]:
    """List all jobs (for debugging/admin)."""
    job_manager: JobManager = request.app.state.job_manager
    use_db: bool = getattr(request.app.state, "use_db", False)

    if use_db:
        from energy_forecast.db.repositories.job_repo import JobRepository

        session_factory = request.app.state.session_factory
        async with session_factory() as session:
            repo = JobRepository(session)
            db_jobs = await repo.get_all()
            stats = await repo.get_stats()
        return {
            "count": len(db_jobs),
            "stats": stats,
            "jobs": [
                {
                    "id": j.id,
                    "status": j.status,
                    "email": mask_email(j.email),
                    "created_at": j.created_at.isoformat(),
                    "completed_at": (j.completed_at.isoformat() if j.completed_at else None),
                }
                for j in db_jobs
            ],
        }

    jobs = job_manager.get_all_jobs_in_memory()
    return {
        "count": len(jobs),
        "stats": job_manager.get_stats_in_memory(),
        "jobs": [
            {
                "id": j.id,
                "status": j.status,
                "email": mask_email(j.email),
                "created_at": j.created_at.isoformat(),
                "completed_at": (j.completed_at.isoformat() if j.completed_at else None),
            }
            for j in jobs
        ],
    }


@app.get("/", include_in_schema=False)
async def root() -> FileResponse:
    """Serve the dashboard."""
    return FileResponse(
        Path(__file__).parent / "static" / "dashboard.html",
        headers={"Cache-Control": "no-cache"},
    )
