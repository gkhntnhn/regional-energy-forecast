"""FastAPI application with prediction endpoints."""

from __future__ import annotations

import secrets
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

from fastapi import BackgroundTasks, Depends, FastAPI, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles
from loguru import logger
from pydantic import EmailStr
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from energy_forecast.config.settings import load_config
from energy_forecast.serving.exceptions import APIError, JobNotFoundError, JobQueueFullError
from energy_forecast.serving.job_manager import JobManager
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
from energy_forecast.utils import TZ_ISTANBUL

if TYPE_CHECKING:
    pass


def _find_latest_model_dir(base_dir: Path, prefix: str) -> Path | None:
    """Find the latest timestamped model subdirectory.

    All models use timestamped subdirectories: ``{prefix}_YYYY-MM-DD_HH-MM/``.
    Scans ``base_dir`` for matching subdirectories and returns the most recent
    one (sorted by name, so timestamps sort chronologically).

    Args:
        base_dir: Parent directory (e.g. ``models/catboost``).
        prefix: Directory name prefix (e.g. ``"catboost"``, ``"prophet"``, ``"tft"``).

    Returns:
        Path to the latest matching subdirectory, or None if no match found.
    """
    if not base_dir.exists():
        return None
    subdirs = sorted(
        (p for p in base_dir.glob(f"{prefix}_*") if p.is_dir()),
        key=lambda p: p.name,
    )
    return subdirs[-1] if subdirs else None

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
    if credentials is None or not secrets.compare_digest(
        credentials.credentials, expected_key
    ):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return credentials

# ---------------------------------------------------------------------------
# Rate Limiter
# ---------------------------------------------------------------------------

limiter = Limiter(key_func=get_remote_address)


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
    logger.info("Starting Energy Forecast API...")

    # Load configuration
    try:
        settings = load_config()
    except FileNotFoundError:
        logger.warning("Config files not found, using defaults")
        from energy_forecast.config.settings import get_default_config

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
        catboost_dir = _find_latest_model_dir(models_dir / "catboost", "catboost")
        catboost_path = (
            catboost_dir / "model.cbm"
            if catboost_dir
            else models_dir / "catboost" / "model.cbm"
        )

        prophet_dir = _find_latest_model_dir(models_dir / "prophet", "prophet")
        prophet_path = (
            prophet_dir / "prophet_model.pkl"
            if prophet_dir
            else models_dir / "prophet" / "prophet_model.pkl"
        )

        tft_dir = _find_latest_model_dir(models_dir / "tft", "tft")
        tft_path = tft_dir if tft_dir else models_dir / "tft"

        ensemble_dir = _find_latest_model_dir(models_dir / "ensemble", "ensemble")
        logger.info("Serving from models/ with timestamped discovery")

    pred_config = PredictionServiceConfig(
        models_dir=models_dir,
        catboost_path=catboost_path,
        prophet_path=prophet_path,
        tft_path=tft_path,
        ensemble_dir=ensemble_dir,
    )
    app.state.prediction_service = PredictionService(pred_config, settings)

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

    # Initialize job manager
    app.state.job_manager = JobManager()

    # Initialize database (if DATABASE_URL is set)
    if settings.env.database_url:
        from energy_forecast.db import create_db_engine, create_session_factory

        engine = create_db_engine(settings.env.database_url, settings.database)
        session_factory = create_session_factory(engine)
        app.state.db_engine = engine
        app.state.session_factory = session_factory
        app.state.use_db = True
        masked_url = settings.env.database_url[:50] + "..."
        logger.info("Database connected: {}", masked_url)
    else:
        app.state.db_engine = None
        app.state.session_factory = None
        app.state.use_db = False
        logger.warning(
            "DATABASE_URL not set — using in-memory job storage (dev mode)"
        )

    # Start weather actuals scheduler (DB mode only)
    _scheduler_task = None
    if app.state.use_db:
        import asyncio

        from energy_forecast.jobs.weather_actuals import run_scheduler

        _scheduler_task = asyncio.create_task(
            run_scheduler(app.state.session_factory, settings)
        )
        logger.info("Weather actuals scheduler started (daily at 04:00)")

    logger.info("Energy Forecast API started successfully")

    yield

    # Cleanup on shutdown
    logger.info("Shutting down Energy Forecast API...")
    if _scheduler_task is not None:
        _scheduler_task.cancel()
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
    version="0.1.0",
    lifespan=lifespan,
)

# Config loaded at module level for CORS middleware; reloaded in lifespan for model paths
# Load CORS origins from config (fallback to ["*"] if config unavailable)
try:
    _cors_settings = load_config()
    _cors_origins = _cors_settings.api.cors_origins
except Exception as e:
    logger.warning("Failed to load CORS config, falling back to wildcard: {}", e)
    _cors_origins = ["*"]

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

# Add rate limiter
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

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
        version="0.1.0",
    )


@app.post("/predict", response_model=JobResponse)
@limiter.limit("10/minute")
async def predict(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile,
    email: Annotated[EmailStr, Form()],
    _auth: HTTPAuthorizationCredentials = Depends(verify_api_key),  # noqa: B008
) -> JobResponse:
    """Submit a prediction job.

    Uploads Excel file, creates a job, and queues it for processing.
    Only one job can run at a time. Returns 429 if a job is already running.

    Args:
        request: FastAPI request object.
        background_tasks: Background task queue.
        file: Uploaded Excel file with consumption data.
        email: Email address to send results.

    Returns:
        Job creation response with job_id.
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
        async with session_factory() as session:
            job = await job_manager.create_job_db(
                session, email=str(email), excel_path=excel_path, file_stem=file_stem
            )

        # Audit log (non-fatal)
        try:
            from energy_forecast.db.repositories.audit_repo import AuditRepository

            async with session_factory() as session:
                audit = AuditRepository(session)
                await audit.log(
                    action="predict_request",
                    user_email=str(email),
                    ip_address=(
                        request.client.host if request.client else None
                    ),
                    details={
                        "job_id": job.id,
                        "file_name": file.filename,
                    },
                )
                await session.commit()
        except Exception as e:
            logger.warning("Audit log failed: {}", e)

        background_tasks.add_task(
            job_manager.process_job_db,
            job_id=job.id,
            excel_path=str(excel_path),
            email=str(email),
            file_stem=file_stem,
            created_at=job.created_at,
            session_factory=session_factory,
            prediction_service=prediction_service,
            file_service=file_service,
            email_service=email_service,
        )

        return JobResponse(
            job_id=job.id,
            status=JobStatus(job.status),
            message="Job queued successfully. Results will be sent to your email.",
            created_at=job.created_at,
        )

    # In-memory fallback
    if job_manager.has_active_job_in_memory():
        active_job = job_manager.get_active_job_in_memory()
        job_id_str = active_job.id if active_job else "unknown"
        raise JobQueueFullError(
            f"A prediction job is currently running (ID: {job_id_str}). "
            "Please try again later."
        )

    job_mem = job_manager.create_job_in_memory(
        email=str(email), excel_path=excel_path, file_stem=file_stem
    )

    background_tasks.add_task(
        job_manager.process_job_in_memory,
        job=job_mem,
        prediction_service=prediction_service,
        file_service=file_service,
        email_service=email_service,
    )

    return JobResponse(
        job_id=job_mem.id,
        status=job_mem.status,
        message="Job queued successfully. Results will be sent to your email.",
        created_at=job_mem.created_at,
    )


@app.get("/status/active")
async def get_active_status(request: Request) -> dict[str, object]:
    """Check if a prediction job is currently running (no auth required).

    Used by the frontend to disable the form when a job is in progress.
    """
    job_manager: JobManager = request.app.state.job_manager
    use_db: bool = getattr(request.app.state, "use_db", False)

    if use_db:
        session_factory = request.app.state.session_factory
        async with session_factory() as session:
            active = await job_manager.get_active_job_db(session)
        if active:
            return {
                "busy": True,
                "message": "Devam eden bir tahmin islemi bulunmaktadir...",
                "job_id": active.id,
                "status": active.status,
                "progress": active.progress,
                "started_at": active.created_at.isoformat(),
            }
        return {"busy": False}

    active_mem = job_manager.get_active_job_in_memory()
    if active_mem:
        return {
            "busy": True,
            "message": "Devam eden bir tahmin islemi bulunmaktadir...",
            "job_id": active_mem.id,
            "status": active_mem.status.value,
            "progress": active_mem.progress,
            "started_at": active_mem.created_at.isoformat(),
        }
    return {"busy": False}


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
                    "email": j.email[:3] + "***",
                    "created_at": j.created_at.isoformat(),
                    "completed_at": (
                        j.completed_at.isoformat() if j.completed_at else None
                    ),
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
                "email": j.email[:3] + "***",
                "created_at": j.created_at.isoformat(),
                "completed_at": (
                    j.completed_at.isoformat() if j.completed_at else None
                ),
            }
            for j in jobs
        ],
    }


@app.get("/", include_in_schema=False)
async def root() -> FileResponse:
    """Serve the dashboard."""
    return FileResponse(Path(__file__).parent / "static" / "dashboard.html")
