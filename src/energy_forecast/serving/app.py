"""FastAPI application with prediction endpoints."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

from fastapi import BackgroundTasks, FastAPI, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import EmailStr
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from energy_forecast.config.settings import load_config
from energy_forecast.serving.exceptions import APIError, JobQueueFullError
from energy_forecast.serving.job_manager import JobManager
from energy_forecast.serving.schemas import (
    ErrorResponse,
    HealthResponse,
    JobResponse,
    JobStatusResponse,
)
from energy_forecast.serving.services.email_service import EmailService, EmailServiceConfig
from energy_forecast.serving.services.file_service import FileService, FileServiceConfig
from energy_forecast.serving.services.prediction_service import (
    PredictionService,
    PredictionServiceConfig,
)

if TYPE_CHECKING:
    pass

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
    )
    app.state.email_service = EmailService(email_config)

    # Initialize prediction service
    pred_config = PredictionServiceConfig(
        models_dir=Path(settings.paths.models_dir),
        catboost_path=Path(settings.paths.models_dir) / "catboost" / "model.cbm",
        prophet_path=Path(settings.paths.models_dir) / "prophet" / "model.pkl",
        tft_path=Path(settings.paths.models_dir) / "tft",
    )
    app.state.prediction_service = PredictionService(pred_config, settings)

    # Try to load models (warn if not available)
    try:
        app.state.prediction_service.load_models()
        logger.info("Models loaded successfully")
    except Exception as e:
        logger.warning("Failed to load models (API will reject predictions): {}", e)

    # Initialize job manager
    app.state.job_manager = JobManager()

    logger.info("Energy Forecast API started successfully")

    yield

    # Cleanup on shutdown
    logger.info("Shutting down Energy Forecast API...")
    app.state.file_service.cleanup_old_files()
    app.state.job_manager.cleanup_old_jobs()
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

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add rate limiter
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


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
        timestamp=datetime.now(),
        version="0.1.0",
    )


@app.post("/predict", response_model=JobResponse)
@limiter.limit("10/minute")
async def predict(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile,
    email: Annotated[EmailStr, Form()],
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

    # Check if a job is already running (returns 429)
    if job_manager.has_active_job():
        active_job = job_manager.get_active_job()
        job_id_str = active_job.id if active_job else "unknown"
        raise JobQueueFullError(
            f"A prediction job is currently running (ID: {job_id_str}). "
            "Please try again later."
        )

    # Save uploaded file
    excel_path = file_service.save_upload(file)

    # Create job
    job = job_manager.create_job(email=str(email), excel_path=excel_path)

    # Queue background task
    background_tasks.add_task(
        job_manager.process_job,
        job=job,
        prediction_service=prediction_service,
        file_service=file_service,
        email_service=email_service,
    )

    return JobResponse(
        job_id=job.id,
        status=job.status,
        message="Job queued successfully. Results will be sent to your email.",
        created_at=job.created_at,
    )


@app.get("/status/{job_id}", response_model=JobStatusResponse)
async def get_status(request: Request, job_id: str) -> JobStatusResponse:
    """Get job status by ID.

    Args:
        request: FastAPI request object.
        job_id: Job identifier from /predict response.

    Returns:
        Job status with progress information.
    """
    job_manager: JobManager = request.app.state.job_manager

    try:
        job = job_manager.get_job(job_id)
    except Exception as e:
        raise HTTPException(status_code=404, detail="Job not found") from e

    return JobStatusResponse(
        job_id=job.id,
        status=job.status,
        progress=job.progress,
        error=job.error,
        created_at=job.created_at,
        completed_at=job.completed_at,
    )


@app.get("/models")
async def get_models(request: Request) -> dict[str, object]:
    """Get information about loaded models."""
    prediction_service: PredictionService = request.app.state.prediction_service
    return prediction_service.get_model_info()


@app.get("/jobs")
async def list_jobs(request: Request) -> dict[str, object]:
    """List all jobs (for debugging/admin)."""
    job_manager: JobManager = request.app.state.job_manager
    jobs = job_manager.get_all_jobs()

    return {
        "count": len(jobs),
        "stats": job_manager.get_stats(),
        "jobs": [
            {
                "id": j.id,
                "status": j.status,
                "email": j.email[:3] + "***",  # Mask email for privacy
                "created_at": j.created_at.isoformat(),
                "completed_at": j.completed_at.isoformat() if j.completed_at else None,
            }
            for j in jobs
        ],
    }
