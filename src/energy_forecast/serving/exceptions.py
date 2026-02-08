"""API exception hierarchy for structured error handling."""

from __future__ import annotations


class APIError(Exception):
    """Base API exception with HTTP status code.

    All API exceptions inherit from this class. Each exception
    defines a default status code and detail message.

    Args:
        detail: Error message to return to client.
        status_code: HTTP status code (overrides class default).
    """

    status_code: int = 500
    default_detail: str = "Internal server error"

    def __init__(
        self,
        detail: str | None = None,
        status_code: int | None = None,
    ) -> None:
        self.detail = detail or self.default_detail
        if status_code is not None:
            self.status_code = status_code
        super().__init__(self.detail)


class ValidationError(APIError):
    """Request validation error (422)."""

    status_code = 422
    default_detail = "Request validation failed"


class FileUploadError(APIError):
    """File upload error (400)."""

    status_code = 400
    default_detail = "File upload failed"


class FileTooLargeError(FileUploadError):
    """File exceeds maximum size limit."""

    default_detail = "File exceeds maximum allowed size"


class InvalidFileTypeError(FileUploadError):
    """File type not allowed."""

    default_detail = "File type not allowed"


class PredictionError(APIError):
    """Prediction pipeline error (500)."""

    status_code = 500
    default_detail = "Prediction failed"


class ModelNotLoadedError(PredictionError):
    """Model not loaded on startup."""

    default_detail = "Model not loaded. Please contact administrator."


class FeaturePipelineError(PredictionError):
    """Feature engineering pipeline error."""

    default_detail = "Feature pipeline failed"


class EmailDeliveryError(APIError):
    """Email sending error (500)."""

    status_code = 500
    default_detail = "Failed to send email"


class JobNotFoundError(APIError):
    """Job not found (404)."""

    status_code = 404
    default_detail = "Job not found"


class JobQueueFullError(APIError):
    """Job queue is full / active job running (429)."""

    status_code = 429
    default_detail = "A prediction job is currently running. Please try again later."
