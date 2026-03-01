"""File upload and output generation service."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from loguru import logger
from pydantic import BaseModel, Field

from energy_forecast.serving.exceptions import (
    FileTooLargeError,
    FileUploadError,
    InvalidFileTypeError,
)
from energy_forecast.utils import TZ_ISTANBUL

if TYPE_CHECKING:
    from fastapi import UploadFile


class FileServiceConfig(BaseModel, frozen=True):
    """File service configuration."""

    upload_dir: Path = Field(default=Path("data/uploads"))
    output_dir: Path = Field(default=Path("data/outputs"))
    allowed_extensions: list[str] = Field(default_factory=lambda: [".xlsx", ".xls"])
    max_file_size_mb: int = Field(default=50)
    cleanup_after_hours: int = Field(default=24)


class FileService:
    """Handles file uploads, output generation, and cleanup.

    Args:
        config: File service configuration.
    """

    def __init__(self, config: FileServiceConfig) -> None:
        self._config = config
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        """Create upload and output directories if they don't exist."""
        self._config.upload_dir.mkdir(parents=True, exist_ok=True)
        self._config.output_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(
            "File service directories ensured: upload={}, output={}",
            self._config.upload_dir,
            self._config.output_dir,
        )

    def save_upload(self, file: UploadFile) -> tuple[Path, str]:
        """Save uploaded file to disk with validation.

        Args:
            file: FastAPI UploadFile object.

        Returns:
            Tuple of (path to saved file, file_stem for output pairing).
            The file_stem is a DD-MM-YYYY_HH-MM-SS timestamp shared between
            the input and output files for traceability.

        Raises:
            InvalidFileTypeError: If file extension not allowed.
            FileTooLargeError: If file exceeds size limit.
            FileUploadError: If file save fails.
        """
        if file.filename is None:
            raise FileUploadError("No filename provided")

        # Validate extension
        ext = Path(file.filename).suffix.lower()
        if ext not in self._config.allowed_extensions:
            raise InvalidFileTypeError(
                f"File type '{ext}' not allowed. Allowed: {self._config.allowed_extensions}"
            )

        # Generate traceable filename: DD-MM-YYYY_HH-MM-SS_Input.xlsx
        file_stem = datetime.now(tz=TZ_ISTANBUL).strftime("%d-%m-%Y_%H-%M-%S")
        safe_filename = f"{file_stem}_Input{ext}"
        save_path = self._config.upload_dir / safe_filename

        try:
            # Read and check size
            content = file.file.read()
            size_mb = len(content) / (1024 * 1024)

            if size_mb > self._config.max_file_size_mb:
                raise FileTooLargeError(
                    f"File size {size_mb:.1f}MB exceeds limit of "
                    f"{self._config.max_file_size_mb}MB"
                )

            # Write to disk
            with open(save_path, "wb") as f:
                f.write(content)

            logger.info(
                "Saved uploaded file: {} ({:.2f}MB)",
                save_path.name,
                size_mb,
            )
            return save_path, file_stem

        except (FileTooLargeError, InvalidFileTypeError):
            raise
        except Exception as e:
            raise FileUploadError(f"Failed to save file: {e}") from e
        finally:
            file.file.seek(0)  # Reset file pointer

    def create_output_xlsx(
        self,
        predictions: pd.DataFrame,
        file_stem: str,
    ) -> Path:
        """Create output Excel file with predictions.

        Args:
            predictions: DataFrame with datetime index and prediction column.
            file_stem: Shared timestamp stem (DD-MM-YYYY_HH-MM-SS) from upload.

        Returns:
            Path to created Excel file.
        """
        filename = f"{file_stem}_Forecast.xlsx"
        output_path = self._config.output_dir / filename

        # Prepare DataFrame for export — customer-friendly column names
        export_df = predictions.reset_index()
        if "datetime" not in export_df.columns and "index" in export_df.columns:
            export_df = export_df.rename(columns={"index": "datetime"})
        export_df = export_df.rename(columns={
            "datetime": "Tarih",
            "consumption_mwh": "Tahmin (MWh)",
        })

        # Write to Excel
        export_df.to_excel(output_path, index=False, engine="openpyxl")

        logger.info("Created output file: {}", output_path.name)
        return output_path

    def cleanup_old_files(self) -> int:
        """Remove files older than configured threshold.

        Returns:
            Number of files removed.
        """
        threshold = datetime.now(tz=TZ_ISTANBUL) - timedelta(hours=self._config.cleanup_after_hours)
        removed = 0

        for directory in [self._config.upload_dir, self._config.output_dir]:
            for file_path in directory.iterdir():
                if file_path.is_file():
                    mtime = datetime.fromtimestamp(file_path.stat().st_mtime, tz=TZ_ISTANBUL)
                    if mtime < threshold:
                        file_path.unlink()
                        removed += 1
                        logger.debug("Removed old file: {}", file_path.name)

        if removed > 0:
            logger.info("Cleaned up {} old files", removed)
        return removed

    def delete_file(self, path: Path) -> bool:
        """Delete a specific file.

        Args:
            path: Path to file.

        Returns:
            True if deleted, False if not found.
        """
        if path.exists():
            path.unlink()
            logger.debug("Deleted file: {}", path.name)
            return True
        return False

    def get_file_path(self, filename: str, output: bool = True) -> Path | None:
        """Get full path for a filename.

        Args:
            filename: Name of file.
            output: If True, look in output dir; otherwise upload dir.

        Returns:
            Full path if exists, None otherwise.
        """
        base_dir = self._config.output_dir if output else self._config.upload_dir
        path = base_dir / filename
        return path if path.exists() else None
