"""Tests for file service."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pandas as pd
import pytest

from energy_forecast.serving.exceptions import (
    FileTooLargeError,
    InvalidFileTypeError,
)
from energy_forecast.serving.services.file_service import FileService, FileServiceConfig

if TYPE_CHECKING:
    pass


@pytest.fixture
def file_service(tmp_path: Path) -> FileService:
    """Create file service with temp directories."""
    config = FileServiceConfig(
        upload_dir=tmp_path / "uploads",
        output_dir=tmp_path / "outputs",
        allowed_extensions=[".xlsx", ".xls"],
        max_file_size_mb=1,
        cleanup_after_hours=1,
    )
    return FileService(config)


@pytest.fixture
def mock_upload_file() -> MagicMock:
    """Create mock UploadFile."""
    mock = MagicMock()
    mock.filename = "test.xlsx"
    mock.file = BytesIO(b"test content")
    return mock


class TestFileServiceConfig:
    """Tests for FileServiceConfig."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = FileServiceConfig()
        assert config.upload_dir == Path("data/uploads")
        assert config.output_dir == Path("data/outputs")
        assert ".xlsx" in config.allowed_extensions

    def test_custom_config(self, tmp_path: Path) -> None:
        """Test custom configuration."""
        config = FileServiceConfig(
            upload_dir=tmp_path / "custom_upload",
            max_file_size_mb=100,
        )
        assert config.upload_dir == tmp_path / "custom_upload"
        assert config.max_file_size_mb == 100


class TestFileService:
    """Tests for FileService."""

    def test_directories_created(self, file_service: FileService) -> None:
        """Test that directories are created on init."""
        assert file_service._config.upload_dir.exists()
        assert file_service._config.output_dir.exists()

    def test_save_upload_success(
        self,
        file_service: FileService,
        mock_upload_file: MagicMock,
    ) -> None:
        """Test successful file upload."""
        path = file_service.save_upload(mock_upload_file)

        assert path.exists()
        assert path.suffix == ".xlsx"
        assert path.parent == file_service._config.upload_dir

    def test_save_upload_invalid_extension(
        self,
        file_service: FileService,
        mock_upload_file: MagicMock,
    ) -> None:
        """Test rejection of invalid file extension."""
        mock_upload_file.filename = "test.csv"

        with pytest.raises(InvalidFileTypeError):
            file_service.save_upload(mock_upload_file)

    def test_save_upload_too_large(
        self,
        file_service: FileService,
        mock_upload_file: MagicMock,
    ) -> None:
        """Test rejection of oversized file."""
        # Create content larger than 1MB limit
        mock_upload_file.file = BytesIO(b"x" * (2 * 1024 * 1024))

        with pytest.raises(FileTooLargeError):
            file_service.save_upload(mock_upload_file)

    def test_create_output_xlsx(self, file_service: FileService) -> None:
        """Test output Excel creation."""
        # Create sample predictions DataFrame
        predictions = pd.DataFrame(
            {
                "consumption_mwh": [1000.0, 1100.0, 1200.0],
            },
            index=pd.date_range("2025-01-01", periods=3, freq="h"),
        )

        path = file_service.create_output_xlsx(predictions, "test123")

        assert path.exists()
        assert path.suffix == ".xlsx"
        assert "test123" in path.name

        # Verify content
        df = pd.read_excel(path)
        assert len(df) == 3
        assert "Tahmin (MWh)" in df.columns

    def test_delete_file(self, file_service: FileService, tmp_path: Path) -> None:
        """Test file deletion."""
        # Create a test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("test")

        assert file_service.delete_file(test_file) is True
        assert not test_file.exists()

    def test_delete_nonexistent_file(self, file_service: FileService) -> None:
        """Test deletion of nonexistent file."""
        result = file_service.delete_file(Path("/nonexistent/file.txt"))
        assert result is False

    def test_cleanup_old_files(
        self,
        file_service: FileService,
    ) -> None:
        """Test old file cleanup."""
        # Create a file (will be new, so not cleaned)
        test_file = file_service._config.upload_dir / "new_file.txt"
        test_file.write_text("test")

        removed = file_service.cleanup_old_files()

        # File is new, should not be removed
        assert removed == 0
        assert test_file.exists()
