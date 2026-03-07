"""Tests for GoogleDriveStorage (mocked, no real API calls)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from energy_forecast.storage.gdrive import GoogleDriveStorage


@pytest.fixture()
def gdrive(tmp_path: Path) -> GoogleDriveStorage:
    """Create a GoogleDriveStorage with a fake credentials path."""
    creds_path = tmp_path / "creds.json"
    creds_path.write_text("{}")
    return GoogleDriveStorage(
        credentials_path=str(creds_path),
        root_folder_id="root_folder_123",
    )


@pytest.fixture()
def mock_service() -> MagicMock:
    """Create a mock Google Drive API service."""
    service = MagicMock()

    # files().create().execute() returns file with id
    create_exec = MagicMock(return_value={"id": "new_folder_id"})
    service.files.return_value.create.return_value.execute = create_exec

    # files().list().execute() returns empty by default
    list_exec = MagicMock(return_value={"files": []})
    service.files.return_value.list.return_value.execute = list_exec

    return service


class TestGoogleDriveStorage:
    """Tests for GoogleDriveStorage."""

    def test_init(self, gdrive: GoogleDriveStorage) -> None:
        """Test initialization stores config."""
        assert gdrive._root_folder_id == "root_folder_123"
        assert gdrive._service is None

    @patch("energy_forecast.storage.gdrive.GoogleDriveStorage._get_service")
    def test_create_folder(
        self,
        mock_get_svc: MagicMock,
        gdrive: GoogleDriveStorage,
        mock_service: MagicMock,
    ) -> None:
        """Test folder creation calls API with correct metadata."""
        mock_get_svc.return_value = mock_service

        folder_id = gdrive._create_folder("test_folder", "parent_123")

        assert folder_id == "new_folder_id"
        mock_service.files.return_value.create.assert_called_once()
        call_kwargs = mock_service.files.return_value.create.call_args
        body = call_kwargs.kwargs.get("body") or call_kwargs[1].get("body")
        assert body["name"] == "test_folder"
        assert body["parents"] == ["parent_123"]

    @patch("energy_forecast.storage.gdrive.GoogleDriveStorage._get_service")
    def test_get_or_create_month_folder_creates_new(
        self,
        mock_get_svc: MagicMock,
        gdrive: GoogleDriveStorage,
        mock_service: MagicMock,
    ) -> None:
        """Test month folder creation when none exists."""
        mock_get_svc.return_value = mock_service

        folder_id = gdrive._get_or_create_month_folder("2026-03")

        assert folder_id == "new_folder_id"
        # Should be cached
        assert gdrive._month_cache["2026-03"] == "new_folder_id"

    @patch("energy_forecast.storage.gdrive.GoogleDriveStorage._get_service")
    def test_get_or_create_month_folder_finds_existing(
        self,
        mock_get_svc: MagicMock,
        gdrive: GoogleDriveStorage,
        mock_service: MagicMock,
    ) -> None:
        """Test month folder lookup when it already exists."""
        mock_service.files.return_value.list.return_value.execute.return_value = {
            "files": [{"id": "existing_id"}]
        }
        mock_get_svc.return_value = mock_service

        folder_id = gdrive._get_or_create_month_folder("2026-03")

        assert folder_id == "existing_id"
        assert gdrive._month_cache["2026-03"] == "existing_id"

    @patch("energy_forecast.storage.gdrive.GoogleDriveStorage._get_service")
    def test_get_or_create_month_folder_uses_cache(
        self,
        mock_get_svc: MagicMock,
        gdrive: GoogleDriveStorage,
        mock_service: MagicMock,
    ) -> None:
        """Test month folder cache hit avoids API call."""
        mock_get_svc.return_value = mock_service
        gdrive._month_cache["2026-03"] = "cached_id"

        folder_id = gdrive._get_or_create_month_folder("2026-03")

        assert folder_id == "cached_id"
        mock_service.files.return_value.list.assert_not_called()

    @patch("energy_forecast.storage.gdrive.GoogleDriveStorage._upload_file")
    @patch(
        "energy_forecast.storage.gdrive.GoogleDriveStorage._get_or_create_month_folder"
    )
    @patch("energy_forecast.storage.gdrive.GoogleDriveStorage._create_folder")
    def test_upload_job_artifacts(
        self,
        mock_create_folder: MagicMock,
        mock_month_folder: MagicMock,
        mock_upload: MagicMock,
        gdrive: GoogleDriveStorage,
        tmp_path: Path,
    ) -> None:
        """Test uploading multiple files for a job."""
        mock_month_folder.return_value = "month_folder_id"
        mock_create_folder.return_value = "job_folder_id"
        mock_upload.return_value = "uploaded_file_id"

        # Create real files
        f1 = tmp_path / "historical.parquet"
        f1.write_text("data1")
        f2 = tmp_path / "forecast.parquet"
        f2.write_text("data2")

        result = gdrive.upload_job_artifacts(
            "job_123", {"historical.parquet": f1, "forecast.parquet": f2}
        )

        assert result == {
            "historical.parquet": "uploaded_file_id",
            "forecast.parquet": "uploaded_file_id",
        }
        assert mock_upload.call_count == 2

    @patch("energy_forecast.storage.gdrive.GoogleDriveStorage._upload_file")
    @patch(
        "energy_forecast.storage.gdrive.GoogleDriveStorage._get_or_create_month_folder"
    )
    @patch("energy_forecast.storage.gdrive.GoogleDriveStorage._create_folder")
    def test_upload_skips_missing_files(
        self,
        mock_create_folder: MagicMock,
        mock_month_folder: MagicMock,
        mock_upload: MagicMock,
        gdrive: GoogleDriveStorage,
        tmp_path: Path,
    ) -> None:
        """Test that missing files are skipped gracefully."""
        mock_month_folder.return_value = "month_folder_id"
        mock_create_folder.return_value = "job_folder_id"

        missing = tmp_path / "nonexistent.parquet"

        result = gdrive.upload_job_artifacts(
            "job_123", {"nonexistent.parquet": missing}
        )

        assert result == {}
        mock_upload.assert_not_called()

    @patch("energy_forecast.storage.gdrive.GoogleDriveStorage._upload_file")
    @patch(
        "energy_forecast.storage.gdrive.GoogleDriveStorage._get_or_create_month_folder"
    )
    @patch("energy_forecast.storage.gdrive.GoogleDriveStorage._create_folder")
    def test_upload_handles_api_error(
        self,
        mock_create_folder: MagicMock,
        mock_month_folder: MagicMock,
        mock_upload: MagicMock,
        gdrive: GoogleDriveStorage,
        tmp_path: Path,
    ) -> None:
        """Test that upload errors are caught and logged."""
        mock_month_folder.return_value = "month_folder_id"
        mock_create_folder.return_value = "job_folder_id"
        mock_upload.side_effect = RuntimeError("API error")

        f1 = tmp_path / "test.parquet"
        f1.write_text("data")

        result = gdrive.upload_job_artifacts(
            "job_123", {"test.parquet": f1}
        )

        assert result == {}
