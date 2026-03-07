"""Tests for GoogleDriveStorage (mocked, no real API calls)."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from energy_forecast.storage.gdrive import GoogleDriveStorage
from energy_forecast.utils import TZ_ISTANBUL


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
    def test_find_folder_returns_id_when_exists(
        self,
        mock_get_svc: MagicMock,
        gdrive: GoogleDriveStorage,
        mock_service: MagicMock,
    ) -> None:
        """Test _find_folder returns folder ID when folder exists."""
        mock_service.files.return_value.list.return_value.execute.return_value = {
            "files": [{"id": "existing_id"}]
        }
        mock_get_svc.return_value = mock_service

        folder_id = gdrive._find_folder("forecasts", "root_folder_123")

        assert folder_id == "existing_id"

    @patch("energy_forecast.storage.gdrive.GoogleDriveStorage._get_service")
    def test_find_folder_returns_none_when_missing(
        self,
        mock_get_svc: MagicMock,
        gdrive: GoogleDriveStorage,
        mock_service: MagicMock,
    ) -> None:
        """Test _find_folder returns None when folder doesn't exist."""
        mock_get_svc.return_value = mock_service

        folder_id = gdrive._find_folder("nonexistent", "root_folder_123")

        assert folder_id is None

    @patch("energy_forecast.storage.gdrive.GoogleDriveStorage._create_folder")
    @patch("energy_forecast.storage.gdrive.GoogleDriveStorage._find_folder")
    def test_ensure_folder_path_creates_hierarchy(
        self,
        mock_find: MagicMock,
        mock_create: MagicMock,
        gdrive: GoogleDriveStorage,
    ) -> None:
        """Test _ensure_folder_path creates nested folders when none exist."""
        mock_find.return_value = None
        mock_create.side_effect = ["f1", "f2", "f3"]

        result = gdrive._ensure_folder_path(["forecasts", "2026", "03"])

        assert result == "f3"
        assert mock_create.call_count == 3
        mock_create.assert_any_call("forecasts", "root_folder_123")
        mock_create.assert_any_call("2026", "f1")
        mock_create.assert_any_call("03", "f2")

    @patch("energy_forecast.storage.gdrive.GoogleDriveStorage._create_folder")
    @patch("energy_forecast.storage.gdrive.GoogleDriveStorage._find_folder")
    def test_ensure_folder_path_reuses_existing(
        self,
        mock_find: MagicMock,
        mock_create: MagicMock,
        gdrive: GoogleDriveStorage,
    ) -> None:
        """Test _ensure_folder_path finds existing folders instead of creating."""
        mock_find.side_effect = ["existing_f1", "existing_f2"]

        result = gdrive._ensure_folder_path(["forecasts", "2026"])

        assert result == "existing_f2"
        mock_create.assert_not_called()

    @patch("energy_forecast.storage.gdrive.GoogleDriveStorage._create_folder")
    @patch("energy_forecast.storage.gdrive.GoogleDriveStorage._find_folder")
    def test_ensure_folder_path_uses_cache(
        self,
        mock_find: MagicMock,
        mock_create: MagicMock,
        gdrive: GoogleDriveStorage,
    ) -> None:
        """Test _ensure_folder_path cache hit avoids API calls."""
        gdrive._folder_cache["forecasts"] = "cached_f1"
        gdrive._folder_cache["forecasts/2026"] = "cached_f2"
        mock_find.return_value = None
        mock_create.return_value = "new_leaf"

        result = gdrive._ensure_folder_path(["forecasts", "2026", "03"])

        assert result == "new_leaf"
        # Only the last part should trigger find+create
        assert mock_find.call_count == 1
        mock_create.assert_called_once_with("03", "cached_f2")

    @patch("energy_forecast.storage.gdrive.GoogleDriveStorage._upload_file")
    @patch("energy_forecast.storage.gdrive.GoogleDriveStorage._ensure_folder_path")
    def test_upload_job_artifacts(
        self,
        mock_ensure: MagicMock,
        mock_upload: MagicMock,
        gdrive: GoogleDriveStorage,
        tmp_path: Path,
    ) -> None:
        """Test uploading multiple files for a job."""
        mock_ensure.return_value = "leaf_folder_id"
        mock_upload.return_value = "uploaded_file_id"

        f1 = tmp_path / "historical.parquet"
        f1.write_text("data1")
        f2 = tmp_path / "forecast.parquet"
        f2.write_text("data2")

        ts = datetime(2026, 3, 7, 14, 34, tzinfo=TZ_ISTANBUL)
        result = gdrive.upload_job_artifacts(
            "job_123",
            {"historical.parquet": f1, "forecast.parquet": f2},
            created_at=ts,
        )

        assert result == {
            "historical.parquet": "uploaded_file_id",
            "forecast.parquet": "uploaded_file_id",
        }
        assert mock_upload.call_count == 2
        mock_ensure.assert_called_once_with(
            ["forecasts", "2026", "03", "07", "14-34_job_123"]
        )

    @patch("energy_forecast.storage.gdrive.GoogleDriveStorage._upload_file")
    @patch("energy_forecast.storage.gdrive.GoogleDriveStorage._ensure_folder_path")
    def test_upload_skips_missing_files(
        self,
        mock_ensure: MagicMock,
        mock_upload: MagicMock,
        gdrive: GoogleDriveStorage,
        tmp_path: Path,
    ) -> None:
        """Test that missing files are skipped gracefully."""
        mock_ensure.return_value = "leaf_folder_id"

        missing = tmp_path / "nonexistent.parquet"

        result = gdrive.upload_job_artifacts(
            "job_123", {"nonexistent.parquet": missing}
        )

        assert result == {}
        mock_upload.assert_not_called()

    @patch("energy_forecast.storage.gdrive.GoogleDriveStorage._upload_file")
    @patch("energy_forecast.storage.gdrive.GoogleDriveStorage._ensure_folder_path")
    def test_upload_handles_api_error(
        self,
        mock_ensure: MagicMock,
        mock_upload: MagicMock,
        gdrive: GoogleDriveStorage,
        tmp_path: Path,
    ) -> None:
        """Test that upload errors are caught and logged."""
        mock_ensure.return_value = "leaf_folder_id"
        mock_upload.side_effect = RuntimeError("API error")

        f1 = tmp_path / "test.parquet"
        f1.write_text("data")

        result = gdrive.upload_job_artifacts(
            "job_123", {"test.parquet": f1}
        )

        assert result == {}

    @patch("energy_forecast.storage.gdrive.GoogleDriveStorage._upload_file")
    @patch("energy_forecast.storage.gdrive.GoogleDriveStorage._ensure_folder_path")
    def test_upload_backup(
        self,
        mock_ensure: MagicMock,
        mock_upload: MagicMock,
        gdrive: GoogleDriveStorage,
        tmp_path: Path,
    ) -> None:
        """Test backup upload creates correct folder hierarchy."""
        mock_ensure.return_value = "backup_folder_id"
        mock_upload.return_value = "backup_file_id"

        backup = tmp_path / "energy_forecast_2026-03-07_14-30.sql.gz"
        backup.write_bytes(b"compressed_dump")

        ts = datetime(2026, 3, 7, 14, 30, tzinfo=TZ_ISTANBUL)
        result = gdrive.upload_backup(backup, ts=ts)

        assert result == "backup_file_id"
        mock_ensure.assert_called_once_with(
            ["backups", "2026", "03", "07", "14-30"]
        )
        mock_upload.assert_called_once_with(
            backup.name, backup, "backup_folder_id"
        )
