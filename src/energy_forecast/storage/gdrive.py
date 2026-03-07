"""Google Drive storage for L3 artifact backup.

Supports two auth modes (auto-detected from credentials JSON):
- OAuth2 user flow: personal Google accounts (first run opens browser)
- Service account: Google Workspace with shared drives

All operations are synchronous — caller must wrap with asyncio.to_thread().
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar

from loguru import logger

from energy_forecast.utils import TZ_ISTANBUL


class GoogleDriveStorage:
    """Upload job artifacts to Google Drive for archival."""

    SCOPES: ClassVar[list[str]] = [
        "https://www.googleapis.com/auth/drive.file"
    ]

    def __init__(
        self, credentials_path: str, root_folder_id: str
    ) -> None:
        self._credentials_path = credentials_path
        self._root_folder_id = root_folder_id
        self._service: Any = None
        self._folder_cache: dict[str, str] = {}

    def _get_service(self) -> Any:
        """Lazy-init Google Drive API service.

        Auto-detects credential type from JSON file:
        - {"type": "service_account"} → service account flow
        - {"installed": ...} → OAuth2 desktop app flow
        """
        if self._service is not None:
            return self._service

        from googleapiclient.discovery import build

        creds_path = Path(self._credentials_path)
        with open(creds_path) as f:
            cred_data = json.load(f)

        if cred_data.get("type") == "service_account":
            creds = self._auth_service_account()
        else:
            creds = self._auth_oauth2(creds_path)

        self._service = build("drive", "v3", credentials=creds)
        return self._service

    def _auth_service_account(self) -> Any:
        """Authenticate via service account key file."""
        from google.oauth2.service_account import Credentials

        return Credentials.from_service_account_file(  # type: ignore[no-untyped-call]
            self._credentials_path, scopes=self.SCOPES
        )

    def _auth_oauth2(self, creds_path: Path) -> Any:
        """Authenticate via OAuth2 user consent flow.

        First run opens browser for authorization. Token is saved to
        ``credentials/gdrive_token.json`` for subsequent runs.
        """
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import (
            Credentials as UserCredentials,
        )
        from google_auth_oauthlib.flow import InstalledAppFlow  # type: ignore[import-untyped]

        token_path = creds_path.parent / "gdrive_token.json"
        creds: Any = None

        if token_path.exists():
            creds = UserCredentials.from_authorized_user_file(  # type: ignore[no-untyped-call]
                str(token_path), self.SCOPES
            )

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    str(creds_path), self.SCOPES
                )
                creds = flow.run_local_server(port=0)
                logger.info("OAuth2 authorization successful")

            token_path.write_text(creds.to_json())
            logger.info("Token saved to {}", token_path)

        return creds

    def upload_job_artifacts(
        self,
        job_id: str,
        files: dict[str, Path],
        created_at: datetime | None = None,
    ) -> dict[str, str]:
        """Upload forecast artifacts to GDrive.

        Structure: ``forecasts/YYYY/MM/DD/HH-MM_<job_id>/``

        Args:
            job_id: Job identifier.
            files: Mapping of filename -> local path.
            created_at: Job creation timestamp (for folder hierarchy).

        Returns:
            Mapping of filename -> GDrive file ID.
        """
        ts = created_at or datetime.now(tz=TZ_ISTANBUL)
        leaf = f"{ts.strftime('%H-%M')}_{job_id}"
        path_parts = [
            "forecasts",
            ts.strftime("%Y"),
            ts.strftime("%m"),
            ts.strftime("%d"),
            leaf,
        ]
        target_folder_id = self._ensure_folder_path(path_parts)

        uploaded: dict[str, str] = {}
        for name, path in files.items():
            if path.exists():
                try:
                    file_id = self._upload_file(
                        name, path, target_folder_id
                    )
                    uploaded[name] = file_id
                except Exception as e:
                    logger.warning(
                        "GDrive upload failed for {}: {}", name, e
                    )

        logger.info(
            "GDrive: uploaded {}/{} files to forecasts/{}/{}/{}/{}",
            len(uploaded),
            len(files),
            *path_parts[1:],
        )
        return uploaded

    def upload_backup(
        self, file_path: Path, ts: datetime | None = None
    ) -> str:
        """Upload a DB backup file to GDrive.

        Structure: ``backups/YYYY/MM/DD/HH-MM/``

        Args:
            file_path: Local path to the gzipped dump.
            ts: Backup timestamp (for folder hierarchy).

        Returns:
            GDrive file ID.
        """
        ts = ts or datetime.now(tz=TZ_ISTANBUL)
        path_parts = [
            "backups",
            ts.strftime("%Y"),
            ts.strftime("%m"),
            ts.strftime("%d"),
            ts.strftime("%H-%M"),
        ]
        target_folder_id = self._ensure_folder_path(path_parts)
        file_id = self._upload_file(
            file_path.name, file_path, target_folder_id
        )
        logger.info(
            "GDrive: backup uploaded to backups/{}/{}/{}/{}",
            *path_parts[1:],
        )
        return file_id

    def _ensure_folder_path(self, parts: list[str]) -> str:
        """Create nested folder chain, caching each level.

        Args:
            parts: Folder names from root, e.g. ["forecasts", "2026", "03", "07", "14-45_abc"].

        Returns:
            GDrive folder ID of the deepest (leaf) folder.
        """
        parent_id = self._root_folder_id

        for i, name in enumerate(parts):
            cache_key = "/".join(parts[: i + 1])
            if cache_key in self._folder_cache:
                parent_id = self._folder_cache[cache_key]
                continue

            # Search for existing folder
            folder_id = self._find_folder(name, parent_id)
            if folder_id is None:
                folder_id = self._create_folder(name, parent_id)

            self._folder_cache[cache_key] = folder_id
            parent_id = folder_id

        return parent_id

    def _find_folder(self, name: str, parent_id: str) -> str | None:
        """Find existing folder by name under parent."""
        service = self._get_service()
        safe_name = name.replace("'", "\\'")
        query = (
            f"name='{safe_name}' and "
            f"'{parent_id}' in parents and "
            f"mimeType='application/vnd.google-apps.folder' and "
            f"trashed=false"
        )
        results = (
            service.files()
            .list(q=query, fields="files(id)")
            .execute()
        )
        existing = results.get("files", [])
        if existing:
            return existing[0]["id"]  # type: ignore[no-any-return]
        return None

    def _create_folder(self, name: str, parent_id: str) -> str:
        """Create a folder in GDrive."""
        service = self._get_service()
        metadata = {
            "name": name,
            "mimeType": "application/vnd.google-apps.folder",
            "parents": [parent_id],
        }
        folder = (
            service.files()
            .create(body=metadata, fields="id")
            .execute()
        )
        folder_id: str = folder["id"]
        return folder_id

    def _upload_file(
        self, name: str, path: Path, folder_id: str
    ) -> str:
        """Upload a single file to a GDrive folder."""
        from googleapiclient.http import MediaFileUpload

        service = self._get_service()
        metadata = {"name": name, "parents": [folder_id]}
        media = MediaFileUpload(str(path), resumable=True)
        result = (
            service.files()
            .create(body=metadata, media_body=media, fields="id")
            .execute()
        )
        file_id: str = result["id"]
        return file_id
