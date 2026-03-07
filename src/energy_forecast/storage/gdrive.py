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
        self._month_cache: dict[str, str] = {}

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
        self, job_id: str, files: dict[str, Path]
    ) -> dict[str, str]:
        """Upload multiple files to GDrive under month/job_id folder.

        Args:
            job_id: Job identifier (used as subfolder name).
            files: Mapping of filename -> local path.

        Returns:
            Mapping of filename -> GDrive file ID.
        """
        month = datetime.now(tz=TZ_ISTANBUL).strftime("%Y-%m")
        month_folder_id = self._get_or_create_month_folder(month)
        job_folder_id = self._create_folder(
            job_id, month_folder_id
        )

        uploaded: dict[str, str] = {}
        for name, path in files.items():
            if path.exists():
                try:
                    file_id = self._upload_file(
                        name, path, job_folder_id
                    )
                    uploaded[name] = file_id
                except Exception as e:
                    logger.warning(
                        "GDrive upload failed for {}: {}", name, e
                    )

        logger.info(
            "GDrive: uploaded {}/{} files for job {}",
            len(uploaded),
            len(files),
            job_id,
        )
        return uploaded

    def _get_or_create_month_folder(self, month: str) -> str:
        """Get or create month folder (e.g. '2026-03')."""
        if month in self._month_cache:
            return self._month_cache[month]

        service = self._get_service()

        # Search for existing folder
        query = (
            f"name='{month}' and "
            f"'{self._root_folder_id}' in parents and "
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
            folder_id: str = existing[0]["id"]
        else:
            folder_id = self._create_folder(
                month, self._root_folder_id
            )

        self._month_cache[month] = folder_id
        return folder_id

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
