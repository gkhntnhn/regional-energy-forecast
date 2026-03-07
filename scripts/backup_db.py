"""Backup PostgreSQL database to a gzipped SQL dump.

Usage:
    uv run python scripts/backup_db.py

    # Or via Makefile:
    make db-backup

Requires DATABASE_URL_SYNC in .env (sync psycopg2 URL for pg_dump).
Optionally uploads to Google Drive if GDRIVE_CREDENTIALS_PATH is set.
"""

from __future__ import annotations

import gzip
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from loguru import logger

_TZ_ISTANBUL = ZoneInfo("Europe/Istanbul")


def backup_database(output_dir: Path | None = None) -> Path | None:
    """Create a gzipped pg_dump of the database.

    Args:
        output_dir: Directory for the dump file. Defaults to /tmp or data/backups/.

    Returns:
        Path to the gzipped dump file.
    """
    db_url = os.environ.get("DATABASE_URL_SYNC", "")
    if not db_url:
        logger.error("DATABASE_URL_SYNC not set")
        sys.exit(1)

    timestamp = datetime.now(tz=_TZ_ISTANBUL).strftime("%Y-%m-%d_%H-%M")
    if output_dir is None:
        output_dir = Path("data/backups")
    output_dir.mkdir(parents=True, exist_ok=True)

    dump_file = output_dir / f"energy_forecast_{timestamp}.sql"
    gz_file = output_dir / f"energy_forecast_{timestamp}.sql.gz"

    # pg_dump — try docker exec first (dev), then local pg_dump (production)
    logger.info("Running pg_dump...")
    container = os.environ.get("POSTGRES_CONTAINER", "regional-energy-forecast-db-1")

    docker_cmd = [
        "docker", "exec", container,
        "pg_dump", "-U", "forecast_user", "-d", "energy_forecast",
        "--no-owner", "--no-acl",
    ]
    local_cmd = [
        "pg_dump", db_url,
        "--no-owner", "--no-acl",
    ]

    for cmd in [docker_cmd, local_cmd]:
        result = subprocess.run(
            cmd, capture_output=True, timeout=300,
        )
        if result.returncode == 0:
            dump_file.write_bytes(result.stdout)
            break
    else:
        logger.error("pg_dump failed: {}", result.stderr.decode())
        sys.exit(1)

    # Gzip
    logger.info("Compressing dump...")
    with open(dump_file, "rb") as f_in, gzip.open(gz_file, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    dump_file.unlink()

    size_mb = gz_file.stat().st_size / (1024 * 1024)
    logger.info("Backup created: {} ({:.2f} MB)", gz_file, size_mb)

    # Upload to Google Drive if credentials are available
    creds_path = os.environ.get("GDRIVE_CREDENTIALS_PATH")
    folder_id = os.environ.get("GDRIVE_BACKUP_FOLDER_ID")
    if creds_path and folder_id:
        try:
            from energy_forecast.storage.gdrive import GoogleDriveStorage

            gdrive = GoogleDriveStorage(creds_path, folder_id)
            gdrive.upload_backup(gz_file)
            gz_file.unlink()
            logger.info("Local backup removed after upload")
            return None
        except Exception as e:
            logger.warning("GDrive upload failed (keeping local): {}", e)
    else:
        logger.info("GDrive not configured — backup kept locally")

    return gz_file


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    backup_database()
