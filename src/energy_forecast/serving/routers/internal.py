"""Internal scheduler endpoints (token-protected, sync response).

These endpoints exist for daily ingestion automation (Cloud Scheduler /
Windows Task Scheduler / cron). They are gated by the ``X-Internal-Token``
header (separate from the user-facing ``API_KEY``).

Endpoints
---------
POST /internal/sync-epias?days_back=N   — reconcile EPIAS market + generation
                                          parquet cache into DB tables
POST /internal/sync-weather?days_back=N — reconcile weather aggregate +
                                          15-grid parquet cache into DB tables

Design
------
Sync (blocking) response — daily ingestion finishes in 5-90s and the
scheduler/operator wants the result inline. Background-task pattern is
not needed here.

Idempotency — backed by ``ON CONFLICT (datetime, ...) DO UPDATE`` upserts in
the existing seed scripts. EPIAS path imports ``seed_db`` helpers in-thread.
Weather path shells out to ``seed_weather.py`` (CLI-only entry point) and
parses its loguru output for row counts; library-extract is a follow-up.
"""

from __future__ import annotations

import asyncio
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from loguru import logger
from pydantic import BaseModel, Field

from energy_forecast.utils import TZ_ISTANBUL

internal_router = APIRouter(prefix="/internal", tags=["internal"])


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class SyncResult(BaseModel):
    """Sync endpoint response payload."""

    sync_type: str = Field(description="Identifier: 'epias' | 'weather'")
    days_back: int = Field(description="Days requested for backfill window")
    rows_fetched: int = Field(description="Total rows pulled from upstream API")
    rows_upserted: int = Field(description="Total rows written to DB (post-conflict)")
    last_date: str | None = Field(
        default=None,
        description="ISO date of most recent row in result, or null if empty",
    )
    duration_ms: int = Field(description="Wall-clock duration in milliseconds")
    errors: list[str] = Field(
        default_factory=list,
        description="Non-fatal error messages — partial success allowed",
    )


# ---------------------------------------------------------------------------
# Helpers (run inside asyncio.to_thread — blocking)
# ---------------------------------------------------------------------------


def _ensure_scripts_importable() -> None:
    """Make ``scripts.ops.*`` import work from the API process.

    The ``scripts/`` directory is not part of the installed package; we add
    the project root (parents[4] from this file) to sys.path so the seed
    helpers can be re-used without duplicating their upsert logic here.
    """
    project_root = Path(__file__).resolve().parents[4]
    root_str = str(project_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


def _run_epias_db_sync(
    settings: Any,
) -> tuple[int, int, int, str | None]:
    """Blocking parquet -> DB upsert for EPIAS market + generation.

    Returns ``(market_files, db_market_upserts, db_gen_upserts, last_date)``.

    Scope: cheap, idempotent reconciliation between the on-disk yearly
    parquet cache and the ``epias_market`` / ``epias_generation`` tables
    via PostgreSQL ON CONFLICT (datetime) DO UPDATE.

    NOT in scope: pulling new data from the EPIAS API. That is a separate
    incremental-fetch concern handled today by ``scripts/data/backfill_epias.py``
    and tracked as a follow-up enhancement (will be wired in via a future
    ``EpiasClient.fetch_incremental`` method).
    """
    env = settings.env
    cfg = settings.epias_api

    if not env.database_url_sync:
        logger.warning(
            "DATABASE_URL_SYNC not set — sync-epias is a no-op (skip DB upsert)"
        )
        return 0, 0, 0, None

    cache_dir = Path(cfg.cache_dir)
    market_paths = sorted(cache_dir.glob("epias_market_*.parquet"))
    gen_paths = sorted(cache_dir.glob("epias_generation_*.parquet"))

    # Determine last_date from most recent market parquet (cheap, indexed read)
    import pandas as pd

    last_date: str | None = None
    if market_paths:
        df_last = pd.read_parquet(market_paths[-1], columns=["datetime"])
        if not df_last.empty:
            last_date = str(pd.to_datetime(df_last["datetime"]).max().date())

    _ensure_scripts_importable()
    from scripts.ops.seed_db import seed_epias_generation, seed_epias_market
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    engine = create_engine(env.database_url_sync, pool_pre_ping=True)
    session_factory = sessionmaker(bind=engine)

    db_market = 0
    db_gen = 0
    with session_factory() as session:
        db_market = seed_epias_market(session, market_paths)
        db_gen = seed_epias_generation(session, gen_paths)
        session.commit()

    engine.dispose()
    return len(market_paths), db_market, db_gen, last_date


async def _audit_log(
    request: Request,
    *,
    action: str,
    details: dict[str, Any],
) -> None:
    """Non-fatal audit log entry. Skips if DB-mode is inactive."""
    if not getattr(request.app.state, "use_db", False):
        return
    try:
        from energy_forecast.db.repositories.audit_repo import AuditRepository

        session_factory = request.app.state.session_factory
        async with session_factory() as session:
            repo = AuditRepository(session)
            await repo.log(
                action=action,
                user_email="scheduler",
                details=details,
            )
            await session.commit()
    except Exception as exc:
        logger.warning("Audit log failed (non-fatal) for action={}: {}", action, exc)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@internal_router.post("/sync-epias", response_model=SyncResult)
async def sync_epias(
    request: Request,
    days_back: int = Query(7, ge=1, le=90, description="Days to backfill from today"),
) -> SyncResult:
    """Sync EPIAS market + generation data into DB (idempotent).

    Designed for scheduled daily invocation (e.g. cron 06:00). Fetches the
    last ``days_back`` days from EPIAS via the cache-aware client and upserts
    into ``epias_market`` and ``epias_generation`` tables.

    Returns a SyncResult with row counts and timing. Partial failures are
    captured in ``errors[]`` rather than raising — the scheduler should
    treat HTTP 200 with non-empty errors as a soft warning.
    """
    started = time.monotonic()
    settings = request.app.state.settings
    env = settings.env

    if not env.epias_username or not env.epias_password:
        raise HTTPException(
            status_code=503,
            detail="EPIAS credentials not configured (EPIAS_USERNAME / EPIAS_PASSWORD)",
        )

    today = datetime.now(tz=TZ_ISTANBUL).date()

    errors: list[str] = []
    parquet_files = db_market = db_gen = 0
    last_date: str | None = None

    try:
        parquet_files, db_market, db_gen, last_date = await asyncio.to_thread(
            _run_epias_db_sync,
            settings,
        )
    except Exception as exc:
        # Redact: only type + first 200 chars (str(e) leak prevention, audit P0-3)
        msg = f"{type(exc).__name__}: {str(exc).split(chr(10))[0][:200]}"
        logger.opt(exception=True).error("sync-epias failed: {}", msg)
        errors.append(msg)

    duration_ms = int((time.monotonic() - started) * 1000)
    rows_upserted = db_market + db_gen

    await _audit_log(
        request,
        action="sync_epias",
        details={
            "days_back": days_back,
            "today": today.isoformat(),
            "parquet_files": parquet_files,
            "db_market_upserts": db_market,
            "db_generation_upserts": db_gen,
            "last_date": last_date,
            "duration_ms": duration_ms,
            "errors": errors,
        },
    )

    return SyncResult(
        sync_type="epias",
        days_back=days_back,
        rows_fetched=0,  # API fetch deferred to future incremental method
        rows_upserted=rows_upserted,
        last_date=last_date,
        duration_ms=duration_ms,
        errors=errors,
    )


def _run_weather_db_sync() -> tuple[int, int, str | None, str]:
    """Blocking parquet -> DB sync for weather aggregate + 15-grid backfill.

    Shells out to ``scripts/data/seed_weather.py`` (CLI-only entry today;
    library-extract is a follow-up cleanup). Parses its loguru log lines
    for aggregate and grid row counts.

    Returns ``(aggregate_rows, grid_rows, last_year, raw_log_tail)``. The
    raw log tail is included in the audit details for ops debuggability.
    """
    project_root = Path(__file__).resolve().parents[4]
    seed_script = project_root / "scripts" / "data" / "seed_weather.py"

    proc = subprocess.run(
        [sys.executable, str(seed_script)],
        cwd=str(project_root),
        capture_output=True,
        timeout=600,
        check=False,
        env={**os.environ},
    )

    log_text = (proc.stderr or b"").decode("utf-8", errors="ignore") + (
        proc.stdout or b""
    ).decode("utf-8", errors="ignore")

    if proc.returncode != 0:
        tail = "\n".join(log_text.splitlines()[-15:])
        raise RuntimeError(
            f"seed_weather.py exit={proc.returncode}; tail:\n{tail}"
        )

    aggregate_total = 0
    grid_total = 0
    last_year: int | None = None

    # Loguru emits with timestamp + level prefix, e.g.:
    #   "15:51:58 | INFO   |   2020 -> DB from parquet (35136 rows)"
    #   "15:51:58 | INFO   |   2020 -> grid DB (131760 rows, 15 points)"
    # Line-internal search (no ^ anchor) so we don't trip on the prefix.
    agg_re = re.compile(r"(\d{4})\s+->\s+DB from parquet\s+\((\d+)\s+rows")
    grid_re = re.compile(r"(\d{4})\s+->\s+grid DB\s+\((\d+)\s+rows")
    for raw_line in log_text.splitlines():
        # Strip ANSI color codes that loguru emits to terminals
        line = re.sub(r"\x1b\[[0-9;]*m", "", raw_line)
        m = agg_re.search(line)
        if m:
            year, count = int(m.group(1)), int(m.group(2))
            aggregate_total += count
            last_year = max(last_year or 0, year)
            continue
        m = grid_re.search(line)
        if m:
            grid_total += int(m.group(2))

    last_year_iso = str(last_year) if last_year else None
    log_tail = "\n".join(log_text.splitlines()[-10:])
    return aggregate_total, grid_total, last_year_iso, log_tail


@internal_router.post("/sync-weather", response_model=SyncResult)
async def sync_weather(
    request: Request,
    days_back: int = Query(2, ge=1, le=14, description="Days of T-N actuals to backfill"),
) -> SyncResult:
    """Sync weather aggregate + 15-grid parquet cache into DB.

    Designed for scheduled daily invocation (e.g. cron 06:30). Re-runs the
    existing ``seed_weather.py`` helper (parquet-first, API fallback per
    its own logic), then captures aggregate and grid row counts for audit.

    Idempotent: ``upsert_weather`` performs ON CONFLICT (datetime, lat,
    lon, source) DO UPDATE under the hood.
    """
    started = time.monotonic()
    today = datetime.now(tz=TZ_ISTANBUL).date()

    errors: list[str] = []
    aggregate_total = grid_total = 0
    last_year: str | None = None
    log_tail = ""

    try:
        aggregate_total, grid_total, last_year, log_tail = await asyncio.to_thread(
            _run_weather_db_sync,
        )
    except Exception as exc:
        msg = f"{type(exc).__name__}: {str(exc).split(chr(10))[0][:200]}"
        logger.opt(exception=True).error("sync-weather failed: {}", msg)
        errors.append(msg)

    duration_ms = int((time.monotonic() - started) * 1000)
    rows_upserted = aggregate_total + grid_total

    await _audit_log(
        request,
        action="sync_weather",
        details={
            "days_back": days_back,
            "today": today.isoformat(),
            "aggregate_rows": aggregate_total,
            "grid_rows": grid_total,
            "last_year": last_year,
            "duration_ms": duration_ms,
            "log_tail": log_tail,
            "errors": errors,
        },
    )

    return SyncResult(
        sync_type="weather",
        days_back=days_back,
        rows_fetched=0,  # API fetch deferred — seed_weather decides parquet vs API internally
        rows_upserted=rows_upserted,
        last_date=last_year,
        duration_ms=duration_ms,
        errors=errors,
    )
