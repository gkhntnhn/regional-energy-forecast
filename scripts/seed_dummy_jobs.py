"""Seed DB with realistic dummy job/prediction data for admin panel testing.

Usage:
    uv run python scripts/seed_dummy_jobs.py              # 90 days
    uv run python scripts/seed_dummy_jobs.py --days 30    # 30 days
    uv run python scripts/seed_dummy_jobs.py --dry-run    # preview only
"""

from __future__ import annotations

import argparse
import random
import uuid
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from loguru import logger

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Seed dummy jobs for admin panel testing")
    p.add_argument("--days", type=int, default=90, help="Number of days to generate")
    p.add_argument("--dry-run", action="store_true", help="Preview without writing")
    return p.parse_args()


def _realistic_consumption(hour: int, month: int) -> float:
    """Generate realistic hourly consumption (MWh) for Uludag region."""
    # Base load ~1100 MWh, peak ~1500 MWh
    base = 1100.0
    # Daily pattern: low at night, peak at 11-13 and 19-21
    hour_factor = {
        0: 0.82, 1: 0.78, 2: 0.75, 3: 0.73, 4: 0.72, 5: 0.74,
        6: 0.80, 7: 0.88, 8: 0.95, 9: 1.02, 10: 1.08, 11: 1.12,
        12: 1.10, 13: 1.08, 14: 1.05, 15: 1.03, 16: 1.05, 17: 1.10,
        18: 1.15, 19: 1.18, 20: 1.15, 21: 1.08, 22: 0.98, 23: 0.90,
    }
    # Seasonal: higher in winter (heating), summer (cooling)
    month_factor = {
        1: 1.15, 2: 1.12, 3: 1.05, 4: 0.95, 5: 0.90, 6: 0.95,
        7: 1.05, 8: 1.08, 9: 0.98, 10: 0.92, 11: 1.00, 12: 1.10,
    }
    val = base * hour_factor[hour] * month_factor[month]
    val += random.gauss(0, 30)  # noise
    return round(max(val, 500.0), 1)


def _realistic_weather(hour: int, month: int) -> dict:
    """Generate realistic weather values for Istanbul region."""
    # Monthly avg temps (Celsius)
    month_temp = {
        1: 5, 2: 6, 3: 9, 4: 14, 5: 19, 6: 24,
        7: 27, 8: 27, 9: 23, 10: 17, 11: 12, 12: 7,
    }
    base_temp = month_temp[month]
    # Daily cycle: coldest at 5-6, warmest at 14-15
    hour_offset = -3 * np.cos(2 * np.pi * (hour - 14) / 24)
    temp = base_temp + hour_offset + random.gauss(0, 1.5)

    return {
        "temperature_2m": round(temp, 1),
        "apparent_temperature": round(temp - 2 + random.gauss(0, 1), 1),
        "relative_humidity_2m": round(max(30, min(95, 65 + random.gauss(0, 12))), 1),
        "dew_point_2m": round(temp - 8 + random.gauss(0, 2), 1),
        "precipitation": round(max(0, random.gauss(0.1, 0.3)), 2),
        "snow_depth": 0.0 if month > 3 else round(max(0, random.gauss(0, 2)), 1),
        "surface_pressure": round(1013 + random.gauss(0, 5), 1),
        "wind_speed_10m": round(max(0, 8 + random.gauss(0, 4)), 1),
        "wind_direction_10m": round(random.uniform(0, 360), 0),
        "shortwave_radiation": round(max(0, 400 * max(0, np.sin(np.pi * (hour - 6) / 12))), 1)
        if 6 <= hour <= 18
        else 0.0,
        "weather_code": random.choice([0, 1, 2, 3, 45, 51, 61, 71, 80, 95]),
        "wth_hdd": round(max(0, 18 - temp), 1),
        "wth_cdd": round(max(0, temp - 22), 1),
    }


def main() -> None:
    import os

    from sqlalchemy import create_engine, text

    args = parse_args()

    db_url = os.environ.get("DATABASE_URL_SYNC", "")
    if not db_url:
        logger.error("DATABASE_URL_SYNC not set")
        return

    engine = create_engine(db_url)

    # Time range: today - N days to today
    from energy_forecast.utils import TZ_ISTANBUL

    end_date = datetime.now(tz=TZ_ISTANBUL).replace(hour=0, minute=0, second=0, microsecond=0)
    start_date = end_date - timedelta(days=args.days)

    logger.info("Generating {} days of dummy data: {} to {}", args.days, start_date.date(), end_date.date())

    jobs = []
    predictions = []
    weather_snaps = []
    weather_actuals = []
    actual_hours_seen: set[datetime] = set()  # deduplicate actual weather per hour
    audit_logs = []

    current = start_date
    job_count = 0
    while current < end_date:
        job_id = uuid.uuid4().hex[:12]
        created = current + timedelta(hours=random.randint(7, 10), minutes=random.randint(0, 59),
                                       seconds=random.randint(0, 59),
                                       microseconds=random.randint(0, 999999))

        # 92% success, 5% failed, 3% archived
        r = random.random()
        if r < 0.92:
            status = "completed"
            duration = timedelta(seconds=random.randint(8, 25))
            completed = created + duration
            progress = "Sonuclar gonderildi"
            error = None
            email_status = "sent"
        elif r < 0.97:
            status = "failed"
            completed = created + timedelta(seconds=random.randint(3, 10))
            progress = None
            error = random.choice([
                "Prediction failed: CatBoost model error",
                "Feature pipeline failed: missing column",
                "Excel parse error: invalid format",
            ])
            email_status = "failed"
        else:
            status = "archived"
            completed = created + timedelta(seconds=random.randint(8, 20))
            progress = "Sonuclar gonderildi"
            error = None
            email_status = "sent"

        file_stem = created.strftime("%d-%m-%Y_%H-%M-%S") + f"-{created.microsecond // 1000:03d}"

        jobs.append({
            "id": job_id,
            "email": "demo@example.com",
            "status": status,
            "progress": progress,
            "error": error,
            "excel_path": f"data/uploads/{file_stem}_Input.xlsx",
            "file_stem": file_stem,
            "result_path": f"data/outputs/{file_stem}_Forecast.xlsx" if status == "completed" else None,
            "created_at": created,
            "completed_at": completed,
            "email_status": email_status,
            "email_attempts": 1 if email_status == "sent" else 0,
        })

        # Generate 48 predictions for completed/archived jobs
        if status in ("completed", "archived"):
            forecast_start = current + timedelta(days=1)  # T+0 day
            for h in range(48):
                fc_dt = forecast_start + timedelta(hours=h)
                actual = _realistic_consumption(fc_dt.hour, fc_dt.month)
                predicted = actual * (1 + random.gauss(0, 0.03))  # ~3% MAPE
                error_pct = abs(predicted - actual) / actual * 100

                predictions.append({
                    "job_id": job_id,
                    "forecast_dt": fc_dt,
                    "consumption_mwh": round(predicted, 1),
                    "period": "intraday" if h < 24 else "day_ahead",
                    "model_source": "ensemble",
                    "created_at": created,
                    "actual_mwh": round(actual, 1),
                    "error_pct": round(error_pct, 2),
                    "matched_at": completed + timedelta(hours=24),
                })

            # Weather snapshots (48 hours) — forecast + actual pairs
            for h in range(48):
                fc_dt = forecast_start + timedelta(hours=h)
                w = _realistic_weather(fc_dt.hour, fc_dt.month)

                # Forecast snapshot (tied to job)
                w_fc = {**w}
                w_fc["job_id"] = job_id
                w_fc["forecast_dt"] = fc_dt
                w_fc["fetched_at"] = created
                w_fc["is_actual"] = False
                weather_snaps.append(w_fc)

                # Actual snapshot — one per hour (deduplicated across jobs)
                if fc_dt not in actual_hours_seen:
                    actual_hours_seen.add(fc_dt)
                    w_act = {**w}
                    # Actuals differ slightly from forecasts
                    w_act["temperature_2m"] = round(w["temperature_2m"] + random.gauss(0, 0.8), 1)
                    w_act["apparent_temperature"] = round(w["apparent_temperature"] + random.gauss(0, 0.9), 1)
                    w_act["wind_speed_10m"] = round(max(0, w["wind_speed_10m"] + random.gauss(0, 1.5)), 1)
                    w_act["shortwave_radiation"] = round(max(0, w["shortwave_radiation"] + random.gauss(0, 20)), 1)
                    w_act["precipitation"] = round(max(0, w["precipitation"] + random.gauss(0, 0.1)), 2)
                    w_act["job_id"] = None
                    w_act["forecast_dt"] = fc_dt
                    w_act["fetched_at"] = fc_dt  # actual fetched at observation time
                    w_act["is_actual"] = True
                    weather_actuals.append(w_act)

        # Audit log
        audit_logs.append({
            "action": "predict_request",
            "user_email": "demo@example.com",
            "ip_address": "127.0.0.1",
            "details": f'{{"job_id": "{job_id}", "file_name": "Input.xlsx"}}',
            "created_at": created,
        })

        # Occasional drift alert
        if random.random() < 0.08:
            audit_logs.append({
                "action": "drift_mape",
                "user_email": None,
                "ip_address": None,
                "details": f'{{"mape": {round(random.uniform(5, 12), 1)}, "threshold": 5.0}}',
                "created_at": created + timedelta(hours=1),
            })

        job_count += 1
        current += timedelta(days=1)

    logger.info(
        "Generated: {} jobs, {} predictions, {} weather fc + {} actuals, {} audit logs",
        len(jobs), len(predictions), len(weather_snaps), len(weather_actuals), len(audit_logs),
    )

    if args.dry_run:
        logger.info("Dry run — no data written")
        return

    # Bulk insert
    with engine.begin() as conn:
        # Jobs
        for j in jobs:
            conn.execute(text("""
                INSERT INTO jobs (id, email, status, progress, error, excel_path, file_stem,
                    result_path, created_at, completed_at, email_status, email_attempts)
                VALUES (:id, :email, :status, :progress, :error, :excel_path, :file_stem,
                    :result_path, :created_at, :completed_at, :email_status, :email_attempts)
            """), j)
        logger.info("Inserted {} jobs", len(jobs))

        # Predictions
        for batch_start in range(0, len(predictions), 500):
            batch = predictions[batch_start:batch_start + 500]
            for p in batch:
                conn.execute(text("""
                    INSERT INTO predictions (job_id, forecast_dt, consumption_mwh, period,
                        model_source, created_at, actual_mwh, error_pct, matched_at)
                    VALUES (:job_id, :forecast_dt, :consumption_mwh, :period,
                        :model_source, :created_at, :actual_mwh, :error_pct, :matched_at)
                """), p)
        logger.info("Inserted {} predictions", len(predictions))

        # Weather snapshots
        for batch_start in range(0, len(weather_snaps), 500):
            batch = weather_snaps[batch_start:batch_start + 500]
            for w in batch:
                conn.execute(text("""
                    INSERT INTO weather_snapshots (job_id, forecast_dt, fetched_at, is_actual,
                        temperature_2m, apparent_temperature, relative_humidity_2m, dew_point_2m,
                        precipitation, snow_depth, surface_pressure, wind_speed_10m,
                        wind_direction_10m, shortwave_radiation, weather_code, wth_hdd, wth_cdd)
                    VALUES (:job_id, :forecast_dt, :fetched_at, :is_actual,
                        :temperature_2m, :apparent_temperature, :relative_humidity_2m, :dew_point_2m,
                        :precipitation, :snow_depth, :surface_pressure, :wind_speed_10m,
                        :wind_direction_10m, :shortwave_radiation, :weather_code, :wth_hdd, :wth_cdd)
                """), w)
        logger.info("Inserted {} weather forecast snapshots", len(weather_snaps))

        # Weather actuals
        for batch_start in range(0, len(weather_actuals), 500):
            batch = weather_actuals[batch_start:batch_start + 500]
            for w in batch:
                conn.execute(text("""
                    INSERT INTO weather_snapshots (job_id, forecast_dt, fetched_at, is_actual,
                        temperature_2m, apparent_temperature, relative_humidity_2m, dew_point_2m,
                        precipitation, snow_depth, surface_pressure, wind_speed_10m,
                        wind_direction_10m, shortwave_radiation, weather_code, wth_hdd, wth_cdd)
                    VALUES (:job_id, :forecast_dt, :fetched_at, :is_actual,
                        :temperature_2m, :apparent_temperature, :relative_humidity_2m, :dew_point_2m,
                        :precipitation, :snow_depth, :surface_pressure, :wind_speed_10m,
                        :wind_direction_10m, :shortwave_radiation, :weather_code, :wth_hdd, :wth_cdd)
                """), w)
        logger.info("Inserted {} weather actual snapshots", len(weather_actuals))

        # Audit logs
        for a in audit_logs:
            conn.execute(text("""
                INSERT INTO audit_logs (action, user_email, ip_address, details, created_at)
                VALUES (:action, :user_email, :ip_address, :details, :created_at)
            """), a)
        logger.info("Inserted {} audit logs", len(audit_logs))

    logger.info("Done! {} days of dummy data seeded.", args.days)


if __name__ == "__main__":
    main()
