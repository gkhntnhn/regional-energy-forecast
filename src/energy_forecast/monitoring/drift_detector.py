"""Model drift detection using production MAPE data.

Analyzes prediction-actual matches from the ``predictions`` table
to detect three types of drift:

1. **Absolute MAPE threshold** — recent average exceeds warning/critical levels
2. **MAPE trend** — weekly MAPE shows sustained upward slope
3. **Bias shift** — systematic over/under-prediction (signed MBE)

All queries use Python ``timedelta`` for date arithmetic (SQLite-compatible).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from loguru import logger
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from energy_forecast.db.models import PredictionModel
from energy_forecast.utils import TZ_ISTANBUL

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DriftConfig:
    """Drift detection thresholds (loaded from configs/monitoring.yaml)."""

    enabled: bool = True
    mape_threshold_warning: float = 5.0
    mape_threshold_critical: float = 8.0
    mape_trend_threshold: float = 0.5
    bias_threshold: float = 3.0
    lookback_days: int = 7
    trend_weeks: int = 4
    min_samples: int = 24
    cooldown_hours: int = 24
    email_on_warning: bool = False
    admin_email: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DriftConfig:
        """Create from YAML dict (configs/monitoring.yaml → drift_detection)."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Alert
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DriftAlert:
    """A single drift detection alert."""

    alert_type: str  # "mape_threshold" | "mape_trend" | "bias_shift"
    severity: str  # "warning" | "critical"
    current_value: float
    threshold: float
    message: str
    window_days: int


# ---------------------------------------------------------------------------
# Core detection
# ---------------------------------------------------------------------------


async def check_model_drift(
    session: AsyncSession,
    config: DriftConfig | None = None,
) -> list[DriftAlert]:
    """Run all drift checks and return any triggered alerts.

    Args:
        session: Async DB session with access to predictions table.
        config: Drift thresholds. Uses defaults if not provided.

    Returns:
        List of triggered alerts (may be empty).
    """
    cfg = config or DriftConfig()
    if not cfg.enabled:
        return []

    alerts: list[DriftAlert] = []

    # 1. Absolute MAPE threshold
    recent_mape, sample_count = await _get_recent_mape(
        session, days=cfg.lookback_days
    )
    if (
        recent_mape is not None
        and sample_count >= cfg.min_samples
        and recent_mape > cfg.mape_threshold_warning
    ):
        alerts.append(
            DriftAlert(
                alert_type="mape_threshold",
                severity=(
                    "critical"
                    if recent_mape > cfg.mape_threshold_critical
                    else "warning"
                ),
                current_value=recent_mape,
                threshold=cfg.mape_threshold_warning,
                message=(
                    f"Son {cfg.lookback_days} gun production MAPE: "
                    f"%{recent_mape:.1f} (threshold: %{cfg.mape_threshold_warning})"
                ),
                window_days=cfg.lookback_days,
            )
        )

    # 2. MAPE trend (weekly slope)
    weekly_mapes = await _get_weekly_mapes(session, weeks=cfg.trend_weeks)
    if len(weekly_mapes) >= 3:
        trend = _compute_trend(weekly_mapes)
        if trend > cfg.mape_trend_threshold:
            alerts.append(
                DriftAlert(
                    alert_type="mape_trend",
                    severity="warning",
                    current_value=trend,
                    threshold=cfg.mape_trend_threshold,
                    message=(
                        f"MAPE haftalik artis trendi: "
                        f"+%{trend:.2f}/hafta (son {cfg.trend_weeks} hafta)"
                    ),
                    window_days=cfg.trend_weeks * 7,
                )
            )

    # 3. Bias shift (signed MBE)
    recent_bias, bias_count = await _get_recent_bias(
        session, days=cfg.lookback_days
    )
    if (
        recent_bias is not None
        and bias_count >= cfg.min_samples
        and abs(recent_bias) > cfg.bias_threshold
    ):
        direction = "fazla" if recent_bias > 0 else "eksik"
        alerts.append(
            DriftAlert(
                alert_type="bias_shift",
                severity="warning",
                current_value=recent_bias,
                threshold=cfg.bias_threshold,
                message=(
                    f"Sistematik {direction} tahmin: "
                    f"ortalama %{abs(recent_bias):.1f} sapma"
                ),
                window_days=cfg.lookback_days,
            )
        )

    return alerts


# ---------------------------------------------------------------------------
# Helper queries
# ---------------------------------------------------------------------------


async def _get_recent_mape(
    session: AsyncSession, days: int
) -> tuple[float | None, int]:
    """Average absolute error_pct and sample count for last N days.

    Returns:
        (average_mape, sample_count) — both None/0 if no data.
    """
    cutoff = datetime.now(tz=TZ_ISTANBUL) - timedelta(days=days)
    result = await session.execute(
        select(
            func.avg(PredictionModel.error_pct),
            func.count(PredictionModel.id),
        )
        .where(PredictionModel.actual_mwh.is_not(None))
        .where(PredictionModel.model_source == "ensemble")
        .where(PredictionModel.forecast_dt >= cutoff)
    )
    row = result.one()
    avg_val = row[0]
    count_val = int(row[1]) if row[1] else 0
    return (float(avg_val) if avg_val is not None else None, count_val)


async def _get_weekly_mapes(
    session: AsyncSession, weeks: int
) -> list[float]:
    """Weekly MAPE values for last N weeks.

    Uses Python-side grouping instead of SQL date_trunc for SQLite compatibility.
    """
    cutoff = datetime.now(tz=TZ_ISTANBUL) - timedelta(weeks=weeks)
    result = await session.execute(
        select(
            PredictionModel.forecast_dt,
            PredictionModel.error_pct,
        )
        .where(PredictionModel.actual_mwh.is_not(None))
        .where(PredictionModel.model_source == "ensemble")
        .where(PredictionModel.forecast_dt >= cutoff)
        .order_by(PredictionModel.forecast_dt)
    )
    rows = result.all()
    if not rows:
        return []

    # Group by ISO week in Python (SQLite-compatible)
    weekly: dict[tuple[int, int], list[float]] = {}
    for forecast_dt, error_pct in rows:
        if error_pct is None:
            continue
        dt = forecast_dt
        iso_year, iso_week, _ = dt.isocalendar()
        key = (iso_year, iso_week)
        weekly.setdefault(key, []).append(float(error_pct))

    # Average per week, sorted chronologically
    sorted_weeks = sorted(weekly.keys())
    return [
        sum(weekly[k]) / len(weekly[k])
        for k in sorted_weeks
        if weekly[k]
    ]


async def _get_recent_bias(
    session: AsyncSession, days: int
) -> tuple[float | None, int]:
    """Mean Bias Error (%) for last N days.

    Positive = over-prediction, negative = under-prediction.
    Computed from raw columns (not error_pct which is absolute).
    """
    cutoff = datetime.now(tz=TZ_ISTANBUL) - timedelta(days=days)
    result = await session.execute(
        select(
            func.avg(
                (PredictionModel.consumption_mwh - PredictionModel.actual_mwh)
                / func.nullif(PredictionModel.actual_mwh, 0)
                * 100
            ),
            func.count(PredictionModel.id),
        )
        .where(PredictionModel.actual_mwh.is_not(None))
        .where(PredictionModel.model_source == "ensemble")
        .where(PredictionModel.forecast_dt >= cutoff)
    )
    row = result.one()
    avg_val = row[0]
    count_val = int(row[1]) if row[1] else 0
    return (float(avg_val) if avg_val is not None else None, count_val)


def _compute_trend(values: list[float]) -> float:
    """Simple linear regression slope (weekly increase rate)."""
    n = len(values)
    if n < 2:
        return 0.0
    x = list(range(n))
    x_mean = sum(x) / n
    y_mean = sum(values) / n
    numerator = sum(
        (xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, values, strict=True)
    )
    denominator = sum((xi - x_mean) ** 2 for xi in x)
    if denominator == 0:
        return 0.0
    slope = numerator / denominator
    logger.debug("MAPE trend: {} points, slope={:.4f}", n, slope)
    return slope
