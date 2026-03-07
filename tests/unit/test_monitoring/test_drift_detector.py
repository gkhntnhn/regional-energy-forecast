"""Tests for model drift detection."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from energy_forecast.db.models import JobModel, PredictionModel
from energy_forecast.monitoring.drift_detector import (
    DriftConfig,
    _compute_trend,
    check_model_drift,
)
from energy_forecast.utils import TZ_ISTANBUL


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now() -> datetime:
    return datetime.now(tz=TZ_ISTANBUL)


async def _seed_predictions(
    session: AsyncSession,
    *,
    count: int = 48,
    error_pct: float = 3.0,
    consumption_mwh: float = 1000.0,
    actual_mwh: float = 1000.0,
    days_ago: int = 1,
    model_source: str = "ensemble",
) -> None:
    """Seed matched predictions for drift detection tests."""
    job = JobModel(
        id=f"drift_{days_ago}_{count}",
        email="test@example.com",
        status="completed",
        excel_path="/tmp/test.xlsx",
        file_stem="stem",
        email_status="pending",
    )
    session.add(job)
    await session.flush()

    base_dt = _now() - timedelta(days=days_ago)
    for i in range(count):
        pred = PredictionModel(
            job_id=job.id,
            forecast_dt=base_dt + timedelta(hours=i),
            consumption_mwh=consumption_mwh,
            actual_mwh=actual_mwh,
            error_pct=error_pct,
            period="day_ahead",
            model_source=model_source,
            matched_at=_now(),
        )
        session.add(pred)
    await session.flush()


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestDriftConfig:
    """Tests for DriftConfig."""

    def test_defaults(self) -> None:
        cfg = DriftConfig()
        assert cfg.mape_threshold_warning == 5.0
        assert cfg.mape_threshold_critical == 8.0
        assert cfg.min_samples == 24

    def test_from_dict(self) -> None:
        data = {
            "enabled": False,
            "mape_threshold_warning": 4.0,
            "min_samples": 10,
            "unknown_key": "ignored",
        }
        cfg = DriftConfig.from_dict(data)
        assert cfg.enabled is False
        assert cfg.mape_threshold_warning == 4.0
        assert cfg.min_samples == 10


# ---------------------------------------------------------------------------
# MAPE threshold
# ---------------------------------------------------------------------------


class TestMapeThreshold:
    """Tests for absolute MAPE threshold detection."""

    @pytest.mark.asyncio
    async def test_no_data_no_alert(self, db_session: AsyncSession) -> None:
        """No matched predictions → no alerts."""
        alerts = await check_model_drift(db_session)
        assert alerts == []

    @pytest.mark.asyncio
    async def test_below_threshold(self, db_session: AsyncSession) -> None:
        """MAPE below warning → no alert."""
        await _seed_predictions(db_session, count=48, error_pct=3.0)
        alerts = await check_model_drift(db_session)
        assert len(alerts) == 0

    @pytest.mark.asyncio
    async def test_warning_threshold(self, db_session: AsyncSession) -> None:
        """MAPE > 5% → warning alert."""
        await _seed_predictions(db_session, count=48, error_pct=6.0)
        alerts = await check_model_drift(db_session)
        mape_alerts = [a for a in alerts if a.alert_type == "mape_threshold"]
        assert len(mape_alerts) == 1
        assert mape_alerts[0].severity == "warning"
        assert mape_alerts[0].current_value == pytest.approx(6.0)

    @pytest.mark.asyncio
    async def test_critical_threshold(self, db_session: AsyncSession) -> None:
        """MAPE > 8% → critical alert."""
        await _seed_predictions(db_session, count=48, error_pct=9.0)
        alerts = await check_model_drift(db_session)
        mape_alerts = [a for a in alerts if a.alert_type == "mape_threshold"]
        assert len(mape_alerts) == 1
        assert mape_alerts[0].severity == "critical"

    @pytest.mark.asyncio
    async def test_insufficient_samples(
        self, db_session: AsyncSession
    ) -> None:
        """Less than min_samples → no alert even if MAPE high."""
        await _seed_predictions(db_session, count=10, error_pct=9.0)
        cfg = DriftConfig(min_samples=24)
        alerts = await check_model_drift(db_session, config=cfg)
        mape_alerts = [a for a in alerts if a.alert_type == "mape_threshold"]
        assert len(mape_alerts) == 0

    @pytest.mark.asyncio
    async def test_non_ensemble_ignored(
        self, db_session: AsyncSession
    ) -> None:
        """Only ensemble predictions are checked."""
        await _seed_predictions(
            db_session, count=48, error_pct=9.0, model_source="catboost"
        )
        alerts = await check_model_drift(db_session)
        assert len(alerts) == 0


# ---------------------------------------------------------------------------
# MAPE trend
# ---------------------------------------------------------------------------


class TestMapeTrend:
    """Tests for weekly MAPE trend detection."""

    @pytest.mark.asyncio
    async def test_increasing_trend(self, db_session: AsyncSession) -> None:
        """Upward trend → warning alert."""
        # Seed 4 weeks of data with increasing error
        for week in range(4):
            error = 2.0 + week * 1.0  # 2%, 3%, 4%, 5%
            await _seed_predictions(
                db_session,
                count=48,
                error_pct=error,
                days_ago=7 * (4 - week),
            )

        cfg = DriftConfig(
            mape_threshold_warning=99.0,  # disable MAPE threshold
            mape_trend_threshold=0.5,
        )
        alerts = await check_model_drift(db_session, config=cfg)
        trend_alerts = [a for a in alerts if a.alert_type == "mape_trend"]
        assert len(trend_alerts) == 1
        assert trend_alerts[0].severity == "warning"
        assert trend_alerts[0].current_value > 0.5

    @pytest.mark.asyncio
    async def test_stable_no_trend(self, db_session: AsyncSession) -> None:
        """Stable MAPE → no trend alert."""
        for week in range(4):
            await _seed_predictions(
                db_session,
                count=48,
                error_pct=3.0,
                days_ago=7 * (4 - week),
            )

        cfg = DriftConfig(mape_threshold_warning=99.0)
        alerts = await check_model_drift(db_session, config=cfg)
        trend_alerts = [a for a in alerts if a.alert_type == "mape_trend"]
        assert len(trend_alerts) == 0


# ---------------------------------------------------------------------------
# Bias shift
# ---------------------------------------------------------------------------


class TestBiasShift:
    """Tests for systematic over/under-prediction detection."""

    @pytest.mark.asyncio
    async def test_over_prediction_bias(
        self, db_session: AsyncSession
    ) -> None:
        """Systematic over-prediction → bias alert."""
        # Predicted 1050, actual 1000 → +5% bias
        await _seed_predictions(
            db_session,
            count=48,
            consumption_mwh=1050.0,
            actual_mwh=1000.0,
            error_pct=5.0,
        )
        cfg = DriftConfig(
            mape_threshold_warning=99.0,  # disable MAPE threshold
            bias_threshold=3.0,
        )
        alerts = await check_model_drift(db_session, config=cfg)
        bias_alerts = [a for a in alerts if a.alert_type == "bias_shift"]
        assert len(bias_alerts) == 1
        assert bias_alerts[0].current_value > 0  # positive = over-prediction
        assert "fazla" in bias_alerts[0].message

    @pytest.mark.asyncio
    async def test_under_prediction_bias(
        self, db_session: AsyncSession
    ) -> None:
        """Systematic under-prediction → bias alert."""
        # Predicted 950, actual 1000 → -5% bias
        await _seed_predictions(
            db_session,
            count=48,
            consumption_mwh=950.0,
            actual_mwh=1000.0,
            error_pct=5.0,
        )
        cfg = DriftConfig(
            mape_threshold_warning=99.0,
            bias_threshold=3.0,
        )
        alerts = await check_model_drift(db_session, config=cfg)
        bias_alerts = [a for a in alerts if a.alert_type == "bias_shift"]
        assert len(bias_alerts) == 1
        assert bias_alerts[0].current_value < 0  # negative = under-prediction
        assert "eksik" in bias_alerts[0].message

    @pytest.mark.asyncio
    async def test_no_bias(self, db_session: AsyncSession) -> None:
        """Balanced predictions → no bias alert."""
        await _seed_predictions(
            db_session,
            count=48,
            consumption_mwh=1000.0,
            actual_mwh=1000.0,
            error_pct=0.0,
        )
        cfg = DriftConfig(mape_threshold_warning=99.0)
        alerts = await check_model_drift(db_session, config=cfg)
        bias_alerts = [a for a in alerts if a.alert_type == "bias_shift"]
        assert len(bias_alerts) == 0


# ---------------------------------------------------------------------------
# Disabled / Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge case tests."""

    @pytest.mark.asyncio
    async def test_disabled(self, db_session: AsyncSession) -> None:
        """Disabled config → no alerts."""
        await _seed_predictions(db_session, count=48, error_pct=9.0)
        cfg = DriftConfig(enabled=False)
        alerts = await check_model_drift(db_session, config=cfg)
        assert alerts == []

    def test_compute_trend_empty(self) -> None:
        assert _compute_trend([]) == 0.0

    def test_compute_trend_single(self) -> None:
        assert _compute_trend([3.0]) == 0.0

    def test_compute_trend_flat(self) -> None:
        assert _compute_trend([3.0, 3.0, 3.0, 3.0]) == pytest.approx(0.0)

    def test_compute_trend_increasing(self) -> None:
        slope = _compute_trend([1.0, 2.0, 3.0, 4.0])
        assert slope == pytest.approx(1.0)
