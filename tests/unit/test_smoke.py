"""Smoke tests to verify project skeleton imports work."""

from __future__ import annotations


def test_package_import() -> None:
    """Verify the main package is importable."""
    import energy_forecast

    assert energy_forecast.__version__ == "0.1.0"


def test_base_feature_engineer_import() -> None:
    """Verify BaseFeatureEngineer is importable."""
    from energy_forecast.features.base import BaseFeatureEngineer

    assert BaseFeatureEngineer is not None


def test_base_forecaster_import() -> None:
    """Verify BaseForecaster is importable."""
    from energy_forecast.models.base import BaseForecaster

    assert BaseForecaster is not None


def test_fastapi_app_import() -> None:
    """Verify FastAPI app is importable."""
    from energy_forecast.serving.app import app

    assert app is not None
