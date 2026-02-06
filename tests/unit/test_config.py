"""Unit tests for the config system."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from energy_forecast.config import Settings, get_default_config, load_config
from energy_forecast.config.settings import (
    CityConfig,
    ConsumptionLagConfig,
    CrossValidationConfig,
    EpiasExpandingConfig,
    EpiasLagConfig,
    ExpandingConfig,
    ModelSearchConfig,
    RegionConfig,
    SearchParamConfig,
    _load_yaml,
)

# ---------------------------------------------------------------------------
# Happy path: load from real YAML files
# ---------------------------------------------------------------------------


class TestLoadConfig:
    """Tests for loading config from project YAML files."""

    def test_load_config_from_project_yamls(self, settings: Settings) -> None:
        """Real YAML files produce a valid Settings object."""
        assert isinstance(settings, Settings)

    def test_settings_has_all_sections(self, settings: Settings) -> None:
        """Root config exposes all expected sub-configs."""
        assert settings.project is not None
        assert settings.logging is not None
        assert settings.region is not None
        assert settings.forecast is not None
        assert settings.pipeline is not None
        assert settings.data_loader is not None
        assert settings.openmeteo is not None
        assert settings.features is not None
        assert settings.catboost is not None
        assert settings.prophet is not None
        assert settings.tft is not None
        assert settings.hyperparameters is not None
        assert settings.env is not None

    def test_project_name(self, settings: Settings) -> None:
        """Project name matches YAML."""
        assert settings.project.name == "energy-forecast"

    def test_forecast_horizon_48(self, settings: Settings) -> None:
        """Forecast horizon is 48 hours."""
        assert settings.forecast.horizon_hours == 48

    def test_forecast_min_lag_48(self, settings: Settings) -> None:
        """Forecast min_lag is 48."""
        assert settings.forecast.min_lag == 48

    def test_openmeteo_variables_count(self, settings: Settings) -> None:
        """OpenMeteo config has 11 weather variables."""
        assert len(settings.openmeteo.variables) == 11

    def test_catboost_has_time_true(self, settings: Settings) -> None:
        """CatBoost has_time is always True."""
        assert settings.catboost.training.has_time is True

    def test_pipeline_drop_raw_epias_default_true(self, settings: Settings) -> None:
        """Pipeline drops raw EPIAS by default."""
        assert settings.pipeline.drop_raw_epias is True

    def test_data_loader_paths(self, settings: Settings) -> None:
        """Data loader paths are set correctly."""
        assert settings.data_loader.paths.raw == Path("data/raw")
        assert settings.data_loader.paths.holidays == Path("data/static/turkish_holidays.parquet")

    def test_prophet_seasonality_mode(self, settings: Settings) -> None:
        """Prophet uses multiplicative seasonality."""
        assert settings.prophet.seasonality.mode == "multiplicative"

    def test_tft_prediction_length(self, settings: Settings) -> None:
        """TFT prediction length matches forecast horizon."""
        assert settings.tft.training.prediction_length == 48

    def test_hyperparameters_cross_validation_no_shuffle(
        self,
        settings: Settings,
    ) -> None:
        """Cross-validation shuffle is disabled (time series)."""
        assert settings.hyperparameters.cross_validation.shuffle is False


# ---------------------------------------------------------------------------
# Default config (no YAML files)
# ---------------------------------------------------------------------------


class TestDefaultConfig:
    """Tests for get_default_config()."""

    def test_get_default_config(self) -> None:
        """Default config can be created without YAML files."""
        cfg = get_default_config()
        assert isinstance(cfg, Settings)

    def test_default_region_has_four_cities(self) -> None:
        """Default region has 4 cities."""
        cfg = get_default_config()
        assert len(cfg.region.cities) == 4

    def test_default_region_weights_sum(self) -> None:
        """Default region weights sum to 1.0."""
        cfg = get_default_config()
        total = sum(c.weight for c in cfg.region.cities)
        assert abs(total - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# Validators: Region weights
# ---------------------------------------------------------------------------


class TestRegionValidation:
    """Tests for region weight validators."""

    def test_region_weights_sum_to_one(self) -> None:
        """Valid region with weights summing to 1.0 passes."""
        region = RegionConfig(
            name="Test",
            cities=[
                CityConfig(name="A", weight=0.7, latitude=40.0, longitude=29.0),
                CityConfig(name="B", weight=0.3, latitude=39.0, longitude=28.0),
            ],
        )
        assert abs(sum(c.weight for c in region.cities) - 1.0) < 1e-6

    def test_region_weights_invalid_sum_raises(self) -> None:
        """Region with weights not summing to 1.0 raises."""
        with pytest.raises(ValidationError, match=r"weights must sum to 1\.0"):
            RegionConfig(
                name="Test",
                cities=[
                    CityConfig(name="A", weight=0.5, latitude=40.0, longitude=29.0),
                    CityConfig(name="B", weight=0.3, latitude=39.0, longitude=28.0),
                ],
            )


# ---------------------------------------------------------------------------
# Validators: Leakage guards
# ---------------------------------------------------------------------------


class TestLeakageGuards:
    """Tests for data leakage prevention validators."""

    def test_consumption_lag_min_48(self) -> None:
        """Consumption lags >= 48 pass validation."""
        lag_cfg = ConsumptionLagConfig(min_lag=48, values=[48, 72, 168])
        assert lag_cfg.min_lag == 48

    def test_consumption_lag_below_min_raises(self) -> None:
        """Consumption lag < 48 raises ValidationError."""
        with pytest.raises(ValidationError, match="leakage"):
            ConsumptionLagConfig(min_lag=48, values=[24, 48, 72])

    def test_consumption_min_lag_below_48_raises(self) -> None:
        """min_lag below 48 raises (Field ge=48)."""
        with pytest.raises(ValidationError):
            ConsumptionLagConfig(min_lag=24, values=[24, 48])

    def test_expanding_min_periods_ge_48(self) -> None:
        """Expanding min_periods >= 48 passes."""
        cfg = ExpandingConfig(min_periods=48, functions=["mean"])
        assert cfg.min_periods == 48

    def test_expanding_min_periods_below_48_raises(self) -> None:
        """Expanding min_periods < 48 raises ValidationError."""
        with pytest.raises(ValidationError):
            ExpandingConfig(min_periods=24, functions=["mean"])

    def test_epias_lag_min_48(self) -> None:
        """EPIAS lags >= 48 pass validation."""
        lag_cfg = EpiasLagConfig(min_lag=48, values=[48, 72, 168])
        assert lag_cfg.min_lag == 48

    def test_epias_lag_below_min_raises(self) -> None:
        """EPIAS lag < 48 raises ValidationError."""
        with pytest.raises(ValidationError, match="leakage"):
            EpiasLagConfig(min_lag=48, values=[24, 48])

    def test_epias_expanding_min_periods_below_48_raises(self) -> None:
        """EPIAS expanding min_periods < 48 raises."""
        with pytest.raises(ValidationError):
            EpiasExpandingConfig(min_periods=24, functions=["mean"])


# ---------------------------------------------------------------------------
# Frozen / immutability
# ---------------------------------------------------------------------------


class TestImmutability:
    """Tests for config immutability (frozen models)."""

    def test_settings_frozen(self, default_settings: Settings) -> None:
        """Settings attributes cannot be mutated."""
        with pytest.raises(ValidationError):
            default_settings.forecast = None  # type: ignore[assignment,misc]

    def test_nested_config_frozen(self, default_settings: Settings) -> None:
        """Nested config attributes cannot be mutated."""
        with pytest.raises(ValidationError):
            default_settings.forecast.horizon_hours = 24  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """Tests for error handling in config loading."""

    def test_missing_yaml_raises(self, tmp_path: Path) -> None:
        """Missing YAML file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path)

    def test_invalid_yaml_content_raises(self, tmp_path: Path) -> None:
        """Non-dict YAML content raises TypeError."""
        bad_yaml = tmp_path / "bad.yaml"
        bad_yaml.write_text("- item1\n- item2\n", encoding="utf-8")
        with pytest.raises(TypeError, match="Expected dict"):
            _load_yaml(bad_yaml)

    def test_load_yaml_valid(self, tmp_path: Path) -> None:
        """Valid YAML file loads as dict."""
        valid_yaml = tmp_path / "test.yaml"
        valid_yaml.write_text("key: value\n", encoding="utf-8")
        result: dict[str, Any] = _load_yaml(valid_yaml)
        assert result == {"key": "value"}


# ---------------------------------------------------------------------------
# Dynamic search space config
# ---------------------------------------------------------------------------


class TestSearchParamConfig:
    """Tests for SearchParamConfig validation."""

    def test_int_valid(self) -> None:
        cfg = SearchParamConfig(type="int", low=1, high=10)
        assert cfg.type == "int"
        assert cfg.low == 1
        assert cfg.high == 10

    def test_float_log_valid(self) -> None:
        cfg = SearchParamConfig(type="float", low=0.01, high=0.1, log=True)
        assert cfg.log is True

    def test_float_step_valid(self) -> None:
        cfg = SearchParamConfig(type="float", low=0.0, high=1.0, step=0.1)
        assert cfg.step == 0.1

    def test_categorical_valid(self) -> None:
        cfg = SearchParamConfig(type="categorical", choices=["RMSE", "MAE"])
        assert cfg.choices == ["RMSE", "MAE"]

    def test_int_missing_low_raises(self) -> None:
        with pytest.raises(ValidationError, match="requires low and high"):
            SearchParamConfig(type="int", high=10)

    def test_int_missing_high_raises(self) -> None:
        with pytest.raises(ValidationError, match="requires low and high"):
            SearchParamConfig(type="int", low=1)

    def test_low_greater_than_high_raises(self) -> None:
        with pytest.raises(ValidationError, match=r"low.*high"):
            SearchParamConfig(type="float", low=10.0, high=1.0)

    def test_log_with_step_raises(self) -> None:
        with pytest.raises(ValidationError, match="mutually exclusive"):
            SearchParamConfig(type="float", low=0.01, high=1.0, log=True, step=0.1)

    def test_categorical_no_choices_raises(self) -> None:
        with pytest.raises(ValidationError, match="non-empty choices"):
            SearchParamConfig(type="categorical")

    def test_categorical_empty_choices_raises(self) -> None:
        with pytest.raises(ValidationError, match="non-empty choices"):
            SearchParamConfig(type="categorical", choices=[])


class TestModelSearchConfig:
    """Tests for ModelSearchConfig."""

    def test_empty_search_space(self) -> None:
        cfg = ModelSearchConfig()
        assert cfg.n_trials == 50
        assert cfg.search_space == {}

    def test_with_search_space(self) -> None:
        cfg = ModelSearchConfig(
            n_trials=10,
            search_space={
                "depth": SearchParamConfig(type="int", low=4, high=7),
            },
        )
        assert cfg.n_trials == 10
        assert "depth" in cfg.search_space


class TestCrossValidationConfig:
    """Tests for calendar-month CrossValidationConfig."""

    def test_defaults(self) -> None:
        cfg = CrossValidationConfig()
        assert cfg.n_splits == 12
        assert cfg.val_months == 1
        assert cfg.test_months == 1
        assert cfg.gap_hours == 0
        assert cfg.shuffle is False

    def test_shuffle_always_false(self) -> None:
        cfg = CrossValidationConfig(shuffle=False)
        assert cfg.shuffle is False
