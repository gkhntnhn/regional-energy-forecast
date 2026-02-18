"""Unit tests for ProphetTrainer."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from energy_forecast.config.settings import (
    CrossValidationConfig,
    HyperparameterConfig,
    ModelSearchConfig,
    ProphetChangepointConfig,
    ProphetConfig,
    ProphetHolidaysConfig,
    ProphetRegressorConfig,
    ProphetSeasonalityConfig,
    ProphetUncertaintyConfig,
    SearchParamConfig,
    Settings,
)
from energy_forecast.training.metrics import MetricsResult
from energy_forecast.training.prophet_trainer import (
    ProphetPipelineResult,
    ProphetSplitResult,
    ProphetTrainer,
    ProphetTrainingResult,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Create sample hourly data for testing (6 months)."""
    dates = pd.date_range("2023-01-01", "2023-06-30 23:00", freq="h")
    n = len(dates)
    rng = np.random.default_rng(42)

    # Synthetic consumption with daily pattern
    hour_effect = np.sin(np.arange(n) * 2 * np.pi / 24) * 100
    trend = np.linspace(1000, 1100, n)
    noise = rng.normal(0, 30, n)
    consumption = trend + hour_effect + noise

    return pd.DataFrame(
        {
            "consumption": consumption,
            "temperature": rng.uniform(5, 30, n),
            "humidity": rng.uniform(30, 80, n),
        },
        index=dates,
    )


@pytest.fixture
def mock_holidays_df() -> pd.DataFrame:
    """Create mock holidays DataFrame with window parameters."""
    return pd.DataFrame(
        {
            "ds": pd.to_datetime(["2023-01-01", "2023-04-23", "2023-05-01"]),
            "holiday": ["New Year", "National Sovereignty", "Labour Day"],
            "lower_window": [0, 0, 0],
            "upper_window": [1, 1, 1],
        }
    )


@pytest.fixture
def settings_with_prophet(tmp_path: Any) -> Settings:
    """Create Settings with Prophet config for testing."""
    # Create mock holidays file
    holidays_path = tmp_path / "holidays.parquet"
    holidays_df = pd.DataFrame(
        {
            "ds": pd.to_datetime(["2023-01-01", "2023-04-23"]),
            "holiday": ["New Year", "National Sovereignty"],
        }
    )
    holidays_df.to_parquet(holidays_path)

    # Create minimal Settings with Prophet config
    from energy_forecast.config.settings import (
        CityConfig,
        DataLoaderConfig,
        PathsConfig,
        RegionConfig,
    )

    return Settings(
        region=RegionConfig(
            name="Test",
            cities=[CityConfig(name="Test", weight=1.0, latitude=40.0, longitude=29.0)],
        ),
        data_loader=DataLoaderConfig(
            paths=PathsConfig(holidays=holidays_path),
        ),
        prophet=ProphetConfig(
            seasonality=ProphetSeasonalityConfig(mode="multiplicative"),
            holidays=ProphetHolidaysConfig(country="TR", include_ramadan=False),
            regressors=[
                ProphetRegressorConfig(name="temperature", mode="additive"),
                ProphetRegressorConfig(name="humidity", mode="additive"),
            ],
            changepoint=ProphetChangepointConfig(prior_scale=0.05, n_changepoints=10),
            uncertainty=ProphetUncertaintyConfig(interval_width=0.95, mcmc_samples=0),
        ),
        hyperparameters=HyperparameterConfig(
            prophet=ModelSearchConfig(
                n_trials=2,
                search_space={
                    "changepoint_prior_scale": SearchParamConfig(
                        type="float", low=0.001, high=0.5, log=True
                    ),
                    "seasonality_mode": SearchParamConfig(
                        type="categorical", choices=["additive", "multiplicative"]
                    ),
                },
            ),
            cross_validation=CrossValidationConfig(
                n_splits=2,
                val_months=1,
                test_months=1,
                gap_hours=0,
            ),
            target_col="consumption",
        ),
    )


@pytest.fixture
def trainer(settings_with_prophet: Settings) -> ProphetTrainer:
    """Create ProphetTrainer instance."""
    return ProphetTrainer(settings_with_prophet)


# ---------------------------------------------------------------------------
# Tests: Format Conversion
# ---------------------------------------------------------------------------


class TestToProphetFormat:
    """Tests for _to_prophet_format method."""

    def test_with_target(self, trainer: ProphetTrainer, sample_df: pd.DataFrame) -> None:
        """Test format conversion with target column."""
        prophet_df = trainer._to_prophet_format(sample_df, include_target=True)

        assert "ds" in prophet_df.columns
        assert "y" in prophet_df.columns
        assert len(prophet_df) == len(sample_df)
        assert prophet_df["ds"].iloc[0] == sample_df.index[0]

    def test_without_target(self, trainer: ProphetTrainer, sample_df: pd.DataFrame) -> None:
        """Test format conversion without target column."""
        prophet_df = trainer._to_prophet_format(sample_df, include_target=False)

        assert "ds" in prophet_df.columns
        assert "y" not in prophet_df.columns

    def test_regressors_included(self, trainer: ProphetTrainer, sample_df: pd.DataFrame) -> None:
        """Test that configured regressors are included."""
        prophet_df = trainer._to_prophet_format(sample_df, include_target=True)

        assert "temperature" in prophet_df.columns
        assert "humidity" in prophet_df.columns

    def test_missing_regressor_skipped(
        self, trainer: ProphetTrainer, sample_df: pd.DataFrame
    ) -> None:
        """Test that missing regressors are skipped without error."""
        # Remove humidity column
        df = sample_df.drop(columns=["humidity"])
        prophet_df = trainer._to_prophet_format(df, include_target=True)

        assert "temperature" in prophet_df.columns
        assert "humidity" not in prophet_df.columns


# ---------------------------------------------------------------------------
# Tests: Model Creation
# ---------------------------------------------------------------------------


class TestCreateProphet:
    """Tests for _create_prophet method."""

    def test_default_params(
        self, trainer: ProphetTrainer, sample_df: pd.DataFrame
    ) -> None:
        """Test Prophet creation with default parameters."""
        # _to_prophet_format must run first to populate _regressor_names
        trainer._to_prophet_format(sample_df, include_target=True)
        model = trainer._create_prophet({})

        assert model is not None
        # Check that regressors were added (temperature + humidity from fixture)
        assert len(model.extra_regressors) == 2

    def test_with_optuna_params(self, trainer: ProphetTrainer) -> None:
        """Test Prophet creation with Optuna parameters."""
        params = {
            "changepoint_prior_scale": 0.1,
            "seasonality_mode": "additive",
        }
        model = trainer._create_prophet(params)

        assert model.changepoint_prior_scale == 0.1
        assert model.seasonality_mode == "additive"


# ---------------------------------------------------------------------------
# Tests: Holidays Loading
# ---------------------------------------------------------------------------


class TestLoadHolidays:
    """Tests for _load_holidays method."""

    def test_holidays_loaded(self, trainer: ProphetTrainer) -> None:
        """Test that holidays are loaded from parquet."""
        holidays = trainer._load_holidays()

        assert holidays is not None
        assert "ds" in holidays.columns
        assert "holiday" in holidays.columns
        assert "lower_window" in holidays.columns
        assert "upper_window" in holidays.columns
        assert len(holidays) > 0

    def test_holidays_cached_in_init(self, trainer: ProphetTrainer) -> None:
        """Test that holidays are cached during __init__."""
        assert trainer._holidays_df is not None
        assert "ds" in trainer._holidays_df.columns

    def test_holiday_windows_resmi(self, trainer: ProphetTrainer) -> None:
        """Test that resmi tatiller get upper_window=1 (day-after effect)."""
        holidays = trainer._load_holidays()
        assert holidays is not None
        # Fixture holidays are "New Year" and "National Sovereignty" — both resmi
        assert (holidays["lower_window"] == 0).all()
        assert (holidays["upper_window"] == 1).all()

    def test_holidays_missing_file(self, settings_with_prophet: Settings) -> None:
        """Test graceful handling of missing holidays file."""
        from energy_forecast.config.settings import DataLoaderConfig, PathsConfig

        # Create settings with non-existent holidays path
        settings = settings_with_prophet.model_copy(
            update={
                "data_loader": DataLoaderConfig(
                    paths=PathsConfig(holidays="nonexistent.parquet"),
                ),
            }
        )
        trainer = ProphetTrainer(settings)
        holidays = trainer._load_holidays()

        assert holidays is None
        assert trainer._holidays_df is None


# ---------------------------------------------------------------------------
# Tests: Split Training
# ---------------------------------------------------------------------------


class TestTrainSplit:
    """Tests for _train_split method."""

    def test_train_split_smoke(self, trainer: ProphetTrainer, sample_df: pd.DataFrame) -> None:
        """Smoke test for single split training."""
        from energy_forecast.training.splitter import SplitInfo

        # Create a simple split
        split_info = SplitInfo(
            split_idx=0,
            train_start=pd.Timestamp("2023-01-01"),
            train_end=pd.Timestamp("2023-02-28 23:00"),
            val_start=pd.Timestamp("2023-03-01"),
            val_end=pd.Timestamp("2023-03-31 23:00"),
            test_start=pd.Timestamp("2023-04-01"),
            test_end=pd.Timestamp("2023-04-30 23:00"),
        )

        train_df = sample_df.loc["2023-01-01":"2023-02-28"]
        val_df = sample_df.loc["2023-03-01":"2023-03-31"]
        test_df = sample_df.loc["2023-04-01":"2023-04-30"]

        result = trainer._train_split(split_info, train_df, val_df, test_df, {})

        assert isinstance(result, ProphetSplitResult)
        assert result.split_idx == 0
        assert result.val_month == "2023-03"
        assert result.test_month == "2023-04"
        assert isinstance(result.train_metrics, MetricsResult)
        assert isinstance(result.val_metrics, MetricsResult)
        assert isinstance(result.test_metrics, MetricsResult)

    def test_split_result_month_labels(
        self, trainer: ProphetTrainer, sample_df: pd.DataFrame
    ) -> None:
        """Test that month labels are correctly formatted."""
        from energy_forecast.training.splitter import SplitInfo

        split_info = SplitInfo(
            split_idx=0,
            train_start=pd.Timestamp("2023-01-01"),
            train_end=pd.Timestamp("2023-02-28 23:00"),
            val_start=pd.Timestamp("2023-03-01"),
            val_end=pd.Timestamp("2023-03-31 23:00"),
            test_start=pd.Timestamp("2023-04-01"),
            test_end=pd.Timestamp("2023-04-30 23:00"),
        )

        train_df = sample_df.loc["2023-01-01":"2023-02-28"]
        val_df = sample_df.loc["2023-03-01":"2023-03-31"]
        test_df = sample_df.loc["2023-04-01":"2023-04-30"]

        result = trainer._train_split(split_info, train_df, val_df, test_df, {})

        assert result.val_month == "2023-03"
        assert result.test_month == "2023-04"


# ---------------------------------------------------------------------------
# Tests: All Splits Training
# ---------------------------------------------------------------------------


class TestTrainAllSplits:
    """Tests for _train_all_splits method."""

    def test_train_all_splits(self, trainer: ProphetTrainer, sample_df: pd.DataFrame) -> None:
        """Test training on all CV splits."""
        result = trainer._train_all_splits(sample_df, {})

        assert isinstance(result, ProphetTrainingResult)
        assert len(result.split_results) == 2  # n_splits=2
        assert result.avg_val_mape > 0
        assert result.avg_test_mape > 0
        assert result.std_val_mape >= 0


# ---------------------------------------------------------------------------
# Tests: Optimization
# ---------------------------------------------------------------------------


class TestOptimize:
    """Tests for optimize method."""

    def test_optimize_smoke(self, trainer: ProphetTrainer, sample_df: pd.DataFrame) -> None:
        """Smoke test for Optuna optimization (n_trials=2)."""
        study, result = trainer.optimize(sample_df)

        assert study is not None
        assert study.best_value > 0
        has_cps = "changepoint_prior_scale" in study.best_params
        has_sm = "seasonality_mode" in study.best_params
        assert has_cps or has_sm
        assert isinstance(result, ProphetTrainingResult)


# ---------------------------------------------------------------------------
# Tests: Final Model
# ---------------------------------------------------------------------------


class TestTrainFinal:
    """Tests for train_final method."""

    def test_train_final(self, trainer: ProphetTrainer, sample_df: pd.DataFrame) -> None:
        """Test final model training on all data."""
        from prophet import Prophet

        params = {"changepoint_prior_scale": 0.05, "seasonality_mode": "multiplicative"}
        model = trainer.train_final(sample_df, params)

        assert isinstance(model, Prophet)


# ---------------------------------------------------------------------------
# Tests: Full Pipeline
# ---------------------------------------------------------------------------


class TestRun:
    """Tests for run method (full pipeline)."""

    def test_run_pipeline(self, trainer: ProphetTrainer, sample_df: pd.DataFrame) -> None:
        """Test full training pipeline."""
        result = trainer.run(sample_df)

        assert isinstance(result, ProphetPipelineResult)
        assert result.study is not None
        assert result.best_params is not None
        assert isinstance(result.training_result, ProphetTrainingResult)
        assert result.final_model is not None
        assert result.training_time_seconds > 0


# ---------------------------------------------------------------------------
# Tests: Dynamic Search Space
# ---------------------------------------------------------------------------


class TestDynamicSearchSpace:
    """Tests for dynamic Optuna search space from YAML."""

    def test_dynamic_search_space(
        self, settings_with_prophet: Settings, sample_df: pd.DataFrame
    ) -> None:
        """Test that search space is correctly read from config."""
        trainer = ProphetTrainer(settings_with_prophet)
        search_space = trainer._search_config.search_space

        assert "changepoint_prior_scale" in search_space
        assert "seasonality_mode" in search_space
        assert search_space["changepoint_prior_scale"].type == "float"
        assert search_space["seasonality_mode"].type == "categorical"


# ---------------------------------------------------------------------------
# Tests: Dataclass Behaviors
# ---------------------------------------------------------------------------


class TestDataclasses:
    """Tests for result dataclasses."""

    def test_prophet_split_result_frozen(self) -> None:
        """Test that ProphetSplitResult is frozen."""
        result = ProphetSplitResult(
            split_idx=0,
            train_metrics=MetricsResult(
                mape=5.0, mae=10.0, rmse=15.0, r2=0.9, smape=5.0, wmape=5.0, mbe=0.0
            ),
            val_metrics=MetricsResult(
                mape=6.0, mae=12.0, rmse=18.0, r2=0.85, smape=6.0, wmape=6.0, mbe=0.0
            ),
            test_metrics=MetricsResult(
                mape=7.0, mae=14.0, rmse=20.0, r2=0.8, smape=7.0, wmape=7.0, mbe=0.0
            ),
            val_month="2023-03",
            test_month="2023-04",
        )

        with pytest.raises(AttributeError):
            result.split_idx = 1  # type: ignore[misc]

    def test_prophet_training_result_frozen(self) -> None:
        """Test that ProphetTrainingResult is frozen."""
        result = ProphetTrainingResult(
            split_results=[],
            avg_val_mape=5.0,
            avg_test_mape=6.0,
            std_val_mape=1.0,
            regressor_names=["temperature"],
        )

        with pytest.raises(AttributeError):
            result.avg_val_mape = 10.0  # type: ignore[misc]
