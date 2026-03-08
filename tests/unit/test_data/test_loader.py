"""Unit tests for DataLoader."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from energy_forecast.config import DataLoaderConfig
from energy_forecast.data.exceptions import DataValidationError
from energy_forecast.data.loader import DataLoader


@pytest.fixture()
def loader(data_loader_config: DataLoaderConfig) -> DataLoader:
    """Create a DataLoader with project config."""
    return DataLoader(data_loader_config)


class TestLoadExcel:
    """Tests for DataLoader.load_excel()."""

    def test_load_valid_excel(
        self,
        loader: DataLoader,
        sample_excel_path: Path,
    ) -> None:
        """Valid Excel file produces a DataFrame with 72 rows."""
        df = loader.load_excel(sample_excel_path)
        assert len(df) == 72
        assert "consumption" in df.columns

    def test_datetime_index_created(
        self,
        loader: DataLoader,
        sample_excel_path: Path,
    ) -> None:
        """Output has a DatetimeIndex."""
        df = loader.load_excel(sample_excel_path)
        assert isinstance(df.index, pd.DatetimeIndex)
        assert df.index.name == "datetime"

    def test_date_time_merge_correct(
        self,
        loader: DataLoader,
        sample_excel_path: Path,
    ) -> None:
        """Date+time columns merge correctly: 2024-01-01 + 15 → 15:00."""
        df = loader.load_excel(sample_excel_path)
        # First row: 2024-01-01 hour 0
        assert df.index[0] == pd.Timestamp("2024-01-01 00:00:00")
        # Hour 15 of day 1
        assert df.index[15] == pd.Timestamp("2024-01-01 15:00:00")

    def test_continuous_index_fills_gaps(
        self,
        loader: DataLoader,
        tmp_path: Path,
    ) -> None:
        """Missing hours in input are filled with NaN."""
        # Create Excel with a gap (skip hour 5)
        df = pd.DataFrame(
            {
                "date": ["2024-01-01"] * 23,
                "time": [h for h in range(24) if h != 5],
                "consumption": [1000.0] * 23,
            }
        )
        path = tmp_path / "gap.xlsx"
        df.to_excel(path, index=False, engine="openpyxl")

        result = loader.load_excel(path)
        assert len(result) == 24  # continuous, including the gap
        # Hour 5 should be NaN
        ts_5 = pd.Timestamp("2024-01-01 05:00:00")
        assert pd.isna(result.loc[ts_5, "consumption"])

    def test_missing_column_raises(
        self,
        loader: DataLoader,
        tmp_path: Path,
    ) -> None:
        """Missing required column raises DataValidationError."""
        df = pd.DataFrame({"date": ["2024-01-01"], "time": [0]})
        path = tmp_path / "bad.xlsx"
        df.to_excel(path, index=False, engine="openpyxl")

        with pytest.raises(DataValidationError, match="Missing required columns"):
            loader.load_excel(path)

    def test_invalid_time_range_raises(
        self,
        loader: DataLoader,
        tmp_path: Path,
    ) -> None:
        """Time value outside 0-23 raises DataValidationError."""
        df = pd.DataFrame(
            {
                "date": ["2024-01-01"],
                "time": [25],
                "consumption": [1000.0],
            }
        )
        path = tmp_path / "bad_time.xlsx"
        df.to_excel(path, index=False, engine="openpyxl")

        with pytest.raises(DataValidationError, match="Invalid time"):
            loader.load_excel(path)

    def test_negative_consumption_raises(
        self,
        loader: DataLoader,
        tmp_path: Path,
    ) -> None:
        """Negative consumption value raises DataValidationError."""
        df = pd.DataFrame(
            {
                "date": ["2024-01-01"],
                "time": [0],
                "consumption": [-10.0],
            }
        )
        path = tmp_path / "neg.xlsx"
        df.to_excel(path, index=False, engine="openpyxl")

        with pytest.raises(DataValidationError, match="below minimum"):
            loader.load_excel(path)

    def test_empty_excel_raises(
        self,
        loader: DataLoader,
        tmp_path: Path,
    ) -> None:
        """Empty Excel file raises DataValidationError."""
        df = pd.DataFrame({"date": [], "time": [], "consumption": []})
        path = tmp_path / "empty.xlsx"
        df.to_excel(path, index=False, engine="openpyxl")

        with pytest.raises(DataValidationError, match="no data rows"):
            loader.load_excel(path)

    def test_file_not_found_raises(self, loader: DataLoader) -> None:
        """Non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            loader.load_excel(Path("/nonexistent/file.xlsx"))

    def test_max_missing_ratio_exceeded(
        self,
        loader: DataLoader,
        tmp_path: Path,
    ) -> None:
        """More than 5% NaN consumption raises DataValidationError."""
        # 20 rows, 2 NaN = 10% > 5% threshold
        df = pd.DataFrame(
            {
                "date": ["2024-01-01"] * 20,
                "time": list(range(20)),
                "consumption": [1000.0] * 18 + [float("nan")] * 2,
            }
        )
        path = tmp_path / "missing.xlsx"
        df.to_excel(path, index=False, engine="openpyxl")

        with pytest.raises(DataValidationError, match="Missing consumption ratio"):
            loader.load_excel(path)


class TestExtendForForecast:
    """Tests for DataLoader.extend_for_forecast()."""

    def test_extends_by_48_hours(
        self,
        loader: DataLoader,
        sample_excel_path: Path,
    ) -> None:
        """Extend adds 48 NaN rows after last data point."""
        df = loader.load_excel(sample_excel_path)
        extended = loader.extend_for_forecast(df, horizon_hours=48)
        assert len(extended) == len(df) + 48

    def test_extended_rows_are_nan(
        self,
        loader: DataLoader,
        sample_excel_path: Path,
    ) -> None:
        """Extended rows have NaN consumption."""
        df = loader.load_excel(sample_excel_path)
        extended = loader.extend_for_forecast(df, horizon_hours=48)
        last_original = df.index.max()
        forecast_mask = extended.index > last_original
        assert extended.loc[forecast_mask, "consumption"].isna().all()
