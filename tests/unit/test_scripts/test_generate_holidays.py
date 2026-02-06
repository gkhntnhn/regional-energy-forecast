"""Unit tests for holiday generation script."""

from __future__ import annotations

import pandas as pd
from scripts.generate_holidays import (
    _map_holiday_name,
    _resolve_collisions,
    generate_holiday_catalog,
)


class TestGenerateHolidayCatalog:
    """Tests for generate_holiday_catalog()."""

    def test_returns_dataframe(self) -> None:
        """Catalog generation returns a DataFrame."""
        df = generate_holiday_catalog(start_year=2024, end_year=2024)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_has_required_columns(self) -> None:
        """Output has date, holiday_name, raw_holiday_name columns."""
        df = generate_holiday_catalog(start_year=2024, end_year=2024)
        assert "date" in df.columns
        assert "holiday_name" in df.columns
        assert "raw_holiday_name" in df.columns

    def test_holiday_name_is_categorical(self) -> None:
        """holiday_name column has categorical dtype."""
        df = generate_holiday_catalog(start_year=2024, end_year=2024)
        assert df["holiday_name"].dtype.name == "category"

    def test_year_range_correct(self) -> None:
        """Multi-year range includes holidays from all years."""
        df = generate_holiday_catalog(start_year=2020, end_year=2022)
        years = pd.to_datetime(df["date"]).dt.year.unique()
        assert 2020 in years
        assert 2021 in years
        assert 2022 in years

    def test_no_duplicate_dates(self) -> None:
        """Each date appears at most once (collisions resolved)."""
        df = generate_holiday_catalog(start_year=2020, end_year=2024)
        assert df["date"].is_unique


class TestHolidayNameMapping:
    """Tests for _map_holiday_name()."""

    def test_known_holiday_mapped(self) -> None:
        """Cumhuriyet Bayramı maps to Republic Day."""
        assert _map_holiday_name("Cumhuriyet Bayramı") == "Republic Day"

    def test_ramazan_mapped(self) -> None:
        """Ramazan Bayramı maps to Eid al-Fitr."""
        assert _map_holiday_name("Ramazan Bayramı") == "Eid al-Fitr"

    def test_estimated_variant_mapped(self) -> None:
        """Estimated (tahmini) variant maps to same name."""
        assert _map_holiday_name("Ramazan Bayramı (tahmini)") == "Eid al-Fitr"

    def test_unknown_returns_raw(self) -> None:
        """Unknown holiday name returns the raw name."""
        assert _map_holiday_name("Bilinmeyen Bayram") == "Bilinmeyen Bayram"


class TestCollisionResolution:
    """Tests for _resolve_collisions()."""

    def test_collision_resolved_by_priority(self) -> None:
        """Same date with two holidays → higher priority kept."""
        df = pd.DataFrame(
            {
                "date": [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-01")],
                "holiday_name": ["New Year", "Republic Day"],
                "raw_holiday_name": ["Yılbaşı", "Cumhuriyet Bayramı"],
            }
        )
        result = _resolve_collisions(df)
        assert len(result) == 1
        # Republic Day has higher priority than New Year
        assert result.iloc[0]["holiday_name"] == "Republic Day"
