"""Tests for prepare_dataset staleness guard (item 235)."""

from __future__ import annotations

import pandas as pd
import pytest
from scripts.data.prepare_dataset import _warn_if_stale_cache


class TestWarnIfStaleCache:
    """Verify the cache-staleness guard surfaces silent gaps."""

    def _df(self, end: str) -> pd.DataFrame:
        idx = pd.date_range("2026-03-01", end=end, freq="h")
        return pd.DataFrame({"v": range(len(idx))}, index=idx)

    def test_fresh_cache_no_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        df = self._df("2026-03-31 23:00")
        with caplog.at_level("WARNING"):
            _warn_if_stale_cache(df, "EPIAS market", "2026-03-31")
        assert not any("STALE CACHE" in r.message for r in caplog.records)

    def test_stale_cache_warns(self) -> None:
        df = self._df("2026-03-09 23:00")
        from loguru import logger

        captured: list[str] = []
        sink_id = logger.add(lambda m: captured.append(str(m)), level="WARNING")
        try:
            _warn_if_stale_cache(df, "EPIAS market", "2026-03-31")
        finally:
            logger.remove(sink_id)

        assert any("STALE CACHE" in m for m in captured)
        assert any("EPIAS market" in m for m in captured)

    def test_within_threshold_no_warning(self) -> None:
        df = self._df("2026-03-30 23:00")
        from loguru import logger

        captured: list[str] = []
        sink_id = logger.add(lambda m: captured.append(str(m)), level="WARNING")
        try:
            _warn_if_stale_cache(df, "EPIAS market", "2026-03-31", threshold_hours=48)
        finally:
            logger.remove(sink_id)

        assert not any("STALE CACHE" in m for m in captured)

    def test_none_df_silent(self) -> None:
        from loguru import logger

        captured: list[str] = []
        sink_id = logger.add(lambda m: captured.append(str(m)), level="WARNING")
        try:
            _warn_if_stale_cache(None, "EPIAS market", "2026-03-31")
        finally:
            logger.remove(sink_id)
        assert not captured

    def test_empty_df_silent(self) -> None:
        from loguru import logger

        captured: list[str] = []
        sink_id = logger.add(lambda m: captured.append(str(m)), level="WARNING")
        try:
            _warn_if_stale_cache(pd.DataFrame(), "EPIAS market", "2026-03-31")
        finally:
            logger.remove(sink_id)
        assert not captured

    def test_non_datetime_index_silent(self) -> None:
        df = pd.DataFrame({"v": [1, 2, 3]})
        from loguru import logger

        captured: list[str] = []
        sink_id = logger.add(lambda m: captured.append(str(m)), level="WARNING")
        try:
            _warn_if_stale_cache(df, "EPIAS market", "2026-03-31")
        finally:
            logger.remove(sink_id)
        assert not captured
