"""Tests for export_weather script."""

from __future__ import annotations

from unittest.mock import patch


class TestParseArgs:
    """Tests for argument parsing."""

    def test_defaults(self) -> None:
        from scripts.export_weather import parse_args

        with patch("sys.argv", ["export_weather.py"]):
            args = parse_args()
        assert args.year is None
        assert args.dry_run is False

    def test_year_flag(self) -> None:
        from scripts.export_weather import parse_args

        with patch("sys.argv", ["export_weather.py", "--year", "2024"]):
            args = parse_args()
        assert args.year == 2024

    def test_dry_run_flag(self) -> None:
        from scripts.export_weather import parse_args

        with patch("sys.argv", ["export_weather.py", "--dry-run"]):
            args = parse_args()
        assert args.dry_run is True


class TestMainNoDB:
    """Test main() when DATABASE_URL_SYNC is not set."""

    def test_returns_1_without_db_url(self) -> None:
        """main() returns 1 when DB URL is missing."""
        from scripts.export_weather import main

        with (
            patch("sys.argv", ["export_weather.py"]),
            patch.dict("os.environ", {}, clear=True),
        ):
            result = main()
        assert result == 1
