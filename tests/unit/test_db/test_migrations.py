"""Tests for Alembic migrations."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

VERSIONS_DIR = Path(__file__).resolve().parents[3] / "alembic" / "versions"


def _load_migration(filename: str) -> ModuleType:
    """Load a migration module by filename (handles numeric-prefix names)."""
    path = VERSIONS_DIR / filename
    spec = importlib.util.spec_from_file_location(filename.removesuffix(".py"), path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestMigration008:
    """Tests for 008_add_queued_status migration."""

    @pytest.fixture(autouse=True)
    def _load(self) -> None:
        self.mod = _load_migration("008_add_queued_status.py")

    def test_upgrade_postgresql(self) -> None:
        """PostgreSQL upgrade drops old constraint and adds queued status."""
        mock_op = MagicMock()
        mock_bind = MagicMock()
        mock_bind.dialect.name = "postgresql"
        mock_op.get_bind.return_value = mock_bind

        with patch.object(self.mod, "op", mock_op):
            self.mod.upgrade()

        mock_op.drop_constraint.assert_called_once_with(
            "ck_jobs_status", "jobs", type_="check"
        )
        mock_op.create_check_constraint.assert_called_once_with(
            "ck_jobs_status",
            "jobs",
            "status IN ('pending', 'queued', 'running', 'completed', 'failed', 'archived')",
        )

    def test_upgrade_sqlite_noop(self) -> None:
        """SQLite upgrade is a no-op (no ALTER CONSTRAINT support)."""
        mock_op = MagicMock()
        mock_bind = MagicMock()
        mock_bind.dialect.name = "sqlite"
        mock_op.get_bind.return_value = mock_bind

        with patch.object(self.mod, "op", mock_op):
            self.mod.upgrade()

        mock_op.drop_constraint.assert_not_called()
        mock_op.create_check_constraint.assert_not_called()

    def test_downgrade_postgresql(self) -> None:
        """PostgreSQL downgrade removes queued from constraint."""
        mock_op = MagicMock()
        mock_bind = MagicMock()
        mock_bind.dialect.name = "postgresql"
        mock_op.get_bind.return_value = mock_bind

        with patch.object(self.mod, "op", mock_op):
            self.mod.downgrade()

        mock_op.drop_constraint.assert_called_once_with(
            "ck_jobs_status", "jobs", type_="check"
        )
        mock_op.create_check_constraint.assert_called_once_with(
            "ck_jobs_status",
            "jobs",
            "status IN ('pending', 'running', 'completed', 'failed', 'archived')",
        )

    def test_downgrade_sqlite_noop(self) -> None:
        """SQLite downgrade is a no-op."""
        mock_op = MagicMock()
        mock_bind = MagicMock()
        mock_bind.dialect.name = "sqlite"
        mock_op.get_bind.return_value = mock_bind

        with patch.object(self.mod, "op", mock_op):
            self.mod.downgrade()

        mock_op.drop_constraint.assert_not_called()
        mock_op.create_check_constraint.assert_not_called()


class TestMigrationChain:
    """Verify migration revision chain is not broken."""

    def test_revision_chain_008(self) -> None:
        """008 correctly depends on 007."""
        mod = _load_migration("008_add_queued_status.py")
        assert mod.revision == "008"
        assert mod.down_revision == "007"

    def test_all_revisions_importable(self) -> None:
        """All migration modules can be imported without errors."""
        filenames = [
            "001_initial_schema.py",
            "2051b306c0c5_fix_datetime_columns_to_timestamptz.py",
            "003_weather_snapshots.py",
            "004_audit_logs.py",
            "005_model_runs.py",
            "006_external_data_tables.py",
            "007_fix_json_to_jsonb.py",
            "008_add_queued_status.py",
        ]
        for fname in filenames:
            mod = _load_migration(fname)
            assert hasattr(mod, "upgrade"), f"{fname} missing upgrade()"
            assert hasattr(mod, "downgrade"), f"{fname} missing downgrade()"
            assert hasattr(mod, "revision"), f"{fname} missing revision"
            assert hasattr(mod, "down_revision"), f"{fname} missing down_revision"

    def test_revision_chain_continuity(self) -> None:
        """Each migration's down_revision matches the previous revision."""
        # Order: 001 → 2051b306c0c5 → 003 → 004 → ... → 008
        filenames = [
            "001_initial_schema.py",
            "2051b306c0c5_fix_datetime_columns_to_timestamptz.py",
            "003_weather_snapshots.py",
            "004_audit_logs.py",
            "005_model_runs.py",
            "006_external_data_tables.py",
            "007_fix_json_to_jsonb.py",
            "008_add_queued_status.py",
        ]
        prev_rev = None
        for fname in filenames:
            mod = _load_migration(fname)
            if prev_rev is not None:
                assert mod.down_revision == prev_rev, (
                    f"{fname}: down_revision={mod.down_revision!r} != expected {prev_rev!r}"
                )
            prev_rev = mod.revision
