"""Shared test fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture()
def project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture()
def configs_dir(project_root: Path) -> Path:
    """Return the configs directory path."""
    return project_root / "configs"


@pytest.fixture()
def data_dir(project_root: Path) -> Path:
    """Return the data directory path."""
    return project_root / "data"
