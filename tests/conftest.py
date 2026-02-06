"""Shared test fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

from energy_forecast.config import Settings, get_default_config, load_config


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


@pytest.fixture()
def settings(configs_dir: Path) -> Settings:
    """Load Settings from project YAML configs."""
    return load_config(configs_dir)


@pytest.fixture()
def default_settings() -> Settings:
    """Get default Settings (no YAML files needed)."""
    return get_default_config()
