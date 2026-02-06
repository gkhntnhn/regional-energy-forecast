"""YAML-based configuration loader."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class Settings:
    """Loads and manages YAML configuration files.

    Args:
        config_dir: Path to the configs/ directory.
    """

    def __init__(self, config_dir: Path | None = None) -> None:
        self.config_dir = config_dir or Path("configs")
        self._cache: dict[str, Any] = {}

    def get(self, name: str) -> dict[str, Any]:
        """Load a YAML config by name.

        Args:
            name: Config file name without extension (e.g. 'settings').

        Returns:
            Parsed YAML as dictionary.
        """
        raise NotImplementedError
