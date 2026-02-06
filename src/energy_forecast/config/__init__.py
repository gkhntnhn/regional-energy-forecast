"""Configuration management.

Public API:
    - ``Settings``: Root config aggregating all YAML configs + env vars.
    - ``EnvConfig``: Environment variables loaded from ``.env``.
    - ``load_config()``: Load and validate all YAML configs.
    - ``get_default_config()``: Get default config (no YAML files required).
"""

from energy_forecast.config.settings import (
    EnvConfig,
    Settings,
    get_default_config,
    load_config,
)

__all__ = ["EnvConfig", "Settings", "get_default_config", "load_config"]
