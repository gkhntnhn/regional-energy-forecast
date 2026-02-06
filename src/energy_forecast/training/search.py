"""Dynamic Optuna search space to trial parameter converter.

Reads the search_space dict from YAML config and automatically generates
the appropriate trial.suggest_* calls. Adding a new parameter to YAML
requires NO code change.

All models (CatBoost, Prophet, TFT) use this same function.
"""

from __future__ import annotations

from typing import Any

from energy_forecast.config.settings import SearchParamConfig


def suggest_params(
    trial: Any,  # optuna.Trial — typed as Any to avoid import at module level
    search_space: dict[str, SearchParamConfig],
) -> dict[str, Any]:
    """Generate Optuna trial parameters from search space config.

    Args:
        trial: Optuna trial object.
        search_space: ``{param_name: SearchParamConfig}`` mapping.

    Returns:
        ``{param_name: suggested_value}`` dict.
    """
    params: dict[str, Any] = {}
    for name, cfg in search_space.items():
        if cfg.type == "int":
            params[name] = trial.suggest_int(
                name,
                int(cfg.low),  # type: ignore[arg-type]
                int(cfg.high),  # type: ignore[arg-type]
                step=int(cfg.step) if cfg.step is not None else 1,
                log=cfg.log,
            )
        elif cfg.type == "float":
            kwargs: dict[str, Any] = {
                "name": name,
                "low": cfg.low,
                "high": cfg.high,
                "log": cfg.log,
            }
            if cfg.step is not None:
                kwargs["step"] = cfg.step
            params[name] = trial.suggest_float(**kwargs)
        elif cfg.type == "categorical":
            params[name] = trial.suggest_categorical(
                name,
                cfg.choices,
            )
    return params
