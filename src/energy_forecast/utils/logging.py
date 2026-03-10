"""Loguru-based logging setup."""

from __future__ import annotations

import sys

from loguru import logger


def setup_logger(level: str = "INFO", log_file: str | None = None) -> None:
    """Configure loguru logger for the project.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR).
        log_file: Optional file path for persistent logging.
            When set, logs are written to the file with rotation (50 MB)
            and retention (5 files). Tracebacks are included automatically.
    """
    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level}</level> | "
            "<cyan>{name}</cyan> - <level>{message}</level>"
        ),
    )

    if log_file is not None:
        from pathlib import Path

        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            log_file,
            level=level,
            format=(
                "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
                "{level:<8} | "
                "{name}:{function}:{line} - {message}"
            ),
            rotation="50 MB",
            retention=5,
            backtrace=True,
            diagnose=True,
        )


def suppress_training_noise() -> None:
    """Suppress noisy warnings and logging from training dependencies.

    Silences Lightning banners/tips/GPU info, pytorch-forecasting hints,
    Optuna per-trial INFO logs, and torch deprecation warnings.
    Keeps WARNING+ from all libs so real errors remain visible.

    Call once before TFT or ensemble training starts.
    """
    import logging as stdlib_logging
    import warnings

    # --- warnings.warn() calls ---
    # Lightning: checkpoint hints, nn.Module save recommendations, deprecations
    warnings.filterwarnings("ignore", category=UserWarning, module="lightning")
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="lightning")
    warnings.filterwarnings("ignore", category=UserWarning, module="pytorch_lightning")
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="pytorch_lightning")
    # neuralforecast: suppress internal warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="neuralforecast")
    # torch: FutureWarning (API changes), DeprecationWarning (cuda.amp → torch.amp)
    warnings.filterwarnings("ignore", category=FutureWarning, module="torch")
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="torch")

    # --- stdlib logging (Lightning banners, Optuna trial logs) ---
    # Lightning: kills "GPU available", "TPU available", "LOCAL_RANK", "Tip" banners
    stdlib_logging.getLogger("lightning.pytorch").setLevel(stdlib_logging.WARNING)
    stdlib_logging.getLogger("lightning.fabric").setLevel(stdlib_logging.WARNING)
    # pytorch_lightning (old package name, used by neuralforecast internally)
    stdlib_logging.getLogger("pytorch_lightning").setLevel(stdlib_logging.WARNING)
    # neuralforecast: suppress internal INFO
    stdlib_logging.getLogger("neuralforecast").setLevel(stdlib_logging.WARNING)

    # Optuna: suppress "[I] Trial X finished..." and "[I] study created..."
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Set Tensor Core precision to avoid "you should set precision" warning
    import torch

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
