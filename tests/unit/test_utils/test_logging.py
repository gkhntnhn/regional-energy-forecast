"""Tests for logging utilities."""

from __future__ import annotations

import logging as stdlib_logging

from loguru import logger

from energy_forecast.utils.logging import setup_logger, suppress_training_noise


class TestSetupLogger:
    """Tests for setup_logger function."""

    def test_setup_logger_removes_default_handler(self) -> None:
        """Logger should have no default handlers after setup."""
        setup_logger(level="DEBUG")
        # loguru internally tracks handlers; after remove+add we have exactly 1
        assert len(logger._core.handlers) == 1

    def test_setup_logger_accepts_level(self) -> None:
        """Should accept valid log levels without error."""
        for level in ("DEBUG", "INFO", "WARNING", "ERROR"):
            setup_logger(level=level)

    def test_setup_logger_with_log_file(self, tmp_path) -> None:
        """setup_logger with log_file creates file handler."""
        log_file = str(tmp_path / "logs" / "test.log")
        setup_logger(level="DEBUG", log_file=log_file)
        # Should have 2 handlers: stderr + file
        assert len(logger._core.handlers) == 2

    def test_setup_logger_creates_parent_directory(self, tmp_path) -> None:
        """setup_logger with log_file creates parent directories."""
        from pathlib import Path

        log_file = str(tmp_path / "nested" / "dir" / "test.log")
        setup_logger(level="INFO", log_file=log_file)
        assert Path(log_file).parent.exists()


class TestSuppressTrainingNoise:
    """Tests for suppress_training_noise function."""

    def test_lightning_loggers_set_to_warning(self) -> None:
        """Lightning loggers should be silenced to WARNING level."""
        suppress_training_noise()

        assert stdlib_logging.getLogger("lightning.pytorch").level >= stdlib_logging.WARNING
        assert stdlib_logging.getLogger("lightning.fabric").level >= stdlib_logging.WARNING
        assert stdlib_logging.getLogger("pytorch_lightning").level >= stdlib_logging.WARNING
        assert stdlib_logging.getLogger("neuralforecast").level >= stdlib_logging.WARNING

    def test_optuna_verbosity_set_to_warning(self) -> None:
        """Optuna logging should be silenced to WARNING."""
        suppress_training_noise()

        import optuna

        assert optuna.logging.get_verbosity() >= optuna.logging.WARNING

    def test_torch_matmul_precision_set_on_cuda(self) -> None:
        """If CUDA available, float32 matmul precision should be set."""
        import torch

        suppress_training_noise()
        # Just verify it doesn't crash — precision check only meaningful on GPU
        if torch.cuda.is_available():
            assert torch.get_float32_matmul_precision() == "high"
