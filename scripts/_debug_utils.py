"""Shared utilities for debug scripts (debug_catboost, debug_prophet, debug_tft).

Extracts common patterns that were duplicated across all three debug scripts:
- TeeLogger: Captures stdout/stderr to both terminal and log file
- SEASON_MAP: Month-to-season mapping
- setup_tee_logging: Standard TeeLogger setup for debug scripts
"""

from __future__ import annotations

import sys
from typing import TextIO


class TeeLogger:
    """Writes to both terminal and log file. Survives bash timeout."""

    def __init__(self, filepath: str, original_stream: TextIO) -> None:
        self.terminal = original_stream
        self.log = open(filepath, "w", encoding="utf-8")  # noqa: SIM115

    def write(self, message: str) -> None:
        try:
            self.terminal.write(message)
        except (OSError, BrokenPipeError):
            pass  # pipe may be broken, continue logging to file
        self.log.write(message)
        self.log.flush()

    def flush(self) -> None:
        self.terminal.flush()
        self.log.flush()

    def close(self) -> None:
        self.log.close()


SEASON_MAP: dict[int, str] = {
    12: "Winter", 1: "Winter", 2: "Winter",
    3: "Spring", 4: "Spring", 5: "Spring",
    6: "Summer", 7: "Summer", 8: "Summer",
    9: "Fall", 10: "Fall", 11: "Fall",
}


def setup_tee_logging(log_path: str) -> tuple[TeeLogger, TeeLogger]:
    """Set up TeeLogger for stdout and stderr.

    Args:
        log_path: Path to the terminal log file.

    Returns:
        Tuple of (stdout_tee, stderr_tee) for cleanup.
    """
    tee_out = TeeLogger(log_path, sys.stdout)
    tee_err = TeeLogger(log_path, sys.stderr)
    sys.stdout = tee_out  # type: ignore[assignment]
    sys.stderr = tee_err  # type: ignore[assignment]
    return tee_out, tee_err
