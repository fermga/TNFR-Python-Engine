"""Logging utilities for TNFR.

Provides a helper to obtain module-specific loggers with a
consistent configuration across the project.
"""

from __future__ import annotations

import logging
import threading

_LOCK = threading.Lock()

__all__ = ["get_logger"]


def get_logger(name: str) -> logging.Logger:
    """Return a logger with a standard configuration."""

    with _LOCK:
        root = logging.getLogger()
        if not root.handlers:
            level = root.level if root.level != logging.WARNING else logging.INFO
            logging.basicConfig(
                level=level,
                format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            )
    return logging.getLogger(name)
