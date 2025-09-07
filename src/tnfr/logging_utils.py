"""Logging utilities for TNFR.

Provides a helper to obtain module-specific loggers with a
consistent configuration across the project.
"""

from __future__ import annotations

import logging
import threading

_LOCK = threading.Lock()


def get_logger(name: str) -> logging.Logger:
    """Return a logger with a standard configuration."""

    with _LOCK:
        root = logging.getLogger()
        if not root.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                )
            )
            root.addHandler(handler)
            root.setLevel(logging.INFO)
    return logging.getLogger(name)
