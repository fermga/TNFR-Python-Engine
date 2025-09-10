"""Logging utilities for TNFR.

Provides a helper to obtain module-specific loggers with a
consistent configuration across the project.
"""

from __future__ import annotations

import logging
import threading

_LOCK = threading.Lock()

__all__ = ("get_logger",)


def get_logger(name: str) -> logging.Logger:
    """Return a logger with a standard configuration.

    If the root logger has no handlers, it is configured with a default
    format. A default log level of ``INFO`` is only set when the root
    logger's level is ``NOTSET``; otherwise the existing level is
    preserved.
    """

    with _LOCK:
        root = logging.getLogger()
        if not root.handlers:
            kwargs = {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            }
            if root.level == logging.NOTSET:
                kwargs["level"] = logging.INFO
            logging.basicConfig(**kwargs)
    return logging.getLogger(name)
