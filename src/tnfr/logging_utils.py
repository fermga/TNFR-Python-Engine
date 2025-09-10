"""Logging utilities for TNFR.

Provides a helper to obtain module-specific loggers with a
consistent configuration across the project.
"""

from __future__ import annotations

import logging

__all__ = ("get_logger", "warn_once")


root = logging.getLogger()
if not root.handlers:
    kwargs = {"format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"}
    if root.level == logging.NOTSET:
        kwargs["level"] = logging.INFO
    logging.basicConfig(**kwargs)


def get_logger(name: str) -> logging.Logger:
    """Return a module-specific logger."""

    return logging.getLogger(name)


def warn_once(key: str, message: str) -> None:
    """Log ``message`` once for ``key`` using a global set with lock."""

    with _WARN_ONCE_LOCK:
        if key in _WARNED_KEYS:
            return
        _WARNED_KEYS.add(key)
    logging.getLogger().warning(message)

