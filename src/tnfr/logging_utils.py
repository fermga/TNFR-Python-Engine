"""Logging utilities for TNFR.

Provides a helper to obtain module-specific loggers with a
consistent configuration across the project.
"""

from __future__ import annotations

import logging
import threading

_LOCK = threading.Lock()
_LOGGING_CONFIGURED = False
_WARN_ONCE_LOCK = threading.Lock()
_WARNED_KEYS: set[str] = set()

__all__ = ("get_logger", "warn_once")


def _configure_root() -> None:
    """Configure the root logger if it has no handlers."""

    global _LOGGING_CONFIGURED

    root = logging.getLogger()
    if root.handlers:
        _LOGGING_CONFIGURED = True
        return

    kwargs = {"format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"}
    if root.level == logging.NOTSET:
        kwargs["level"] = logging.INFO
    logging.basicConfig(**kwargs)
    _LOGGING_CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a logger with a standard configuration.

    If the root logger has no handlers, it is configured with a default
    format. A default log level of ``INFO`` is only set when the root
    logger's level is ``NOTSET``; otherwise the existing level is
    preserved.
    """

    if not _LOGGING_CONFIGURED:
        with _LOCK:
            if not _LOGGING_CONFIGURED:
                _configure_root()
    return logging.getLogger(name)


def warn_once(key: str, message: str) -> None:
    """Log ``message`` once for ``key`` using a global set with lock."""

    with _WARN_ONCE_LOCK:
        if key in _WARNED_KEYS:
            return
        _WARNED_KEYS.add(key)
    logging.getLogger().warning(message)

