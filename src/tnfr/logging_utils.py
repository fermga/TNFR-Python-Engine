"""Logging utilities for TNFR.

Provides a helper to obtain module-specific loggers with a
consistent configuration across the project.
"""

from __future__ import annotations

import logging
import threading
from functools import lru_cache
from typing import Any, Hashable, Mapping

_LOCK = threading.Lock()
_LOGGING_CONFIGURED = False

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


def warn_once(
    logger: logging.Logger,
    msg: str,
    *,
    maxsize: int = 1024,
) -> callable:
    """Return a function that logs ``msg`` once per key.

    The returned callable accepts a mapping of keys to values. Keys are
    tracked using an LRU cache limited to ``maxsize`` entries. When
    called, new keys trigger a warning with their associated values while
    repeated keys are ignored. The callable exposes ``clear()`` to reset
    the tracked keys, useful for tests.
    """

    @lru_cache(maxsize=maxsize)
    def _seen(key: Hashable) -> None:  # pragma: no cover - simple cache
        return None

    def _log(mapping: Mapping[Hashable, Any]) -> None:
        new: dict[Hashable, Any] = {}
        for k, v in mapping.items():
            info = _seen.cache_info()
            _seen(k)
            if _seen.cache_info().misses > info.misses:
                new[k] = v
        if new:
            logger.warning(msg, new)

    _log.clear = _seen.cache_clear  # type: ignore[attr-defined]
    return _log
