"""Logging utilities for TNFR.

Provides a helper to obtain module-specific loggers with a
consistent configuration across the project.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Hashable, Mapping

from cachetools import LRUCache

from .logging_base import _configure_root

__all__ = ("get_logger", "WarnOnce", "warn_once")


def get_logger(name: str) -> logging.Logger:
    """Return a module-specific logger."""
    _configure_root()
    return logging.getLogger(name)


class WarnOnce:
    """Log a message once per unique key.

    The callable maintains an LRU set of keys limited by ``maxsize`` to
    preserve coherence while avoiding unbounded growth. New keys trigger
    a warning with their associated values, repeated keys are ignored.
    ``clear()`` resets the tracked keys, aiding controlled tests.
    """

    def __init__(self, logger: logging.Logger, msg: str, *, maxsize: int = 1024) -> None:
        self._logger = logger
        self._msg = msg
        self._seen: LRUCache[Hashable, None] = LRUCache(maxsize)
        self._lock = threading.Lock()

    def __call__(self, mapping: Mapping[Hashable, Any]) -> None:
        new: dict[Hashable, Any] = {}
        with self._lock:
            for k, v in mapping.items():
                if k in self._seen:
                    self._seen[k]
                else:
                    self._seen[k] = None
                    new[k] = v
        if new:
            self._logger.warning(self._msg, new)

    def clear(self) -> None:
        """Reset tracked keys."""
        with self._lock:
            self._seen.clear()


def warn_once(
    logger: logging.Logger,
    msg: str,
    *,
    maxsize: int = 1024,
) -> WarnOnce:
    """Return a :class:`WarnOnce` logger."""
    return WarnOnce(logger, msg, maxsize=maxsize)
