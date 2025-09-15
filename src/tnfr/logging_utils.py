"""Logging utilities for TNFR.

Centralises creation of module-specific loggers so that all TNFR
modules share a consistent configuration.  ``get_module_logger`` is
retained as a lightweight alias to ``get_logger`` for backwards
compatibility.
"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from typing import Any, Hashable, Mapping

__all__ = ("_configure_root", "get_logger", "get_module_logger", "WarnOnce", "warn_once")

_LOGGING_CONFIGURED = False


def _configure_root() -> None:
    """Ensure the root logger has handlers and a default format."""

    global _LOGGING_CONFIGURED
    if _LOGGING_CONFIGURED:
        return

    root = logging.getLogger()
    if not root.handlers:
        kwargs = {"format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"}
        if root.level == logging.NOTSET:
            kwargs["level"] = logging.INFO
        logging.basicConfig(**kwargs)

    _LOGGING_CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a module-specific logger."""
    _configure_root()
    return logging.getLogger(name)


# Backwards compatibility --------------------------------------------------

get_module_logger = get_logger


class WarnOnce:
    """Log a message once per unique key.

    The callable maintains an LRU set of keys limited by ``maxsize`` using
    an :class:`collections.OrderedDict` to preserve coherence while avoiding
    unbounded growth. New keys trigger a warning with their associated
    values, repeated keys are ignored. ``clear()`` resets the tracked keys,
    aiding controlled tests.
    """

    def __init__(self, logger: logging.Logger, msg: str, *, maxsize: int = 1024) -> None:
        self._logger = logger
        self._msg = msg
        self._maxsize = maxsize
        self._seen: OrderedDict[Hashable, None] = OrderedDict()
        self._lock = threading.Lock()

    def __call__(self, mapping: Mapping[Hashable, Any]) -> None:
        new: dict[Hashable, Any] = {}
        with self._lock:
            for k, v in mapping.items():
                if k not in self._seen:
                    self._seen[k] = None
                    self._seen.move_to_end(k)
                    new[k] = v
                    if len(self._seen) > self._maxsize:
                        self._seen.popitem(last=False)
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
