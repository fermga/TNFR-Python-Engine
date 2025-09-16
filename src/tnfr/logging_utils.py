"""Logging utilities for TNFR.

Centralises creation of module-specific loggers so that all TNFR
modules share a consistent configuration.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Hashable

__all__ = ("_configure_root", "get_logger", "WarnOnce", "warn_once")

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


class WarnOnce:
    """Log a message once per unique ``key``.

    The class keeps track of seen keys using a plain :class:`set` guarded by
    a :class:`threading.Lock`. The cache size is bounded by ``maxsize``; when
    the limit is reached the cache is cleared to keep memory usage bounded
    without maintaining ordering metadata. ``clear()`` resets the cache making
    the instance suitable for deterministic tests.
    """

    def __init__(self, logger: logging.Logger, msg: str, *, maxsize: int | None = 1024) -> None:
        self._logger = logger
        self._msg = msg
        self._maxsize = maxsize
        self._seen: set[Hashable] = set()
        self._lock = threading.Lock()

    def check(self, key: Hashable) -> bool:
        """Return ``True`` if ``key`` has not been seen before.

        When ``maxsize`` is a positive integer the cache is cleared whenever
        the number of tracked keys reaches the limit, preventing unbounded
        growth while favouring simplicity over ordered eviction.
        """

        with self._lock:
            if self._maxsize is not None and self._maxsize <= 0:
                return True
            if key in self._seen:
                return False
            if self._maxsize is not None and len(self._seen) >= self._maxsize:
                self._seen.clear()
            self._seen.add(key)
            return True

    def __call__(self, key: Hashable, *args: Any, **kwargs: Any) -> bool:
        """Emit a warning for ``key`` if it was not previously logged.

        Additional ``*args`` and ``**kwargs`` are forwarded to
        :meth:`logging.Logger.warning`. The method returns ``True`` when the
        warning is emitted and ``False`` otherwise.
        """

        if self.check(key):
            self._logger.warning(self._msg, *args, **kwargs)
            return True
        return False

    def clear(self) -> None:
        """Reset tracked keys."""
        with self._lock:
            self._seen.clear()


def warn_once(
    logger: logging.Logger,
    msg: str,
    *,
    maxsize: int | None = 1024,
) -> WarnOnce:
    """Return a :class:`WarnOnce` helper bound to ``logger`` and ``msg``."""

    return WarnOnce(logger, msg, maxsize=maxsize)
