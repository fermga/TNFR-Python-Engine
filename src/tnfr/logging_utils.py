"""Logging utilities for TNFR.

Centralises creation of module-specific loggers so that all TNFR
modules share a consistent configuration.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Hashable, Mapping

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
    """Log a warning only once for each unique key.

    ``WarnOnce`` tracks seen keys in a bounded :class:`set`. When ``maxsize`` is
    reached an arbitrary key is evicted to keep memory usage stable; ordered
    eviction is intentionally avoided to keep the implementation lightweight.
    Instances are callable and accept either a mapping of keys to values or a
    single key/value pair. Passing ``maxsize <= 0`` disables caching and logs on
    every invocation.
    """

    def __init__(self, logger: logging.Logger, msg: str, *, maxsize: int = 1024) -> None:
        self._logger = logger
        self._msg = msg
        self._maxsize = maxsize
        self._seen: set[Hashable] = set()
        self._lock = threading.Lock()

    def _mark_seen(self, key: Hashable) -> bool:
        """Return ``True`` when ``key`` has not been seen before."""

        if self._maxsize <= 0:
            # Caching disabled – always log.
            return True
        if key in self._seen:
            return False
        if len(self._seen) >= self._maxsize:
            # ``set.pop()`` removes an arbitrary element which is acceptable for
            # this lightweight cache.
            self._seen.pop()
        self._seen.add(key)
        return True

    def __call__(
        self,
        data: Mapping[Hashable, Any] | Hashable,
        value: Any | None = None,
    ) -> None:
        """Log new keys found in ``data``.

        ``data`` may be a mapping of keys to payloads or a single key. When
        called with a single key ``value`` customises the payload passed to the
        logging message; the key itself is used when ``value`` is omitted.
        """

        if isinstance(data, Mapping):
            new_items: dict[Hashable, Any] = {}
            with self._lock:
                for key, item_value in data.items():
                    if self._mark_seen(key):
                        new_items[key] = item_value
            if new_items:
                self._logger.warning(self._msg, new_items)
            return

        key = data
        payload = value if value is not None else data
        with self._lock:
            should_log = self._mark_seen(key)
        if should_log:
            self._logger.warning(self._msg, payload)

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
