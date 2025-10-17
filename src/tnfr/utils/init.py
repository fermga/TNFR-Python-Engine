"""Core logging and import helpers for :mod:`tnfr`.

This module merges the functionality that historically lived in
``tnfr.logging_utils`` and ``tnfr.import_utils``.  The behaviour is kept
identical so downstream consumers can keep relying on the same APIs while
benefiting from a consolidated entry point under :mod:`tnfr.utils`.
"""

from __future__ import annotations

import importlib
import logging
import threading
import warnings
from collections import OrderedDict
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Callable, Hashable, Literal, Mapping

__all__ = (
    "_configure_root",
    "cached_import",
    "get_logger",
    "get_numpy",
    "get_nodonx",
    "prune_failed_imports",
    "WarnOnce",
    "warn_once",
    "IMPORT_LOG",
    "EMIT_MAP",
    "_warn_failure",
    "_IMPORT_STATE",
    "_reset_logging_state",
    "_reset_import_state",
    "_FAILED_IMPORT_LIMIT",
    "_DEFAULT_CACHE_SIZE",
)


_LOGGING_CONFIGURED = False


def _reset_logging_state() -> None:
    """Reset cached logging configuration state."""

    global _LOGGING_CONFIGURED
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


_FAILED_IMPORT_LIMIT = 128
_DEFAULT_CACHE_SIZE = 128


@dataclass(slots=True)
class ImportRegistry:
    """Process-wide registry tracking failed imports and emitted warnings."""

    limit: int = 128
    failed: OrderedDict[str, None] = field(default_factory=OrderedDict)
    warned: set[str] = field(default_factory=set)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def _insert(self, key: str) -> None:
        self.failed[key] = None
        self.failed.move_to_end(key)
        while len(self.failed) > self.limit:
            self.failed.popitem(last=False)

    def record_failure(self, key: str, *, module: str | None = None) -> None:
        """Record ``key`` and, optionally, ``module`` as failed imports."""

        with self.lock:
            self._insert(key)
            if module and module != key:
                self._insert(module)

    def discard(self, key: str) -> None:
        """Remove ``key`` from the registry and clear its warning state."""

        with self.lock:
            self.failed.pop(key, None)
            self.warned.discard(key)

    def mark_warning(self, module: str) -> bool:
        """Mark ``module`` as warned and return ``True`` if it was new."""

        with self.lock:
            if module in self.warned:
                return False
            self.warned.add(module)
            return True

    def clear(self) -> None:
        """Remove all failure records and warning markers."""

        with self.lock:
            self.failed.clear()
            self.warned.clear()

    def __contains__(self, key: str) -> bool:  # pragma: no cover - trivial
        with self.lock:
            return key in self.failed


_IMPORT_STATE = ImportRegistry()
# Public alias to ease direct introspection in tests and diagnostics.
IMPORT_LOG = _IMPORT_STATE


def _reset_import_state() -> None:
    """Reset cached import tracking structures."""

    global _IMPORT_STATE, IMPORT_LOG
    _IMPORT_STATE = ImportRegistry()
    IMPORT_LOG = _IMPORT_STATE


@lru_cache(maxsize=_DEFAULT_CACHE_SIZE)
def _import_cached(module_name: str, attr: str | None) -> tuple[bool, Any]:
    """Import ``module_name`` (and optional ``attr``) capturing failures."""

    try:
        module = importlib.import_module(module_name)
        obj = getattr(module, attr) if attr else module
    except (ImportError, AttributeError) as exc:
        return False, exc
    return True, obj


logger = get_logger(__name__)


def _format_failure_message(module: str, attr: str | None, err: Exception) -> str:
    """Return a standardised failure message."""

    return (
        f"Failed to import module '{module}': {err}"
        if isinstance(err, ImportError)
        else f"Module '{module}' has no attribute '{attr}': {err}"
    )


EMIT_MAP: dict[str, Callable[[str], None]] = {
    "warn": lambda msg: _emit(msg, "warn"),
    "log": lambda msg: _emit(msg, "log"),
    "both": lambda msg: _emit(msg, "both"),
}


def _emit(message: str, mode: Literal["warn", "log", "both"]) -> None:
    """Emit ``message`` via :mod:`warnings`, logger or both."""

    if mode in ("warn", "both"):
        warnings.warn(message, RuntimeWarning, stacklevel=2)
    if mode in ("log", "both"):
        logger.warning(message)


def _warn_failure(
    module: str,
    attr: str | None,
    err: Exception,
    *,
    emit: Literal["warn", "log", "both"] = "warn",
) -> None:
    """Emit a warning about a failed import."""

    msg = _format_failure_message(module, attr, err)
    if _IMPORT_STATE.mark_warning(module):
        EMIT_MAP[emit](msg)
    else:
        logger.debug(msg)


def cached_import(
    module_name: str,
    attr: str | None = None,
    *,
    fallback: Any | None = None,
    emit: Literal["warn", "log", "both"] = "warn",
) -> Any | None:
    """Import ``module_name`` (and optional ``attr``) with caching and fallback."""

    key = module_name if attr is None else f"{module_name}.{attr}"
    success, result = _import_cached(module_name, attr)
    if success:
        _IMPORT_STATE.discard(key)
        if attr is not None:
            _IMPORT_STATE.discard(module_name)
        return result
    exc = result
    include_module = isinstance(exc, ImportError)
    _warn_failure(module_name, attr, exc, emit=emit)
    _IMPORT_STATE.record_failure(key, module=module_name if include_module else None)
    return fallback


def _clear_default_cache() -> None:
    global _NP_MISSING_LOGGED

    _import_cached.cache_clear()
    _NP_MISSING_LOGGED = False


cached_import.cache_clear = _clear_default_cache  # type: ignore[attr-defined]


_NP_MISSING_LOGGED = False


def get_numpy() -> Any | None:
    """Return the cached :mod:`numpy` module when available."""

    global _NP_MISSING_LOGGED

    np = cached_import("numpy")
    if np is None:
        if not _NP_MISSING_LOGGED:
            logger.debug("Failed to import numpy; continuing in non-vectorised mode")
            _NP_MISSING_LOGGED = True
        return None

    if _NP_MISSING_LOGGED:
        _NP_MISSING_LOGGED = False
    return np


def get_nodonx() -> type | None:
    """Return :class:`tnfr.node.NodoNX` using import caching."""

    return cached_import("tnfr.node", "NodoNX")


def prune_failed_imports() -> None:
    """Clear the registry of recorded import failures and warnings."""

    _IMPORT_STATE.clear()
