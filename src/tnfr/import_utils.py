"""Helpers for optional imports and cached access to heavy modules.

This module centralises caching for optional dependencies. It exposes
:func:`cached_import`, backed by a small :func:`functools.lru_cache`, alongside a
light-weight registry that tracks failed imports and warnings. Use
:func:`prune_failed_imports` or ``cached_import.cache_clear`` to reset state when
new packages become available at runtime.
"""

from __future__ import annotations

import importlib
import warnings
from collections import OrderedDict
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Callable, Literal, MutableMapping
import threading

from .logging_utils import get_logger

__all__ = (
    "cached_import",
    "get_numpy",
    "get_nodonx",
    "prune_failed_imports",
    "IMPORT_LOG",
)


logger = get_logger(__name__)


def _emit(message: str, mode: Literal["warn", "log", "both"]) -> None:
    """Emit ``message`` via :mod:`warnings`, logger or both."""

    if mode in ("warn", "both"):
        warnings.warn(message, RuntimeWarning, stacklevel=2)
    if mode in ("log", "both"):
        logger.warning(message)


EMIT_MAP: dict[str, Callable[[str], None]] = {
    "warn": lambda msg: _emit(msg, "warn"),
    "log": lambda msg: _emit(msg, "log"),
    "both": lambda msg: _emit(msg, "both"),
}


def _format_failure_message(module: str, attr: str | None, err: Exception) -> str:
    """Return a standardised failure message."""

    return (
        f"Failed to import module '{module}': {err}"
        if isinstance(err, ImportError)
        else f"Module '{module}' has no attribute '{attr}': {err}"
    )


_FAILED_IMPORT_LIMIT = 128
_DEFAULT_CACHE_SIZE = 128
_FAIL = object()
_MANUAL_CACHE_LOCK = threading.Lock()


@dataclass(slots=True)
class ImportRegistry:
    """Process-wide registry tracking failed imports and emitted warnings."""

    limit: int = _FAILED_IMPORT_LIMIT
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


@lru_cache(maxsize=_DEFAULT_CACHE_SIZE)
def _import_cached(module_name: str, attr: str | None) -> tuple[bool, Any]:
    """Import ``module_name`` (and optional ``attr``) capturing failures."""

    try:
        module = importlib.import_module(module_name)
        obj = getattr(module, attr) if attr else module
    except (ImportError, AttributeError) as exc:
        return False, exc
    return True, obj


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
    cache: MutableMapping[str, Any] | None = None,
    lock: threading.Lock | None = None,
    emit: Literal["warn", "log", "both"] = "warn",
) -> Any | None:
    """Import ``module_name`` (and optional ``attr``) with caching and fallback.

    Parameters
    ----------
    module_name:
        Module to import.
    attr:
        Optional attribute to fetch from the module.
    fallback:
        Value returned when the import fails.
    cache:
        Custom mapping used to store cached results. When ``None`` a process-wide
        :func:`functools.lru_cache` is used.
    lock:
        Lock guarding ``cache`` when a custom mapping is supplied.
    emit:
        Destination for warnings emitted on failure (``"warn"``/``"log"``/``"both"``).
    """

    key = module_name if attr is None else f"{module_name}.{attr}"
    if cache is None:
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

    lock = lock or _MANUAL_CACHE_LOCK
    with lock:
        try:
            value = cache[key]
        except KeyError:
            pass
        else:
            return fallback if value is _FAIL else value

    try:
        module = importlib.import_module(module_name)
        obj = getattr(module, attr) if attr else module
    except (ImportError, AttributeError) as exc:
        include_module = isinstance(exc, ImportError)
        _warn_failure(module_name, attr, exc, emit=emit)
        _IMPORT_STATE.record_failure(key, module=module_name if include_module else None)
        with lock:
            cache[key] = _FAIL
        return fallback

    with lock:
        cache[key] = obj
    _IMPORT_STATE.discard(key)
    if attr is not None:
        _IMPORT_STATE.discard(module_name)
    return obj


def _clear_default_cache() -> None:
    global _NP_MISSING_LOGGED

    _import_cached.cache_clear()
    _NP_MISSING_LOGGED = False


cached_import.cache_clear = _clear_default_cache  # type: ignore[attr-defined]


_NP_MISSING_LOGGED = False


def get_numpy() -> Any | None:
    """Return the cached :mod:`numpy` module when available.

    Import attempts are delegated to :func:`cached_import`, which already caches
    successes and failures. A lightweight flag suppresses duplicate debug logs
    when :mod:`numpy` is unavailable so callers can repeatedly probe without
    spamming the logger.
    """

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
