"""Helpers for optional imports and cached access to heavy modules.

The module maintains a registry of failed import attempts. Entries older than
``_FAILED_IMPORT_MAX_AGE`` seconds are pruned automatically; use
``prune_failed_imports`` to trigger cleanup manually.
"""

from __future__ import annotations

import importlib
import warnings
from functools import lru_cache
from typing import Any, Literal
from collections import OrderedDict
from dataclasses import dataclass, field
import threading
import time
from .logging_utils import get_logger
from .locking import get_lock

__all__ = (
    "optional_import",
    "get_numpy",
    "import_nodonx",
    "prune_failed_imports",
    "clear_optional_import_cache",
)


logger = get_logger(__name__)


def _emit_warn(message: str) -> None:
    """Emit a warning preserving caller context."""
    warnings.warn(message, RuntimeWarning, stacklevel=2)


def _emit_log(message: str) -> None:
    """Log a warning message using the module logger."""
    logger.warning(message)


def _emit_both(message: str) -> None:
    """Emit a warning and log it."""
    warnings.warn(message, RuntimeWarning, stacklevel=2)
    logger.warning(message)


EMIT_MAP: dict[str, tuple] = {
    "warn": (_emit_warn,),
    "log": (_emit_log,),
    "both": (_emit_both,),
}

_FAILED_IMPORT_LIMIT = 128  # keep only this many recent failures
_FAILED_IMPORT_MAX_AGE = 3600.0  # seconds
_FAILED_IMPORT_PRUNE_INTERVAL = 60.0  # seconds between automatic prunes


@dataclass(slots=True)
class _ImportState:
    failed: OrderedDict[str, float] = field(default_factory=OrderedDict)
    lock: threading.Lock = field(default_factory=threading.Lock)
    limit: int = _FAILED_IMPORT_LIMIT
    max_age: float = _FAILED_IMPORT_MAX_AGE
    last_prune: float = 0.0

    def prune(self, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        expiry = now - self.max_age
        failed = self.failed
        while failed:
            key, timestamp = next(iter(failed.items()))
            if timestamp >= expiry:
                break
            failed.pop(key)
        while len(failed) > self.limit:
            failed.popitem(last=False)

    def record(self, item: str) -> None:
        now = time.monotonic()
        self.failed[item] = now
        self.prune(now)

    def discard(self, item: str) -> None:
        self.failed.pop(item, None)

    def __contains__(self, item: str) -> bool:  # pragma: no cover - trivial
        return item in self.failed

    def clear(self) -> None:
        self.failed.clear()


_IMPORT_STATE = _ImportState()


_WARNED_STATE = _ImportState(lock=get_lock("import_warned"))
_WARNED_MODULES = _WARNED_STATE.failed
_WARNED_LOCK = _WARNED_STATE.lock


def _warn_failure(
    module: str,
    attr: str | None,
    err: Exception,
    *,
    emit: Literal["warn", "log", "both"] = "warn",
) -> None:
    """Emit a warning about a failed import.

    Parameters
    ----------
    module:
        Module name that failed to import.
    attr:
        Optional attribute that was looked up on the module.
    err:
        Exception that was raised during import or attribute access.
    emit:
        Destination for the warning: ``"warn"`` (default) uses
        :func:`warnings.warn`, ``"log"`` uses :func:`logger.warning` and
        ``"both"`` emits to both destinations.
    """
    now = time.monotonic()
    state = _IMPORT_STATE
    if now - state.last_prune >= _FAILED_IMPORT_PRUNE_INTERVAL:
        prune_failed_imports()
    msg = (
        f"Failed to import module '{module}': {err}"
        if isinstance(err, ImportError)
        else f"Module '{module}' has no attribute '{attr}': {err}"
    )
    with _WARNED_STATE.lock:
        first = module not in _WARNED_STATE.failed
        _WARNED_STATE.record(module)

    if not first:
        logger.debug(msg)
        return
    for fn in EMIT_MAP[emit]:
        fn(msg)


@lru_cache(maxsize=128)
def _optional_import_cached(name: str) -> Any | None:
    """Internal helper implementing ``optional_import`` logic without fallback."""

    module_name, attr = (name.rsplit(".", 1) + [None])[:2]
    try:
        module = importlib.import_module(module_name)
        obj = getattr(module, attr) if attr else module
        with _IMPORT_STATE.lock:
            for item in (name, module_name):
                _IMPORT_STATE.discard(item)
        with _WARNED_STATE.lock:
            _WARNED_STATE.discard(module_name)
        return obj
    except (ImportError, AttributeError) as e:
        _warn_failure(module_name, attr, e)
        with _IMPORT_STATE.lock:
            if isinstance(e, ImportError):
                _IMPORT_STATE.record(module_name)
                _IMPORT_STATE.record(name)
            else:
                _IMPORT_STATE.record(name)
    return None


def optional_import(name: str, fallback: Any | None = None) -> Any | None:
    """Import ``name`` returning ``fallback`` if it fails.

    This function is thread-safe: concurrent failures are recorded in a bounded
    registry protected by a lock to avoid race conditions. Results are cached by
    ``name`` only, so ``fallback`` can be any object, even if unhashable.

    ``name`` may refer to a module, submodule or attribute. If the import or
    attribute access fails a warning is emitted and ``fallback`` is returned.

    Parameters
    ----------
    name:
        Fully qualified module, submodule or attribute path.
    fallback:
        Value to return when import fails. Defaults to ``None``.

    Returns
    -------
    Any | None
        Imported object or ``fallback`` if an error occurs.

    Notes
    -----
    ``fallback`` is returned when the module is unavailable or the requested
    attribute does not exist. In both cases a warning is emitted and logged.
    Use :func:`clear_optional_import_cache` to reset the internal cache and
    failure registry after installing new optional dependencies.
    """

    result = _optional_import_cached(name)
    return fallback if result is None else result


# Expose cache management utilities for backward compatibility
optional_import.cache_clear = _optional_import_cached.cache_clear  # type: ignore[attr-defined]


def clear_optional_import_cache() -> None:
    """Clear ``optional_import`` cache, failure records and warning state."""
    optional_import.cache_clear()
    with _IMPORT_STATE.lock:
        _IMPORT_STATE.clear()
    with _WARNED_STATE.lock:
        _WARNED_STATE.clear()


def prune_failed_imports() -> None:
    """Remove expired entries from the failed import registry."""
    now = time.monotonic()
    state = _IMPORT_STATE
    with state.lock:
        state.prune(now)
        state.last_prune = now
    with _WARNED_STATE.lock:
        _WARNED_STATE.prune(now)
        _WARNED_STATE.last_prune = now


@lru_cache(maxsize=1)
def get_numpy(*, warn: bool = False) -> Any | None:
    """Return :mod:`numpy` or ``None`` if unavailable.

    Parameters
    ----------
    warn:
        When ``True`` a warning is logged if import fails; otherwise a
        ``DEBUG`` message is recorded.
    """

    module = optional_import("numpy")
    if module is None:
        log = logger.warning if warn else logger.debug
        log("Failed to import numpy; continuing in non-vectorised mode")
    return module


@lru_cache(maxsize=1)
def import_nodonx():
    """Lazily import :class:`NodoNX` to avoid circular dependencies.

    The import is cached after the first successful call.
    """
    from .node import NodoNX

    return NodoNX
