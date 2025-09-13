"""Helpers for optional imports and cached access to heavy modules.

The module maintains a registry of failed import attempts. Entries older than
``_FAILED_IMPORT_MAX_AGE`` seconds are pruned automatically; use
``prune_failed_imports`` to trigger cleanup manually.
"""

from __future__ import annotations

import importlib
import warnings
from typing import Any, Literal, MutableMapping
from dataclasses import dataclass, field
from cachetools import TTLCache
import threading
import time
from .logging_utils import get_logger
from .locking import get_lock

__all__ = (
    "cached_import",
    "prune_failed_imports",
    "clear_cached_imports",
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


def _format_failure_message(module: str, attr: str | None, err: Exception) -> str:
    """Return a standardised failure message."""
    return (
        f"Failed to import module '{module}': {err}"
        if isinstance(err, ImportError)
        else f"Module '{module}' has no attribute '{attr}': {err}"
    )


_FAILED_IMPORT_LIMIT = 128  # keep only this many recent failures
_FAILED_IMPORT_MAX_AGE = 3600.0  # seconds
_FAILED_IMPORT_PRUNE_INTERVAL = 60.0  # seconds between automatic prunes


@dataclass(slots=True)
class _ImportState:
    limit: int = _FAILED_IMPORT_LIMIT
    max_age: float = _FAILED_IMPORT_MAX_AGE
    failed: TTLCache = field(init=False)
    lock: threading.Lock = field(default_factory=threading.Lock)
    last_prune: float = 0.0

    def __post_init__(self) -> None:
        self.failed = TTLCache(
            self.limit, self.max_age, timer=lambda: time.monotonic()
        )

    def prune(self, now: float | None = None) -> None:
        self.failed.expire(now)

    def record(self, item: str) -> None:
        self.failed[item] = True

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
    msg = _format_failure_message(module, attr, err)
    with _WARNED_STATE.lock:
        first = module not in _WARNED_STATE.failed
        _WARNED_STATE.record(module)

    if not first:
        logger.debug(msg)
        return
    for fn in EMIT_MAP[emit]:
        fn(msg)


_IMPORT_CACHE: MutableMapping[str, Any] = TTLCache(128, 3600.0)


def cached_import(
    module_name: str,
    attr: str | None = None,
    *,
    cache: MutableMapping[str, Any] | None = _IMPORT_CACHE,
    emit: Literal["warn", "log", "both"] = "warn",
) -> Any | None:
    """Import ``module_name`` and cache the result with optional attribute.

    Parameters
    ----------
    module_name:
        Name of the module to import.
    attr:
        Optional attribute to retrieve from the imported module.
    cache:
        Mapping used to memoise successful imports. Pass ``None`` to disable
        caching for the call.
    emit:
        Destination for warnings when imports fail. See :func:`_warn_failure`.
    """

    key = f"{module_name}.{attr}" if attr else module_name
    if cache is not None:
        try:
            return cache[key]
        except KeyError:
            pass
    try:
        module = importlib.import_module(module_name)
        obj = getattr(module, attr) if attr else module
        with _IMPORT_STATE.lock, _WARNED_STATE.lock:
            for item in (key, module_name):
                _IMPORT_STATE.discard(item)
            _WARNED_STATE.discard(module_name)
        if cache is not None:
            cache[key] = obj
        return obj
    except (ImportError, AttributeError) as e:
        _warn_failure(module_name, attr, e, emit=emit)
        with _IMPORT_STATE.lock:
            if isinstance(e, ImportError):
                _IMPORT_STATE.record(module_name)
                _IMPORT_STATE.record(key)
            else:
                _IMPORT_STATE.record(key)
    return None


def _cache_clear() -> None:
    _IMPORT_CACHE.clear()


cached_import.cache_clear = _cache_clear  # type: ignore[attr-defined]


def clear_cached_imports() -> None:
    """Clear cached imports, failure records and warning state."""
    cached_import.cache_clear()
    with _IMPORT_STATE.lock, _WARNED_STATE.lock:
        _IMPORT_STATE.clear()
        _WARNED_STATE.clear()


def prune_failed_imports() -> None:
    """Remove expired entries from the failed import registry."""
    now = time.monotonic()
    with _IMPORT_STATE.lock, _WARNED_STATE.lock:
        _IMPORT_STATE.failed.expire(now)
        _IMPORT_STATE.last_prune = now
        _WARNED_STATE.failed.expire(now)
        _WARNED_STATE.last_prune = now
