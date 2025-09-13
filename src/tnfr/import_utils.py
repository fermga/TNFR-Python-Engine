"""Helpers for optional imports and cached access to heavy modules.

The module maintains caches for successful imports alongside registries of
failed attempts and previously warned modules. Entries older than
``_FAILED_IMPORT_MAX_AGE`` seconds are pruned automatically; use
``prune_failed_imports`` to trigger cleanup manually and
``clear_optional_import_cache`` to reset both caches and logs.
Additional helpers like :func:`get_nodonx` provide light-weight access to
TNFR-specific structures on demand.
"""

from __future__ import annotations

import importlib
import warnings
from typing import Any, Literal, MutableMapping
from dataclasses import dataclass, field
from cachetools import TTLCache
import threading
import time
from .logging import get_module_logger
from .locking import get_lock

__all__ = (
    "cached_import",
    "optional_numpy",
    "get_nodonx",
    "prune_failed_imports",
    "clear_optional_import_cache",
)


logger = get_module_logger(__name__)


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
        self.last_prune = time.monotonic() if now is None else now

    def record(self, item: str) -> None:
        self.failed[item] = True

    def discard(self, item: str) -> None:
        self.failed.pop(item, None)

    def __contains__(self, item: str) -> bool:  # pragma: no cover - trivial
        return item in self.failed

    def clear(self) -> None:
        self.failed.clear()
        self.last_prune = 0.0


_IMPORT_STATE = _ImportState()


_WARNED_STATE = _ImportState(lock=get_lock("import_warned"))
_WARNED_MODULES = _WARNED_STATE.failed
_WARNED_LOCK = _WARNED_STATE.lock

_DEFAULT_CACHE_SIZE = 128
_DEFAULT_CACHE_TTL = _FAILED_IMPORT_MAX_AGE
_IMPORT_CACHE: TTLCache[str, Any] = TTLCache(
    _DEFAULT_CACHE_SIZE, _DEFAULT_CACHE_TTL, timer=lambda: time.monotonic()
)
_CACHE_LOCK = threading.Lock()
_FAIL = object()


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


def cached_import(
    module_name: str,
    attr: str | None = None,
    *,
    fallback: Any | None = None,
    cache: MutableMapping[str, Any] | None = None,
    ttl: float = _DEFAULT_CACHE_TTL,
    emit: Literal["warn", "log", "both"] = "warn",
) -> Any | None:
    """Import ``module_name`` (and optional ``attr``) with caching and fallback."""

    key = module_name if attr is None else f"{module_name}.{attr}"
    global _IMPORT_CACHE
    if cache is None:
        lock = _CACHE_LOCK
        if ttl != _IMPORT_CACHE.ttl:
            with lock:
                if _IMPORT_CACHE.ttl != ttl:
                    _IMPORT_CACHE = TTLCache(
                        _DEFAULT_CACHE_SIZE, ttl, timer=lambda: time.monotonic()
                    )
        cache = _IMPORT_CACHE
    else:
        lock = threading.Lock()

    with lock:
        try:
            value = cache[key]
            return fallback if value is _FAIL else value
        except KeyError:
            pass

    try:
        module = importlib.import_module(module_name)
        obj = getattr(module, attr) if attr else module
        with lock:
            cache[key] = obj
        with _IMPORT_STATE.lock, _WARNED_STATE.lock:
            for item in (key, module_name):
                _IMPORT_STATE.discard(item)
            _WARNED_STATE.discard(module_name)
        return obj
    except (ImportError, AttributeError) as e:
        _warn_failure(module_name, attr, e, emit=emit)
        with _IMPORT_STATE.lock:
            if isinstance(e, ImportError):
                _IMPORT_STATE.record(module_name)
                _IMPORT_STATE.record(key)
            else:
                _IMPORT_STATE.record(key)
        with lock:
            cache[key] = _FAIL
        return fallback


def optional_numpy(logger: Any) -> Any | None:
    """Return NumPy module or ``None`` if unavailable, logging the failure."""

    np = cached_import("numpy")
    if np is None:
        logger.debug("Failed to import numpy; continuing in non-vectorised mode")
    return np


def get_nodonx() -> type | None:
    """Return :class:`tnfr.node.NodoNX` using import caching.

    The helper centralises access to the lightweight ``NodoNX`` wrapper so
    modules can interact with graph nodes without incurring repeated import
    overhead.

    Returns
    -------
    type | None
        ``NodoNX`` when available, otherwise ``None``.
    """

    return cached_import("tnfr.node", "NodoNX")


def optional_import(name: str, fallback: Any | None = None) -> Any | None:
    """Deprecated wrapper around :func:`cached_import` parsing dotted names.

    This helper will be removed in a future release. Use
    :func:`cached_import` directly instead:

    ``cached_import("pkg", "attr")``.
    """

    warnings.warn(
        "optional_import is deprecated; use cached_import instead",
        DeprecationWarning,
        stacklevel=2,
    )
    module_name, attr = (name.rsplit(".", 1) + [None])[:2]
    return cached_import(module_name, attr, fallback=fallback)


def _clear_cache() -> None:
    with _CACHE_LOCK:
        _IMPORT_CACHE.clear()


cached_import.cache_clear = _clear_cache  # type: ignore[attr-defined]
optional_import.cache_clear = _clear_cache  # type: ignore[attr-defined]


def clear_optional_import_cache() -> None:
    """Deprecated cache reset for :func:`optional_import`.

    The function now clears the internal import-result cache as well as the
    registries tracking failed imports and previously warned modules. Prefer
    calling ``cached_import.cache_clear()`` together with
    :func:`prune_failed_imports` in new code.
    """

    warnings.warn(
        "clear_optional_import_cache is deprecated; use cached_import.cache_clear() "
        "and prune_failed_imports()",
        DeprecationWarning,
        stacklevel=2,
    )
    cached_import.cache_clear()
    with _IMPORT_STATE.lock, _WARNED_STATE.lock:
        _IMPORT_STATE.clear()
        _WARNED_STATE.clear()


def prune_failed_imports() -> None:
    """Prune expired entries from failure and warning registries."""
    now = time.monotonic()
    with _IMPORT_STATE.lock, _WARNED_STATE.lock:
        _IMPORT_STATE.prune(now)
        _WARNED_STATE.prune(now)


