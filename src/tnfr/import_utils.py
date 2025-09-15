"""Helpers for optional imports and cached access to heavy modules.

The module maintains caches for successful imports alongside registries of
failed attempts and previously warned modules. Entries older than
``_FAILED_IMPORT_MAX_AGE`` seconds are pruned automatically; use
``prune_failed_imports`` to trigger cleanup manually or call
``cached_import.cache_clear`` to reset import results.
Additional helpers like :func:`get_nodonx` provide light-weight access to
TNFR-specific structures on demand.
"""

from __future__ import annotations

import importlib
import warnings
from functools import partial
from typing import Any, Callable, Literal, MutableMapping
from dataclasses import dataclass, field
from cachetools import TTLCache
import threading
import time
from .logging_utils import get_logger
from .locking import get_lock

__all__ = (
    "cached_import",
    "optional_numpy",
    "get_numpy",
    "get_nodonx",
    "prune_failed_imports",
)


logger = get_logger(__name__)


def _emit(message: str, mode: Literal["warn", "log", "both"]) -> None:
    """Emit ``message`` via warning, logger or both."""
    if mode in ("warn", "both"):
        warnings.warn(message, RuntimeWarning, stacklevel=2)
    if mode in ("log", "both"):
        logger.warning(message)


EMIT_MAP: dict[str, Callable[[str], None]] = {
    "warn": partial(_emit, mode="warn"),
    "log": partial(_emit, mode="log"),
    "both": partial(_emit, mode="both"),
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
    EMIT_MAP[emit](msg)


def _update_import_cache(ttl: float) -> MutableMapping[str, Any]:
    """Return the shared import cache.

    Updates its TTL when required to maintain deterministic behaviour.
    """
    global _IMPORT_CACHE
    if _IMPORT_CACHE.ttl != ttl:
        with _CACHE_LOCK:
            if _IMPORT_CACHE.ttl != ttl:
                _IMPORT_CACHE = TTLCache(
                    _DEFAULT_CACHE_SIZE, ttl, timer=lambda: time.monotonic()
                )
    return _IMPORT_CACHE


def _get_cache_lock(
    cache: MutableMapping[str, Any] | None,
    lock: threading.Lock | None,
    ttl: float,
) -> tuple[MutableMapping[str, Any], threading.Lock]:
    """Resolve cache and lock arguments for :func:`cached_import`.

    Ensures the shared cache respects the requested ``ttl``.
    """
    if cache is None:
        cache = _update_import_cache(ttl)
        lock = _CACHE_LOCK
    elif lock is None:
        lock = _CACHE_LOCK
    return cache, lock


def _record_failure(key: str, module: str, err: Exception) -> None:
    """Record a failed import attempt in the process-wide registry.

    This keeps deterministic failure accounting across modules.
    """
    with _IMPORT_STATE.lock:
        if isinstance(err, ImportError):
            _IMPORT_STATE.record(module)
            _IMPORT_STATE.record(key)
        else:
            _IMPORT_STATE.record(key)


def cached_import(
    module_name: str,
    attr: str | None = None,
    *,
    fallback: Any | None = None,
    cache: MutableMapping[str, Any] | None = None,
    lock: threading.Lock | None = None,
    ttl: float = _DEFAULT_CACHE_TTL,
    emit: Literal["warn", "log", "both"] = "warn",
) -> Any | None:
    """Import ``module_name`` (and optional ``attr``) with caching and fallback.

    Parameters
    ----------
    module_name:
        Name of the module to import.
    attr:
        Optional attribute to fetch from ``module_name``.
    fallback:
        Value returned when the import fails.
    cache:
        Mapping used to store cached results. When ``None`` the internal
        process-wide cache is used.
    lock:
        Optional :class:`threading.Lock` guarding ``cache``. When using an
        external cache, passing a lock is recommended to avoid repeated
        creations. If omitted, the shared ``_CACHE_LOCK`` is used.
    ttl:
        Time-to-live for entries when using the internal cache.
    emit:
        Destination for warnings emitted on import failures.
    """

    key = module_name if attr is None else f"{module_name}.{attr}"
    cache, lock = _get_cache_lock(cache, lock, ttl)

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
        with lock:
            cache[key] = obj
        with _IMPORT_STATE.lock, _WARNED_STATE.lock:
            for item in (key, module_name):
                _IMPORT_STATE.discard(item)
            _WARNED_STATE.discard(module_name)
        return obj
    except (ImportError, AttributeError) as e:
        _warn_failure(module_name, attr, e, emit=emit)
        _record_failure(key, module_name, e)
        with lock:
            cache[key] = _FAIL
        return fallback


def optional_numpy(logger: Any) -> Any | None:
    """Return NumPy module or ``None`` if unavailable, logging the failure."""

    np = cached_import("numpy")
    if np is None:
        logger.debug("Failed to import numpy; continuing in non-vectorised mode")
    return np


_NP_CACHE_SENTINEL = object()
_NP_CACHE: Any | None | object = _NP_CACHE_SENTINEL


def get_numpy(logger: Any | None = None) -> Any | None:
    """Return cached NumPy module if available."""

    global _NP_CACHE
    if _NP_CACHE is _NP_CACHE_SENTINEL:
        _NP_CACHE = optional_numpy(logger or get_logger(__name__))
    return _NP_CACHE


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


def _clear_cache() -> None:
    with _CACHE_LOCK:
        _IMPORT_CACHE.clear()


cached_import.cache_clear = _clear_cache  # type: ignore[attr-defined]


def prune_failed_imports() -> None:
    """Prune expired entries from failure and warning registries."""
    now = time.monotonic()
    with _IMPORT_STATE.lock, _WARNED_STATE.lock:
        _IMPORT_STATE.prune(now)
        _WARNED_STATE.prune(now)


