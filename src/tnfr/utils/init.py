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
import weakref
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Hashable, Iterable, Literal, Mapping

__all__ = (
    "_configure_root",
    "cached_import",
    "warm_cached_import",
    "LazyImportProxy",
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


def _import_key(module_name: str, attr: str | None) -> str:
    return module_name if attr is None else f"{module_name}.{attr}"


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


# Successful imports are cached so lazy proxies can resolve once and later
# requests return the concrete object without recreating the proxy. The cache
# stores weak references whenever possible so unused imports can be collected
# after external references disappear.


class _CacheEntry:
    """Container storing either a weak or strong reference to a value."""

    __slots__ = ("_kind", "_value")

    def __init__(
        self,
        value: Any,
        *,
        key: str,
        remover: Callable[[str, weakref.ReferenceType[Any]], None],
    ) -> None:
        try:
            reference = weakref.ref(value, lambda ref, key=key: remover(key, ref))
        except TypeError:
            self._kind = "strong"
            self._value = value
        else:
            self._kind = "weak"
            self._value = reference

    def get(self) -> Any | None:
        if self._kind == "weak":
            return self._value()
        return self._value

    def matches(self, ref: weakref.ReferenceType[Any]) -> bool:
        return self._kind == "weak" and self._value is ref


_CACHE_LOCK = threading.Lock()
_SUCCESS_CACHE: OrderedDict[str, _CacheEntry] = OrderedDict()
_FAILURE_CACHE: OrderedDict[str, Exception] = OrderedDict()


def _remove_success_entry(key: str, ref: weakref.ReferenceType[Any]) -> None:
    with _CACHE_LOCK:
        entry = _SUCCESS_CACHE.get(key)
        if entry is not None and entry.matches(ref):
            _SUCCESS_CACHE.pop(key, None)


def _trim_cache(cache: OrderedDict[str, Any]) -> None:
    while len(cache) > _DEFAULT_CACHE_SIZE:
        cache.popitem(last=False)


def _get_success(key: str) -> Any | None:
    with _CACHE_LOCK:
        entry = _SUCCESS_CACHE.get(key)
        if entry is None:
            return None
        value = entry.get()
        if value is None:
            _SUCCESS_CACHE.pop(key, None)
            return None
        _SUCCESS_CACHE.move_to_end(key)
        return value


def _store_success(key: str, value: Any) -> None:
    entry = _CacheEntry(value, key=key, remover=_remove_success_entry)
    with _CACHE_LOCK:
        _SUCCESS_CACHE[key] = entry
        _SUCCESS_CACHE.move_to_end(key)
        _FAILURE_CACHE.pop(key, None)
        _trim_cache(_SUCCESS_CACHE)


def _get_failure(key: str) -> Exception | None:
    with _CACHE_LOCK:
        exc = _FAILURE_CACHE.get(key)
        if exc is None:
            return None
        _FAILURE_CACHE.move_to_end(key)
        return exc


def _store_failure(key: str, exc: Exception) -> None:
    with _CACHE_LOCK:
        _FAILURE_CACHE[key] = exc
        _FAILURE_CACHE.move_to_end(key)
        _SUCCESS_CACHE.pop(key, None)
        _trim_cache(_FAILURE_CACHE)


def _clear_import_cache() -> None:
    with _CACHE_LOCK:
        _SUCCESS_CACHE.clear()
        _FAILURE_CACHE.clear()


_IMPORT_STATE = ImportRegistry()
# Public alias to ease direct introspection in tests and diagnostics.
IMPORT_LOG = _IMPORT_STATE


def _reset_import_state() -> None:
    """Reset cached import tracking structures."""

    global _IMPORT_STATE, IMPORT_LOG
    _IMPORT_STATE = ImportRegistry()
    IMPORT_LOG = _IMPORT_STATE
    _clear_import_cache()


def _import_cached(module_name: str, attr: str | None) -> tuple[bool, Any]:
    """Import ``module_name`` (and optional ``attr``) capturing failures."""

    key = _import_key(module_name, attr)
    cached_value = _get_success(key)
    if cached_value is not None:
        return True, cached_value

    cached_failure = _get_failure(key)
    if cached_failure is not None:
        return False, cached_failure

    try:
        module = importlib.import_module(module_name)
        obj = getattr(module, attr) if attr else module
    except (ImportError, AttributeError) as exc:
        _store_failure(key, exc)
        return False, exc

    _store_success(key, obj)
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


class LazyImportProxy:
    """Descriptor that defers imports until first use."""

    __slots__ = ("_module", "_attr", "_emit", "_target", "_lock", "_key")

    _UNRESOLVED = object()

    def __init__(
        self,
        module_name: str,
        attr: str | None,
        emit: Literal["warn", "log", "both"],
    ) -> None:
        self._module = module_name
        self._attr = attr
        self._emit = emit
        self._target: Any = self._UNRESOLVED
        self._lock = threading.Lock()
        self._key = _import_key(module_name, attr)

    def _resolve(self) -> Any:
        target = self._target
        if target is not self._UNRESOLVED:
            return target
        with self._lock:
            target = self._target
            if target is self._UNRESOLVED:
                target = _resolve_import(self._module, self._attr, self._emit, None)
                self._target = target
        return target

    def __getattr__(self, item: str) -> Any:
        return getattr(self._resolve(), item)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._resolve()(*args, **kwargs)

    def __bool__(self) -> bool:
        return bool(self._resolve())

    def __repr__(self) -> str:  # pragma: no cover - representation helper
        target = self._target
        if target is self._UNRESOLVED:
            return f"<LazyImportProxy pending={self._key!r}>"
        return repr(target)

    def __str__(self) -> str:  # pragma: no cover - representation helper
        return str(self._resolve())

    def __iter__(self):  # pragma: no cover - passthrough helper
        return iter(self._resolve())


def _resolve_import(
    module_name: str,
    attr: str | None,
    emit: Literal["warn", "log", "both"],
    fallback: Any | None,
) -> Any | None:
    key = _import_key(module_name, attr)
    success, result = _import_cached(module_name, attr)
    if success:
        _IMPORT_STATE.discard(key)
        if attr is not None:
            _IMPORT_STATE.discard(module_name)
        return result

    exc: Exception = result
    include_module = isinstance(exc, ImportError)
    _warn_failure(module_name, attr, exc, emit=emit)
    _IMPORT_STATE.record_failure(key, module=module_name if include_module else None)
    return fallback


def cached_import(
    module_name: str,
    attr: str | None = None,
    *,
    fallback: Any | None = None,
    emit: Literal["warn", "log", "both"] = "warn",
    lazy: bool = False,
) -> Any | None:
    """Import ``module_name`` (and optional ``attr``) with caching and fallback.

    When ``lazy`` is ``True`` the import is deferred until the returned proxy is
    first used. The proxy integrates with the shared cache so subsequent calls
    return the resolved object directly. Passing ``fallback`` disables lazy
    importing because a concrete success or failure value must be returned
    immediately.
    """

    key = _import_key(module_name, attr)

    if lazy and fallback is None:
        cached_obj = _get_success(key)
        if cached_obj is not None:
            return cached_obj
        return LazyImportProxy(module_name, attr, emit)

    return _resolve_import(module_name, attr, emit, fallback)


_ModuleSpec = str | tuple[str, str | None]


def _normalise_warm_specs(
    module: _ModuleSpec | Iterable[_ModuleSpec],
    extra: tuple[_ModuleSpec, ...],
    attr: str | None,
) -> list[tuple[str, str | None]]:
    if attr is not None:
        if extra:
            raise ValueError("'attr' can only be combined with a single module name")
        if not isinstance(module, str):
            raise TypeError("'attr' requires the first argument to be a module name string")
        return [(module, attr)]

    specs: list[_ModuleSpec]
    if extra:
        specs = [module, *extra]
    elif isinstance(module, tuple) and len(module) == 2:
        specs = [module]
    elif isinstance(module, str):
        specs = [module]
    else:
        if isinstance(module, Iterable):
            specs = list(module)
            if not specs:
                raise ValueError("At least one module specification is required")
        else:
            raise TypeError("Unsupported module specification for warm_cached_import")

    normalised: list[tuple[str, str | None]] = []
    for spec in specs:
        if isinstance(spec, str):
            normalised.append((spec, None))
            continue
        if isinstance(spec, tuple) and len(spec) == 2:
            module_name, module_attr = spec
            if not isinstance(module_name, str) or (
                module_attr is not None and not isinstance(module_attr, str)
            ):
                raise TypeError("Invalid module specification for warm_cached_import")
            normalised.append((module_name, module_attr))
            continue
        raise TypeError("Module specifications must be strings or (module, attr) tuples")

    return normalised


def warm_cached_import(
    module: _ModuleSpec | Iterable[_ModuleSpec],
    *extra: _ModuleSpec,
    attr: str | None = None,
    fallback: Any | None = None,
    emit: Literal["warn", "log", "both"] = "warn",
    lazy: bool = False,
) -> Any | dict[str, Any | None]:
    """Pre-populate the import cache for the provided module specifications."""

    specs = _normalise_warm_specs(module, extra, attr)
    results: dict[str, Any | None] = {}
    for module_name, module_attr in specs:
        key = _import_key(module_name, module_attr)
        results[key] = cached_import(
            module_name,
            module_attr,
            fallback=fallback,
            emit=emit,
            lazy=lazy,
        )

    if len(results) == 1:
        return next(iter(results.values()))
    return results


def _clear_default_cache() -> None:
    global _NP_MISSING_LOGGED

    _clear_import_cache()
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
    with _CACHE_LOCK:
        _FAILURE_CACHE.clear()
