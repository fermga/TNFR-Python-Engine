"""Utilidades para importaciones opcionales."""

from __future__ import annotations

import importlib
import warnings
from functools import lru_cache
from typing import Any
from collections import OrderedDict
from dataclasses import dataclass, field
import threading
from .logging_utils import get_logger

__all__ = ["optional_import", "get_numpy", "import_nodonx"]


logger = get_logger(__name__)


_FAILED_IMPORT_LIMIT = 128  # keep only this many recent failures


@dataclass
class _ImportState:
    failed: OrderedDict[str, None] = field(default_factory=OrderedDict)
    lock: threading.Lock = field(default_factory=threading.Lock)
    limit: int = _FAILED_IMPORT_LIMIT

    def record(self, item: str) -> None:
        if item in self.failed:
            return
        self.failed[item] = None
        if len(self.failed) > self.limit:
            self.failed.popitem(last=False)

    def discard(self, item: str) -> None:
        self.failed.pop(item, None)

    def __contains__(self, item: str) -> bool:  # pragma: no cover - trivial
        return item in self.failed

    def clear(self) -> None:
        self.failed.clear()


_IMPORT_STATE = _ImportState()


_WARNED_MODULES: set[str] = set()
_WARNED_LOCK = threading.Lock()


def _warn_failure(module: str, attr: str | None, err: Exception) -> None:
    msg = (
        f"Failed to import module '{module}': {err}" if isinstance(err, ImportError)
        else f"Module '{module}' has no attribute '{attr}': {err}"
    )
    with _WARNED_LOCK:
        first = module not in _WARNED_MODULES
        if first:
            _WARNED_MODULES.add(module)
    if first:
        warnings.warn(msg, RuntimeWarning, stacklevel=2)
    else:
        warnings.warn(msg, RuntimeWarning)


@lru_cache(maxsize=128)
def optional_import(name: str, fallback: Any | None = None) -> Any | None:
    """Import ``name`` returning ``fallback`` if it fails.

    This function is thread-safe: concurrent failures are recorded in a bounded
    deque protected by a lock to avoid race conditions.

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
    attribute does not exist. In both cases a warning is emitted.
    Use ``optional_import.cache_clear()`` to reset the internal cache and
    failure registry after installing new optional dependencies.
    """

    module_name, attr = (name.rsplit(".", 1) + [None])[:2]
    try:
        module = importlib.import_module(module_name)
        obj = getattr(module, attr) if attr else module
        with _IMPORT_STATE.lock:
            for item in (name, module_name):
                _IMPORT_STATE.discard(item)
        with _WARNED_LOCK:
            _WARNED_MODULES.discard(module_name)
        return obj
    except (ImportError, AttributeError) as e:
        _warn_failure(module_name, attr, e)
        with _IMPORT_STATE.lock:
            if isinstance(e, ImportError):
                _IMPORT_STATE.record(module_name)
                _IMPORT_STATE.record(name)
            else:
                _IMPORT_STATE.record(name)
    return fallback


_optional_import_cache_clear = optional_import.cache_clear


def _cache_clear() -> None:
    """Clear ``optional_import`` cache and failure records."""
    _optional_import_cache_clear()
    with _IMPORT_STATE.lock:
        _IMPORT_STATE.clear()
    with _WARNED_LOCK:
        _WARNED_MODULES.clear()


optional_import.cache_clear = _cache_clear  # type: ignore[attr-defined]


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
        log(
            "Failed to import numpy; continuing in non-vectorised mode"
        )
    return module


@lru_cache(maxsize=1)
def import_nodonx():
    """Lazily import :class:`NodoNX` to avoid circular dependencies."""
    from .node import NodoNX

    return NodoNX
