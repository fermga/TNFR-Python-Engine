"""Utilidades para importaciones opcionales."""

from __future__ import annotations

import importlib
import warnings
import logging
from functools import lru_cache
from typing import Any
from collections import deque
import threading

__all__ = ["optional_import", "get_numpy", "import_nodonx"]


logger = logging.getLogger(__name__)


_FAILED_IMPORT_LIMIT = 128  # keep only this many recent failures
_FAILED_IMPORTS: deque[str] = deque(maxlen=_FAILED_IMPORT_LIMIT)
_FAILED_IMPORTS_LOCK = threading.Lock()


def _warn_failure(module: str, attr: str | None, err: Exception) -> None:
    msg = (
        f"Failed to import module '{module}': {err}" if isinstance(err, ImportError)
        else f"Module '{module}' has no attribute '{attr}': {err}"
    )
    warnings.warn(msg, RuntimeWarning, stacklevel=2)


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
    with _FAILED_IMPORTS_LOCK:
        previously_failed = name in _FAILED_IMPORTS or module_name in _FAILED_IMPORTS
    try:
        module = importlib.import_module(module_name)
        obj = getattr(module, attr) if attr else module
        with _FAILED_IMPORTS_LOCK:
            for item in (name, module_name):
                try:
                    _FAILED_IMPORTS.remove(item)
                except ValueError:
                    pass
        return obj
    except (ImportError, AttributeError) as e:
        if not previously_failed:
            _warn_failure(module_name, attr, e)
        with _FAILED_IMPORTS_LOCK:
            if isinstance(e, ImportError):
                if module_name not in _FAILED_IMPORTS:
                    _FAILED_IMPORTS.append(module_name)
                if name not in _FAILED_IMPORTS:
                    _FAILED_IMPORTS.append(name)
            elif name not in _FAILED_IMPORTS:
                _FAILED_IMPORTS.append(name)
    return fallback


_optional_import_cache_clear = optional_import.cache_clear


def _cache_clear() -> None:
    """Clear ``optional_import`` cache and failure records."""
    _optional_import_cache_clear()
    with _FAILED_IMPORTS_LOCK:
        _FAILED_IMPORTS.clear()


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
