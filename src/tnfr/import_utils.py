"""Utilidades para importaciones opcionales."""

from __future__ import annotations

import importlib
import warnings
import logging
from functools import lru_cache
from typing import Any
import threading

__all__ = ["optional_import", "get_numpy", "import_nodonx"]


logger = logging.getLogger(__name__)


_FAILED_IMPORTS: set[str] = set()
_FAILED_IMPORTS_LOCK = threading.Lock()


def _warn_failure(module: str, attr: str | None, err: Exception) -> None:
    msg = (
        f"Failed to import module '{module}': {err}" if isinstance(err, ImportError)
        else f"Module '{module}' has no attribute '{attr}': {err}"
    )
    warnings.warn(msg, RuntimeWarning, stacklevel=2)


@lru_cache(maxsize=None)
def optional_import(name: str, fallback: Any | None = None) -> Any | None:
    """Import ``name`` returning ``fallback`` if it fails.

    This function is thread-safe: concurrent failures are recorded in a shared
    set protected by a lock to avoid race conditions.

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
    """

    module_name, attr = (name.rsplit(".", 1) + [None])[:2]
    with _FAILED_IMPORTS_LOCK:
        previously_failed = name in _FAILED_IMPORTS or module_name in _FAILED_IMPORTS
    try:
        module = importlib.import_module(module_name)
        obj = getattr(module, attr) if attr else module
        with _FAILED_IMPORTS_LOCK:
            _FAILED_IMPORTS.discard(name)
            _FAILED_IMPORTS.discard(module_name)
        return obj
    except (ImportError, AttributeError) as e:
        if not previously_failed:
            _warn_failure(module_name, attr, e)
        with _FAILED_IMPORTS_LOCK:
            if isinstance(e, ImportError):
                _FAILED_IMPORTS.update({name, module_name})
            else:
                _FAILED_IMPORTS.add(name)
    return fallback


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
