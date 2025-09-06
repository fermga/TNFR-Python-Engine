"""Utilidades para importaciones opcionales."""

from __future__ import annotations

import importlib
import warnings
import logging
from functools import lru_cache
from typing import Any

__all__ = ["optional_import", "get_numpy"]


logger = logging.getLogger(__name__)


def optional_import(name: str, fallback: Any | None = None) -> Any | None:
    """Import ``name`` returning ``fallback`` if it fails.

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
    try:
        module = importlib.import_module(module_name)
        return getattr(module, attr) if attr else module
    except ImportError as e:
        warnings.warn(
            f"Failed to import module '{module_name}': {e}",
            RuntimeWarning,
            stacklevel=2,
        )
    except AttributeError as e:
        warnings.warn(
            f"Module '{module_name}' has no attribute '{attr}': {e}",
            RuntimeWarning,
            stacklevel=2,
        )
    return fallback


@lru_cache(maxsize=1)
def get_numpy(*, warn: bool = False) -> Any | None:
    """Return :mod:`numpy` or ``None`` if unavailable.

    Parameters
    ----------
    warn:
        When ``True`` a warning is logged if import fails; otherwise a ``DEBUG``
        message is recorded.
    """

    module = optional_import("numpy")
    if module is None:
        log = logger.warning if warn else logger.debug
        log(
            "Failed to import numpy; continuing in non-vectorised mode"
        )
    return module
