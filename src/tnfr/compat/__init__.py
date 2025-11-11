"""Compatibility layer for optional dependencies.

This module provides lightweight stubs and fallback interfaces for optional
dependencies like numpy, matplotlib, and jsonschema. When these packages are
not installed, the stubs allow type checking and imports to succeed without
runtime errors, while maintaining TNFR semantic clarity.

TYPE_CHECKING guards
--------------------
Use ``if TYPE_CHECKING:`` blocks to import optional typing-only dependencies.
At runtime, fallback stubs are used instead.

Example::

    from typing import TYPE_CHECKING
    if TYPE_CHECKING:
        import numpy as np
    else:
        from tnfr.compat import numpy_stub as np
"""

from __future__ import annotations

__all__ = [
    "get_numpy_or_stub",
    "get_matplotlib_or_stub",
    "get_jsonschema_or_stub",
    "numpy_stub",
    "matplotlib_stub",
    "jsonschema_stub",
]

from typing import Any

# Import stubs so they're available as module attributes
from . import numpy_stub, matplotlib_stub, jsonschema_stub


def get_numpy_or_stub() -> Any:
    """Return numpy module if available, otherwise return a stub.

    Returns
    -------
    module or stub
        The actual numpy module when installed, or a minimal stub providing
        basic type compatibility for type checking.
    """
    try:
        import numpy

        return numpy
    except ImportError:
        return numpy_stub


def get_matplotlib_or_stub() -> Any:
    """Return matplotlib module if available, otherwise return a stub.

    Returns
    -------
    module or stub
        The actual matplotlib module when installed, or a minimal stub.
    """
    try:
        import matplotlib

        return matplotlib
    except ImportError:
        return matplotlib_stub


def get_jsonschema_or_stub() -> Any:
    """Return jsonschema module if available, otherwise return a stub.

    Returns
    -------
    module or stub
        The actual jsonschema module when installed, or a minimal stub.
    """
    try:
        import jsonschema

        return jsonschema
    except ImportError:
        return jsonschema_stub
