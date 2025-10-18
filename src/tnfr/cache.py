"""Deprecated shim re-exporting :mod:`tnfr.utils.cache`."""

from __future__ import annotations

import warnings

from .utils.cache import *  # noqa: F401,F403
from .utils.cache import __all__  # type: ignore  # re-export explicit names

warnings.warn(
    "tnfr.cache is deprecated and will be removed in a future release; "
    "import from tnfr.utils.cache instead.",
    DeprecationWarning,
    stacklevel=2,
)
