"""Compatibility wrapper for value helpers moved to :mod:`tnfr.utils`."""

from __future__ import annotations

import warnings

from .utils.data import convert_value

__all__ = ("convert_value",)

warnings.warn(
    "'tnfr.value_utils' is deprecated; import from 'tnfr.utils.data' instead",
    DeprecationWarning,
    stacklevel=2,
)
