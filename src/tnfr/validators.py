"""Compatibility shim for :mod:`tnfr.utils.validators`."""

from __future__ import annotations

import warnings

from .utils.validators import run_validators, validate_window

__all__ = ("validate_window", "run_validators")

warnings.warn(
    "'tnfr.validators' is deprecated; import from 'tnfr.utils.validators' instead.",
    DeprecationWarning,
    stacklevel=2,
)
