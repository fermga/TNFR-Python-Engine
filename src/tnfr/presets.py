"""Backward compatibility shim for configuration presets."""

from __future__ import annotations

import warnings

from .config.presets import get_preset

warnings.warn(
    "'tnfr.presets' is deprecated; use 'tnfr.config.presets' instead",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ("get_preset",)
