"""Dependency-light validation for canonical structural coherence values."""

from __future__ import annotations

import math
from typing import Any


def validate_structural_coherence(
    value: Any, *, name: str = "coherence"
) -> float:
    """Return one finite canonical coherence value in the closed unit interval."""

    if isinstance(value, (bool, str, bytes)) or type(value).__name__ == "bool_":
        raise TypeError(f"{name} must be a finite real scalar, not bool")
    if getattr(value, "ndim", 0) != 0:
        raise TypeError(f"{name} must be a finite real scalar")
    try:
        normalized = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{name} must be a finite real scalar") from exc
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    if not 0.0 <= normalized <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return normalized


__all__ = ["validate_structural_coherence"]