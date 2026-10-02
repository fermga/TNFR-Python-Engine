"""Dependency-light validation for canonical structural coherence values."""

from __future__ import annotations

from typing import Any

from ._exact_time import finite_represented_real


def validate_structural_coherence(value: Any, *, name: str = "coherence") -> float:
    """Admit a real coherence sample without rounding it into the unit interval.

    Reuse runtime scalar admission: booleans, text, non-finite values and
    nonzero sources lost to binary64 zero are invalid. Check the original
    ordered value as well, so an out-of-range rational cannot round to one.
    """
    normalized, _ = finite_represented_real(value, name)
    if not 0 <= value <= 1 or not 0.0 <= normalized <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return normalized


__all__ = ["validate_structural_coherence"]
