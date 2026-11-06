"""Finite floating-point arithmetic shared by quotient diagnostics.

Model-specific input admission belongs to each caller. These helpers preserve
scale-safe norms and reject arithmetic overflow instead of reporting infinite
closure residuals. They do not provide exact-real closure certificates.
"""

from __future__ import annotations

import numpy as np


def _finite_product(left: np.ndarray, right: np.ndarray, name: str) -> np.ndarray:
    """Multiply finite arrays or reject an unrepresentable result explicitly."""
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            result = left @ right
    except (FloatingPointError, OverflowError) as exc:
        raise ValueError(f"{name} exceeds finite floating-point range") from exc
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} exceeds finite floating-point range")
    return result


def _finite_difference(left: np.ndarray, right: np.ndarray, name: str) -> np.ndarray:
    """Subtract finite arrays without allowing an infinite residual."""
    try:
        with np.errstate(over="raise", invalid="raise"):
            result = left - right
    except (FloatingPointError, OverflowError) as exc:
        raise ValueError(f"{name} exceeds finite floating-point range") from exc
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} exceeds finite floating-point range")
    return result


def _finite_norm(value: np.ndarray, *, matrix: bool, name: str) -> float:
    """Return a scale-safe 2-norm, rejecting an unrepresentable norm."""
    array = np.asarray(value, dtype=float)
    scale = float(np.max(np.abs(array), initial=0.0))
    if scale == 0.0:
        return 0.0
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            normalized = array / scale
            norm = float(np.linalg.norm(normalized, 2 if matrix else None))
            result = scale * norm
    except (FloatingPointError, OverflowError, np.linalg.LinAlgError) as exc:
        raise ValueError(f"{name} exceeds finite floating-point range") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} exceeds finite floating-point range")
    return result
