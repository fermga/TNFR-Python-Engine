"""Factory helpers to assemble TNFR coherence and frequency operators."""
from __future__ import annotations

import numpy as np

from .operators import CoherenceOperator, FrequencyOperator

__all__ = ["make_coherence_operator", "make_frequency_operator"]

_ATOL = 1e-9


def _validate_dimension(dim: int) -> int:
    if int(dim) != dim:
        raise ValueError("Operator dimension must be an integer.")
    if dim <= 0:
        raise ValueError("Operator dimension must be strictly positive.")
    return int(dim)


def make_coherence_operator(
    dim: int,
    *,
    spectrum: np.ndarray | None = None,
    c_min: float = 0.1,
) -> CoherenceOperator:
    """Return a Hermitian positive semidefinite :class:`CoherenceOperator`."""

    dimension = _validate_dimension(dim)
    if not np.isfinite(c_min):
        raise ValueError("Coherence threshold ``c_min`` must be finite.")

    if spectrum is None:
        eigenvalues = np.full(dimension, float(c_min), dtype=float)
    else:
        eigenvalues = np.asarray(spectrum, dtype=np.complex128)
        if eigenvalues.ndim != 1:
            raise ValueError("Coherence spectrum must be one-dimensional.")
        if eigenvalues.shape[0] != dimension:
            raise ValueError("Coherence spectrum size must match operator dimension.")
        if np.any(np.abs(eigenvalues.imag) > _ATOL):
            raise ValueError("Coherence spectrum must be real-valued within tolerance.")
        eigenvalues = eigenvalues.real.astype(float, copy=False)

    operator = CoherenceOperator(eigenvalues, c_min=c_min)
    if not operator.is_hermitian(atol=_ATOL):
        raise ValueError("Coherence operator must be Hermitian.")
    if not operator.is_positive_semidefinite(atol=_ATOL):
        raise ValueError("Coherence operator must be positive semidefinite.")
    return operator


def make_frequency_operator(matrix: np.ndarray) -> FrequencyOperator:
    """Return a Hermitian PSD :class:`FrequencyOperator` from ``matrix``."""

    array = np.asarray(matrix, dtype=np.complex128)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError("Frequency operator matrix must be square.")
    if not np.allclose(array, array.conj().T, atol=_ATOL):
        raise ValueError("Frequency operator must be Hermitian within tolerance.")

    operator = FrequencyOperator(array)
    if not operator.is_positive_semidefinite(atol=_ATOL):
        raise ValueError("Frequency operator must be positive semidefinite.")
    return operator
