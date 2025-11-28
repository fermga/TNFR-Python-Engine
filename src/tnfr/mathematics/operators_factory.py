"""Factory helpers to assemble TNFR coherence and frequency operators."""

from __future__ import annotations

import numpy as np

from .backend import ensure_array, ensure_numpy, get_backend
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
    """Return a Hermitian positive semidefinite :class:`CoherenceOperator`.

    This factory validates inputs, ensures structural invariants (Hermiticity
    and positive semi-definiteness), and integrates with the TNFR backend
    abstraction layer.

    Parameters
    ----------
    dim : int
        Dimensionality of the operator's Hilbert space. Must be positive.
    spectrum : np.ndarray | None, optional
        Custom eigenvalue spectrum. If None, uses uniform c_min values.
        Must be real-valued and match dimension.
    c_min : float, optional
        Minimum coherence threshold for default spectrum (default: 0.1).

    Returns
    -------
    CoherenceOperator
        Validated coherence operator with backend-native arrays.

    Raises
    ------
    ValueError
        If dimension is invalid, spectrum has wrong shape, or operator
        violates Hermiticity/PSD constraints.
    """

    dimension = _validate_dimension(dim)
    if not np.isfinite(c_min):
        raise ValueError("Coherence threshold ``c_min`` must be finite.")

    backend = get_backend()

    if spectrum is None:
        eigenvalues_backend = ensure_array(
            np.full(dimension, float(c_min), dtype=float), backend=backend
        )
    else:
        eigenvalues_backend = ensure_array(spectrum, dtype=np.complex128, backend=backend)
        eigenvalues_np = ensure_numpy(eigenvalues_backend, backend=backend)
        if eigenvalues_np.ndim != 1:
            raise ValueError("Coherence spectrum must be one-dimensional.")
        if eigenvalues_np.shape[0] != dimension:
            raise ValueError("Coherence spectrum size must match operator dimension.")
        if np.any(np.abs(eigenvalues_np.imag) > _ATOL):
            raise ValueError("Coherence spectrum must be real-valued within tolerance.")
        eigenvalues_backend = ensure_array(
            eigenvalues_np.real.astype(float, copy=False), backend=backend
        )

    operator = CoherenceOperator(eigenvalues_backend, c_min=c_min, backend=backend)
    if not operator.is_hermitian(atol=_ATOL):
        raise ValueError("Coherence operator must be Hermitian.")
    if not operator.is_positive_semidefinite(atol=_ATOL):
        raise ValueError("Coherence operator must be positive semidefinite.")
    return operator


def make_frequency_operator(matrix: np.ndarray) -> FrequencyOperator:
    """Return a Hermitian PSD :class:`FrequencyOperator` from ``matrix``.

    This factory validates the input matrix for Hermiticity and constructs
    a frequency operator that enforces positive semi-definiteness.

    Parameters
    ----------
    matrix : np.ndarray
        Square Hermitian matrix representing the frequency operator.
        Must be complex128 compatible.

    Returns
    -------
    FrequencyOperator
        Validated frequency operator with backend-native arrays.

    Raises
    ------
    ValueError
        If matrix is not square, not Hermitian, or not positive semidefinite.
    """

    backend = get_backend()
    array_backend = ensure_array(matrix, dtype=np.complex128, backend=backend)
    array_np = ensure_numpy(array_backend, backend=backend)
    if array_np.ndim != 2 or array_np.shape[0] != array_np.shape[1]:
        raise ValueError("Frequency operator matrix must be square.")
    if not np.allclose(array_np, array_np.conj().T, atol=_ATOL):
        raise ValueError("Frequency operator must be Hermitian within tolerance.")

    operator = FrequencyOperator(array_backend, backend=backend)
    if not operator.is_positive_semidefinite(atol=_ATOL):
        raise ValueError("Frequency operator must be positive semidefinite.")
    return operator
