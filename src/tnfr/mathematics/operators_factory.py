"""Factory helpers to assemble TNFR coherence and frequency operators."""

from __future__ import annotations

from ..constants.canonical import MATH_COHERENCE_MIN_CANONICAL
from ..errors import TNFRValueError
from .backend import ensure_array, ensure_numpy, get_backend
from .operators import CoherenceOperator, FrequencyOperator
from .unified_numerical import np

__all__ = ["make_coherence_operator", "make_frequency_operator"]

_ATOL = 1e-9


def _validate_dimension(dim: int) -> int:
    if int(dim) != dim:
        raise TNFRValueError(
            "Operator dimension must be an integer.",
            context={"dimension": dim},
            suggestion="Provide an integer dimension.",
        )
    if dim <= 0:
        raise TNFRValueError(
            "Operator dimension must be strictly positive.",
            context={"dimension": dim},
            suggestion="Provide a positive integer dimension.",
        )
    return int(dim)


def make_coherence_operator(
    dim: int,
    *,
    spectrum: np.ndarray | None = None,
    c_min: float = MATH_COHERENCE_MIN_CANONICAL,
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
    TNFRValueError
        If dimension is invalid, spectrum has wrong shape, or operator
        violates Hermiticity/PSD constraints.
    """

    dimension = _validate_dimension(dim)
    if not np.isfinite(c_min):
        raise TNFRValueError(
            "Coherence threshold ``c_min`` must be finite.",
            context={"c_min": c_min},
            suggestion="Provide a finite value for c_min.",
        )

    backend = get_backend()

    if spectrum is None:
        eigenvalues_backend = ensure_array(
            np.full(dimension, float(c_min), dtype=float), backend=backend
        )
    else:
        eigenvalues_backend = ensure_array(
            spectrum, dtype=np.complex128, backend=backend
        )
        eigenvalues_np = ensure_numpy(eigenvalues_backend, backend=backend)
        if eigenvalues_np.ndim != 1:
            raise TNFRValueError(
                "Coherence spectrum must be one-dimensional.",
                context={"ndim": eigenvalues_np.ndim},
                suggestion="Provide a 1D spectrum array.",
            )
        if eigenvalues_np.shape[0] != dimension:
            raise TNFRValueError(
                "Coherence spectrum size must match operator dimension.",
                context={
                    "spectrum_size": eigenvalues_np.shape[0],
                    "dimension": dimension,
                },
                suggestion="Ensure spectrum size matches dimension.",
            )
        if np.any(np.abs(eigenvalues_np.imag) > _ATOL):
            raise TNFRValueError(
                "Coherence spectrum must be real-valued within tolerance.",
                context={
                    "max_imag": float(np.max(np.abs(eigenvalues_np.imag))),
                    "atol": _ATOL,
                },
                suggestion="Ensure spectrum is real-valued.",
            )
        eigenvalues_backend = ensure_array(
            eigenvalues_np.real.astype(float, copy=False), backend=backend
        )

    operator = CoherenceOperator(eigenvalues_backend, c_min=c_min, backend=backend)
    if not operator.is_hermitian(atol=_ATOL):
        raise TNFRValueError(
            "Coherence operator must be Hermitian.",
            context={"is_hermitian": False},
            suggestion="Ensure the operator is Hermitian.",
        )
    if not operator.is_positive_semidefinite(atol=_ATOL):
        raise TNFRValueError(
            "Coherence operator must be positive semidefinite.",
            context={"is_psd": False},
            suggestion="Ensure the operator is positive semidefinite.",
        )
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
    TNFRValueError
        If matrix is not square, not Hermitian, or not positive semidefinite.
    """

    backend = get_backend()
    array_backend = ensure_array(matrix, dtype=np.complex128, backend=backend)
    array_np = ensure_numpy(array_backend, backend=backend)
    if array_np.ndim != 2 or array_np.shape[0] != array_np.shape[1]:
        raise TNFRValueError(
            "Frequency operator matrix must be square.",
            context={"shape": array_np.shape},
            suggestion="Provide a square matrix.",
        )
    if not np.allclose(array_np, array_np.conj().T, atol=_ATOL):
        raise TNFRValueError(
            "Frequency operator must be Hermitian within tolerance.",
            context={"atol": _ATOL},
            suggestion="Ensure the matrix is Hermitian.",
        )

    operator = FrequencyOperator(array_backend, backend=backend)
    if not operator.is_positive_semidefinite(atol=_ATOL):
        raise TNFRValueError(
            "Frequency operator must be positive semidefinite.",
            context={"is_psd": False},
            suggestion="Ensure the operator is positive semidefinite.",
        )
    return operator
