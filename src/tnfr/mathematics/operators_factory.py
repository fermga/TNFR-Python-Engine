"""Factory helpers for auxiliary spectral-expectation and frequency operators."""

from __future__ import annotations

from .._spectral_expectation import finite_spectral_real, positive_spectral_dimension
from ..constants.canonical import MATH_SPECTRAL_EXPECTATION_FLOOR_DEFAULT
from ..errors import TNFRValueError
from ._complex_arrays import backend_complex_array
from .backend import ensure_numpy, get_backend
from .operators import CoherenceOperator, FrequencyOperator, SpectralExpectationOperator
from .unified_numerical import np

__all__ = [
    "make_spectral_expectation_operator",
    "make_coherence_operator",
    "make_frequency_operator",
]

_ATOL = 1e-9


def make_spectral_expectation_operator(
    dim: int,
    *,
    spectrum: np.ndarray | None = None,
    expectation_floor: float = MATH_SPECTRAL_EXPECTATION_FLOOR_DEFAULT,
) -> SpectralExpectationOperator:
    """Return a positive-semidefinite spectral expectation operator.

    The resulting Hermitian expectation is an auxiliary real observable in the
    supplied spectrum's units.  It is unbounded above and does not represent or
    certify canonical structural ``C(t)``.

    Parameters
    ----------
    dim : int
        Dimensionality of the operator's Hilbert space. Must be positive.
    spectrum : np.ndarray | None, optional
        Custom eigenvalue spectrum. If None, uses uniform
        expectation-floor values.
        Must be real-valued and match dimension.
    expectation_floor : float, optional
        Auxiliary comparison floor and default uniform eigenvalue (default: 0.1).

    Returns
    -------
    SpectralExpectationOperator
        Validated auxiliary operator with backend-native arrays.

    Raises
    ------
    TNFRValueError
        If dimension is invalid, spectrum has wrong shape, or operator
        violates Hermiticity/PSD constraints.
    """

    dimension = positive_spectral_dimension(dim, label="Operator dimension")
    expectation_floor = finite_spectral_real(
        expectation_floor, label="Spectral expectation floor"
    )

    backend = get_backend()

    if spectrum is None:
        eigenvalues_backend = np.full(dimension, expectation_floor, dtype=float)
    else:
        eigenvalues_backend = backend_complex_array(
            spectrum, backend=backend, label="Spectral expectation spectrum"
        )
        eigenvalues_np = ensure_numpy(eigenvalues_backend, backend=backend)
        if eigenvalues_np.ndim != 1:
            raise TNFRValueError(
                "Spectral expectation spectrum must be one-dimensional.",
                context={"ndim": eigenvalues_np.ndim},
                suggestion="Provide a 1D spectrum array.",
            )
        if eigenvalues_np.shape[0] != dimension:
            raise TNFRValueError(
                "Spectral expectation spectrum size must match operator dimension.",
                context={
                    "spectrum_size": eigenvalues_np.shape[0],
                    "dimension": dimension,
                },
                suggestion="Ensure spectrum size matches dimension.",
            )
        if np.any(np.abs(eigenvalues_np.imag) > _ATOL):
            raise TNFRValueError(
                "Spectral expectation spectrum must be real-valued within tolerance.",
                context={
                    "max_imag": float(np.max(np.abs(eigenvalues_np.imag))),
                    "atol": _ATOL,
                },
                suggestion="Ensure spectrum is real-valued.",
            )
        eigenvalues_backend = eigenvalues_backend.real

    operator = SpectralExpectationOperator(
        eigenvalues_backend,
        expectation_floor=expectation_floor,
        backend=backend,
    )
    if not operator.is_positive_semidefinite(atol=_ATOL):
        raise TNFRValueError(
            "Spectral expectation operator must be positive semidefinite.",
            context={"is_psd": False},
            suggestion="Ensure the operator is positive semidefinite.",
        )
    return operator


def make_coherence_operator(
    dim: int,
    *,
    spectrum: np.ndarray | None = None,
    c_min: float = MATH_SPECTRAL_EXPECTATION_FLOOR_DEFAULT,
) -> CoherenceOperator:
    """Compatibility factory for an auxiliary spectral expectation operator.

    Parameters
    ----------
    dim : int
        Hilbert-space dimension.
    spectrum : numpy.ndarray, optional
        Real positive-semidefinite spectrum. Values may exceed one.
    c_min : float, optional
        Historical name for the auxiliary spectral expectation floor.

    Returns
    -------
    CoherenceOperator
        The compatibility alias of :class:`SpectralExpectationOperator`.

    Raises
    ------
    TNFRValueError
        If the dimension, floor, or spectrum violates the spectral contract.

    This API does not compute or certify canonical structural ``C(t)``.
    """

    return make_spectral_expectation_operator(
        dim,
        spectrum=spectrum,
        expectation_floor=c_min,
    )


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
    array_backend = backend_complex_array(
        matrix, backend=backend, label="Frequency operator matrix"
    )
    if array_backend.ndim != 2 or array_backend.shape[0] != array_backend.shape[1]:
        raise TNFRValueError(
            "Frequency operator matrix must be square.",
            context={"shape": array_backend.shape},
            suggestion="Provide a square matrix.",
        )

    operator = FrequencyOperator(array_backend, backend=backend)
    if not operator.is_positive_semidefinite(atol=_ATOL):
        raise TNFRValueError(
            "Frequency operator must be positive semidefinite.",
            context={"is_psd": False},
            suggestion="Ensure the operator is positive semidefinite.",
        )
    return operator
