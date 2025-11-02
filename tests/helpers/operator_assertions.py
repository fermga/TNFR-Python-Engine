"""Shared assertions for operator testing.

This module provides reusable assertions for testing
operator properties, spectral analysis, and numerical stability.
"""

from __future__ import annotations

from typing import Any

import pytest

np = pytest.importorskip("numpy")


def assert_operator_hermitian(
    operator: Any,
    *,
    atol: float = 1e-9,
    rtol: float = 1e-9,
) -> None:
    """Assert that operator is Hermitian (self-adjoint).
    
    Verifies O† = O where † denotes conjugate transpose.
    
    Parameters
    ----------
    operator : array-like or operator object
        Operator to test. Can be numpy array or object with .matrix attribute.
    atol : float, default=1e-9
        Absolute tolerance for comparison.
    rtol : float, default=1e-9
        Relative tolerance for comparison.
    
    Raises
    ------
    AssertionError
        If operator is not Hermitian within tolerance.
    
    Examples
    --------
    >>> op = build_delta_nfr(4)
    >>> assert_operator_hermitian(op)
    """
    matrix = _extract_matrix(operator)
    
    assert np.allclose(matrix, matrix.conj().T, atol=atol, rtol=rtol), \
        "Operator is not Hermitian: O† ≠ O"


def assert_operator_positive_semidefinite(
    operator: Any,
    *,
    atol: float = 1e-9,
) -> None:
    """Assert that operator is positive semi-definite (PSD).
    
    Verifies all eigenvalues are non-negative: λᵢ ≥ 0.
    
    Parameters
    ----------
    operator : array-like or operator object
        Operator to test.
    atol : float, default=1e-9
        Absolute tolerance for zero eigenvalues.
    
    Raises
    ------
    AssertionError
        If operator has negative eigenvalues beyond tolerance.
    
    Examples
    --------
    >>> op = CoherenceOperator([1.0, 2.0, 3.0])
    >>> assert_operator_positive_semidefinite(op)
    """
    matrix = _extract_matrix(operator)
    eigenvalues = np.linalg.eigvalsh(matrix)
    
    min_eigenvalue = eigenvalues.min()
    assert min_eigenvalue >= -atol, \
        f"Operator has negative eigenvalue: λ_min = {min_eigenvalue}"


def assert_eigenvalues_real(
    operator: Any,
    *,
    atol: float = 1e-9,
) -> None:
    """Assert that operator has real eigenvalues.
    
    This is a property of Hermitian operators.
    
    Parameters
    ----------
    operator : array-like or operator object
        Operator to test.
    atol : float, default=1e-9
        Absolute tolerance for imaginary parts.
    
    Raises
    ------
    AssertionError
        If eigenvalues have significant imaginary components.
    
    Examples
    --------
    >>> op = build_delta_nfr(3, topology="laplacian")
    >>> assert_eigenvalues_real(op)
    """
    matrix = _extract_matrix(operator)
    eigenvalues = np.linalg.eigvals(matrix)
    
    max_imag = np.abs(eigenvalues.imag).max()
    assert max_imag < atol, \
        f"Eigenvalues not real: max |Im(λ)| = {max_imag}"


def assert_operator_finite(operator: Any) -> None:
    """Assert that operator contains only finite values.
    
    Checks for NaN and infinity values.
    
    Parameters
    ----------
    operator : array-like or operator object
        Operator to test.
    
    Raises
    ------
    AssertionError
        If operator contains NaN or infinite values.
    
    Examples
    --------
    >>> op = build_delta_nfr(5)
    >>> assert_operator_finite(op)
    """
    matrix = _extract_matrix(operator)
    
    assert np.all(np.isfinite(matrix)), \
        "Operator contains non-finite values (NaN or Inf)"


def assert_operator_dimension(
    operator: Any,
    expected_dimension: int,
) -> None:
    """Assert operator has expected dimension.
    
    Parameters
    ----------
    operator : array-like or operator object
        Operator to test.
    expected_dimension : int
        Expected dimension (for square matrix: N×N).
    
    Raises
    ------
    AssertionError
        If operator dimension doesn't match expected.
    
    Examples
    --------
    >>> op = build_delta_nfr(4)
    >>> assert_operator_dimension(op, 4)
    """
    matrix = _extract_matrix(operator)
    
    assert matrix.shape == (expected_dimension, expected_dimension), \
        f"Expected {expected_dimension}×{expected_dimension}, got {matrix.shape}"


def assert_spectral_properties(
    operator: Any,
    *,
    min_eigenvalue: float | None = None,
    max_eigenvalue: float | None = None,
    num_eigenvalues: int | None = None,
) -> None:
    """Assert operator spectrum satisfies constraints.
    
    Parameters
    ----------
    operator : array-like or operator object
        Operator to test.
    min_eigenvalue : float, optional
        Minimum expected eigenvalue.
    max_eigenvalue : float, optional
        Maximum expected eigenvalue.
    num_eigenvalues : int, optional
        Expected number of eigenvalues.
    
    Raises
    ------
    AssertionError
        If spectrum doesn't satisfy constraints.
    
    Examples
    --------
    >>> op = build_delta_nfr(3)
    >>> assert_spectral_properties(op, num_eigenvalues=3, min_eigenvalue=0.0)
    """
    matrix = _extract_matrix(operator)
    eigenvalues = np.linalg.eigvalsh(matrix)
    
    if num_eigenvalues is not None:
        assert len(eigenvalues) == num_eigenvalues, \
            f"Expected {num_eigenvalues} eigenvalues, got {len(eigenvalues)}"
    
    if min_eigenvalue is not None:
        assert eigenvalues.min() >= min_eigenvalue, \
            f"Minimum eigenvalue {eigenvalues.min()} < {min_eigenvalue}"
    
    if max_eigenvalue is not None:
        assert eigenvalues.max() <= max_eigenvalue, \
            f"Maximum eigenvalue {eigenvalues.max()} > {max_eigenvalue}"


def assert_operators_close(
    operator1: Any,
    operator2: Any,
    *,
    atol: float = 1e-9,
    rtol: float = 1e-9,
) -> None:
    """Assert two operators are element-wise close.
    
    Parameters
    ----------
    operator1, operator2 : array-like or operator objects
        Operators to compare.
    atol : float, default=1e-9
        Absolute tolerance.
    rtol : float, default=1e-9
        Relative tolerance.
    
    Raises
    ------
    AssertionError
        If operators differ beyond tolerance.
    
    Examples
    --------
    >>> op1 = build_delta_nfr(3, rng=np.random.default_rng(42))
    >>> op2 = build_delta_nfr(3, rng=np.random.default_rng(42))
    >>> assert_operators_close(op1, op2)
    """
    matrix1 = _extract_matrix(operator1)
    matrix2 = _extract_matrix(operator2)
    
    assert matrix1.shape == matrix2.shape, \
        f"Shape mismatch: {matrix1.shape} vs {matrix2.shape}"
    
    assert np.allclose(matrix1, matrix2, atol=atol, rtol=rtol), \
        "Operators are not close"


def assert_commutator_properties(
    operator1: Any,
    operator2: Any,
    *,
    is_hermitian1: bool = False,
    is_hermitian2: bool = False,
) -> None:
    """Assert commutator [A,B] = AB - BA has valid properties.
    
    Parameters
    ----------
    operator1, operator2 : array-like or operator objects
        Operators to compute commutator for.
    is_hermitian1, is_hermitian2 : bool, default=False
        If True, skip Hermitian check (performance optimization).
    
    Raises
    ------
    AssertionError
        If commutator has invalid properties.
    
    Examples
    --------
    >>> op1 = build_delta_nfr(3, topology="laplacian")
    >>> op2 = build_delta_nfr(3, topology="adjacency")
    >>> assert_commutator_properties(op1, op2, is_hermitian1=True, is_hermitian2=True)
    """
    matrix1 = _extract_matrix(operator1)
    matrix2 = _extract_matrix(operator2)
    
    # Compute commutator [A,B] = AB - BA
    commutator = matrix1 @ matrix2 - matrix2 @ matrix1
    
    # Commutator should be finite
    assert np.all(np.isfinite(commutator)), \
        "Commutator contains non-finite values"
    
    # Commutator of Hermitian operators is anti-Hermitian: [A,B]† = -[A,B]
    if is_hermitian1 and is_hermitian2:
        # Skip expensive Hermitian check if caller confirms
        assert np.allclose(commutator.conj().T, -commutator, atol=1e-9), \
            "Commutator of Hermitian operators should be anti-Hermitian"
    elif (np.allclose(matrix1, matrix1.conj().T, atol=1e-9) and 
          np.allclose(matrix2, matrix2.conj().T, atol=1e-9)):
        assert np.allclose(commutator.conj().T, -commutator, atol=1e-9), \
            "Commutator of Hermitian operators should be anti-Hermitian"


def get_spectral_bandwidth(operator: Any) -> float:
    """Compute spectral bandwidth (max eigenvalue - min eigenvalue).
    
    Parameters
    ----------
    operator : array-like or operator object
        Operator to analyze.
    
    Returns
    -------
    float
        Spectral bandwidth.
    
    Examples
    --------
    >>> op = build_delta_nfr(4)
    >>> bandwidth = get_spectral_bandwidth(op)
    >>> assert bandwidth >= 0
    """
    matrix = _extract_matrix(operator)
    eigenvalues = np.linalg.eigvalsh(matrix)
    
    return float(eigenvalues.max() - eigenvalues.min())


def _extract_matrix(operator: Any) -> np.ndarray:
    """Extract matrix from operator object or array.
    
    Parameters
    ----------
    operator : array-like or operator object
        Operator that may have .matrix attribute or be array itself.
        Expected interface: either numpy array or object with .matrix attribute
        returning array-like data.
    
    Returns
    -------
    np.ndarray
        Numpy array representation of operator.
    
    Raises
    ------
    ValueError
        If operator type is not supported.
    """
    if hasattr(operator, "matrix"):
        matrix_attr = operator.matrix
        if not isinstance(matrix_attr, (np.ndarray, list, tuple)):
            raise ValueError(
                f"Operator .matrix attribute has unsupported type: {type(matrix_attr)}"
            )
        return np.asarray(matrix_attr)
    else:
        try:
            return np.asarray(operator)
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"Cannot convert operator of type {type(operator)} to array: {e}"
            ) from e
