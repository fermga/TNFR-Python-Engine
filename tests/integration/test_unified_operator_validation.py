"""Unified operator validation tests using parametrized fixtures.

This module consolidates redundant operator tests from:
- integration/test_operator_generation.py
- integration/test_operator_generation_extended.py
- math_integration/test_generators.py
- mathematics/test_operators.py

By using parametrized fixtures, we reduce redundancy while maintaining
comprehensive coverage of operator properties.
"""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from tnfr.mathematics.generators import build_delta_nfr
from tnfr.mathematics.operators import CoherenceOperator, FrequencyOperator
from tnfr.mathematics.operators_factory import (
    make_coherence_operator,
    make_frequency_operator,
)


# Parametrized operator dimensions
@pytest.fixture(params=[2, 3, 4, 5, 8])
def unified_dimension(request):
    """Unified dimension parameter consolidating multiple dimension tests."""
    return request.param


# Parametrized operator topologies
@pytest.fixture(params=["laplacian", "adjacency"])
def unified_topology(request):
    """Unified topology parameter consolidating topology tests."""
    return request.param


# Parametrized operator scales
@pytest.fixture(params=[0.1, 1.0, 2.0, 10.0])
def unified_scale(request):
    """Unified scale parameter consolidating scale tests."""
    return request.param


def test_build_delta_nfr_hermitian_unified(unified_dimension) -> None:
    """Unified test verifying operators are Hermitian.
    
    Consolidates:
    - test_build_delta_nfr_returns_hermitian_operator
    - test_build_delta_nfr_returns_hermitian_operators (math_integration)
    """
    dim = unified_dimension
    dnfr_matrix = build_delta_nfr(dim)
    
    # Hermitian operators should equal their conjugate transpose
    assert np.allclose(dnfr_matrix, dnfr_matrix.conj().T)


def test_build_delta_nfr_dimension_consistency_unified(unified_dimension) -> None:
    """Unified test verifying operator dimensions.
    
    Consolidates:
    - test_build_delta_nfr_respects_dimension
    - Multiple dimension-specific tests
    """
    dim = unified_dimension
    dnfr_matrix = build_delta_nfr(dim)
    assert dnfr_matrix.shape == (dim, dim)


def test_build_delta_nfr_topology_unified(unified_dimension, unified_topology) -> None:
    """Unified test verifying different topologies produce valid operators.
    
    Consolidates:
    - test_build_delta_nfr_laplacian_topology
    - test_build_delta_nfr_adjacency_topology
    """
    dim = unified_dimension
    topology = unified_topology
    
    dnfr_matrix = build_delta_nfr(dim, topology=topology)
    
    # Should be Hermitian
    assert np.allclose(dnfr_matrix, dnfr_matrix.conj().T)
    assert dnfr_matrix.shape == (dim, dim)


def test_build_delta_nfr_scale_unified(unified_dimension, unified_scale) -> None:
    """Unified test verifying scale parameter effects.
    
    Consolidates:
    - test_build_delta_nfr_scale_parameter
    - test_build_delta_nfr_small_scale_precision
    - test_build_delta_nfr_large_scale_stability
    """
    dim = unified_dimension
    scale = unified_scale
    
    dnfr_matrix = build_delta_nfr(dim, scale=scale)
    
    # Should maintain structural properties regardless of scale
    assert dnfr_matrix.shape == (dim, dim)
    assert np.allclose(dnfr_matrix, dnfr_matrix.conj().T)
    assert np.all(np.isfinite(dnfr_matrix))


@pytest.mark.parametrize("nu_f", [0.0, 0.5, 1.0, 2.5, 10.0])
def test_build_delta_nfr_frequency_unified(unified_dimension, nu_f) -> None:
    """Unified test verifying frequency parameter handling.
    
    Consolidates:
    - test_build_delta_nfr_frequency_scaling
    - test_build_delta_nfr_nu_f_zero_valid
    - test_build_delta_nfr_nu_f_extremes
    """
    dim = unified_dimension
    
    dnfr_matrix = build_delta_nfr(dim, nu_f=nu_f)
    
    assert dnfr_matrix.shape == (dim, dim)
    assert np.allclose(dnfr_matrix, dnfr_matrix.conj().T)
    assert np.all(np.isfinite(dnfr_matrix))


def test_build_delta_nfr_eigenvalues_real_unified(unified_dimension) -> None:
    """Unified test verifying eigenvalues are real.
    
    Consolidates:
    - test_build_delta_nfr_eigenvalues_real
    - test_build_delta_nfr_spectrum_properties
    """
    dim = unified_dimension
    dnfr_matrix = build_delta_nfr(dim)
    
    eigenvalues = np.linalg.eigvalsh(dnfr_matrix)
    
    # All eigenvalues should be real (characteristic of Hermitian)
    assert np.all(np.isreal(eigenvalues))
    assert len(eigenvalues) == dim


def test_build_delta_nfr_finite_values_unified(unified_dimension) -> None:
    """Unified test verifying operators contain finite values.
    
    Consolidates:
    - test_build_delta_nfr_produces_finite_values
    - Multiple finite value checks
    """
    dim = unified_dimension
    dnfr_matrix = build_delta_nfr(dim)
    
    assert np.all(np.isfinite(dnfr_matrix))


def test_build_delta_nfr_reproducibility_unified() -> None:
    """Unified test verifying reproducibility with seeds.
    
    Consolidates:
    - test_build_delta_nfr_reproducibility_with_seed
    - test_build_delta_nfr_consistent_across_calls
    """
    dim = 4
    
    rng1 = np.random.default_rng(seed=42)
    rng2 = np.random.default_rng(seed=42)
    
    matrix1 = build_delta_nfr(dim, rng=rng1)
    matrix2 = build_delta_nfr(dim, rng=rng2)
    
    # Should produce identical operators with same seed
    assert np.allclose(matrix1, matrix2)


def test_build_delta_nfr_different_seeds_unified() -> None:
    """Unified test verifying different seeds produce different results.
    
    Consolidates seed variation tests.
    """
    dim = 4
    
    rng1 = np.random.default_rng(seed=42)
    rng2 = np.random.default_rng(seed=123)
    
    matrix1 = build_delta_nfr(dim, rng=rng1)
    matrix2 = build_delta_nfr(dim, rng=rng2)
    
    # Should produce different operators with different seeds
    assert not np.allclose(matrix1, matrix2)


@pytest.mark.parametrize("invalid_dim", [0, -1, -5])
def test_build_delta_nfr_rejects_invalid_dimension_unified(invalid_dim) -> None:
    """Unified test verifying invalid dimensions are rejected.
    
    Consolidates:
    - test_build_delta_nfr_rejects_invalid_dimension
    """
    with pytest.raises((ValueError, TypeError, AssertionError)):
        build_delta_nfr(invalid_dim)


def test_build_delta_nfr_rejects_invalid_topology_unified() -> None:
    """Unified test verifying invalid topologies are rejected."""
    with pytest.raises((ValueError, KeyError)):
        build_delta_nfr(4, topology="unknown_topology")


def test_coherence_operator_hermitian_unified() -> None:
    """Unified test for coherence operator Hermitian property.
    
    Consolidates coherence operator tests.
    """
    matrix = np.array([[2.0, 1.0 - 1.0j], [1.0 + 1.0j, 3.0]], dtype=np.complex128)
    operator = CoherenceOperator(matrix)
    
    # Should be Hermitian
    assert operator.is_hermitian()


def test_coherence_operator_positive_semidefinite_unified() -> None:
    """Unified test for coherence operator positive semidefiniteness."""
    matrix = np.array([[2.0, 1.0 - 1.0j], [1.0 + 1.0j, 3.0]], dtype=np.complex128)
    operator = CoherenceOperator(matrix)
    
    assert operator.is_positive_semidefinite()


def test_coherence_operator_rejects_non_hermitian_unified() -> None:
    """Unified test verifying non-Hermitian matrices are rejected.
    
    Consolidates:
    - test_coherence_operator_non_hermitian_rejected
    """
    matrix = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.complex128)
    with pytest.raises(ValueError):
        CoherenceOperator(matrix)


def test_frequency_operator_positive_spectrum_unified() -> None:
    """Unified test for frequency operator with positive spectrum."""
    operator = FrequencyOperator([0.5, 1.5, 3.0])
    
    spectrum = operator.spectrum()
    assert spectrum.dtype == float
    assert np.all(spectrum >= 0)


def test_frequency_operator_negative_spectrum_detected_unified() -> None:
    """Unified test verifying negative spectrum detection."""
    operator = FrequencyOperator([-0.5, 0.5])
    assert not operator.is_positive_semidefinite()


def test_make_coherence_operator_from_spectrum_unified() -> None:
    """Unified test for coherence operator factory."""
    operator = make_coherence_operator(3, spectrum=np.array([0.2, 0.3, 0.4]))
    
    assert isinstance(operator, CoherenceOperator)
    np.testing.assert_allclose(operator.matrix, np.diag([0.2, 0.3, 0.4]))
    assert operator.is_positive_semidefinite()


def test_make_frequency_operator_rejects_negative_eigenvalues_unified() -> None:
    """Unified test for frequency operator validation."""
    matrix = np.array([[1.0, 0.0], [0.0, -0.5]], dtype=np.complex128)
    
    with pytest.raises(ValueError, match="positive semidefinite"):
        make_frequency_operator(matrix)


@pytest.mark.parametrize("combined_params", [
    {"topology": "laplacian", "nu_f": 1.0, "scale": 1.0},
    {"topology": "adjacency", "nu_f": 2.0, "scale": 0.5},
    {"topology": "laplacian", "nu_f": 0.5, "scale": 2.0},
])
def test_build_delta_nfr_combined_parameters_unified(combined_params) -> None:
    """Unified test for combined parameter configurations.
    
    Consolidates:
    - test_build_delta_nfr_combined_parameters
    """
    dim = 4
    rng = np.random.default_rng(seed=999)
    
    dnfr_matrix = build_delta_nfr(dim, rng=rng, **combined_params)
    
    # All structural properties should hold
    assert dnfr_matrix.shape == (dim, dim)
    assert np.allclose(dnfr_matrix, dnfr_matrix.conj().T)
    assert np.all(np.isfinite(dnfr_matrix))
    
    # Eigenvalues should be real
    eigenvalues = np.linalg.eigvalsh(dnfr_matrix)
    assert np.all(np.isreal(eigenvalues))


def test_build_delta_nfr_orthogonality_preserved_unified(unified_dimension) -> None:
    """Unified test verifying eigenvector orthogonality.
    
    Consolidates:
    - test_build_delta_nfr_orthogonality_preservation
    """
    dim = unified_dimension
    dnfr_matrix = build_delta_nfr(dim)
    
    # Hermitian operators have orthogonal eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(dnfr_matrix)
    
    # Check orthonormality of eigenvectors
    identity = np.eye(dim)
    computed = eigenvectors.conj().T @ eigenvectors
    assert np.allclose(computed, identity, atol=1e-10)
