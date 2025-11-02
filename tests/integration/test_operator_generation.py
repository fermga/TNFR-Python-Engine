"""DEPRECATED: Comprehensive tests for operator generation and factory wiring.

⚠️ DEPRECATION NOTICE:
This module has been consolidated into test_unified_operator_validation.py
and test_operator_generation_critical_paths.py using parametrized fixtures.

See:
- tests/integration/test_unified_operator_validation.py for unified operator tests
- tests/integration/test_operator_generation_critical_paths.py for critical path coverage
- tests/README_TEST_OPTIMIZATION.md for usage guidelines
- tests/TEST_CONSOLIDATION_SUMMARY.md for detailed consolidation mapping

This module tests critical paths for operator generation including:
- Operator factory parameter validation
- Operator instance creation
- Matrix generation
"""

import pytest

# Mark entire module as deprecated - tests are redundant with unified suite
pytestmark = pytest.mark.skip(
    reason="DEPRECATED: Consolidated into test_unified_operator_validation.py"
)

np = pytest.importorskip("numpy")

from tnfr.mathematics.generators import build_delta_nfr


def test_build_delta_nfr_returns_hermitian_operator() -> None:
    """Verify that build_delta_nfr generates Hermitian operator."""
    dimension = 4

    dnfr_matrix = build_delta_nfr(dimension)

    assert isinstance(dnfr_matrix, np.ndarray)

    # Hermitian operators should equal their conjugate transpose
    assert np.allclose(dnfr_matrix, dnfr_matrix.conj().T)


def test_build_delta_nfr_respects_dimension() -> None:
    """Verify operator dimensions match requested size."""
    for dim in [2, 3, 5, 8]:
        dnfr_matrix = build_delta_nfr(dim)

        assert dnfr_matrix.shape == (dim, dim)


def test_build_delta_nfr_laplacian_topology() -> None:
    """Verify Laplacian topology produces appropriate structure."""
    dim = 4
    dnfr_matrix = build_delta_nfr(dim, topology="laplacian")

    # Laplacian should be Hermitian
    assert np.allclose(dnfr_matrix, dnfr_matrix.conj().T)

    # Check it's a square matrix of correct size
    assert dnfr_matrix.shape == (dim, dim)


def test_build_delta_nfr_adjacency_topology() -> None:
    """Verify adjacency topology produces appropriate structure."""
    dim = 4
    dnfr_matrix = build_delta_nfr(dim, topology="adjacency")

    # Adjacency should be Hermitian
    assert np.allclose(dnfr_matrix, dnfr_matrix.conj().T)

    assert dnfr_matrix.shape == (dim, dim)


def test_build_delta_nfr_frequency_scaling() -> None:
    """Test frequency scaling parameter."""
    dim = 3
    nu_f = 2.5

    dnfr_matrix = build_delta_nfr(dim, nu_f=nu_f)

    assert dnfr_matrix.shape == (dim, dim)
    assert np.allclose(dnfr_matrix, dnfr_matrix.conj().T)


def test_build_delta_nfr_scale_parameter() -> None:
    """Test scale parameter affects operator amplitude."""
    dim = 3

    base_matrix = build_delta_nfr(dim, scale=1.0)
    scaled_matrix = build_delta_nfr(dim, scale=2.0)

    # Scaled version should have larger values
    assert np.linalg.norm(scaled_matrix) >= np.linalg.norm(base_matrix)


def test_build_delta_nfr_rejects_invalid_dimension() -> None:
    """Verify operator factory rejects invalid dimensions."""
    with pytest.raises((ValueError, TypeError, AssertionError)):
        build_delta_nfr(0)

    with pytest.raises((ValueError, TypeError, AssertionError)):
        build_delta_nfr(-1)


def test_build_delta_nfr_rejects_invalid_topology() -> None:
    """Verify operator factory rejects unknown topologies."""
    with pytest.raises((ValueError, KeyError)):
        build_delta_nfr(4, topology="unknown_topology")


def test_build_delta_nfr_reproducibility_with_seed() -> None:
    """Verify operator generation is deterministic with RNG seed."""
    dim = 4

    rng1 = np.random.default_rng(seed=42)
    rng2 = np.random.default_rng(seed=42)

    matrix1 = build_delta_nfr(dim, rng=rng1)
    matrix2 = build_delta_nfr(dim, rng=rng2)

    # Should produce identical operators with same seed
    assert np.allclose(matrix1, matrix2)


def test_build_delta_nfr_different_seeds_produce_different_results() -> None:
    """Verify different seeds produce different operators."""
    dim = 4

    rng1 = np.random.default_rng(seed=42)
    rng2 = np.random.default_rng(seed=123)

    matrix1 = build_delta_nfr(dim, rng=rng1)
    matrix2 = build_delta_nfr(dim, rng=rng2)

    # Should produce different operators with different seeds
    assert not np.allclose(matrix1, matrix2)


def test_build_delta_nfr_eigenvalues_real() -> None:
    """Verify Hermitian operators have real eigenvalues."""
    dim = 5
    dnfr_matrix = build_delta_nfr(dim)

    eigenvalues = np.linalg.eigvalsh(dnfr_matrix)

    # All eigenvalues should be real (characteristic of Hermitian)
    assert np.all(np.isreal(eigenvalues))


def test_build_delta_nfr_produces_finite_values() -> None:
    """Verify generated operators contain only finite values."""
    dim = 6
    dnfr_matrix = build_delta_nfr(dim)

    assert np.all(np.isfinite(dnfr_matrix))
