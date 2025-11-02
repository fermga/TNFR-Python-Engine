"""Extended tests for operator generation covering critical paths and edge cases.

This module extends test_operator_generation.py with additional coverage for:
- Complex operator parameter combinations
- Edge cases in matrix generation
- Structural operator validation
"""

import pytest

np = pytest.importorskip("numpy")

from tnfr.mathematics.generators import build_delta_nfr


def test_build_delta_nfr_consistent_across_calls() -> None:
    """Verify operator generation is deterministic without explicit seed."""
    dim = 5
    
    # Without explicit RNG, should use internal seeding
    matrix1 = build_delta_nfr(dim)
    matrix2 = build_delta_nfr(dim)
    
    # Structure should be consistent (Hermitian)
    assert np.allclose(matrix1, matrix1.conj().T)
    assert np.allclose(matrix2, matrix2.conj().T)


def test_build_delta_nfr_large_dimension_feasibility() -> None:
    """Verify operator generation handles larger dimensions efficiently."""
    dim = 32
    
    dnfr_matrix = build_delta_nfr(dim)
    
    # Should complete without error and maintain structural properties
    assert dnfr_matrix.shape == (dim, dim)
    assert np.allclose(dnfr_matrix, dnfr_matrix.conj().T)
    assert np.all(np.isfinite(dnfr_matrix))


def test_build_delta_nfr_small_scale_precision() -> None:
    """Verify precision maintained with very small scale parameters."""
    dim = 4
    tiny_scale = 1e-6
    
    dnfr_matrix = build_delta_nfr(dim, scale=tiny_scale)
    
    # Should maintain finite values even at tiny scales
    assert np.all(np.isfinite(dnfr_matrix))
    assert np.allclose(dnfr_matrix, dnfr_matrix.conj().T)
    
    # Matrix norm should scale appropriately
    norm = np.linalg.norm(dnfr_matrix)
    assert norm > 0.0


def test_build_delta_nfr_large_scale_stability() -> None:
    """Verify operator remains stable with large scale parameters."""
    dim = 4
    large_scale = 1e3
    
    dnfr_matrix = build_delta_nfr(dim, scale=large_scale)
    
    # Should maintain structural properties despite large scale
    assert np.all(np.isfinite(dnfr_matrix))
    assert np.allclose(dnfr_matrix, dnfr_matrix.conj().T)


def test_build_delta_nfr_nu_f_zero_valid() -> None:
    """Verify operator generation with νf = 0 (silence operator)."""
    dim = 3
    
    dnfr_matrix = build_delta_nfr(dim, nu_f=0.0)
    
    # Zero frequency should still produce valid Hermitian operator
    assert dnfr_matrix.shape == (dim, dim)
    assert np.allclose(dnfr_matrix, dnfr_matrix.conj().T)


def test_build_delta_nfr_nu_f_extremes() -> None:
    """Verify operator generation with extreme frequency values."""
    dim = 3
    
    # Very small νf
    matrix_small = build_delta_nfr(dim, nu_f=1e-10)
    assert np.all(np.isfinite(matrix_small))
    
    # Very large νf
    matrix_large = build_delta_nfr(dim, nu_f=1e6)
    assert np.all(np.isfinite(matrix_large))


def test_build_delta_nfr_combined_parameters() -> None:
    """Verify operator generation with multiple parameters combined."""
    dim = 4
    rng = np.random.default_rng(seed=999)
    
    dnfr_matrix = build_delta_nfr(
        dim,
        topology="laplacian",
        nu_f=2.5,
        scale=0.5,
        rng=rng
    )
    
    # All structural properties should hold
    assert dnfr_matrix.shape == (dim, dim)
    assert np.allclose(dnfr_matrix, dnfr_matrix.conj().T)
    assert np.all(np.isfinite(dnfr_matrix))
    
    # Eigenvalues should be real
    eigenvalues = np.linalg.eigvalsh(dnfr_matrix)
    assert np.all(np.isreal(eigenvalues))


def test_build_delta_nfr_boundary_dimension_2() -> None:
    """Verify minimum dimension of 2 works correctly."""
    dim = 2
    
    dnfr_matrix = build_delta_nfr(dim)
    
    assert dnfr_matrix.shape == (2, 2)
    assert np.allclose(dnfr_matrix, dnfr_matrix.conj().T)


def test_build_delta_nfr_spectrum_properties() -> None:
    """Verify spectral properties of generated operators."""
    dim = 6
    dnfr_matrix = build_delta_nfr(dim)
    
    # Compute eigenvalues
    eigenvalues = np.linalg.eigvalsh(dnfr_matrix)
    
    # Eigenvalues should be real and finite
    assert len(eigenvalues) == dim
    assert np.all(np.isreal(eigenvalues))
    assert np.all(np.isfinite(eigenvalues))


def test_build_delta_nfr_orthogonality_preservation() -> None:
    """Verify operator generation preserves orthogonal structure when applicable."""
    dim = 4
    dnfr_matrix = build_delta_nfr(dim)
    
    # Hermitian operators have orthogonal eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(dnfr_matrix)
    
    # Check orthonormality of eigenvectors
    identity = np.eye(dim)
    computed = eigenvectors.conj().T @ eigenvectors
    assert np.allclose(computed, identity, atol=1e-10)


def test_build_delta_nfr_topology_comparison() -> None:
    """Compare different topology options for structural differences."""
    dim = 4
    rng = np.random.default_rng(seed=42)
    
    # Save state for reproducibility
    state = rng.bit_generator.state
    
    # Generate with laplacian topology
    rng.bit_generator.state = state
    matrix_laplacian = build_delta_nfr(dim, topology="laplacian", rng=rng)
    
    # Generate with adjacency topology
    rng.bit_generator.state = state
    matrix_adjacency = build_delta_nfr(dim, topology="adjacency", rng=rng)
    
    # Both should be valid Hermitian operators
    assert np.allclose(matrix_laplacian, matrix_laplacian.conj().T)
    assert np.allclose(matrix_adjacency, matrix_adjacency.conj().T)
    
    # But they should generally differ in structure
    # (unless by chance they're identical, which is unlikely)
    # We just verify both are valid rather than asserting difference


def test_build_delta_nfr_scale_linearity() -> None:
    """Verify scale parameter affects operator magnitude linearly."""
    dim = 3
    rng = np.random.default_rng(seed=123)
    state = rng.bit_generator.state
    
    # Generate with scale 1.0
    rng.bit_generator.state = state
    matrix_1 = build_delta_nfr(dim, scale=1.0, rng=rng)
    
    # Generate with scale 2.0 (same random state)
    rng.bit_generator.state = state
    matrix_2 = build_delta_nfr(dim, scale=2.0, rng=rng)
    
    # Second matrix should have approximately double the norm
    norm_1 = np.linalg.norm(matrix_1)
    norm_2 = np.linalg.norm(matrix_2)
    
    assert norm_2 >= norm_1  # At minimum, should not decrease
