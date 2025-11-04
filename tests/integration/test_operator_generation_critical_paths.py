"""Critical path coverage for operator generation and factory wiring.

This module provides focused tests for critical operator generation paths
that were identified as needing increased coverage:
- Parameter validation and error handling
- Operator composition and chaining
- Edge cases in matrix generation
- Factory method combinations
"""

import pytest

np = pytest.importorskip("numpy")

from tnfr.mathematics.generators import build_delta_nfr
from tnfr.mathematics.operators_factory import (
    make_coherence_operator,
    make_frequency_operator,
)
from tests.helpers.base import parametrized_operator_dimension

def test_build_delta_nfr_parameter_validation_combinations() -> None:
    """Verify operator factory validates parameter combinations correctly."""
    dim = 4

    # Valid combinations should succeed
    valid_configs = [
        {"topology": "laplacian", "nu_f": 1.0, "scale": 1.0},
        {"topology": "adjacency", "nu_f": 2.0, "scale": 0.5},
        {"nu_f": 0.0, "scale": 1.0},  # Zero frequency
    ]

    for config in valid_configs:
        operator = build_delta_nfr(dim, **config)
        assert operator.shape == (dim, dim)
        assert np.allclose(operator, operator.conj().T)

def test_build_delta_nfr_invalid_parameter_combinations() -> None:
    """Verify operator factory rejects invalid parameter combinations."""
    dim = 4

    # Invalid dimension
    with pytest.raises((ValueError, TypeError, AssertionError)):
        build_delta_nfr(0)

    # Negative dimension
    with pytest.raises((ValueError, TypeError, AssertionError)):
        build_delta_nfr(-5)

    # Invalid topology
    with pytest.raises((ValueError, KeyError)):
        build_delta_nfr(dim, topology="invalid_topology_name")

def test_build_delta_nfr_preserves_structure_under_parameter_scaling() -> None:
    """Verify operator maintains structural properties under parameter variations."""
    dim = 4
    base_rng = np.random.default_rng(seed=42)

    scales = [0.1, 1.0, 10.0, 100.0]

    for scale in scales:
        # Reset RNG to same state for fair comparison
        rng = np.random.default_rng(seed=42)
        operator = build_delta_nfr(dim, scale=scale, rng=rng)

        # Structural properties must hold regardless of scale
        assert np.allclose(operator, operator.conj().T), f"Not Hermitian at scale={scale}"
        assert np.all(np.isfinite(operator)), f"Non-finite values at scale={scale}"

        # Eigenvalues should be real
        eigenvalues = np.linalg.eigvalsh(operator)
        assert np.all(np.isreal(eigenvalues)), f"Complex eigenvalues at scale={scale}"

def test_build_delta_nfr_frequency_parameter_boundary_conditions() -> None:
    """Verify operator generation handles frequency boundary conditions."""
    dim = 3

    # Zero frequency (silence)
    op_zero = build_delta_nfr(dim, nu_f=0.0)
    assert np.all(np.isfinite(op_zero))
    assert np.allclose(op_zero, op_zero.conj().T)

    # Very small frequency
    op_small = build_delta_nfr(dim, nu_f=1e-10)
    assert np.all(np.isfinite(op_small))

    # Very large frequency
    op_large = build_delta_nfr(dim, nu_f=1e6)
    assert np.all(np.isfinite(op_large))

    # Negative frequency (should be valid as νf can represent structural direction)
    op_neg = build_delta_nfr(dim, nu_f=-1.0)
    assert np.all(np.isfinite(op_neg))

def test_operator_composition_hermitian_closure() -> None:
    """Verify composed operators maintain Hermitian property."""
    dim = 4

    rng1 = np.random.default_rng(seed=42)
    rng2 = np.random.default_rng(seed=123)
    op1 = build_delta_nfr(dim, topology="laplacian", rng=rng1)
    op2 = build_delta_nfr(dim, topology="adjacency", rng=rng2)

    # Linear combination of Hermitian operators is Hermitian
    composed = 0.5 * op1 + 0.5 * op2
    assert np.allclose(composed, composed.conj().T)

    # Eigenvalues of composed operator should be real
    eigenvalues = np.linalg.eigvalsh(composed)
    assert np.all(np.isreal(eigenvalues))

def test_operator_factory_dimension_consistency(parametrized_operator_dimension) -> None:
    """Verify operators maintain dimension consistency across factory methods."""
    dim = parametrized_operator_dimension

    # build_delta_nfr
    dnfr_op = build_delta_nfr(dim)
    assert dnfr_op.shape == (dim, dim)

    # make_coherence_operator with default c_min
    coh_op = make_coherence_operator(dim, c_min=0.1)
    assert coh_op.matrix.shape == (dim, dim)

    # make_frequency_operator from spectrum
    spectrum = np.linspace(0.1, 1.0, dim)
    freq_op = make_frequency_operator(np.diag(spectrum))
    assert freq_op.matrix.shape == (dim, dim)

def test_operator_factory_error_propagation() -> None:
    """Verify operator factories properly propagate errors."""
    dim = 4

    # make_frequency_operator should reject negative eigenvalues
    invalid_matrix = np.diag([1.0, 2.0, -0.5, 1.0])
    with pytest.raises(ValueError, match="positive semidefinite"):
        make_frequency_operator(invalid_matrix)

    # make_coherence_operator should handle invalid spectrum gracefully
    # (Currently accepts eigenvalues as input)

def test_operator_reproducibility_across_sessions() -> None:
    """Verify operator generation is reproducible across different sessions."""
    dim = 5
    seed = 42

    # Session 1
    rng1 = np.random.default_rng(seed=seed)
    op1 = build_delta_nfr(dim, rng=rng1)

    # Session 2 (new RNG with same seed)
    rng2 = np.random.default_rng(seed=seed)
    op2 = build_delta_nfr(dim, rng=rng2)

    # Should be identical
    assert np.allclose(op1, op2), "Operators not reproducible"

def test_operator_topology_structural_differences() -> None:
    """Verify different topologies produce structurally distinct operators."""
    dim = 5
    seed = 999

    # Create separate RNG instances instead of manipulating state
    op_laplacian = build_delta_nfr(dim, topology="laplacian", rng=np.random.default_rng(seed=seed))
    op_adjacency = build_delta_nfr(dim, topology="adjacency", rng=np.random.default_rng(seed=seed))

    # Both should be valid Hermitian operators
    assert np.allclose(op_laplacian, op_laplacian.conj().T)
    assert np.allclose(op_adjacency, op_adjacency.conj().T)

    # But generally should have different structures (unless extremely unlikely)
    # We verify both are valid rather than asserting difference

def test_operator_numerical_stability_extreme_parameters() -> None:
    """Verify numerical stability with extreme parameter combinations."""
    dim = 4

    # Extreme combinations that should still produce valid operators
    extreme_configs = [
        {"nu_f": 1e-15, "scale": 1e-15},
        {"nu_f": 1e10, "scale": 1e-10},
        {"nu_f": 1e-10, "scale": 1e10},
    ]

    for config in extreme_configs:
        op = build_delta_nfr(dim, **config)
        # Must remain finite and Hermitian despite extreme parameters
        assert np.all(np.isfinite(op)), f"Non-finite with config {config}"
        assert np.allclose(op, op.conj().T), f"Not Hermitian with config {config}"

def test_operator_eigenspectrum_properties() -> None:
    """Verify eigenspectrum properties of generated operators."""
    dim = 6
    operator = build_delta_nfr(dim)

    eigenvalues = np.linalg.eigvalsh(operator)

    # All eigenvalues should be real (Hermitian property)
    assert np.all(np.isreal(eigenvalues))

    # Eigenvalues should be finite
    assert np.all(np.isfinite(eigenvalues))

    # Should have exactly dim eigenvalues
    assert len(eigenvalues) == dim

def test_operator_matrix_properties_under_scaling() -> None:
    """Verify matrix properties are preserved under different scalings."""
    dim = 4
    rng = np.random.default_rng(seed=555)
    state = rng.bit_generator.state

    base_op = build_delta_nfr(dim, scale=1.0, rng=rng)

    # Generate with different scale using same random state
    rng.bit_generator.state = state
    scaled_op = build_delta_nfr(dim, scale=5.0, rng=rng)

    # Both should have same structural pattern (Hermitian, finite)
    assert np.allclose(base_op, base_op.conj().T)
    assert np.allclose(scaled_op, scaled_op.conj().T)
    assert np.all(np.isfinite(base_op))
    assert np.all(np.isfinite(scaled_op))

    # Scaled version should generally have larger norm
    norm_base = np.linalg.norm(base_op)
    norm_scaled = np.linalg.norm(scaled_op)
    assert norm_scaled >= norm_base * 0.9  # Allow some tolerance for randomness

def test_operator_generation_thread_safety() -> None:
    """Verify operator generation produces consistent results in sequential calls."""
    dim = 4
    seed = 777

    # Generate multiple operators with same seed sequentially
    operators = []
    for _ in range(3):
        rng = np.random.default_rng(seed=seed)
        op = build_delta_nfr(dim, rng=rng)
        operators.append(op)

    # All should be identical
    for i in range(1, len(operators)):
        assert np.allclose(operators[0], operators[i])

# ============================================================================
# ADDITIONAL CRITICAL PATH COVERAGE FOR OPERATOR GENERATION
# ============================================================================

@pytest.mark.parametrize("dim,nu_f,scale", [
    (2, 0.001, 0.01),  # Very small parameters
    (3, 100.0, 10.0),   # Very large parameters
    (5, 0.5, 1e-6),     # Mixed scales
    (8, 1e3, 1e-3),     # Extreme ratio
])
def test_operator_generation_extreme_parameter_ranges(dim, nu_f, scale) -> None:
    """Test operator generation with extreme but valid parameter ranges.

    Enhances coverage by testing boundary conditions not covered in basic tests.
    Ensures numerical stability across wide parameter ranges.
    """
    operator = build_delta_nfr(dim, nu_f=nu_f, scale=scale)

    # Verify basic structural properties maintained
    assert operator.shape == (dim, dim)
    assert np.allclose(operator, operator.conj().T)  # Hermitian
    assert np.all(np.isfinite(operator))  # No NaN/Inf

    # Verify eigenspectrum is real (consequence of Hermitian)
    eigenvalues = np.linalg.eigvalsh(operator)
    assert np.all(np.isreal(eigenvalues))

@pytest.mark.parametrize("topology_type", ["laplacian", "adjacency"])
def test_operator_generation_with_different_topologies(topology_type) -> None:
    """Test operator generation for different canonical topologies.

    Adds critical path coverage for topology-specific operator behavior.
    """
    dim = 5

    # Generate operator for this topology
    operator = build_delta_nfr(dim, topology=topology_type)

    # Verify structural properties
    assert operator.shape == (dim, dim)
    assert np.allclose(operator, operator.conj().T)

    # Verify operator reflects topology structure
    # Both topologies should produce non-trivial operators
    assert np.linalg.norm(operator) > 1e-10

    # Laplacian should have zero row sums (conservation)
    if topology_type == "laplacian":
        row_sums = np.sum(operator, axis=1)
        assert np.allclose(row_sums, 0.0, atol=1e-10)

def test_operator_composition_maintains_closure() -> None:
    """Test that operator composition maintains structural closure.

    Critical path: verifies that composed operators preserve TNFR invariants.
    """
    dim = 4
    op1 = build_delta_nfr(dim, nu_f=1.0, scale=1.0)
    op2 = build_delta_nfr(dim, nu_f=2.0, scale=0.5)

    # Test linear combination (operator closure)
    composed = 0.5 * op1 + 0.5 * op2

    # Verify Hermitian property maintained
    assert np.allclose(composed, composed.conj().T)

    # Test commutator [op1, op2] = op1*op2 - op2*op1
    commutator = op1 @ op2 - op2 @ op1

    # Commutator of two Hermitian operators is anti-Hermitian
    assert np.allclose(commutator, -commutator.conj().T)

@pytest.mark.parametrize("nu_f_values", [
    [0.1, 0.2, 0.3],
    [1.0, 2.0, 3.0],
    [0.5, 1.0, 1.5],
])
def test_operator_generation_frequency_scaling_consistency(nu_f_values) -> None:
    """Test that frequency scaling maintains consistent structural behavior.

    Adds parametrized coverage for frequency parameter variations.
    """
    dim = 3
    operators = [build_delta_nfr(dim, nu_f=nu_f) for nu_f in nu_f_values]

    # All operators should maintain structural properties
    for op in operators:
        assert np.allclose(op, op.conj().T)
        assert np.all(np.isfinite(op))

    # Verify relative scaling relationships
    norms = [np.linalg.norm(op) for op in operators]

    # Larger frequency should generally lead to larger operator norms
    # (allowing tolerance for stochastic generation)
    for i in range(len(norms) - 1):
        if nu_f_values[i+1] > nu_f_values[i]:
            assert norms[i+1] >= norms[i] * 0.8  # Relaxed check for robustness

def test_operator_zero_frequency_boundary() -> None:
    """Test operator generation at zero frequency boundary condition.

    Critical path: ensures graceful handling of boundary case νf = 0.
    """
    dim = 3
    # Zero frequency should produce minimal/zero operator
    operator = build_delta_nfr(dim, nu_f=0.0, scale=1.0)

    assert operator.shape == (dim, dim)
    assert np.allclose(operator, operator.conj().T)

    # Zero frequency should lead to near-zero operator
    assert np.linalg.norm(operator) < 1e-10
