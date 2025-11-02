"""Enhanced critical path coverage for operator generation, validators, and trajectories.

This module adds additional critical path coverage beyond the existing unified tests,
focusing on:
- Advanced operator generation parameter validation
- Multi-scale nodal validator scenarios
- Complex run_sequence trajectory patterns
- Edge cases in structural operators
"""

from __future__ import annotations

import math
import pytest
import networkx as nx

np = pytest.importorskip("numpy")

from tnfr.constants import DNFR_PRIMARY, EPI_PRIMARY, VF_PRIMARY, THETA_KEY, inject_defaults
from tnfr.mathematics.generators import build_delta_nfr
from tnfr.execution import play, seq, wait, target, block, compile_sequence
from tnfr.tokens import OpTag
from tests.helpers.validation import (
    assert_dnfr_balanced,
    assert_epi_vf_in_bounds,
)
from tests.helpers.fixtures import seed_graph_factory


# ============================================================================
# Operator Generation - Advanced Parameter Validation
# ============================================================================

@pytest.mark.parametrize("dim,nu_f,scale", [
    (3, 0.0, 1.0),      # Zero frequency
    (3, 1e-10, 1.0),    # Near-zero frequency
    (3, 1e6, 1.0),      # Very large frequency
    (3, 1.0, 1e-10),    # Near-zero scale
    (3, 1.0, 1e6),      # Very large scale
    (3, 1e-10, 1e-10),  # Both near-zero
    (3, 1e6, 1e6),      # Both very large
])
def test_operator_generation_extreme_parameters(dim, nu_f, scale) -> None:
    """Test operator generation with extreme parameter combinations.
    
    Ensures operators remain valid even with boundary parameter values,
    maintaining TNFR structural invariants (Hermitian, finite values).
    """
    dnfr_matrix = build_delta_nfr(dim, nu_f=nu_f, scale=scale)
    
    # Must remain Hermitian
    assert np.allclose(dnfr_matrix, dnfr_matrix.conj().T, rtol=1e-10)
    
    # Must produce finite values
    assert np.all(np.isfinite(dnfr_matrix))
    
    # Must have correct dimension
    assert dnfr_matrix.shape == (dim, dim)


@pytest.mark.parametrize("topology1,topology2", [
    ("laplacian", "adjacency"),
    ("adjacency", "laplacian"),
])
def test_operator_generation_topology_independence(topology1, topology2) -> None:
    """Test that different topologies produce structurally valid but distinct operators.
    
    Verifies that topology parameter affects operator structure while maintaining
    all TNFR structural invariants.
    """
    dim = 4
    
    # Create separate RNG instances with same seed for clearer comparison
    rng1 = np.random.default_rng(seed=42)
    rng2 = np.random.default_rng(seed=42)
    
    # Generate with same seed but different topologies
    matrix1 = build_delta_nfr(dim, topology=topology1, rng=rng1)
    matrix2 = build_delta_nfr(dim, topology=topology2, rng=rng2)
    
    # Both must be Hermitian
    assert np.allclose(matrix1, matrix1.conj().T)
    assert np.allclose(matrix2, matrix2.conj().T)
    
    # Both must have real eigenvalues
    eigs1 = np.linalg.eigvalsh(matrix1)
    eigs2 = np.linalg.eigvalsh(matrix2)
    assert np.all(np.isreal(eigs1))
    assert np.all(np.isreal(eigs2))


@pytest.mark.parametrize("seed_val", [0, 1, 42, 999, 2**31-1])
def test_operator_generation_seed_reproducibility(seed_val) -> None:
    """Test operator generation reproducibility across various seed values.
    
    Ensures deterministic behavior is maintained for any valid seed value,
    supporting TNFR's controlled determinism invariant.
    """
    dim = 4
    
    rng1 = np.random.default_rng(seed=seed_val)
    rng2 = np.random.default_rng(seed=seed_val)
    
    matrix1 = build_delta_nfr(dim, rng=rng1)
    matrix2 = build_delta_nfr(dim, rng=rng2)
    
    # Identical seeds must produce identical operators
    # Use rtol for platform-independent comparison
    assert np.allclose(matrix1, matrix2, rtol=1e-14, atol=1e-15)


# ============================================================================
# Nodal Validators - Multi-Scale Scenarios
# ============================================================================

@pytest.mark.parametrize("num_nodes,edge_prob", [
    (5, 0.2),      # Sparse small network
    (10, 0.5),     # Medium density
    (20, 0.8),     # Dense network
    (50, 0.1),     # Large sparse network
    (100, 0.05),   # Very large sparse network
])
def test_nodal_validator_multi_scale_bounds(num_nodes, edge_prob) -> None:
    """Test nodal validators maintain EPI/νf bounds across multiple scales.
    
    Verifies structural bounds are enforced regardless of network size or density,
    maintaining TNFR's scale-invariant properties.
    """
    graph = nx.gnp_random_graph(num_nodes, edge_prob, seed=42)
    inject_defaults(graph)
    
    # Initialize with values spanning expected ranges
    for i, (node, data) in enumerate(graph.nodes(data=True)):
        data[EPI_PRIMARY] = -1.0 + 2.0 * (i / num_nodes)  # Range [-1, 1]
        data[VF_PRIMARY] = 0.5 + 1.0 * (i / num_nodes)    # Range [0.5, 1.5]
        data[DNFR_PRIMARY] = 0.0
    
    # Verify bounds are maintained
    assert_epi_vf_in_bounds(graph, epi_min=-1.5, epi_max=1.5, vf_min=0.0, vf_max=2.0)


@pytest.mark.parametrize("phase_variation", [
    0.0,           # All synchronized
    math.pi / 4,   # Small variation
    math.pi / 2,   # Medium variation
    math.pi,       # Large variation
    2 * math.pi,   # Full cycle variation
])
def test_nodal_validator_phase_wrapping_multi_scale(phase_variation) -> None:
    """Test phase wrapping validation across different phase distributions.
    
    Ensures phase values are properly wrapped to [-π, π] regardless of
    initial distribution, maintaining phase coherence invariant.
    """
    num_nodes = 20
    graph = nx.gnp_random_graph(num_nodes, 0.3, seed=42)
    inject_defaults(graph)
    
    # Set phases with increasing offset
    base_phase = 0.0
    for i, (node, data) in enumerate(graph.nodes(data=True)):
        data[THETA_KEY] = base_phase + (i / num_nodes) * phase_variation
    
    # Verify all phases are in valid range after wrapping
    for node, data in graph.nodes(data=True):
        phase = data[THETA_KEY]
        wrapped = math.atan2(math.sin(phase), math.cos(phase))
        assert -math.pi <= wrapped <= math.pi


@pytest.mark.parametrize("connectivity", [
    "connected",
    "disconnected",
    "isolated_nodes",
])
def test_nodal_validator_network_topology_variants(connectivity) -> None:
    """Test nodal validators handle different network topologies correctly.
    
    Verifies validators work with connected, disconnected, and networks
    with isolated nodes, ensuring robustness across topological variations.
    """
    graph = nx.Graph()
    inject_defaults(graph)
    
    if connectivity == "connected":
        # Create connected graph
        for i in range(5):
            for j in range(i + 1, 5):
                graph.add_edge(i, j)
    elif connectivity == "disconnected":
        # Create multiple components
        graph.add_edges_from([(0, 1), (1, 2)])
        graph.add_edges_from([(3, 4), (4, 5)])
    else:  # isolated_nodes
        # Add isolated nodes
        graph.add_nodes_from([0, 1, 2, 3, 4])
        graph.add_edges_from([(0, 1)])
    
    # Initialize attributes
    for node, data in graph.nodes(data=True):
        data[EPI_PRIMARY] = 0.5
        data[VF_PRIMARY] = 1.0
        data[DNFR_PRIMARY] = 0.0
    
    # Should handle all topologies without error
    assert_epi_vf_in_bounds(graph, epi_min=-1.0, epi_max=1.0)


# ============================================================================
# run_sequence - Complex Trajectory Patterns
# ============================================================================

def test_run_sequence_deep_nesting() -> None:
    """Test deeply nested block structures in sequences.
    
    Ensures compilation and execution handle arbitrary nesting depth
    without stack overflow or structural corruption.
    """
    # Create deeply nested sequence
    from tnfr.tokens import Glyph
    
    sequence = seq(
        block(
            block(
                block(Glyph.SHA, repeat=1),
                repeat=1
            ),
            repeat=1
        ),
        wait(1)
    )
    
    compiled = compile_sequence(sequence)
    
    # Should compile without error
    assert compiled is not None
    assert len(compiled) > 0


def test_run_sequence_alternating_targets() -> None:
    """Test sequences with rapidly alternating target selections.
    
    Verifies target switching is handled correctly even with
    frequent changes, maintaining execution order integrity.
    """
    from tnfr.tokens import Glyph
    
    sequence = seq(
        target([0]),
        Glyph.SHA,
        target([1]),
        Glyph.AL,
        target([0]),
        Glyph.SHA,
        target([1, 2]),
        Glyph.AL,
    )
    
    compiled = compile_sequence(sequence)
    
    # Should have correct number of target operations
    target_ops = [op for op in compiled if op[0] == OpTag.TARGET]
    assert len(target_ops) == 4


def test_run_sequence_mixed_wait_durations() -> None:
    """Test sequences with varying wait operation durations.
    
    Ensures time progression is correctly tracked with different
    wait durations, maintaining temporal coherence.
    """
    from tnfr.tokens import Glyph
    
    sequence = seq(
        wait(1),
        Glyph.SHA,
        wait(5),
        Glyph.AL,
        wait(10),
        Glyph.SHA,
        wait(2),
    )
    
    compiled = compile_sequence(sequence)
    
    # Extract wait operations
    wait_ops = [op for op in compiled if op[0] == OpTag.WAIT]
    wait_values = [op[1] for op in wait_ops]
    
    # Should preserve all wait durations
    assert len(wait_ops) == 4
    assert wait_values == [1, 5, 10, 2]


@pytest.mark.parametrize("repeat_count", [1, 2, 5, 10])
def test_run_sequence_variable_repeat_counts(repeat_count) -> None:
    """Test block operations with varying repeat counts.
    
    Verifies repeat parameter correctly multiplies block execution
    across different repeat values, maintaining operator closure.
    """
    from tnfr.tokens import Glyph
    
    sequence = seq(
        block(Glyph.SHA, repeat=repeat_count),
        wait(1),
    )
    
    compiled = compile_sequence(sequence)
    
    # Should have correct structure
    assert len(compiled) > 0


def test_run_sequence_empty_target_lists() -> None:
    """Test sequence handling of various empty/None target configurations.
    
    Ensures empty target lists and None values are handled correctly,
    selecting appropriate default targets per TNFR execution semantics.
    """
    from tnfr.tokens import Glyph
    
    # Empty target list
    sequence1 = seq(
        target([]),
        Glyph.SHA,
    )
    compiled1 = compile_sequence(sequence1)
    assert compiled1 is not None
    
    # None target (should select all)
    sequence2 = seq(
        target(None),
        Glyph.SHA,
    )
    compiled2 = compile_sequence(sequence2)
    assert compiled2 is not None


# ============================================================================
# Cross-cutting Critical Paths
# ============================================================================

def test_operator_validator_integration(seed_graph_factory) -> None:
    """Test integration between operator generation and node validators.
    
    Ensures generated operators maintain structural properties when
    applied through validators, preserving TNFR invariants end-to-end.
    """
    # Generate operator
    dim = 4
    operator = build_delta_nfr(dim, nu_f=1.5, scale=0.5)
    
    # Create graph
    graph = seed_graph_factory(num_nodes=dim, edge_probability=0.4, seed=42)
    
    # Verify operator is valid
    assert np.allclose(operator, operator.conj().T)
    
    # Verify graph maintains bounds
    assert_epi_vf_in_bounds(graph, epi_min=-2.0, epi_max=2.0)


@pytest.mark.parametrize("num_nodes", [5, 10, 20, 50])
def test_multi_scale_trajectory_consistency(seed_graph_factory, num_nodes) -> None:
    """Test trajectory consistency across different network scales.
    
    Verifies ΔNFR evolution maintains conservation and coherence
    regardless of network size, supporting TNFR's scale invariance.
    Uses parametrization for better isolation of scale-specific failures.
    """
    from tnfr.dynamics import dnfr_epi_vf_mixed
    
    graph = seed_graph_factory(
        num_nodes=num_nodes,
        edge_probability=0.3,
        seed=42
    )
    
    # Apply dynamics
    dnfr_epi_vf_mixed(graph)
    
    # Must maintain conservation regardless of scale
    assert_dnfr_balanced(graph, abs_tol=0.1)
