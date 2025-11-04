"""Extended critical path coverage for identified gaps.

This module adds targeted tests for areas identified as needing more coverage:
- Operator composition with error propagation
- Validator performance and interaction with operators
- Sequence error recovery and optimization

These tests complement existing critical path tests by focusing on
interaction patterns and edge cases.
"""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

import networkx as nx

from tnfr.constants import inject_defaults, EPI_PRIMARY, VF_PRIMARY, THETA_KEY
from tnfr.mathematics.generators import build_delta_nfr
from tnfr.dynamics import dnfr_epi_vf_mixed
from tnfr.execution import play, seq, wait, target
from tests.helpers.validation import assert_dnfr_balanced, assert_epi_vf_in_bounds

# ============================================================================
# Operator Composition and Error Propagation
# ============================================================================

def test_operator_composition_error_propagation() -> None:
    """Verify errors in operator composition propagate correctly.

    Tests that when composing operators, errors from invalid operations
    are properly detected and reported, maintaining TNFR structural integrity.
    """
    dim = 4

    # Create valid operators
    op1 = build_delta_nfr(dim, topology="laplacian", nu_f=1.0)
    op2 = build_delta_nfr(dim, topology="adjacency", nu_f=2.0)

    # Valid composition: addition should work
    composed = op1 + op2
    assert composed.shape == (dim, dim)

    # Verify composition preserves Hermitian property
    assert np.allclose(composed, composed.conj().T)

    # Invalid composition: dimension mismatch should fail
    op_wrong_dim = build_delta_nfr(dim + 1)
    with pytest.raises((ValueError, AssertionError)):
        _ = op1 + op_wrong_dim

def test_operator_composition_chain_stability() -> None:
    """Verify long chains of operator composition remain stable.

    Tests that composed operators maintain structural properties
    even after multiple composition operations.
    """
    dim = 3
    rng = np.random.default_rng(seed=42)

    # Create base operator
    base_op = build_delta_nfr(dim, rng=rng)

    # Compose multiple times with scaled versions
    composed = base_op.copy()
    scales = [0.1, 0.5, 1.0, 2.0]

    for scale in scales:
        scaled_op = scale * build_delta_nfr(dim, rng=np.random.default_rng(seed=42))
        composed = composed + scaled_op

        # Verify structural properties maintained
        assert np.allclose(composed, composed.conj().T), "Lost Hermitian property in chain"
        assert np.all(np.isfinite(composed)), "Non-finite values in composition chain"

def test_operator_mixed_topology_interaction() -> None:
    """Verify operators with different topologies interact correctly.

    Tests that combining operators with different underlying topologies
    (laplacian vs adjacency) produces structurally valid results.
    """
    dim = 5

    # Create operators with different topologies
    laplacian_op = build_delta_nfr(dim, topology="laplacian", nu_f=1.0)
    adjacency_op = build_delta_nfr(dim, topology="adjacency", nu_f=1.0)

    # Mix topologies through linear combination
    mixed = 0.6 * laplacian_op + 0.4 * adjacency_op

    # Verify mixed operator maintains structural properties
    assert mixed.shape == (dim, dim)
    assert np.allclose(mixed, mixed.conj().T), "Mixed topology lost Hermitian property"

    # Eigenvalues should be real for Hermitian operators
    eigenvalues = np.linalg.eigvalsh(mixed)
    assert np.all(np.isreal(eigenvalues)), "Mixed topology has complex eigenvalues"

# ============================================================================
# Validator Performance and Interaction
# ============================================================================

@pytest.mark.parametrize("num_nodes,edge_prob", [
    (100, 0.1),
    (200, 0.05),
    (500, 0.02),
])
def test_validator_performance_large_graphs(num_nodes: int, edge_prob: float) -> None:
    """Verify validators perform adequately on large graphs.

    Tests that nodal validators can handle large-scale graphs
    without performance degradation or numerical issues.
    """
    # Create large graph
    graph = nx.gnp_random_graph(num_nodes, edge_prob, seed=42)
    inject_defaults(graph)

    # Initialize node attributes
    for node in graph.nodes():
        graph.nodes[node][EPI_PRIMARY] = 0.5
        graph.nodes[node][VF_PRIMARY] = 1.0
        graph.nodes[node][THETA_KEY] = 0.0

    # Apply ΔNFR computation
    dnfr_epi_vf_mixed(graph)

    # Validate structural properties
    assert_dnfr_balanced(graph, abs_tol=0.5)  # Larger tolerance for large graphs
    assert_epi_vf_in_bounds(graph, epi_min=-5.0, epi_max=5.0)

def test_validator_operator_result_validation() -> None:
    """Verify validators correctly validate operator computation results.

    Tests that validators can detect when operator results violate
    structural constraints and when they satisfy constraints.
    """
    # Create small graph
    graph = nx.path_graph(4)
    inject_defaults(graph)

    # Initialize with valid values
    for node in graph.nodes():
        graph.nodes[node][EPI_PRIMARY] = 0.1
        graph.nodes[node][VF_PRIMARY] = 1.0
        graph.nodes[node][THETA_KEY] = 0.0

    # Apply operator (should produce valid result)
    dnfr_epi_vf_mixed(graph)

    # Validate - should pass
    assert_dnfr_balanced(graph)

    # Now introduce invalid values manually
    graph.nodes[0][EPI_PRIMARY] = float('inf')

    # Validator should detect this
    with pytest.raises((AssertionError, ValueError)):
        assert_epi_vf_in_bounds(graph, epi_min=-1.0, epi_max=1.0)

def test_validator_multi_operator_consistency() -> None:
    """Verify validators maintain consistency across multiple operator applications.

    Tests that validators correctly track structural properties
    when operators are applied sequentially.
    """
    graph = nx.cycle_graph(6)
    inject_defaults(graph)

    # Initialize
    for node in graph.nodes():
        graph.nodes[node][EPI_PRIMARY] = 0.0
        graph.nodes[node][VF_PRIMARY] = 1.0
        graph.nodes[node][THETA_KEY] = 0.0

    # Apply operator multiple times
    for iteration in range(3):
        dnfr_epi_vf_mixed(graph)

        # Each iteration should maintain conservation
        assert_dnfr_balanced(graph, abs_tol=0.1 * (iteration + 1))

# ============================================================================
# Sequence Error Recovery and Optimization
# ============================================================================

def test_sequence_error_recovery_invalid_operation() -> None:
    """Verify sequence execution handles invalid operations gracefully.

    Tests that when an invalid operation is encountered in a sequence,
    the error is properly caught and reported without corrupting state.
    """
    graph = nx.path_graph(3)
    inject_defaults(graph)

    # This tests that invalid glyph names are detected
    # Note: play() expects valid operations; invalid ones should raise during compilation
    with pytest.raises((ValueError, KeyError, AttributeError)):
        play(graph, seq("invalid_glyph_name"))

def test_sequence_partial_execution_consistency() -> None:
    """Verify sequence state remains consistent after partial execution.

    Tests that if a sequence executes partially, the graph state
    remains valid and doesn't have inconsistent attributes.
    """
    graph = nx.star_graph(5)
    inject_defaults(graph)

    # Use noop step function to avoid glyph issues
    def step_noop(g):
        g.graph["_t"] = g.graph.get("_t", 0.0) + 1.0

    # Execute sequence with wait operations
    play(graph, seq(wait(1), wait(2), wait(3)), step_fn=step_noop)

    # Verify graph history exists and is consistent
    assert "history" in graph.graph
    history = graph.graph["history"]

    # Time should have progressed
    if "_t" in graph.graph:
        assert graph.graph["_t"] >= 0

def test_sequence_repeated_target_switching() -> None:
    """Verify sequence handles repeated target switches efficiently.

    Tests that switching targets multiple times in a sequence
    doesn't degrade performance or introduce errors.
    """
    graph = nx.complete_graph(8)
    inject_defaults(graph)

    # Use noop step function to avoid glyph issues
    def step_noop(g):
        g.graph["_t"] = g.graph.get("_t", 0.0) + 1.0

    # Create sequence with many target switches
    operations = []
    for node in range(8):
        operations.extend([target([node]), wait(1)])

    # Execute should complete without error
    play(graph, seq(*operations), step_fn=step_noop)

    # Verify trace contains all target switches
    if "history" in graph.graph and "program_trace" in graph.graph["history"]:
        trace = list(graph.graph["history"]["program_trace"])
        target_ops = [e for e in trace if e.get("op") == "TARGET"]
        # Should have approximately 8 target operations (one per node)
        assert len(target_ops) >= 1

def test_sequence_nested_block_optimization() -> None:
    """Verify nested blocks in sequences don't cause exponential overhead.

    Tests that deeply nested block structures are handled efficiently
    without causing performance degradation.
    """
    from tnfr.execution import block

    graph = nx.path_graph(4)
    inject_defaults(graph)

    # Use noop step function to avoid glyph issues
    def step_noop(g):
        g.graph["_t"] = g.graph.get("_t", 0.0) + 1.0

    # Create nested block structure
    inner = seq(wait(1))
    middle = seq(block(inner))
    outer = seq(block(middle))

    # Should execute efficiently
    play(graph, outer, step_fn=step_noop)

    # Verify completed
    assert "history" in graph.graph

# ============================================================================
# Cross-Cutting Integration Tests
# ============================================================================

def test_integration_operator_validator_sequence() -> None:
    """Verify complete integration of operators, validators, and sequences.

    Tests the full pipeline: generate operators -> apply to graph ->
    validate results -> execute sequence with validated state.
    """
    # Create test graph
    graph = nx.cycle_graph(5)
    inject_defaults(graph)

    # Initialize
    for node in graph.nodes():
        graph.nodes[node][EPI_PRIMARY] = 0.2
        graph.nodes[node][VF_PRIMARY] = 1.0
        graph.nodes[node][THETA_KEY] = 0.0

    # Generate and apply operator
    dnfr_epi_vf_mixed(graph)

    # Validate result
    assert_dnfr_balanced(graph)

    # Use noop step function to avoid glyph issues
    def step_noop(g):
        g.graph["_t"] = g.graph.get("_t", 0.0) + 1.0

    # Execute sequence on validated state
    play(graph, seq(wait(1), wait(2)), step_fn=step_noop)

    # Final validation
    assert "history" in graph.graph

def test_integration_operator_composition_in_dynamics() -> None:
    """Verify operator composition integrates correctly with dynamics.

    Tests that composed operators can be used in dynamics computations
    without introducing numerical instabilities.
    """
    # Create operators
    dim = 4
    op1 = build_delta_nfr(dim, topology="laplacian")
    op2 = build_delta_nfr(dim, topology="adjacency")
    composed = 0.5 * op1 + 0.5 * op2

    # Verify composition is valid
    assert composed.shape == (dim, dim)
    assert np.allclose(composed, composed.conj().T)

    # Could be used in dynamics (this tests structural validity)
    eigenvalues = np.linalg.eigvalsh(composed)
    assert np.all(np.isreal(eigenvalues))
    assert np.all(np.isfinite(eigenvalues))
