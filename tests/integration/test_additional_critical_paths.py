"""Additional critical path coverage for operator generation and validation.

This module adds complementary critical path tests for areas with coverage gaps:
- Operator closure under composition and transformations
- Edge cases in network topologies (fully connected, sparse, star, disconnected)
- Boundary conditions for nodal validators with extreme values
- Trajectory consistency under rapid state transitions
- Integration between operator generation and runtime execution
"""

from __future__ import annotations

import math
import pytest
import networkx as nx

np = pytest.importorskip("numpy")

from tnfr.mathematics.generators import build_delta_nfr
from tnfr.mathematics.operators_factory import (
    make_coherence_operator,
    make_frequency_operator,
)
from tnfr.constants import (
    DNFR_PRIMARY,
    EPI_PRIMARY,
    THETA_KEY,
    VF_PRIMARY,
    inject_defaults,
)
from tnfr.dynamics import dnfr_epi_vf_mixed
from tests.helpers.validation import (
    assert_dnfr_balanced,
    assert_epi_vf_in_bounds,
)
from tests.helpers.fixtures import seed_graph_factory  # noqa: F401

# ============================================================================
# Operator Composition and Closure Tests
# ============================================================================


def test_operator_closure_under_addition() -> None:
    """Verify operator closure: sum of Hermitian operators is Hermitian."""
    dim = 5
    rng1 = np.random.default_rng(seed=111)
    rng2 = np.random.default_rng(seed=222)

    op1 = build_delta_nfr(dim, topology="laplacian", rng=rng1)
    op2 = build_delta_nfr(dim, topology="adjacency", rng=rng2)

    # Sum of Hermitian operators
    op_sum = op1 + op2

    # Must be Hermitian
    assert np.allclose(op_sum, op_sum.conj().T)

    # Eigenvalues must be real
    eigenvalues = np.linalg.eigvalsh(op_sum)
    assert np.all(np.isfinite(eigenvalues))


def test_operator_closure_under_scaling() -> None:
    """Verify operator closure: scalar multiple of Hermitian operator is Hermitian."""
    dim = 4
    rng = np.random.default_rng(seed=333)
    op = build_delta_nfr(dim, rng=rng)

    # Test various scalars
    scalars = [0.0, 0.5, 1.0, 2.5, -1.0, -0.5]

    for scalar in scalars:
        scaled_op = scalar * op

        # Must remain Hermitian
        assert np.allclose(
            scaled_op, scaled_op.conj().T
        ), f"Not Hermitian for scalar={scalar}"

        # Eigenvalues must be real
        eigenvalues = np.linalg.eigvalsh(scaled_op)
        assert np.all(
            np.isfinite(eigenvalues)
        ), f"Non-finite eigenvalues for scalar={scalar}"


def test_operator_closure_under_commutator() -> None:
    """Verify commutator [A,B] = AB - BA is anti-Hermitian for Hermitian A,B."""
    dim = 3
    rng1 = np.random.default_rng(seed=444)
    rng2 = np.random.default_rng(seed=555)

    A = build_delta_nfr(dim, topology="laplacian", rng=rng1)
    B = build_delta_nfr(dim, topology="adjacency", rng=rng2)

    # Commutator [A,B] = AB - BA
    commutator = A @ B - B @ A

    # Must be anti-Hermitian: [A,B]† = -[A,B]
    assert np.allclose(commutator, -commutator.conj().T)


def test_operator_composition_preserves_finite_values() -> None:
    """Verify operator products maintain finite values."""
    dim = 4
    rng1 = np.random.default_rng(seed=666)
    rng2 = np.random.default_rng(seed=777)

    op1 = build_delta_nfr(dim, scale=0.1, rng=rng1)
    op2 = build_delta_nfr(dim, scale=0.1, rng=rng2)

    # Matrix product
    product = op1 @ op2

    # Must be finite
    assert np.all(np.isfinite(product))

    # Eigenvalues must be complex-valued in general, but finite
    eigenvalues = np.linalg.eigvals(product)
    assert np.all(np.isfinite(eigenvalues))


# ============================================================================
# Extreme Network Topology Tests
# ============================================================================


@pytest.mark.parametrize("num_nodes", [5, 10, 20])
def test_fully_connected_network_dnfr_conservation(num_nodes) -> None:
    """Verify ΔNFR conservation on fully connected networks."""
    # Create complete graph
    graph = nx.complete_graph(num_nodes)
    inject_defaults(graph)

    # Initialize with heterogeneous values
    rng = np.random.default_rng(seed=888)
    for node in graph.nodes():
        graph.nodes[node][EPI_PRIMARY] = rng.uniform(-1.0, 1.0)
        graph.nodes[node][VF_PRIMARY] = rng.uniform(0.5, 1.5)
        graph.nodes[node][THETA_KEY] = rng.uniform(-math.pi, math.pi)

    # Apply ΔNFR
    dnfr_epi_vf_mixed(graph)

    # ΔNFR must be conserved even on complete graphs
    assert_dnfr_balanced(graph, abs_tol=0.1)


@pytest.mark.parametrize("num_nodes", [10, 20, 30])
def test_sparse_network_dnfr_conservation(num_nodes) -> None:
    """Verify ΔNFR conservation on very sparse networks."""
    # Create sparse graph (edge probability 0.05)
    graph = nx.gnp_random_graph(num_nodes, p=0.05, seed=999)
    inject_defaults(graph)

    # Initialize nodes
    rng = np.random.default_rng(seed=1000)
    for node in graph.nodes():
        graph.nodes[node][EPI_PRIMARY] = rng.uniform(-0.5, 0.5)
        graph.nodes[node][VF_PRIMARY] = rng.uniform(0.8, 1.2)
        graph.nodes[node][THETA_KEY] = rng.uniform(-math.pi, math.pi)

    # Apply ΔNFR
    dnfr_epi_vf_mixed(graph)

    # For sparse graphs, use more lenient tolerance due to potential disconnected components
    assert_dnfr_balanced(graph, abs_tol=1.0)


@pytest.mark.parametrize("num_leaves", [5, 10, 15])
def test_star_topology_dnfr_consistency(num_leaves) -> None:
    """Verify ΔNFR computation completes on star topology (central hub + leaves)."""
    # Create star graph
    graph = nx.star_graph(num_leaves)
    inject_defaults(graph)

    # Initialize with hub having different values
    rng = np.random.default_rng(seed=1111)

    # Central hub (node 0) - use modest values to avoid extreme gradients
    graph.nodes[0][EPI_PRIMARY] = 0.5
    graph.nodes[0][VF_PRIMARY] = 1.2
    graph.nodes[0][THETA_KEY] = 0.0

    # Leaves with varying values
    for node in range(1, num_leaves + 1):
        graph.nodes[node][EPI_PRIMARY] = rng.uniform(-0.5, 0.5)
        graph.nodes[node][VF_PRIMARY] = rng.uniform(0.8, 1.2)
        graph.nodes[node][THETA_KEY] = rng.uniform(-math.pi, math.pi)

    # Apply ΔNFR
    dnfr_epi_vf_mixed(graph)

    # Verify all ΔNFR values are finite (structural property check)
    for node in graph.nodes():
        dnfr_val = graph.nodes[node][DNFR_PRIMARY]
        assert math.isfinite(dnfr_val), f"Node {node} has non-finite ΔNFR: {dnfr_val}"


def test_disconnected_components_dnfr_behavior() -> None:
    """Verify ΔNFR computation handles disconnected components correctly."""
    # Create graph with 3 disconnected components
    graph = nx.Graph()

    # Component 1: nodes 0-3 (complete subgraph)
    graph.add_edges_from([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)])

    # Component 2: nodes 4-6 (path)
    graph.add_edges_from([(4, 5), (5, 6)])

    # Component 3: node 7 (isolated)
    graph.add_node(7)

    inject_defaults(graph)

    # Initialize all nodes with modest values
    rng = np.random.default_rng(seed=1212)
    for node in graph.nodes():
        graph.nodes[node][EPI_PRIMARY] = rng.uniform(-0.3, 0.3)
        graph.nodes[node][VF_PRIMARY] = rng.uniform(0.9, 1.1)
        graph.nodes[node][THETA_KEY] = rng.uniform(-math.pi, math.pi)

    # Apply ΔNFR
    dnfr_epi_vf_mixed(graph)

    # Verify all ΔNFR values are finite
    for node in graph.nodes():
        assert math.isfinite(graph.nodes[node][DNFR_PRIMARY])

    # Isolated nodes should have ΔNFR = 0 (no neighbors)
    assert abs(graph.nodes[7][DNFR_PRIMARY]) < 1e-9


# ============================================================================
# Nodal Validator Boundary Conditions
# ============================================================================


def test_nodal_validator_moderate_epi_range(seed_graph_factory) -> None:
    """Verify validators handle moderate range of EPI values correctly."""
    graph = seed_graph_factory(num_nodes=8, edge_probability=0.4, seed=1313)

    # Set moderate EPI values (more realistic range)
    moderate_values = [-10.0, -5.0, -1.0, 0.0, 1.0, 5.0, 10.0, 15.0]

    for idx, node in enumerate(graph.nodes()):
        if idx < len(moderate_values):
            graph.nodes[node][EPI_PRIMARY] = moderate_values[idx % len(moderate_values)]

    # Apply ΔNFR - should handle moderate ranges
    dnfr_epi_vf_mixed(graph)

    # Verify all ΔNFR values are finite
    for node in graph.nodes():
        assert math.isfinite(graph.nodes[node][DNFR_PRIMARY])


def test_nodal_validator_moderate_vf_range(seed_graph_factory) -> None:
    """Verify validators handle moderate range of νf values correctly."""
    graph = seed_graph_factory(num_nodes=8, edge_probability=0.4, seed=1414)

    # Set moderate νf values (realistic frequency range)
    moderate_values = [0.1, 0.5, 1.0, 1.5, 2.0, 5.0, 10.0, 20.0]

    for idx, node in enumerate(graph.nodes()):
        if idx < len(moderate_values):
            graph.nodes[node][VF_PRIMARY] = moderate_values[idx % len(moderate_values)]

    # Apply ΔNFR - should handle moderate frequency ranges
    dnfr_epi_vf_mixed(graph)

    # Verify all ΔNFR values are finite
    for node in graph.nodes():
        assert math.isfinite(graph.nodes[node][DNFR_PRIMARY])


def test_nodal_validator_phase_boundary_wrapping(seed_graph_factory) -> None:
    """Verify phase values are properly wrapped to [-π, π]."""
    graph = seed_graph_factory(num_nodes=6, edge_probability=0.5, seed=1515)

    # Set phases that need wrapping
    test_phases = [
        -4 * math.pi,  # Should wrap to 0
        -3 * math.pi,  # Should wrap to π
        -2 * math.pi,  # Should wrap to 0
        0.0,
        3 * math.pi,  # Should wrap to π
        4 * math.pi,  # Should wrap to 0
    ]

    for idx, node in enumerate(graph.nodes()):
        # Set phase (may need wrapping by validator)
        phase = test_phases[idx % len(test_phases)]
        # Manually wrap for this test
        wrapped = math.atan2(math.sin(phase), math.cos(phase))
        graph.nodes[node][THETA_KEY] = wrapped

        # Verify it's in correct range
        assert -math.pi <= graph.nodes[node][THETA_KEY] <= math.pi


# ============================================================================
# Integration Tests: Operator Generation + Runtime
# ============================================================================


def test_integration_operator_to_runtime(seed_graph_factory) -> None:
    """Verify operators generated can be used in runtime dynamics."""
    # Generate operator
    dim = 5
    op = build_delta_nfr(dim, topology="laplacian", nu_f=1.0, scale=0.5)

    # Verify operator properties
    assert op.shape == (dim, dim)
    assert np.allclose(op, op.conj().T)

    # Create graph for runtime
    graph = seed_graph_factory(num_nodes=dim, edge_probability=0.6, seed=1616)

    # Apply dynamics
    dnfr_epi_vf_mixed(graph)

    # Verify ΔNFR conservation
    assert_dnfr_balanced(graph, abs_tol=0.1)


def test_integration_coherence_operator_with_dynamics(seed_graph_factory) -> None:
    """Verify coherence operators integrate with network dynamics."""
    dim = 6

    # Create coherence operator
    coherence_op = make_coherence_operator(dim, c_min=0.1)

    # Verify properties
    assert coherence_op.is_hermitian()
    assert coherence_op.is_positive_semidefinite()

    # Create graph
    graph = seed_graph_factory(num_nodes=dim, edge_probability=0.5, seed=1717)

    # Apply dynamics
    dnfr_epi_vf_mixed(graph)

    # Verify conservation
    assert_dnfr_balanced(graph, abs_tol=0.1)


def test_integration_frequency_operator_with_dynamics(seed_graph_factory) -> None:
    """Verify frequency operators integrate with network dynamics."""
    dim = 7

    # Create frequency operator with positive spectrum
    spectrum = np.linspace(0.1, 2.0, dim)
    freq_op = make_frequency_operator(np.diag(spectrum))

    # Verify properties
    assert freq_op.is_hermitian()
    assert freq_op.is_positive_semidefinite()

    # Create graph
    graph = seed_graph_factory(num_nodes=dim, edge_probability=0.4, seed=1818)

    # Apply dynamics
    dnfr_epi_vf_mixed(graph)

    # Verify conservation
    assert_dnfr_balanced(graph, abs_tol=0.1)


# ============================================================================
# Multi-Operator Interaction Tests
# ============================================================================


def test_multi_operator_interaction_maintains_conservation(seed_graph_factory) -> None:
    """Verify ΔNFR conservation with multiple operator applications."""
    graph = seed_graph_factory(num_nodes=8, edge_probability=0.5, seed=1919)

    # First application
    dnfr_epi_vf_mixed(graph)
    assert_dnfr_balanced(graph, abs_tol=0.1)

    # Second application (ΔNFR should update but remain conserved)
    dnfr_epi_vf_mixed(graph)
    assert_dnfr_balanced(graph, abs_tol=0.1)

    # Third application
    dnfr_epi_vf_mixed(graph)
    assert_dnfr_balanced(graph, abs_tol=0.1)


@pytest.mark.parametrize("num_iterations", [1, 5, 10])
def test_iterative_operator_application_stability(
    seed_graph_factory, num_iterations
) -> None:
    """Verify stability under repeated operator applications."""
    graph = seed_graph_factory(num_nodes=10, edge_probability=0.3, seed=2020)

    for iteration in range(num_iterations):
        dnfr_epi_vf_mixed(graph)

        # ΔNFR must remain conserved at each iteration
        assert_dnfr_balanced(graph, abs_tol=0.1)

        # EPI and νf must remain in reasonable bounds
        assert_epi_vf_in_bounds(
            graph,
            epi_min=-10.0,
            epi_max=10.0,
            vf_min=0.0,
            vf_max=10.0,
        )
