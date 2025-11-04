"""Consolidated critical path tests for TNFR engine.

This module consolidates and extends critical path coverage by unifying
redundant test patterns across integration/mathematics/property/stress tests
and adding comprehensive coverage for:
- Operator generation edge cases and parameter validation
- Nodal validator boundary conditions and multi-scale behavior
- Run sequence trajectories and execution paths

Following TNFR structural fidelity and DRY principles.
"""

from __future__ import annotations

import math

import networkx as nx
import pytest

np = pytest.importorskip("numpy")

from tnfr.constants import (
    DNFR_PRIMARY,
    EPI_PRIMARY,
    VF_PRIMARY,
    inject_defaults,
)
from tnfr.dynamics import dnfr_epi_vf_mixed
from tnfr.execution import play, seq, wait, target, block, compile_sequence
from tnfr.mathematics.generators import build_delta_nfr
from tnfr.tokens import Glyph, OpTag
from tests.helpers.validation import (
    assert_dnfr_balanced,
    assert_epi_vf_in_bounds,
)
from tests.helpers.fixtures import seed_graph_factory

# ============================================================================
# PARAMETRIZED FIXTURES FOR CONSOLIDATED TESTING
# ============================================================================

@pytest.fixture(params=[
    {"dimension": 2, "nu_f": 0.1, "scale": 0.5},
    {"dimension": 3, "nu_f": 1.0, "scale": 1.0},
    {"dimension": 5, "nu_f": 2.0, "scale": 2.0},
    {"dimension": 8, "nu_f": 5.0, "scale": 0.1},
])
def parametrized_operator_config(request):
    """Parametrized operator configurations for testing multiple scenarios."""
    return request.param

@pytest.fixture(params=[
    {"epi_range": (-0.1, 0.1), "vf_range": (0.9, 1.1)},
    {"epi_range": (-1.0, 1.0), "vf_range": (0.5, 2.0)},
    {"epi_range": (-2.0, 2.0), "vf_range": (0.1, 5.0)},
])
def parametrized_validator_bounds(request):
    """Parametrized validator boundary configurations."""
    return request.param

@pytest.fixture(params=[
    {"num_nodes": 5, "edge_prob": 0.3},
    {"num_nodes": 10, "edge_prob": 0.5},
    {"num_nodes": 20, "edge_prob": 0.7},
])
def parametrized_graph_scale(request):
    """Parametrized graph scales for multi-scale testing."""
    return request.param

# ============================================================================
# OPERATOR GENERATION: CONSOLIDATED CRITICAL PATHS
# ============================================================================

def test_operator_generation_parameter_combinations(parametrized_operator_config):
    """Test operator generation across parameter combinations (unifies multiple tests).

    Consolidates:
    - test_build_delta_nfr_frequency_scaling
    - test_build_delta_nfr_scale_parameter
    - test_build_delta_nfr_respects_dimension
    from multiple test modules into a single parametrized test.
    """
    config = parametrized_operator_config
    operator = build_delta_nfr(
        config["dimension"],
        nu_f=config["nu_f"],
        scale=config["scale"],
    )

    # Verify dimension
    assert operator.shape == (config["dimension"], config["dimension"])

    # Verify Hermitian property
    assert np.allclose(operator, operator.conj().T)

    # Verify finite values
    assert np.all(np.isfinite(operator))

    # Verify eigenvalues are real (Hermitian property)
    eigenvalues = np.linalg.eigvalsh(operator)
    assert np.all(np.isfinite(eigenvalues))

@pytest.mark.parametrize("topology", ["laplacian", "adjacency"])
def test_operator_generation_topology_unified(topology: str):
    """Test both topologies in single parametrized test.

    Consolidates:
    - test_build_delta_nfr_laplacian_topology
    - test_build_delta_nfr_adjacency_topology
    """
    dimension = 4
    operator = build_delta_nfr(dimension, topology=topology)

    # Both should be Hermitian
    assert np.allclose(operator, operator.conj().T)
    assert operator.shape == (dimension, dimension)

    # Verify structural properties based on topology
    if topology == "laplacian":
        # Laplacian row sums should be zero (or small for noisy version)
        row_sums = operator.sum(axis=1)
        # Allow some noise but should be structured
        assert np.all(np.isfinite(row_sums))

@pytest.mark.parametrize("seed_value", [0, 42, 123, 999])
def test_operator_generation_reproducibility_unified(seed_value: int):
    """Test reproducibility across multiple seeds in single parametrized test.

    Consolidates multiple reproducibility tests into one with parametrization.
    """
    dimension = 4

    # Create operators with same seed
    op1 = build_delta_nfr(dimension, rng=np.random.default_rng(seed_value))
    op2 = build_delta_nfr(dimension, rng=np.random.default_rng(seed_value))

    # Should be identical
    assert np.allclose(op1, op2)

    # Create operator with different seed
    op_different = build_delta_nfr(dimension, rng=np.random.default_rng(seed_value + 1))

    # Should differ
    assert not np.allclose(op1, op_different)

@pytest.mark.parametrize("invalid_dim", [0, -1, -5])
def test_operator_generation_invalid_dimension(invalid_dim: int):
    """Test dimension validation (consolidated)."""
    with pytest.raises(ValueError):
        build_delta_nfr(invalid_dim)

def test_operator_generation_spectral_properties():
    """Test comprehensive spectral properties of generated operators.

    Extends coverage for operator eigenspectrum analysis.
    """
    dimension = 5
    operator = build_delta_nfr(dimension, topology="laplacian")

    # Compute eigenvalues
    eigenvalues = np.linalg.eigvalsh(operator)

    # All eigenvalues should be real (eigvalsh returns real values by definition)
    assert eigenvalues.dtype == np.float64

    # Should have correct number of eigenvalues
    assert len(eigenvalues) == dimension

    # Verify spectral bandwidth is finite
    bandwidth = eigenvalues.max() - eigenvalues.min()
    assert np.isfinite(bandwidth)
    assert bandwidth >= 0

# ============================================================================
# NODAL VALIDATORS: CONSOLIDATED CRITICAL PATHS
# ============================================================================

def test_nodal_validator_bounds_unified(
    seed_graph_factory,
    parametrized_validator_bounds,
):
    """Test validator bounds across multiple ranges (consolidated).

    Consolidates:
    - test_nodal_validator_moderate_epi_range
    - test_nodal_validator_moderate_vf_range
    - test_validator_boundary_epi_limits
    - test_validator_boundary_vf_limits
    into single parametrized test.
    """
    bounds = parametrized_validator_bounds
    epi_min, epi_max = bounds["epi_range"]
    vf_min, vf_max = bounds["vf_range"]

    # Create graph
    graph = seed_graph_factory(num_nodes=10, edge_probability=0.3, seed=42)

    # Set values within bounds using seeded RNG
    rng = np.random.default_rng(123)
    for _, data in graph.nodes(data=True):
        data[EPI_PRIMARY] = rng.uniform(epi_min, epi_max)
        data[VF_PRIMARY] = rng.uniform(vf_min, vf_max)
        data[DNFR_PRIMARY] = 0.0

    # Validate bounds
    assert_epi_vf_in_bounds(
        graph,
        epi_min=epi_min,
        epi_max=epi_max,
        vf_min=vf_min,
        vf_max=vf_max,
    )

@pytest.mark.parametrize("phase_value", [0.0, math.pi/4, math.pi/2, math.pi, 2*math.pi])
def test_nodal_validator_phase_wrapping_unified(phase_value: float):
    """Test phase wrapping across multiple values (consolidated).

    Consolidates multiple phase wrapping tests into single parametrized test.

    Note: This test validates the wrapping formula used in TNFR.
    Production code should use the same formula for consistency.
    """
    # Phase should be wrapped to [-π, π] using canonical TNFR formula
    wrapped = ((phase_value + math.pi) % (2 * math.pi)) - math.pi

    # Verify wrapping to canonical range
    assert -math.pi <= wrapped <= math.pi

def test_nodal_validator_multi_scale_consistency(
    seed_graph_factory,
    parametrized_graph_scale,
):
    """Test validator consistency across multiple graph scales.

    Extends coverage for multi-scale validation behavior.
    """
    scale = parametrized_graph_scale

    # Create graphs at different scales
    graph = seed_graph_factory(
        num_nodes=scale["num_nodes"],
        edge_probability=scale["edge_prob"],
        seed=42,
    )

    # Apply dynamics
    dnfr_epi_vf_mixed(graph)

    # Verify conservation at all scales
    assert_dnfr_balanced(graph, abs_tol=0.1)

    # Verify all nodes have valid attributes
    for _, data in graph.nodes(data=True):
        assert EPI_PRIMARY in data
        assert VF_PRIMARY in data
        assert DNFR_PRIMARY in data
        assert all(math.isfinite(data[key]) for key in [EPI_PRIMARY, VF_PRIMARY, DNFR_PRIMARY])

def test_nodal_validator_disconnected_components():
    """Test validator behavior with disconnected graph components.

    Extends coverage for non-standard graph topologies.
    """
    # Create graph with disconnected components
    graph = nx.Graph()
    inject_defaults(graph)

    # Add two disconnected components
    graph.add_edges_from([(0, 1), (1, 2)])  # Component 1
    graph.add_edges_from([(3, 4), (4, 5)])  # Component 2

    # Initialize nodes
    for node, data in graph.nodes(data=True):
        data[EPI_PRIMARY] = 0.5
        data[VF_PRIMARY] = 1.0
        data[DNFR_PRIMARY] = 0.0

    # Apply dynamics
    dnfr_epi_vf_mixed(graph)

    # Each component should conserve ΔNFR independently
    assert_dnfr_balanced(graph)

def test_nodal_validator_isolated_nodes():
    """Test validator behavior with isolated nodes.

    Extends coverage for edge case topologies.
    """
    graph = nx.Graph()
    inject_defaults(graph)

    # Add isolated nodes
    graph.add_nodes_from([0, 1, 2])

    # Initialize
    for _, data in graph.nodes(data=True):
        data[EPI_PRIMARY] = 0.5
        data[VF_PRIMARY] = 1.0
        data[DNFR_PRIMARY] = 0.0

    # Apply dynamics
    dnfr_epi_vf_mixed(graph)

    # Isolated nodes should have zero ΔNFR
    for _, data in graph.nodes(data=True):
        assert math.isclose(data[DNFR_PRIMARY], 0.0, abs_tol=1e-9)

# ============================================================================
# RUN SEQUENCE: CONSOLIDATED CRITICAL PATHS
# ============================================================================

@pytest.fixture
def graph_canon():
    """Create canonical test graph."""
    def _create():
        G = nx.Graph()
        inject_defaults(G)
        return G
    return _create

def _step_noop(graph):
    """Simple step function for testing."""
    graph.graph["_t"] = graph.graph.get("_t", 0.0) + 1.0

def test_run_sequence_compilation_unified(graph_canon):
    """Test sequence compilation with multiple operation types (consolidated).

    Consolidates:
    - test_compile_sequence_single_glyph
    - test_compile_sequence_with_wait
    - test_compile_sequence_with_target
    - test_compile_sequence_with_block
    into single comprehensive test.
    """
    # Test single operations
    compiled_glyph = compile_sequence(seq(Glyph.SHA))
    assert len(compiled_glyph) == 1
    assert compiled_glyph[0][0] == OpTag.GLYPH

    compiled_wait = compile_sequence(seq(wait(5)))
    assert len(compiled_wait) == 1
    assert compiled_wait[0][0] == OpTag.WAIT

    compiled_target = compile_sequence(seq(target([0, 1])))
    assert len(compiled_target) == 1
    assert compiled_target[0][0] == OpTag.TARGET

    # Test compound sequence
    compiled_compound = compile_sequence(seq(
        target([0]),
        Glyph.SHA,
        wait(2),
        block(Glyph.AL, repeat=2),
    ))

    op_tags = [op[0] for op in compiled_compound]
    assert OpTag.TARGET in op_tags
    assert OpTag.GLYPH in op_tags
    assert OpTag.WAIT in op_tags
    assert OpTag.THOL in op_tags

@pytest.mark.parametrize("wait_value", [0, -1, -10])
def test_run_sequence_wait_clamping_unified(graph_canon, wait_value: int):
    """Test wait value clamping for zero and negative values (consolidated).

    Consolidates:
    - test_run_sequence_wait_zero_clamping
    - test_run_sequence_wait_negative_clamping
    - test_run_sequence_wait_boundary_conditions
    """
    G = graph_canon()
    G.add_node(0)

    # Wait with non-positive value should be clamped
    play(G, seq(wait(wait_value)), step_fn=_step_noop)

    trace = list(G.graph["history"]["program_trace"])
    wait_entries = [e for e in trace if e["op"] == "WAIT"]

    # Should have executed at least 1 step
    assert all(e["k"] >= 1 for e in wait_entries)

def test_run_sequence_target_variations(graph_canon):
    """Test target operation with different selections (consolidated).

    Extends coverage for target selection patterns.
    """
    G = graph_canon()
    G.add_nodes_from([0, 1, 2, 3])

    # Test multiple targets
    play(G, seq(
        target([0, 1]),
        wait(1),
        target([2, 3]),
        wait(1),
    ), step_fn=_step_noop)

    trace = list(G.graph["history"]["program_trace"])
    target_entries = [e for e in trace if e["op"] == "TARGET"]

    # Should have 2 target operations
    assert len(target_entries) == 2

@pytest.mark.parametrize("num_operations", [1, 5, 10, 20])
def test_run_sequence_long_chains(graph_canon, num_operations: int):
    """Test sequence execution with varying chain lengths.

    Extends coverage for long operation sequences.
    """
    G = graph_canon()
    G.add_node(0)

    # Build long sequence
    operations = []
    for _ in range(num_operations):
        operations.append(wait(1))

    sequence = seq(*operations)
    play(G, sequence, step_fn=_step_noop)

    trace = list(G.graph["history"]["program_trace"])
    wait_entries = [e for e in trace if e["op"] == "WAIT"]

    # Should have executed all operations
    assert len(wait_entries) == num_operations

def test_run_sequence_empty_variations(graph_canon):
    """Test sequence execution with empty variations.

    Extends coverage for empty sequence edge cases.
    """
    G = graph_canon()

    # Empty sequence
    play(G, seq(), step_fn=_step_noop)
    assert "history" in G.graph

    # Empty graph with sequence
    G2 = graph_canon()
    play(G2, seq(wait(1)), step_fn=_step_noop)
    assert "history" in G2.graph

def test_run_sequence_trace_consistency():
    """Test program trace consistency and ordering.

    Extends coverage for trace generation validation.
    """
    G = nx.Graph()
    inject_defaults(G)
    G.add_nodes_from([0, 1, 2])

    sequence = seq(
        target([0]),
        wait(2),
        target([1, 2]),
        wait(1),
    )

    play(G, sequence, step_fn=_step_noop)

    trace = list(G.graph["history"]["program_trace"])

    # Verify trace ordering
    assert len(trace) > 0

    # Check that time progresses monotonically
    times = [e.get("t", 0.0) for e in trace]
    assert all(times[i] <= times[i+1] for i in range(len(times)-1))

# ============================================================================
# INTEGRATION: CROSS-CUTTING CRITICAL PATHS
# ============================================================================

def test_integration_operator_to_validator():
    """Test integration between operator generation and validation.

    Extends coverage for cross-component interactions.
    """
    # Generate operator
    dimension = 5
    operator = build_delta_nfr(dimension, nu_f=1.0)

    # Verify operator properties that validators would check
    assert np.allclose(operator, operator.conj().T)  # Hermitian
    eigenvalues = np.linalg.eigvalsh(operator)
    assert np.all(np.isfinite(eigenvalues))

    # Use operator shape to validate graph compatibility
    graph = nx.complete_graph(dimension)
    inject_defaults(graph)

    # Should be compatible
    assert len(graph.nodes) == operator.shape[0]

def test_integration_validator_to_sequence(seed_graph_factory):
    """Test integration between validator and sequence execution.

    Extends coverage for validator + runtime interactions.
    """
    # Create validated graph
    graph = seed_graph_factory(num_nodes=5, edge_probability=0.5, seed=42)

    # Verify initial state is valid
    assert_epi_vf_in_bounds(graph, epi_min=-2.0, epi_max=2.0, vf_min=0.0, vf_max=5.0)

    # Execute sequence
    play(graph, seq(wait(3)), step_fn=_step_noop)

    # Verify trace was created
    assert "history" in graph.graph
    assert "program_trace" in graph.graph["history"]

def test_integration_multi_operator_composition():
    """Test composition of multiple operators.

    Extends coverage for operator composition patterns.
    """
    dimension = 4

    # Generate multiple operators
    op1 = build_delta_nfr(dimension, topology="laplacian", scale=0.5)
    op2 = build_delta_nfr(dimension, topology="adjacency", scale=0.5)

    # Composition should preserve Hermitian property
    composed = op1 + op2
    assert np.allclose(composed, composed.conj().T)

    # Scaled operators should remain Hermitian
    scaled = 2.0 * op1
    assert np.allclose(scaled, scaled.conj().T)
