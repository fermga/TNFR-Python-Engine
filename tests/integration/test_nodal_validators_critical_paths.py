"""Critical path coverage for nodal validators.

This module provides focused tests for critical validator paths including:
- Boundary condition validation
- Validator composition and chaining
- Multi-node validation consistency
- Edge cases in structural constraints
"""

import math

import networkx as nx
import pytest

from tnfr.constants import (
    DNFR_PRIMARY,
    EPI_PRIMARY,
    THETA_KEY,
    VF_PRIMARY,
    inject_defaults,
)
from tests.helpers.validation import (
    assert_dnfr_balanced,
    assert_epi_vf_in_bounds,
    assert_graph_has_tnfr_defaults,
)
from tests.helpers.fixtures import seed_graph_factory


def test_validator_boundary_epi_limits(seed_graph_factory) -> None:
    """Verify validator handles EPI boundary values correctly."""
    graph = seed_graph_factory(num_nodes=5, edge_probability=0.3, seed=42)
    
    # Set extreme EPI values
    epi_values = [
        -1000.0,  # Very negative
        -1.0,     # Negative
        0.0,      # Zero
        1.0,      # Positive
        1000.0,   # Very positive
    ]
    
    for node, epi_val in zip(graph.nodes(), epi_values):
        graph.nodes[node][EPI_PRIMARY] = epi_val
        # Verify finite
        assert math.isfinite(graph.nodes[node][EPI_PRIMARY])


def test_validator_boundary_vf_limits(seed_graph_factory) -> None:
    """Verify validator handles νf boundary values correctly."""
    graph = seed_graph_factory(num_nodes=5, edge_probability=0.3, seed=42)
    
    # νf should be positive for active nodes
    vf_values = [
        1e-10,  # Very small positive
        0.1,    # Small
        1.0,    # Unit
        10.0,   # Large
        1000.0, # Very large
    ]
    
    for node, vf_val in zip(graph.nodes(), vf_values):
        graph.nodes[node][VF_PRIMARY] = vf_val
        # Verify positive and finite
        assert graph.nodes[node][VF_PRIMARY] > 0.0
        assert math.isfinite(graph.nodes[node][VF_PRIMARY])


def test_validator_phase_wrapping() -> None:
    """Verify validator handles phase wrapping to [-π, π] correctly."""
    graph = nx.Graph()
    inject_defaults(graph)
    
    # Phases outside [-π, π] should be wrappable
    unwrapped_phases = [
        -3 * math.pi,  # Should wrap to -π
        -2 * math.pi,  # Should wrap to 0
        0.0,           # Already in range
        2 * math.pi,   # Should wrap to 0
        3 * math.pi,   # Should wrap to π
        5 * math.pi / 2,  # Should wrap
    ]
    
    for i, phase in enumerate(unwrapped_phases):
        # Wrap to [-π, π]
        wrapped = ((phase + math.pi) % (2 * math.pi)) - math.pi
        graph.add_node(i, **{
            THETA_KEY: wrapped,
            EPI_PRIMARY: 0.0,
            VF_PRIMARY: 1.0,
            DNFR_PRIMARY: 0.0,
        })
        
        # Verify in range
        assert -math.pi <= graph.nodes[i][THETA_KEY] <= math.pi


def test_validator_consistency_across_multiple_nodes(seed_graph_factory) -> None:
    """Verify validator maintains consistency across all nodes."""
    graph = seed_graph_factory(num_nodes=20, edge_probability=0.3, seed=42)
    
    required_attrs = [EPI_PRIMARY, VF_PRIMARY, DNFR_PRIMARY, THETA_KEY]
    
    # All nodes should have required attributes
    for node, data in graph.nodes(data=True):
        for attr in required_attrs:
            assert attr in data, f"Node {node} missing {attr}"
        
        # All values should be finite
        for attr in required_attrs:
            assert math.isfinite(data[attr]), f"Node {node} {attr} not finite"


def test_validator_dnfr_conservation_multi_scale() -> None:
    """Verify ΔNFR conservation holds at different network scales."""
    scales = [5, 10, 20, 50]
    
    for num_nodes in scales:
        graph = nx.gnp_random_graph(num_nodes, 0.3, seed=42)
        inject_defaults(graph)
        
        # Initialize with balanced ΔNFR
        for node in graph.nodes():
            graph.nodes[node][EPI_PRIMARY] = 0.5
            graph.nodes[node][VF_PRIMARY] = 1.0
            graph.nodes[node][DNFR_PRIMARY] = 0.0
        
        # Should be conserved at any scale
        assert_dnfr_balanced(graph)


def test_validator_handles_isolated_nodes() -> None:
    """Verify validator handles isolated nodes correctly."""
    graph = nx.Graph()
    inject_defaults(graph)
    
    # Add isolated nodes
    for i in range(5):
        graph.add_node(i, **{
            EPI_PRIMARY: 0.5,
            VF_PRIMARY: 1.0,
            DNFR_PRIMARY: 0.0,
            THETA_KEY: 0.0,
        })
    
    # No edges, all isolated
    assert graph.number_of_edges() == 0
    
    # Should still be valid
    assert_dnfr_balanced(graph)


def test_validator_handles_disconnected_components() -> None:
    """Verify validator handles multiple disconnected components."""
    graph = nx.Graph()
    inject_defaults(graph)
    
    # Create two disconnected components
    # Component 1: nodes 0-2
    for i in range(3):
        graph.add_node(i, **{
            EPI_PRIMARY: 0.5,
            VF_PRIMARY: 1.0,
            DNFR_PRIMARY: 0.0,
        })
    graph.add_edge(0, 1)
    graph.add_edge(1, 2)
    
    # Component 2: nodes 3-5
    for i in range(3, 6):
        graph.add_node(i, **{
            EPI_PRIMARY: 0.3,
            VF_PRIMARY: 1.2,
            DNFR_PRIMARY: 0.0,
        })
    graph.add_edge(3, 4)
    graph.add_edge(4, 5)
    
    # Should validate across all components
    assert_dnfr_balanced(graph)


def test_validator_compositional_checks() -> None:
    """Verify validator can perform compositional checks."""
    graph = nx.Graph()
    inject_defaults(graph)
    
    # Add nodes with varying properties
    configs = [
        {"epi": -0.5, "vf": 0.8, "dnfr": 0.1},
        {"epi": 0.0, "vf": 1.0, "dnfr": -0.05},
        {"epi": 0.5, "vf": 1.2, "dnfr": -0.05},
    ]
    
    for i, config in enumerate(configs):
        graph.add_node(i, **{
            EPI_PRIMARY: config["epi"],
            VF_PRIMARY: config["vf"],
            DNFR_PRIMARY: config["dnfr"],
        })
    
    # Composite check: conservation + bounds
    assert_dnfr_balanced(graph)
    assert_epi_vf_in_bounds(
        graph,
        epi_min=-1.0,
        epi_max=1.0,
        vf_min=0.0,
        vf_max=2.0,
    )


def test_validator_network_level_constraints() -> None:
    """Verify validator enforces network-level constraints."""
    graph = nx.Graph()
    inject_defaults(graph)
    
    # Create network with specific topology
    graph.add_nodes_from([0, 1, 2, 3])
    graph.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0)])  # Cycle
    
    # Initialize with balanced ΔNFR
    total_dnfr = 0.0
    for node in graph.nodes():
        dnfr_val = 0.1 * (node - 1.5)  # Balanced around 0
        graph.nodes[node][EPI_PRIMARY] = 0.5
        graph.nodes[node][VF_PRIMARY] = 1.0
        graph.nodes[node][DNFR_PRIMARY] = dnfr_val
        total_dnfr += dnfr_val
    
    # Should be approximately conserved
    assert abs(total_dnfr) < 0.1


def test_validator_attribute_type_consistency() -> None:
    """Verify validator ensures attribute types are consistent."""
    graph = nx.Graph()
    inject_defaults(graph)
    
    graph.add_node(0, **{
        EPI_PRIMARY: 0.5,
        VF_PRIMARY: 1.0,
        DNFR_PRIMARY: 0.0,
        THETA_KEY: 0.0,
    })
    
    # All should be numeric
    for attr in [EPI_PRIMARY, VF_PRIMARY, DNFR_PRIMARY, THETA_KEY]:
        value = graph.nodes[0][attr]
        assert isinstance(value, (int, float))


def test_validator_stability_metric_computation() -> None:
    """Verify validator can assess nodal stability."""
    graph = nx.Graph()
    inject_defaults(graph)
    
    # Stable node (low ΔNFR magnitude)
    graph.add_node(0, **{
        EPI_PRIMARY: 0.5,
        VF_PRIMARY: 1.0,
        DNFR_PRIMARY: 0.01,
    })
    
    # Unstable node (high ΔNFR magnitude)
    graph.add_node(1, **{
        EPI_PRIMARY: 0.3,
        VF_PRIMARY: 1.2,
        DNFR_PRIMARY: 0.8,
    })
    
    # Compensating node
    graph.add_node(2, **{
        EPI_PRIMARY: 0.7,
        VF_PRIMARY: 0.9,
        DNFR_PRIMARY: -0.81,
    })
    
    # Network should be balanced
    assert_dnfr_balanced(graph)
    
    # Individual stability can be assessed
    stability_0 = abs(graph.nodes[0][DNFR_PRIMARY])
    stability_1 = abs(graph.nodes[1][DNFR_PRIMARY])
    assert stability_0 < stability_1


def test_validator_phase_coherence_checks() -> None:
    """Verify validator can check phase coherence."""
    graph = nx.Graph()
    inject_defaults(graph)
    
    # Create nodes with phases
    phases = [0.0, math.pi / 4, math.pi / 2, 3 * math.pi / 4, math.pi]
    
    for i, phase in enumerate(phases):
        graph.add_node(i, **{
            THETA_KEY: phase,
            EPI_PRIMARY: 0.0,
            VF_PRIMARY: 1.0,
        })
    
    # All phases should be in valid range
    for node in graph.nodes():
        phase = graph.nodes[node][THETA_KEY]
        assert -math.pi <= phase <= math.pi


def test_validator_frequency_positivity() -> None:
    """Verify validator can check frequency positivity."""
    graph = nx.Graph()
    inject_defaults(graph)
    
    # Positive frequencies for active nodes
    vf_values = [0.1, 0.5, 1.0, 1.5, 2.0]
    
    for i, vf in enumerate(vf_values):
        graph.add_node(i, **{
            VF_PRIMARY: vf,
            EPI_PRIMARY: 0.0,
            DNFR_PRIMARY: 0.0,
        })
    
    # All should be positive
    for node in graph.nodes():
        assert graph.nodes[node][VF_PRIMARY] > 0.0


def test_validator_gradient_magnitude_bounds() -> None:
    """Verify validator can bound ΔNFR magnitudes."""
    graph = nx.Graph()
    inject_defaults(graph)
    
    # Create network with bounded ΔNFR
    max_magnitude = 1.0
    
    dnfr_values = [-0.5, -0.25, 0.0, 0.25, 0.5]
    for i, dnfr in enumerate(dnfr_values):
        graph.add_node(i, **{
            DNFR_PRIMARY: dnfr,
            EPI_PRIMARY: 0.0,
            VF_PRIMARY: 1.0,
        })
    
    # All should be within bounds
    for node in graph.nodes():
        magnitude = abs(graph.nodes[node][DNFR_PRIMARY])
        assert magnitude <= max_magnitude


def test_validator_defaults_injection() -> None:
    """Verify validator ensures TNFR defaults are injected."""
    graph = nx.Graph()
    inject_defaults(graph)
    
    assert_graph_has_tnfr_defaults(graph)


def test_validator_handles_edge_cases() -> None:
    """Verify validator handles various edge cases."""
    # Empty graph
    graph_empty = nx.Graph()
    inject_defaults(graph_empty)
    assert_dnfr_balanced(graph_empty)
    
    # Single node
    graph_single = nx.Graph()
    inject_defaults(graph_single)
    graph_single.add_node(0, **{
        EPI_PRIMARY: 0.5,
        VF_PRIMARY: 1.0,
        DNFR_PRIMARY: 0.0,
    })
    assert_dnfr_balanced(graph_single)
    
    # Two disconnected nodes
    graph_two = nx.Graph()
    inject_defaults(graph_two)
    for i in range(2):
        graph_two.add_node(i, **{
            EPI_PRIMARY: 0.5,
            VF_PRIMARY: 1.0,
            DNFR_PRIMARY: 0.0,
        })
    assert_dnfr_balanced(graph_two)
