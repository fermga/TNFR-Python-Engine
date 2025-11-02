"""Consolidated structural validation tests using shared utilities.

This module replaces redundant test patterns across integration, mathematics,
property, and stress test suites by using shared validation helpers.
Following DRY principles while maintaining TNFR structural fidelity.
"""

from __future__ import annotations

import math

import networkx as nx
import pytest

from tnfr.constants import DNFR_PRIMARY, EPI_PRIMARY, THETA_KEY, VF_PRIMARY
from tnfr.dynamics import dnfr_epi_vf_mixed, dnfr_phase_only
from tests.helpers.validation import (
    assert_dnfr_balanced,
    assert_dnfr_homogeneous_stable,
    get_dnfr_values,
)
from tests.helpers.fixtures import (
    seed_graph_factory,
    homogeneous_graph_factory,
    bicluster_graph_factory,
)


def test_dnfr_conservation_small_network(seed_graph_factory) -> None:
    """Verify ΔNFR conservation on small deterministic network."""
    graph = seed_graph_factory(num_nodes=10, edge_probability=0.3, seed=42)
    dnfr_epi_vf_mixed(graph)
    assert_dnfr_balanced(graph, abs_tol=0.1)


def test_dnfr_conservation_medium_network(seed_graph_factory) -> None:
    """Verify ΔNFR conservation on medium-sized network."""
    graph = seed_graph_factory(num_nodes=50, edge_probability=0.2, seed=123)
    dnfr_epi_vf_mixed(graph)
    assert_dnfr_balanced(graph, abs_tol=0.1)


def test_dnfr_homogeneous_stability(homogeneous_graph_factory) -> None:
    """Verify homogeneous graphs remain stable (consolidated from property tests)."""
    graph = homogeneous_graph_factory(
        num_nodes=8, edge_probability=0.4, seed=42, epi_value=0.5, vf_value=1.0
    )
    dnfr_epi_vf_mixed(graph)
    assert_dnfr_homogeneous_stable(graph)


def test_dnfr_homogeneous_multiple_configurations(homogeneous_graph_factory) -> None:
    """Verify stability holds across different homogeneous configurations."""
    configs = [
        {"epi_value": 0.0, "vf_value": 1.0},
        {"epi_value": 0.5, "vf_value": 1.5},
        {"epi_value": -0.3, "vf_value": 0.8},
    ]

    for i, config in enumerate(configs):
        graph = homogeneous_graph_factory(
            num_nodes=6, edge_probability=0.5, seed=100 + i, **config
        )
        dnfr_epi_vf_mixed(graph)
        assert_dnfr_homogeneous_stable(graph)


def test_dnfr_bicluster_gradient(bicluster_graph_factory) -> None:
    """Verify ΔNFR creates gradients between contrasting clusters."""
    graph, (left_nodes, right_nodes) = bicluster_graph_factory(
        cluster_size=4,
        epi_left=0.0,
        epi_right=1.0,
        vf_left=1.0,
        vf_right=1.0,
    )

    dnfr_epi_vf_mixed(graph)

    # Extract ΔNFR from each cluster
    left_dnfr = [float(graph.nodes[n][DNFR_PRIMARY]) for n in left_nodes]
    right_dnfr = [float(graph.nodes[n][DNFR_PRIMARY]) for n in right_nodes]

    # Clusters should have opposite ΔNFR signs due to gradient
    left_sign = math.copysign(1, sum(left_dnfr))
    right_sign = math.copysign(1, sum(right_dnfr))
    assert left_sign * right_sign <= 0.0

    # Total should still be conserved
    assert_dnfr_balanced(graph)


def test_dnfr_phase_only_synchronization() -> None:
    """Verify synchronized phases remain stable (consolidated from property tests)."""
    graph = nx.gnp_random_graph(6, 0.4, seed=42)

    from tnfr.constants import inject_defaults

    inject_defaults(graph)

    # All nodes at same phase
    sync_phase = math.pi / 4
    for _, data in graph.nodes(data=True):
        data[THETA_KEY] = sync_phase
        data[DNFR_PRIMARY] = 0.0

    dnfr_phase_only(graph)
    assert_dnfr_homogeneous_stable(graph)


def test_dnfr_invariance_under_relabeling(seed_graph_factory) -> None:
    """Verify ΔNFR values are invariant under node relabeling."""
    import copy

    base_graph = seed_graph_factory(num_nodes=8, edge_probability=0.3, seed=789)
    permuted_graph = copy.deepcopy(base_graph)

    # Create new labels that don't overlap
    nodes = list(base_graph.nodes())
    new_labels = [f"n{i}" for i in range(len(nodes))]
    mapping = dict(zip(nodes, new_labels))
    nx.relabel_nodes(permuted_graph, mapping, copy=False)

    # Compute ΔNFR on both
    dnfr_epi_vf_mixed(base_graph)
    dnfr_epi_vf_mixed(permuted_graph)

    # Sorted ΔNFR values should match
    base_values = get_dnfr_values(base_graph)
    permuted_values = get_dnfr_values(permuted_graph)

    from tests.helpers.validation import assert_dnfr_lists_close

    assert_dnfr_lists_close(base_values, permuted_values)


def test_dnfr_computation_consistency(seed_graph_factory) -> None:
    """Verify ΔNFR computation is deterministic and consistent."""
    graph1 = seed_graph_factory(num_nodes=12, edge_probability=0.25, seed=555)
    graph2 = seed_graph_factory(num_nodes=12, edge_probability=0.25, seed=555)

    dnfr_epi_vf_mixed(graph1)
    dnfr_epi_vf_mixed(graph2)

    values1 = get_dnfr_values(graph1)
    values2 = get_dnfr_values(graph2)

    from tests.helpers.validation import assert_dnfr_lists_close

    assert_dnfr_lists_close(values1, values2)


def test_dnfr_phase_rotation_invariance() -> None:
    """Verify phase-only ΔNFR is invariant under global rotation."""
    import copy

    from tnfr.constants import inject_defaults

    graph1 = nx.gnp_random_graph(5, 0.5, seed=111)
    inject_defaults(graph1)

    # Set initial phases
    base_phases = [0.0, math.pi / 6, math.pi / 3, math.pi / 2, 2 * math.pi / 3]
    for node, phase in zip(graph1.nodes(), base_phases):
        graph1.nodes[node][THETA_KEY] = phase
        graph1.nodes[node][DNFR_PRIMARY] = 0.0

    # Create rotated copy
    graph2 = copy.deepcopy(graph1)
    rotation = math.pi / 4
    for node in graph2.nodes():
        graph2.nodes[node][THETA_KEY] += rotation

    # Compute ΔNFR
    dnfr_phase_only(graph1)
    dnfr_phase_only(graph2)

    # Magnitude distribution should be the same
    values1 = get_dnfr_values(graph1)
    values2 = get_dnfr_values(graph2)

    from tests.helpers.validation import assert_dnfr_lists_close

    assert_dnfr_lists_close(values1, values2)


def test_structural_conservation_across_topologies(seed_graph_factory) -> None:
    """Verify ΔNFR conservation holds across different network topologies."""
    # Test various edge probabilities
    probabilities = [0.1, 0.3, 0.5, 0.7]

    for prob in probabilities:
        graph = seed_graph_factory(num_nodes=10, edge_probability=prob, seed=999)
        dnfr_epi_vf_mixed(graph)
        assert_dnfr_balanced(graph, abs_tol=0.1)
