"""Unified structural validation tests using parametrized fixtures.

This module consolidates redundant structural validation tests from:
- integration/test_consolidated_structural_validation.py
- integration/test_nodal_validators.py
- property/test_dnfr_properties.py
- stress/test_dnfr_runtime.py

By using parametrized fixtures, we reduce redundancy while maintaining
comprehensive coverage across different network scales and configurations.
"""

from __future__ import annotations

import copy
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
from tnfr.dynamics import dnfr_epi_vf_mixed, dnfr_phase_only
from tests.helpers.validation import (
    assert_dnfr_balanced,
    assert_dnfr_homogeneous_stable,
    assert_dnfr_lists_close,
    get_dnfr_values,
)
from tests.helpers.fixtures import (  # noqa: F401
    seed_graph_factory,  # noqa: F401
    homogeneous_graph_factory,  # noqa: F401
    bicluster_graph_factory,  # noqa: F401
)


# Parametrized network scales for consolidated testing
@pytest.fixture(
    params=[
        {"num_nodes": 5, "edge_probability": 0.3, "seed": 42},
        {"num_nodes": 10, "edge_probability": 0.3, "seed": 100},
        {"num_nodes": 50, "edge_probability": 0.2, "seed": 123},
    ]
)
def parametrized_network_scale(request):
    """Parametrized network scales consolidating small/medium/large tests.

    Uses different seeds for each scale to ensure independent test coverage.
    """
    return request.param


# Parametrized homogeneous configurations
@pytest.fixture(
    params=[
        {"epi_value": 0.0, "vf_value": 1.0},
        {"epi_value": 0.5, "vf_value": 1.5},
        {"epi_value": -0.3, "vf_value": 0.8},
    ]
)
def parametrized_homogeneous_config(request):
    """Parametrized homogeneous configurations consolidating multiple config tests."""
    return request.param


def test_dnfr_conservation_unified(
    seed_graph_factory, parametrized_network_scale
) -> None:
    """Unified ΔNFR conservation test consolidating multiple scale tests.

    Consolidates:
    - test_dnfr_conservation_small_network
    - test_dnfr_conservation_medium_network
    - test_validator_dnfr_conservation_multi_scale
    """
    config = parametrized_network_scale
    graph = seed_graph_factory(
        num_nodes=config["num_nodes"],
        edge_probability=config["edge_probability"],
        seed=config["seed"],
    )

    dnfr_epi_vf_mixed(graph)
    assert_dnfr_balanced(graph, abs_tol=0.1)


def test_dnfr_homogeneous_stability_unified(
    homogeneous_graph_factory,
    parametrized_homogeneous_config,
) -> None:
    """Unified homogeneous stability test consolidating multiple configuration tests.

    Consolidates:
    - test_dnfr_homogeneous_stability
    - test_dnfr_homogeneous_multiple_configurations
    - test_dnfr_epi_vf_mixed_stable_on_homogeneous (property)
    """
    config = parametrized_homogeneous_config
    graph = homogeneous_graph_factory(
        num_nodes=8,
        edge_probability=0.4,
        seed=42,
        **config,
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
    """Verify synchronized phases remain stable."""
    graph = nx.gnp_random_graph(6, 0.4, seed=42)
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

    assert_dnfr_lists_close(base_values, permuted_values)


def test_dnfr_computation_consistency(seed_graph_factory) -> None:
    """Verify ΔNFR computation is deterministic and consistent."""
    graph1 = seed_graph_factory(num_nodes=12, edge_probability=0.25, seed=555)
    graph2 = seed_graph_factory(num_nodes=12, edge_probability=0.25, seed=555)

    dnfr_epi_vf_mixed(graph1)
    dnfr_epi_vf_mixed(graph2)

    values1 = get_dnfr_values(graph1)
    values2 = get_dnfr_values(graph2)

    assert_dnfr_lists_close(values1, values2)


def test_dnfr_phase_rotation_invariance() -> None:
    """Verify phase-only ΔNFR is invariant under global rotation."""
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

    assert_dnfr_lists_close(values1, values2)


@pytest.mark.parametrize("edge_prob", [0.1, 0.3, 0.5, 0.7])
def test_structural_conservation_across_topologies(
    seed_graph_factory, edge_prob
) -> None:
    """Verify ΔNFR conservation holds across different network topologies.

    Consolidates topology variation tests using parametrization.
    """
    graph = seed_graph_factory(num_nodes=10, edge_probability=edge_prob, seed=999)
    dnfr_epi_vf_mixed(graph)
    assert_dnfr_balanced(graph, abs_tol=0.1)


@pytest.mark.parametrize("num_nodes", [5, 10, 20, 50])
def test_dnfr_conservation_scale_independence(num_nodes) -> None:
    """Verify ΔNFR conservation is scale-independent.

    Consolidates multi-scale validation tests.
    """
    graph = nx.gnp_random_graph(num_nodes, 0.3, seed=42)
    inject_defaults(graph)

    # Initialize with balanced ΔNFR
    for node in graph.nodes():
        graph.nodes[node][EPI_PRIMARY] = 0.5
        graph.nodes[node][VF_PRIMARY] = 1.0
        graph.nodes[node][DNFR_PRIMARY] = 0.0

    dnfr_epi_vf_mixed(graph)
    assert_dnfr_balanced(graph, abs_tol=0.1)


@pytest.mark.parametrize("phase", [0.0, math.pi / 4, math.pi / 2, math.pi])
def test_dnfr_phase_synchronization_at_different_phases(phase) -> None:
    """Verify synchronization stability at different phase values.

    Consolidates phase synchronization tests.
    """
    graph = nx.gnp_random_graph(6, 0.4, seed=42)
    inject_defaults(graph)

    # All nodes at same phase
    for _, data in graph.nodes(data=True):
        data[THETA_KEY] = phase
        data[DNFR_PRIMARY] = 0.0

    dnfr_phase_only(graph)
    assert_dnfr_homogeneous_stable(graph)
