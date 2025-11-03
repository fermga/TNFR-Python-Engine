"""Property-based checks for ΔNFR hooks and coherence metrics."""

from __future__ import annotations

import copy
import math
from typing import Iterable

import networkx as nx
import pytest

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import given, strategies as st

from tnfr.constants import DNFR_PRIMARY, EPI_PRIMARY, THETA_KEY, VF_PRIMARY, dEPI_PRIMARY
from tnfr.dynamics import dnfr_epi_vf_mixed, dnfr_phase_only
from tnfr.metrics.common import compute_coherence
from tests.helpers.validation import (
    assert_dnfr_balanced,
    assert_dnfr_homogeneous_stable,
    assert_dnfr_lists_close,
    get_dnfr_values,
)

from .strategies import (
    PROPERTY_TEST_SETTINGS,
    ClusteredGraph,
    PhaseGraph,
    homogeneous_graphs,
    phase_graphs,
    prepare_network,
    two_cluster_graphs,
)


def _expected_dnfr_mixed(graph, node) -> float:
    """Return the analytic ΔNFR expected from ``dnfr_epi_vf_mixed``."""

    neighbors = list(graph.neighbors(node))
    if not neighbors:
        return 0.0
    epi_val = float(graph.nodes[node][EPI_PRIMARY])
    vf_val = float(graph.nodes[node][VF_PRIMARY])
    epi_avg = sum(
        float(graph.nodes[neigh][EPI_PRIMARY]) for neigh in neighbors
    ) / len(neighbors)
    vf_avg = sum(
        float(graph.nodes[neigh][VF_PRIMARY]) for neigh in neighbors
    ) / len(neighbors)
    return 0.5 * (epi_avg - epi_val) + 0.5 * (vf_avg - vf_val)


@PROPERTY_TEST_SETTINGS
@given(graph=homogeneous_graphs())
def test_dnfr_epi_vf_mixed_stable_on_homogeneous(graph) -> None:
    """Homogeneous graphs should remain balanced after ΔNFR evaluation."""

    dnfr_epi_vf_mixed(graph)

    # Use shared validation helper for consistency with integration tests
    assert_dnfr_homogeneous_stable(graph)


@PROPERTY_TEST_SETTINGS
@given(clustered=two_cluster_graphs())
def test_dnfr_epi_vf_mixed_balances_clusters(clustered: ClusteredGraph) -> None:
    """ΔNFR should align with the analytic neighbour gradient in bi-clusters."""

    graph = clustered.graph
    dnfr_epi_vf_mixed(graph)

    total_dnfr = 0.0
    cluster_signatures: list[float] = []
    for cluster in clustered.clusters:
        cluster_values: list[float] = []
        for node in cluster:
            actual = float(graph.nodes[node][DNFR_PRIMARY])
            expected = _expected_dnfr_mixed(graph, node)
            assert math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9)
            cluster_values.append(actual)
            total_dnfr += actual
        if cluster_values:
            cluster_signatures.append(cluster_values[0])

    # Use shared validation helper for conservation check
    assert_dnfr_balanced(graph, abs_tol=1e-9)
    
    if len(cluster_signatures) == 2:
        # Cluster signatures should oppose or cancel depending on the gradient.
        assert cluster_signatures[0] * cluster_signatures[1] <= 0.0


@PROPERTY_TEST_SETTINGS
@given(
    data=st.data(),
    graph=prepare_network(min_nodes=2, connected=True),
)
def test_dnfr_epi_vf_mixed_invariant_under_relabel(data, graph) -> None:
    """ΔNFR values should remain stable after node relabelling."""

    base_graph = copy.deepcopy(graph)

    # Create permutation and apply using two-step relabeling to avoid overlapping labels
    nodes = list(base_graph.nodes())
    permutation = data.draw(
        st.permutations(nodes),
        label="node_permutation",
    )
    
    # Step 1: Relabel to temporary non-overlapping labels
    temp_labels = [f"__tmp_{i}" for i in range(len(nodes))]
    temp_mapping = dict(zip(nodes, temp_labels))
    permuted_graph = nx.relabel_nodes(base_graph, temp_mapping, copy=True)
    
    # Step 2: Apply the actual permutation from temp labels to final labels
    final_mapping = dict(zip(temp_labels, permutation))
    nx.relabel_nodes(permuted_graph, final_mapping, copy=False)

    # Ensure DNFR initialized before running dynamics
    for _node, data_dict in base_graph.nodes(data=True):
        for key in (EPI_PRIMARY, VF_PRIMARY):
            assert key in data_dict
        if DNFR_PRIMARY not in data_dict:
            data_dict[DNFR_PRIMARY] = 0.0
    for _node, data_dict in permuted_graph.nodes(data=True):
        if DNFR_PRIMARY not in data_dict:
            data_dict[DNFR_PRIMARY] = 0.0

    dnfr_epi_vf_mixed(base_graph)
    dnfr_epi_vf_mixed(permuted_graph)

    # Use shared validation helpers for consistency
    base_values = get_dnfr_values(base_graph)
    permuted_values = get_dnfr_values(permuted_graph)
    assert_dnfr_lists_close(base_values, permuted_values)


def _apply_noise(
    base_graph,
    noise_scale: float,
    noise_pairs: Iterable[tuple[float, float]],
):
    graph = copy.deepcopy(base_graph)
    for (node, data), (noise_dnfr, noise_depi) in zip(
        graph.nodes(data=True), noise_pairs
    ):
        data[DNFR_PRIMARY] = float(data.get(DNFR_PRIMARY, 0.0)) + noise_scale * noise_dnfr
        data[dEPI_PRIMARY] = float(data.get(dEPI_PRIMARY, 0.0)) + noise_scale * noise_depi
    return graph


@PROPERTY_TEST_SETTINGS
@given(
    data=st.data(),
    graph=prepare_network(min_nodes=2, max_nodes=6, connected=True),
)
def test_compute_coherence_decreases_with_noise(data, graph) -> None:
    """Coherence should not improve after injecting ΔNFR/dEPI noise."""

    base_coherence, base_dnfr, base_depi = compute_coherence(graph, return_means=True)

    noise_pair = data.draw(
        st.tuples(
            st.floats(min_value=0.0, max_value=0.5, allow_nan=False, allow_infinity=False),
            st.floats(min_value=0.0, max_value=0.5, allow_nan=False, allow_infinity=False),
        ),
        label="noise_pair",
    )
    small_scale, large_scale = sorted(noise_pair)

    count = graph.number_of_nodes()
    noise_vectors = data.draw(
        st.lists(
            st.tuples(
                st.floats(
                    min_value=-1.0,
                    max_value=1.0,
                    allow_nan=False,
                    allow_infinity=False,
                ),
                st.floats(
                    min_value=-1.0,
                    max_value=1.0,
                    allow_nan=False,
                    allow_infinity=False,
                ),
            ),
            min_size=count,
            max_size=count,
        ),
        label="noise_vectors",
    )

    small_graph = _apply_noise(graph, small_scale, noise_vectors)
    large_graph = _apply_noise(graph, large_scale, noise_vectors)

    small_coherence, small_dnfr, small_depi = compute_coherence(
        small_graph, return_means=True
    )
    large_coherence, large_dnfr, large_depi = compute_coherence(
        large_graph, return_means=True
    )

    tol = 1e-9
    assert small_dnfr + tol >= base_dnfr
    assert small_depi + tol >= base_depi
    assert large_dnfr + tol >= small_dnfr
    assert large_depi + tol >= small_depi
    assert base_coherence + tol >= small_coherence
    assert small_coherence + tol >= large_coherence


@PROPERTY_TEST_SETTINGS
@given(
    graph=prepare_network(min_nodes=2, max_nodes=8, connected=True),
    phase=st.floats(
        min_value=-4.0 * math.pi,
        max_value=4.0 * math.pi,
        allow_nan=False,
        allow_infinity=False,
    ),
)
def test_dnfr_phase_only_stable_on_synchronised(graph, phase) -> None:
    """Synchronised phases should remain balanced under ΔNFR."""

    for _node, data in graph.nodes(data=True):
        data[THETA_KEY] = phase
        data[DNFR_PRIMARY] = 0.0

    dnfr_phase_only(graph)

    # Use shared validation helper for consistency
    assert_dnfr_homogeneous_stable(graph)


@PROPERTY_TEST_SETTINGS
@given(data=st.data())
def test_dnfr_phase_only_rotation_invariant(data) -> None:
    """Global phase rotations should not affect ΔNFR magnitudes."""

    phased: PhaseGraph = data.draw(phase_graphs(), label="phase_graph")
    rotation = data.draw(
        st.floats(
            min_value=-4.0 * math.pi,
            max_value=4.0 * math.pi,
            allow_nan=False,
            allow_infinity=False,
        ),
        label="rotation_angle",
    )

    base_graph = copy.deepcopy(phased.graph)
    rotated_graph = copy.deepcopy(phased.graph)
    base_phases = dict(phased.base_phases)

    for node in base_graph.nodes:
        base_value = base_phases[node] + phased.offset
        base_graph.nodes[node][THETA_KEY] = base_value
        base_graph.nodes[node][DNFR_PRIMARY] = 0.0
        rotated_graph.nodes[node][THETA_KEY] = base_value + rotation
        rotated_graph.nodes[node][DNFR_PRIMARY] = 0.0

    dnfr_phase_only(base_graph)
    dnfr_phase_only(rotated_graph)

    # Use shared validation helpers for consistency
    base_values = get_dnfr_values(base_graph)
    rotated_values = get_dnfr_values(rotated_graph)
    assert_dnfr_lists_close(base_values, rotated_values)
