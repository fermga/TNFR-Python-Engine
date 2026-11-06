"""Compare the held graph-wave diagnostic in actual nodal coordinates."""

import math
import sys

import networkx as nx
import numpy as np
import pytest

from tnfr.physics.structural_diffusion import verify_overdamped_projection


def _direct_reference(graph, gamma, samples):
    """Independent block exponential for x''+gamma*x'+L_rw*x=0."""
    expm = pytest.importorskip("scipy.linalg").expm
    nodes = list(graph)
    adjacency = nx.to_numpy_array(graph, nodelist=nodes)
    degree = adjacency.sum(axis=1)
    walk = np.divide(
        adjacency,
        degree[:, None],
        out=np.zeros_like(adjacency),
        where=degree[:, None] > 0,
    )
    laplacian = np.diag((degree > 0).astype(float)) - walk
    n = len(nodes)
    initial = np.array([graph.nodes[node]["EPI"] for node in nodes])
    wave = np.block(
        [
            [np.zeros((n, n)), np.eye(n)],
            [-laplacian, -gamma * np.eye(n)],
        ]
    )
    positive = np.linalg.eigvals(laplacian).real
    positive = positive[positive > 1e-9]
    horizon = 3 * gamma / min(positive) if positive.size else gamma
    errors = []
    for time in np.linspace(0.05 * horizon, horizon, samples):
        actual_wave = (expm(time * wave) @ np.r_[initial, np.zeros(n)])[:n]
        diffusion = expm(-time * laplacian / gamma) @ initial
        errors.append(
            np.linalg.norm(actual_wave - diffusion)
            / (np.linalg.norm(diffusion) + 1e-12)
        )
    return max(errors)


def test_irregular_uniform_epi_is_stationary_in_both_declared_models():
    graph = nx.path_graph(3)
    nx.set_node_attributes(graph, 0.75, "EPI")

    result = verify_overdamped_projection(graph, gamma=5, n_time_samples=4)

    # L_rw*1=0; raw 1 is not the zero eigenvector of L_sym on P3.
    assert result.trajectory_max_rel_error < 1e-13


def test_weighted_irregular_nonuniform_epi_matches_raw_block_exponential():
    graph = nx.path_graph(3)
    graph[0][1]["weight"] = 1.0
    graph[1][2]["weight"] = 4.0
    nx.set_node_attributes(graph, dict(enumerate((0.25, -0.5, 1.75))), "EPI")

    result = verify_overdamped_projection(graph, gamma=5, n_time_samples=5)

    assert result.trajectory_max_rel_error == pytest.approx(
        _direct_reference(graph, gamma=5, samples=5), abs=2e-13
    )


def test_parallel_edges_loop_and_isolate_follow_adjacency_row_strengths():
    graph = nx.MultiGraph()
    graph.add_nodes_from(("left", "middle", "right", "isolated"))
    graph.add_edge("left", "middle", weight=1.0)
    graph.add_edge("left", "middle", weight=2.0)
    graph.add_edge("middle", "right", weight=4.0)
    graph.add_edge("middle", "middle", weight=2.0)
    nx.set_node_attributes(graph, dict(zip(graph, (0.25, -0.5, 1.75, 2.0))), "EPI")

    result = verify_overdamped_projection(graph, gamma=5, n_time_samples=5)

    assert result.trajectory_max_rel_error == pytest.approx(
        _direct_reference(graph, gamma=5, samples=5), abs=2e-13
    )


@pytest.mark.parametrize("weight", (sys.float_info.max, math.ulp(0.0)))
def test_common_conductance_scale_does_not_require_raw_degree_range(weight):
    graph = nx.path_graph(3)
    nx.set_node_attributes(graph, dict(enumerate((0.25, -0.5, 1.75))), "EPI")
    reference = verify_overdamped_projection(graph, gamma=5, n_time_samples=4)
    nx.set_edge_attributes(graph, weight, "weight")

    result = verify_overdamped_projection(graph, gamma=5, n_time_samples=4)

    assert result.trajectory_max_rel_error == pytest.approx(
        reference.trajectory_max_rel_error, abs=2e-13
    )


def test_zero_strength_components_keep_their_independent_stationary_values():
    graph = nx.empty_graph(3)
    nx.set_node_attributes(graph, dict(enumerate((0.25, -0.5, 1.75))), "EPI")

    result = verify_overdamped_projection(graph, gamma=5, n_time_samples=4)

    assert result.trajectory_max_rel_error == 0.0
