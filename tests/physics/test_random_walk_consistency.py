"""Analytic transport checks for weighted, disconnected and absorbing walks."""

import networkx as nx
import numpy as np
import pytest

from tnfr.constants.aliases import ALIAS_EPI
from tnfr.physics.structural_diffusion import (
    commute_time,
    current_divergence,
    effective_resistance,
    random_walk_matrix,
    stationary_distribution,
    structural_current,
    structural_diffusion_operator,
    verify_structural_flow,
    verify_structural_random_walk,
)


@pytest.mark.parametrize("graph_type", [nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph])
def test_zero_strength_nodes_are_absorbing(graph_type):
    graph = graph_type()
    graph.add_nodes_from(range(3))
    graph.add_edge(0, 1, weight=2.0)
    graph.add_edge(1, 2, weight=0.0)
    nodes, transition = random_walk_matrix(graph)
    _, laplacian = structural_diffusion_operator(graph)

    np.testing.assert_allclose(transition.sum(axis=1), 1.0)
    np.testing.assert_allclose(laplacian, np.eye(len(nodes)) - transition)
    np.testing.assert_array_equal(transition[2], [0.0, 0.0, 1.0])
    if graph.is_directed():
        np.testing.assert_array_equal(transition[1], [0.0, 1.0, 0.0])


def test_parallel_conductances_match_the_collapsed_weighted_network():
    graph = nx.MultiGraph()
    graph.add_weighted_edges_from([(0, 1, 2.0), (0, 1, 3.0), (0, 2, 1.0)])
    _, transition = random_walk_matrix(graph)
    _, resistance = effective_resistance(graph)
    _, commute = commute_time(graph)

    np.testing.assert_allclose(transition[0], [0.0, 5.0 / 6.0, 1.0 / 6.0])
    expected_resistance = np.array(
        [[0.0, 0.2, 1.0], [0.2, 0.0, 1.2], [1.0, 1.2, 0.0]]
    )
    np.testing.assert_allclose(resistance, expected_resistance, atol=1e-12)
    np.testing.assert_allclose(commute, 12.0 * expected_resistance, atol=1e-12)
    assert verify_structural_random_walk(graph).is_valid_random_walk


def test_disconnected_resistance_and_commute_use_each_component():
    graph = nx.Graph()
    graph.add_nodes_from(range(5))
    graph.add_weighted_edges_from([(0, 1, 2.0), (2, 3, 4.0), (1, 2, 0.0)])
    _, resistance = effective_resistance(graph)
    _, commute = commute_time(graph)

    assert resistance[0, 1] == pytest.approx(0.5)
    assert resistance[2, 3] == pytest.approx(0.25)
    # Each two-node walk needs one step out and one back, regardless of
    # conductance or the volume of an unreachable component.
    assert commute[0, 1] == pytest.approx(2.0)
    assert commute[2, 3] == pytest.approx(2.0)
    for left, right in [(0, 2), (0, 4), (3, 4)]:
        assert np.isinf(resistance[left, right])
        assert np.isinf(commute[left, right])
    np.testing.assert_array_equal(np.diag(commute), 0.0)
    assert verify_structural_random_walk(graph).is_valid_random_walk


@pytest.mark.parametrize("size", [0, 1, 3])
def test_edgeless_walk_has_normalized_stationarity_and_infinite_cross_transport(size):
    graph = nx.empty_graph(size)
    _, transition = random_walk_matrix(graph)
    _, stationary = stationary_distribution(graph)
    _, resistance = effective_resistance(graph)
    _, commute = commute_time(graph)

    np.testing.assert_array_equal(transition, np.eye(size))
    np.testing.assert_allclose(stationary, np.full(size, 1.0 / size) if size else [])
    expected = np.full((size, size), np.inf)
    np.fill_diagonal(expected, 0.0)
    np.testing.assert_array_equal(resistance, expected)
    np.testing.assert_array_equal(commute, expected)
    assert verify_structural_random_walk(graph).is_valid_random_walk


def test_self_loop_holding_time_changes_commute_but_not_resistance():
    graph = nx.Graph()
    graph.add_weighted_edges_from([(0, 1, 2.0), (0, 0, 6.0)])
    _, transition = random_walk_matrix(graph)
    _, stationary = stationary_distribution(graph)
    _, resistance = effective_resistance(graph)
    _, commute = commute_time(graph)

    np.testing.assert_allclose(transition, [[0.75, 0.25], [1.0, 0.0]])
    np.testing.assert_allclose(stationary, [0.8, 0.2])
    assert resistance[0, 1] == pytest.approx(0.5)
    # H(0,1)=1/P(0,1)=4, H(1,0)=1; the diagonal weight is counted once.
    assert commute[0, 1] == pytest.approx(5.0)
    assert verify_structural_random_walk(graph).is_valid_random_walk


@pytest.mark.parametrize(
    "function",
    [stationary_distribution, effective_resistance, commute_time, verify_structural_random_walk],
)
def test_undirected_transport_formulas_reject_asymmetric_adjacency(function):
    graph = nx.DiGraph([(0, 1), (1, 2), (2, 0)])
    with pytest.raises(ValueError, match="symmetric"):
        function(graph)


def test_symmetric_directed_adjacency_retains_undirected_transport():
    graph = nx.path_graph(3).to_directed()
    _, resistance = effective_resistance(graph)
    _, stationary = stationary_distribution(graph)
    np.testing.assert_allclose(resistance, [[0, 1, 2], [1, 0, 1], [2, 1, 0]], atol=1e-12)
    np.testing.assert_allclose(stationary, [0.25, 0.5, 0.25])
    assert verify_structural_random_walk(graph).is_valid_random_walk


def test_weighted_commute_matches_independent_first_step_hitting_equations():
    graph = nx.Graph()
    graph.add_weighted_edges_from(
        [(0, 1, 2.0), (1, 2, 3.0), (2, 3, 4.0), (3, 0, 1.0), (1, 1, 5.0)]
    )
    adjacency = nx.to_numpy_array(graph, weight="weight")
    transition = adjacency / adjacency.sum(axis=1, keepdims=True)
    hitting = np.zeros((4, 4))
    for target in range(4):
        others = np.asarray([node for node in range(4) if node != target])
        # H(i,t) = 1 + sum_j P(i,j) H(j,t), with H(t,t) = 0.
        hitting[others, target] = np.linalg.solve(
            np.eye(3) - transition[np.ix_(others, others)], np.ones(3)
        )
    _, commute = commute_time(graph)
    np.testing.assert_allclose(commute, hitting + hitting.T, atol=1e-10)


def test_unreachable_conductance_scale_does_not_change_local_resistance():
    graph = nx.Graph()
    graph.add_weighted_edges_from([(0, 1, 1e-20), (2, 3, 1e20)])
    _, resistance = effective_resistance(graph)
    _, commute = commute_time(graph)
    assert resistance[0, 1] == pytest.approx(1e20)
    assert resistance[2, 3] == pytest.approx(1e-20, rel=1e-12, abs=0.0)
    assert commute[0, 1] == pytest.approx(2.0)
    assert commute[2, 3] == pytest.approx(2.0)


@pytest.mark.parametrize("tolerance", [0.0, -1.0, np.nan, np.inf])
@pytest.mark.parametrize("function", [verify_structural_random_walk, verify_structural_flow])
def test_certificate_requires_finite_positive_tolerance(tolerance, function):
    with pytest.raises(ValueError, match="tolerance"):
        function(nx.path_graph(2), tolerance=tolerance)


@pytest.mark.parametrize("graph_type", [nx.Graph, nx.MultiGraph])
def test_weighted_current_preserves_kirchhoff_and_nodal_continuity(graph_type):
    graph = graph_type()
    graph.add_edge(0, 1, weight=2.0)
    if graph.is_multigraph():
        graph.add_edge(0, 1, weight=3.0)
    graph.nodes[0].update(EPI=3.0, nu_f=1.0)
    graph.nodes[1][ALIAS_EPI[-1]] = 1.0
    graph.nodes[1]["nu_f"] = 4.0
    conductance = 5.0 if graph.is_multigraph() else 2.0
    _, current = structural_current(graph)
    _, divergence = current_divergence(graph)
    _, laplacian = structural_diffusion_operator(graph)
    np.testing.assert_allclose(current, [[0.0, 2.0 * conductance], [-2.0 * conductance, 0.0]])
    np.testing.assert_allclose(divergence, [2.0 * conductance, -2.0 * conductance])
    frequency = np.array([1.0, 4.0])
    derivative = -frequency * (laplacian @ np.array([3.0, 1.0]))
    np.testing.assert_allclose(conductance / frequency * derivative + divergence, 0.0)
    assert verify_structural_flow(graph).is_valid_flow


@pytest.mark.parametrize("function", [structural_current, current_divergence, verify_structural_flow])
def test_antisymmetric_current_requires_symmetric_conductance(function):
    graph = nx.DiGraph([(0, 1)])
    with pytest.raises(ValueError, match="symmetric"):
        function(graph)


def test_ohm_certificate_checks_the_injected_current_equation(monkeypatch):
    # A broken inverse makes both its entries and its computed voltage zero;
    # comparing those alone would falsely certify Ohm's law.
    monkeypatch.setattr(np.linalg, "pinv", lambda matrix, **kwargs: np.zeros_like(matrix))
    certificate = verify_structural_flow(nx.path_graph(2))
    assert not certificate.ohm_law_holds
    assert not certificate.is_valid_flow


def test_flow_certificate_handles_reachable_pairs_in_disconnected_components():
    graph = nx.Graph()
    graph.add_weighted_edges_from([(0, 1, 2.0), (2, 3, 4.0)])
    graph.add_node(4)
    for node in graph:
        graph.nodes[node]["EPI"] = float(node)
    assert verify_structural_flow(graph).is_valid_flow
