"""Shared constitutive flux, Dirichlet gradient and sparse read-out contracts."""

import copy

import networkx as nx
import numpy as np
import pytest

from tnfr.physics._conductance import ConductanceSnapshot, read_conductance
from tnfr.physics.structural_diffusion import (
    compute_diffusion_energy,
    current_divergence,
    degree_weighted_total,
    structural_current,
    structural_diffusion_operator,
    stationary_distribution,
    symmetric_normalized_laplacian,
)


def _fixture(kind):
    graph = kind()
    graph.add_nodes_from(["left", 2, ("right", 0), "isolated"])
    graph.add_weighted_edges_from([("left", 2, 2.0), (2, ("right", 0), 0.5), (2, 2, 3.0)])
    if graph.is_multigraph():
        graph.add_edge("left", 2, weight=0.75)
    if graph.is_directed():
        graph.add_weighted_edges_from([(2, "left", 2.0), (("right", 0), 2, 0.5)])
        if graph.is_multigraph():
            graph.add_edge(2, "left", weight=0.75)
    for node, epi, vf in zip(graph, [1.5, -0.25, 0.75, 8.0], [0.2, 1.0, 2.0, 0.8]):
        graph.nodes[node].update(EPI=epi, nu_f=vf, theta=0.0)
    return graph


@pytest.mark.parametrize("kind", [nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph])
def test_current_divergence_gradient_and_nodal_rate_share_one_conductance_convention(kind):
    graph = _fixture(kind)
    before = copy.deepcopy(graph)
    nodes, current = structural_current(graph)
    _, divergence = current_divergence(graph)
    balance = compute_diffusion_energy(graph)
    weights = nx.to_numpy_array(graph, nodelist=nodes)
    field = np.array([graph.nodes[node]["EPI"] for node in nodes])
    frequency = np.array([graph.nodes[node]["nu_f"] for node in nodes])
    laplacian = np.diag(weights.sum(axis=1)) - weights
    expected = laplacian @ field
    np.testing.assert_allclose(current, weights * (field[:, None] - field[None, :]))
    np.testing.assert_allclose(current, -current.T)
    np.testing.assert_allclose(divergence, expected)
    np.testing.assert_array_equal(balance.gradient, divergence)
    _, lap_rw = structural_diffusion_operator(graph)
    np.testing.assert_allclose(balance.epi_rate, -frequency * (lap_rw @ field))
    assert balance.energy == pytest.approx(float(0.5 * field @ expected))
    assert balance.energy_rate == pytest.approx(float(expected @ balance.epi_rate))
    assert nx.utils.graphs_equal(graph, before)


@pytest.mark.parametrize("zero_edge", [False, True])
def test_disconnected_extreme_finite_fields_have_zero_flux_without_nan(zero_edge):
    graph = nx.empty_graph(2)
    graph.nodes[0].update(EPI=1e308, nu_f=1.0)
    graph.nodes[1].update(EPI=-1e308, nu_f=1.0)
    if zero_edge:
        graph.add_edge(0, 1, weight=0.0)
    with np.errstate(all="raise"):
        np.testing.assert_array_equal(structural_current(graph)[1], np.zeros((2, 2)))
        np.testing.assert_array_equal(current_divergence(graph)[1], [0.0, 0.0])
        np.testing.assert_array_equal(compute_diffusion_energy(graph).gradient, [0.0, 0.0])


def test_zero_weight_directed_arc_and_cancelled_parallel_conductance():
    graph = nx.MultiDiGraph()
    graph.add_nodes_from([0, 1, 2])
    graph.add_edge(0, 1, weight=-2.0)
    graph.add_edge(0, 1, weight=3.0)
    graph.add_edge(1, 0, weight=1.0)
    graph.add_edge(2, 0, weight=0.0)
    nx.set_node_attributes(graph, {0: 2.0, 1: 0.0, 2: 7.0}, "EPI")
    np.testing.assert_array_equal(structural_current(graph)[1],
                                  [[0.0, 2.0, 0.0], [-2.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    np.testing.assert_array_equal(read_conductance(graph, symmetric=True).dense(),
                                  nx.to_numpy_array(graph))


@pytest.mark.parametrize("sign", [1, -1])
def test_parallel_integer_aggregation_precedes_float_conversion(sign):
    graph = nx.MultiGraph()
    graph.add_edge(0, 1, weight=sign * (2**53 + 1))
    graph.add_edge(0, 1, weight=-sign * 2**53)
    expected = nx.to_numpy_array(graph)
    assert expected[0, 1] == sign
    if sign < 0:
        with pytest.raises(ValueError, match="nonnegative"):
            read_conductance(graph)
    else:
        np.testing.assert_array_equal(read_conductance(graph).dense(), expected)


def test_constitutive_flux_does_not_require_valid_or_nonzero_capacity():
    graph = _fixture(nx.Graph)
    expected = current_divergence(graph)[1]
    nx.set_node_attributes(graph, float("nan"), "nu_f")
    np.testing.assert_array_equal(current_divergence(graph)[1], expected)
    np.testing.assert_allclose(structural_current(graph)[1].sum(axis=1), expected)
    with pytest.raises(ValueError, match="frequency"):
        compute_diffusion_energy(graph)


def test_sparse_vector_readouts_never_materialize_a_dense_matrix(monkeypatch):
    graph = nx.path_graph(5000)
    nx.set_node_attributes(graph, 1.0, "nu_f")
    nx.set_node_attributes(graph, {node: node % 2 for node in graph}, "EPI")
    monkeypatch.setattr(ConductanceSnapshot, "dense", lambda *args: pytest.fail("dense allocation"))
    _, divergence = current_divergence(graph)
    balance = compute_diffusion_energy(graph)
    np.testing.assert_array_equal(divergence, balance.gradient)
    assert balance.energy == 0.5 * (len(graph) - 1)
    assert degree_weighted_total(graph) == len(graph) - 1
    assert stationary_distribution(graph)[1].sum() == pytest.approx(1.0)


def test_induced_node_order_matches_the_matrix_api_and_rejects_invalid_lists():
    graph = _fixture(nx.MultiGraph)
    nodes = [("right", 0), 2, "left"]
    snapshot = read_conductance(graph, nodes)
    assert snapshot.nodes == nodes
    np.testing.assert_array_equal(snapshot.dense(), nx.to_numpy_array(graph, nodelist=nodes))
    reordered, lap = symmetric_normalized_laplacian(graph, nodes)
    assert reordered == nodes
    assert lap.shape == (3, 3)
    with pytest.raises(nx.NetworkXError, match="duplicates"):
        read_conductance(graph, [2, 2])
    with pytest.raises(nx.NetworkXError, match="not in"):
        read_conductance(graph, ["missing"])


def test_fresh_readouts_track_weight_and_epi_changes_without_cache_invalidation():
    graph = _fixture(nx.Graph)
    initial = compute_diffusion_energy(graph)
    graph["left"][2]["weight"] *= 3
    graph.nodes[2]["EPI"] += 0.5
    updated = compute_diffusion_energy(graph)
    assert updated.energy != initial.energy
    np.testing.assert_array_equal(updated.gradient, current_divergence(graph)[1])
    updated.gradient[:] = 0.0
    assert np.any(current_divergence(graph)[1])


@pytest.mark.parametrize("reader", [structural_current, current_divergence, compute_diffusion_energy])
def test_invalid_epi_and_overflow_cannot_produce_valid_readouts(reader):
    graph = nx.path_graph(2)
    nx.set_node_attributes(graph, 1.0, "nu_f")
    graph.nodes[0]["EPI"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        reader(graph)
    graph.nodes[0]["EPI"] = 1e308
    graph.nodes[1]["EPI"] = -1e308
    with pytest.raises(ValueError, match="floating-point range"):
        reader(graph)


def test_unrepresentable_raw_row_strength_remains_an_explicit_error():
    graph = nx.Graph()
    graph.add_weighted_edges_from([(0, 1, 1e308), (0, 2, 1e308)])
    with pytest.raises(ValueError, match="row strength"):
        read_conductance(graph).strength


@pytest.mark.parametrize("weight", [1e-320, 1.0, 1e308])
def test_stationary_measure_survives_common_conductance_scaling(weight):
    graph = nx.Graph()
    graph.add_edge(0, 1, weight=weight)
    with np.errstate(over="raise", invalid="raise"):
        np.testing.assert_array_equal(stationary_distribution(graph)[1], [0.5, 0.5])
