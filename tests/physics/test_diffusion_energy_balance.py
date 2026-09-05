"""Independent energy/pressure checks for the restricted variational bridge."""

import copy

import networkx as nx
import numpy as np
import pytest

from tnfr.alias import get_attr
from tnfr.constants.aliases import ALIAS_DNFR
from tnfr.dynamics import default_compute_delta_nfr
from tnfr.physics.structural_diffusion import compute_diffusion_energy


def _graph(kind=nx.Graph):
    graph = kind()
    graph.add_nodes_from(["a", "b", "c", "isolated"])
    graph.add_edge("a", "b", weight=2.0)
    graph.add_edge("b", "c", weight=0.5)
    graph.add_edge("b", "b", weight=3.0)
    if graph.is_multigraph():
        graph.add_edge("a", "b", weight=0.75)
    for node, epi, vf in zip(graph, [1.5, -0.25, 0.75, 8.0], [0.2, 1.0, 2.0, 0.8]):
        graph.nodes[node].update(EPI=epi, nu_f=vf, theta=0.0)
    graph.graph["DNFR_WEIGHTS"] = dict(phase=0.0, epi=1.0, vf=0.0, topo=0.0)
    return graph


def _edge_energy(graph, field):
    # Each undirected edge appears once; parallel edges contribute separately.
    return sum(0.5 * data.get("weight", 1.0) * (field[u] - field[v]) ** 2
               for u, v, data in graph.edges(data=True))


@pytest.mark.parametrize("kind", [nx.Graph, nx.MultiGraph])
def test_gradient_and_dissipation_match_independent_finite_differences(kind):
    graph = _graph(kind)
    before = copy.deepcopy(graph)
    balance = compute_diffusion_energy(graph)
    field = nx.get_node_attributes(graph, "EPI")
    epsilon = 1e-6
    numerical_gradient = []
    for node in graph:
        plus, minus = dict(field), dict(field)
        plus[node] += epsilon
        minus[node] -= epsilon
        numerical_gradient.append((_edge_energy(graph, plus) - _edge_energy(graph, minus))
                                  / (2 * epsilon))
    np.testing.assert_allclose(balance.gradient, numerical_gradient, atol=2e-9)
    assert balance.energy == pytest.approx(_edge_energy(graph, field))
    plus = {n: field[n] + epsilon * balance.epi_rate[i] for i, n in enumerate(graph)}
    minus = {n: field[n] - epsilon * balance.epi_rate[i] for i, n in enumerate(graph)}
    derivative = (_edge_energy(graph, plus) - _edge_energy(graph, minus)) / (2 * epsilon)
    assert balance.energy_rate == pytest.approx(derivative, abs=3e-9)
    assert balance.energy_rate < 0.0
    assert nx.utils.graphs_equal(graph, before)


@pytest.mark.parametrize("kind", [nx.Graph, nx.MultiGraph])
def test_mobility_flow_matches_canonical_pressure_with_loops_and_heterogeneous_capacity(kind):
    graph = _graph(kind)
    balance = compute_diffusion_energy(graph)
    default_compute_delta_nfr(graph)
    actual = [graph.nodes[n]["nu_f"] * get_attr(graph.nodes[n], ALIAS_DNFR)
              for n in balance.nodes]
    np.testing.assert_allclose(balance.epi_rate, actual, atol=1e-13)
    assert balance.mobility[-1] == balance.epi_rate[-1] == 0.0


def test_zero_capacity_freezes_nonuniform_field_without_erasing_pressure():
    graph = _graph()
    nx.set_node_attributes(graph, 0.0, "nu_f")
    balance = compute_diffusion_energy(graph)
    assert balance.energy > 0.0
    assert np.any(balance.gradient != 0.0)
    np.testing.assert_array_equal(balance.epi_rate, 0.0)
    assert balance.energy_rate == 0.0


def test_empty_and_uniform_fields_have_zero_energy_and_flow():
    for graph in [nx.Graph(), _graph()]:
        nx.set_node_attributes(graph, 1e12, "EPI")
        balance = compute_diffusion_energy(graph)
        assert balance.energy == balance.energy_rate == 0.0
        np.testing.assert_array_equal(balance.gradient, np.zeros(len(graph)))
        np.testing.assert_array_equal(balance.epi_rate, np.zeros(len(graph)))


def test_large_common_offset_preserves_energy():
    graph = _graph()
    original = compute_diffusion_energy(graph)
    for node in graph:
        graph.nodes[node]["EPI"] += 1e12
    shifted = compute_diffusion_energy(graph)
    assert shifted.energy == original.energy
    np.testing.assert_array_equal(shifted.gradient, original.gradient)


def test_extreme_disconnected_fields_do_not_form_spurious_pairwise_differences():
    graph = nx.empty_graph(2)
    graph.nodes[0].update(EPI=1e308, nu_f=1.0)
    graph.nodes[1].update(EPI=-1e308, nu_f=1.0)
    with np.errstate(all="raise"):
        balance = compute_diffusion_energy(graph)
    assert balance.energy == balance.energy_rate == 0.0
    np.testing.assert_array_equal(balance.epi_rate, [0.0, 0.0])


def test_unrepresentable_connected_balance_raises_instead_of_returning_nan():
    graph = nx.path_graph(2)
    graph.nodes[0].update(EPI=1e308, nu_f=1.0)
    graph.nodes[1].update(EPI=-1e308, nu_f=1.0)
    with pytest.raises(ValueError, match="floating-point range"):
        compute_diffusion_energy(graph)


def test_positive_mobility_underflow_cannot_falsely_certify_stationarity():
    graph = nx.path_graph(2)
    graph[0][1]["weight"] = 1e200
    graph.nodes[0].update(EPI=1.0, nu_f=1e-200)
    graph.nodes[1].update(EPI=0.0, nu_f=1e-200)
    # The actual generator has representable nonzero rate and dissipation -2,
    # but nu_f/degree is 1e-400 and cannot be returned as a float mobility.
    with pytest.raises(ValueError, match="mobility.*floating-point range"):
        compute_diffusion_energy(graph)


@pytest.mark.parametrize("attribute,value", [("EPI", float("nan")), ("EPI", float("inf")),
                                           ("nu_f", -1.0), ("nu_f", float("inf"))])
def test_invalid_nodal_state_is_rejected(attribute, value):
    graph = _graph()
    graph.nodes["a"][attribute] = value
    with pytest.raises(ValueError):
        compute_diffusion_energy(graph)


@pytest.mark.parametrize("weight", [-1.0, float("nan"), float("inf")])
def test_invalid_conductance_is_rejected(weight):
    graph = _graph()
    graph["a"]["b"]["weight"] = weight
    with pytest.raises(ValueError, match="nonnegative"):
        compute_diffusion_energy(graph)


def test_asymmetric_adjacency_rejected_and_symmetric_directed_representation_agrees():
    graph = _graph()
    directed = graph.to_directed()
    assert compute_diffusion_energy(directed).energy == compute_diffusion_energy(graph).energy
    directed.remove_edge("a", "b")
    with pytest.raises(ValueError, match="symmetric"):
        compute_diffusion_energy(directed)


def test_dirichlet_energy_is_distinct_from_tetrad_potential_one_edge_counterexample():
    from tnfr.physics.variational import compute_potential_density

    graph = nx.path_graph(2)
    for node in graph:
        graph.nodes[node].update(EPI=1.0 - node, nu_f=1.0, theta=0.0)
    graph.graph["DNFR_WEIGHTS"] = dict(phase=0.0, epi=1.0, vf=0.0, topo=0.0)
    default_compute_delta_nfr(graph)
    balance = compute_diffusion_energy(graph)
    assert balance.energy == 0.5
    assert sum(compute_potential_density(graph).values()) == 1.0
    np.testing.assert_allclose(balance.epi_rate, [-1.0, 1.0])
