"""Declared geometry work is separate from EPI transport dissipation."""

from copy import deepcopy
from dataclasses import replace
from fractions import Fraction as Q

import networkx as nx
import pytest

from tnfr.physics.support_transport import (
    observe_support_transport,
    observe_support_transport_derivative,
)


def _graph():
    graph = nx.path_graph(3)
    graph[0][1]["weight"] = 2.0
    graph[1][2]["weight"] = 1.0
    for node, x, nu, p in zip(graph, (0, 1, 3), (1, 2, 3), (0.25, -0.5, 0.125)):
        graph.nodes[node].update(EPI=x, nu_f=nu, delta_nfr=p, theta=0.0)
    return graph


def test_exact_geometry_term_and_flow_term_have_distinct_sources():
    graph = _graph()
    original = deepcopy(graph)
    source = observe_support_transport(graph)
    result = observe_support_transport_derivative(
        source, conductance_rates=(1, 1, -1, -1)
    )
    assert source.rate == (Q(1, 4), -1, Q(3, 8))
    assert result.flow_gradient_rate == (Q(-5, 4), Q(31, 24), Q(-11, 8))
    assert result.geometry_gradient_rate == (0, -1, 0)
    assert result.epi_gradient_rate == (Q(-5, 4), Q(7, 24), Q(-11, 8))
    assert result.nodal_work == Q(1, 4)
    assert result.conductance_work == Q(-3, 2)
    assert result.energy_rate == Q(-5, 4)
    assert nx.utils.graphs_equal(graph, original)


def test_energy_derivative_matches_independent_cubic_edge_polynomial():
    source = observe_support_transport(_graph())
    result = observe_support_transport_derivative(
        source, conductance_rates=(1, 1, -1, -1)
    )

    def energy(t):
        # Detached local jet x+t*x', W+t*W', not a numerical TNFR trajectory.
        x = tuple(a + t * b for a, b in zip(source.epi, source.rate))
        return ((2 + t) * (x[1] - x[0]) ** 2 + (1 - t) * (x[2] - x[1]) ** 2) / 2

    def symmetric_difference(t):
        return (energy(t) - energy(-t)) / (2 * t)

    # The edge energy is cubic, so cancellation removes the t^2 term exactly.
    derivative = (4 * symmetric_difference(Q(1, 8)) - symmetric_difference(Q(1, 4))) / 3
    assert derivative == result.energy_rate


def test_positive_geometry_work_can_exceed_pure_epi_dissipation():
    graph = nx.path_graph(2)
    for node, x, p in ((0, 0, 1), (1, 1, -1)):
        graph.nodes[node].update(EPI=x, nu_f=1, delta_nfr=p)
    source = observe_support_transport(graph)
    result = observe_support_transport_derivative(source, conductance_rates=(6, 6))
    assert result.geometry_gradient_rate == (0, 0)  # normalized walk is unchanged
    assert result.nodal_work == -2
    assert result.conductance_work == 3
    assert result.energy_rate == 1  # changing the energy's own conductance


def test_loops_change_normalized_pressure_without_direct_energy_work():
    graph = nx.Graph()
    graph.add_edge(0, 0, weight=2)
    graph.add_edge(0, 1, weight=1)
    for node, x in ((0, 0), (1, 1)):
        graph.nodes[node].update(EPI=x, nu_f=1, delta_nfr=0)
    source = observe_support_transport(graph)
    result = observe_support_transport_derivative(source, conductance_rates=(1, 0, 0))
    assert result.geometry_gradient_rate == (Q(-1, 9), 0)
    assert result.energy_rate == result.conductance_work == 0


def test_empty_and_zero_conductance_rows_stay_zero_in_this_fixed_active_set():
    for graph in (nx.Graph(), nx.empty_graph(2)):
        for node in graph:
            graph.nodes[node].update(EPI=node, nu_f=1, delta_nfr=1)
        result = observe_support_transport_derivative(
            observe_support_transport(graph),
            conductance_rates=(),
        )
        assert result.epi_gradient_rate == (0,) * len(graph)
        assert result.energy_rate == 0


@pytest.mark.parametrize("rates", [(1,), (1, 0, 0, 0), (1, 1, float("nan"), 0)])
def test_bad_size_asymmetry_or_nonfinite_rates_are_rejected(rates):
    with pytest.raises(ValueError):
        observe_support_transport_derivative(
            observe_support_transport(_graph()), conductance_rates=rates
        )


def test_cached_snapshot_results_cannot_supply_a_false_derivative():
    source = observe_support_transport(_graph())
    altered = replace(source, rate=(0, 0, 0), epi_gradient=(0, 0, 0), energy_rate=0)
    result = observe_support_transport_derivative(altered, conductance_rates=(0,) * 4)
    assert result.source == source
    assert result.nodal_work == source.energy_rate == Q(1, 4)


def test_rates_are_explicit_ordered_coefficients_not_a_claim_dictionary():
    with pytest.raises(TypeError, match="ordered sequence"):
        observe_support_transport_derivative(
            observe_support_transport(_graph()), conductance_rates={"certified": True}
        )
