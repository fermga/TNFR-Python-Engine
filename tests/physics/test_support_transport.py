"""Exact transport controls for attachment, support and held nodal flow."""

from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
from fractions import Fraction

import networkx as nx
import pytest

from tnfr.constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_VF
from tnfr.dynamics.integrators import update_epi_via_nodal_equation
from tnfr.mathematics import BEPIElement
from tnfr.physics.support_transport import (
    observe_support_transport,
    observe_support_transport_euler,
    observe_support_transport_reset,
)


F = Fraction


def _initialize(graph, epi, capacity, pressure):
    for node, x, nu, p in zip(graph, epi, capacity, pressure, strict=True):
        graph.nodes[node].update({
            ALIAS_EPI[0]: x, ALIAS_VF[0]: nu, ALIAS_DNFR[0]: p, "theta": 0.0,
        })
    return graph


def _attachment():
    graph = nx.Graph()
    graph.add_nodes_from(("parent", "peer", "child"))
    graph.add_edge("parent", "peer", weight=2.0)
    return _initialize(graph, (1.0, 0.5, 0.25), (2.0, 1.0, 0.5), (-0.25, 0.5, 0.75))


def _flow_graph(epi, capacity, pressure):
    graph = _initialize(nx.path_graph(2), epi, capacity, pressure)
    graph.graph.update(
        _t=0.0, _gamma_spec={"type": "none"}, GAMMA={"type": "none"},
        use_extended_dynamics=False, DT_MIN=0.0,
        EPI_MIN=-4.0, EPI_MAX=4.0, CLIP_MODE="hard",
    )
    return graph


def test_weighted_child_attachment_has_distinct_channels_and_exact_energy_reset():
    graph = _attachment()
    before = observe_support_transport(graph)
    assert before.epi_gradient == (F(-1, 2), F(1, 2), 0)
    assert before.capacity_gradient == (-1, 1, 0)
    assert before.topology_gradient == (0, 0, 0)
    assert before.dirichlet_gradient == (1, -1, 0)
    assert before.dirichlet_energy == F(1, 4)
    assert before.rate == (F(-1, 2), F(1, 2), F(3, 8))
    assert before.energy_rate == -1

    graph.add_edge("parent", "child", weight=0.5)
    after = observe_support_transport(graph)
    assert after.nodes == ("parent", "peer", "child")
    assert after.conductance == ((0, 1, F(2)), (0, 2, F(1, 2)),
                                 (1, 0, F(2)), (2, 0, F(1, 2)))
    assert after.support_neighbors == ((1, 2), (0,), (0,))
    assert after.epi_gradient == (F(-11, 20), F(1, 2), F(3, 4))
    assert after.capacity_gradient == (F(-5, 4), 1, F(3, 2))
    assert after.topology_gradient == (-1, 1, 1)
    assert after.dirichlet_gradient == (F(11, 8), -1, F(-3, 8))
    assert after.dirichlet_energy == F(25, 64)
    assert after.energy_rate == F(-85, 64)
    reset = observe_support_transport_reset(before, after)
    assert reset.energy_change == reset.edge_energy_change == F(9, 64)
    assert reset.identity_residual == 0
    removed = observe_support_transport_reset(after, before)
    assert removed.energy_change == removed.edge_energy_change == F(-9, 64)


def test_zero_weight_edge_changes_support_channels_without_epi_transport():
    graph = _initialize(nx.Graph(), (), (), ())
    graph.add_nodes_from((0, 1, 2))
    _initialize(graph, (1.0, 0.0, 2.0), (1.0, 2.0, 4.0), (0.0,) * 3)
    graph.add_edge(0, 1, weight=1.0)
    before = observe_support_transport(graph)
    graph.add_edge(0, 2, weight=0.0)
    after = observe_support_transport(graph)
    assert after.conductance == before.conductance
    assert after.epi_gradient == before.epi_gradient == (-1, 1, 0)
    assert after.capacity_gradient == (2, -1, -3)
    assert after.topology_gradient == (-1, 1, 1)
    assert after.support_neighbors == ((1, 2), (0,), (0,))
    reset = observe_support_transport_reset(before, after)
    assert reset.energy_change == reset.edge_energy_change == reset.identity_residual == 0


def test_effectively_symmetric_zero_conductance_can_have_directed_support():
    graph = nx.DiGraph()
    graph.add_edge(0, 1, weight=0.0)
    _initialize(graph, (1.0, -1.0), (1.0, 3.0), (0.0, 0.0))
    result = observe_support_transport(graph)
    assert result.conductance == ()
    assert result.support_neighbors == ((1,), ())
    assert result.epi_gradient == (0, 0)
    assert result.capacity_gradient == (2, 0)
    assert result.topology_gradient == (-1, 0)
    assert result.dirichlet_energy == 0


def test_parallel_edges_aggregate_and_loops_count_once_per_unique_support_row():
    graph = nx.MultiGraph()
    graph.add_nodes_from((0, 1, 2))
    graph.add_edge(0, 1, weight=1.0)
    graph.add_edge(0, 1, weight=2.0)
    graph.add_edge(0, 0, weight=4.0)
    _initialize(graph, (1.0, 0.0, 2.0), (1.0, 3.0, 5.0), (-2.0, 1.0, 0.5))
    result = observe_support_transport(graph)
    assert result.conductance == ((0, 0, F(4)), (0, 1, F(3)), (1, 0, F(3)))
    assert result.support_neighbors == ((0, 1), (0,), ())
    assert result.epi_gradient == (F(-3, 7), 1, 0)
    assert result.capacity_gradient == (1, -2, 0)
    assert result.topology_gradient == (F(-1, 2), 1, 0)
    assert result.dirichlet_gradient == (3, -3, 0)
    assert result.dirichlet_energy == F(3, 2)


def test_attached_capacity_channel_has_nonzero_weighted_total():
    graph = nx.cycle_graph(8)
    graph.add_node("child")
    _initialize(graph, (1.0,) * 9, (1.0,) * 8 + (0.75,), (0.0,) * 9)
    graph.add_edge(0, "child", weight=0.5)
    result = observe_support_transport(graph)
    assert result.capacity_gradient == (F(-1, 12),) + (0,) * 7 + (F(1, 4),)
    strengths = tuple(sum(w for i, _, w in result.conductance if i == n)
                      for n in range(9))
    total = sum(d * g for d, g in zip(strengths, result.capacity_gradient))
    assert total == F(-1, 12)
    assert total == 2 * (F(1, 2) - 1) * (1 - F(3, 4)) / 3
    assert result.epi_gradient == (0,) * 9


@pytest.mark.parametrize(("location", "value"), [
    ("weight", -1.0), ("weight", float("nan")), ("weight", float("inf")),
    (ALIAS_VF[0], -0.5), (ALIAS_VF[0], float("inf")),
    (ALIAS_DNFR[0], float("nan")), (ALIAS_EPI[0], float("inf")),
])
def test_invalid_materialized_state_or_conductance_is_rejected(location, value):
    graph = _attachment()
    if location == "weight":
        graph.edges["parent", "peer"][location] = value
    else:
        graph.nodes["parent"][location] = value
    with pytest.raises(ValueError):
        observe_support_transport(graph)


def test_asymmetric_positive_conductance_is_rejected():
    graph = nx.DiGraph()
    graph.add_edge(0, 1, weight=1.0)
    _initialize(graph, (1.0, 0.0), (1.0, 1.0), (0.0, 0.0))
    with pytest.raises(ValueError, match="symmetric"):
        observe_support_transport(graph)


def test_exact_shared_euler_has_negative_drift_and_positive_quadratic_remainder():
    graph = _flow_graph((1.0, 0.0), (1.0, 2.0), (-0.5, 0.25))
    graph.edges[0, 1]["weight"] = 2.0
    before = observe_support_transport(graph)
    update_epi_via_nodal_equation(graph, dt=0.5, method="euler")
    after = observe_support_transport(graph)
    result = observe_support_transport_euler(before, after, F(1, 2))
    assert result.expected_epi == after.epi == (F(3, 4), F(1, 4))
    assert result.state_defect == (0, 0)
    assert result.drift_term == -1
    assert result.quadratic_term == F(1, 4)
    assert result.defect_term == 0
    assert result.energy_change == F(-3, 4)
    assert result.identity_residual == 0


def test_binary64_euler_defect_has_an_independent_two_node_energy_expansion():
    graph = _flow_graph((0.7, 0.2), (1.1, 0.9), (-0.3, 0.4))
    before = observe_support_transport(graph)
    update_epi_via_nodal_equation(graph, dt=0.1, method="euler")
    after = observe_support_transport(graph)
    result = observe_support_transport_euler(before, after, 0.1)
    h = F(0.1)
    expected = tuple(x + h * nu * p for x, nu, p in
                     zip(before.epi, before.capacity, before.stored_pressure))
    defect = tuple(actual - ideal for actual, ideal in zip(after.epi, expected))
    assert result.expected_epi == expected
    assert result.state_defect == defect and any(defect)
    assert result.defect_term == ((expected[0] - expected[1]) * (defect[0] - defect[1])
                                  + (defect[0] - defect[1])**2 / 2)
    expected_change = ((after.epi[0] - after.epi[1])**2
                       - (before.epi[0] - before.epi[1])**2) / 2
    assert result.energy_change == expected_change
    assert result.energy_change == result.drift_term + result.quadratic_term + result.defect_term
    assert result.identity_residual == 0


def test_transition_readers_rebuild_cached_fields_without_trusting_public_data():
    graph = _attachment()
    before = observe_support_transport(graph)
    graph.add_edge("parent", "child", weight=0.5)
    after = observe_support_transport(graph)
    forged = replace(before, dirichlet_energy=F(999), energy_rate=F(999),
                     rate=(F(999),) * 3, epi_gradient=(F(999),) * 3)
    assert observe_support_transport_reset(forged, after) == (
        observe_support_transport_reset(before, after)
    )
    endpoint = replace(before, epi=(F(3, 4), F(3, 4), F(1, 4)))
    assert observe_support_transport_euler(forged, endpoint, F(1, 2)) == (
        observe_support_transport_euler(before, endpoint, F(1, 2))
    )
    invalid = replace(before, capacity=(F(-1), F(1), F(1)))
    with pytest.raises(ValueError, match="capacity"):
        observe_support_transport_reset(invalid, after)


@pytest.mark.parametrize("changed", ["nodes", "epi"])
def test_reset_rejects_unaligned_nodes_or_an_epi_jump(changed):
    before = observe_support_transport(_attachment())
    after = replace(before, **{
        changed: tuple(reversed(getattr(before, changed))),
    })
    with pytest.raises(ValueError, match="identical node order and EPI"):
        observe_support_transport_reset(before, after)


@pytest.mark.parametrize("changed", ["nodes", "conductance", "capacity"])
def test_euler_requires_the_declared_fixed_operator_and_capacity(changed):
    before = observe_support_transport(_attachment())
    values = {
        "nodes": tuple(reversed(before.nodes)),
        "conductance": tuple((i, j, 2 * w) for i, j, w in before.conductance),
        "capacity": (F(1),) * 3,
    }
    after = replace(before, **{changed: values[changed]})
    with pytest.raises(ValueError, match="fixed node order, conductance and capacity"):
        observe_support_transport_euler(before, after, F(1, 4))


def test_snapshot_reuses_signed_uniform_real_bepi_and_remains_detached():
    graph = _flow_graph((-1.0, 0.0), (1.0, 1.0), (0.25, -0.5))
    expected = observe_support_transport(graph)
    graph.nodes[0][ALIAS_EPI[0]] = BEPIElement((-1.0,) * 2, (-1.0,) * 2, (0.0, 1.0))
    graph.graph["research_tag"] = {"values": [1, 2]}
    nodes = deepcopy(dict(graph.nodes(data=True)))
    metadata = deepcopy(graph.graph)
    edges = deepcopy(dict(graph.edges))
    observed = observe_support_transport(graph)
    assert observed == expected
    assert dict(graph.nodes(data=True)) == nodes
    assert graph.graph == metadata and dict(graph.edges) == edges
    with pytest.raises(FrozenInstanceError):
        observed.dirichlet_energy = F(999)
    graph.nodes[0][ALIAS_EPI[0]] = 3.0
    graph.edges[0, 1]["weight"] = 2.0
    assert observed.epi == (-1, 0)
    assert observed.conductance == ((0, 1, F(1)), (1, 0, F(1)))
