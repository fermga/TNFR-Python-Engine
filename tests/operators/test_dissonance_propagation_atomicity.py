"""Atomic validation and directed conductance for OZ propagation."""

from __future__ import annotations

import math
import sys
from copy import deepcopy

import networkx as nx
import pytest

from tnfr.constants.aliases import ALIAS_DNFR, ALIAS_THETA, ALIAS_VF
from tnfr.constants.canonical import DELTA_PHI_MAX
from tnfr.dynamics.propagation import propagate_dissonance
from tnfr.errors import TNFRValueError


def _add_state(graph: nx.Graph, node: int, *, dnfr: float = 0.0) -> None:
    graph.add_node(
        node,
        **{
            ALIAS_THETA[0]: 0.0,
            ALIAS_VF[0]: 1.0,
            ALIAS_DNFR[0]: dnfr,
            "glyph_history": ["AL"],
        },
    )


def _two_neighbor_graph(graph_type: type[nx.Graph] = nx.Graph) -> nx.Graph:
    graph = graph_type()
    for node in range(3):
        _add_state(graph, node)
    graph.add_edge(0, 1, weight=1.0)
    graph.add_edge(0, 2, weight=1.0)
    return graph


def _state_snapshot(graph: nx.Graph) -> tuple[dict, dict, list]:
    if graph.is_multigraph():
        edges = list(graph.edges(keys=True, data=True))
    else:
        edges = list(graph.edges(data=True))
    return (
        deepcopy(graph.graph),
        {node: deepcopy(dict(data)) for node, data in graph.nodes(data=True)},
        deepcopy(edges),
    )


def _assert_state_unchanged(graph: nx.Graph, before: tuple[dict, dict, list]) -> None:
    assert _state_snapshot(graph) == before


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"OZ_PHASE_THRESHOLD": 0.0}, "phase threshold must be positive"),
        ({"OZ_PHASE_THRESHOLD": math.nan}, "phase threshold must be finite"),
        ({"OZ_PHASE_THRESHOLD": math.inf}, "phase threshold must be finite"),
        ({"OZ_PHASE_THRESHOLD": True}, "phase threshold must be a finite"),
        ({"OZ_PHASE_THRESHOLD": "1.0"}, "phase threshold must be a finite"),
        ({"OZ_MIN_PROPAGATION": -0.1}, "minimum propagation must be nonnegative"),
        ({"OZ_MIN_PROPAGATION": math.nan}, "minimum propagation must be finite"),
        ({"OZ_MIN_PROPAGATION": math.inf}, "minimum propagation must be finite"),
        ({"OZ_MIN_PROPAGATION": False}, "minimum propagation must be a finite"),
    ],
)
def test_invalid_policy_rejects_without_state_or_telemetry_commit(config, message):
    graph = _two_neighbor_graph()
    graph.graph.update(config)
    before = _state_snapshot(graph)

    with pytest.raises(TNFRValueError, match=message):
        propagate_dissonance(graph, 0, 0.2)

    _assert_state_unchanged(graph, before)


@pytest.mark.parametrize(
    ("magnitude", "mode", "message"),
    [
        (math.nan, "phase_weighted", "magnitude must be finite"),
        (math.inf, "phase_weighted", "magnitude must be finite"),
        (-0.1, "phase_weighted", "magnitude must be nonnegative"),
        (True, "phase_weighted", "magnitude must be a finite"),
        ("0.2", "phase_weighted", "magnitude must be a finite"),
        (0.2, "unknown", "propagation_mode must be one of"),
    ],
)
def test_invalid_argument_rejects_without_state_or_telemetry_commit(
    magnitude, mode, message
):
    graph = _two_neighbor_graph()
    before = _state_snapshot(graph)

    with pytest.raises(TNFRValueError, match=message):
        propagate_dissonance(graph, 0, magnitude, propagation_mode=mode)

    _assert_state_unchanged(graph, before)


def test_invalid_late_edge_weight_rolls_back_whole_outgoing_neighborhood():
    graph = _two_neighbor_graph()
    existing_events = [{"prior": True}]
    graph.nodes[1]["_oz_propagation"] = existing_events
    graph[0][2]["weight"] = math.nan
    before = _state_snapshot(graph)

    with pytest.raises(TNFRValueError, match="edge weight.*must be finite"):
        propagate_dissonance(graph, 0, 0.2)

    _assert_state_unchanged(graph, before)
    assert graph.nodes[1]["_oz_propagation"] is existing_events


def test_overflowing_late_proposal_rolls_back_earlier_finite_proposal():
    graph = _two_neighbor_graph()
    graph.nodes[2][ALIAS_DNFR[0]] = sys.float_info.max
    before = _state_snapshot(graph)

    with pytest.raises(TNFRValueError, match="proposed pressure.*must be finite"):
        propagate_dissonance(graph, 0, sys.float_info.max)

    _assert_state_unchanged(graph, before)


def test_invalid_late_telemetry_sink_rolls_back_state_and_existing_history():
    graph = _two_neighbor_graph()
    existing_events = [{"prior": True}]
    graph.nodes[1]["_oz_propagation"] = existing_events
    graph.nodes[2]["_oz_propagation"] = ("invalid",)
    before = _state_snapshot(graph)

    with pytest.raises(TNFRValueError, match="_oz_propagation.*must be a list"):
        propagate_dissonance(graph, 0, 0.2)

    _assert_state_unchanged(graph, before)
    assert graph.nodes[1]["_oz_propagation"] is existing_events


@pytest.mark.parametrize(
    ("node", "attribute", "value", "message"),
    [
        (0, ALIAS_THETA[0], math.nan, "source phase.*must be .*finite"),
        (0, ALIAS_VF[0], -0.1, "source structural frequency.*nonnegative"),
        (2, ALIAS_THETA[0], math.inf, "neighbor phase.*must be .*finite"),
        (2, ALIAS_VF[0], math.nan, "neighbor structural frequency.*finite"),
        (2, ALIAS_DNFR[0], math.inf, "neighbor pressure.*must be .*finite"),
    ],
)
def test_noncanonical_node_state_rejects_before_any_neighbor_commit(
    node, attribute, value, message
):
    graph = _two_neighbor_graph()
    graph.nodes[node][attribute] = value
    before = _state_snapshot(graph)

    with pytest.raises(TNFRValueError, match=message):
        propagate_dissonance(graph, 0, 0.2)

    _assert_state_unchanged(graph, before)


def test_directed_propagation_uses_only_outgoing_arc_and_its_weight():
    graph = nx.DiGraph()
    for node in range(3):
        _add_state(graph, node)
    graph.add_edge(0, 1, weight=0.5)
    graph.add_edge(1, 0, weight=100.0)
    graph.add_edge(2, 0, weight=math.nan)

    affected = propagate_dissonance(graph, 0, 0.2)

    assert affected == {1}
    assert graph.nodes[1][ALIAS_DNFR[0]] == pytest.approx(0.1)
    assert graph.nodes[2][ALIAS_DNFR[0]] == 0.0
    event = graph.nodes[1]["_oz_propagation"][0]
    assert event["coupling_weight"] == 0.5


def test_multidigraph_sums_parallel_outgoing_conductances():
    graph = nx.MultiDiGraph()
    _add_state(graph, 0)
    _add_state(graph, 1)
    graph.add_edge(0, 1, weight=0.25)
    graph.add_edge(0, 1, weight=0.75)
    graph.add_edge(1, 0, weight=10.0)

    affected = propagate_dissonance(graph, 0, 0.2)

    assert affected == {1}
    assert graph.nodes[1][ALIAS_DNFR[0]] == pytest.approx(0.2)
    event = graph.nodes[1]["_oz_propagation"][0]
    assert event["coupling_weight"] == pytest.approx(1.0)


def test_frequency_weighted_mode_preserves_existing_coupling_formula():
    graph = nx.DiGraph()
    _add_state(graph, 0)
    _add_state(graph, 1)
    graph.nodes[0][ALIAS_VF[0]] = 2.0
    graph.nodes[1][ALIAS_VF[0]] = 1.0
    graph.add_edge(0, 1, weight=0.5)

    affected = propagate_dissonance(
        graph, 0, 0.4, propagation_mode="frequency_weighted"
    )

    assert affected == {1}
    assert graph.nodes[1][ALIAS_DNFR[0]] == pytest.approx(0.1)


class _FailingNodeDict(dict):
    fail_next_event_write = False

    def __setitem__(self, key, value):
        if key == "_oz_propagation" and self.fail_next_event_write:
            self.fail_next_event_write = False
            raise RuntimeError("simulated telemetry write failure")
        super().__setitem__(key, value)


class _FailingGraph(nx.Graph):
    node_attr_dict_factory = _FailingNodeDict


def test_unexpected_commit_failure_restores_all_prior_writes():
    graph = _two_neighbor_graph(_FailingGraph)
    existing_events = [{"prior": True}]
    graph.nodes[1]["_oz_propagation"] = existing_events
    graph.nodes[2].fail_next_event_write = True
    before = _state_snapshot(graph)

    with pytest.raises(RuntimeError, match="simulated telemetry write failure"):
        propagate_dissonance(graph, 0, 0.2)

    _assert_state_unchanged(graph, before)
    assert graph.nodes[1]["_oz_propagation"] is existing_events


def test_zero_propagation_is_not_reported_as_an_affected_neighbor():
    graph = _two_neighbor_graph()
    graph.graph["OZ_MIN_PROPAGATION"] = 0.0
    graph.nodes[1][ALIAS_THETA[0]] = DELTA_PHI_MAX
    graph.remove_edge(0, 2)

    affected = propagate_dissonance(graph, 0, 0.2)

    assert affected == set()
    assert "_oz_propagation" not in graph.nodes[1]
    assert graph.nodes[1][ALIAS_DNFR[0]] == 0.0
