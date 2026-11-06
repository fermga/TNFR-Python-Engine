"""A propagation record and an acceleration crossing do not prove causation."""

from copy import deepcopy

import networkx as nx
import pytest

from tnfr.constants.aliases import ALIAS_D2EPI, ALIAS_EPI
from tnfr.dynamics.propagation import detect_bifurcation_cascade, propagate_dissonance
from tnfr.errors import TNFRValueError


def _graph():
    graph = nx.Graph([(0, 1), (0, 2), (3, 1)])
    for node in graph:
        graph.nodes[node].update(
            {ALIAS_EPI[0]: 4.0, "epi_time_history": [(0, 0), (1, 1), (2, 4)]}
        )
    return graph


def _state(graph):
    return deepcopy(
        (graph.graph, dict(graph.nodes(data=True)), list(graph.edges(data=True)))
    )


def test_input_from_another_source_is_not_attributed_to_requested_source():
    graph = _graph()
    propagate_dissonance(graph, 3, 0.2)
    before = _state(graph)
    assert detect_bifurcation_cascade(graph, 0, threshold=0.5) == []
    assert _state(graph) == before


def test_actual_input_and_crossing_are_reported_without_causal_promotion():
    graph = _graph()
    propagate_dissonance(graph, 0, 0.2)
    assert detect_bifurcation_cascade(graph, 0, threshold=0.5) == [1, 2]
    for node in (1, 2):
        record = graph.nodes[node]["_bifurcation_cascade"]
        assert record["triggered_by"] == 0
        assert record["causal_attribution"] is False
        assert record["history_source"] == "epi_time_history"
        assert record["d2epi"] == 2.0
        assert graph.nodes[node][ALIAS_D2EPI[0]] == 2.0


def test_short_authoritative_history_does_not_use_old_cache_or_legacy_history():
    graph = _graph()
    propagate_dissonance(graph, 0, 0.2)
    for node in (1, 2):
        graph.nodes[node]["epi_time_history"] = [(2.0, 4.0)]
        graph.nodes[node]["_epi_history"] = [0.0, 0.0, 100.0]
        graph.nodes[node][ALIAS_D2EPI[0]] = 100.0
    before = _state(graph)
    assert detect_bifurcation_cascade(graph, 0, threshold=0.0) == []
    assert _state(graph) == before


def test_late_invalid_history_does_not_commit_earlier_observation():
    graph = _graph()
    propagate_dissonance(graph, 0, 0.2)
    graph.nodes[2]["epi_time_history"][-1] = (2.0, 99.0)
    before = _state(graph)
    with pytest.raises(TNFRValueError, match="current nodal state"):
        detect_bifurcation_cascade(graph, 0)
    assert _state(graph) == before


@pytest.mark.parametrize("threshold", [-1.0, True, "1", float("inf"), float("nan")])
def test_invalid_threshold_rejects_without_writing(threshold):
    graph = _graph()
    before = _state(graph)
    with pytest.raises(TNFRValueError, match="threshold"):
        detect_bifurcation_cascade(graph, 0, threshold=threshold)
    assert _state(graph) == before


def test_threshold_equality_and_zero_record_do_not_establish_crossing():
    graph = _graph()
    propagate_dissonance(graph, 0, 0.2)
    assert detect_bifurcation_cascade(graph, 0, threshold=2.0) == []
    graph.nodes[1]["_oz_propagation"] = [{"from_node": 0, "magnitude": 0.0}]
    graph.nodes[2]["_oz_propagation"] = []
    assert detect_bifurcation_cascade(graph, 0, threshold=0.0) == []


def test_directed_incoming_only_neighbor_is_not_a_cascade_candidate():
    graph = _graph().to_directed()
    graph.remove_edge(0, 1)
    graph.nodes[1]["_oz_propagation"] = [{"from_node": 0, "magnitude": 0.2}]
    assert detect_bifurcation_cascade(graph, 0) == []
