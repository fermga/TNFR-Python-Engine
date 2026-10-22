"""Distinguish canonical REMESH advice from actual delayed EPI evolution."""

from collections import deque
from copy import deepcopy

import pytest

from benchmarks.capacity_localization import build_cycle
from tnfr.operators import (
    build_operator_event_schedule,
    execute_operator_event_schedule,
)
from tnfr.utils.numeric import angle_diff


def _prepared_cycle():
    graph = build_cycle(8, epi=(1.0,) + (0.5,) * 7)
    graph.graph["_epi_hist"] = deque(
        [{node: graph.nodes[node]["EPI"] for node in graph}], maxlen=8
    )
    return graph


def test_explicit_legacy_coupling_recursivity_rejects_without_physical_advance():
    graph = _prepared_cycle()
    # Rollback may replace internal containers; physical values are the claim.
    original_nodes = deepcopy(dict(graph.nodes(data=True)))
    original_edges = deepcopy(list(graph.edges(data=True)))
    original_history = deepcopy(graph.graph["_epi_hist"])
    original_time = graph.graph["_t"]
    schedule = build_operator_event_schedule(
        ("coupling", "recursivity"),
        start_time=0.0,
        flow_durations=(0.0, 0.0, 0.0),
    )
    with pytest.raises(ValueError, match="(?i)(coupling|UM|recursivity|REMESH)"):
        execute_operator_event_schedule(
            graph,
            schedule,
            context={"initial_epi_nonzero": True, "compatibility_profile": "legacy"},
        )
    assert dict(graph.nodes(data=True)) == original_nodes
    assert list(graph.edges(data=True)) == original_edges
    assert graph.graph["_epi_hist"] == original_history
    assert graph.graph["_t"] == original_time


@pytest.mark.parametrize(
    "names,expected_history",
    [
        (("coupling", "recursivity"), ("UM", "REMESH")),
        (("coupling", "coherence", "recursivity"), ("UM", "IL", "REMESH")),
    ],
)
def test_admitted_core_word_records_advice_without_mixing_or_advancing_history(
    names, expected_history
):
    graph = _prepared_cycle()
    original_epi = tuple(graph.nodes[node]["EPI"] for node in graph)
    original_phase = tuple(graph.nodes[node]["theta"] for node in graph)
    original_history = deepcopy(graph.graph["_epi_hist"])
    schedule = build_operator_event_schedule(
        names,
        start_time=0.0,
        flow_durations=(0.0,) * (len(names) + 1),
    )
    result = execute_operator_event_schedule(
        graph, schedule, context={"initial_epi_nonzero": True}
    )
    assert result.whole_schedule_graph_state_atomic
    assert tuple(graph.nodes[node]["EPI"] for node in graph) == original_epi
    assert tuple(graph.nodes[node]["nu_f"] for node in graph) == (1.0,) * 8
    assert graph.graph["_epi_hist"] == original_history
    assert graph.graph["_t"] == 0.0
    assert (
        max(
            abs(angle_diff(graph.nodes[node]["theta"], original_phase[node]))
            for node in graph
        )
        < 1e-14
    )
    assert all(
        tuple(graph.nodes[node]["glyph_history"])[-len(names) :] == expected_history
        for node in graph
    )
