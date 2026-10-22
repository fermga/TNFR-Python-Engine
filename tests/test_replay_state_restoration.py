"""Core trace snapshots retain detached recorded state.

Retired lab-specific replay controls are recoverable through
applications/factorization-lab/RETIREMENT.json.
"""

from types import MappingProxyType

import networkx as nx
import pytest


@pytest.mark.parametrize(
    "graph_key,field",
    [
        ("GAMMA", "gamma"),
        ("GRAMMAR_CANON", "grammar"),
        ("DNFR_WEIGHTS", "dnfr_weights"),
        ("_Si_weights", "si_weights"),
    ],
)
def test_trace_mapping_is_detached_from_later_graph_edits(graph_key, field):
    from tnfr.trace import mapping_field

    graph = nx.Graph()
    graph.graph[graph_key] = {"value": 0.5, "nested": {"values": [1, 2]}}
    snapshot = mapping_field(graph, graph_key, field)[field]
    graph.graph[graph_key]["value"] = 0.9
    graph.graph[graph_key]["nested"]["values"].append(3)
    assert snapshot["value"] == 0.5
    assert snapshot["nested"]["values"] == [1, 2]
    with pytest.raises(TypeError):
        snapshot["value"] = 7


def test_recorded_trace_detaches_custom_field_and_nested_proxy():
    from tnfr.trace import _trace_capture

    graph = nx.Graph()
    graph.graph["TRACE"] = {"enabled": True, "capture": ["custom"]}
    values = {"items": [1], "nested": MappingProxyType({"values": [2]})}
    _trace_capture(graph, "before", {"custom": lambda graph: {"custom": values}})
    snapshot = graph.graph["history"]["trace_meta"][-1]["custom"]
    values["items"].append(9)
    values["nested"]["values"].append(9)
    assert snapshot["items"] == [1]
    assert snapshot["nested"]["values"] == [2]


def test_default_trace_context_detaches_transition_inputs_and_saved_history():
    from tnfr.core.default_implementations import DefaultTraceContext

    graph = nx.Graph()
    pre = {"coherence": 0.5, "nested": [1]}
    post = {"coherence": 0.8, "nested": [2]}
    with DefaultTraceContext(graph) as context:
        context.record_transition("IL", pre, post)
        pre["nested"].append(9)
        post["coherence"] = 0.1
    saved = graph.graph["_trace_transitions"][0]
    assert saved["pre"]["nested"] == [1]
    assert saved["post"]["coherence"] == 0.8
    context.transitions[0]["post"]["nested"].append(9)
    assert saved["post"]["nested"] == [2]
