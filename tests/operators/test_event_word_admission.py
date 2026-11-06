"""Scheduled execution admits actual operator instances once under shared policy."""

from copy import deepcopy

import networkx as nx
import pytest

from tnfr.errors import TNFRValueError
from tnfr.operators.definitions import Emission, Silence
from tnfr.operators.event_runtime import execute_operator_event_schedule
from tnfr.operators.event_timing import build_operator_event_schedule
from tnfr.operators.grammar_core import GrammarValidator


def graph_and_schedule():
    graph = nx.path_graph(2)
    graph.graph.update(_t=0.0, RANDOM_SEED=7)
    for node in graph:
        graph.nodes[node].update(
            EPI=0.0, nu_f=1.0, theta=0.0, delta_nfr=0.0, glyph_history=[]
        )
    schedule = build_operator_event_schedule(
        ("emission", "silence"),
        start_time=0.0,
        flow_durations=(0.0, 0.0, 0.0),
    )
    return graph, schedule


@pytest.mark.parametrize("profile", ["core", "legacy"])
def test_schedule_uses_one_actual_instance_admission(profile, monkeypatch):
    calls = []
    original = GrammarValidator.validate_checks

    def checked(self, sequence, *args, **kwargs):
        calls.append(tuple(type(operator) for operator in sequence))
        return original(self, sequence, *args, **kwargs)

    monkeypatch.setattr(GrammarValidator, "validate_checks", checked)
    graph, schedule = graph_and_schedule()
    before = deepcopy((graph.graph, dict(graph.nodes(data=True))))
    if profile == "legacy":
        with pytest.raises(
            TNFRValueError, match="Invalid operator-event sequence"
        ) as error:
            execute_operator_event_schedule(
                graph, schedule, context={"compatibility_profile": profile}
            )
        assert isinstance(error.value.__cause__, TNFRValueError)
        assert "legacy" in error.value.message
        assert (graph.graph, dict(graph.nodes(data=True))) == before
    else:
        result = execute_operator_event_schedule(graph, schedule)
        assert result.whole_schedule_graph_state_atomic
        assert all(
            tuple(graph.nodes[node]["glyph_history"]) == ("AL", "SHA") for node in graph
        )
        assert graph.graph["_t"] == 0.0
    assert calls == [(Emission, Silence)]


@pytest.mark.parametrize("profile", [True, "unknown"])
def test_invalid_profile_rejects_before_scheduled_state_changes(profile):
    graph, schedule = graph_and_schedule()
    before = deepcopy((graph.graph, dict(graph.nodes(data=True))))
    with pytest.raises((ValueError, TypeError), match="compatibility_profile"):
        execute_operator_event_schedule(
            graph, schedule, context={"compatibility_profile": profile}
        )
    assert (graph.graph, dict(graph.nodes(data=True))) == before
