"""History bounds limit retained observations, never metric availability."""

from collections import deque

import networkx as nx
import pytest

from tnfr.constants import inject_defaults
from tnfr.glyph_history import HistoryDict, ensure_history


@pytest.mark.parametrize("invalid_bound", [True, 1.5, "2"])
def test_history_bound_rejects_coercion_before_replacing_retained_history(
    invalid_bound,
):
    graph = nx.Graph()
    inject_defaults(graph)
    original = {"C_steps": [0.1, 0.2]}
    graph.graph.update(HISTORY_MAXLEN=invalid_bound, history=original)
    with pytest.raises(TypeError):
        ensure_history(graph)
    assert graph.graph["history"] is original
    assert original == {"C_steps": [0.1, 0.2]}


def test_sample_bound_preserves_all_metric_streams_and_explicit_eviction():
    graph = nx.Graph()
    inject_defaults(graph)
    graph.graph.update(
        HISTORY_MAXLEN=2,
        history={"C_steps": [0.1, 0.2, 0.3], "stable_frac": [0.0], "W_bar": [0.2]},
    )
    history = ensure_history(graph)
    assert isinstance(history, HistoryDict)
    assert set(history) == {"C_steps", "stable_frac", "W_bar"}
    assert list(history["C_steps"]) == [0.2, 0.3]
    history["C_steps"].append(0.4)
    assert ensure_history(graph) is history
    assert list(history["C_steps"]) == [0.3, 0.4]
    assert set(history) == {"C_steps", "stable_frac", "W_bar"}

    history.get_increment("C_steps")
    history.get_increment("W_bar")
    assert list(history.pop_least_used()) == [0.0]
    assert set(history) == {"C_steps", "W_bar"}


def test_changing_history_bound_resizes_existing_series_and_preserves_payloads():
    graph = nx.Graph()
    inject_defaults(graph)
    payload = {"node": "metadata"}
    graph.graph.update(
        HISTORY_MAXLEN=3,
        history={"C_steps": deque([0.1, 0.2, 0.3], maxlen=3), "metadata": payload},
    )
    original = ensure_history(graph)["C_steps"]
    graph.graph["HISTORY_MAXLEN"] = 2
    history = ensure_history(graph)
    assert history["C_steps"].maxlen == 2
    assert list(history["C_steps"]) == [0.2, 0.3]
    assert list(original) == [0.1, 0.2, 0.3]
    assert history["metadata"] is payload

    graph.graph["HISTORY_MAXLEN"] = 4
    history = ensure_history(graph)
    history["C_steps"].extend([0.4, 0.5])
    assert history["C_steps"].maxlen == 4
    assert list(history["C_steps"]) == [0.2, 0.3, 0.4, 0.5]

    graph.graph["HISTORY_MAXLEN"] = 0
    history = ensure_history(graph)
    assert type(history) is dict
    assert isinstance(history["C_steps"], list)
    history["C_steps"].append(0.6)
    assert history["C_steps"] == [0.2, 0.3, 0.4, 0.5, 0.6]
    assert history["metadata"] is payload
