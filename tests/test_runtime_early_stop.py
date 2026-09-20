"""Early stopping requires valid policy and consecutive fresh observations."""

from argparse import Namespace
from copy import deepcopy

import networkx as nx
import pytest

from tnfr.cli.execution import apply_cli_config
from tnfr.constants import inject_defaults
from tnfr.dynamics import runtime
from tnfr.glyph_history import ensure_history
from tnfr.metrics.coherence import _track_stability
from tnfr.metrics.core import register_metrics_callbacks


def _graph():
    graph = nx.Graph()
    inject_defaults(graph)
    graph.graph.update(
        STOP_EARLY={"enabled": True, "window": 2, "fraction": 0.9},
        history={"stable_frac": []},
    )
    return graph


@pytest.mark.parametrize(
    "override",
    [
        {"enabled": "false"},
        {"window": True},
        {"window": 1.5},
        {"window": 0},
        {"fraction": True},
        {"fraction": "0.9"},
        {"fraction": float("inf")},
        {"fraction": -0.1},
        {"fraction": 1.1},
    ],
)
def test_invalid_policy_rejects_before_step(monkeypatch, override):
    graph = _graph()
    graph.graph["STOP_EARLY"].update(override)
    before = deepcopy(graph.graph)
    calls = []
    monkeypatch.setattr(runtime, "step", lambda *a, **k: calls.append(1))
    with pytest.raises((TypeError, ValueError)):
        runtime.run(graph, 3)
    assert not calls
    assert graph.graph == before


@pytest.mark.parametrize("gap", [None, "missing", True, float("nan"), 1.1])
def test_missing_or_invalid_sample_breaks_consecutive_window(monkeypatch, gap):
    graph = _graph()
    samples = iter([1.0, gap, 1.0, 1.0, 0.0])
    monkeypatch.setattr(
        runtime,
        "step",
        lambda graph, **k: graph.graph["history"]["stable_frac"].append(next(samples)),
    )
    runtime.run(graph, 5)
    assert len(graph.graph["history"]["stable_frac"]) == 4


def test_old_telemetry_without_new_samples_cannot_stop_run(monkeypatch):
    graph = _graph()
    graph.graph["history"]["stable_frac"] = [1.0, 1.0]
    calls = []
    monkeypatch.setattr(runtime, "step", lambda *a, **k: calls.append(1))
    runtime.run(graph, 3)
    assert len(calls) == 3


def test_full_bounded_history_stops_after_two_new_stable_observations(monkeypatch):
    graph = _graph()
    graph.add_node(0, EPI=0.0, nu_f=1.0, theta=0.0, delta_nfr=0.0)
    graph.graph["HISTORY_MAXLEN"] = 2
    graph.graph["history"]["stable_frac"] = [0.0, 0.0]
    history = ensure_history(graph)
    original_series = history["stable_frac"]
    calls = []

    def record_observation(graph, **kwargs):
        calls.append(1)
        # Exercise the actual metrics producer and its real bounded HistoryDict.
        _track_stability(graph, history, 1.0, 1e-3, 1e-3)

    monkeypatch.setattr(runtime, "step", record_observation)
    runtime.run(graph, 5)
    assert len(calls) == 2
    assert history["stable_frac"] is original_series
    assert list(original_series) == [1.0, 1.0]
    assert graph.graph["_stability_observation_revision"] == 2


def test_normalizing_retained_history_does_not_create_a_new_observation(monkeypatch):
    graph = _graph()
    graph.graph["HISTORY_MAXLEN"] = 2
    graph.graph["history"]["stable_frac"] = [1.0, 1.0]
    original_series = graph.graph["history"]["stable_frac"]
    calls = []

    def normalize_only(graph, **kwargs):
        calls.append(1)
        ensure_history(graph)

    monkeypatch.setattr(runtime, "step", normalize_only)
    runtime.run(graph, 3)
    assert len(calls) == 3
    assert graph.graph["history"]["stable_frac"] is not original_series
    assert "_stability_observation_revision" not in graph.graph


def test_invalid_producer_revision_rejects_before_metric_commit():
    graph = _graph()
    graph.add_node(0, EPI=0.0, nu_f=1.0, theta=0.0, delta_nfr=0.0)
    graph.graph["_stability_observation_revision"] = True
    before_graph = deepcopy(graph.graph)
    before_nodes = deepcopy(dict(graph.nodes(data=True)))
    with pytest.raises(TypeError, match="observation revision"):
        _track_stability(graph, graph.graph["history"], 1.0, 1e-3, 1e-3)
    assert graph.graph == before_graph
    assert dict(graph.nodes(data=True)) == before_nodes


def test_invalid_run_revision_rejects_before_step(monkeypatch):
    graph = _graph()
    graph.graph["_stability_observation_revision"] = -1
    calls = []
    monkeypatch.setattr(runtime, "step", lambda *a, **k: calls.append(1))
    with pytest.raises(ValueError, match="observation revision"):
        runtime.run(graph, 3)
    assert calls == []


def test_only_tail_is_read_and_prior_valid_window_can_continue(monkeypatch):
    class TailOnly(list):
        def __iter__(self):
            raise AssertionError("must not scan complete retained history")

    graph = _graph()
    series = TailOnly(["unavailable"] * 1000 + [1.0])
    graph.graph["history"]["stable_frac"] = series
    monkeypatch.setattr(runtime, "step", lambda *a, **k: series.append(1.0))
    runtime.run(graph, 3)
    assert len(series) == 1002


def test_disabled_policy_does_not_consume_unused_fields(monkeypatch):
    graph = _graph()
    graph.graph["STOP_EARLY"].update(enabled=False, window="unused", fraction=None)
    calls = []
    monkeypatch.setattr(runtime, "step", lambda *a, **k: calls.append(1))
    runtime.run(graph, 3)
    assert len(calls) == 3


@pytest.mark.parametrize(
    "options", [{"stop_early_window": 2}, {"stop_early_fraction": 0.8}]
)
def test_explicit_cli_stop_option_enables_policy(options):
    graph = nx.Graph()
    inject_defaults(graph)
    assert graph.graph["STOP_EARLY"]["enabled"] is False
    apply_cli_config(graph, Namespace(config=None, **options))
    assert graph.graph["STOP_EARLY"]["enabled"] is True
    window, fraction = runtime._resolve_early_stop(graph)
    assert window == options.get("stop_early_window", 25)
    assert fraction == options.get("stop_early_fraction", 0.9)


@pytest.mark.parametrize("history_maxlen", [0, 2])
def test_real_metrics_callback_stops_after_two_equilibrium_observations(
    history_maxlen, caplog
):
    graph = nx.path_graph(2)
    inject_defaults(graph)
    graph.graph.update(
        STOP_EARLY={"enabled": True, "window": 2, "fraction": 1.0},
        PHASE_ADAPT={"enabled": False},
        HISTORY_MAXLEN=history_maxlen,
        history={"stable_frac": [0.0, 0.0]} if history_maxlen else {},
    )
    for node in graph:
        graph.nodes[node].update(EPI=0.0, nu_f=1.0, theta=0.0, delta_nfr=0.0, Si=0.5)
    register_metrics_callbacks(graph)
    runtime.run(graph, 5, dt=0.125, use_Si=False, apply_glyphs=False)
    assert graph.graph["_t"] == 0.25
    history = graph.graph["history"]
    assert list(history["stable_frac"]) == [1.0, 1.0]
    assert list(history["C_steps"]) == [1.0, 1.0]
    assert list(history["W_bar"]) == [1.0, 1.0]
    assert graph.graph["_stability_observation_revision"] == 2
    assert not [record for record in caplog.records if record.levelno >= 40]
