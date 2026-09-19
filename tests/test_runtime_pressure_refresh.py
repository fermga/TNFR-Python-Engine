"""Focused tests for centralized runtime pressure refresh."""

from __future__ import annotations

import networkx as nx
import pytest

import tnfr.dynamics as dynamics
from tnfr.dynamics import runtime


def _graph() -> nx.Graph:
    graph = nx.Graph()
    graph.add_node(0)
    graph.graph["_sel_norms"] = {"stale": True}
    return graph


def test_refresh_uses_default_callback_and_returns_it(monkeypatch) -> None:
    graph = _graph()
    calls = []

    def default_callback(live_graph, *, n_jobs=None):
        calls.append((live_graph, n_jobs))

    monkeypatch.setattr(runtime, "default_compute_delta_nfr", default_callback)

    callback = runtime._refresh_delta_nfr(graph, n_jobs=3)

    assert callback is default_callback
    assert calls == [(graph, 3)]
    assert "_sel_norms" not in graph.graph


def test_refresh_calls_legacy_callback_without_keyword() -> None:
    graph = _graph()
    calls = []

    def legacy_callback(live_graph):
        calls.append(live_graph)

    graph.graph["compute_delta_nfr"] = legacy_callback

    callback = runtime._refresh_delta_nfr(graph, n_jobs=4)

    assert callback is legacy_callback
    assert calls == [graph]
    assert "_sel_norms" not in graph.graph


def test_refresh_propagates_internal_type_error_without_retry() -> None:
    graph = _graph()
    calls = 0

    def broken_callback(live_graph, *, n_jobs=None):
        nonlocal calls
        calls += 1
        raise TypeError("n_jobs failed inside pressure computation")

    graph.graph["compute_delta_nfr"] = broken_callback

    with pytest.raises(TypeError, match="inside pressure computation"):
        runtime._refresh_delta_nfr(graph, n_jobs=2)

    assert calls == 1
    assert graph.graph["_sel_norms"] == {"stale": True}


def test_uninspectable_legacy_callback_retains_no_keyword_fallback(
    monkeypatch,
) -> None:
    graph = _graph()
    calls = []

    def legacy_callback(live_graph):
        calls.append(live_graph)

    graph.graph["compute_delta_nfr"] = legacy_callback

    def unavailable_signature(_callback):
        raise ValueError("signature unavailable")

    monkeypatch.setattr(runtime.inspect, "signature", unavailable_signature)

    callback = runtime._refresh_delta_nfr(graph, n_jobs=5)

    assert callback is legacy_callback
    assert calls == [graph]
    assert "_sel_norms" not in graph.graph


def test_uninspectable_callback_internal_type_error_is_not_retried(
    monkeypatch,
) -> None:
    graph = _graph()
    calls = 0

    def broken_callback(live_graph, **kwargs):
        nonlocal calls
        calls += 1
        raise TypeError("n_jobs is invalid inside callback")

    graph.graph["compute_delta_nfr"] = broken_callback

    def unavailable_signature(_callback):
        raise ValueError("signature unavailable")

    monkeypatch.setattr(runtime.inspect, "signature", unavailable_signature)

    with pytest.raises(TypeError, match="inside callback"):
        runtime._refresh_delta_nfr(graph, n_jobs=1)

    assert calls == 1
    assert graph.graph["_sel_norms"] == {"stale": True}


def test_prepare_dnfr_preserves_si_dispatch(monkeypatch) -> None:
    graph = _graph()
    graph.graph.update(DNFR_N_JOBS=2, SI_N_JOBS=3)
    calls = []

    def refresh(live_graph, *, n_jobs):
        calls.append(("dnfr", live_graph, n_jobs))
        return object()

    def compute_si(live_graph, *, inplace, n_jobs):
        calls.append(("si", live_graph, inplace, n_jobs))

    monkeypatch.setattr(runtime, "_refresh_delta_nfr", refresh)
    monkeypatch.setattr(dynamics, "compute_Si", compute_si)

    runtime._prepare_dnfr(graph, use_Si=True)

    assert calls == [
        ("dnfr", graph, 2),
        ("si", graph, True, 3),
    ]
