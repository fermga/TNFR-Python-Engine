"""Focused tests for centralized runtime pressure refresh."""

from __future__ import annotations

import networkx as nx
import pytest

import tnfr.dynamics as dynamics
from tnfr.dynamics import dnfr, runtime


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


@pytest.mark.parametrize("through_runtime", [False, True])
@pytest.mark.parametrize("inspectable", [False, True])
def test_registered_callback_internal_error_never_replays_graph_writes(
    monkeypatch, through_runtime, inspectable
):
    graph = _graph()

    def broken_callback(live_graph, *, n_jobs=None):
        live_graph.nodes[0]["calls"] = live_graph.nodes[0].get("calls", 0) + 1
        raise TypeError("n_jobs payload failed inside registered callback")

    dnfr.set_delta_nfr_hook(graph, broken_callback)
    if not inspectable:

        def unavailable_signature(_callback):
            raise ValueError("signature unavailable")

        monkeypatch.setattr(dnfr.inspect, "signature", unavailable_signature)
    with pytest.raises(TypeError, match="inside registered callback"):
        if through_runtime:
            runtime._refresh_delta_nfr(graph, n_jobs=2)
        else:
            graph.graph["compute_delta_nfr"](graph, n_jobs=2)
    assert graph.nodes[0]["calls"] == 1
    assert graph.graph["_sel_norms"] == {"stale": True}


@pytest.mark.parametrize("through_runtime", [False, True])
@pytest.mark.parametrize(
    "signature_kind", ["legacy", "keyword", "kwargs", "positional"]
)
def test_registered_hook_keeps_supported_worker_hint_and_calls_once(
    through_runtime, signature_kind
):
    graph = _graph()
    calls = []

    def legacy(live_graph):
        calls.append((live_graph, None))

    def keyword(live_graph, *, n_jobs=None):
        calls.append((live_graph, n_jobs))

    def kwargs(live_graph, **options):
        calls.append((live_graph, options["n_jobs"]))

    def positional(live_graph, n_jobs=None, /):
        calls.append((live_graph, n_jobs))

    callback = dict(
        legacy=legacy, keyword=keyword, kwargs=kwargs, positional=positional
    )[signature_kind]
    dnfr.set_delta_nfr_hook(graph, callback)
    if through_runtime:
        runtime._refresh_delta_nfr(graph, n_jobs=3)
    else:
        graph.graph["compute_delta_nfr"](graph, n_jobs=3)
    expected_hint = 3 if signature_kind in ("keyword", "kwargs") else None
    assert calls == [(graph, expected_hint)]


def test_registered_uninspectable_legacy_hook_retains_binding_fallback(monkeypatch):
    graph = _graph()
    calls = []

    def legacy(live_graph):
        calls.append(live_graph)

    def unavailable_signature(_callback):
        raise ValueError("signature unavailable")

    dnfr.set_delta_nfr_hook(graph, legacy)
    monkeypatch.setattr(dnfr.inspect, "signature", unavailable_signature)
    runtime._refresh_delta_nfr(graph, n_jobs=4)
    assert calls == [graph]


def test_pressure_accumulator_failure_is_not_retried(monkeypatch):
    graph = nx.path_graph(2)
    graph.graph.update(
        vectorized_dnfr=False,
        DNFR_WEIGHTS={"epi": 1.0, "phase": 0.0, "vf": 0.0, "topo": 0.0},
    )
    for node in graph:
        graph.nodes[node].update(EPI=float(node), nu_f=1.0, theta=0.0, delta_nfr=7.0)
    calls = []

    def broken_accumulation(*_args, **_kwargs):
        calls.append(1)
        raise TypeError("n_jobs failed inside accumulation")

    monkeypatch.setattr(dnfr, "_build_neighbor_sums_common", broken_accumulation)
    with pytest.raises(TypeError, match="inside accumulation"):
        dnfr.default_compute_delta_nfr(graph)
    assert calls == [1]
    assert tuple(graph.nodes[node]["delta_nfr"] for node in graph) == (7.0, 7.0)
