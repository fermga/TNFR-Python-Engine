"""Tests for dynamics helpers."""

import inspect
import sys
from inspect import Parameter, Signature
from typing import Any

import pytest

from tnfr.alias import get_attr, set_attr
from tnfr.constants import get_aliases
from tnfr.dynamics import (
    _choose_glyph,
    _compute_neighbor_means,
    _init_dnfr_cache,
    _prepare_dnfr,
    _refresh_dnfr_vectors,
    default_glyph_selector,
    run,
)
from tnfr.types import Glyph


@pytest.fixture
def compute_delta_nfr_hook():
    """Fixture that records DNFR hook invocations and populates ΔNFR values."""

    dnfr_alias = get_aliases("DNFR")
    recorded: dict[str, list[Any]] = {"n_jobs": []}

    def hook(graph, **kwargs):
        recorded["n_jobs"].append(kwargs.get("n_jobs"))
        for node in graph.nodes:
            set_attr(graph.nodes[node], dnfr_alias, 0.0)

    return hook, recorded


def test_init_and_refresh_dnfr_cache(graph_canon):
    G = graph_canon()
    for i in range(2):
        G.add_node(i, theta=0.1 * i, EPI=float(i), VF=float(i))
    nodes = list(G.nodes())
    cache, idx, th, epi, vf, _cx, _sx, refreshed = _init_dnfr_cache(
        G, nodes, None, 1, False
    )
    assert refreshed
    _refresh_dnfr_vectors(G, nodes, cache)
    assert th[1] == pytest.approx(0.1)
    cache2, *_rest, refreshed2 = _init_dnfr_cache(G, nodes, cache, 1, False)
    assert not refreshed2
    assert cache2 is cache


def test_compute_neighbor_means_list(graph_canon):
    G = graph_canon()
    G.add_edge(0, 1)
    data = {
        "w_topo": 0.0,
        "theta": [0.0, 0.0],
        "epi": [0.0, 0.0],
        "vf": [0.0, 0.0],
        "cos_theta": [1.0, 1.0],
        "sin_theta": [0.0, 0.0],
        "idx": {0: 0, 1: 1},
        "nodes": [0, 1],
    }
    x = [1.0, 0.0]
    y = [0.0, 0.0]
    epi_sum = [2.0, 0.0]
    vf_sum = [0.0, 0.0]
    count = [1, 0]
    th_bar, epi_bar, vf_bar, deg_bar = _compute_neighbor_means(
        G, data, x=x, y=y, epi_sum=epi_sum, vf_sum=vf_sum, count=count
    )
    assert th_bar[0] == pytest.approx(0.0)
    assert epi_bar[0] == pytest.approx(2.0)
    assert vf_bar[0] == pytest.approx(0.0)
    assert deg_bar is None


def test_choose_glyph_respects_lags(graph_canon):
    G = graph_canon()
    G.add_node(0)

    def selector(G, n):
        return "RA"

    h_al = {0: 2}
    h_en = {0: 0}
    g = _choose_glyph(G, 0, selector, False, h_al, h_en, 1, 5)
    assert g == Glyph.AL
    h_al[0] = 0
    h_en[0] = 6
    g = _choose_glyph(G, 0, selector, False, h_al, h_en, 1, 5)
    assert g == Glyph.EN


def test_run_rejects_negative_steps(graph_canon):
    G = graph_canon()
    with pytest.raises(ValueError):
        run(G, steps=-1)


def test_default_selector_refreshes_norms(graph_canon):
    G = graph_canon()
    G.add_nodes_from((0, 1))

    dnfr_alias = get_aliases("DNFR")
    accel_alias = get_aliases("D2EPI")
    si_alias = get_aliases("SI")

    for node in G.nodes:
        set_attr(G.nodes[node], si_alias, 0.5)

    def assign_metrics(dnfr_map, accel_map):
        def _cb(graph):
            for node, value in dnfr_map.items():
                set_attr(graph.nodes[node], dnfr_alias, value)
            for node, value in accel_map.items():
                set_attr(graph.nodes[node], accel_alias, value)

        return _cb

    G.graph["compute_delta_nfr"] = assign_metrics(
        {0: 10.0, 1: 6.0},
        {0: 8.0, 1: 5.0},
    )
    _prepare_dnfr(G, use_Si=False)
    default_glyph_selector(G, 0)
    norms_initial = G.graph["_sel_norms"]
    assert norms_initial["dnfr_max"] == pytest.approx(10.0)
    assert norms_initial["accel_max"] == pytest.approx(8.0)

    G.graph["compute_delta_nfr"] = assign_metrics(
        {0: 4.0, 1: 2.0},
        {0: 3.0, 1: 1.0},
    )
    _prepare_dnfr(G, use_Si=False)
    default_glyph_selector(G, 0)
    norms_updated = G.graph["_sel_norms"]
    assert norms_updated["dnfr_max"] == pytest.approx(4.0)
    assert norms_updated["accel_max"] == pytest.approx(3.0)

    nd = G.nodes[0]
    dnfr_norm = abs(get_attr(nd, dnfr_alias, 0.0)) / norms_updated["dnfr_max"]
    accel_norm = abs(get_attr(nd, accel_alias, 0.0)) / norms_updated["accel_max"]
    assert dnfr_norm == pytest.approx(1.0)
    assert accel_norm == pytest.approx(1.0)


def test_prepare_dnfr_passes_configured_jobs(monkeypatch, graph_canon):
    G = graph_canon()
    dnfr_alias = get_aliases("DNFR")
    for node in range(2):
        G.add_node(node)
    G.add_edge(0, 1)
    captured = {}

    def fake_compute(graph, data, *, use_numpy=None, n_jobs=None):
        captured["n_jobs"] = n_jobs
        # emulate ΔNFR assignment to keep downstream expectations stable
        for node in graph.nodes:
            set_attr(graph.nodes[node], dnfr_alias, 0.0)

    monkeypatch.setattr("tnfr.dynamics.dnfr._compute_dnfr", fake_compute)
    G.graph["DNFR_N_JOBS"] = "4"
    _prepare_dnfr(G, use_Si=False)
    assert captured["n_jobs"] == 4


def test_prepare_dnfr_supports_hooks_without_jobs_kw(graph_canon):
    G = graph_canon()
    dnfr_alias = get_aliases("DNFR")
    for node in range(2):
        G.add_node(node)
    G.add_edge(0, 1)
    G.graph["DNFR_N_JOBS"] = 3

    calls = {"count": 0}

    def hook(graph):
        calls["count"] += 1
        for node in graph.nodes:
            set_attr(graph.nodes[node], dnfr_alias, 0.0)

    G.graph["compute_delta_nfr"] = hook
    _prepare_dnfr(G, use_Si=False)
    assert calls["count"] == 1


def test_prepare_dnfr_passes_jobs_kw_to_hook(graph_canon, compute_delta_nfr_hook):
    G = graph_canon()
    hook, recorded = compute_delta_nfr_hook
    for node in range(2):
        G.add_node(node)
    G.add_edge(0, 1)

    G.graph["compute_delta_nfr"] = hook
    G.graph["DNFR_N_JOBS"] = 5

    _prepare_dnfr(G, use_Si=False)

    assert recorded["n_jobs"] == [5]


def test_prepare_dnfr_falls_back_when_jobs_kw_rejected(graph_canon):
    G = graph_canon()
    dnfr_alias = get_aliases("DNFR")
    for node in range(2):
        G.add_node(node)
    G.add_edge(0, 1)

    calls: dict[str, Any] = {"count": 0, "kwargs": []}

    def hook(graph, **kwargs):
        calls["count"] += 1
        calls["kwargs"].append(kwargs)
        if "n_jobs" in kwargs:
            raise TypeError("unexpected keyword argument 'n_jobs'")
        for node in graph.nodes:
            set_attr(graph.nodes[node], dnfr_alias, 0.0)

    hook.__signature__ = Signature(  # type: ignore[attr-defined]
        [Parameter("graph", Parameter.POSITIONAL_OR_KEYWORD)]
    )

    G.graph["compute_delta_nfr"] = hook
    G.graph["DNFR_N_JOBS"] = 2
    G.graph["_sel_norms"] = {"dnfr_max": 1.0}

    _prepare_dnfr(G, use_Si=False)

    assert calls["count"] == 2
    assert calls["kwargs"][0] == {"n_jobs": 2}
    assert calls["kwargs"][1] == {}
    assert "_sel_norms" not in G.graph


def test_prepare_dnfr_handles_signature_errors(
    monkeypatch, graph_canon, compute_delta_nfr_hook
):
    G = graph_canon()
    dnfr_alias = get_aliases("DNFR")
    hook, recorded = compute_delta_nfr_hook
    for node in range(2):
        G.add_node(node)
        set_attr(G.nodes[node], dnfr_alias, 1.0)
    G.add_edge(0, 1)

    original_signature = inspect.signature

    def failing_signature(obj, *args, **kwargs):
        if obj is hook:
            raise TypeError("signature boom")
        return original_signature(obj, *args, **kwargs)

    monkeypatch.setattr("tnfr.dynamics.runtime.inspect.signature", failing_signature)

    G.graph["compute_delta_nfr"] = hook
    G.graph["DNFR_N_JOBS"] = "7"
    G.graph["_sel_norms"] = {"dnfr_max": 10.0, "accel_max": 5.0}

    _prepare_dnfr(G, use_Si=False)

    assert recorded["n_jobs"] == [7]
    for node in G.nodes:
        assert get_attr(G.nodes[node], dnfr_alias, None) == pytest.approx(0.0)
    assert "_sel_norms" not in G.graph


def test_prepare_dnfr_reraises_hook_type_error_without_jobs_kw(graph_canon):
    G = graph_canon()
    for node in range(2):
        G.add_node(node)
    G.add_edge(0, 1)

    mutated_key = "_boom_marker"

    def boom_hook(graph):
        graph.graph[mutated_key] = True
        raise TypeError("boom")

    G.graph["compute_delta_nfr"] = boom_hook

    with pytest.raises(TypeError, match="boom"):
        _prepare_dnfr(G, use_Si=False)

    if mutated_key in G.graph:
        del G.graph[mutated_key]


def test_prepare_dnfr_passes_si_jobs_to_compute_si(monkeypatch, graph_canon):
    G = graph_canon()
    dnfr_alias = get_aliases("DNFR")
    for node in range(3):
        G.add_node(node)
    G.add_edges_from(((0, 1), (1, 2)))

    def fake_compute_delta(graph, n_jobs=None):
        for node in graph.nodes:
            set_attr(graph.nodes[node], dnfr_alias, 0.0)

    captured = {}

    def fake_compute_si(graph, *, inplace=True, n_jobs=None):
        captured["n_jobs"] = n_jobs

    G.graph["compute_delta_nfr"] = fake_compute_delta
    G.graph["SI_N_JOBS"] = "5"
    monkeypatch.setattr("tnfr.dynamics.compute_Si", fake_compute_si)

    _prepare_dnfr(G, use_Si=True)

    assert captured["n_jobs"] == 5


def test_prepare_dnfr_falls_back_to_metrics_compute_si(monkeypatch, graph_canon):
    G = graph_canon()
    dnfr_alias = get_aliases("DNFR")
    for node in range(2):
        G.add_node(node)
    G.add_edge(0, 1)

    def fake_compute_delta(graph, *, n_jobs=None):
        for node in graph.nodes:
            set_attr(graph.nodes[node], dnfr_alias, 0.0)

    recorded: dict[str, Any] = {}

    def capture_compute_si(graph, *, inplace=True, n_jobs=None):
        recorded["call"] = {"graph": graph, "inplace": inplace, "n_jobs": n_jobs}

    monkeypatch.setattr(
        "tnfr.metrics.sense_index.compute_Si",
        capture_compute_si,
    )
    monkeypatch.setattr("tnfr.dynamics.runtime.compute_Si", capture_compute_si)

    original_module = sys.modules.pop("tnfr.dynamics", None)
    try:
        G.graph["compute_delta_nfr"] = fake_compute_delta
        G.graph["SI_N_JOBS"] = "6"

        _prepare_dnfr(G, use_Si=True)
    finally:
        if original_module is not None:
            sys.modules["tnfr.dynamics"] = original_module

    assert recorded["call"]["graph"] is G
    assert recorded["call"]["inplace"] is True
    assert recorded["call"]["n_jobs"] == 6
