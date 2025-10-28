"""Tests for dynamics helpers."""

import inspect
import sys
import time
from inspect import Parameter, Signature
from typing import Any

import networkx as nx
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
from tnfr.dynamics.dnfr import _accumulate_neighbors_broadcasted
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


def test_compute_dnfr_common_numpy_matches_python(monkeypatch):
    np = pytest.importorskip("numpy")
    del np

    import tnfr.dynamics.dnfr as dnfr_module

    base = nx.path_graph(7)
    theta_alias = get_aliases("THETA")
    epi_alias = get_aliases("EPI")
    vf_alias = get_aliases("VF")
    for idx, node in enumerate(base.nodes):
        set_attr(base.nodes[node], theta_alias, 0.2 * (idx + 1))
        set_attr(base.nodes[node], epi_alias, 0.5 * (idx + 1))
        set_attr(base.nodes[node], vf_alias, 0.1 * (idx + 1))
    base.graph["DNFR_WEIGHTS"] = {"phase": 0.4, "epi": 0.3, "vf": 0.2, "topo": 0.1}

    vector_graph = base.copy()
    vector_data = dnfr_module._prepare_dnfr_data(vector_graph)
    dnfr_module._compute_dnfr(vector_graph, vector_data)

    dnfr_alias = get_aliases("DNFR")
    vector_values = [
        get_attr(vector_graph.nodes[node], dnfr_alias, 0.0)
        for node in vector_graph.nodes
    ]

    with monkeypatch.context() as ctx:
        ctx.setattr(dnfr_module, "get_numpy", lambda: None)
        python_graph = base.copy()
        python_data = dnfr_module._prepare_dnfr_data(python_graph)
        dnfr_module._compute_dnfr(python_graph, python_data)

    python_values = [
        get_attr(python_graph.nodes[node], dnfr_alias, 0.0)
        for node in python_graph.nodes
    ]

    assert vector_values == pytest.approx(python_values, rel=1e-9, abs=1e-9)


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


def _legacy_broadcast_accumulate(
    np_module,
    *,
    edge_src,
    edge_dst,
    cos,
    sin,
    epi,
    vf,
    x,
    y,
    epi_sum,
    vf_sum,
    count,
    deg_sum,
    deg_array,
    chunk_size,
):
    n = x.shape[0]
    include_count = count is not None
    use_topology = deg_sum is not None and deg_array is not None
    component_rows = 4 + (1 if include_count else 0) + (1 if use_topology else 0)
    accum = np_module.zeros((component_rows, n), dtype=float)

    edge_count = int(edge_src.size)
    if edge_count:
        edge_src_int = edge_src.astype(np_module.intp, copy=False)
        edge_dst_int = edge_dst.astype(np_module.intp, copy=False)

        if chunk_size is None:
            resolved_chunk = edge_count
        else:
            try:
                resolved_chunk = int(chunk_size)
            except (TypeError, ValueError):
                resolved_chunk = edge_count
            else:
                if resolved_chunk <= 0:
                    resolved_chunk = edge_count
        resolved_chunk = max(1, min(edge_count, resolved_chunk))
        component_indices = np_module.arange(component_rows, dtype=np_module.intp)
        flat_length = component_rows * n

        row = 0
        cos_row = row
        row += 1
        sin_row = row
        row += 1
        epi_row = row
        row += 1
        vf_row = row
        row += 1
        count_row = row if include_count else None
        if count_row is not None:
            row += 1
        deg_row = row if use_topology else None

        for start in range(0, edge_count, resolved_chunk):
            end = min(start + resolved_chunk, edge_count)
            if start >= end:
                continue
            src_slice = edge_src_int[start:end]
            dst_slice = edge_dst_int[start:end]
            slice_len = end - start

            chunk_values = np_module.empty((slice_len, component_rows), dtype=float)

            np_module.take(cos, dst_slice, out=chunk_values[:, cos_row])
            np_module.take(sin, dst_slice, out=chunk_values[:, sin_row])
            np_module.take(epi, dst_slice, out=chunk_values[:, epi_row])
            np_module.take(vf, dst_slice, out=chunk_values[:, vf_row])

            if count_row is not None:
                chunk_values[:, count_row].fill(1.0)

            if deg_row is not None and deg_array is not None:
                np_module.take(deg_array, dst_slice, out=chunk_values[:, deg_row])

            repeated_src = np_module.repeat(src_slice, component_rows)
            repeated_components = np_module.tile(component_indices, slice_len)
            combined_index = repeated_src * component_rows + repeated_components
            weights = chunk_values.reshape(-1)

            chunk_accum = np_module.bincount(
                combined_index,
                weights=weights,
                minlength=flat_length,
            )
            accum += chunk_accum.reshape(n, component_rows).T

    row = 0
    np_module.copyto(x, accum[row], casting="unsafe")
    row += 1
    np_module.copyto(y, accum[row], casting="unsafe")
    row += 1
    np_module.copyto(epi_sum, accum[row], casting="unsafe")
    row += 1
    np_module.copyto(vf_sum, accum[row], casting="unsafe")
    row += 1

    if include_count and count is not None:
        np_module.copyto(count, accum[row], casting="unsafe")
        row += 1

    if use_topology and deg_sum is not None:
        np_module.copyto(deg_sum, accum[row], casting="unsafe")

    return {
        "accumulator": accum,
        "edge_values": None,
    }


def test_broadcast_accumulator_matches_legacy_and_speed():
    np_module = pytest.importorskip("numpy")
    rng = np_module.random.default_rng(1337)

    n = 96
    edge_count = 720
    edge_src = rng.integers(0, n, size=edge_count, dtype=np_module.int64)
    edge_dst = rng.integers(0, n, size=edge_count, dtype=np_module.int64)

    cos = rng.standard_normal(n)
    sin = rng.standard_normal(n)
    epi = rng.random(n)
    vf = rng.standard_normal(n)
    deg_array = rng.uniform(0.25, 3.0, size=n)

    x_new = np_module.zeros(n, dtype=float)
    y_new = np_module.zeros(n, dtype=float)
    epi_new = np_module.zeros(n, dtype=float)
    vf_new = np_module.zeros(n, dtype=float)
    count_new = np_module.zeros(n, dtype=float)
    deg_sum_new = np_module.zeros(n, dtype=float)

    x_old = np_module.zeros_like(x_new)
    y_old = np_module.zeros_like(y_new)
    epi_old = np_module.zeros_like(epi_new)
    vf_old = np_module.zeros_like(vf_new)
    count_old = np_module.zeros_like(count_new)
    deg_sum_old = np_module.zeros_like(deg_sum_new)

    chunk_size = 128

    def run_new():
        x_new.fill(0.0)
        y_new.fill(0.0)
        epi_new.fill(0.0)
        vf_new.fill(0.0)
        count_new.fill(0.0)
        deg_sum_new.fill(0.0)
        return _accumulate_neighbors_broadcasted(
            edge_src=edge_src,
            edge_dst=edge_dst,
            cos=cos,
            sin=sin,
            epi=epi,
            vf=vf,
            x=x_new,
            y=y_new,
            epi_sum=epi_new,
            vf_sum=vf_new,
            count=count_new,
            deg_sum=deg_sum_new,
            deg_array=deg_array,
            cache=None,
            np=np_module,
            chunk_size=chunk_size,
        )

    def run_old():
        x_old.fill(0.0)
        y_old.fill(0.0)
        epi_old.fill(0.0)
        vf_old.fill(0.0)
        count_old.fill(0.0)
        deg_sum_old.fill(0.0)
        return _legacy_broadcast_accumulate(
            np_module,
            edge_src=edge_src,
            edge_dst=edge_dst,
            cos=cos,
            sin=sin,
            epi=epi,
            vf=vf,
            x=x_old,
            y=y_old,
            epi_sum=epi_old,
            vf_sum=vf_old,
            count=count_old,
            deg_sum=deg_sum_old,
            deg_array=deg_array,
            chunk_size=chunk_size,
        )

    legacy = run_old()
    modern = run_new()

    np_module.testing.assert_allclose(x_new, x_old, rtol=1e-12, atol=1e-12)
    np_module.testing.assert_allclose(y_new, y_old, rtol=1e-12, atol=1e-12)
    np_module.testing.assert_allclose(epi_new, epi_old, rtol=1e-12, atol=1e-12)
    np_module.testing.assert_allclose(vf_new, vf_old, rtol=1e-12, atol=1e-12)
    np_module.testing.assert_allclose(count_new, count_old, rtol=1e-12, atol=1e-12)
    np_module.testing.assert_allclose(
        deg_sum_new, deg_sum_old, rtol=1e-12, atol=1e-12
    )

    np_module.testing.assert_allclose(
        modern["accumulator"], legacy["accumulator"], rtol=1e-12, atol=1e-12
    )

    repeats = 6
    for _ in range(2):
        run_new()
        run_old()

    start_new = time.perf_counter()
    for _ in range(repeats):
        run_new()
    new_elapsed = time.perf_counter() - start_new

    start_old = time.perf_counter()
    for _ in range(repeats):
        run_old()
    old_elapsed = time.perf_counter() - start_old

    assert new_elapsed <= old_elapsed * 1.1
