import sys

import networkx as nx
import pytest

import tnfr.dynamics.dnfr as dnfr_module
from tnfr.alias import collect_attr, set_attr
from tnfr.constants import get_aliases
from tnfr.dynamics.dnfr import (
    _build_neighbor_sums_common,
    _compute_dnfr,
    _compute_dnfr_common,
    _prepare_dnfr_data,
)

ALIAS_THETA = get_aliases("THETA")
ALIAS_EPI = get_aliases("EPI")
ALIAS_VF = get_aliases("VF")
ALIAS_DNFR = get_aliases("DNFR")


def _graph_fixture(size: int = 4) -> nx.Graph:
    G = nx.path_graph(size)
    for n in G.nodes:
        set_attr(G.nodes[n], ALIAS_THETA, 0.1 * (n + 1))
        set_attr(G.nodes[n], ALIAS_EPI, 0.2 * (n + 1))
        set_attr(G.nodes[n], ALIAS_VF, 0.3 * (n + 1))
    G.graph["DNFR_WEIGHTS"] = {
        "phase": 0.4,
        "epi": 0.3,
        "vf": 0.2,
        "topo": 0.1,
    }
    return G


def _assert_loop_state(data):
    cache = data.get("cache")
    assert data.get("edge_src") is None
    assert data.get("neighbor_edge_values_np") is None
    if cache is not None:
        assert cache.edge_src is None
        assert cache.neighbor_edge_values_np is None
        assert cache.neighbor_accum_np is None


def _assert_vector_state(data, np):
    accum = data.get("neighbor_accum_np")
    assert accum is not None
    assert isinstance(accum, np.ndarray)
    assert accum.ndim == 2
    cache = data.get("cache")
    if cache is not None:
        assert cache.neighbor_accum_np is accum
        assert cache.edge_src is not None
        edge_values = cache.neighbor_edge_values_np
        assert data.get("neighbor_edge_values_np") is edge_values
        edge_count = data.get("edge_count", 0)
        if edge_count:
            assert isinstance(edge_values, np.ndarray)
            chunk_size = data.get("neighbor_chunk_size") or edge_count
            assert edge_values.shape == (chunk_size, accum.shape[0])
        else:
            assert edge_values is None


def test_compute_dnfr_uses_numpy_even_when_graph_disables_vectorization():
    np = pytest.importorskip("numpy")

    G = _graph_fixture()
    G.graph["vectorized_dnfr"] = False
    data = _prepare_dnfr_data(G)

    data["prefer_sparse"] = True
    data["A"] = None

    _compute_dnfr(G, data)

    _assert_vector_state(data, np)
    dnfr_values = collect_attr(G, G.nodes, ALIAS_DNFR, 0.0)
    assert all(isinstance(val, float) for val in dnfr_values)


def test_compute_dnfr_reuses_cached_numpy_when_flag_disabled_again():
    np = pytest.importorskip("numpy")

    G = _graph_fixture()
    data = _prepare_dnfr_data(G)

    data["prefer_sparse"] = True
    data["A"] = None

    _compute_dnfr(G, data)
    cached_accum = data.get("neighbor_accum_np")
    assert cached_accum is not None

    G.graph["vectorized_dnfr"] = False
    _compute_dnfr(G, data)

    _assert_vector_state(data, np)
    assert data.get("neighbor_accum_np") is cached_accum
    dnfr_values = collect_attr(G, G.nodes, ALIAS_DNFR, 0.0)
    assert all(isinstance(val, float) for val in dnfr_values)


def test_compute_dnfr_prefers_numpy_even_when_requested_off(monkeypatch):
    np = pytest.importorskip("numpy")

    G = _graph_fixture()
    data = _prepare_dnfr_data(G)

    data["prefer_sparse"] = True
    data["A"] = None

    calls = {"numpy": 0}

    original_numpy = dnfr_module._accumulate_neighbors_numpy

    def _spy_numpy(*args, **kwargs):
        calls["numpy"] += 1
        return original_numpy(*args, **kwargs)

    monkeypatch.setattr(dnfr_module, "_accumulate_neighbors_numpy", _spy_numpy)

    _compute_dnfr(G, data, use_numpy=False)

    assert calls["numpy"] >= 1
    _assert_vector_state(data, np)
    dnfr_values = collect_attr(G, G.nodes, ALIAS_DNFR, 0.0)
    assert all(isinstance(val, float) for val in dnfr_values)


def test_compute_dnfr_loop_path_without_numpy_module(monkeypatch):
    pytest.importorskip("numpy")

    monkeypatch.delitem(sys.modules, "numpy", raising=False)
    monkeypatch.setattr(dnfr_module, "get_numpy", lambda: None)
    monkeypatch.setattr(
        dnfr_module, "_has_cached_numpy_buffers", lambda *_, **__: False
    )

    G = _graph_fixture()
    data = _prepare_dnfr_data(G)

    _compute_dnfr(G, data)

    _assert_loop_state(data)
    dnfr_values = collect_attr(G, G.nodes, ALIAS_DNFR, 0.0)
    assert all(isinstance(val, float) for val in dnfr_values)


def test_build_neighbor_sums_uses_cached_numpy_when_get_numpy_none(monkeypatch):
    np = pytest.importorskip("numpy")

    G = _graph_fixture()
    data = _prepare_dnfr_data(G)

    data["prefer_sparse"] = True
    data["A"] = None

    _compute_dnfr(G, data)

    baseline_dnfr = list(collect_attr(G, G.nodes, ALIAS_DNFR, 0.0))

    baseline_stats = _build_neighbor_sums_common(G, data, use_numpy=True)
    assert baseline_stats is not None
    baseline_arrays = []
    for arr in baseline_stats[:-1]:
        if arr is None:
            baseline_arrays.append(None)
        else:
            baseline_arrays.append(np.array(arr, copy=True))

    baseline_degs = baseline_stats[-1]
    if baseline_degs is not None:
        baseline_degs_copy = np.array(baseline_degs, copy=True)
    else:
        baseline_degs_copy = None

    accum = data.get("neighbor_accum_np")
    assert isinstance(accum, np.ndarray)
    cache = data.get("cache")
    assert cache is not None
    assert cache.neighbor_accum_np is accum

    numpy_module = sys.modules.get("numpy")
    assert numpy_module is np

    monkeypatch.setattr(dnfr_module, "get_numpy", lambda: None)

    fallback_stats = _build_neighbor_sums_common(G, data, use_numpy=True)
    assert fallback_stats is not None

    fallback_arrays = fallback_stats[:-1]
    for baseline_arr, fallback_arr in zip(baseline_arrays, fallback_arrays):
        if baseline_arr is None:
            assert fallback_arr is None
        else:
            assert isinstance(fallback_arr, np.ndarray)
            np.testing.assert_allclose(fallback_arr, baseline_arr)

    fallback_degs = fallback_stats[-1]
    if baseline_degs_copy is None:
        assert fallback_degs is None
    else:
        assert isinstance(fallback_degs, np.ndarray)
        np.testing.assert_allclose(fallback_degs, baseline_degs_copy)

    accum_after = data.get("neighbor_accum_np")
    assert isinstance(accum_after, np.ndarray)
    assert cache.neighbor_accum_np is accum_after
    edge_workspace = data.get("neighbor_edge_values_np")
    edge_count = data.get("edge_count", 0)
    if edge_count:
        assert isinstance(edge_workspace, np.ndarray)
        chunk_size = data.get("neighbor_chunk_size") or edge_count
        expected_rows = accum_after.shape[0]
        assert edge_workspace.shape == (chunk_size, expected_rows)
    else:
        assert edge_workspace is None

    _compute_dnfr_common(
        G,
        data,
        x=fallback_stats[0],
        y=fallback_stats[1],
        epi_sum=fallback_stats[2],
        vf_sum=fallback_stats[3],
        count=fallback_stats[4],
        deg_sum=fallback_stats[5],
        degs=fallback_stats[6],
    )

    fallback_dnfr = list(collect_attr(G, G.nodes, ALIAS_DNFR, 0.0))
    np.testing.assert_allclose(fallback_dnfr, baseline_dnfr)
