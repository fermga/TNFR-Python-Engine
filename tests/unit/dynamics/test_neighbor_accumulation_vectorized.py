import time
from contextlib import contextmanager

import networkx as nx
import pytest

from tnfr.alias import set_attr
from tnfr.constants import get_aliases
from tnfr.dynamics.dnfr import (
    _accumulate_neighbors_broadcasted,
    _build_edge_index_arrays,
    _build_neighbor_sums_common,
    _ensure_numpy_state_vectors,
    _init_neighbor_sums,
    _prepare_dnfr_data,
)

ALIAS_THETA = get_aliases("THETA")
ALIAS_EPI = get_aliases("EPI")
ALIAS_VF = get_aliases("VF")


@contextmanager
def numpy_disabled(monkeypatch):
    import tnfr.dynamics.dnfr as dnfr_module

    with monkeypatch.context() as ctx:
        ctx.setattr(dnfr_module, "get_numpy", lambda: None)
        yield


def _dense_weighted_graph(np_module, *, nodes: int, topo_weight: float):
    graph = nx.complete_graph(nodes)
    phases = np_module.linspace(-np_module.pi, np_module.pi, nodes, endpoint=False)
    epi_values = np_module.linspace(0.05, 0.95, nodes)
    vf_values = np_module.linspace(-0.35, 0.35, nodes)

    for idx, node in enumerate(graph.nodes):
        set_attr(graph.nodes[node], ALIAS_THETA, float(phases[idx]))
        set_attr(graph.nodes[node], ALIAS_EPI, float(epi_values[idx]))
        set_attr(graph.nodes[node], ALIAS_VF, float(vf_values[idx]))

    graph.graph["DNFR_WEIGHTS"] = {
        "phase": 0.4,
        "epi": 0.3,
        "vf": 0.2,
        "topo": topo_weight,
    }
    return graph


def _sparse_weighted_graph(np_module, *, nodes: int, topo_weight: float):
    graph = nx.path_graph(nodes)
    phases = np_module.linspace(-np_module.pi / 2, np_module.pi / 2, nodes)
    epi_values = np_module.linspace(-0.2, 0.8, nodes)
    vf_values = np_module.linspace(0.15, 0.55, nodes)

    for idx, node in enumerate(graph.nodes):
        set_attr(graph.nodes[node], ALIAS_THETA, float(phases[idx]))
        set_attr(graph.nodes[node], ALIAS_EPI, float(epi_values[idx]))
        set_attr(graph.nodes[node], ALIAS_VF, float(vf_values[idx]))

    graph.graph["DNFR_WEIGHTS"] = {
        "phase": 0.35,
        "epi": 0.25,
        "vf": 0.25,
        "topo": topo_weight,
    }
    return graph


@pytest.mark.parametrize("topo_weight", [0.0, 0.45])
def test_vectorized_neighbor_sums_match_loop(topo_weight, monkeypatch):
    np_module = pytest.importorskip("numpy")

    vector_graph = _dense_weighted_graph(np_module, nodes=32, topo_weight=topo_weight)
    vector_data = _prepare_dnfr_data(vector_graph)
    vector_result = _build_neighbor_sums_common(
        vector_graph, vector_data, use_numpy=True
    )

    loop_graph = _dense_weighted_graph(np_module, nodes=32, topo_weight=topo_weight)
    with numpy_disabled(monkeypatch):
        loop_data = _prepare_dnfr_data(loop_graph)
        loop_result = _build_neighbor_sums_common(
            loop_graph, loop_data, use_numpy=False
        )

    for vector_arr, loop_arr in zip(vector_result[:-1], loop_result[:-1]):
        if vector_arr is None or loop_arr is None:
            assert vector_arr is loop_arr is None
        else:
            loop_np = np_module.asarray(loop_arr, dtype=float)
            np_module.testing.assert_allclose(vector_arr, loop_np, rtol=1e-9, atol=1e-9)

    vec_degrees = vector_result[-1]
    loop_degrees = loop_result[-1]
    if vec_degrees is None or loop_degrees is None:
        assert vec_degrees is loop_degrees is None
    else:
        np_module.testing.assert_allclose(
            vec_degrees, loop_degrees, rtol=1e-9, atol=1e-9
        )


@pytest.mark.parametrize("topo_weight", [0.0, 0.5])
def test_sparse_broadcast_neighbor_sums_match_loop(topo_weight, monkeypatch):
    np_module = pytest.importorskip("numpy")

    vector_graph = _sparse_weighted_graph(np_module, nodes=48, topo_weight=topo_weight)
    vector_data = _prepare_dnfr_data(vector_graph)
    vector_data["prefer_sparse"] = True
    vector_data["A"] = None
    vector_result = _build_neighbor_sums_common(
        vector_graph, vector_data, use_numpy=True
    )

    loop_graph = _sparse_weighted_graph(np_module, nodes=48, topo_weight=topo_weight)
    with numpy_disabled(monkeypatch):
        loop_data = _prepare_dnfr_data(loop_graph)
        loop_result = _build_neighbor_sums_common(
            loop_graph, loop_data, use_numpy=False
        )

    for vector_arr, loop_arr in zip(vector_result[:-1], loop_result[:-1]):
        if vector_arr is None or loop_arr is None:
            assert vector_arr is loop_arr is None
        else:
            loop_np = np_module.asarray(loop_arr, dtype=float)
            np_module.testing.assert_allclose(vector_arr, loop_np, rtol=1e-9, atol=1e-9)

    vec_degrees = vector_result[-1]
    loop_degrees = loop_result[-1]
    if vec_degrees is None or loop_degrees is None:
        assert vec_degrees is loop_degrees is None
    else:
        np_module.testing.assert_allclose(
            vec_degrees, loop_degrees, rtol=1e-9, atol=1e-9
        )


def test_neighbor_chunking_matches_unchunked():
    np_module = pytest.importorskip("numpy")

    nodes = 96
    topo_weight = 0.35

    chunk_graph = _dense_weighted_graph(np_module, nodes=nodes, topo_weight=topo_weight)
    chunk_data = _prepare_dnfr_data(chunk_graph)
    chunk_data["prefer_sparse"] = True
    chunk_data["A"] = None
    chunk_data["neighbor_chunk_hint"] = 8
    chunk_result = _build_neighbor_sums_common(chunk_graph, chunk_data, use_numpy=True)

    chunk_size = chunk_data.get("neighbor_chunk_size")
    assert isinstance(chunk_size, int)
    assert 1 <= chunk_size <= 8

    edge_workspace = chunk_data.get("neighbor_edge_values_np")
    assert edge_workspace is not None
    accum = chunk_data.get("neighbor_accum_np")
    assert accum is not None
    assert getattr(edge_workspace, "shape", None) == (chunk_size, accum.shape[0])

    baseline_graph = _dense_weighted_graph(
        np_module, nodes=nodes, topo_weight=topo_weight
    )
    baseline_data = _prepare_dnfr_data(baseline_graph)
    baseline_data["prefer_sparse"] = True
    baseline_data["A"] = None
    baseline_data["neighbor_chunk_hint"] = baseline_graph.number_of_edges() * 2
    baseline_result = _build_neighbor_sums_common(
        baseline_graph, baseline_data, use_numpy=True
    )

    for chunk_arr, baseline_arr in zip(chunk_result[:-1], baseline_result[:-1]):
        if chunk_arr is None or baseline_arr is None:
            assert chunk_arr is baseline_arr is None
        else:
            np_module.testing.assert_allclose(
                chunk_arr, baseline_arr, rtol=1e-9, atol=1e-9
            )

    chunk_degrees = chunk_result[-1]
    baseline_degrees = baseline_result[-1]
    if chunk_degrees is None or baseline_degrees is None:
        assert chunk_degrees is baseline_degrees is None
    else:
        np_module.testing.assert_allclose(
            chunk_degrees, baseline_degrees, rtol=1e-9, atol=1e-9
        )


def test_vectorized_neighbor_sums_outperform_loop(monkeypatch):
    np_module = pytest.importorskip("numpy")

    repeats = 6
    vector_graph = _dense_weighted_graph(np_module, nodes=220, topo_weight=0.4)
    vector_data = _prepare_dnfr_data(vector_graph)
    _build_neighbor_sums_common(vector_graph, vector_data, use_numpy=True)

    loop_graph = _dense_weighted_graph(np_module, nodes=220, topo_weight=0.4)
    with numpy_disabled(monkeypatch):
        loop_data = _prepare_dnfr_data(loop_graph)
        _build_neighbor_sums_common(loop_graph, loop_data, use_numpy=False)

        start_loop = time.perf_counter()
        for _ in range(repeats):
            _build_neighbor_sums_common(loop_graph, loop_data, use_numpy=False)
        loop_elapsed = time.perf_counter() - start_loop

    start_vector = time.perf_counter()
    for _ in range(repeats):
        _build_neighbor_sums_common(vector_graph, vector_data, use_numpy=True)
    vector_elapsed = time.perf_counter() - start_vector

    assert vector_elapsed < loop_elapsed
    assert vector_elapsed <= loop_elapsed * 0.9


@pytest.mark.parametrize("topo_weight", [0.0, 0.35])
def test_broadcast_accumulator_matches_without_cache(topo_weight):
    np_module = pytest.importorskip("numpy")

    graph = _sparse_weighted_graph(np_module, nodes=36, topo_weight=topo_weight)
    data_cached = _prepare_dnfr_data(graph)
    data_cached["prefer_sparse"] = True
    data_cached["A"] = None

    state = _ensure_numpy_state_vectors(data_cached, np_module)
    cos = state["cos"]
    sin = state["sin"]
    epi = state["epi"]
    vf = state["vf"]
    assert cos is not None and sin is not None and epi is not None and vf is not None

    nodes = data_cached["nodes"]
    edge_src, edge_dst = _build_edge_index_arrays(
        graph, nodes, data_cached["idx"], np_module
    )
    data_cached["edge_src"] = edge_src
    data_cached["edge_dst"] = edge_dst

    cache = data_cached.get("cache")
    if cache is not None:
        cache.edge_src = edge_src
        cache.edge_dst = edge_dst

    buffers_cached = _init_neighbor_sums(data_cached, np=np_module)
    (
        x_cached,
        y_cached,
        epi_cached,
        vf_cached,
        count_cached,
        deg_sum_cached,
        _,
    ) = buffers_cached

    deg_array = None
    if deg_sum_cached is not None:
        deg_array = np_module.asarray(
            [graph.degree(node) for node in nodes], dtype=float
        )

    cached_result = _accumulate_neighbors_broadcasted(
        edge_src=edge_src,
        edge_dst=edge_dst,
        cos=cos,
        sin=sin,
        epi=epi,
        vf=vf,
        x=x_cached,
        y=y_cached,
        epi_sum=epi_cached,
        vf_sum=vf_cached,
        count=count_cached,
        deg_sum=deg_sum_cached,
        deg_array=deg_array,
        cache=cache,
        np=np_module,
        chunk_size=None,
    )

    data_no_cache = data_cached.copy()
    data_no_cache["cache"] = None
    data_no_cache["neighbor_accum_np"] = None
    data_no_cache["neighbor_edge_values_np"] = None
    data_no_cache["edge_src"] = edge_src
    data_no_cache["edge_dst"] = edge_dst

    (
        x_plain,
        y_plain,
        epi_plain,
        vf_plain,
        count_plain,
        deg_sum_plain,
        _,
    ) = _init_neighbor_sums(data_no_cache, np=np_module)

    deg_array_plain = None
    if deg_array is not None:
        deg_array_plain = np_module.array(deg_array, copy=True)

    plain_result = _accumulate_neighbors_broadcasted(
        edge_src=edge_src,
        edge_dst=edge_dst,
        cos=cos,
        sin=sin,
        epi=epi,
        vf=vf,
        x=x_plain,
        y=y_plain,
        epi_sum=epi_plain,
        vf_sum=vf_plain,
        count=count_plain,
        deg_sum=deg_sum_plain,
        deg_array=deg_array_plain,
        cache=None,
        np=np_module,
        chunk_size=None,
    )

    np_module.testing.assert_allclose(x_cached, x_plain, rtol=1e-12, atol=1e-12)
    np_module.testing.assert_allclose(y_cached, y_plain, rtol=1e-12, atol=1e-12)
    np_module.testing.assert_allclose(
        epi_cached, epi_plain, rtol=1e-12, atol=1e-12
    )
    np_module.testing.assert_allclose(vf_cached, vf_plain, rtol=1e-12, atol=1e-12)

    if count_cached is not None and count_plain is not None:
        np_module.testing.assert_allclose(
            count_cached, count_plain, rtol=1e-12, atol=1e-12
        )

    if deg_sum_cached is not None and deg_sum_plain is not None:
        np_module.testing.assert_allclose(
            deg_sum_cached, deg_sum_plain, rtol=1e-12, atol=1e-12
        )

    assert isinstance(cached_result.get("accumulator"), np_module.ndarray)
    assert isinstance(plain_result.get("accumulator"), np_module.ndarray)
    np_module.testing.assert_allclose(
        cached_result["accumulator"], plain_result["accumulator"], rtol=1e-12, atol=1e-12
    )
