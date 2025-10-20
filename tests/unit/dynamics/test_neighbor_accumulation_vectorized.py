import time
from contextlib import contextmanager

import networkx as nx
import pytest

from tnfr.alias import set_attr
from tnfr.constants import get_aliases
from tnfr.dynamics.dnfr import (
    _build_neighbor_sums_common,
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
    vector_result = _build_neighbor_sums_common(vector_graph, vector_data, use_numpy=True)

    loop_graph = _dense_weighted_graph(np_module, nodes=32, topo_weight=topo_weight)
    with numpy_disabled(monkeypatch):
        loop_data = _prepare_dnfr_data(loop_graph)
        loop_result = _build_neighbor_sums_common(loop_graph, loop_data, use_numpy=False)

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
        np_module.testing.assert_allclose(vec_degrees, loop_degrees, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("topo_weight", [0.0, 0.5])
def test_sparse_broadcast_neighbor_sums_match_loop(topo_weight, monkeypatch):
    np_module = pytest.importorskip("numpy")

    vector_graph = _sparse_weighted_graph(np_module, nodes=48, topo_weight=topo_weight)
    vector_data = _prepare_dnfr_data(vector_graph)
    vector_data["prefer_sparse"] = True
    vector_data["A"] = None
    vector_result = _build_neighbor_sums_common(vector_graph, vector_data, use_numpy=True)

    loop_graph = _sparse_weighted_graph(np_module, nodes=48, topo_weight=topo_weight)
    with numpy_disabled(monkeypatch):
        loop_data = _prepare_dnfr_data(loop_graph)
        loop_result = _build_neighbor_sums_common(loop_graph, loop_data, use_numpy=False)

    for vector_arr, loop_arr in zip(vector_result[:-1], loop_result[:-1]):
        if vector_arr is None or loop_arr is None:
            assert vector_arr is loop_arr is None
        else:
            loop_np = np_module.asarray(loop_arr, dtype=float)
            np_module.testing.assert_allclose(
                vector_arr, loop_np, rtol=1e-9, atol=1e-9
            )

    vec_degrees = vector_result[-1]
    loop_degrees = loop_result[-1]
    if vec_degrees is None or loop_degrees is None:
        assert vec_degrees is loop_degrees is None
    else:
        np_module.testing.assert_allclose(
            vec_degrees, loop_degrees, rtol=1e-9, atol=1e-9
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
