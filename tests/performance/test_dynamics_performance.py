"""Performance regression coverage for ΔNFR computations."""

from __future__ import annotations

import time
from typing import Callable

import networkx as nx
import pytest

np = pytest.importorskip("numpy")
import numpy.testing as npt

from tnfr.alias import collect_attr, get_attr, set_attr
from tnfr.constants import get_aliases
from tnfr.dynamics import _prepare_dnfr_data, default_compute_delta_nfr
from tnfr.dynamics.dnfr import (
    _accumulate_neighbors_numpy,
    _build_edge_index_arrays,
    _init_neighbor_sums,
    _resolve_numpy_degree_array,
)
from tnfr.utils import cached_nodes_and_A

pytestmark = pytest.mark.slow

ALIAS_THETA = get_aliases("THETA")
ALIAS_EPI = get_aliases("EPI")
ALIAS_VF = get_aliases("VF")
ALIAS_DNFR = get_aliases("DNFR")

DNFR_WEIGHTS = {
    "phase": 0.4,
    "epi": 0.3,
    "vf": 0.2,
    "topo": 0.1,
}


def _seed_graph(
    num_nodes: int = 160, edge_probability: float = 0.35, *, seed: int = 42
) -> nx.Graph:
    graph = nx.gnp_random_graph(num_nodes, edge_probability, seed=seed)
    for node in graph.nodes:
        set_attr(graph.nodes[node], ALIAS_THETA, 0.1 * (node + 1))
        set_attr(graph.nodes[node], ALIAS_EPI, 0.05 * (node + 2))
        set_attr(graph.nodes[node], ALIAS_VF, 0.02 * (node + 3))
    graph.graph["DNFR_WEIGHTS"] = dict(DNFR_WEIGHTS)
    return graph


def _naive_prepare(graph: nx.Graph):
    nodes, _ = cached_nodes_and_A(graph, cache_size=1)
    theta = collect_attr(graph, nodes, ALIAS_THETA, 0.0)
    epi = collect_attr(graph, nodes, ALIAS_EPI, 0.0)
    vf = collect_attr(graph, nodes, ALIAS_VF, 0.0)
    return theta, epi, vf


def _legacy_numpy_stack_accumulation(graph, data, *, buffers):
    x, y, epi_sum, vf_sum, count, deg_sum, _ = buffers
    nodes = data["nodes"]
    if not nodes:
        return buffers

    cache = data.get("cache")

    epi = data.get("epi_np")
    if epi is None:
        epi = np.array(data["epi"], dtype=float)
        data["epi_np"] = epi
        if cache is not None:
            cache.epi_np = epi

    cos_th = data.get("cos_theta_np")
    if cos_th is None:
        cos_th = np.array(data["cos_theta"], dtype=float)
        data["cos_theta_np"] = cos_th
        if cache is not None:
            cache.cos_theta_np = cos_th

    sin_th = data.get("sin_theta_np")
    if sin_th is None:
        sin_th = np.array(data["sin_theta"], dtype=float)
        data["sin_theta_np"] = sin_th
        if cache is not None:
            cache.sin_theta_np = sin_th

    vf = data.get("vf_np")
    if vf is None:
        vf = np.array(data["vf"], dtype=float)
        data["vf_np"] = vf
        if cache is not None:
            cache.vf_np = vf

    edge_src = data.get("edge_src")
    edge_dst = data.get("edge_dst")
    if edge_src is None or edge_dst is None:
        edge_src, edge_dst = _build_edge_index_arrays(graph, nodes, data["idx"], np)
        data["edge_src"] = edge_src
        data["edge_dst"] = edge_dst
        if cache is not None:
            cache.edge_src = edge_src
            cache.edge_dst = edge_dst

    count.fill(0.0)
    if edge_src.size:
        np.add.at(count, edge_src, 1.0)

    component_sources = [cos_th, sin_th, epi, vf]
    deg_column = None
    deg_array = None
    if deg_sum is not None:
        deg_sum.fill(0.0)
        deg_array = _resolve_numpy_degree_array(
            data, count if count is not None else None, cache=cache, np=np
        )
        if deg_array is not None:
            deg_column = len(component_sources)
            component_sources.append(deg_array)

    stacked = np.empty((len(nodes), len(component_sources)), dtype=float)
    for col, src_vec in enumerate(component_sources):
        np.copyto(stacked[:, col], src_vec, casting="unsafe")

    accum = np.zeros_like(stacked)
    if edge_src.size:
        edge_values = np.empty((edge_src.size, len(component_sources)), dtype=float)
        np.copyto(edge_values, stacked[edge_dst], casting="unsafe")
        np.add.at(accum, edge_src, edge_values)

    np.copyto(x, accum[:, 0], casting="unsafe")
    np.copyto(y, accum[:, 1], casting="unsafe")
    np.copyto(epi_sum, accum[:, 2], casting="unsafe")
    np.copyto(vf_sum, accum[:, 3], casting="unsafe")
    if deg_column is not None and deg_sum is not None:
        np.copyto(deg_sum, accum[:, deg_column], casting="unsafe")

    return (
        x,
        y,
        epi_sum,
        vf_sum,
        count,
        deg_sum,
        deg_array,
    )


def _measure(runtime_fn: Callable[[], None], loops: int) -> float:
    start = time.perf_counter()
    for _ in range(loops):
        runtime_fn()
    return time.perf_counter() - start


def test_default_compute_delta_nfr_vectorized_is_faster_and_equivalent():
    base_graph = _seed_graph()
    vectorized_graph = base_graph.copy()
    fallback_graph = base_graph.copy()
    fallback_graph.graph["vectorized_dnfr"] = False

    # Warm caches before measuring.
    default_compute_delta_nfr(vectorized_graph)
    default_compute_delta_nfr(fallback_graph)

    loops = 2
    vector_time = _measure(
        lambda: default_compute_delta_nfr(vectorized_graph),
        loops,
    )
    fallback_time = _measure(
        lambda: default_compute_delta_nfr(fallback_graph),
        loops,
    )

    assert vector_time < fallback_time
    assert vector_time <= fallback_time * 0.95

    vector_dnfr = [
        get_attr(vectorized_graph.nodes[n], ALIAS_DNFR, 0.0)
        for n in vectorized_graph.nodes
    ]
    fallback_dnfr = [
        get_attr(fallback_graph.nodes[n], ALIAS_DNFR, 0.0)
        for n in fallback_graph.nodes
    ]
    npt.assert_allclose(vector_dnfr, fallback_dnfr, rtol=1e-9, atol=1e-9)


def test_prepare_dnfr_data_stays_faster_than_naive_collector():
    graph_opt = _seed_graph(seed=7)
    graph_naive = graph_opt.copy()

    loops = 5

    # Warm up caches so both paths reuse prepared buffers.
    _ = _prepare_dnfr_data(graph_opt)
    _ = _prepare_dnfr_data(graph_naive)

    opt_time = _measure(lambda: _prepare_dnfr_data(graph_opt), loops)
    naive_time = _measure(lambda: _naive_prepare(graph_naive), loops)

    assert opt_time < naive_time
    assert opt_time <= naive_time * 0.9

    theta, epi, vf = _naive_prepare(graph_opt)
    optimized_data = _prepare_dnfr_data(graph_opt)
    npt.assert_allclose(optimized_data["theta"], theta, rtol=0.0, atol=0.0)
    npt.assert_allclose(optimized_data["epi"], epi, rtol=0.0, atol=0.0)
    npt.assert_allclose(optimized_data["vf"], vf, rtol=0.0, atol=0.0)


def test_neighbor_accumulation_numpy_outperforms_stack_strategy():
    graph_modern = _seed_graph(seed=9)
    graph_legacy = graph_modern.copy()

    modern_data = _prepare_dnfr_data(graph_modern)
    legacy_data = _prepare_dnfr_data(graph_legacy)

    modern_buffers = _init_neighbor_sums(modern_data, np=np)
    legacy_buffers = _init_neighbor_sums(legacy_data, np=np)

    # Warm up buffers before timing.
    _accumulate_neighbors_numpy(
        graph_modern,
        modern_data,
        np=np,
        **{
            "x": modern_buffers[0],
            "y": modern_buffers[1],
            "epi_sum": modern_buffers[2],
            "vf_sum": modern_buffers[3],
            "count": modern_buffers[4],
            "deg_sum": modern_buffers[5],
        },
    )
    _legacy_numpy_stack_accumulation(graph_legacy, legacy_data, buffers=legacy_buffers)

    loops = 5
    modern_time = _measure(
        lambda: _accumulate_neighbors_numpy(
            graph_modern,
            modern_data,
            x=modern_buffers[0],
            y=modern_buffers[1],
            epi_sum=modern_buffers[2],
            vf_sum=modern_buffers[3],
            count=modern_buffers[4],
            deg_sum=modern_buffers[5],
            np=np,
        ),
        loops,
    )
    legacy_time = _measure(
        lambda: _legacy_numpy_stack_accumulation(
            graph_legacy, legacy_data, buffers=legacy_buffers
        ),
        loops,
    )

    assert modern_time < legacy_time
    assert modern_time <= legacy_time * 0.85

    modern_result = _accumulate_neighbors_numpy(
        graph_modern,
        modern_data,
        x=modern_buffers[0],
        y=modern_buffers[1],
        epi_sum=modern_buffers[2],
        vf_sum=modern_buffers[3],
        count=modern_buffers[4],
        deg_sum=modern_buffers[5],
        np=np,
    )
    legacy_result = _legacy_numpy_stack_accumulation(
        graph_legacy, legacy_data, buffers=legacy_buffers
    )
    for modern_arr, legacy_arr in zip(modern_result, legacy_result):
        if modern_arr is None or legacy_arr is None:
            assert modern_arr is legacy_arr is None
        else:
            npt.assert_allclose(modern_arr, legacy_arr, rtol=1e-9, atol=1e-9)
