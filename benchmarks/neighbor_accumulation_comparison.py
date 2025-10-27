"""Benchmark neighbor accumulation strategies for ΔNFR."""

from __future__ import annotations

import statistics
import time

import networkx as nx

from tnfr.constants import get_aliases
from tnfr.dynamics.dnfr import (
    _accumulate_neighbors_numpy,
    _build_edge_index_arrays,
    _init_neighbor_sums,
    _prepare_dnfr_data,
    _resolve_numpy_degree_array,
)

"""Compare neighbour accumulation kernels for the ΔNFR broadcast path.

This benchmark contrasts the modern single ``np.add.at`` accumulator with the
legacy stack-and-add kernel. On the hosted x86\_64 container (Python 3.11,
NumPy 2.3.4) using the defaults (320 nodes, p=0.65, 5×10 loops) the broadcast
accumulator reached ~0.115 s median versus ~0.166 s for the legacy variant.
"""

try:
    import numpy as np
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency guard
    raise RuntimeError("NumPy is required to run this benchmark") from exc

ALIAS_THETA = get_aliases("THETA")
ALIAS_EPI = get_aliases("EPI")
ALIAS_VF = get_aliases("VF")


def _build_graph(num_nodes: int, edge_probability: float, seed: int) -> nx.Graph:
    """Create a reproducible dense-ish graph with TNFR aliases initialised."""
    graph = nx.gnp_random_graph(num_nodes, edge_probability, seed=seed)
    for node in graph.nodes:
        nd = graph.nodes[node]
        nd[ALIAS_THETA] = 0.1 * (node + 1)
        nd[ALIAS_EPI] = 0.05 * (node + 2)
        nd[ALIAS_VF] = 0.08 * (node + 3)
    graph.graph["DNFR_WEIGHTS"] = {
        "phase": 0.4,
        "epi": 0.3,
        "vf": 0.2,
        "topo": 0.1,
    }
    return graph


def _legacy_numpy_stack_accumulation(G, data, *, buffers):
    """Legacy stack-based neighbour accumulation used as comparison baseline."""

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
        edge_src, edge_dst = _build_edge_index_arrays(G, nodes, data["idx"], np)
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
        deg_array = _resolve_numpy_degree_array(data, count, cache=cache, np=np)
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


def _run_modern(G, data, buffers):
    return _accumulate_neighbors_numpy(
        G,
        data,
        x=buffers[0],
        y=buffers[1],
        epi_sum=buffers[2],
        vf_sum=buffers[3],
        count=buffers[4],
        deg_sum=buffers[5],
        np=np,
    )


def run(
    num_nodes: int = 320,
    edge_probability: float = 0.65,
    repeats: int = 5,
    loops: int = 10,
) -> None:
    """Compare the new accumulation kernel with the legacy stack-based version."""

    modern_times: list[float] = []
    legacy_times: list[float] = []

    for rep in range(repeats):
        base_graph = _build_graph(num_nodes, edge_probability, seed=rep + 1)

        modern_graph = base_graph.copy()
        modern_data = _prepare_dnfr_data(modern_graph)
        modern_buffers = _init_neighbor_sums(modern_data, np=np)

        start = time.perf_counter()
        for _ in range(loops):
            _run_modern(modern_graph, modern_data, modern_buffers)
        modern_times.append(time.perf_counter() - start)

        legacy_graph = base_graph.copy()
        legacy_data = _prepare_dnfr_data(legacy_graph)
        legacy_buffers = _init_neighbor_sums(legacy_data, np=np)

        start = time.perf_counter()
        for _ in range(loops):
            _legacy_numpy_stack_accumulation(
                legacy_graph, legacy_data, buffers=legacy_buffers
            )
        legacy_times.append(time.perf_counter() - start)

        # Validate that both strategies produce the same neighbour sums.
        modern_result = _run_modern(modern_graph, modern_data, modern_buffers)
        legacy_result = _legacy_numpy_stack_accumulation(
            legacy_graph, legacy_data, buffers=legacy_buffers
        )
        for new_arr, old_arr in zip(modern_result, legacy_result):
            if new_arr is None or old_arr is None:
                assert new_arr is old_arr is None
            else:
                np.testing.assert_allclose(new_arr, old_arr, rtol=1e-9, atol=1e-9)

    def _stats(values: list[float]) -> tuple[float, float, float, float]:
        return (
            min(values),
            statistics.median(values),
            sum(values) / len(values),
            max(values),
        )

    modern_stats = _stats(modern_times)
    legacy_stats = _stats(legacy_times)

    print(
        "Neighbor accumulation (modern vs legacy)"
        f" on {num_nodes} nodes (p={edge_probability}):"
    )
    print(
        "modern  best={:.6f}s median={:.6f}s mean={:.6f}s worst={:.6f}s".format(
            *modern_stats
        )
    )
    print(
        "legacy  best={:.6f}s median={:.6f}s mean={:.6f}s worst={:.6f}s".format(
            *legacy_stats
        )
    )


if __name__ == "__main__":
    run()
