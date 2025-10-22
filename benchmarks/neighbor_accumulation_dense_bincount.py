"""Benchmark dense-graph neighbour accumulation (bincount vs np.add.at)."""

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

try:
    import numpy as np
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency guard
    raise RuntimeError("NumPy is required to run this benchmark") from exc


ALIAS_THETA = get_aliases("THETA")
ALIAS_EPI = get_aliases("EPI")
ALIAS_VF = get_aliases("VF")


def _build_dense_graph(num_nodes: int, *, seed: int, probability: float) -> nx.Graph:
    """Create a dense random graph with deterministic TNFR state."""

    graph = nx.gnp_random_graph(num_nodes, probability, seed=seed)
    for node in graph.nodes:
        attr = graph.nodes[node]
        attr[ALIAS_THETA] = 0.03 * (node + 1)
        attr[ALIAS_EPI] = 0.11 * (node + 2)
        attr[ALIAS_VF] = 0.09 * (node + 3)
    graph.graph["DNFR_WEIGHTS"] = {
        "phase": 0.45,
        "epi": 0.25,
        "vf": 0.2,
        "topo": 0.1,
    }
    return graph


def _legacy_add_at_accumulation(G, data, *, buffers):
    """Reference implementation using ``np.add.at`` for accumulation."""

    x, y, epi_sum, vf_sum, count, deg_sum, _ = buffers
    nodes = data["nodes"]
    if not nodes:
        return buffers

    cache = data.get("cache")
    state = {
        key: data.get(key)
        for key in ("cos_theta_np", "sin_theta_np", "epi_np", "vf_np")
    }
    for key, fallback in (
        ("cos_theta_np", "cos_theta"),
        ("sin_theta_np", "sin_theta"),
        ("epi_np", "epi"),
        ("vf_np", "vf"),
    ):
        if state[key] is None:
            state[key] = np.array(data[fallback], dtype=float)
            data[key] = state[key]
            if cache is not None:
                setattr(cache, key, state[key])

    edge_src = data.get("edge_src")
    edge_dst = data.get("edge_dst")
    if edge_src is None or edge_dst is None:
        edge_src, edge_dst = _build_edge_index_arrays(G, nodes, data["idx"], np)
        data["edge_src"] = edge_src
        data["edge_dst"] = edge_dst
        if cache is not None:
            cache.edge_src = edge_src
            cache.edge_dst = edge_dst

    x.fill(0.0)
    y.fill(0.0)
    epi_sum.fill(0.0)
    vf_sum.fill(0.0)
    count.fill(0.0)
    deg_array = None
    if deg_sum is not None:
        deg_sum.fill(0.0)
        deg_array = _resolve_numpy_degree_array(data, count, cache=cache, np=np)

    if edge_src.size:
        stacked = np.empty(
            (edge_src.size, 4 + (1 if deg_array is not None else 0)), dtype=float
        )
        np.take(state["cos_theta_np"], edge_dst, out=stacked[:, 0])
        np.take(state["sin_theta_np"], edge_dst, out=stacked[:, 1])
        np.take(state["epi_np"], edge_dst, out=stacked[:, 2])
        np.take(state["vf_np"], edge_dst, out=stacked[:, 3])
        np.add.at(count, edge_src, 1.0)
        np.add.at(x, edge_src, stacked[:, 0])
        np.add.at(y, edge_src, stacked[:, 1])
        np.add.at(epi_sum, edge_src, stacked[:, 2])
        np.add.at(vf_sum, edge_src, stacked[:, 3])
        if deg_array is not None and deg_sum is not None:
            np.take(deg_array, edge_dst, out=stacked[:, -1])
            np.add.at(deg_sum, edge_src, stacked[:, -1])
    else:
        if deg_sum is not None:
            deg_sum.fill(0.0)

    return buffers


def _run_bincount(G, data, buffers):
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
    num_nodes: int = 360,
    edge_probability: float = 0.7,
    repeats: int = 5,
    loops: int = 8,
) -> None:
    """Time bincount-based accumulation versus the legacy add.at path."""

    bincount_times: list[float] = []
    add_at_times: list[float] = []

    for rep in range(repeats):
        base = _build_dense_graph(
            num_nodes, seed=rep + 42, probability=edge_probability
        )

        modern_graph = base.copy()
        modern_data = _prepare_dnfr_data(modern_graph)
        modern_buffers = _init_neighbor_sums(modern_data, np=np)

        start = time.perf_counter()
        for _ in range(loops):
            _run_bincount(modern_graph, modern_data, modern_buffers)
        bincount_times.append(time.perf_counter() - start)

        legacy_graph = base.copy()
        legacy_data = _prepare_dnfr_data(legacy_graph)
        legacy_buffers = _init_neighbor_sums(legacy_data, np=np)

        start = time.perf_counter()
        for _ in range(loops):
            _legacy_add_at_accumulation(
                legacy_graph, legacy_data, buffers=legacy_buffers
            )
        add_at_times.append(time.perf_counter() - start)

        modern_result = _run_bincount(modern_graph, modern_data, modern_buffers)
        legacy_result = _legacy_add_at_accumulation(
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

    bincount_stats = _stats(bincount_times)
    add_at_stats = _stats(add_at_times)

    if bincount_stats[1] >= add_at_stats[1]:  # pragma: no cover - guard rail
        raise RuntimeError(
            "Bincount accumulation did not outperform the np.add.at baseline"
        )

    print(
        "Dense neighbor accumulation (bincount vs add.at)"
        f" on {num_nodes} nodes (p={edge_probability}):"
    )
    print(
        "bincount best={:.6f}s median={:.6f}s mean={:.6f}s worst={:.6f}s".format(
            *bincount_stats
        )
    )
    print(
        "add.at   best={:.6f}s median={:.6f}s mean={:.6f}s worst={:.6f}s".format(
            *add_at_stats
        )
    )


if __name__ == "__main__":
    run()
