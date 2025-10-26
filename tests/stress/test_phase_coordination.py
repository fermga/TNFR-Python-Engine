"""Stress test coverage for phase coordination when NumPy is unavailable."""

from __future__ import annotations

import copy
import math
import time

import networkx as nx
import pytest

from tnfr.alias import get_theta_attr, set_attr
from tnfr.constants import get_aliases, inject_defaults
from tnfr.dynamics import coordination as coordination_module
from tnfr.dynamics.coordination import coordinate_global_local_phase


class _RecordingExecutor:
    """ProcessPoolExecutor stand-in that records chunk payloads for inspection."""

    def __init__(self, max_workers: int):
        self.max_workers = max_workers
        self.chunks: list[coordination_module.ChunkArgs] = []  # type: ignore[attr-defined]
        _RECORDED_EXECUTORS.append(self)

    def __enter__(self) -> "_RecordingExecutor":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def map(self, func, iterable):
        for item in iterable:
            self.chunks.append(item)
            yield func(item)


_RECORDED_EXECUTORS: list[_RecordingExecutor] = []

ALIAS_THETA = get_aliases("THETA")

pytestmark = [pytest.mark.slow, pytest.mark.stress]


def _build_reproducible_graph(*, seed: int, nodes: int, probability: float) -> nx.Graph:
    """Return a deterministic TNFR graph with canonical theta assignments."""

    graph = nx.gnp_random_graph(nodes, probability, seed=seed)
    inject_defaults(graph)
    graph.graph["history"] = {}

    twopi = 2.0 * math.pi
    for node, data in graph.nodes(data=True):
        base = seed * 17 + int(node)
        theta = ((base * 0.037) % twopi) - math.pi
        set_attr(data, ALIAS_THETA, theta)

    return graph


def _snapshot_theta(graph: nx.Graph) -> dict[int, float]:
    """Return a copy of the theta map ensuring all values are finite."""

    snapshot: dict[int, float] = {}
    for node, data in graph.nodes(data=True):
        theta = get_theta_attr(data)
        assert theta is not None
        value = float(theta)
        assert math.isfinite(value)
        snapshot[int(node)] = value
    return snapshot


@pytest.mark.timeout(30)
def test_coordinate_phase_parallel_matches_sequential(monkeypatch: pytest.MonkeyPatch) -> None:
    """Parallel coordination without NumPy must match sequential results quickly."""

    seed = 2601
    node_count = 240
    probability = 0.065
    workers = 4

    base_graph = _build_reproducible_graph(seed=seed, nodes=node_count, probability=probability)

    sequential_graph = copy.deepcopy(base_graph)
    parallel_graph = copy.deepcopy(base_graph)

    monkeypatch.setattr("tnfr.dynamics.get_numpy", lambda: None)
    monkeypatch.setattr(coordination_module, "get_numpy", lambda: None)
    monkeypatch.setattr(coordination_module, "ProcessPoolExecutor", _RecordingExecutor)
    global _RECORDED_EXECUTORS
    _RECORDED_EXECUTORS = []

    coordinate_global_local_phase(sequential_graph, n_jobs=None)
    sequential_snapshot = _snapshot_theta(sequential_graph)

    start = time.perf_counter()
    coordinate_global_local_phase(parallel_graph, n_jobs=workers)
    elapsed = time.perf_counter() - start

    parallel_snapshot = _snapshot_theta(parallel_graph)
    assert _RECORDED_EXECUTORS, "phase coordination should instantiate a process pool"
    executor_record = _RECORDED_EXECUTORS[-1]
    assert executor_record.max_workers == workers
    chunk_sizes = [len(chunk[0]) for chunk in executor_record.chunks]

    assert elapsed < 30.0

    total_nodes = parallel_graph.number_of_nodes()
    assert total_nodes == node_count >= 200

    expected_chunk_size = max(1, math.ceil(total_nodes / workers))
    expected_chunks = math.ceil(total_nodes / expected_chunk_size)

    assert chunk_sizes, "parallel execution must create worker chunks"
    assert len(chunk_sizes) == expected_chunks
    assert chunk_sizes[0] == expected_chunk_size
    assert all(1 <= size <= expected_chunk_size for size in chunk_sizes)

    assert set(parallel_snapshot) == set(sequential_snapshot)
    for node, theta in sequential_snapshot.items():
        assert parallel_snapshot[node] == pytest.approx(theta, abs=1e-9)
