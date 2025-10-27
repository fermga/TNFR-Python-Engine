import math
from typing import Any

import pytest

from tnfr.alias import set_attr
from tnfr.constants import get_aliases
from tnfr.metrics.sense_index import compute_Si

ALIAS_THETA = get_aliases("THETA")
ALIAS_VF = get_aliases("VF")
ALIAS_DNFR = get_aliases("DNFR")


def test_compute_Si_vectorized_uses_bulk_helper(monkeypatch, graph_canon):
    np = pytest.importorskip("numpy")

    calls: list[dict[str, Any]] = []

    original_helper = compute_Si.__globals__["neighbor_phase_mean_bulk"]

    def capture_helper(edge_src, edge_dst, **kwargs):
        calls.append(
            {
                "edge_src": np.asarray(edge_src, dtype=np.intp).copy(),
                "edge_dst": np.asarray(edge_dst, dtype=np.intp).copy(),
                "node_count": kwargs.get("node_count"),
            }
        )
        return original_helper(edge_src, edge_dst, **kwargs)

    monkeypatch.setattr(
        "tnfr.metrics.sense_index.neighbor_phase_mean_bulk", capture_helper
    )
    monkeypatch.setattr("tnfr.metrics.sense_index.get_numpy", lambda: np)

    G = graph_canon()
    G.add_edge(1, 2)
    for n in G.nodes:
        set_attr(G.nodes[n], ALIAS_THETA, 0.0)
        set_attr(G.nodes[n], ALIAS_VF, 0.0)
        set_attr(G.nodes[n], ALIAS_DNFR, 0.0)

    compute_Si(G, inplace=False)

    assert len(calls) == 1
    recorded = calls[0]
    assert recorded["node_count"] == G.number_of_nodes()
    assert recorded["edge_src"].shape == recorded["edge_dst"].shape
    assert recorded["edge_dst"].size > 0


def _configure_graph(graph):
    graph.add_nodes_from(range(4))
    graph.add_edges_from(((0, 1), (1, 2), (2, 3)))
    phases = [0.0, math.pi / 5, math.pi / 3, math.pi / 2]
    vf_values = [0.2, 0.5, 0.8, 1.1]
    dnfr_values = [0.1, 0.3, 0.4, 0.6]
    for node in graph.nodes:
        set_attr(graph.nodes[node], ALIAS_THETA, phases[node])
        set_attr(graph.nodes[node], ALIAS_VF, vf_values[node])
        set_attr(graph.nodes[node], ALIAS_DNFR, dnfr_values[node])


def test_compute_Si_vectorized_matches_python(monkeypatch, graph_canon):
    pytest.importorskip("numpy")

    G_vec = graph_canon()
    G_py = graph_canon()
    _configure_graph(G_vec)
    _configure_graph(G_py)

    vectorized = compute_Si(G_vec, inplace=False)

    monkeypatch.setattr("tnfr.metrics.sense_index.get_numpy", lambda: None)
    python = compute_Si(G_py, inplace=False)

    assert set(vectorized) == set(python)
    for node in vectorized:
        assert python[node] == pytest.approx(vectorized[node])


def test_compute_Si_python_parallel_matches(monkeypatch, graph_canon):
    monkeypatch.setattr("tnfr.metrics.sense_index.get_numpy", lambda: None)

    G = graph_canon()
    _configure_graph(G)

    sequential = compute_Si(G, inplace=False, n_jobs=1)
    parallel = compute_Si(G, inplace=False, n_jobs=2)

    assert set(sequential) == set(parallel)
    for node in sequential:
        assert parallel[node] == pytest.approx(sequential[node])


def test_compute_Si_reads_jobs_from_graph(monkeypatch, graph_canon):
    monkeypatch.setattr("tnfr.metrics.sense_index.get_numpy", lambda: None)

    captured = []

    class DummyExecutor:
        def __init__(self, max_workers):
            captured.append(max_workers)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn, *args, **kwargs):
            result = fn(*args, **kwargs)

            class DummyFuture:
                def result(self_inner):
                    return result

            return DummyFuture()

    monkeypatch.setattr("tnfr.metrics.sense_index.ProcessPoolExecutor", DummyExecutor)

    G = graph_canon()
    _configure_graph(G)
    G.graph["SI_N_JOBS"] = "3"

    compute_Si(G, inplace=False)

    assert captured == [3]
