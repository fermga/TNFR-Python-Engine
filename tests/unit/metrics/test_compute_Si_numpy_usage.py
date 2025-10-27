import math

import pytest

from tnfr.alias import set_attr
from tnfr.constants import get_aliases
from tnfr.metrics.sense_index import compute_Si

ALIAS_THETA = get_aliases("THETA")
ALIAS_VF = get_aliases("VF")
ALIAS_DNFR = get_aliases("DNFR")


def test_compute_Si_vectorized_accumulates_neighbors(monkeypatch, graph_canon):
    np = pytest.importorskip("numpy")

    calls: list[np.ndarray] = []

    original_bincount = np.bincount

    def capture_bincount(*args, **kwargs):
        calls.append(args[0])
        return original_bincount(*args, **kwargs)

    monkeypatch.setattr(np, "bincount", capture_bincount)
    monkeypatch.setattr("tnfr.metrics.sense_index.get_numpy", lambda: np)

    G = graph_canon()
    G.add_edge(1, 2)
    for n in G.nodes:
        set_attr(G.nodes[n], ALIAS_THETA, 0.0)
        set_attr(G.nodes[n], ALIAS_VF, 0.0)
        set_attr(G.nodes[n], ALIAS_DNFR, 0.0)

    compute_Si(G, inplace=False)

    assert len(calls) == 1


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
