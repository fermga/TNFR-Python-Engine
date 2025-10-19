import math

import pytest

from tnfr.alias import set_attr
from tnfr.constants import get_aliases
from tnfr.metrics.sense_index import compute_Si
import tnfr.utils.init as utils_init

ALIAS_THETA = get_aliases("THETA")
ALIAS_VF = get_aliases("VF")
ALIAS_DNFR = get_aliases("DNFR")


def test_compute_Si_uses_module_numpy_and_propagates(monkeypatch, graph_canon):
    class DummyNP:
        def fromiter(self, iterable, dtype=float, count=-1):
            _ = dtype
            return list(iterable)

        def cos(self, arr):
            return [math.cos(x) for x in arr]

        def sin(self, arr):
            return [math.sin(x) for x in arr]

    sentinel = DummyNP()

    captured = []

    def fake_neighbor_phase_mean_list(
        _neigh, cos_th, sin_th, np=None, fallback=0.0
    ):
        captured.append(np)
        return 0.0

    monkeypatch.setattr(
        utils_init,
        "cached_import",
        lambda module, attr=None, **kwargs: sentinel if module == "numpy" else None,
    )
    monkeypatch.setattr(
        "tnfr.metrics.sense_index.neighbor_phase_mean_list",
        fake_neighbor_phase_mean_list,
    )

    G = graph_canon()
    G.add_edge(1, 2)
    for n in G.nodes:
        set_attr(G.nodes[n], ALIAS_THETA, 0.0)
        set_attr(G.nodes[n], ALIAS_VF, 0.0)
        set_attr(G.nodes[n], ALIAS_DNFR, 0.0)

    compute_Si(G, inplace=False)

    assert captured == [sentinel] * G.number_of_nodes()


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

    monkeypatch.setattr(
        "tnfr.metrics.sense_index.ProcessPoolExecutor", DummyExecutor
    )

    G = graph_canon()
    _configure_graph(G)
    G.graph["SI_N_JOBS"] = "3"

    compute_Si(G, inplace=False)

    assert captured == [3]
