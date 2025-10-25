"""Regression tests for deterministic diagnosis computations."""

import math
from contextlib import contextmanager

import pytest

from tnfr.alias import set_attr
from tnfr.constants import get_aliases, get_param
from tnfr.glyph_history import ensure_history
from tnfr.metrics.diagnosis import _diagnosis_step

ALIAS_EPI = get_aliases("EPI")
ALIAS_VF = get_aliases("VF")
ALIAS_SI = get_aliases("SI")
ALIAS_DNFR = get_aliases("DNFR")
ALIAS_THETA = get_aliases("THETA")


def _build_ring_graph(graph_factory, *, size: int = 6) -> "nx.Graph":
    G = graph_factory()
    for idx in range(size):
        G.add_node(idx)
        base = 0.25 + 0.05 * idx
        set_attr(G.nodes[idx], ALIAS_SI, base % 1.0)
        set_attr(G.nodes[idx], ALIAS_EPI, 0.3 + 0.07 * idx)
        set_attr(G.nodes[idx], ALIAS_VF, 0.2 + 0.03 * idx)
        set_attr(G.nodes[idx], ALIAS_DNFR, (-1) ** idx * 0.04 * (idx + 1))
        set_attr(G.nodes[idx], ALIAS_THETA, (idx / size) * math.tau)
    for idx in range(size):
        G.add_edge(idx, (idx + 1) % size)
    return G


def _capture_diagnostics(G, *, jobs: int | None) -> dict:
    hist = ensure_history(G)
    key = get_param(G, "DIAGNOSIS").get("history_key", "nodal_diag")
    _diagnosis_step(G, n_jobs=jobs)
    return hist[key][-1]


@pytest.fixture
def graph_with_mixed_case_history(graph_canon):
    graph = _build_ring_graph(graph_canon)
    key = get_param(graph, "DIAGNOSIS").get("history_key", "nodal_diag")
    placeholder = object()
    snapshot = {
        0: {"state": "Stable", "note": "needs canonical form"},
        1: {"state": "DISSONANT"},
        2: placeholder,
    }
    graph.graph.setdefault("history", {})[key] = [snapshot]
    return graph, key, snapshot, placeholder


@contextmanager
def numpy_disabled(monkeypatch):
    from tnfr.metrics import diagnosis as diagnosis_module

    with monkeypatch.context() as ctx:
        ctx.setattr(diagnosis_module, "get_numpy", lambda: None)
        yield


@pytest.mark.parametrize("workers", [None, 3])
def test_parallel_diagnosis_matches_serial(graph_canon, workers):
    serial_graph = _build_ring_graph(graph_canon)
    parallel_graph = _build_ring_graph(graph_canon)

    baseline = _capture_diagnostics(serial_graph, jobs=1)
    parallel = _capture_diagnostics(parallel_graph, jobs=workers)

    assert parallel == baseline


def test_diagnosis_vectorized_matches_python(graph_canon, monkeypatch):
    pytest.importorskip("numpy")

    python_graph = _build_ring_graph(graph_canon)
    vector_graph = _build_ring_graph(graph_canon)

    with numpy_disabled(monkeypatch):
        baseline = _capture_diagnostics(python_graph, jobs=1)

    vectorized = _capture_diagnostics(vector_graph, jobs=4)

    assert vectorized.keys() == baseline.keys()
    for node_id, expected in baseline.items():
        observed = vectorized[node_id]
        assert observed.keys() == expected.keys()
        for key, value in expected.items():
            if isinstance(value, float):
                assert observed[key] == pytest.approx(value)
            else:
                assert observed[key] == value


def test_diagnosis_python_parallel_without_numpy(graph_canon, monkeypatch):
    serial_graph = _build_ring_graph(graph_canon)
    parallel_graph = _build_ring_graph(graph_canon)

    with numpy_disabled(monkeypatch):
        baseline = _capture_diagnostics(serial_graph, jobs=1)
        parallel = _capture_diagnostics(parallel_graph, jobs=3)

    assert parallel == baseline


def test_existing_history_states_are_canonicalised(graph_with_mixed_case_history):
    graph, key, snapshot, placeholder = graph_with_mixed_case_history

    _capture_diagnostics(graph, jobs=1)

    assert snapshot[0]["state"] == "stable"
    assert snapshot[1]["state"] == "dissonant"
    assert snapshot[2] is placeholder
