from __future__ import annotations

import pytest

import tnfr.dynamics as dynamics
from tnfr.alias import set_attr


def _build_graph(graph_canon, samples):
    G = graph_canon()
    for idx, (si, dnfr, accel) in enumerate(samples):
        G.add_node(idx)
        nd = G.nodes[idx]
        set_attr(nd, dynamics.ALIAS_SI, si)
        set_attr(nd, dynamics.ALIAS_DNFR, dnfr)
        set_attr(nd, dynamics.ALIAS_D2EPI, accel)
    return G


def test_collect_selector_metrics_numpy_vectorized(monkeypatch, graph_canon):
    class _FakeArray(list):
        def astype(self, dtype):
            return _FakeArray(dtype(v) for v in self)

        def tolist(self):
            return list(self)

        def __truediv__(self, other):
            return _FakeArray(float(v) / other for v in self)

    class _FakeNumpy:
        @staticmethod
        def fromiter(iterable, dtype, count=None):
            return _FakeArray(dtype(value) for value in iterable)

        @staticmethod
        def clip(array, a_min, a_max):
            return _FakeArray(
                max(a_min, min(a_max, float(value))) for value in array
            )

        @staticmethod
        def abs(array):
            return _FakeArray(abs(float(value)) for value in array)

    monkeypatch.setattr(dynamics, "get_numpy", lambda: _FakeNumpy)

    samples = [
        (0.1, -0.5, 0.25),
        (1.5, 1.0, -0.5),
    ]
    G = _build_graph(graph_canon, samples)
    nodes = list(G.nodes)
    norms = {"dnfr_max": 2.0, "accel_max": 4.0}

    metrics = dynamics._collect_selector_metrics(G, nodes, norms)

    assert metrics[0] == pytest.approx((0.1, 0.25, 0.0625))
    assert metrics[1] == pytest.approx((1.0, 0.5, 0.125))


def test_collect_selector_metrics_process_pool(monkeypatch, graph_canon):
    monkeypatch.setattr(dynamics, "get_numpy", lambda: None)

    samples = [
        (1.2, -1.0, 2.0),
        (-0.1, 0.5, -3.0),
        (0.5, -2.0, 1.0),
        (0.8, 0.0, 0.0),
        (0.3, 0.2, -4.0),
    ]
    G = _build_graph(graph_canon, samples)
    nodes = list(G.nodes)
    norms = {"dnfr_max": 2.0, "accel_max": 4.0}

    captured: dict[str, object] = {}

    class StubExecutor:
        def __init__(self, *args, **kwargs):
            captured["max_workers"] = kwargs.get("max_workers") or (args[0] if args else None)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def map(self, func, iterable):
            items = list(iterable)
            captured["chunk_lengths"] = [len(chunk[0]) for chunk in items]
            return (func(chunk) for chunk in items)

    monkeypatch.setattr(dynamics, "ProcessPoolExecutor", StubExecutor)

    metrics = dynamics._collect_selector_metrics(G, nodes, norms, n_jobs=3)

    assert captured["max_workers"] == 3
    assert captured["chunk_lengths"] == [2, 2, 1]

    expected = {
        0: (1.0, 0.5, 0.5),
        1: (0.0, 0.25, 0.75),
        2: (0.5, 1.0, 0.25),
        3: (0.8, 0.0, 0.0),
        4: (0.3, 0.1, 1.0),
    }
    for node, triple in expected.items():
        assert metrics[node] == pytest.approx(triple)
