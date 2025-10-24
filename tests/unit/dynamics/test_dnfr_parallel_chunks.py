"""Exercise ΔNFR parallel scheduling and fallbacks in pure Python."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pickle

from tnfr.alias import get_attr
from tnfr.constants import get_aliases
from tnfr.dynamics.dnfr import _apply_dnfr_hook

ALIAS_DNFR = get_aliases("DNFR")


class _ImmediateFuture:
    """Future returning the provided value immediately."""

    def __init__(self, value: Any) -> None:
        self._value = value

    def result(self) -> Any:
        return self._value


def _serial_totals(G, grads, weights):
    """Return reference ΔNFR totals by iterating nodes serially."""

    totals: dict[Any, float] = {}
    for node, data in G.nodes(data=True):
        total = 0.0
        for name, func in grads.items():
            w = float(weights.get(name, 0.0))
            if w:
                total += w * float(func(G, node, data))
        totals[node] = total
    return totals


def _configure_graph(graph_factory, count: int) -> Any:
    """Create a line graph with ``count`` nodes carrying ``bias`` values."""

    G = graph_factory()
    for idx in range(count):
        G.add_node(idx, bias=float(idx) + 0.5)
        if idx:
            G.add_edge(idx - 1, idx)
    return G


def _grad_bias(graph, node, data):
    return float(data.get("bias", 0.0))


def _grad_degree(graph, node, _data):
    return float(graph.degree(node))


def test_parallel_chunks_cover_all_nodes_once(monkeypatch, graph_canon):
    """Chunk scheduling records non-overlapping node slices with deterministic results."""

    monkeypatch.setattr("tnfr.dynamics.dnfr.get_numpy", lambda: None)

    recorded_chunks: list[tuple[int, ...]] = []
    max_workers_used: list[int] = []

    class _RecordingExecutor:
        def __init__(self, *, max_workers: int | None = None) -> None:
            max_workers_used.append(int(max_workers or 0))

        def __enter__(self) -> "_RecordingExecutor":
            return self

        def __exit__(self, exc_type, exc, tb) -> None:  # noqa: D401 - standard context proto
            return None

        def submit(self, func, G, node_ids: Iterable[int], grad_items, weights):
            nodes = tuple(node_ids)
            assert nodes, "chunks must not be empty"
            recorded_chunks.append(nodes)
            return _ImmediateFuture(func(G, nodes, grad_items, weights))

    monkeypatch.setattr("tnfr.dynamics.dnfr.ProcessPoolExecutor", _RecordingExecutor)

    grads = {"bias": _grad_bias, "degree": _grad_degree}
    weights = {"bias": 1.0, "degree": -0.5}

    G_parallel = _configure_graph(graph_canon, count=5)
    expected = _serial_totals(G_parallel, grads, weights)

    _apply_dnfr_hook(
        G_parallel,
        grads,
        weights=weights,
        hook_name="test_parallel",
        n_jobs=4,
    )

    flattened = [node for chunk in recorded_chunks for node in chunk]
    assert flattened == list(G_parallel.nodes), "every node scheduled exactly once"
    assert len(set(flattened)) == len(flattened)

    assert max_workers_used == [4], "effective worker count respected"

    observed = {
        node: get_attr(G_parallel.nodes[node], ALIAS_DNFR, 0.0)
        for node in G_parallel.nodes
    }
    assert observed == expected


def test_pickle_failure_falls_back_to_serial(monkeypatch, graph_canon):
    """When payloads are not picklable the serial path still applies ΔNFR updates."""

    monkeypatch.setattr("tnfr.dynamics.dnfr.get_numpy", lambda: None)

    calls: list[str] = []

    class _FailingExecutor:
        def __init__(self, *args, **kwargs):
            calls.append("instantiated")
            raise AssertionError("ProcessPoolExecutor should not be used when pickle fails")

    monkeypatch.setattr("tnfr.dynamics.dnfr.ProcessPoolExecutor", _FailingExecutor)

    def _raising_dumps(*_args, **_kwargs):
        raise pickle.PicklingError("payload rejected during test")

    monkeypatch.setattr(pickle, "dumps", _raising_dumps)

    grads = {"bias": _grad_bias, "degree": _grad_degree}
    weights = {"bias": 0.75, "degree": 0.25}

    G_serial = _configure_graph(graph_canon, count=4)
    expected = _serial_totals(G_serial, grads, weights)

    _apply_dnfr_hook(
        G_serial,
        grads,
        weights=weights,
        hook_name="test_pickle_fallback",
        n_jobs=3,
    )

    assert not calls, "parallel executor must not be instantiated"

    observed = {
        node: get_attr(G_serial.nodes[node], ALIAS_DNFR, 0.0)
        for node in G_serial.nodes
    }
    assert observed == expected
