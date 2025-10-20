"""Performance regression coverage for the vectorized Si path.

This benchmark ensures the NumPy-accelerated ``compute_Si`` implementation
remains significantly faster than the pure-Python fallback while producing
identical sense index outputs.  The optimization is critical for large graphs
where Si is recomputed frequently, so the threshold guards against accidental
regressions to non-vectorized behaviour.
"""

from __future__ import annotations

import math
import time

import pytest

np = pytest.importorskip("numpy")
import numpy.testing as npt

from tnfr.alias import set_attr
from tnfr.constants import get_aliases
from tnfr.metrics.sense_index import compute_Si

pytestmark = pytest.mark.slow

ALIAS_THETA = get_aliases("THETA")
ALIAS_VF = get_aliases("VF")
ALIAS_DNFR = get_aliases("DNFR")

TRIG_CACHE_KEYS = ("_cos_th", "_sin_th", "_thetas", "_trig_cache")


def _seed_graph(graph, node_count: int = 240) -> None:
    """Populate ``graph`` with deterministic θ, νf, and ΔNFR values."""

    graph.add_nodes_from(range(node_count))
    graph.add_edges_from(((idx, (idx + 1) % node_count) for idx in range(node_count)))
    graph.add_edges_from(((idx, (idx + 7) % node_count) for idx in range(node_count)))

    for node in graph.nodes:
        theta = (node % 36) * (math.pi / 18)
        vf = 0.15 + 0.02 * (node % 25)
        dnfr = 0.05 + 0.015 * ((node * 3) % 30)
        set_attr(graph.nodes[node], ALIAS_THETA, theta)
        set_attr(graph.nodes[node], ALIAS_VF, vf)
        set_attr(graph.nodes[node], ALIAS_DNFR, dnfr)


def _invalidate_trig_cache(graph) -> None:
    """Reset cached trigonometric data for ``graph``."""

    graph.graph["_trig_version"] = graph.graph.get("_trig_version", 0) + 1
    for key in TRIG_CACHE_KEYS:
        graph.graph.pop(key, None)


def _measure(callback, loops: int) -> float:
    start = time.perf_counter()
    for _ in range(loops):
        callback()
    return time.perf_counter() - start


def test_compute_Si_vectorized_outperforms_python(monkeypatch, graph_canon):
    fast_graph = graph_canon()
    slow_graph = graph_canon()
    _seed_graph(fast_graph)
    _seed_graph(slow_graph)

    # Warm caches for both graphs before timing runs.
    compute_Si(fast_graph, inplace=False)
    compute_Si(slow_graph, inplace=False)

    fast_reference = compute_Si(fast_graph, inplace=False)

    loops = 6
    fast_time = _measure(lambda: compute_Si(fast_graph, inplace=False), loops)

    monkeypatch.setattr("tnfr.metrics.sense_index.get_numpy", lambda: None)
    _invalidate_trig_cache(slow_graph)

    slow_reference = compute_Si(slow_graph, inplace=False)
    slow_time = _measure(lambda: compute_Si(slow_graph, inplace=False), loops)

    nodes = sorted(fast_reference)
    fast_values = np.fromiter((fast_reference[n] for n in nodes), dtype=float)
    slow_values = np.fromiter((slow_reference[n] for n in nodes), dtype=float)
    npt.assert_allclose(fast_values, slow_values, rtol=1e-9, atol=1e-9)

    assert fast_time <= slow_time * 0.6
