"""Performance regression coverage for the vectorized Si path.

This benchmark ensures the NumPy-accelerated ``compute_Si`` implementation
remains significantly faster than the pure-Python fallback while producing
identical sense index outputs.  The optimization is critical for large graphs
where Si is recomputed frequently, so the threshold guards against accidental
regressions to non-vectorized behaviour.
"""

from __future__ import annotations

import math
import statistics
import time

import pytest

np = pytest.importorskip("numpy")
import numpy.testing as npt

from tnfr.alias import get_attr, set_attr
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

    def time_fast() -> float:
        return _measure(lambda: compute_Si(fast_graph, inplace=False), loops)

    target = next(iter(fast_graph.nodes))
    base_vf = float(get_attr(fast_graph.nodes[target], ALIAS_VF, 0.0))
    alt_vf = base_vf + 0.041

    def time_dirty() -> float:
        toggle = False

        def dirty_iteration() -> None:
            nonlocal toggle
            toggle = not toggle
            value = alt_vf if toggle else base_vf
            set_attr(fast_graph.nodes[target], ALIAS_VF, value)
            compute_Si(fast_graph, inplace=False)

        duration = _measure(dirty_iteration, loops)
        set_attr(fast_graph.nodes[target], ALIAS_VF, base_vf)
        compute_Si(fast_graph, inplace=False)
        return duration

    fast_time = min(time_fast() for _ in range(5))
    dirty_time = min(time_dirty() for _ in range(5))

    assert fast_time <= dirty_time * 0.98

    monkeypatch.setattr("tnfr.metrics.sense_index.get_numpy", lambda: None)
    _invalidate_trig_cache(slow_graph)

    slow_reference = compute_Si(slow_graph, inplace=False)
    slow_time = _measure(lambda: compute_Si(slow_graph, inplace=False), loops)

    nodes = sorted(fast_reference)
    fast_values = np.fromiter((fast_reference[n] for n in nodes), dtype=float)
    slow_values = np.fromiter((slow_reference[n] for n in nodes), dtype=float)
    npt.assert_allclose(fast_values, slow_values, rtol=1e-9, atol=1e-9)

    assert fast_time <= slow_time * 0.85


def test_compute_Si_large_graph_chunk_penalty_removed(graph_canon):
    """Large graphs should no longer regress when chunk hints are small."""

    graph = graph_canon()
    _seed_graph(graph, node_count=4096)

    # Warm caches so the timing loop exercises only the vectorised path.
    compute_Si(graph, inplace=False)

    loops = 4
    baseline_samples = [
        _measure(lambda: compute_Si(graph, inplace=False), loops)
        for _ in range(3)
    ]
    hinted_samples = [
        _measure(
            lambda: compute_Si(graph, inplace=False, chunk_size=32),
            loops,
        )
        for _ in range(3)
    ]
    baseline = statistics.fmean(baseline_samples)
    hinted = statistics.fmean(hinted_samples)

    values_default = compute_Si(graph, inplace=False)
    values_hinted = compute_Si(graph, inplace=False, chunk_size=32)

    nodes = sorted(values_default)
    fast_values = np.fromiter((values_default[n] for n in nodes), dtype=float)
    hinted_values = np.fromiter((values_hinted[n] for n in nodes), dtype=float)
    npt.assert_allclose(fast_values, hinted_values, rtol=1e-9, atol=1e-9)

    assert hinted <= baseline * 1.05


def test_compute_Si_buffer_cache_preserves_results(graph_canon):
    """Repeated cache reuse must keep Si parity to the uncached baseline."""

    graph = graph_canon()
    _seed_graph(graph, node_count=512)

    baseline = compute_Si(graph, inplace=False)
    node_order = list(baseline)
    reference = np.fromiter((baseline[n] for n in node_order), dtype=float)

    inplace_values = compute_Si(graph, inplace=True)
    npt.assert_allclose(inplace_values, reference, rtol=1e-9, atol=1e-9)

    for _ in range(5):
        cached = compute_Si(graph, inplace=False)
        cached_values = np.fromiter((cached[n] for n in node_order), dtype=float)
        npt.assert_allclose(cached_values, reference, rtol=1e-9, atol=1e-9)
