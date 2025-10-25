"""Deterministic stress coverage for ΔNFR and runtime orchestration."""

from __future__ import annotations

import math
import time
from collections.abc import Mapping

import networkx as nx
import pytest

pytest.importorskip("numpy")

from tnfr.alias import get_attr, set_attr
from tnfr.constants import DEFAULTS, get_aliases, inject_defaults
from tnfr.dynamics import default_compute_delta_nfr
from tnfr.dynamics.dnfr import _apply_dnfr_hook
import tnfr.dynamics.runtime as runtime
from tnfr.metrics import register_metrics_callbacks
from tnfr.glyph_history import ensure_history

ALIAS_THETA = get_aliases("THETA")
ALIAS_EPI = get_aliases("EPI")
ALIAS_VF = get_aliases("VF")
ALIAS_DNFR = get_aliases("DNFR")

pytestmark = [pytest.mark.slow, pytest.mark.stress]


def _bias_gradient(
    G: nx.Graph,
    node: int,
    node_data: Mapping[str, object],
) -> float:
    """Return a constant contribution regardless of ``node``."""

    return 1.0


def _degree_gradient(
    G: nx.Graph,
    node: int,
    node_data: Mapping[str, object],
) -> float:
    """Return the structural degree contribution for ``node``."""

    return float(G.degree(node))


def _seed_graph(
    *,
    num_nodes: int,
    edge_probability: float,
    seed: int,
) -> nx.Graph:
    """Create a graph with deterministic TNFR attributes and defaults applied."""

    graph = nx.gnp_random_graph(num_nodes, edge_probability, seed=seed)
    inject_defaults(graph)
    graph.graph["DNFR_WEIGHTS"] = dict(DEFAULTS["DNFR_WEIGHTS"])
    graph.graph["compute_delta_nfr"] = default_compute_delta_nfr
    graph.graph.setdefault("RANDOM_SEED", seed)

    twopi = 2.0 * math.pi
    for node, data in graph.nodes(data=True):
        base = seed + int(node)
        theta = ((base * 0.017) % twopi) - math.pi
        epi = math.sin(base * 0.031) * 0.45
        vf = 0.35 + 0.05 * ((base % 11) / 10.0)
        set_attr(data, ALIAS_THETA, theta)
        set_attr(data, ALIAS_EPI, epi)
        set_attr(data, ALIAS_VF, vf)
        set_attr(data, ALIAS_DNFR, 0.0)

    return graph


def _sum_dnfr(graph: nx.Graph) -> float:
    """Return the total ΔNFR across the nodes of ``graph``."""

    return sum(
        float(get_attr(data, ALIAS_DNFR, 0.0)) for _, data in graph.nodes(data=True)
    )


@pytest.mark.timeout(30)
def test_default_compute_delta_nfr_large_graph_consistent() -> None:
    """Ensure ΔNFR computations remain deterministic across cache modes."""

    node_count = 512
    edge_probability = 0.18
    seed = 4201

    cached_graph = _seed_graph(num_nodes=node_count, edge_probability=edge_probability, seed=seed)
    uncached_graph = _seed_graph(num_nodes=node_count, edge_probability=edge_probability, seed=seed)

    start_cached = time.perf_counter()
    default_compute_delta_nfr(cached_graph, cache_size=128)
    cached_duration = time.perf_counter() - start_cached

    start_uncached = time.perf_counter()
    default_compute_delta_nfr(uncached_graph, cache_size=0)
    uncached_duration = time.perf_counter() - start_uncached

    assert cached_duration < 30.0
    assert uncached_duration < 30.0

    dnfr_cached = _sum_dnfr(cached_graph)
    dnfr_uncached = _sum_dnfr(uncached_graph)

    assert dnfr_uncached == pytest.approx(dnfr_cached, rel=0.0, abs=1e-9)


@pytest.mark.timeout(30)
def test_runtime_run_long_trajectory_history_integrity() -> None:
    """Run the runtime loop for many steps ensuring metrics persist without STOP_EARLY."""

    steps = 200
    graph = _seed_graph(num_nodes=96, edge_probability=0.12, seed=2025)
    graph.graph["STOP_EARLY"] = {"enabled": False, "window": 32, "fraction": 0.92}
    graph.graph["HISTORY_MAXLEN"] = 0

    register_metrics_callbacks(graph)

    start = time.perf_counter()
    runtime.run(graph, steps=steps, dt=0.05, use_Si=True, apply_glyphs=False)
    duration = time.perf_counter() - start

    assert duration < 30.0

    history = ensure_history(graph)
    stable_series = history.get("stable_frac", [])
    coherence_series = history.get("C_steps", [])

    assert isinstance(stable_series, list)
    assert isinstance(coherence_series, list)
    assert len(stable_series) >= steps
    assert len(coherence_series) >= steps
    assert len(stable_series) == len(coherence_series)

    assert graph.graph.get("STOP_EARLY", {}).get("enabled") is False

    dnfr_total = _sum_dnfr(graph)
    assert math.isfinite(dnfr_total)


@pytest.mark.slow
@pytest.mark.timeout(30)
def test_apply_dnfr_hook_parallel_python_matches_serial(monkeypatch: pytest.MonkeyPatch) -> None:
    """Parallel ΔNFR hook execution must match the serial reference assignments."""

    node_count = 640
    edge_probability = 0.15
    seed = 8421

    serial_graph = _seed_graph(
        num_nodes=node_count,
        edge_probability=edge_probability,
        seed=seed,
    )
    parallel_graph = _seed_graph(
        num_nodes=node_count,
        edge_probability=edge_probability,
        seed=seed,
    )

    grads = {"bias": _bias_gradient, "degree": _degree_gradient}
    weights = {"bias": 0.5, "degree": 0.5}

    import tnfr.dynamics.dnfr as dnfr

    original_get_numpy = dnfr.get_numpy
    monkeypatch.setattr(dnfr, "get_numpy", lambda: None, raising=False)
    try:
        _apply_dnfr_hook(
            serial_graph,
            grads,
            weights=weights,
            hook_name="serial_reference",
            n_jobs=None,
        )

        start = time.perf_counter()
        _apply_dnfr_hook(
            parallel_graph,
            grads,
            weights=weights,
            hook_name="parallel_reference",
            n_jobs=8,
        )
        elapsed = time.perf_counter() - start
    finally:
        monkeypatch.setattr(dnfr, "get_numpy", original_get_numpy, raising=False)

    assert elapsed < 30.0

    for node in parallel_graph.nodes:
        parallel_value = float(get_attr(parallel_graph.nodes[node], ALIAS_DNFR, 0.0))
        serial_value = float(get_attr(serial_graph.nodes[node], ALIAS_DNFR, 0.0))
        assert parallel_value == pytest.approx(serial_value, rel=0.0, abs=1e-9)

    assert _sum_dnfr(parallel_graph) == pytest.approx(
        _sum_dnfr(serial_graph), rel=0.0, abs=1e-9
    )
