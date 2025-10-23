"""Performance regression tests for increment application paths."""

from __future__ import annotations

import time
from typing import Callable

import networkx as nx
import pytest

numpy = pytest.importorskip("numpy")

from tnfr.alias import set_attr
from tnfr.dynamics import integrators as integrators_mod

pytestmark = pytest.mark.slow


def _build_graph_and_increments(
    method: str,
    *,
    node_count: int = 512,
) -> tuple[nx.Graph, float, dict[int, tuple[float, ...]]]:
    graph = nx.path_graph(node_count)

    epi = numpy.linspace(-0.5, 0.75, node_count)
    dEPI_prev = numpy.sin(numpy.linspace(0.0, numpy.pi * 3.0, node_count)) * 0.25
    freq = 0.6 + numpy.linspace(0.0, 0.3, node_count)
    dnfr = 0.4 + numpy.cos(numpy.linspace(0.0, numpy.pi * 2.5, node_count)) * 0.2

    for idx, node in enumerate(graph.nodes):
        nd = graph.nodes[node]
        set_attr(nd, integrators_mod.ALIAS_EPI, float(epi[idx]))
        set_attr(nd, integrators_mod.ALIAS_DEPI, float(dEPI_prev[idx]))
        set_attr(nd, integrators_mod.ALIAS_VF, float(freq[idx]))
        set_attr(nd, integrators_mod.ALIAS_DNFR, float(dnfr[idx]))

    base_rate = freq * dnfr
    phase = numpy.linspace(-numpy.pi / 3.0, numpy.pi / 2.0, node_count)

    if method == "rk4":
        offsets = numpy.stack(
            [numpy.sin(phase + shift) for shift in (0.0, 0.35, 0.7, 1.05)],
            axis=1,
        )
        staged = base_rate[:, None] + 0.08 * offsets
    else:
        staged = (base_rate + 0.08 * numpy.sin(phase))[:, None]

    increments = {
        node: tuple(float(value) for value in staged[idx])
        for idx, node in enumerate(graph.nodes)
    }

    dt_step = 0.05
    return graph, dt_step, increments


def _measure(runtime_fn: Callable[[], None], loops: int) -> float:
    start = time.perf_counter()
    for _ in range(loops):
        runtime_fn()
    return time.perf_counter() - start


@pytest.mark.parametrize("method", ["euler", "rk4"])
def test_apply_increments_numpy_branch_is_faster(monkeypatch, method):
    graph, dt_step, increments = _build_graph_and_increments(method)

    original_chunk = integrators_mod._apply_increment_chunk

    # Exercise the vectorised branch.
    monkeypatch.setattr(integrators_mod, "get_numpy", lambda: numpy)
    integrators_mod._apply_increments(
        graph,
        dt_step,
        increments,
        method=method,
        n_jobs=None,
    )
    numpy_results = integrators_mod._apply_increments(
        graph,
        dt_step,
        increments,
        method=method,
        n_jobs=None,
    )

    numpy_time = _measure(
        lambda: integrators_mod._apply_increments(
            graph,
            dt_step,
            increments,
            method=method,
            n_jobs=None,
        ),
        loops=8,
    )

    # Force the scalar fallback and ensure the chunk helper executes.
    monkeypatch.setattr(integrators_mod, "get_numpy", lambda: None)
    chunk_calls = 0

    def tracked_chunk(
        chunk: list[tuple[int, float, float, tuple[float, ...]]],
        dt_arg: float,
        method_arg: str,
    ) -> list[tuple[int, tuple[float, float, float]]]:
        nonlocal chunk_calls
        chunk_calls += 1
        return original_chunk(chunk, dt_arg, method_arg)

    monkeypatch.setattr(integrators_mod, "_apply_increment_chunk", tracked_chunk)

    integrators_mod._apply_increments(
        graph,
        dt_step,
        increments,
        method=method,
        n_jobs=None,
    )
    fallback_results = integrators_mod._apply_increments(
        graph,
        dt_step,
        increments,
        method=method,
        n_jobs=None,
    )

    fallback_time = _measure(
        lambda: integrators_mod._apply_increments(
            graph,
            dt_step,
            increments,
            method=method,
            n_jobs=None,
        ),
        loops=8,
    )

    assert chunk_calls > 0
    # Allow modest timing variance so minor CI noise does not trip the check, while
    # still bounding the optimized path from regressing significantly.
    assert numpy_time <= fallback_time * 1.1

    for node in graph.nodes:
        assert numpy_results[node] == pytest.approx(fallback_results[node])
