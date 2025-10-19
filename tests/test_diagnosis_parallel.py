"""Regression tests for deterministic diagnosis computations."""

import math

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


@pytest.mark.parametrize("workers", [None, 3])
def test_parallel_diagnosis_matches_serial(graph_canon, workers):
    serial_graph = _build_ring_graph(graph_canon)
    parallel_graph = _build_ring_graph(graph_canon)

    baseline = _capture_diagnostics(serial_graph, jobs=1)
    parallel = _capture_diagnostics(parallel_graph, jobs=workers)

    assert parallel == baseline
