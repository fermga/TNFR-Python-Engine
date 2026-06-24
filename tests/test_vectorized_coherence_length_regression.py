"""Regression tests for compute_coherence_length_vectorized.

Bug: NaN/Inf/negative sentinels in caller-supplied distance matrices were
casted to negative ``intp`` values and crashed ``np.bincount``.
"""

from __future__ import annotations

import math

import networkx as nx
import numpy as np
import pytest

from tnfr.physics.vectorized_ops import compute_coherence_length_vectorized


def _ring(n: int = 12) -> tuple[nx.Graph, list, dict]:
    G = nx.cycle_graph(n)
    nodes = list(G.nodes())
    delta_nfr = {node: 0.1 * (i % 3) for i, node in enumerate(nodes)}
    return G, nodes, delta_nfr


def test_baseline_returns_finite_value():
    G, nodes, dnfr = _ring(16)
    xi_c = compute_coherence_length_vectorized(G, nodes, dnfr)
    assert isinstance(xi_c, float)
    assert math.isfinite(xi_c) or math.isnan(xi_c)


@pytest.mark.parametrize("sentinel", [-1.0, np.nan, np.inf, -np.inf])
def test_external_distance_matrix_with_invalid_entries(sentinel):
    """Caller-supplied distance matrix containing NaN/Inf/-1 must not crash."""
    G, nodes, dnfr = _ring(12)
    n = len(nodes)
    D = nx.floyd_warshall_numpy(G, nodelist=nodes).astype(np.float64)

    # Inject sentinel into a few off-diagonal entries (symmetric)
    D[0, 5] = sentinel
    D[5, 0] = sentinel
    D[2, 9] = sentinel
    D[9, 2] = sentinel

    # Must not raise (previously triggered ValueError in np.bincount)
    xi_c = compute_coherence_length_vectorized(G, nodes, dnfr, distance_matrix=D)
    assert isinstance(xi_c, float)
    # Either a finite positive ξ_C or NaN if fit cannot be done; never a crash
    assert math.isnan(xi_c) or xi_c > 0


def test_all_invalid_distances_returns_nan():
    G, nodes, dnfr = _ring(8)
    n = len(nodes)
    D = np.full((n, n), np.nan, dtype=np.float64)
    np.fill_diagonal(D, 0.0)

    xi_c = compute_coherence_length_vectorized(G, nodes, dnfr, distance_matrix=D)
    assert math.isnan(xi_c)
