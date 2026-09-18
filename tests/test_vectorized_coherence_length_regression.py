"""Caller-declared distance matrices must satisfy the explicit distance domain."""

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


@pytest.mark.parametrize("sentinel", [-1.0, np.nan, -np.inf])
def test_external_distance_matrix_with_invalid_entries(sentinel):
    """Invalid data raises a domain error, not an allocation or bincount failure."""
    G, nodes, dnfr = _ring(12)
    n = len(nodes)
    D = nx.floyd_warshall_numpy(G, nodelist=nodes).astype(np.float64)

    # Inject sentinel into a few off-diagonal entries (symmetric)
    D[0, 5] = sentinel
    D[5, 0] = sentinel
    D[2, 9] = sentinel
    D[9, 2] = sentinel

    with pytest.raises(ValueError, match="nonnegative distances"):
        compute_coherence_length_vectorized(G, nodes, dnfr, distance_matrix=D)


def test_all_invalid_distances_are_not_a_failed_fit():
    G, nodes, dnfr = _ring(8)
    n = len(nodes)
    D = np.full((n, n), np.nan, dtype=np.float64)
    np.fill_diagonal(D, 0.0)

    with pytest.raises(ValueError, match="nonnegative distances"):
        compute_coherence_length_vectorized(G, nodes, dnfr, distance_matrix=D)


def test_declared_unreachable_pairs_return_unavailable_fit():
    G, nodes, dnfr = _ring(8)
    distances = np.full((len(nodes), len(nodes)), np.inf)
    np.fill_diagonal(distances, 0.0)
    result = compute_coherence_length_vectorized(G, nodes, dnfr, distance_matrix=distances)
    assert math.isnan(result)
