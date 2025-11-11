"""Tests for local coherence fallback used in operator metrics.

Validates that the fallback local coherence proxy remains within a
reasonable tolerance of compute_coherence applied to the induced
subgraph of a node's neighborhood.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add src to path for test execution
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import math
from typing import Any, cast

import networkx as nx

from tnfr.metrics.local_coherence import compute_local_coherence_fallback
from tnfr.metrics.common import compute_coherence
from tnfr.constants.aliases import ALIAS_DNFR, ALIAS_DEPI

TOLERANCE = 0.15  # allowable deviation


def _set_attrs(G: nx.Graph, node: str, dnfr: float, depi: float) -> None:
    G.nodes[node][ALIAS_DNFR[0]] = dnfr
    G.nodes[node][ALIAS_DEPI[0]] = depi


def build_graph() -> nx.Graph:
    G = nx.Graph()
    for i in range(6):
        G.add_node(f"n{i}")
    edges = [("n0", "n1"), ("n0", "n2"), ("n1", "n3"), ("n2", "n4"), ("n3", "n5")]
    G.add_edges_from(edges)
    # Assign attributes with slight variation
    vals = [0.2, 0.25, 0.18, 0.22, 0.27, 0.19]
    for idx, v in enumerate(vals):
        _set_attrs(G, f"n{idx}", dnfr=v, depi=v * 0.5)
    return G


def test_local_coherence_close_to_induced_subgraph():
    G = build_graph()
    target = "n0"
    # Fallback local coherence
    c_local = compute_local_coherence_fallback(G, target)

    # Induced subgraph (neighbors + target)
    neighbors = list(G.neighbors(target))
    induced_nodes = [target] + neighbors
    H = G.subgraph(induced_nodes).copy()

    # Compute coherence on induced subgraph approximating local coherence
    # Cast to Any to satisfy GraphLike protocol in test context
    c_induced = compute_coherence(cast(Any, H))

    assert 0.0 <= c_local <= 1.0
    assert 0.0 <= c_induced <= 1.0

    diff = abs(c_local - c_induced)
    assert diff <= TOLERANCE, (
        f"Local coherence fallback deviation {diff:.3f} exceeds tolerance {TOLERANCE}"
        + f" (c_local={c_local:.3f}, c_induced={c_induced:.3f})"
    )


def test_local_coherence_no_neighbors_zero():
    G = nx.Graph()
    G.add_node("solo")
    _set_attrs(G, "solo", dnfr=0.3, depi=0.1)
    assert compute_local_coherence_fallback(G, "solo") == 0.0


if __name__ == "__main__":
    print("Running local coherence fallback tests...")
    try:
        test_local_coherence_close_to_induced_subgraph()
        print("✓ test_local_coherence_close_to_induced_subgraph PASSED")
    except AssertionError as e:
        print(f"✗ test_local_coherence_close_to_induced_subgraph FAILED: {e}")

    try:
        test_local_coherence_no_neighbors_zero()
        print("✓ test_local_coherence_no_neighbors_zero PASSED")
    except AssertionError as e:
        print(f"✗ test_local_coherence_no_neighbors_zero FAILED: {e}")

    print("\nAll tests completed.")
