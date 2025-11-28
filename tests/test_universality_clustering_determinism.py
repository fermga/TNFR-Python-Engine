"""Determinism tests for universality clustering.

Verifies that `deterministic_kmeans` produces identical assignments and
centroids across multiple invocations with identical input ordering.

Physics alignment: Purely observational analytics layer; no modification of
TNFR dynamics or operators. Invariants #1, #4, #8 preserved.
"""
from __future__ import annotations

import importlib.util
import pathlib
from typing import Dict, List

# Dynamically load the clustering module to avoid package import constraints.
MOD_PATH = pathlib.Path("benchmarks/universality_clusters.py").resolve()
_spec = importlib.util.spec_from_file_location(
    "universality_clusters", str(MOD_PATH)
)
uc = importlib.util.module_from_spec(_spec)  # type: ignore
assert _spec and _spec.loader
_spec.loader.exec_module(uc)  # type: ignore


def _make_vectors() -> Dict[str, List[float]]:
    # Synthetic feature vectors (phi_s, grad, curv, xi_c, snapshot)
    return {
        "ring": [0.95, 0.60, 0.40, 1.10, 0.72],
        "grid": [1.02, 0.62, 0.41, 1.08, 0.70],
        "scale_free": [0.55, 0.30, 0.20, 0.90, 0.40],
        "ws": [0.58, 0.32, 0.22, 0.88, 0.42],
        "random": [0.80, 0.55, 0.33, 1.00, 0.60],
    }


def test_kmeans_deterministic_assignments_centroids():
    vectors = _make_vectors()
    result1 = uc.deterministic_kmeans(vectors, k=2)
    result2 = uc.deterministic_kmeans(vectors, k=2)
    assert result1["assignments"] == result2["assignments"], (
        "Assignments differ across runs"
    )
    assert result1["centroids"] == result2["centroids"], (
        "Centroids differ across runs"
    )
    assert result1["iterations"] == result2["iterations"], (
        "Iteration counts differ"
    )


def test_single_cluster_case():
    vectors = _make_vectors()
    res = uc.deterministic_kmeans(vectors, k=1)
    # All topologies must exist in single cluster 0
    assert set(res["clusters"].keys()) == {0}
    assert set(res["assignments"].values()) == {0}
    assert len(res["clusters"][0]) == len(vectors)


def test_identity_partition_case():
    vectors = _make_vectors()
    # k >= number of topologies => identity partition
    res = uc.deterministic_kmeans(vectors, k=len(vectors))
    assert len(res["clusters"]) == len(vectors)
    for topo, cid in res["assignments"].items():
        assert res["clusters"][cid] == [topo]
        assert res["centroids"][cid] == vectors[topo]


def test_no_side_effects():
    vectors = _make_vectors()
    before = {k: v[:] for k, v in vectors.items()}
    _ = uc.deterministic_kmeans(vectors, k=2)
    # Confirm vectors unchanged (read-only behavior)
    assert vectors == before
