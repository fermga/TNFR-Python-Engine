"""Tests for canonical structural potential landmark approximation.

Focus: validate adaptive landmark refinement keeps relative mean absolute
error (RMAE) below epsilon across representative topologies (ER + scale-free)
within safe ratio band 0.01–0.04 for mid-scale graphs.

Physics integrity:
  - Uses degree‑centered ΔNFR (telemetry heuristic) preserving spatial
    gradients (Invariant #3 semantics of ΔNFR as structural pressure).
  - Does not mutate EPI; read‑only field computation (Invariant #1).
  - Adaptive refinement logic ensures confinement metrics (U6) remain
    meaningful by bounding approximation error.
"""
from __future__ import annotations

import math
import random

import networkx as nx

from tnfr.physics import canonical as _canonical  # type: ignore

compute_structural_potential = _canonical.compute_structural_potential


def _assign_delta_nfr_degree_centered(G: nx.Graph) -> None:
    degs = dict(G.degree())
    mean_deg = sum(degs.values()) / len(degs) if degs else 0.0
    for n, d in degs.items():
        # Degree-centered ΔNFR (telemetry heuristic)
        G.nodes[n]["delta_nfr"] = float(d - mean_deg)


def _build_er(n: int, p: float, seed: int) -> nx.Graph:
    return nx.fast_gnp_random_graph(n, p, seed=seed)


def _build_scale_free(n: int, m: int, seed: int) -> nx.Graph:
    random.seed(seed)
    return nx.barabasi_albert_graph(n, m, seed=seed)


def _extract_rmae(potential: dict) -> float:
    return float(potential.get("__phi_s_rmae__", math.inf))


def test_landmark_adaptive_refinement_er_scale_free() -> None:
    """Adaptive landmark refinement keeps RMAE ≤ epsilon for ER + scale-free.

    Uses landmark_ratio starting at 0.01; validate=True triggers refinement
    (ratio may increase) until error ≤ 0.05 or max_refinements exhausted.
    """
    n = 600  # mid-scale (forces landmark heuristic in canonical path)
    epsilon = 0.05
    ratios_to_try = [0.01, 0.02, 0.04]

    # Erdős–Rényi
    er = _build_er(n, 0.05, seed=41)
    _assign_delta_nfr_degree_centered(er)
    for ratio in ratios_to_try:
        er_phi = compute_structural_potential(
            er,
            alpha=2.0,
            landmark_ratio=ratio,
            validate=True,
            error_epsilon=epsilon,
        )
        rmae = _extract_rmae(er_phi)
        assert rmae <= epsilon, (
            f"ER RMAE {rmae:.4f} > {epsilon:.4f}"
        )

    # Scale-free (Barabási–Albert)
    sf = _build_scale_free(n, m=5, seed=43)
    _assign_delta_nfr_degree_centered(sf)
    for ratio in ratios_to_try:
        sf_phi = compute_structural_potential(
            sf,
            alpha=2.0,
            landmark_ratio=ratio,
            validate=True,
            error_epsilon=epsilon,
        )
        rmae = _extract_rmae(sf_phi)
        assert rmae <= epsilon, (
            f"SF RMAE {rmae:.4f} > {epsilon:.4f}"
        )


def test_landmark_metadata_present() -> None:
    """Metadata keys appear when validate=True and landmark path active."""
    g = _build_er(600, 0.05, seed=55)
    _assign_delta_nfr_degree_centered(g)
    phi = compute_structural_potential(
        g,
        alpha=2.0,
        landmark_ratio=0.02,
        validate=True,
    )
    assert "__phi_s_landmark_ratio__" in phi, "Missing landmark ratio metadata"
    assert "__phi_s_rmae__" in phi, "Missing RMAE metadata"
    assert isinstance(phi["__phi_s_rmae__"], float)
    assert phi["__phi_s_rmae__"] <= 0.05, "RMAE exceeds default epsilon"
