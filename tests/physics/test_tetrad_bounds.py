"""Genuine structural-field tetrad bounds (π is the sole structural scale).

The four-field tetrad (Φ_s, |∇φ|, K_φ, ξ_C) is the minimal derivative tower.
The only genuine structural scale is π: both phase derivatives are wrapped
angles, so |∇φ| ≤ π and |K_φ| ≤ π for any configuration. These tests check
field computability and the genuine π bounds — no φ/γ/e correspondence.
"""

from __future__ import annotations

import math

import networkx as nx

from tnfr.physics.fields import (
    compute_phase_curvature,
    compute_phase_gradient,
    compute_structural_potential,
    estimate_coherence_length,
)


def _ring_with_phases(n: int) -> nx.Graph:
    G = nx.cycle_graph(n)
    for i in G.nodes():
        G.nodes[i]["phase"] = (i * 2.0 * math.pi / n) % (2.0 * math.pi)
        G.nodes[i]["ΔNFR"] = 0.1
        G.nodes[i]["νf"] = 1.0
        G.nodes[i]["EPI"] = float(i) / n
    return G


class TestPhaseSectorBounds:
    """Both phase derivatives are wrapped angles bounded by π."""

    def test_phase_gradient_bounded_by_pi(self):
        grad = compute_phase_gradient(_ring_with_phases(8))
        assert max(abs(v) for v in grad.values()) <= math.pi + 1e-9

    def test_phase_curvature_bounded_by_pi(self):
        curv = compute_phase_curvature(_ring_with_phases(8))
        assert max(abs(v) for v in curv.values()) <= math.pi + 1e-9


class TestFieldComputability:
    """The aggregation/non-local fields are computable and finite."""

    def test_structural_potential_finite(self):
        Phi_s = compute_structural_potential(nx.complete_graph(5))
        assert all(math.isfinite(v) for v in Phi_s.values())

    def test_coherence_length_nonnegative(self):
        G = nx.path_graph(12)
        for node in G.nodes():
            G.nodes[node]["ΔNFR"] = 0.1 * math.exp(-node / 3.0)
            G.nodes[node]["νf"] = 1.0
            G.nodes[node]["phase"] = node * 0.1
        xi = estimate_coherence_length(G)
        assert isinstance(xi, (int, float))
        if not math.isnan(xi):
            assert xi > 0.0
