from __future__ import annotations

import networkx as nx

from tnfr.validation.health import compute_structural_health


def _graph():
    G = nx.erdos_renyi_graph(10, 0.3)
    for n in G.nodes:
        G.nodes[n]["phase"] = 0.2 * n
        G.nodes[n]["delta_nfr"] = 0.02 * (n + 1)
    return G


def test_health_basic_payload():
    G = _graph()
    health = compute_structural_health(G, sequence=["AL", "UM", "IL", "SHA"])
    assert health["status"] == "valid"
    assert "risk_level" in health
    assert "recommended_actions" in health
    assert isinstance(health["field_metrics_subset"]["xi_c"], (int, float))


def test_health_recommendation_when_phase_gradient_forced():
    G = _graph()
    health = compute_structural_health(
        G,
        sequence=["AL", "UM", "IL", "SHA"],
        max_phase_gradient=0.00001,  # force flag
    )
    assert any(
        a in health["recommended_actions"]
        for a in ["phase_resync", "apply_coherence"]
    )
    assert health["thresholds_exceeded"]["phase_gradient_max"] is True
