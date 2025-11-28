from __future__ import annotations

import networkx as nx

from tnfr.validation.aggregator import run_structural_validation


def _graph():
    G = nx.erdos_renyi_graph(12, 0.2)
    for n in G.nodes:
        G.nodes[n]["phase"] = 0.1 * n  # simple distinct phases
        G.nodes[n]["delta_nfr"] = 0.01 * (n + 1)
    return G


def test_validation_report_structure():
    G = _graph()
    seq = ["AL", "UM", "IL", "SHA"]  # canonical bootstrap variant
    report = run_structural_validation(G, sequence=seq)
    assert report.status in {"valid", "invalid"}
    assert isinstance(report.field_metrics["xi_c"], (int, float))
    assert "phase_gradient" in report.field_metrics
    assert report.sequence == tuple(seq)
    # Grammar should pass for canonical bootstrap
    assert report.status == "valid", report.notes


def test_threshold_override_triggers_elevated():
    G = _graph()
    seq = ["AL", "UM", "IL", "SHA"]
    report = run_structural_validation(
        G,
        sequence=seq,
        max_phase_gradient=0.00001,  # force exceed
    )
    assert report.thresholds_exceeded["phase_gradient_max"] is True
    assert report.risk_level in {"elevated", "critical"}
