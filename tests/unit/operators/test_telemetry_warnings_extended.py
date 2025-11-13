from __future__ import annotations

import math
import networkx as nx

from tnfr.operators.grammar import (
    warn_phase_gradient_telemetry,
    warn_phase_curvature_telemetry,
    warn_coherence_length_telemetry,
)


def _wrap_angle(x: float) -> float:
    return (x + math.pi) % (2 * math.pi) - math.pi


def test_warn_phase_gradient_pass_and_warn():
    G = nx.path_graph(4)
    # PASS: set all phases equal
    for n in G.nodes():
        G.nodes[n]["phase"] = 0.0
    safe, stats, msg, flagged = warn_phase_gradient_telemetry(G)
    assert safe is True
    assert "PASS" in msg

    # WARN: create a sharp phase difference at the center
    for n in G.nodes():
        G.nodes[n]["phase"] = 0.0
    # center node pi apart from neighbors
    center = 1
    G.nodes[center]["phase"] = math.pi
    safe2, stats2, msg2, flagged2 = warn_phase_gradient_telemetry(G)
    assert safe2 is False
    assert "WARN" in msg2
    assert len(flagged2) >= 1


def test_warn_phase_curvature_pass_and_warn():
    G = nx.path_graph(3)
    # PASS: equal phases => K_phi approx 0 everywhere
    for n in G.nodes():
        G.nodes[n]["phase"] = 0.0
    safe, stats, msg, hotspots = warn_phase_curvature_telemetry(G, multiscale_check=False)
    assert safe is True
    assert "PASS" in msg

    # WARN: make node 1 out of sync ~pi vs neighbors => |K_phi| ~ pi > 3.0
    for n in G.nodes():
        G.nodes[n]["phase"] = 0.0
    G.nodes[1]["phase"] = math.pi
    safe2, stats2, msg2, hotspots2 = warn_phase_curvature_telemetry(G, multiscale_check=False)
    assert safe2 is False
    assert "WARN" in msg2
    assert len(hotspots2) >= 1


def test_warn_coherence_length_basic_contract():
    G = nx.watts_strogatz_graph(n=20, k=4, p=0.1, seed=123)
    # set reasonable delta_nfr values
    for n in G.nodes():
        G.nodes[n]["delta_nfr"] = 0.5
    safe, stats, msg = warn_coherence_length_telemetry(G)
    assert isinstance(safe, bool)
    assert isinstance(msg, str) and msg.startswith("U6 (ξ_C): ")
    assert set(["xi_c", "mean_path_length", "diameter", "severity"]).issubset(stats.keys())
