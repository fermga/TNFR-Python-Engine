import math

import networkx as nx

from tnfr.operators.grammar import (
    warn_phase_gradient_telemetry,
    warn_phase_curvature_telemetry,
    warn_coherence_length_telemetry,
)


def _set_phase(G, mapping):
    for n, th in mapping.items():
        G.nodes[n]["phase"] = float(th)


def _set_dnfr(G, value: float):
    for n in G.nodes():
        G.nodes[n]["delta_nfr"] = float(value)


class TestU6PhaseGradientWarnings:
    def test_phase_gradient_warns_when_exceeding_threshold(self):
        G = nx.path_graph(6)
        # Alternate phases 0 and pi to maximize neighbor diffs
        phases = {i: (0.0 if i % 2 == 0 else math.pi) for i in G.nodes()}
        _set_phase(G, phases)

        safe, stats, msg, flagged = warn_phase_gradient_telemetry(G, threshold=0.38)

        assert not safe
        assert len(flagged) > 0
        assert stats["max"] >= 0.38
        assert "WARN" in msg

    def test_phase_gradient_passes_when_uniform_phase(self):
        G = nx.path_graph(5)
        _set_phase(G, {i: 0.0 for i in G.nodes()})

        safe, stats, msg, flagged = warn_phase_gradient_telemetry(G, threshold=0.38)

        assert safe
        assert stats["max"] == 0.0
        assert stats["mean"] == 0.0
        assert flagged == []
        assert "PASS" in msg


class TestU6PhaseCurvatureWarnings:
    def test_phase_curvature_hotspot_detected(self):
        # Star graph: center has phase pi, leaves 0 -> center curvature ~ pi
        G = nx.star_graph(4)  # nodes 0..4, center 0
        _set_phase(G, {0: math.pi, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0})

        safe, stats, msg, hotspots = warn_phase_curvature_telemetry(
            G, abs_threshold=3.0, multiscale_check=True
        )

        assert not safe
        assert stats["hotspots"] >= 1
        assert any(h == 0 for h in hotspots)  # center flagged
        assert "WARN" in msg

    def test_phase_curvature_passes_when_uniform(self):
        G = nx.cycle_graph(6)
        _set_phase(G, {i: 0.0 for i in G.nodes()})

        safe, stats, msg, hotspots = warn_phase_curvature_telemetry(
            G, abs_threshold=3.0, multiscale_check=True
        )

        assert safe
        assert stats["hotspots"] == 0
        assert stats["max_abs"] == 0.0
        assert hotspots == []
        assert "PASS" in msg


class TestU6CoherenceLengthWarnings:
    def test_coherence_length_stable_when_uniform_low_pressure(self):
        G = nx.path_graph(8)
        _set_dnfr(G, 0.1)

        safe, stats, msg = warn_coherence_length_telemetry(G)

        assert safe
        assert stats["severity"] == "stable"
        assert "PASS" in msg

    def test_coherence_length_critical_with_monkeypatch(self, monkeypatch):
        G = nx.path_graph(10)
        _set_dnfr(G, 0.2)

        # Compute diameter to set a value safely above
        diam = nx.diameter(G)

        from tnfr.physics import fields as fields_mod

        monkeypatch.setattr(
            fields_mod,
            "estimate_coherence_length",
            lambda H: float(diam + 10.0),
            raising=True,
        )

        safe, stats, msg = warn_coherence_length_telemetry(G)

        assert not safe
        assert stats["severity"] == "critical"
        assert "WARN" in msg
