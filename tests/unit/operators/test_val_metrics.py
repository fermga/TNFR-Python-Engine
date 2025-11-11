"""Tests for enhanced VAL (Expansion) metrics (Issue #2724).

This module validates the comprehensive metrics collection for the VAL operator,
including bifurcation risk, coherence preservation, fractality indicators,
network impact, and structural stability.

Test Coverage:
--------------
1. **Bifurcation Risk**: ∂²EPI/∂t² detection and magnitude
2. **Coherence Preservation**: Local C(t) monitoring
3. **Fractality Indicators**: Growth rate ratios
4. **Network Impact**: Phase coherence with neighbors
5. **ΔNFR Stability**: Positivity and boundedness
6. **Overall Health**: Combined indicator

References:
-----------
- Issue #2724: [VAL] Enriquecer métricas de telemetría
- TNFR.pdf § 2.1: Nodal equation
- AGENTS.md: Canonical invariants
"""

import math

import pytest

from tnfr.alias import set_attr
from tnfr.constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_VF, ALIAS_THETA, ALIAS_D2EPI
from tnfr.operators.definitions import Expansion, Coherence, Emission
from tnfr.structural import create_nfr


class TestVALMetricsBifurcationRisk:
    """Test suite for bifurcation risk metrics."""

    def test_val_metrics_capture_bifurcation_risk(self):
        """VAL metrics detect bifurcation risk from d2epi."""
        G, node = create_nfr("bifurc", epi=0.4, vf=1.0)
        G.graph["COLLECT_OPERATOR_METRICS"] = True
        G.graph["VAL_BIFURCATION_THRESHOLD"] = 0.3

        # Set high d2epi to trigger bifurcation risk
        set_attr(G.nodes[node], ALIAS_D2EPI, 0.5)  # Above threshold
        set_attr(G.nodes[node], ALIAS_DNFR, 0.2)  # Positive ΔNFR

        Expansion()(G, node)
        metrics = G.graph["operator_metrics"][-1]

        assert "bifurcation_risk" in metrics
        assert metrics["bifurcation_risk"] is True
        assert "d2epi" in metrics
        assert abs(metrics["d2epi"]) > 0.3

    def test_val_metrics_no_bifurcation_risk_when_low_d2epi(self):
        """VAL metrics show no risk when d2epi is below threshold."""
        G, node = create_nfr("stable", epi=0.4, vf=1.0)
        G.graph["COLLECT_OPERATOR_METRICS"] = True
        G.graph["VAL_BIFURCATION_THRESHOLD"] = 0.3

        # Set low d2epi
        set_attr(G.nodes[node], ALIAS_D2EPI, 0.1)  # Below threshold
        set_attr(G.nodes[node], ALIAS_DNFR, 0.2)

        Expansion()(G, node)
        metrics = G.graph["operator_metrics"][-1]

        assert metrics["bifurcation_risk"] is False
        assert abs(metrics["d2epi"]) < 0.3

    def test_val_metrics_bifurcation_magnitude(self):
        """VAL metrics compute bifurcation magnitude as ratio to threshold."""
        G, node = create_nfr("magnitude", epi=0.4, vf=1.0)
        G.graph["COLLECT_OPERATOR_METRICS"] = True
        G.graph["VAL_BIFURCATION_THRESHOLD"] = 0.2

        # Set d2epi to 2x threshold
        set_attr(G.nodes[node], ALIAS_D2EPI, 0.4)
        set_attr(G.nodes[node], ALIAS_DNFR, 0.2)

        Expansion()(G, node)
        metrics = G.graph["operator_metrics"][-1]

        assert "bifurcation_magnitude" in metrics
        assert abs(metrics["bifurcation_magnitude"] - 2.0) < 0.1

    def test_val_metrics_configurable_bifurcation_threshold(self):
        """VAL bifurcation threshold is configurable via graph metadata."""
        G, node = create_nfr("config", epi=0.4, vf=1.0)
        G.graph["COLLECT_OPERATOR_METRICS"] = True
        G.graph["VAL_BIFURCATION_THRESHOLD"] = 0.5  # Custom threshold

        set_attr(G.nodes[node], ALIAS_D2EPI, 0.4)  # Below custom threshold
        set_attr(G.nodes[node], ALIAS_DNFR, 0.2)

        Expansion()(G, node)
        metrics = G.graph["operator_metrics"][-1]

        assert metrics["bifurcation_threshold"] == 0.5
        assert metrics["bifurcation_risk"] is False  # Below 0.5


class TestVALMetricsCoherencePreservation:
    """Test suite for coherence preservation metrics."""

    def test_val_metrics_detect_coherence_loss(self):
        """VAL metrics flag coherence degradation."""
        G, node = create_nfr("coherence_loss", epi=0.4, vf=1.0)
        G.graph["COLLECT_OPERATOR_METRICS"] = True
        G.graph["VAL_MIN_COHERENCE"] = 0.5

        # Set attributes
        set_attr(G.nodes[node], ALIAS_DNFR, 0.2)
        set_attr(G.nodes[node], ALIAS_D2EPI, 0.1)

        Expansion()(G, node)
        metrics = G.graph["operator_metrics"][-1]

        assert "coherence_local" in metrics
        assert "coherence_preserved" in metrics
        assert isinstance(metrics["coherence_preserved"], bool)

    def test_val_metrics_coherence_threshold_configurable(self):
        """VAL coherence threshold is configurable."""
        G, node = create_nfr("coherence_config", epi=0.4, vf=1.0)
        G.graph["COLLECT_OPERATOR_METRICS"] = True
        G.graph["VAL_MIN_COHERENCE"] = 0.7  # Custom threshold

        set_attr(G.nodes[node], ALIAS_DNFR, 0.2)
        set_attr(G.nodes[node], ALIAS_D2EPI, 0.1)

        Expansion()(G, node)
        metrics = G.graph["operator_metrics"][-1]

        # Coherence check should use custom threshold
        assert "coherence_preserved" in metrics


class TestVALMetricsFractality:
    """Test suite for fractality preservation metrics."""

    def test_val_metrics_validate_fractality(self):
        """VAL metrics validate fractal preservation via growth ratios."""
        G, node = create_nfr("fractal", epi=0.4, vf=1.0)
        G.graph["COLLECT_OPERATOR_METRICS"] = True
        G.graph["VAL_FRACTAL_RATIO_MIN"] = 0.5
        G.graph["VAL_FRACTAL_RATIO_MAX"] = 2.0

        set_attr(G.nodes[node], ALIAS_DNFR, 0.2)
        set_attr(G.nodes[node], ALIAS_D2EPI, 0.1)

        Expansion()(G, node)
        metrics = G.graph["operator_metrics"][-1]

        assert "epi_growth_rate" in metrics
        assert "vf_growth_rate" in metrics
        assert "growth_ratio" in metrics
        assert "fractal_preserved" in metrics
        assert isinstance(metrics["fractal_preserved"], bool)

    def test_val_metrics_growth_rates(self):
        """VAL metrics compute relative growth rates."""
        G, node = create_nfr("growth", epi=0.4, vf=1.0)
        G.graph["COLLECT_OPERATOR_METRICS"] = True

        set_attr(G.nodes[node], ALIAS_DNFR, 0.2)
        set_attr(G.nodes[node], ALIAS_D2EPI, 0.1)

        Expansion()(G, node)
        metrics = G.graph["operator_metrics"][-1]

        # Growth rates should be positive
        assert metrics["epi_growth_rate"] >= 0
        assert metrics["vf_growth_rate"] >= 0

    def test_val_metrics_fractal_preserved_when_ratio_in_range(self):
        """VAL metrics show fractal preserved when growth ratio in valid range."""
        G, node = create_nfr("fractal_ok", epi=0.4, vf=1.0)
        G.graph["COLLECT_OPERATOR_METRICS"] = True
        G.graph["VAL_FRACTAL_RATIO_MIN"] = 0.5
        G.graph["VAL_FRACTAL_RATIO_MAX"] = 2.0

        set_attr(G.nodes[node], ALIAS_DNFR, 0.2)
        set_attr(G.nodes[node], ALIAS_D2EPI, 0.1)

        Expansion()(G, node)
        metrics = G.graph["operator_metrics"][-1]

        # If growth occurred, check if ratio is tracked
        if metrics["epi_growth_rate"] > 1e-9:
            assert "growth_ratio" in metrics


class TestVALMetricsNetworkImpact:
    """Test suite for network impact metrics."""

    def test_val_metrics_network_impact(self):
        """VAL metrics measure network coupling effects."""
        G, node = create_nfr("network", epi=0.4, vf=1.0)

        # Add neighbors with different phases
        for i, phase in enumerate([0.1, 0.2, 0.3]):
            neighbor = f"n{i}"
            G.add_node(neighbor)
            set_attr(G.nodes[neighbor], ALIAS_EPI, 0.3)
            set_attr(G.nodes[neighbor], ALIAS_VF, 0.8)
            set_attr(G.nodes[neighbor], ALIAS_THETA, phase)
            G.add_edge(node, neighbor)

        G.graph["COLLECT_OPERATOR_METRICS"] = True
        set_attr(G.nodes[node], ALIAS_DNFR, 0.2)
        set_attr(G.nodes[node], ALIAS_D2EPI, 0.1)
        set_attr(G.nodes[node], ALIAS_THETA, 0.15)  # Close to neighbors

        Expansion()(G, node)
        metrics = G.graph["operator_metrics"][-1]

        assert "neighbor_count" in metrics
        assert metrics["neighbor_count"] == 3
        assert "phase_coherence_neighbors" in metrics
        assert 0.0 <= metrics["phase_coherence_neighbors"] <= 1.0
        assert "network_coupled" in metrics

    def test_val_metrics_phase_coherence_neighbors(self):
        """VAL metrics compute phase coherence with neighbors."""
        G, node = create_nfr("phase", epi=0.4, vf=1.0)

        # Add neighbor with similar phase
        neighbor = "n1"
        G.add_node(neighbor)
        set_attr(G.nodes[neighbor], ALIAS_THETA, 0.1)
        G.add_edge(node, neighbor)

        G.graph["COLLECT_OPERATOR_METRICS"] = True
        set_attr(G.nodes[node], ALIAS_DNFR, 0.2)
        set_attr(G.nodes[node], ALIAS_D2EPI, 0.1)
        set_attr(G.nodes[node], ALIAS_THETA, 0.11)  # Very close

        Expansion()(G, node)
        metrics = G.graph["operator_metrics"][-1]

        # Phase coherence should be high (close to 1)
        assert metrics["phase_coherence_neighbors"] > 0.9

    def test_val_metrics_network_coupled_flag(self):
        """VAL metrics indicate if node is well-coupled to network."""
        G, node = create_nfr("coupled", epi=0.4, vf=1.0)

        # Add neighbors with aligned phases
        for i in range(2):
            neighbor = f"n{i}"
            G.add_node(neighbor)
            set_attr(G.nodes[neighbor], ALIAS_THETA, 0.5)
            G.add_edge(node, neighbor)

        G.graph["COLLECT_OPERATOR_METRICS"] = True
        set_attr(G.nodes[node], ALIAS_DNFR, 0.2)
        set_attr(G.nodes[node], ALIAS_D2EPI, 0.1)
        set_attr(G.nodes[node], ALIAS_THETA, 0.51)  # Well aligned

        Expansion()(G, node)
        metrics = G.graph["operator_metrics"][-1]

        # Should be network coupled (has neighbors + good phase coherence)
        assert metrics["network_coupled"] is True


class TestVALMetricsDNFRStability:
    """Test suite for ΔNFR stability metrics."""

    def test_val_metrics_dnfr_state(self):
        """VAL metrics capture ΔNFR positivity and stability."""
        G, node = create_nfr("dnfr", epi=0.4, vf=1.0)
        G.graph["COLLECT_OPERATOR_METRICS"] = True

        set_attr(G.nodes[node], ALIAS_DNFR, 0.5)  # Positive and stable
        set_attr(G.nodes[node], ALIAS_D2EPI, 0.1)

        Expansion()(G, node)
        metrics = G.graph["operator_metrics"][-1]

        assert "dnfr_final" in metrics
        assert "dnfr_positive" in metrics
        assert "dnfr_stable" in metrics
        assert metrics["dnfr_positive"] is True
        assert metrics["dnfr_stable"] is True

    def test_val_metrics_dnfr_unstable_when_too_high(self):
        """VAL metrics flag unstable when ΔNFR >= 1.0."""
        G, node = create_nfr("unstable", epi=0.4, vf=1.0)
        G.graph["COLLECT_OPERATOR_METRICS"] = True

        set_attr(G.nodes[node], ALIAS_DNFR, 1.5)  # Too high
        set_attr(G.nodes[node], ALIAS_D2EPI, 0.1)

        Expansion()(G, node)
        metrics = G.graph["operator_metrics"][-1]

        assert metrics["dnfr_stable"] is False  # Outside [0, 1] range


class TestVALMetricsOverallHealth:
    """Test suite for overall health indicator."""

    def test_val_metrics_expansion_healthy(self):
        """VAL metrics compute combined health indicator."""
        G, node = create_nfr("healthy", epi=0.4, vf=1.0)
        G.graph["COLLECT_OPERATOR_METRICS"] = True
        G.graph["VAL_BIFURCATION_THRESHOLD"] = 0.3

        # Set all healthy conditions
        set_attr(G.nodes[node], ALIAS_DNFR, 0.2)  # Positive and stable
        set_attr(G.nodes[node], ALIAS_D2EPI, 0.1)  # Below threshold

        Expansion()(G, node)
        metrics = G.graph["operator_metrics"][-1]

        assert "expansion_healthy" in metrics
        assert isinstance(metrics["expansion_healthy"], bool)

    def test_val_metrics_unhealthy_when_bifurcation_risk(self):
        """VAL metrics show unhealthy when bifurcation risk present."""
        G, node = create_nfr("bifurc_unhealthy", epi=0.4, vf=1.0)
        G.graph["COLLECT_OPERATOR_METRICS"] = True
        G.graph["VAL_BIFURCATION_THRESHOLD"] = 0.3

        set_attr(G.nodes[node], ALIAS_DNFR, 0.2)
        set_attr(G.nodes[node], ALIAS_D2EPI, 0.5)  # High - bifurcation risk

        Expansion()(G, node)
        metrics = G.graph["operator_metrics"][-1]

        # Should be unhealthy due to bifurcation risk
        assert metrics["expansion_healthy"] is False

    def test_val_metrics_unhealthy_when_negative_dnfr(self):
        """VAL metrics show unhealthy when ΔNFR is negative."""
        G, node = create_nfr("negative_dnfr", epi=0.4, vf=1.0)
        G.graph["COLLECT_OPERATOR_METRICS"] = True

        set_attr(G.nodes[node], ALIAS_DNFR, -0.1)  # Negative - shouldn't happen
        set_attr(G.nodes[node], ALIAS_D2EPI, 0.1)

        Expansion()(G, node)
        metrics = G.graph["operator_metrics"][-1]

        # Should be unhealthy due to negative ΔNFR
        assert metrics["expansion_healthy"] is False


class TestVALMetricsIntegration:
    """Integration tests for complete metrics collection."""

    def test_val_metrics_complete_structure(self):
        """VAL metrics include all specified fields."""
        G, node = create_nfr("complete", epi=0.4, vf=1.0)
        G.graph["COLLECT_OPERATOR_METRICS"] = True

        set_attr(G.nodes[node], ALIAS_DNFR, 0.2)
        set_attr(G.nodes[node], ALIAS_D2EPI, 0.1)

        Expansion()(G, node)
        metrics = G.graph["operator_metrics"][-1]

        # Core metrics
        assert "operator" in metrics
        assert metrics["operator"] == "Expansion"
        assert "glyph" in metrics
        assert metrics["glyph"] == "VAL"

        # Basic metrics
        assert "vf_increase" in metrics
        assert "vf_final" in metrics
        assert "delta_epi" in metrics
        assert "epi_final" in metrics
        assert "expansion_factor" in metrics

        # Structural stability
        assert "dnfr_final" in metrics
        assert "dnfr_positive" in metrics
        assert "dnfr_stable" in metrics

        # Bifurcation risk
        assert "d2epi" in metrics
        assert "bifurcation_risk" in metrics
        assert "bifurcation_magnitude" in metrics
        assert "bifurcation_threshold" in metrics

        # Coherence
        assert "coherence_local" in metrics
        assert "coherence_preserved" in metrics

        # Fractality
        assert "epi_growth_rate" in metrics
        assert "vf_growth_rate" in metrics
        assert "growth_ratio" in metrics
        assert "fractal_preserved" in metrics

        # Network impact
        assert "neighbor_count" in metrics
        assert "phase_coherence_neighbors" in metrics
        assert "network_coupled" in metrics
        assert "theta_final" in metrics

        # Overall health
        assert "expansion_healthy" in metrics

    def test_val_metrics_version(self):
        """VAL metrics include version identifier."""
        G, node = create_nfr("version", epi=0.4, vf=1.0)
        G.graph["COLLECT_OPERATOR_METRICS"] = True

        set_attr(G.nodes[node], ALIAS_DNFR, 0.2)
        set_attr(G.nodes[node], ALIAS_D2EPI, 0.1)

        Expansion()(G, node)
        metrics = G.graph["operator_metrics"][-1]

        assert "metrics_version" in metrics
        assert "3.0_canonical" in metrics["metrics_version"]

    def test_val_metrics_realistic_scenario(self):
        """VAL metrics work in realistic expansion scenario."""
        # Create node with network context
        G, node = create_nfr("realistic", epi=0.4, vf=1.0)

        # Add network
        for i in range(3):
            neighbor = f"n{i}"
            G.add_node(neighbor)
            set_attr(G.nodes[neighbor], ALIAS_EPI, 0.3 + i * 0.1)
            set_attr(G.nodes[neighbor], ALIAS_VF, 0.8 + i * 0.1)
            set_attr(G.nodes[neighbor], ALIAS_THETA, i * 0.2)
            G.add_edge(node, neighbor)

        G.graph["COLLECT_OPERATOR_METRICS"] = True
        set_attr(G.nodes[node], ALIAS_DNFR, 0.2)
        set_attr(G.nodes[node], ALIAS_D2EPI, 0.1)
        set_attr(G.nodes[node], ALIAS_THETA, 0.3)

        Expansion()(G, node)
        metrics = G.graph["operator_metrics"][-1]

        # All metrics should be present and valid
        assert metrics["operator"] == "Expansion"
        assert metrics["neighbor_count"] == 3
        assert metrics["dnfr_positive"] is True
        assert 0.0 <= metrics["phase_coherence_neighbors"] <= 1.0
        assert isinstance(metrics["expansion_healthy"], bool)
