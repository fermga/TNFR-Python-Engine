"""Tests for THOL vibrational metabolism - canonical network signal digestion.

This test module validates the canonical principle that THOL (Self-organization)
metabolizes external network patterns into internal structure (sub-EPIs).

Canonical Principle (from "El pulso que nos atraviesa", §2.2.10):
    "THOL reorganiza la forma desde dentro, en respuesta a la coherencia
    vibracional del campo. La autoorganización es resonancia estructurada
    desde el interior del nodo."

Test Coverage:
1. Network gradient metabolization (high-EPI neighbors increase sub-EPI)
2. Isolated node fallback (no neighbors = internal bifurcation only)
3. Phase variance complexity bonus (dissonant field = larger sub-EPIs)
4. Metabolic metadata recording (traceability)
5. Configuration controls (enable/disable metabolism, adjust weights)
"""

from __future__ import annotations

import math

import networkx as nx
import pytest

from tnfr.alias import get_attr, set_attr
from tnfr.constants import DNFR_PRIMARY, EPI_PRIMARY, THETA_PRIMARY, VF_PRIMARY
from tnfr.constants.aliases import ALIAS_EPI, ALIAS_THETA
from tnfr.operators.definitions import SelfOrganization
from tnfr.operators.metabolism import capture_network_signals, metabolize_signals_into_subepi
from tnfr.structural import create_nfr, run_sequence


class TestNetworkSignalCapture:
    """Test capture_network_signals() function."""

    def test_capture_returns_none_for_isolated_node(self):
        """Isolated node (no neighbors) returns None."""
        G, node = create_nfr("isolated", epi=0.5, vf=1.0, theta=0.0)

        signals = capture_network_signals(G, node)

        assert signals is None

    def test_capture_computes_epi_gradient(self):
        """EPI gradient = mean(neighbor_EPIs) - node_EPI."""
        G = nx.Graph()
        G.add_node(0, **{EPI_PRIMARY: 0.40, VF_PRIMARY: 1.0, THETA_PRIMARY: 0.0})
        G.add_node(1, **{EPI_PRIMARY: 0.60, VF_PRIMARY: 1.0, THETA_PRIMARY: 0.1})
        G.add_node(2, **{EPI_PRIMARY: 0.80, VF_PRIMARY: 1.0, THETA_PRIMARY: 0.2})
        G.add_edge(0, 1)
        G.add_edge(0, 2)

        signals = capture_network_signals(G, 0)

        # Mean neighbor EPI = (0.60 + 0.80) / 2 = 0.70
        # Gradient = 0.70 - 0.40 = 0.30
        assert signals is not None
        assert abs(signals["epi_gradient"] - 0.30) < 1e-6
        assert signals["mean_neighbor_epi"] == 0.70
        assert signals["neighbor_count"] == 2

    def test_capture_computes_phase_variance(self):
        """Phase variance reflects dispersion of neighbor phases."""
        G = nx.Graph()
        G.add_node(0, **{EPI_PRIMARY: 0.5, VF_PRIMARY: 1.0, THETA_PRIMARY: 0.0})
        # Coherent neighbors (low variance)
        G.add_node(1, **{EPI_PRIMARY: 0.5, VF_PRIMARY: 1.0, THETA_PRIMARY: 0.05})
        G.add_node(2, **{EPI_PRIMARY: 0.5, VF_PRIMARY: 1.0, THETA_PRIMARY: 0.10})
        G.add_edge(0, 1)
        G.add_edge(0, 2)

        signals_coherent = capture_network_signals(G, 0)

        # Dissonant neighbors (high variance)
        G2 = nx.Graph()
        G2.add_node(0, **{EPI_PRIMARY: 0.5, VF_PRIMARY: 1.0, THETA_PRIMARY: 0.0})
        G2.add_node(1, **{EPI_PRIMARY: 0.5, VF_PRIMARY: 1.0, THETA_PRIMARY: 1.0})
        G2.add_node(2, **{EPI_PRIMARY: 0.5, VF_PRIMARY: 1.0, THETA_PRIMARY: 2.0})
        G2.add_edge(0, 1)
        G2.add_edge(0, 2)

        signals_dissonant = capture_network_signals(G2, 0)

        assert signals_coherent["phase_variance"] < signals_dissonant["phase_variance"]

    def test_capture_computes_coupling_strength(self):
        """Coupling strength based on phase alignment."""
        G = nx.Graph()
        G.add_node(0, **{EPI_PRIMARY: 0.5, VF_PRIMARY: 1.0, THETA_PRIMARY: 0.0})
        # In-phase neighbor (strong coupling)
        G.add_node(1, **{EPI_PRIMARY: 0.5, VF_PRIMARY: 1.0, THETA_PRIMARY: 0.1})
        # Out-of-phase neighbor (weak coupling)
        G.add_node(2, **{EPI_PRIMARY: 0.5, VF_PRIMARY: 1.0, THETA_PRIMARY: math.pi})
        G.add_edge(0, 1)
        G.add_edge(0, 2)

        signals = capture_network_signals(G, 0)

        # Mean coupling should be moderate (one strong, one weak)
        assert 0.0 < signals["coupling_strength_mean"] < 1.0


class TestMetabolicDigestion:
    """Test metabolize_signals_into_subepi() function."""

    def test_metabolize_isolated_returns_base_bifurcation(self):
        """Isolated node (signals=None) returns base internal bifurcation."""
        parent_epi = 0.60
        base_expected = parent_epi * 0.25  # 0.15

        sub_epi = metabolize_signals_into_subepi(
            parent_epi=parent_epi,
            signals=None,
            d2_epi=0.12,
            scaling_factor=0.25,
        )

        assert abs(sub_epi - base_expected) < 1e-6

    def test_metabolize_positive_gradient_increases_subepi(self):
        """Positive EPI gradient (pressure from neighbors) increases sub-EPI."""
        parent_epi = 0.60
        base_bifurcation = parent_epi * 0.25  # 0.15

        # Positive gradient (neighbors have higher EPI)
        signals = {
            "epi_gradient": 0.20,
            "phase_variance": 0.02,
            "neighbor_count": 2,
            "coupling_strength_mean": 0.8,
            "mean_neighbor_epi": 0.80,
        }

        sub_epi = metabolize_signals_into_subepi(
            parent_epi=parent_epi,
            signals=signals,
            d2_epi=0.12,
            scaling_factor=0.25,
            gradient_weight=0.15,
            complexity_weight=0.10,
        )

        # Should exceed base bifurcation due to network contribution
        assert sub_epi > base_bifurcation

    def test_metabolize_negative_gradient_decreases_subepi(self):
        """Negative EPI gradient (node higher than neighbors) decreases sub-EPI."""
        parent_epi = 0.60
        base_bifurcation = parent_epi * 0.25  # 0.15

        # Negative gradient (neighbors have lower EPI)
        signals = {
            "epi_gradient": -0.20,
            "phase_variance": 0.02,
            "neighbor_count": 2,
            "coupling_strength_mean": 0.8,
            "mean_neighbor_epi": 0.40,
        }

        sub_epi = metabolize_signals_into_subepi(
            parent_epi=parent_epi,
            signals=signals,
            d2_epi=0.12,
            scaling_factor=0.25,
            gradient_weight=0.15,
            complexity_weight=0.10,
        )

        # Should be less than base bifurcation due to negative contribution
        assert sub_epi < base_bifurcation

    def test_metabolize_high_phase_variance_increases_complexity(self):
        """High phase variance (complex field) increases sub-EPI."""
        parent_epi = 0.60

        # Low variance field
        signals_coherent = {
            "epi_gradient": 0.0,
            "phase_variance": 0.01,
            "neighbor_count": 2,
            "coupling_strength_mean": 0.9,
            "mean_neighbor_epi": 0.60,
        }

        # High variance field
        signals_dissonant = {
            "epi_gradient": 0.0,
            "phase_variance": 0.50,
            "neighbor_count": 2,
            "coupling_strength_mean": 0.5,
            "mean_neighbor_epi": 0.60,
        }

        sub_epi_coherent = metabolize_signals_into_subepi(
            parent_epi=parent_epi,
            signals=signals_coherent,
            d2_epi=0.12,
            scaling_factor=0.25,
            gradient_weight=0.15,
            complexity_weight=0.10,
        )

        sub_epi_dissonant = metabolize_signals_into_subepi(
            parent_epi=parent_epi,
            signals=signals_dissonant,
            d2_epi=0.12,
            scaling_factor=0.25,
            gradient_weight=0.15,
            complexity_weight=0.10,
        )

        # Dissonant field should produce larger sub-EPI
        assert sub_epi_dissonant > sub_epi_coherent

    def test_metabolize_bounds_output(self):
        """Sub-EPI is always bounded to [0, 1]."""
        # Extreme positive gradient
        signals_extreme = {
            "epi_gradient": 10.0,
            "phase_variance": 5.0,
            "neighbor_count": 3,
            "coupling_strength_mean": 1.0,
            "mean_neighbor_epi": 10.0,
        }

        sub_epi = metabolize_signals_into_subepi(
            parent_epi=0.80,
            signals=signals_extreme,
            d2_epi=0.20,
            scaling_factor=0.25,
            gradient_weight=0.15,
            complexity_weight=0.10,
        )

        assert 0.0 <= sub_epi <= 1.0


class TestTHOLMetabolismIntegration:
    """Test THOL operator with vibrational metabolism enabled."""

    def test_thol_metabolizes_network_gradient(self):
        """THOL sub-EPIs should reflect network EPI pressure."""
        # Create network with gradient
        G = nx.Graph()
        G.add_node(
            0, **{EPI_PRIMARY: 0.50, VF_PRIMARY: 1.0, DNFR_PRIMARY: 0.10, THETA_PRIMARY: 0.1}
        )
        G.add_node(
            1, **{EPI_PRIMARY: 0.80, VF_PRIMARY: 1.0, DNFR_PRIMARY: 0.05, THETA_PRIMARY: 0.15}
        )
        G.add_edge(0, 1)

        # Enable history for d²EPI computation with acceleration
        # d²EPI = abs(0.50 - 2*0.42 + 0.30) = abs(0.50 - 0.84 + 0.30) = 0.04
        # But we need > 0.1, so let's use stronger acceleration
        # d²EPI = abs(0.50 - 2*0.38 + 0.20) = abs(0.50 - 0.76 + 0.20) = 0.06 (still too low)
        # d²EPI = abs(0.50 - 2*0.35 + 0.10) = abs(0.50 - 0.70 + 0.10) = 0.10 (borderline)
        # d²EPI = abs(0.50 - 2*0.33 + 0.05) = abs(0.50 - 0.66 + 0.05) = 0.11 (good!)
        G.nodes[0]["epi_history"] = [0.05, 0.33, 0.50]  # Accelerating growth

        # Apply THOL
        SelfOrganization()(G, 0, tau=0.1)

        # Verify sub-EPI was influenced by high-EPI neighbor
        sub_epis = G.nodes[0].get("sub_epis", [])
        assert len(sub_epis) > 0, "Bifurcation should have occurred"

        sub_epi = sub_epis[-1]
        assert sub_epi["metabolized"], "Should have metabolized network signals"
        assert sub_epi["network_signals"] is not None

        # Sub-EPI should be larger than pure internal bifurcation
        # due to positive epi_gradient from neighbor
        base_bifurcation = 0.50 * 0.25  # 0.125
        assert sub_epi["epi"] > base_bifurcation, "Metabolized sub-EPI should exceed base"

    def test_thol_isolated_node_falls_back_to_internal(self):
        """THOL on isolated node should use internal bifurcation only."""
        G, node = create_nfr("isolated", epi=0.50, vf=1.0)
        G.nodes[node][DNFR_PRIMARY] = 0.15
        # Use accelerating history: d²EPI = abs(0.50 - 2*0.33 + 0.05) = 0.11 > 0.1
        G.nodes[node]["epi_history"] = [0.05, 0.33, 0.50]

        SelfOrganization()(G, node, tau=0.1)

        sub_epis = G.nodes[node].get("sub_epis", [])
        assert len(sub_epis) > 0

        sub_epi = sub_epis[-1]
        assert not sub_epi["metabolized"], "Isolated node cannot metabolize network"
        assert sub_epi["network_signals"] is None

    def test_thol_high_phase_variance_increases_complexity(self):
        """THOL should generate larger sub-EPIs in complex (high phase variance) fields."""
        # Create dissonant network
        G_dissonant = nx.Graph()
        G_dissonant.add_node(
            0, **{EPI_PRIMARY: 0.50, VF_PRIMARY: 1.0, DNFR_PRIMARY: 0.15, THETA_PRIMARY: 0.0}
        )
        G_dissonant.add_node(1, **{EPI_PRIMARY: 0.50, VF_PRIMARY: 1.0, THETA_PRIMARY: 0.5})
        G_dissonant.add_node(2, **{EPI_PRIMARY: 0.50, VF_PRIMARY: 1.0, THETA_PRIMARY: 1.5})
        G_dissonant.add_edge(0, 1)
        G_dissonant.add_edge(0, 2)

        # Accelerating history
        G_dissonant.nodes[0]["epi_history"] = [0.05, 0.33, 0.50]

        SelfOrganization()(G_dissonant, 0, tau=0.1)

        sub_epi_dissonant = G_dissonant.nodes[0]["sub_epis"][-1]["epi"]

        # Compare with coherent field
        G_coherent = nx.Graph()
        G_coherent.add_node(
            0, **{EPI_PRIMARY: 0.50, VF_PRIMARY: 1.0, DNFR_PRIMARY: 0.15, THETA_PRIMARY: 0.0}
        )
        G_coherent.add_node(1, **{EPI_PRIMARY: 0.50, VF_PRIMARY: 1.0, THETA_PRIMARY: 0.05})
        G_coherent.add_node(2, **{EPI_PRIMARY: 0.50, VF_PRIMARY: 1.0, THETA_PRIMARY: 0.10})
        G_coherent.add_edge(0, 1)
        G_coherent.add_edge(0, 2)
        G_coherent.nodes[0]["epi_history"] = [0.05, 0.33, 0.50]

        SelfOrganization()(G_coherent, 0, tau=0.1)

        sub_epi_coherent = G_coherent.nodes[0]["sub_epis"][-1]["epi"]

        # High variance field should produce more complex (larger) sub-EPI
        assert sub_epi_dissonant > sub_epi_coherent

    def test_thol_metabolism_can_be_disabled(self):
        """THOL metabolism can be disabled via configuration."""
        # Create network with gradient
        G = nx.Graph()
        G.add_node(
            0, **{EPI_PRIMARY: 0.50, VF_PRIMARY: 1.0, DNFR_PRIMARY: 0.10, THETA_PRIMARY: 0.1}
        )
        G.add_node(1, **{EPI_PRIMARY: 0.80, VF_PRIMARY: 1.0, THETA_PRIMARY: 0.15})
        G.add_edge(0, 1)

        # Disable metabolism
        G.graph["THOL_METABOLIC_ENABLED"] = False

        # Accelerating history
        G.nodes[0]["epi_history"] = [0.05, 0.33, 0.50]

        SelfOrganization()(G, 0, tau=0.1)

        sub_epis = G.nodes[0].get("sub_epis", [])
        assert len(sub_epis) > 0

        sub_epi = sub_epis[-1]
        assert not sub_epi["metabolized"], "Metabolism disabled"
        assert sub_epi["network_signals"] is None

        # Should use base bifurcation
        base_expected = 0.50 * 0.25
        assert abs(sub_epi["epi"] - base_expected) < 1e-6

    def test_thol_respects_custom_weights(self):
        """THOL respects custom metabolic weights from graph config."""
        G = nx.Graph()
        G.add_node(
            0, **{EPI_PRIMARY: 0.50, VF_PRIMARY: 1.0, DNFR_PRIMARY: 0.10, THETA_PRIMARY: 0.0}
        )
        G.add_node(1, **{EPI_PRIMARY: 0.80, VF_PRIMARY: 1.0, THETA_PRIMARY: 0.1})
        G.add_edge(0, 1)

        # Custom high weights
        G.graph["THOL_METABOLIC_GRADIENT_WEIGHT"] = 0.30
        G.graph["THOL_METABOLIC_COMPLEXITY_WEIGHT"] = 0.20

        # Accelerating history
        G.nodes[0]["epi_history"] = [0.05, 0.33, 0.50]

        SelfOrganization()(G, 0, tau=0.1)

        sub_epi_high_weights = G.nodes[0]["sub_epis"][-1]["epi"]

        # Compare with default weights
        G2 = nx.Graph()
        G2.add_node(
            0, **{EPI_PRIMARY: 0.50, VF_PRIMARY: 1.0, DNFR_PRIMARY: 0.10, THETA_PRIMARY: 0.0}
        )
        G2.add_node(1, **{EPI_PRIMARY: 0.80, VF_PRIMARY: 1.0, THETA_PRIMARY: 0.1})
        G2.add_edge(0, 1)
        G2.nodes[0]["epi_history"] = [0.05, 0.33, 0.50]

        SelfOrganization()(G2, 0, tau=0.1)

        sub_epi_default = G2.nodes[0]["sub_epis"][-1]["epi"]

        # Higher weights should produce larger sub-EPI (due to positive gradient)
        assert sub_epi_high_weights > sub_epi_default

    def test_thol_metadata_records_metabolic_state(self):
        """Sub-EPI records contain metabolic metadata for traceability."""
        G = nx.Graph()
        G.add_node(
            0, **{EPI_PRIMARY: 0.50, VF_PRIMARY: 1.0, DNFR_PRIMARY: 0.10, THETA_PRIMARY: 0.0}
        )
        G.add_node(1, **{EPI_PRIMARY: 0.70, VF_PRIMARY: 1.0, THETA_PRIMARY: 0.1})
        G.add_edge(0, 1)

        # Accelerating history
        G.nodes[0]["epi_history"] = [0.05, 0.33, 0.50]

        SelfOrganization()(G, 0, tau=0.1)

        sub_epi = G.nodes[0]["sub_epis"][-1]

        # Verify metadata structure
        assert "metabolized" in sub_epi
        assert "network_signals" in sub_epi
        assert sub_epi["metabolized"] is True
        assert sub_epi["network_signals"] is not None

        # Verify signal structure
        signals = sub_epi["network_signals"]
        assert "epi_gradient" in signals
        assert "phase_variance" in signals
        assert "neighbor_count" in signals
        assert "coupling_strength_mean" in signals
        assert "mean_neighbor_epi" in signals


class TestBackwardCompatibility:
    """Test that metabolism doesn't break existing THOL behavior."""

    def test_thol_still_requires_acceleration_threshold(self):
        """THOL doesn't bifurcate if d²EPI/dt² < τ, even with metabolism."""
        G = nx.Graph()
        G.add_node(
            0, **{EPI_PRIMARY: 0.50, VF_PRIMARY: 1.0, DNFR_PRIMARY: 0.10, THETA_PRIMARY: 0.0}
        )
        G.add_node(1, **{EPI_PRIMARY: 0.80, VF_PRIMARY: 1.0, THETA_PRIMARY: 0.1})
        G.add_edge(0, 1)

        # History with low acceleration
        G.nodes[0]["epi_history"] = [0.49, 0.50, 0.51]

        SelfOrganization()(G, 0, tau=0.1)

        # No bifurcation despite strong gradient
        sub_epis = G.nodes[0].get("sub_epis", [])
        assert len(sub_epis) == 0

    def test_thol_parent_epi_still_increases(self):
        """Parent EPI still increases after bifurcation (emergence contribution)."""
        G = nx.Graph()
        G.add_node(
            0, **{EPI_PRIMARY: 0.50, VF_PRIMARY: 1.0, DNFR_PRIMARY: 0.10, THETA_PRIMARY: 0.0}
        )
        G.add_node(1, **{EPI_PRIMARY: 0.70, VF_PRIMARY: 1.0, THETA_PRIMARY: 0.1})
        G.add_edge(0, 1)

        # Accelerating history
        G.nodes[0]["epi_history"] = [0.05, 0.33, 0.50]

        epi_before = float(get_attr(G.nodes[0], ALIAS_EPI, 0.0))

        SelfOrganization()(G, 0, tau=0.1)

        epi_after = float(get_attr(G.nodes[0], ALIAS_EPI, 0.0))

        # Parent EPI should have increased
        assert epi_after > epi_before
