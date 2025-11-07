"""Tests for T'HOL bifurcation logic and structural metabolism."""

from __future__ import annotations

import pytest

from tnfr.constants import DNFR_PRIMARY, EPI_PRIMARY, VF_PRIMARY
from tnfr.dynamics import set_delta_nfr_hook
from tnfr.metrics.emergence import (
    compute_bifurcation_rate,
    compute_emergence_index,
    compute_metabolic_efficiency,
    compute_structural_complexity,
)
from tnfr.operators.definitions import SelfOrganization
from tnfr.structural import create_nfr


class TestBifurcationLogic:
    """Test T'HOL bifurcation when ∂²EPI/∂t² > τ."""

    def test_no_bifurcation_without_history(self):
        """T'HOL without EPI history does not bifurcate."""
        G, node = create_nfr("test", epi=0.5, vf=1.0)

        # Set positive ΔNFR for precondition
        G.nodes[node][DNFR_PRIMARY] = 0.1

        # No epi_history means no acceleration, no bifurcation
        SelfOrganization()(G, node, tau=0.05)

        # Verify no sub-EPIs created
        sub_epis = G.nodes[node].get("sub_epis", [])
        assert len(sub_epis) == 0

    def test_no_bifurcation_with_insufficient_history(self):
        """T'HOL with < 3 history points does not bifurcate."""
        G, node = create_nfr("test", epi=0.5, vf=1.0)

        # Set positive ΔNFR
        G.nodes[node][DNFR_PRIMARY] = 0.1

        # Only 2 history points (need 3 for second derivative)
        G.nodes[node]["epi_history"] = [0.3, 0.4]

        SelfOrganization()(G, node, tau=0.05)

        # Verify no bifurcation
        sub_epis = G.nodes[node].get("sub_epis", [])
        assert len(sub_epis) == 0

    def test_no_bifurcation_below_threshold(self):
        """T'HOL with low acceleration (< τ) does not bifurcate."""
        G, node = create_nfr("test", epi=0.5, vf=1.0)

        # Set positive ΔNFR
        G.nodes[node][DNFR_PRIMARY] = 0.1

        # History with minimal acceleration
        # d²EPI = (0.51 - 2*0.50 + 0.49) = 0.0
        G.nodes[node]["epi_history"] = [0.49, 0.50, 0.51]

        SelfOrganization()(G, node, tau=0.1)

        # Verify no bifurcation (acceleration too low)
        sub_epis = G.nodes[node].get("sub_epis", [])
        assert len(sub_epis) == 0

    def test_bifurcation_above_threshold(self):
        """T'HOL with high acceleration (> τ) spawns sub-EPI."""
        G, node = create_nfr("test", epi=0.5, vf=1.0)

        # Set positive ΔNFR
        G.nodes[node][DNFR_PRIMARY] = 0.2

        # History with strong acceleration
        # d²EPI = abs(0.7 - 2*0.5 + 0.3) = abs(0.0) = 0.0? No, let's make it clearer
        # d²EPI = abs(epi_t - 2*epi_{t-1} + epi_{t-2})
        # For strong acceleration: [0.3, 0.45, 0.7]
        # d²EPI = abs(0.7 - 2*0.45 + 0.3) = abs(0.1) = 0.1
        G.nodes[node]["epi_history"] = [0.3, 0.45, 0.7]

        # Low threshold to trigger bifurcation
        SelfOrganization()(G, node, tau=0.08)

        # Verify bifurcation occurred
        sub_epis = G.nodes[node].get("sub_epis", [])
        assert len(sub_epis) == 1

        # Verify sub-EPI properties
        sub_epi = sub_epis[0]
        assert "epi" in sub_epi
        assert "vf" in sub_epi
        assert "timestamp" in sub_epi
        assert "d2_epi" in sub_epi
        assert "tau" in sub_epi

        # Sub-EPI should be ~25% of parent
        assert abs(sub_epi["epi"] - 0.5 * 0.25) < 0.01

    def test_multiple_bifurcations(self):
        """Multiple T'HOL applications can create multiple sub-EPIs."""
        G, node = create_nfr("test", epi=0.6, vf=1.0)

        # Set positive ΔNFR
        G.nodes[node][DNFR_PRIMARY] = 0.2

        # First bifurcation with strong acceleration
        G.nodes[node]["epi_history"] = [0.4, 0.5, 0.7]
        # d²EPI = abs(0.7 - 2*0.5 + 0.4) = abs(0.1) = 0.1 > 0.08
        SelfOrganization()(G, node, tau=0.08)

        sub_epis = G.nodes[node].get("sub_epis", [])
        assert len(sub_epis) == 1

        # Second bifurcation with different strong acceleration pattern
        # Create a history that shows strong acceleration
        # d²EPI = abs(1.0 - 2*0.7 + 0.5) = abs(0.1) = 0.1 > 0.08
        G.nodes[node]["epi_history"] = [0.5, 0.7, 1.0]
        SelfOrganization()(G, node, tau=0.08)

        sub_epis = G.nodes[node].get("sub_epis", [])
        assert len(sub_epis) == 2

    def test_epi_increases_after_bifurcation(self):
        """Bifurcation increases parent EPI (emergence contribution)."""
        G, node = create_nfr("test", epi=0.5, vf=1.0)

        # Set positive ΔNFR
        G.nodes[node][DNFR_PRIMARY] = 0.2

        # Record EPI before
        epi_before = G.nodes[node][EPI_PRIMARY]

        # Bifurcate
        G.nodes[node]["epi_history"] = [0.3, 0.45, 0.7]
        SelfOrganization()(G, node, tau=0.08)

        # EPI should have increased (glyph + emergence contribution)
        epi_after = G.nodes[node][EPI_PRIMARY]
        # At minimum, emergence contribution of 0.5 * 0.25 * 0.1 = 0.0125
        assert epi_after > epi_before

    def test_custom_threshold_from_graph_config(self):
        """Bifurcation threshold can be configured in graph metadata."""
        G, node = create_nfr("test", epi=0.5, vf=1.0)

        # Set graph-level threshold
        G.graph["THOL_BIFURCATION_THRESHOLD"] = 0.2

        # Set positive ΔNFR
        G.nodes[node][DNFR_PRIMARY] = 0.2

        # Acceleration = 0.1 (below custom threshold)
        G.nodes[node]["epi_history"] = [0.3, 0.45, 0.7]

        # Should not bifurcate (0.1 < 0.2)
        SelfOrganization()(G, node)  # Uses graph config tau=0.2

        sub_epis = G.nodes[node].get("sub_epis", [])
        assert len(sub_epis) == 0


class TestEmergenceMetrics:
    """Test emergence metrics for T'HOL metabolism."""

    def test_structural_complexity_empty(self):
        """Complexity is 0 for nodes without bifurcations."""
        G, node = create_nfr("test", epi=0.5, vf=1.0)
        complexity = compute_structural_complexity(G, node)
        assert complexity == 0

    def test_structural_complexity_with_bifurcations(self):
        """Complexity equals number of sub-EPIs."""
        G, node = create_nfr("test", epi=0.5, vf=1.0)

        # Add sub-EPIs
        G.nodes[node]["sub_epis"] = [
            {"epi": 0.1, "timestamp": 1},
            {"epi": 0.12, "timestamp": 3},
            {"epi": 0.08, "timestamp": 5},
        ]

        complexity = compute_structural_complexity(G, node)
        assert complexity == 3

    def test_bifurcation_rate_zero_without_history(self):
        """Bifurcation rate is 0 without sub-EPIs."""
        G, node = create_nfr("test", epi=0.5, vf=1.0)
        rate = compute_bifurcation_rate(G, node, window=10)
        assert rate == 0.0

    def test_bifurcation_rate_calculation(self):
        """Bifurcation rate counts recent events in window."""
        G, node = create_nfr("test", epi=0.5, vf=1.0)

        # Current time = 15 (from glyph history)
        G.nodes[node]["glyph_history"] = ["AL"] * 15

        # Sub-EPIs: 2 recent (in window), 1 old (outside window)
        G.nodes[node]["sub_epis"] = [
            {"timestamp": 3},  # Outside window (15 - 10 = 5)
            {"timestamp": 8},  # In window
            {"timestamp": 12},  # In window
        ]

        rate = compute_bifurcation_rate(G, node, window=10)
        assert rate == 0.2  # 2 bifurcations in 10 steps

    def test_metabolic_efficiency_zero_without_thol(self):
        """Efficiency is 0 without T'HOL applications."""
        G, node = create_nfr("test", epi=0.5, vf=1.0)
        G.nodes[node]["epi_initial"] = 0.3
        G.nodes[node]["glyph_history"] = ["AL", "IL"]  # No THOL

        efficiency = compute_metabolic_efficiency(G, node)
        assert efficiency == 0.0

    def test_metabolic_efficiency_calculation(self):
        """Efficiency is EPI gain per T'HOL application."""
        G, node = create_nfr("test", epi=0.5, vf=1.0)

        # Initial EPI
        G.nodes[node]["epi_initial"] = 0.3

        # 2 THOL applications
        G.nodes[node]["glyph_history"] = ["THOL", "IL", "THOL"]

        # Current EPI = 0.5, initial = 0.3, gain = 0.2
        # Efficiency = 0.2 / 2 = 0.1
        efficiency = compute_metabolic_efficiency(G, node)
        assert abs(efficiency - 0.1) < 0.01

    def test_emergence_index_combines_metrics(self):
        """Emergence index is geometric mean of complexity, rate, efficiency."""
        G, node = create_nfr("test", epi=0.6, vf=1.0)

        # Setup for emergence
        G.nodes[node]["epi_initial"] = 0.3
        G.nodes[node]["glyph_history"] = ["THOL"] * 10

        # 3 sub-EPIs, 2 recent
        G.nodes[node]["sub_epis"] = [
            {"timestamp": 2},
            {"timestamp": 8},
            {"timestamp": 9},
        ]

        index = compute_emergence_index(G, node)

        # Complexity = 3, Rate = 0.2, Efficiency = 0.03
        # Index = (3 * 0.2 * 0.03)^(1/3) = 0.018^(1/3) ≈ 0.26
        # With epsilon adjustment: slightly higher
        assert index > 0.2
        assert index < 0.5


class TestStructuralMetabolismIntegration:
    """Integration tests for structural metabolism cycles."""

    def test_metabolism_can_be_imported(self):
        """Metabolism module is importable."""
        from tnfr.dynamics.metabolism import (
            StructuralMetabolism,
            adaptive_metabolism,
            cascading_reorganization,
            digest_stimulus,
        )

        assert StructuralMetabolism is not None
        assert digest_stimulus is not None
        assert adaptive_metabolism is not None
        assert cascading_reorganization is not None

    def test_structural_metabolism_init(self):
        """StructuralMetabolism can be initialized."""
        from tnfr.dynamics.metabolism import StructuralMetabolism

        G, node = create_nfr("test", epi=0.5, vf=1.0)
        metabolism = StructuralMetabolism(G, node)

        assert metabolism.G is G
        assert metabolism.node == node
        assert metabolism.metabolic_rate == 1.0

    def test_digest_applies_sequence(self):
        """Digest applies EN → THOL → IL sequence."""
        from tnfr.dynamics.metabolism import StructuralMetabolism

        G, node = create_nfr("test", epi=0.5, vf=1.0)

        # Add neighbor for Reception (EN)
        G.add_node("neighbor", **{EPI_PRIMARY: 0.4, VF_PRIMARY: 1.0})
        G.add_edge(node, "neighbor")

        # Set positive ΔNFR
        G.nodes[node][DNFR_PRIMARY] = 0.2

        # Initialize history
        G.nodes[node]["epi_history"] = [0.3, 0.45, 0.7]

        metabolism = StructuralMetabolism(G, node)
        metabolism.digest(tau=0.08)

        # Verify glyph history shows sequence
        history = G.nodes[node].get("glyph_history", [])
        assert "EN" in history
        assert "THOL" in history
        assert "IL" in history


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
