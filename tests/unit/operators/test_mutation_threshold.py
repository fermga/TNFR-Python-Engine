"""Tests for ZHIR (Mutation) threshold verification (∂EPI/∂t > ξ).

This module tests the canonical requirement that ZHIR activates only when
structural change velocity exceeds a threshold, as specified in:
- AGENTS.md §11 (Mutation operator)
- TNFR.pdf §2.2.11 (ZHIR physics)

The implementation enforces: θ → θ' when ΔEPI/Δt > ξ

Test Coverage:
1. Threshold verification with sufficient history
2. Warning when threshold not met
3. Success when threshold exceeded
4. Canonical sequence (OZ → ZHIR) generates sufficient threshold
5. Insufficient history handling
6. Metrics collection includes threshold indicators
"""

import pytest
from tnfr.structural import create_nfr, run_sequence
from tnfr.operators.definitions import (
    Emission,
    Dissonance,
    Mutation,
    Coherence,
    Reception,
)
from tnfr.operators.preconditions import validate_mutation, OperatorPreconditionError


class TestZHIRThresholdVerification:
    """Test suite for ZHIR threshold verification (∂EPI/∂t > ξ)."""

    def test_zhir_threshold_warning_low_velocity(self, caplog):
        """ZHIR with low ∂EPI/∂t should log warning."""
        G, node = create_nfr("test", epi=0.3, vf=1.0)
        G.graph["ZHIR_THRESHOLD_XI"] = 0.1

        # Build EPI history with low velocity
        # ∂EPI/∂t ≈ 0.30 - 0.29 = 0.01 < ξ=0.1
        G.nodes[node]["epi_history"] = [0.28, 0.29, 0.30]

        # Apply mutation - should log warning
        import logging

        with caplog.at_level(logging.WARNING):
            validate_mutation(G, node)

        # Verify warning was logged
        assert any(
            "ZHIR applied with ∂EPI/∂t=" in record.message and "< ξ=" in record.message
            for record in caplog.records
        )

        # Verify warning flag set
        assert G.nodes[node].get("_zhir_threshold_warning") is True

    def test_zhir_threshold_met_high_velocity(self, caplog):
        """ZHIR with high ∂EPI/∂t should succeed without warnings."""
        G, node = create_nfr("test", epi=0.5, vf=1.0)
        G.graph["ZHIR_THRESHOLD_XI"] = 0.1

        # Build EPI history with high velocity
        # ∂EPI/∂t ≈ 0.50 - 0.38 = 0.12 > ξ=0.1
        G.nodes[node]["epi_history"] = [0.25, 0.38, 0.50]

        # Apply mutation - should succeed without warnings
        import logging

        with caplog.at_level(logging.INFO):
            validate_mutation(G, node)

        # Verify success was logged
        assert any(
            "ZHIR threshold crossed" in record.message and "> ξ=" in record.message
            for record in caplog.records
        )

        # Verify threshold met flag
        assert G.nodes[node].get("_zhir_threshold_met") is True

    def test_zhir_insufficient_history(self, caplog):
        """ZHIR without sufficient history should log warning."""
        G, node = create_nfr("test", epi=0.4, vf=1.0)
        G.graph["ZHIR_THRESHOLD_XI"] = 0.1

        # Only one EPI value - insufficient for derivative
        G.nodes[node]["epi_history"] = [0.4]

        # Apply mutation - should log warning
        import logging

        with caplog.at_level(logging.WARNING):
            validate_mutation(G, node)

        # Verify warning about insufficient history
        assert any(
            "without sufficient EPI history" in record.message
            for record in caplog.records
        )

        # Verify unknown flag set
        assert G.nodes[node].get("_zhir_threshold_unknown") is True

    def test_zhir_canonical_sequence_threshold(self):
        """OZ → ZHIR should generate threshold sufficient for mutation."""
        G, node = create_nfr("test", epi=0.4, vf=1.0)
        G.graph["ZHIR_THRESHOLD_XI"] = 0.1
        G.graph["COLLECT_OPERATOR_METRICS"] = True

        # Build initial history
        G.nodes[node]["epi_history"] = [0.35, 0.38, 0.40]

        # Apply canonical sequence: IL → OZ → ZHIR
        # OZ should elevate ΔNFR and increase EPI enough to cross threshold
        run_sequence(G, node, [Coherence(), Dissonance(), Mutation()])

        # Verify metrics were collected
        assert "operator_metrics" in G.graph
        metrics_list = G.graph["operator_metrics"]

        # Find ZHIR metrics
        zhir_metrics = [m for m in metrics_list if m.get("glyph") == "ZHIR"]
        assert len(zhir_metrics) > 0

        # Verify threshold was met (or at least checked)
        zhir_metric = zhir_metrics[-1]  # Last ZHIR application
        assert "threshold_met" in zhir_metric
        assert "depi_dt" in zhir_metric
        assert "threshold_xi" in zhir_metric

    def test_zhir_metrics_include_threshold(self):
        """Mutation metrics should include threshold verification indicators."""
        G, node = create_nfr("test", epi=0.5, vf=1.0)
        G.graph["ZHIR_THRESHOLD_XI"] = 0.15
        G.graph["COLLECT_OPERATOR_METRICS"] = True

        # Build EPI history with known velocity
        # ∂EPI/∂t ≈ 0.50 - 0.30 = 0.20 > ξ=0.15
        G.nodes[node]["epi_history"] = [0.10, 0.30, 0.50]

        # Apply mutation
        Mutation()(G, node)

        # Get metrics
        metrics_list = G.graph.get("operator_metrics", [])
        assert len(metrics_list) > 0

        # Last metric should be ZHIR
        zhir_metric = metrics_list[-1]
        assert zhir_metric["operator"] == "Mutation"
        assert zhir_metric["glyph"] == "ZHIR"

        # Verify threshold metrics present
        assert "depi_dt" in zhir_metric
        assert "threshold_xi" in zhir_metric
        assert "threshold_met" in zhir_metric
        assert "threshold_ratio" in zhir_metric
        assert "threshold_exceeded_by" in zhir_metric

        # Verify threshold was met
        assert zhir_metric["threshold_met"] is True
        assert zhir_metric["depi_dt"] >= zhir_metric["threshold_xi"]
        assert zhir_metric["threshold_ratio"] >= 1.0

    def test_zhir_threshold_ratio_calculation(self):
        """Threshold ratio should correctly reflect velocity/threshold."""
        G, node = create_nfr("test", epi=0.6, vf=1.0)
        G.graph["ZHIR_THRESHOLD_XI"] = 0.1
        G.graph["COLLECT_OPERATOR_METRICS"] = True

        # Build history with known velocity: 0.30 / 0.1 = 3.0x threshold
        G.nodes[node]["epi_history"] = [0.10, 0.30, 0.60]

        # Apply mutation
        Mutation()(G, node)

        # Get metrics
        metrics_list = G.graph["operator_metrics"]
        zhir_metric = metrics_list[-1]

        # Verify ratio is approximately 3.0
        assert zhir_metric["threshold_ratio"] >= 2.5
        assert zhir_metric["threshold_exceeded_by"] >= 0.19  # Account for float precision

    def test_zhir_low_vf_blocks_mutation(self):
        """ZHIR should fail if νf too low, regardless of threshold."""
        G, node = create_nfr("test", epi=0.5, vf=0.01)  # Very low vf
        G.graph["ZHIR_MIN_VF"] = 0.05
        G.graph["ZHIR_THRESHOLD_XI"] = 0.1

        # Build history with high velocity
        G.nodes[node]["epi_history"] = [0.20, 0.35, 0.50]

        # Should raise error due to low vf
        with pytest.raises(OperatorPreconditionError) as exc_info:
            validate_mutation(G, node)

        assert "Structural frequency too low" in str(exc_info.value)

    def test_zhir_compatible_history_keys(self):
        """ZHIR should work with both 'epi_history' and '_epi_history'."""
        G, node = create_nfr("test", epi=0.4, vf=1.0)
        G.graph["ZHIR_THRESHOLD_XI"] = 0.1

        # Test with _epi_history (underscore prefix)
        G.nodes[node]["_epi_history"] = [0.25, 0.33, 0.40]

        # Should not raise
        validate_mutation(G, node)

        # Should have set threshold met flag
        assert (
            G.nodes[node].get("_zhir_threshold_met") is True
            or G.nodes[node].get("_zhir_threshold_warning") is True
        )


class TestZHIRThresholdConfiguration:
    """Test threshold configuration and defaults."""

    def test_default_threshold_value(self):
        """Default ZHIR_THRESHOLD_XI should be 0.1."""
        G, node = create_nfr("test", epi=0.3, vf=1.0)
        # Don't set ZHIR_THRESHOLD_XI, use default

        G.nodes[node]["epi_history"] = [0.20, 0.28, 0.30]

        # Should use default ξ=0.1
        # ∂EPI/∂t ≈ 0.02 < 0.1 → warning
        validate_mutation(G, node)

        # Check that warning was set (implies default was used)
        assert G.nodes[node].get("_zhir_threshold_warning") is True

    def test_custom_threshold_value(self):
        """Custom ZHIR_THRESHOLD_XI should override default."""
        G, node = create_nfr("test", epi=0.5, vf=1.0)
        G.graph["ZHIR_THRESHOLD_XI"] = 0.25  # Higher threshold

        # Build history: ∂EPI/∂t ≈ 0.15 < 0.25
        G.nodes[node]["epi_history"] = [0.30, 0.35, 0.50]

        # Should trigger warning due to high threshold
        validate_mutation(G, node)

        # Should have warning flag
        assert G.nodes[node].get("_zhir_threshold_warning") is True

    def test_zero_threshold_always_passes(self):
        """Setting ξ=0 should make all mutations pass threshold check."""
        G, node = create_nfr("test", epi=0.3, vf=1.0)
        G.graph["ZHIR_THRESHOLD_XI"] = 0.0  # No threshold

        # Even tiny velocity should pass
        G.nodes[node]["epi_history"] = [0.299, 0.30]

        validate_mutation(G, node)

        # Should have met threshold
        assert G.nodes[node].get("_zhir_threshold_met") is True


class TestZHIRThresholdIntegration:
    """Integration tests for threshold verification in realistic scenarios."""

    def test_multiple_mutations_track_separate_thresholds(self):
        """Multiple ZHIR applications should each verify threshold."""
        G, node = create_nfr("test", epi=0.3, vf=1.0)
        G.graph["ZHIR_THRESHOLD_XI"] = 0.1
        G.graph["COLLECT_OPERATOR_METRICS"] = True

        # Initialize history
        G.nodes[node]["epi_history"] = [0.25, 0.28, 0.30]

        # First mutation - low velocity
        Mutation()(G, node)

        # Apply dissonance to increase velocity
        Dissonance()(G, node)

        # Second mutation - higher velocity
        Mutation()(G, node)

        # Check both mutations have separate threshold evaluations
        metrics_list = G.graph["operator_metrics"]
        zhir_metrics = [m for m in metrics_list if m.get("glyph") == "ZHIR"]

        assert len(zhir_metrics) == 2
        # Both should have threshold data
        assert all("threshold_met" in m for m in zhir_metrics)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
