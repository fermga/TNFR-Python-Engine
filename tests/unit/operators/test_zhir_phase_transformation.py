"""Tests for ZHIR (Mutation) active phase transformation (θ → θ').

This module tests the canonical requirement that ZHIR actively transforms
the phase θ → θ' based on structural dynamics (ΔNFR), as specified in:
- AGENTS.md §11 (Mutation operator: "Effect: θ → θ' when ΔEPI/Δt > ξ")
- TNFR.pdf §2.2.11 (ZHIR physics)

The implementation must APPLY phase transformation, not just measure it.

Test Coverage:
1. Phase transformation is applied (theta changes)
2. Transformation direction based on ΔNFR sign
3. Transformation magnitude proportional to parameters
4. Regime change detection (quadrant crossings)
5. Backward compatibility with fixed shifts
6. Reproducibility with seeds
7. Metrics capture transformation telemetry
"""

import math
import pytest
from tnfr.structural import create_nfr, run_sequence
from tnfr.operators.definitions import Mutation, Dissonance, Coherence
from tnfr.constants.aliases import ALIAS_THETA, ALIAS_DNFR


class TestZHIRPhaseTransformation:
    """Test suite for ZHIR active phase transformation."""

    def test_zhir_applies_phase_transformation(self):
        """ZHIR MUST transform phase θ → θ', not just measure it."""
        G, node = create_nfr("test", epi=0.5, vf=1.0)
        theta_before = 0.0
        G.nodes[node]["theta"] = theta_before
        G.nodes[node]["delta_nfr"] = 0.3  # Positive pressure

        # Apply mutation
        Mutation()(G, node)

        theta_after = G.nodes[node]["theta"]

        # CRITICAL: Phase MUST have changed
        assert theta_after != theta_before, "ZHIR must transform phase, not just measure"
        assert abs(theta_after - theta_before) > 0.1, "Phase change must be significant"

    def test_zhir_positive_dnfr_forward_shift(self):
        """Positive ΔNFR → forward phase shift (positive direction)."""
        G, node = create_nfr("test", epi=0.5, vf=1.0)
        theta_before = math.pi / 4  # 45°
        G.nodes[node]["theta"] = theta_before
        G.nodes[node]["delta_nfr"] = 0.5  # Positive reorganization pressure

        # Apply mutation
        Mutation()(G, node)

        theta_after = G.nodes[node]["theta"]

        # Phase should shift forward (increase)
        # Note: May wrap around 2π, but shift should be positive
        shift = theta_after - theta_before
        if shift < -math.pi:  # Handle wrapping
            shift += 2 * math.pi

        assert shift > 0, f"Positive ΔNFR should shift phase forward, got shift={shift}"

    def test_zhir_negative_dnfr_backward_shift(self):
        """Negative ΔNFR → backward phase shift (negative direction)."""
        G, node = create_nfr("test", epi=0.5, vf=1.0)
        theta_before = math.pi  # 180°
        G.nodes[node]["theta"] = theta_before
        G.nodes[node]["delta_nfr"] = -0.5  # Negative reorganization pressure

        # Apply mutation
        Mutation()(G, node)

        theta_after = G.nodes[node]["theta"]

        # Phase should shift backward (decrease, modulo wrapping)
        shift = theta_after - theta_before
        if shift > math.pi:  # Handle wrapping
            shift -= 2 * math.pi

        assert shift < 0, f"Negative ΔNFR should shift phase backward, got shift={shift}"

    def test_zhir_regime_change_detection(self):
        """ZHIR should detect regime changes (quadrant crossings)."""
        G, node = create_nfr("test", epi=0.5, vf=1.0)
        G.graph["COLLECT_OPERATOR_METRICS"] = True

        # Start near quadrant boundary (π/2 - ε)
        theta_before = math.pi / 2 - 0.1
        G.nodes[node]["theta"] = theta_before
        G.nodes[node]["delta_nfr"] = 0.4  # Strong positive pressure

        # Build EPI history to satisfy threshold
        G.nodes[node]["epi_history"] = [0.3, 0.4, 0.5]

        # Apply mutation - should cross into next quadrant
        Mutation()(G, node)

        # Check telemetry
        assert "_zhir_regime_changed" in G.nodes[node]

        # Get metrics
        metrics = G.graph.get("operator_metrics", [])
        assert len(metrics) > 0

        zhir_metric = metrics[-1]
        assert zhir_metric["glyph"] == "ZHIR"

        # Should have regime change indicators
        assert "theta_regime_change" in zhir_metric
        assert "regime_before" in zhir_metric
        assert "regime_after" in zhir_metric

    def test_zhir_no_regime_change_small_shift(self):
        """Small shifts within same quadrant should not trigger regime change."""
        G, node = create_nfr("test", epi=0.5, vf=1.0)
        G.graph["COLLECT_OPERATOR_METRICS"] = True

        # Start in middle of quadrant
        theta_before = math.pi / 4  # 45°, in quadrant 0
        G.nodes[node]["theta"] = theta_before
        G.nodes[node]["delta_nfr"] = 0.1  # Small pressure
        G.nodes[node]["epi_history"] = [0.4, 0.45, 0.5]

        # Apply mutation with small shift factor
        G.graph["GLYPH_FACTORS"] = {"ZHIR_theta_shift_factor": 0.1}
        Mutation()(G, node)

        theta_after = G.nodes[node]["theta"]

        # Should still be in same quadrant
        regime_before = int(theta_before // (math.pi / 2))
        regime_after = int(theta_after // (math.pi / 2))

        assert regime_before == regime_after, "Small shift should stay in same regime"

        # Metrics should reflect no regime change
        metrics = G.graph["operator_metrics"]
        zhir_metric = metrics[-1]
        assert zhir_metric["theta_regime_change"] is False

    def test_zhir_phase_wrapping(self):
        """Phase transformation should wrap correctly at 2π boundary."""
        G, node = create_nfr("test", epi=0.5, vf=1.0)

        # Start near 2π boundary
        theta_before = 2 * math.pi - 0.2
        G.nodes[node]["theta"] = theta_before
        G.nodes[node]["delta_nfr"] = 0.5  # Positive shift
        G.nodes[node]["epi_history"] = [0.3, 0.4, 0.5]

        # Apply mutation
        Mutation()(G, node)

        theta_after = G.nodes[node]["theta"]

        # Phase should wrap to small positive value
        assert 0 <= theta_after < 2 * math.pi, "Phase must be in [0, 2π)"
        assert theta_after < 1.0, f"Should have wrapped to small value, got {theta_after}"

    def test_zhir_backward_compatibility_fixed_shift(self):
        """Explicit ZHIR_theta_shift should override canonical behavior."""
        G, node = create_nfr("test", epi=0.5, vf=1.0)
        G.graph["COLLECT_OPERATOR_METRICS"] = True

        theta_before = 0.0
        G.nodes[node]["theta"] = theta_before
        G.nodes[node]["delta_nfr"] = 0.8  # This should be ignored
        G.nodes[node]["epi_history"] = [0.3, 0.4, 0.5]

        # Apply fixed shift (backward compatibility)
        fixed_shift = math.pi / 2  # 90°
        G.graph["GLYPH_FACTORS"] = {"ZHIR_theta_shift": fixed_shift}

        Mutation()(G, node)

        theta_after = G.nodes[node]["theta"]

        # Should apply fixed shift, ignoring ΔNFR
        assert abs(theta_after - fixed_shift) < 0.01, "Fixed shift should be exact"

        # Metrics should indicate fixed mode
        metrics = G.graph["operator_metrics"]
        zhir_metric = metrics[-1]
        assert zhir_metric.get("transformation_mode") == "fixed"

    def test_zhir_magnitude_proportional_to_factor(self):
        """Transformation magnitude should scale with theta_shift_factor."""
        G, node = create_nfr("test", epi=0.5, vf=1.0)

        theta_before = 1.0
        G.nodes[node]["theta"] = theta_before
        G.nodes[node]["delta_nfr"] = 0.5
        G.nodes[node]["epi_history"] = [0.3, 0.4, 0.5]

        # Small factor
        G.graph["GLYPH_FACTORS"] = {"ZHIR_theta_shift_factor": 0.1}
        Mutation()(G, node)
        theta_small = G.nodes[node]["theta"]
        shift_small = abs(theta_small - theta_before)

        # Reset
        G2, node2 = create_nfr("test2", epi=0.5, vf=1.0)
        G2.nodes[node2]["theta"] = theta_before
        G2.nodes[node2]["delta_nfr"] = 0.5
        G2.nodes[node2]["epi_history"] = [0.3, 0.4, 0.5]

        # Large factor
        G2.graph["GLYPH_FACTORS"] = {"ZHIR_theta_shift_factor": 0.5}
        Mutation()(G2, node2)
        theta_large = G2.nodes[node2]["theta"]
        shift_large = abs(theta_large - theta_before)

        # Larger factor should produce larger shift
        assert shift_large > shift_small, "Larger factor should produce larger phase shift"


class TestZHIRMetricsTelemetry:
    """Test metrics capture transformation telemetry."""

    def test_zhir_metrics_include_transformation_data(self):
        """Mutation metrics should include detailed transformation telemetry."""
        G, node = create_nfr("test", epi=0.5, vf=1.0)
        G.graph["COLLECT_OPERATOR_METRICS"] = True

        G.nodes[node]["theta"] = 1.0
        G.nodes[node]["delta_nfr"] = 0.3
        G.nodes[node]["epi_history"] = [0.3, 0.4, 0.5]

        Mutation()(G, node)

        metrics = G.graph["operator_metrics"]
        assert len(metrics) > 0

        zhir_metric = metrics[-1]
        assert zhir_metric["operator"] == "Mutation"

        # Check transformation telemetry
        required_keys = [
            "theta_shift",
            "theta_shift_signed",
            "theta_before",
            "theta_after",
            "theta_regime_change",
            "regime_before",
            "regime_after",
            "transformation_mode",
        ]

        for key in required_keys:
            assert key in zhir_metric, f"Missing telemetry key: {key}"

    def test_zhir_metrics_signed_shift(self):
        """Metrics should include signed shift (preserving direction)."""
        G, node = create_nfr("test", epi=0.5, vf=1.0)
        G.graph["COLLECT_OPERATOR_METRICS"] = True

        theta_before = 1.0
        G.nodes[node]["theta"] = theta_before
        G.nodes[node]["delta_nfr"] = -0.4  # Negative ΔNFR
        G.nodes[node]["epi_history"] = [0.3, 0.4, 0.5]

        Mutation()(G, node)

        metrics = G.graph["operator_metrics"]
        zhir_metric = metrics[-1]

        # Signed shift should be negative
        assert zhir_metric["theta_shift_signed"] < 0, "Negative ΔNFR should produce negative shift"


class TestZHIRReproducibility:
    """Test deterministic behavior with seeds."""

    def test_zhir_reproducible_with_seed(self):
        """Same seed should produce identical transformations."""
        import random
        import numpy as np

        # First run
        random.seed(42)
        np.random.seed(42)
        G1, node1 = create_nfr("test1", epi=0.5, vf=1.0)
        G1.nodes[node1]["theta"] = 1.0
        G1.nodes[node1]["delta_nfr"] = 0.3
        G1.nodes[node1]["epi_history"] = [0.3, 0.4, 0.5]
        Mutation()(G1, node1)
        theta1 = G1.nodes[node1]["theta"]

        # Second run with same seed
        random.seed(42)
        np.random.seed(42)
        G2, node2 = create_nfr("test2", epi=0.5, vf=1.0)
        G2.nodes[node2]["theta"] = 1.0
        G2.nodes[node2]["delta_nfr"] = 0.3
        G2.nodes[node2]["epi_history"] = [0.3, 0.4, 0.5]
        Mutation()(G2, node2)
        theta2 = G2.nodes[node2]["theta"]

        # Should be identical
        assert abs(theta1 - theta2) < 1e-10, "Same seed should produce identical results"


class TestZHIRCanonicalSequences:
    """Test ZHIR in canonical operator sequences."""

    def test_oz_zhir_applies_transformation(self):
        """OZ → ZHIR sequence should apply phase transformation."""
        G, node = create_nfr("test", epi=0.4, vf=1.0)
        G.graph["COLLECT_OPERATOR_METRICS"] = True

        theta_before = G.nodes[node]["theta"]
        G.nodes[node]["epi_history"] = [0.35, 0.38, 0.40]

        # Apply canonical destabilizer → mutation sequence
        run_sequence(G, node, [Dissonance(), Mutation()])

        theta_after = G.nodes[node]["theta"]

        # Phase must have changed
        assert theta_after != theta_before, "OZ → ZHIR must transform phase"

        # Check metrics captured transformation
        metrics = G.graph["operator_metrics"]
        zhir_metrics = [m for m in metrics if m.get("glyph") == "ZHIR"]
        assert len(zhir_metrics) > 0
        assert zhir_metrics[-1]["theta_shift"] > 0

    def test_il_oz_zhir_il_full_sequence(self):
        """Full stabilize-destabilize-mutate-stabilize sequence."""
        G, node = create_nfr("test", epi=0.4, vf=1.0)
        G.graph["COLLECT_OPERATOR_METRICS"] = True

        theta_initial = G.nodes[node]["theta"]
        G.nodes[node]["epi_history"] = [0.35, 0.38, 0.40]

        # IL → OZ → ZHIR → IL (canonical mutation pattern)
        run_sequence(G, node, [Coherence(), Dissonance(), Mutation(), Coherence()])

        theta_final = G.nodes[node]["theta"]

        # Phase should have transformed during ZHIR
        assert theta_final != theta_initial, "ZHIR in full sequence must transform phase"

        # Verify sequence was successful
        metrics = G.graph["operator_metrics"]
        operators = [m.get("operator") for m in metrics]

        # Should have all four operators
        assert "Coherence" in operators
        assert "Dissonance" in operators
        assert "Mutation" in operators


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
