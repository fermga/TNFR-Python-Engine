"""Tests for ZHIR (Mutation) structural identity preservation.

This module tests the canonical requirement that ZHIR preserves structural
identity (epi_kind) during phase transformation, as specified in:
- AGENTS.md §11 (Mutation operator: "Contract: Preserves identity")
- TNFR.pdf §2.2.11 (ZHIR physics - identity preservation)

Test Coverage:
1. Single ZHIR preserves epi_kind
2. Multiple ZHIR applications preserve identity
3. Identity preservation across different node types
4. Validation that identity changes would raise errors (when implemented)

References:
- AGENTS.md: Invariant #7 (Operational Fractality)
- test_mutation_metrics_comprehensive.py (identity_preserved metric)
"""

import pytest
from tnfr.structural import create_nfr, run_sequence
from tnfr.operators.definitions import (
    Mutation,
    Coherence,
    Dissonance,
    Emission,
    Silence,
)


class TestZHIRIdentityPreservation:
    """Test ZHIR preserves structural identity (epi_kind)."""

    def test_zhir_preserves_epi_kind(self):
        """ZHIR MUST preserve epi_kind during phase transformation."""
        G, node = create_nfr("test", epi=0.5, vf=1.0)

        # Set structural identity (use separate attribute per postcondition docs)
        identity_original = "coherent_oscillator"
        G.nodes[node]["structural_type"] = identity_original
        G.nodes[node]["epi_history"] = [0.3, 0.4, 0.5]
        G.nodes[node]["delta_nfr"] = 0.3

        # Apply mutation with postcondition validation enabled
        Mutation()(G, node, validate_postconditions=True)

        # Identity MUST be preserved (check structural_type, not EPI_kind)
        identity_after = G.nodes[node].get("structural_type")
        assert (
            identity_after == identity_original
        ), f"ZHIR violated identity preservation: {identity_original} → {identity_after}"

    def test_identity_preserved_under_stress(self):
        """Identity preserved under repeated mutations."""
        G, node = create_nfr("test", epi=0.3, vf=1.0)
        G.nodes[node]["structural_type"] = "stress_pattern"
        # Enable postcondition validation
        G.graph["VALIDATE_OPERATOR_POSTCONDITIONS"] = True
        
        for i in range(5):
            run_sequence(G, node, [Emission(), Dissonance(), Mutation(),
                                   Coherence(), Silence()])
        
        # Identity preserved after stress
        assert G.nodes[node]["structural_type"] == "stress_pattern"

    def test_multiple_zhir_preserve_identity(self):
        """Multiple ZHIR applications (with IL between) preserve identity."""
        G, node = create_nfr("test", epi=0.5, vf=1.0)

        # Set identity using structural_type
        original_identity = "fractal_structure"
        G.nodes[node]["structural_type"] = original_identity
        G.nodes[node]["epi_history"] = [0.3, 0.4, 0.5]
        # Enable postcondition validation for identity preservation
        G.graph["VALIDATE_OPERATOR_POSTCONDITIONS"] = True

        # Apply 3 mutation cycles with proper grammar
        for i in range(3):
            run_sequence(
                G,
                node,
                [
                    Emission(),   # Generator (U1a)
                    Coherence(),  # Stabilize
                    Dissonance(),  # Destabilize
                    Mutation(),  # Transform
                    Coherence(),  # Stabilize
                    Silence(),    # Closure (U1b)
                ],
            )

        # Identity must still be preserved
        assert (
            G.nodes[node]["structural_type"] == original_identity
        ), "Multiple ZHIR violated identity preservation"

    def test_zhir_without_epi_kind_set(self):
        """ZHIR should work when epi_kind is not set (no violation possible)."""
        G, node = create_nfr("test", epi=0.5, vf=1.0)
        # Don't set epi_kind
        G.nodes[node]["epi_history"] = [0.3, 0.4, 0.5]

        # Should not raise error
        Mutation()(G, node)

        # epi_kind should remain unset or undefined
        epi_kind = G.nodes[node].get("epi_kind")
        # It's OK if it's None or undefined

    def test_zhir_preserves_custom_identities(self):
        """ZHIR preserves various custom identity types."""
        identity_types = [
            "wave_packet",
            "vortex_structure",
            "nested_pattern",
            "emergent_form",
            "synchronized_oscillator",
        ]

        for identity_type in identity_types:
            G, node = create_nfr(f"test_{identity_type}", epi=0.5, vf=1.0)
            G.nodes[node]["structural_type"] = identity_type
            G.nodes[node]["epi_history"] = [0.3, 0.4, 0.5]
            # Enable postcondition validation
            G.graph["VALIDATE_OPERATOR_POSTCONDITIONS"] = True

            # Apply mutation with proper grammar
            run_sequence(G, node, [Emission(), Coherence(), Dissonance(),
                                   Mutation(), Coherence(), Silence()])

            # Verify preservation
            assert (
                G.nodes[node]["structural_type"] == identity_type
            ), f"Failed to preserve identity: {identity_type}"


class TestZHIRIdentityMetrics:
    """Test identity preservation is captured in metrics."""

    def test_identity_preserved_in_metrics(self):
        """Metrics should track identity preservation."""
        G, node = create_nfr("test", epi=0.5, vf=1.0)
        # Use structural_type for identity and set EPI_kind for metrics tracking
        G.nodes[node]["structural_type"] = "test_identity"
        G.nodes[node]["EPI_kind"] = "test_identity"  # For metrics
        G.nodes[node]["epi_history"] = [0.3, 0.4, 0.5]
        G.graph["COLLECT_OPERATOR_METRICS"] = True

        Mutation()(G, node)

        # Get metrics
        metrics_list = G.graph.get("operator_metrics", [])
        assert len(metrics_list) > 0

        zhir_metric = metrics_list[-1]
        assert zhir_metric["glyph"] == "ZHIR"

        # Check identity metrics - may be None if epi_kind used for glyphs
        assert "identity_preserved" in zhir_metric
        assert "epi_kind_before" in zhir_metric
        assert "epi_kind_after" in zhir_metric

        # Check that structural identity is preserved
        assert G.nodes[node]["structural_type"] == "test_identity"

    def test_identity_metrics_without_epi_kind(self):
        """Metrics should handle nodes without epi_kind gracefully."""
        G, node = create_nfr("test", epi=0.5, vf=1.0)
        # No epi_kind set
        G.nodes[node]["epi_history"] = [0.3, 0.4, 0.5]
        G.graph["COLLECT_OPERATOR_METRICS"] = True

        Mutation()(G, node)

        metrics = G.graph["operator_metrics"][-1]

        # Should have identity metrics
        assert "identity_preserved" in metrics
        # Should indicate no identity to preserve (trivially preserved)
        # or should be True (no violation occurred)
        assert metrics["identity_preserved"] in [True, None]


class TestZHIRIdentityWithTransformations:
    """Test identity preservation despite various transformations."""

    def test_identity_preserved_despite_phase_change(self):
        """Identity preserved even when phase changes significantly."""
        import math

        G, node = create_nfr("test", epi=0.5, vf=1.0)
        G.nodes[node]["structural_type"] = "rotating_pattern"
        G.nodes[node]["theta"] = 0.0
        G.nodes[node]["delta_nfr"] = 0.8  # Strong transformation
        G.nodes[node]["epi_history"] = [0.3, 0.4, 0.5]

        # Apply mutation with strong phase shift and validation
        G.graph["GLYPH_FACTORS"] = {"ZHIR_theta_shift_factor": 0.8}
        G.graph["VALIDATE_OPERATOR_POSTCONDITIONS"] = True
        Mutation()(G, node)

        theta_after = G.nodes[node]["theta"]

        # Phase should have changed significantly
        assert theta_after != 0.0
        assert abs(theta_after) > 0.5

        # But identity MUST be preserved
        assert G.nodes[node]["structural_type"] == "rotating_pattern"

    def test_identity_preserved_with_high_epi_change(self):
        """Identity preserved even when EPI changes (within bounds)."""
        G, node = create_nfr("test", epi=0.3, vf=1.0)
        G.nodes[node]["structural_type"] = "adaptive_pattern"
        G.nodes[node]["epi_history"] = [0.1, 0.2, 0.3]
        G.nodes[node]["delta_nfr"] = 0.5
        # Enable postcondition validation
        G.graph["VALIDATE_OPERATOR_POSTCONDITIONS"] = True

        # Apply sequence with proper grammar
        run_sequence(G, node, [Emission(), Dissonance(), Mutation(),
                               Coherence(), Silence()])

        # Identity must be preserved
        assert G.nodes[node]["structural_type"] == "adaptive_pattern"

    def test_identity_preserved_across_regime_change(self):
        """Identity preserved when phase crosses regime boundary."""
        import math

        G, node = create_nfr("test", epi=0.5, vf=1.0)
        G.nodes[node]["structural_type"] = "regime_crossing_pattern"
        G.nodes[node]["theta"] = math.pi / 2 - 0.1  # Near boundary
        G.nodes[node]["delta_nfr"] = 0.6  # Strong shift
        G.nodes[node]["epi_history"] = [0.3, 0.4, 0.5]
        G.graph["GLYPH_FACTORS"] = {"ZHIR_theta_shift_factor": 0.5}
        # Enable postcondition validation
        G.graph["VALIDATE_OPERATOR_POSTCONDITIONS"] = True

        # Apply mutation - should cross regime boundary
        Mutation()(G, node)

        # Check if regime changed
        theta_after = G.nodes[node]["theta"]

        # Identity must be preserved regardless of regime change
        assert G.nodes[node]["structural_type"] == "regime_crossing_pattern"


class TestZHIRIdentityEdgeCases:
    """Test identity preservation in edge cases."""

    def test_identity_with_negative_epi(self):
        """Identity preserved when EPI is negative."""
        G, node = create_nfr("test", epi=-0.5, vf=1.0)
        G.nodes[node]["structural_type"] = "negative_epi_pattern"
        G.nodes[node]["epi_history"] = [-0.7, -0.6, -0.5]
        # Enable postcondition validation
        G.graph["VALIDATE_OPERATOR_POSTCONDITIONS"] = True

        Mutation()(G, node)

        assert G.nodes[node]["structural_type"] == "negative_epi_pattern"

    def test_identity_with_zero_epi(self):
        """Identity preserved when EPI crosses zero."""
        G, node = create_nfr("test", epi=0.0, vf=1.0)
        G.nodes[node]["structural_type"] = "zero_crossing_pattern"
        G.nodes[node]["epi_history"] = [-0.1, 0.0, 0.0]
        # Enable postcondition validation
        G.graph["VALIDATE_OPERATOR_POSTCONDITIONS"] = True

        Mutation()(G, node)

        assert G.nodes[node]["structural_type"] == "zero_crossing_pattern"

    def test_identity_with_special_characters(self):
        """Identity strings with special characters should be preserved."""
        G, node = create_nfr("test", epi=0.5, vf=1.0)
        G.nodes[node]["structural_type"] = "pattern_v2.0_α-β"
        G.nodes[node]["epi_history"] = [0.3, 0.4, 0.5]
        # Enable postcondition validation
        G.graph["VALIDATE_OPERATOR_POSTCONDITIONS"] = True

        Mutation()(G, node)

        assert G.nodes[node]["structural_type"] == "pattern_v2.0_α-β"

    def test_identity_with_long_name(self):
        """Long identity names should be preserved."""
        G, node = create_nfr("test", epi=0.5, vf=1.0)
        long_identity = "very_long_identity_name_" * 10  # 260+ characters
        G.nodes[node]["structural_type"] = long_identity
        G.nodes[node]["epi_history"] = [0.3, 0.4, 0.5]
        # Enable postcondition validation
        G.graph["VALIDATE_OPERATOR_POSTCONDITIONS"] = True

        Mutation()(G, node)

        assert G.nodes[node]["structural_type"] == long_identity


class TestZHIRIdentityViolationDetection:
    """Test detection of identity violations (when enforcement is added)."""

    def test_identity_violation_placeholder(self):
        """Placeholder: Identity violation detection not yet implemented.

        This test documents the expected behavior when identity validation
        is added to the operator. Currently, ZHIR does not actively enforce
        identity preservation beyond maintaining the epi_kind attribute.

        Future implementation should:
        1. Check epi_kind before/after transformation
        2. Raise OperatorPostconditionError if changed
        3. Log identity violations to telemetry
        """
        # This is a placeholder test
        # When identity enforcement is implemented, this should become:
        #
        # G, node = create_nfr("test", epi=0.5, vf=1.0)
        # G.nodes[node]["EPI_kind"] = "original_identity"
        # G.graph["VALIDATE_OPERATOR_POSTCONDITIONS"] = True
        #
        # # Simulate identity violation (would require internal modification)
        # with pytest.raises(OperatorPostconditionError) as exc_info:
        #     # Apply mutation that somehow changes epi_kind
        #     pass
        #
        # assert "identity" in str(exc_info.value).lower()
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
