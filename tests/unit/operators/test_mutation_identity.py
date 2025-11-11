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
)


class TestZHIRIdentityPreservation:
    """Test ZHIR preserves structural identity (epi_kind)."""

    def test_zhir_preserves_epi_kind(self):
        """ZHIR MUST preserve epi_kind during phase transformation."""
        G, node = create_nfr("test", epi=0.5, vf=1.0)
        
        # Set structural identity
        epi_kind_original = "coherent_oscillator"
        G.nodes[node]["EPI_kind"] = epi_kind_original
        G.nodes[node]["epi_history"] = [0.3, 0.4, 0.5]
        G.nodes[node]["delta_nfr"] = 0.3
        
        # Apply mutation
        Mutation()(G, node)
        
        # Identity MUST be preserved
        epi_kind_after = G.nodes[node].get("epi_kind")
        assert epi_kind_after == epi_kind_original, \
            f"ZHIR violated identity preservation: {epi_kind_original} → {epi_kind_after}"

    def test_zhir_preserves_identity_in_canonical_sequence(self):
        """Identity preserved in IL → OZ → ZHIR → IL sequence."""
        G, node = create_nfr("test", epi=0.4, vf=1.0)
        
        # Set identity
        G.nodes[node]["EPI_kind"] = "test_pattern"
        G.nodes[node]["epi_history"] = [0.35, 0.38, 0.40]
        
        # Apply canonical sequence
        run_sequence(G, node, [
            Coherence(),    # IL
            Dissonance(),   # OZ
            Mutation(),     # ZHIR - must preserve identity
            Coherence(),    # IL
        ])
        
        # Identity must be preserved through entire sequence
        assert G.nodes[node]["EPI_kind"] == "test_pattern"

    def test_multiple_zhir_preserve_identity(self):
        """Multiple ZHIR applications (with IL between) preserve identity."""
        G, node = create_nfr("test", epi=0.5, vf=1.0)
        
        # Set identity
        original_identity = "fractal_structure"
        G.nodes[node]["EPI_kind"] = original_identity
        G.nodes[node]["epi_history"] = [0.3, 0.4, 0.5]
        
        # Apply 3 mutation cycles
        for i in range(3):
            run_sequence(G, node, [
                Coherence(),    # Stabilize
                Dissonance(),   # Destabilize
                Mutation(),     # Transform
                Coherence(),    # Stabilize
            ])
        
        # Identity must still be preserved
        assert G.nodes[node]["EPI_kind"] == original_identity, \
            "Multiple ZHIR violated identity preservation"

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
            G.nodes[node]["EPI_kind"] = identity_type
            G.nodes[node]["epi_history"] = [0.3, 0.4, 0.5]
            
            # Apply mutation
            run_sequence(G, node, [Coherence(), Dissonance(), Mutation()])
            
            # Verify preservation
            assert G.nodes[node]["EPI_kind"] == identity_type, \
                f"Failed to preserve identity: {identity_type}"


class TestZHIRIdentityMetrics:
    """Test identity preservation is captured in metrics."""

    def test_identity_preserved_in_metrics(self):
        """Metrics should track identity preservation."""
        G, node = create_nfr("test", epi=0.5, vf=1.0)
        G.nodes[node]["EPI_kind"] = "test_identity"
        G.nodes[node]["epi_history"] = [0.3, 0.4, 0.5]
        G.graph["COLLECT_OPERATOR_METRICS"] = True
        
        Mutation()(G, node)
        
        # Get metrics
        metrics_list = G.graph.get("operator_metrics", [])
        assert len(metrics_list) > 0
        
        zhir_metric = metrics_list[-1]
        assert zhir_metric["glyph"] == "ZHIR"
        
        # Check identity metrics
        assert "identity_preserved" in zhir_metric
        assert "epi_kind_before" in zhir_metric
        assert "epi_kind_after" in zhir_metric
        
        # Identity should be preserved
        assert zhir_metric["identity_preserved"] is True
        assert zhir_metric["epi_kind_before"] == "test_identity"
        assert zhir_metric["epi_kind_after"] == "test_identity"

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
        G.nodes[node]["EPI_kind"] = "rotating_pattern"
        G.nodes[node]["theta"] = 0.0
        G.nodes[node]["delta_nfr"] = 0.8  # Strong transformation
        G.nodes[node]["epi_history"] = [0.3, 0.4, 0.5]
        
        # Apply mutation with strong phase shift
        G.graph["GLYPH_FACTORS"] = {"ZHIR_theta_shift_factor": 0.8}
        Mutation()(G, node)
        
        theta_after = G.nodes[node]["theta"]
        
        # Phase should have changed significantly
        assert theta_after != 0.0
        assert abs(theta_after) > 0.5
        
        # But identity MUST be preserved
        assert G.nodes[node]["EPI_kind"] == "rotating_pattern"

    def test_identity_preserved_with_high_epi_change(self):
        """Identity preserved even when EPI changes (within bounds)."""
        G, node = create_nfr("test", epi=0.3, vf=1.0)
        G.nodes[node]["EPI_kind"] = "adaptive_pattern"
        G.nodes[node]["epi_history"] = [0.1, 0.2, 0.3]
        G.nodes[node]["delta_nfr"] = 0.5
        
        # Apply sequence that increases EPI
        run_sequence(G, node, [Dissonance(), Mutation()])
        
        # EPI may have changed
        epi_after = G.nodes[node]["EPI"]
        # (exact value depends on implementation)
        
        # Identity must be preserved
        assert G.nodes[node]["EPI_kind"] == "adaptive_pattern"

    def test_identity_preserved_across_regime_change(self):
        """Identity preserved when phase crosses regime boundary."""
        import math
        
        G, node = create_nfr("test", epi=0.5, vf=1.0)
        G.nodes[node]["EPI_kind"] = "regime_crossing_pattern"
        G.nodes[node]["theta"] = math.pi / 2 - 0.1  # Near boundary
        G.nodes[node]["delta_nfr"] = 0.6  # Strong shift
        G.nodes[node]["epi_history"] = [0.3, 0.4, 0.5]
        G.graph["GLYPH_FACTORS"] = {"ZHIR_theta_shift_factor": 0.5}
        
        # Apply mutation - should cross regime boundary
        Mutation()(G, node)
        
        # Check if regime changed
        theta_after = G.nodes[node]["theta"]
        regime_before = int((math.pi / 2 - 0.1) // (math.pi / 2))
        regime_after = int(theta_after // (math.pi / 2))
        
        # Identity must be preserved regardless of regime change
        assert G.nodes[node]["EPI_kind"] == "regime_crossing_pattern"


class TestZHIRIdentityEdgeCases:
    """Test identity preservation in edge cases."""

    def test_identity_with_negative_epi(self):
        """Identity preserved when EPI is negative."""
        G, node = create_nfr("test", epi=-0.5, vf=1.0)
        G.nodes[node]["EPI_kind"] = "negative_epi_pattern"
        G.nodes[node]["epi_history"] = [-0.7, -0.6, -0.5]
        
        Mutation()(G, node)
        
        assert G.nodes[node]["EPI_kind"] == "negative_epi_pattern"

    def test_identity_with_zero_epi(self):
        """Identity preserved when EPI crosses zero."""
        G, node = create_nfr("test", epi=0.0, vf=1.0)
        G.nodes[node]["EPI_kind"] = "zero_crossing_pattern"
        G.nodes[node]["epi_history"] = [-0.1, 0.0, 0.0]
        
        Mutation()(G, node)
        
        assert G.nodes[node]["EPI_kind"] == "zero_crossing_pattern"

    def test_identity_with_special_characters(self):
        """Identity strings with special characters should be preserved."""
        G, node = create_nfr("test", epi=0.5, vf=1.0)
        G.nodes[node]["EPI_kind"] = "pattern_v2.0_α-β"
        G.nodes[node]["epi_history"] = [0.3, 0.4, 0.5]
        
        Mutation()(G, node)
        
        assert G.nodes[node]["EPI_kind"] == "pattern_v2.0_α-β"

    def test_identity_with_long_name(self):
        """Long identity names should be preserved."""
        G, node = create_nfr("test", epi=0.5, vf=1.0)
        long_identity = "very_long_identity_name_" * 10  # 260+ characters
        G.nodes[node]["EPI_kind"] = long_identity
        G.nodes[node]["epi_history"] = [0.3, 0.4, 0.5]
        
        Mutation()(G, node)
        
        assert G.nodes[node]["EPI_kind"] == long_identity


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
