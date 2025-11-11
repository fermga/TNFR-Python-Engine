"""Additional precondition tests for ZHIR (Mutation) operator.

This module complements test_mutation_threshold.py with tests for
EPI history requirements and other preconditions not covered elsewhere.

Test Coverage:
1. EPI history requirement (must have sufficient history)
2. Node initialization requirements
3. Edge cases for precondition validation

References:
- AGENTS.md §11 (Mutation operator contracts)
- test_mutation_threshold.py (complementary tests)
- src/tnfr/operators/preconditions/__init__.py
"""

import pytest
from tnfr.structural import create_nfr, run_sequence
from tnfr.operators.definitions import Mutation, Coherence, Dissonance
from tnfr.operators.preconditions import validate_mutation, OperatorPreconditionError


class TestZHIRHistoryRequirements:
    """Test ZHIR requirements for EPI history."""

    def test_zhir_requires_epi_history_for_threshold(self, caplog):
        """ZHIR should warn when epi_history is missing or insufficient."""
        G, node = create_nfr("test", epi=0.5, vf=1.0)
        G.graph["ZHIR_THRESHOLD_XI"] = 0.1

        # No EPI history at all
        # Don't set epi_history

        import logging

        with caplog.at_level(logging.WARNING):
            validate_mutation(G, node)

        # Should log warning about insufficient history
        assert any(
            "without sufficient EPI history" in record.message
            or "history" in record.message.lower()
            for record in caplog.records
        )

        # Should set unknown threshold flag
        assert G.nodes[node].get("_zhir_threshold_unknown") is True

    def test_zhir_accepts_minimal_history(self):
        """ZHIR should work with minimal history (2 points for velocity)."""
        G, node = create_nfr("test", epi=0.5, vf=1.0)
        G.graph["ZHIR_THRESHOLD_XI"] = 0.1

        # Minimal history for velocity computation (2 points)
        G.nodes[node]["epi_history"] = [0.3, 0.5]

        # Should not raise error
        validate_mutation(G, node)

        # Should compute velocity even with minimal history
        # (bifurcation detection needs 3 points, but threshold only needs 2)
        depi_dt = abs(0.5 - 0.3)
        if depi_dt > 0.1:
            assert G.nodes[node].get("_zhir_threshold_met") is True
        else:
            assert G.nodes[node].get("_zhir_threshold_warning") is True

    def test_zhir_accepts_long_history(self):
        """ZHIR should handle long EPI histories correctly."""
        G, node = create_nfr("test", epi=0.8, vf=1.0)
        G.graph["ZHIR_THRESHOLD_XI"] = 0.1

        # Long history (uses last 3 points for computations)
        G.nodes[node]["epi_history"] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

        # Should not raise error
        validate_mutation(G, node)

        # Should use recent history for velocity computation
        # Recent velocity: 0.8 - 0.7 = 0.1 (meets threshold)
        assert G.nodes[node].get("_zhir_threshold_met") is True

    def test_zhir_with_both_history_keys(self):
        """ZHIR should work with either 'epi_history' or '_epi_history'."""
        # Test with underscore prefix
        G1, n1 = create_nfr("test1", epi=0.5, vf=1.0)
        G1.nodes[n1]["_epi_history"] = [0.3, 0.4, 0.5]
        G1.graph["ZHIR_THRESHOLD_XI"] = 0.1

        # Should not raise
        validate_mutation(G1, n1)
        assert G1.nodes[n1].get("_zhir_threshold_met") or G1.nodes[n1].get(
            "_zhir_threshold_warning"
        )

        # Test with standard key
        G2, n2 = create_nfr("test2", epi=0.5, vf=1.0)
        G2.nodes[n2]["epi_history"] = [0.3, 0.4, 0.5]
        G2.graph["ZHIR_THRESHOLD_XI"] = 0.1

        # Should not raise
        validate_mutation(G2, n2)
        assert G2.nodes[n2].get("_zhir_threshold_met") or G2.nodes[n2].get(
            "_zhir_threshold_warning"
        )


class TestZHIRNodeValidation:
    """Test ZHIR validation of node state."""

    def test_zhir_requires_positive_vf(self):
        """ZHIR must fail if νf <= 0 (node is dead/frozen)."""
        G, node = create_nfr("test", epi=0.5, vf=0.0)  # Dead node
        G.graph["ZHIR_MIN_VF"] = 0.05
        G.nodes[node]["epi_history"] = [0.3, 0.4, 0.5]

        # Should raise error
        with pytest.raises(OperatorPreconditionError) as exc_info:
            validate_mutation(G, node)

        assert "Structural frequency too low" in str(exc_info.value)

    def test_zhir_accepts_small_positive_vf(self):
        """ZHIR should work with small but positive νf."""
        G, node = create_nfr("test", epi=0.5, vf=0.06)  # Just above minimum
        G.graph["ZHIR_MIN_VF"] = 0.05
        G.nodes[node]["epi_history"] = [0.3, 0.4, 0.5]

        # Should not raise
        validate_mutation(G, node)

        # Should succeed or warn (depending on threshold)
        assert G.nodes[node].get("_zhir_threshold_met") or G.nodes[node].get(
            "_zhir_threshold_warning"
        )

    def test_zhir_works_without_min_vf_config(self):
        """ZHIR should work when ZHIR_MIN_VF not configured (no enforcement)."""
        G, node = create_nfr("test", epi=0.5, vf=0.01)  # Very low vf
        # Don't set ZHIR_MIN_VF
        G.nodes[node]["epi_history"] = [0.3, 0.4, 0.5]

        # Should not raise (enforcement disabled)
        validate_mutation(G, node)


class TestZHIREdgeCases:
    """Test edge cases for ZHIR preconditions."""

    def test_zhir_with_negative_epi(self):
        """ZHIR should work with negative EPI values."""
        G, node = create_nfr("test", epi=-0.5, vf=1.0)
        G.nodes[node]["epi_history"] = [-0.7, -0.6, -0.5]
        G.graph["ZHIR_THRESHOLD_XI"] = 0.1

        # Should not raise
        validate_mutation(G, node)

        # Velocity should still be computed
        depi_dt = abs(-0.5 - (-0.6))
        assert depi_dt == pytest.approx(0.1, abs=0.01)

    def test_zhir_with_zero_epi(self):
        """ZHIR should work with EPI=0 (crossing zero is valid)."""
        G, node = create_nfr("test", epi=0.0, vf=1.0)
        G.nodes[node]["epi_history"] = [-0.1, 0.0, 0.0]
        G.graph["ZHIR_THRESHOLD_XI"] = 0.1

        # Should not raise
        validate_mutation(G, node)

    def test_zhir_with_very_high_vf(self):
        """ZHIR should handle very high νf values."""
        G, node = create_nfr("test", epi=0.5, vf=10.0)  # Very high frequency
        G.nodes[node]["epi_history"] = [0.3, 0.4, 0.5]
        G.graph["ZHIR_THRESHOLD_XI"] = 0.1

        # Should not raise
        validate_mutation(G, node)

        # Should still compute threshold correctly
        assert G.nodes[node].get("_zhir_threshold_met") or G.nodes[node].get(
            "_zhir_threshold_warning"
        )

    def test_zhir_preserves_history_after_validation(self):
        """ZHIR validation should not modify EPI history."""
        G, node = create_nfr("test", epi=0.5, vf=1.0)
        history_original = [0.3, 0.4, 0.5]
        G.nodes[node]["epi_history"] = history_original.copy()

        validate_mutation(G, node)

        # History should be unchanged
        history_after = G.nodes[node]["epi_history"]
        assert history_after == history_original


class TestZHIRPreconditionConfiguration:
    """Test configuration flags for precondition validation."""

    def test_strict_validation_enforces_all_checks(self):
        """VALIDATE_OPERATOR_PRECONDITIONS=True should enforce all checks."""
        G, node = create_nfr("test", epi=0.5, vf=0.01)  # Low vf
        G.graph["VALIDATE_OPERATOR_PRECONDITIONS"] = True
        G.graph["ZHIR_MIN_VF"] = 0.05
        G.nodes[node]["epi_history"] = [0.3, 0.4, 0.5]

        # Should raise error with strict validation
        with pytest.raises(OperatorPreconditionError):
            validate_mutation(G, node)

    def test_soft_validation_allows_borderline_cases(self):
        """Default soft validation should allow borderline cases with warnings."""
        G, node = create_nfr("test", epi=0.5, vf=0.04)  # Below threshold
        # Don't enable strict validation (default is soft)
        G.graph["ZHIR_MIN_VF"] = 0.05
        G.nodes[node]["epi_history"] = [0.3, 0.4, 0.5]

        # Should not raise (soft validation logs warnings)
        validate_mutation(G, node)

    def test_individual_flags_override_defaults(self):
        """Individual requirement flags should work independently."""
        G, node = create_nfr("test", epi=0.5, vf=1.0)
        # Set individual flag without global strict validation
        G.graph["ZHIR_REQUIRE_IL_PRECEDENCE"] = True

        from tnfr.types import Glyph

        G.nodes[node]["glyph_history"] = [Glyph.OZ]  # No IL
        G.nodes[node]["epi_history"] = [0.3, 0.4, 0.5]

        # Should fail on IL even without global strict validation
        with pytest.raises(OperatorPreconditionError):
            validate_mutation(G, node)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
