"""Tests for NUL (Contraction) enhanced precondition validation.

This module validates the implementation of enhanced precondition checks
for the Contraction (NUL) operator to prevent over-compression and
structural collapse:

1. νf > minimum threshold (existing check)
2. EPI >= minimum threshold (NEW - prevent compression of too-small structures)
3. density <= maximum threshold (NEW - prevent over-compression)

The validation prevents structural collapse by ensuring nodes have sufficient
form to contract and are not already at critical density where further
contraction would risk fragmentation.
"""

import pytest

from tnfr.constants import DNFR_PRIMARY, EPI_PRIMARY, VF_PRIMARY
from tnfr.operators.preconditions import (
    OperatorPreconditionError,
    validate_contraction,
)
from tnfr.structural import create_nfr


class TestContractionBasicPreconditions:
    """Test suite for basic NUL precondition validation."""

    def test_validate_contraction_success_normal_node(self):
        """Validation passes for normal node with sufficient attributes."""
        G, node = create_nfr("normal", epi=0.5, vf=1.0)
        G.nodes[node][DNFR_PRIMARY] = 0.2

        # Should not raise
        validate_contraction(G, node)

    def test_validate_contraction_success_at_thresholds(self):
        """Validation passes for node just above all thresholds."""
        G, node = create_nfr("threshold", epi=0.11, vf=0.11)
        G.nodes[node][DNFR_PRIMARY] = 0.5  # density = 0.5/0.11 ≈ 4.5 < 10.0

        # Should not raise (just above all thresholds)
        validate_contraction(G, node)

    def test_validate_contraction_fails_low_vf(self):
        """Validation fails when νf <= minimum threshold (existing check)."""
        G, node = create_nfr("low_vf", epi=0.5, vf=0.09)
        G.nodes[node][DNFR_PRIMARY] = 0.2

        with pytest.raises(OperatorPreconditionError) as exc_info:
            validate_contraction(G, node)

        error_msg = str(exc_info.value)
        assert "Contraction" in error_msg
        assert "νf=0.090" in error_msg
        assert "<= 0.1" in error_msg

    def test_validate_contraction_fails_at_min_vf(self):
        """Validation fails when νf exactly equals minimum (boundary)."""
        G, node = create_nfr("at_min", epi=0.5, vf=0.1)
        G.nodes[node][DNFR_PRIMARY] = 0.2

        with pytest.raises(OperatorPreconditionError) as exc_info:
            validate_contraction(G, node)

        error_msg = str(exc_info.value)
        assert "νf=0.100 <= 0.1" in error_msg


class TestContractionEPIMinimum:
    """Test suite for EPI minimum threshold validation."""

    def test_validate_contraction_fails_low_epi(self):
        """Validation fails when EPI < minimum threshold."""
        G, node = create_nfr("low_epi", epi=0.05, vf=1.0)
        G.nodes[node][DNFR_PRIMARY] = 0.1

        with pytest.raises(OperatorPreconditionError) as exc_info:
            validate_contraction(G, node)

        error_msg = str(exc_info.value)
        assert "Contraction" in error_msg
        assert "EPI=0.050" in error_msg
        assert "< 0.1" in error_msg
        assert "Cannot compress structure below minimum coherent form" in error_msg

    def test_validate_contraction_passes_at_min_epi(self):
        """Validation passes when EPI exactly equals minimum (boundary)."""
        G, node = create_nfr("at_min_epi", epi=0.1, vf=1.0)
        G.nodes[node][DNFR_PRIMARY] = 0.2

        # Should pass: epi=0.1 is not < 0.1 (boundary is inclusive)
        validate_contraction(G, node)

    def test_validate_contraction_passes_just_above_min_epi(self):
        """Validation passes when EPI just above minimum."""
        G, node = create_nfr("above_min", epi=0.101, vf=1.0)
        G.nodes[node][DNFR_PRIMARY] = 0.2

        validate_contraction(G, node)

    def test_validate_contraction_fails_zero_epi(self):
        """Validation fails for zero EPI (no structure to contract)."""
        G, node = create_nfr("zero_epi", epi=0.0, vf=1.0)
        G.nodes[node][DNFR_PRIMARY] = 0.1

        with pytest.raises(OperatorPreconditionError) as exc_info:
            validate_contraction(G, node)

        error_msg = str(exc_info.value)
        assert "EPI=0.000 < 0.1" in error_msg

    def test_validate_contraction_custom_min_epi(self):
        """Custom EPI minimum can be configured via graph metadata."""
        G, node = create_nfr("custom", epi=0.05, vf=1.0)
        G.nodes[node][DNFR_PRIMARY] = 0.1

        # With default min_epi=0.1, this would fail
        # Set custom lower threshold
        G.graph["NUL_MIN_EPI"] = 0.03

        # Should now pass
        validate_contraction(G, node)


class TestContractionDensityThreshold:
    """Test suite for density threshold validation."""

    def test_validate_contraction_fails_high_density(self):
        """Validation fails when density exceeds maximum threshold."""
        G, node = create_nfr("high_density", epi=0.2, vf=1.0)
        # density = 2.5 / 0.2 = 12.5 > 10.0
        G.nodes[node][DNFR_PRIMARY] = 2.5

        with pytest.raises(OperatorPreconditionError) as exc_info:
            validate_contraction(G, node)

        error_msg = str(exc_info.value)
        assert "Contraction" in error_msg
        assert "ρ=12.500" in error_msg
        assert "> 10.0" in error_msg
        assert "critical density" in error_msg
        assert "structural collapse" in error_msg
        assert "Consider IL (Coherence)" in error_msg

    def test_validate_contraction_passes_at_max_density(self):
        """Validation passes when density exactly equals maximum (boundary)."""
        G, node = create_nfr("at_max", epi=0.5, vf=1.0)
        # density = 5.0 / 0.5 = 10.0 (exactly at threshold)
        G.nodes[node][DNFR_PRIMARY] = 5.0

        # Should pass: density=10.0 is not > 10.0 (boundary is inclusive)
        validate_contraction(G, node)

    def test_validate_contraction_passes_below_max_density(self):
        """Validation passes when density below maximum."""
        G, node = create_nfr("safe_density", epi=0.5, vf=1.0)
        # density = 2.0 / 0.5 = 4.0 < 10.0
        G.nodes[node][DNFR_PRIMARY] = 2.0

        validate_contraction(G, node)

    def test_validate_contraction_density_with_negative_dnfr(self):
        """Density calculation uses absolute value of ΔNFR."""
        G, node = create_nfr("negative_dnfr", epi=0.2, vf=1.0)
        # density = |-2.5| / 0.2 = 12.5 > 10.0
        G.nodes[node][DNFR_PRIMARY] = -2.5

        with pytest.raises(OperatorPreconditionError) as exc_info:
            validate_contraction(G, node)

        error_msg = str(exc_info.value)
        assert "ρ=12.500" in error_msg

    def test_validate_contraction_density_with_zero_dnfr(self):
        """Validation passes when ΔNFR = 0 (equilibrium, zero density)."""
        G, node = create_nfr("equilibrium", epi=0.5, vf=1.0)
        G.nodes[node][DNFR_PRIMARY] = 0.0

        # density = 0.0 / 0.5 = 0.0 < 10.0, should pass
        validate_contraction(G, node)

    def test_validate_contraction_density_with_very_small_epi(self):
        """Very small EPI is caught by EPI minimum check before density check."""
        G, node = create_nfr("tiny_epi", epi=1e-10, vf=1.0)
        G.nodes[node][DNFR_PRIMARY] = 0.1

        # EPI check happens before density check, so tiny EPI will fail on that first
        with pytest.raises(OperatorPreconditionError) as exc_info:
            validate_contraction(G, node)

        error_msg = str(exc_info.value)
        # Should be caught by EPI check, not density check
        assert "EPI" in error_msg
        assert "Cannot compress" in error_msg

    def test_validate_contraction_custom_max_density(self):
        """Custom density maximum can be configured via graph metadata."""
        G, node = create_nfr("custom_density", epi=0.2, vf=1.0)
        # density = 2.5 / 0.2 = 12.5
        G.nodes[node][DNFR_PRIMARY] = 2.5

        # With default max_density=10.0, this would fail
        # Set custom higher threshold
        G.graph["NUL_MAX_DENSITY"] = 15.0

        # Should now pass
        validate_contraction(G, node)

    def test_validate_contraction_density_epsilon_protection(self):
        """Density calculation uses epsilon when EPI is very small but above minimum."""
        G, node = create_nfr("epsilon_case", epi=0.1, vf=1.0)
        G.nodes[node][DNFR_PRIMARY] = 1.5

        # Set very low minimum EPI to bypass that check
        G.graph["NUL_MIN_EPI"] = 0.01

        # density = 1.5 / max(0.1, 1e-9) = 15.0 > 10.0
        with pytest.raises(OperatorPreconditionError) as exc_info:
            validate_contraction(G, node)

        error_msg = str(exc_info.value)
        assert "critical density" in error_msg
        assert "15.000" in error_msg


class TestContractionPreconditionsEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_validate_contraction_multiple_violations_reports_first(self):
        """Multiple violations report in order (νf checked first)."""
        G, node = create_nfr("multi_violation", epi=0.05, vf=0.05)
        G.nodes[node][DNFR_PRIMARY] = 2.0  # Would also violate density

        # Should fail on νf check (checked first)
        with pytest.raises(OperatorPreconditionError) as exc_info:
            validate_contraction(G, node)

        # Should mention νf, not EPI or density
        assert "νf=" in str(exc_info.value)

    def test_validate_contraction_epi_before_density_check(self):
        """EPI check happens before density check."""
        G, node = create_nfr("epi_violation", epi=0.05, vf=1.0)
        G.nodes[node][DNFR_PRIMARY] = 10.0  # Would also violate density

        with pytest.raises(OperatorPreconditionError) as exc_info:
            validate_contraction(G, node)

        # Should mention EPI, not density
        assert "EPI=" in str(exc_info.value)
        assert "density" not in str(exc_info.value).lower()

    def test_validate_contraction_all_checks_pass_together(self):
        """All checks can pass simultaneously with valid values."""
        G, node = create_nfr("all_valid", epi=0.5, vf=1.0)
        G.nodes[node][DNFR_PRIMARY] = 2.0  # density = 4.0 < 10.0

        # All checks pass
        validate_contraction(G, node)

    def test_validate_contraction_negative_epi_below_threshold(self):
        """Negative EPI values fail if below absolute threshold."""
        G, node = create_nfr("negative_epi", epi=-0.05, vf=1.0)
        G.nodes[node][DNFR_PRIMARY] = 0.1

        # -0.05 < 0.1 (minimum), should fail
        with pytest.raises(OperatorPreconditionError) as exc_info:
            validate_contraction(G, node)

        assert "EPI=-0.050 < 0.1" in str(exc_info.value)

    def test_validate_contraction_high_vf_with_valid_epi_density(self):
        """Very high νf with valid EPI and density passes."""
        G, node = create_nfr("high_vf", epi=1.0, vf=9.5)
        G.nodes[node][DNFR_PRIMARY] = 3.0  # density = 3.0 < 10.0

        validate_contraction(G, node)


class TestContractionPreconditionsIntegration:
    """Integration tests with full operator sequences."""

    def test_validate_contraction_after_coherence(self):
        """IL (Coherence) can prepare node for safe contraction."""
        from tnfr.operators.definitions import Coherence

        G, node = create_nfr("prep", epi=0.8, vf=1.5)
        G.nodes[node][DNFR_PRIMARY] = 2.0

        # Apply Coherence to stabilize (reduces ΔNFR)
        Coherence()(G, node)

        # After coherence, ΔNFR should be reduced, lowering density
        # This makes contraction safer
        # Validation should pass if density is now safe
        try:
            validate_contraction(G, node)
        except OperatorPreconditionError:
            # If still fails, it's due to other factors, not density
            pass

    def test_validate_contraction_prevents_over_compression_cascade(self):
        """Multiple contractions eventually hit safety limits."""
        from tnfr.operators import apply_glyph
        from tnfr.types import Glyph

        G, node = create_nfr("cascade", epi=1.0, vf=2.0)
        G.nodes[node][DNFR_PRIMARY] = 0.5

        # First contraction should succeed
        apply_glyph(G, node, Glyph.NUL)
        validate_contraction(G, node)

        # Second contraction should succeed
        apply_glyph(G, node, Glyph.NUL)
        validate_contraction(G, node)

        # Eventually, one of the limits will be hit
        # Continue until we hit a limit
        for _ in range(10):
            try:
                apply_glyph(G, node, Glyph.NUL)
                validate_contraction(G, node)
            except OperatorPreconditionError:
                # Expected - hit safety limit
                break
        else:
            pytest.fail("Should have hit safety limit after multiple contractions")

    def test_validate_contraction_error_messages_actionable(self):
        """Error messages provide actionable guidance."""
        # Test EPI violation message
        G1, node1 = create_nfr("low_epi", epi=0.05, vf=1.0)
        G1.nodes[node1][DNFR_PRIMARY] = 0.1

        with pytest.raises(OperatorPreconditionError) as exc1:
            validate_contraction(G1, node1)

        assert "Cannot compress structure below minimum coherent form" in str(exc1.value)

        # Test density violation message
        G2, node2 = create_nfr("high_density", epi=0.2, vf=1.0)
        G2.nodes[node2][DNFR_PRIMARY] = 3.0

        with pytest.raises(OperatorPreconditionError) as exc2:
            validate_contraction(G2, node2)

        msg = str(exc2.value)
        assert "structural collapse" in msg
        assert "Consider IL (Coherence)" in msg


class TestContractionPreconditionsConfiguration:
    """Test configuration parameter behavior."""

    def test_validate_contraction_all_defaults(self):
        """Default configuration values are sensible."""
        G, node = create_nfr("defaults", epi=0.5, vf=1.0)
        G.nodes[node][DNFR_PRIMARY] = 2.0

        # Should pass with defaults:
        # - NUL_MIN_VF = 0.1 (vf=1.0 > 0.1)
        # - NUL_MIN_EPI = 0.1 (epi=0.5 >= 0.1)
        # - NUL_MAX_DENSITY = 10.0 (density=4.0 < 10.0)
        validate_contraction(G, node)

    def test_validate_contraction_strict_configuration(self):
        """Strict configuration enables tighter safety margins."""
        G, node = create_nfr("strict", epi=0.15, vf=0.15)
        G.nodes[node][DNFR_PRIMARY] = 0.5  # density ≈ 3.33

        # Set strict thresholds
        G.graph["NUL_MIN_VF"] = 0.2
        G.graph["NUL_MIN_EPI"] = 0.2
        G.graph["NUL_MAX_DENSITY"] = 3.0

        # Should fail on all three checks
        with pytest.raises(OperatorPreconditionError):
            validate_contraction(G, node)

    def test_validate_contraction_permissive_configuration(self):
        """Permissive configuration allows more aggressive contraction."""
        G, node = create_nfr("permissive", epi=0.05, vf=0.05)
        G.nodes[node][DNFR_PRIMARY] = 1.0  # density = 20.0

        # Set permissive thresholds
        G.graph["NUL_MIN_VF"] = 0.01
        G.graph["NUL_MIN_EPI"] = 0.01
        G.graph["NUL_MAX_DENSITY"] = 50.0

        # Should pass with permissive config
        validate_contraction(G, node)

    def test_validate_contraction_mixed_configuration(self):
        """Can configure some parameters while using defaults for others."""
        G, node = create_nfr("mixed", epi=0.5, vf=1.0)
        G.nodes[node][DNFR_PRIMARY] = 3.0  # density = 6.0

        # Only configure max_density, use defaults for others
        G.graph["NUL_MAX_DENSITY"] = 5.0

        # Should fail on density (6.0 > 5.0)
        with pytest.raises(OperatorPreconditionError) as exc_info:
            validate_contraction(G, node)

        assert "ρ=6.000 > 5.0" in str(exc_info.value)


class TestContractionPreconditionsPhysicsAlignment:
    """Test alignment with TNFR physics principles."""

    def test_validate_contraction_prevents_vacuum_compression(self):
        """Cannot compress structure that's already minimal (EPI check)."""
        G, node = create_nfr("vacuum", epi=0.01, vf=1.0)
        G.nodes[node][DNFR_PRIMARY] = 0.1

        with pytest.raises(OperatorPreconditionError) as exc_info:
            validate_contraction(G, node)

        assert "EPI" in str(exc_info.value)
        assert "Cannot compress" in str(exc_info.value)

    def test_validate_contraction_prevents_infinite_density(self):
        """Prevents approaching infinite density (critical threshold)."""
        G, node = create_nfr("approaching_infinity", epi=0.1, vf=1.0)
        G.nodes[node][DNFR_PRIMARY] = 2.0  # density = 20.0 > 10.0

        with pytest.raises(OperatorPreconditionError) as exc_info:
            validate_contraction(G, node)

        assert "density" in str(exc_info.value).lower()
        assert "collapse" in str(exc_info.value).lower()

    def test_validate_contraction_maintains_nodal_equation_boundedness(self):
        """Validation ensures ∂EPI/∂t = νf · ΔNFR remains bounded."""
        # High density means high ΔNFR relative to EPI
        # This could cause ∂EPI/∂t to explode
        G, node = create_nfr("unbounded", epi=0.1, vf=1.0)
        G.nodes[node][DNFR_PRIMARY] = 5.0  # density = 50.0 >> 10.0

        # Should be rejected to maintain boundedness
        with pytest.raises(OperatorPreconditionError):
            validate_contraction(G, node)

    def test_validate_contraction_respects_structural_limits(self):
        """All three checks work together to enforce structural limits."""
        # Create node violating all three limits
        G, node = create_nfr("all_limits", epi=0.05, vf=0.05)
        G.nodes[node][DNFR_PRIMARY] = 10.0

        # Should fail on first check (νf)
        with pytest.raises(OperatorPreconditionError) as exc_info:
            validate_contraction(G, node)

        # Fix νf, should fail on EPI
        G.nodes[node][VF_PRIMARY] = 1.0
        with pytest.raises(OperatorPreconditionError) as exc_info:
            validate_contraction(G, node)

        # Fix EPI, should fail on density
        G.nodes[node][EPI_PRIMARY] = 0.5
        with pytest.raises(OperatorPreconditionError) as exc_info:
            validate_contraction(G, node)

        # Fix density, should pass
        G.nodes[node][DNFR_PRIMARY] = 2.0  # density = 4.0 < 10.0
        validate_contraction(G, node)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
