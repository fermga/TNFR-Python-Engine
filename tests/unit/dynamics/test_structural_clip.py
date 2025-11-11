"""Tests for structural boundary preservation (structural_clip).

This test module validates the canonical TNFR structural clipping functionality
that ensures EPI values remain within valid structural boundaries while preserving
coherence through smooth operator behavior.
"""

import math
import pytest

from tnfr.dynamics.structural_clip import (
    structural_clip,
    StructuralClipStats,
    get_clip_stats,
    reset_clip_stats,
)


class TestStructuralClipHardMode:
    """Test hard clipping mode for immediate boundary enforcement."""

    def test_clip_above_maximum(self):
        """Test that values above maximum are clamped down."""
        assert structural_clip(1.1, -1.0, 1.0, mode="hard") == 1.0
        assert structural_clip(2.5, -1.0, 1.0, mode="hard") == 1.0
        assert structural_clip(999.9, -1.0, 1.0, mode="hard") == 1.0

    def test_clip_below_minimum(self):
        """Test that values below minimum are clamped up."""
        assert structural_clip(-1.1, -1.0, 1.0, mode="hard") == -1.0
        assert structural_clip(-2.5, -1.0, 1.0, mode="hard") == -1.0
        assert structural_clip(-999.9, -1.0, 1.0, mode="hard") == -1.0

    def test_no_clip_within_bounds(self):
        """Test that values within bounds are unchanged."""
        assert structural_clip(0.0, -1.0, 1.0, mode="hard") == 0.0
        assert structural_clip(0.5, -1.0, 1.0, mode="hard") == 0.5
        assert structural_clip(-0.5, -1.0, 1.0, mode="hard") == -0.5
        assert structural_clip(0.999, -1.0, 1.0, mode="hard") == 0.999
        assert structural_clip(-0.999, -1.0, 1.0, mode="hard") == -0.999

    def test_boundary_values(self):
        """Test that boundary values themselves are preserved."""
        assert structural_clip(1.0, -1.0, 1.0, mode="hard") == 1.0
        assert structural_clip(-1.0, -1.0, 1.0, mode="hard") == -1.0

    def test_custom_bounds(self):
        """Test clipping with custom structural boundaries."""
        assert structural_clip(5.5, 0.0, 5.0, mode="hard") == 5.0
        assert structural_clip(-0.5, 0.0, 5.0, mode="hard") == 0.0
        assert structural_clip(2.5, 0.0, 5.0, mode="hard") == 2.5

    def test_critical_VAL_expansion_case(self):
        """Test the critical case mentioned in issue: EPI=0.95 with VAL_scale=1.15.

        After VAL application, EPI could become 0.95 * (some factor based on vf and dnfr).
        This tests that even if integration pushes EPI to 1.0925, it gets clipped.
        """
        # Simulating the result after VAL operator with scale 1.15
        epi_after_expansion = 1.0925  # 0.95 * 1.15
        clipped = structural_clip(epi_after_expansion, -1.0, 1.0, mode="hard")
        assert clipped == 1.0
        assert clipped <= 1.0  # Structural invariant preserved


class TestStructuralClipSoftMode:
    """Test soft clipping mode for smooth boundary preservation."""

    def test_soft_clip_preserves_interior_values(self):
        """Test that interior values stay within bounds with soft mode."""
        for val in [0.0, 0.3, -0.3]:
            clipped = structural_clip(val, -1.0, 1.0, mode="soft", k=3.0)
            # Soft mode applies sigmoid transformation
            # Values should remain within bounds
            assert -1.0 <= clipped <= 1.0
            # Sign should be preserved
            if val > 0:
                assert clipped > 0
            elif val < 0:
                assert clipped < 0
            else:
                assert abs(clipped) < 0.1

    def test_soft_clip_constrains_extreme_values(self):
        """Test that extreme values are pulled back within bounds."""
        clipped_high = structural_clip(1.5, -1.0, 1.0, mode="soft", k=3.0)
        assert clipped_high <= 1.0
        assert clipped_high > 0.7  # Should approach boundary

        clipped_low = structural_clip(-1.5, -1.0, 1.0, mode="soft", k=3.0)
        assert clipped_low >= -1.0
        assert clipped_low < -0.7  # Should approach boundary

    def test_soft_clip_is_monotonic(self):
        """Test that soft clipping produces monotonic values."""
        # Sample values across boundary
        values = [0.9, 0.95, 1.0, 1.05, 1.1]
        clipped = [structural_clip(v, -1.0, 1.0, mode="soft", k=3.0) for v in values]

        # Check monotonicity (values increase)
        for i in range(len(clipped) - 1):
            assert clipped[i] <= clipped[i + 1] + 1e-10  # Allow tiny numerical error

        # All should be within bounds
        assert all(-1.0 <= c <= 1.0 for c in clipped)

    def test_soft_clip_steepness_parameter(self):
        """Test that k parameter controls transition sharpness."""
        val = 1.2

        # Lower k = softer transition
        soft_k1 = structural_clip(val, -1.0, 1.0, mode="soft", k=1.0)
        # Higher k = sharper transition (closer to hard clip)
        soft_k10 = structural_clip(val, -1.0, 1.0, mode="soft", k=10.0)

        # Both within bounds
        assert soft_k1 <= 1.0
        assert soft_k10 <= 1.0

        # Both should approach boundary
        assert soft_k1 > 0.5
        assert soft_k10 > 0.5

    def test_soft_clip_critical_expansion_case(self):
        """Test soft mode on the critical VAL expansion case."""
        epi_after_expansion = 1.0925
        clipped = structural_clip(epi_after_expansion, -1.0, 1.0, mode="soft", k=3.0)
        assert clipped <= 1.0
        assert clipped > 0.8  # Should be close to boundary but smooth


class TestStructuralClipEdgeCases:
    """Test edge cases and error handling."""

    def test_invalid_bounds_raises_error(self):
        """Test that inverted bounds raise ValueError."""
        with pytest.raises(ValueError, match="Lower bound.*must be <= upper bound"):
            structural_clip(0.5, 1.0, -1.0, mode="hard")

    def test_invalid_mode_raises_error(self):
        """Test that invalid mode raises ValueError."""
        with pytest.raises(ValueError, match="mode must be"):
            structural_clip(0.5, -1.0, 1.0, mode="invalid")  # type: ignore[arg-type]

    def test_equal_bounds(self):
        """Test clipping when lo == hi."""
        result = structural_clip(5.0, 2.0, 2.0, mode="hard")
        assert result == 2.0

        result_soft = structural_clip(5.0, 2.0, 2.0, mode="soft")
        assert result_soft == 2.0

    def test_zero_width_bounds(self):
        """Test behavior with zero-width bounds."""
        assert structural_clip(10.0, 0.0, 0.0, mode="hard") == 0.0
        assert structural_clip(-10.0, 0.0, 0.0, mode="soft") == 0.0


class TestStructuralClipStats:
    """Test telemetry and statistics collection."""

    def test_stats_not_recorded_by_default(self):
        """Test that stats are not recorded when record_stats=False."""
        reset_clip_stats()
        structural_clip(1.5, -1.0, 1.0, mode="hard", record_stats=False)
        stats = get_clip_stats()
        assert stats.total_adjustments == 0

    def test_stats_recorded_when_enabled(self):
        """Test that stats are recorded when record_stats=True."""
        reset_clip_stats()
        structural_clip(1.5, -1.0, 1.0, mode="hard", record_stats=True)
        stats = get_clip_stats()
        assert stats.hard_clips == 1
        assert stats.total_adjustments == 1

    def test_stats_track_hard_clips(self):
        """Test that hard clip statistics are tracked correctly."""
        reset_clip_stats()

        structural_clip(1.2, -1.0, 1.0, mode="hard", record_stats=True)
        structural_clip(1.5, -1.0, 1.0, mode="hard", record_stats=True)
        structural_clip(-1.3, -1.0, 1.0, mode="hard", record_stats=True)

        stats = get_clip_stats()
        assert stats.hard_clips == 3
        assert stats.soft_clips == 0
        assert stats.max_delta_hard > 0

    def test_stats_track_soft_clips(self):
        """Test that soft clip statistics are tracked correctly."""
        reset_clip_stats()

        structural_clip(1.2, -1.0, 1.0, mode="soft", record_stats=True)
        structural_clip(1.5, -1.0, 1.0, mode="soft", record_stats=True)

        stats = get_clip_stats()
        assert stats.soft_clips == 2
        assert stats.hard_clips == 0
        assert stats.max_delta_soft > 0

    def test_stats_summary(self):
        """Test that summary returns complete statistics."""
        reset_clip_stats()

        structural_clip(1.2, -1.0, 1.0, mode="hard", record_stats=True)
        structural_clip(1.5, -1.0, 1.0, mode="soft", record_stats=True)

        summary = get_clip_stats().summary()
        assert "hard_clips" in summary
        assert "soft_clips" in summary
        assert "total_adjustments" in summary
        assert summary["total_adjustments"] == 2

    def test_stats_reset(self):
        """Test that reset clears all statistics."""
        reset_clip_stats()

        structural_clip(1.5, -1.0, 1.0, mode="hard", record_stats=True)
        assert get_clip_stats().total_adjustments > 0

        reset_clip_stats()
        assert get_clip_stats().total_adjustments == 0
        assert get_clip_stats().hard_clips == 0
        assert get_clip_stats().max_delta_hard == 0.0


class TestStructuralClipIntegration:
    """Test structural_clip in realistic TNFR integration scenarios."""

    def test_repeated_val_applications(self):
        """Test that repeated VAL expansions don't break boundaries."""
        epi = 0.8
        val_scale = 1.15

        # Simulate multiple VAL applications
        for _ in range(10):
            epi = epi * val_scale
            epi = structural_clip(epi, -1.0, 1.0, mode="hard")
            # After clipping, must never exceed boundary
            assert epi <= 1.0

    def test_repeated_nul_applications(self):
        """Test that repeated NUL contractions don't break boundaries."""
        epi = -0.8
        nul_scale = 0.85

        # Simulate multiple NUL applications
        for _ in range(10):
            epi = epi * nul_scale
            epi = structural_clip(epi, -1.0, 1.0, mode="hard")
            # After clipping, must stay within boundaries
            assert epi >= -1.0

    def test_boundary_preservation_under_noise(self):
        """Test that boundaries are preserved even with numerical noise."""
        # Simulate floating point imprecision
        epi_values = [1.0 + 1e-10, 1.0 + 1e-8, -1.0 - 1e-10, -1.0 - 1e-8]

        for epi in epi_values:
            clipped = structural_clip(epi, -1.0, 1.0, mode="hard")
            assert -1.0 <= clipped <= 1.0

    def test_soft_mode_preserves_gradient_continuity(self):
        """Test that soft mode produces smoother transitions than hard mode."""
        values = [0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2]

        hard_clipped = [structural_clip(v, -1.0, 1.0, mode="hard") for v in values]
        soft_clipped = [structural_clip(v, -1.0, 1.0, mode="soft", k=3.0) for v in values]

        # Hard mode has discontinuous derivative at boundary
        hard_diffs = [hard_clipped[i + 1] - hard_clipped[i] for i in range(len(hard_clipped) - 1)]
        # Soft mode should have more continuous changes
        soft_diffs = [soft_clipped[i + 1] - soft_clipped[i] for i in range(len(soft_clipped) - 1)]

        # Soft mode differences should not have abrupt jumps to zero
        # (though they may decrease smoothly)
        assert all(d >= 0 for d in soft_diffs)  # Monotonic
        # Hard mode will have zero differences after boundary
        assert 0.0 in hard_diffs


class TestVALNULOperatorBoundaries:
    """Specific tests for VAL and NUL operator boundary cases from the issue."""

    def test_val_critical_threshold_0_869565(self):
        """Test the critical EPI threshold where VAL_scale=1.15 causes overflow.

        From issue: EPI >= 0.869565 with VAL_scale=1.15 → overflow.
        0.869565 * 1.15 ≈ 1.0 (or slightly above due to precision)
        """
        critical_epi = 0.869565
        val_scale = 1.15

        # After applying VAL operator scaling and integration
        epi_after = critical_epi * val_scale

        # This is right at the edge, might be slightly below 1.0
        # The key is that with clipping, it definitely stays within bounds
        epi_clipped = structural_clip(epi_after, -1.0, 1.0, mode="hard")
        assert epi_clipped <= 1.0
        assert -1.0 <= epi_clipped <= 1.0

    def test_val_example_case_0_95(self):
        """Test the specific example from issue: EPI=0.95 with VAL_scale=1.15."""
        epi = 0.95
        val_scale = 1.15

        epi_after = epi * val_scale  # = 1.0925
        assert epi_after > 1.0  # Overflows without clipping

        epi_clipped = structural_clip(epi_after, -1.0, 1.0, mode="hard")
        assert epi_clipped == 1.0

    def test_nul_symmetric_contraction_case(self):
        """Test symmetric case for NUL contraction at negative boundary."""
        critical_epi = -0.869565
        nul_scale = 0.85

        # Simulate effect (though NUL doesn't multiply EPI directly,
        # it affects vf which affects integration)
        epi_tendency = critical_epi / nul_scale  # Going more negative

        if epi_tendency < -1.0:
            epi_clipped = structural_clip(epi_tendency, -1.0, 1.0, mode="hard")
            assert epi_clipped == -1.0
            assert epi_clipped >= -1.0

    def test_soft_mode_near_val_boundary(self):
        """Test soft mode behavior near VAL overflow boundary."""
        # Test values approaching and crossing the boundary
        test_values = [0.95, 0.99, 1.0, 1.05, 1.1]

        for val in test_values:
            clipped = structural_clip(val, -1.0, 1.0, mode="soft", k=3.0)
            # All must stay within bounds
            assert -1.0 <= clipped <= 1.0
            # Values near boundary should be smoothly mapped
            if val <= 1.0:
                assert abs(clipped - val) < 0.05  # Close to original
            else:
                assert clipped < 1.0  # Pulled back from boundary
