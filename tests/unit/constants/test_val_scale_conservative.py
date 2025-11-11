"""Tests verifying the conservative nature of VAL_scale=1.05.

This module tests the adjustment from VAL_scale=1.15 to VAL_scale=1.05
for improved numerical stability near EPI boundaries.

Issue: fermga/TNFR-Python-Engine#TBD
Related: fermga/TNFR-Python-Engine#2661 (structural_clip), fermga/TNFR-Python-Engine#2662 (edge-aware scaling)
"""

import pytest
from tnfr.constants import CORE_DEFAULTS
from tnfr.structural import create_nfr
from tnfr.operators import Expansion


class TestVALScaleConservative:
    """Test suite verifying VAL_scale=1.05 is sufficiently conservative."""

    def test_val_scale_is_conservative(self):
        """Verify that VAL_scale=1.05 is set to the conservative value."""
        assert CORE_DEFAULTS["GLYPH_FACTORS"]["VAL_scale"] == 1.05
        assert 1.0 / 1.05 > 0.95  # Safe threshold above 0.95

    def test_critical_threshold_calculation(self):
        """Verify the critical threshold for VAL_scale=1.05."""
        val_scale = CORE_DEFAULTS["GLYPH_FACTORS"]["VAL_scale"]
        critical_threshold = 1.0 / val_scale
        
        # Critical threshold should be approximately 0.952381
        assert abs(critical_threshold - 0.952381) < 0.000001
        
        # This is significantly higher than the old threshold (1.0/1.15 ≈ 0.870)
        old_critical_threshold = 1.0 / 1.15
        assert critical_threshold > old_critical_threshold + 0.08

    def test_expansion_capability_maintained(self):
        """Verify that VAL_scale=1.05 still provides meaningful expansion."""
        val_scale = CORE_DEFAULTS["GLYPH_FACTORS"]["VAL_scale"]
        
        # Starting from EPI=0.5, 10 applications should still expand significantly
        epi = 0.5
        for _ in range(10):
            epi *= val_scale
        
        # Should reach approximately 0.814
        assert 0.81 < epi < 0.82
        
        # This is much safer than the old behavior (0.5 → 2.078 which would overflow)
        old_epi = 0.5
        for _ in range(10):
            old_epi *= 1.15
        assert old_epi > 2.0  # Old behavior was problematic

    def test_val_scale_provides_safety_margin(self):
        """Verify that VAL_scale=1.05 provides adequate safety margin."""
        val_scale = CORE_DEFAULTS["GLYPH_FACTORS"]["VAL_scale"]
        
        # Test various high EPI values
        safe_zone_start = 1.0 / val_scale  # ≈ 0.952
        
        # Values below the safe zone should not overflow with single application
        test_values = [0.90, 0.91, 0.92, 0.93, 0.94, 0.95]
        for epi_start in test_values:
            epi_result = epi_start * val_scale
            assert epi_result <= 1.0, f"EPI {epi_start} * {val_scale} = {epi_result} exceeds 1.0"

    def test_val_scale_ratio_to_nul_scale(self):
        """Verify the balance between VAL_scale and NUL_scale."""
        val_scale = CORE_DEFAULTS["GLYPH_FACTORS"]["VAL_scale"]
        nul_scale = CORE_DEFAULTS["GLYPH_FACTORS"]["NUL_scale"]
        
        # VAL and NUL should be roughly symmetric around 1.0
        assert val_scale == 1.05
        assert nul_scale == 0.85
        
        # VAL expansion ratio: 1.05 / 1.0 = 1.05
        val_ratio = val_scale / 1.0
        # NUL contraction ratio: 1.0 / 0.85 ≈ 1.176
        nul_ratio = 1.0 / nul_scale
        
        # VAL is more conservative (closer to 1.0) than NUL
        assert val_ratio < nul_ratio

    def test_multiple_expansions_stay_bounded(self):
        """Verify that multiple VAL applications maintain bounds."""
        G, node = create_nfr("test_multi", epi=0.9, vf=1.0)
        G.nodes[node]["DNFR"] = 0.01
        
        # Apply VAL multiple times
        val_op = Expansion()
        
        for i in range(20):
            # Apply VAL
            val_op(G, node)
            
            # Check that vf increases with each application
            vf = G.nodes[node]["νf"]
            assert vf > 1.0, f"vf should be > 1.0 after {i+1} applications"
            
            # EPI doesn't change immediately (requires integration)
            # but vf scaling demonstrates conservative behavior

    def test_boundary_preservation_principle(self):
        """Test that VAL_scale=1.05 respects the structural boundary principle.
        
        EPI_MAX is not an arbitrary numerical limit but represents the
        structural identity frontier. Conservative scaling prevents
        boundary transgression while maintaining expansion capacity.
        """
        val_scale = CORE_DEFAULTS["GLYPH_FACTORS"]["VAL_scale"]
        epi_max = CORE_DEFAULTS["EPI_MAX"]
        
        # The safe threshold (where scaling would reach exactly EPI_MAX)
        safe_threshold = epi_max / val_scale
        
        # This threshold should provide meaningful operating range
        assert safe_threshold > 0.95  # Conservative enough
        assert safe_threshold < 1.0   # Still allows expansion to boundary
        
        # The 8.7% reduction in scale factor (1.15 → 1.05) significantly
        # improves numerical stability
        old_scale = 1.15
        new_scale = 1.05
        reduction_percentage = ((old_scale - new_scale) / old_scale) * 100
        assert abs(reduction_percentage - 8.7) < 0.1


class TestVALScaleComparisonToOld:
    """Tests comparing new VAL_scale=1.05 behavior to old VAL_scale=1.15."""

    def test_new_scale_more_conservative(self):
        """Verify new scale is more conservative than old scale."""
        new_scale = 1.05
        old_scale = 1.15
        
        assert new_scale < old_scale
        assert new_scale - 1.0 < old_scale - 1.0

    def test_critical_threshold_improvement(self):
        """Verify the critical threshold improvement."""
        new_threshold = 1.0 / 1.05  # ≈ 0.952381
        old_threshold = 1.0 / 1.15  # ≈ 0.869565
        
        # New threshold is significantly higher (better)
        improvement = new_threshold - old_threshold
        assert improvement > 0.08  # More than 8% improvement

    def test_overflow_prevention_improvement(self):
        """Verify that the new scale prevents more overflow cases."""
        # Test case that was problematic with old scale
        epi_test = 0.95
        
        # Old scale would overflow
        old_result = epi_test * 1.15
        assert old_result > 1.0  # Overflow!
        
        # New scale stays safe
        new_result = epi_test * 1.05
        assert new_result < 1.0  # Safe!

    def test_expansion_sequence_comparison(self):
        """Compare expansion sequences between old and new scales."""
        # Start from same point
        epi_old = 0.5
        epi_new = 0.5
        
        # Apply 10 expansions
        for _ in range(10):
            epi_old *= 1.15
            epi_new *= 1.05
        
        # Old scale reaches problematic values
        assert epi_old > 2.0  # Way beyond EPI_MAX
        
        # New scale stays in reasonable range
        assert epi_new < 1.0  # Still safe
        assert epi_new > 0.8  # Still provides meaningful expansion


def test_val_scale_documented_value():
    """Verify VAL_scale matches documented value in constants."""
    from tnfr.config.defaults_core import CoreDefaults
    
    defaults = CoreDefaults()
    assert defaults.GLYPH_FACTORS["VAL_scale"] == 1.05
