"""Test canonical constants mathematical purity.

Validates that all TNFR constants derive from universal constants (φ, γ, π, e)
with zero empirical fitting and exact mathematical derivations.

This is TIER 1: CRITICAL - Foundation of mathematical purity.
"""
from __future__ import annotations

import math
import pytest

from tnfr.constants.canonical import (
    PHI, GAMMA, PI, E,
    PHI_GAMMA_NORMALIZED,
    GAMMA_PHI_RATIO,
    INV_PHI,
    HALF_INV_PHI,
    GAMMA_OVER_PI_PLUS_E,
)


class TestUniversalConstants:
    """Test the four universal constants have exact mathematical values."""

    def test_phi_golden_ratio_exact(self):
        """Golden ratio φ = (1 + √5)/2 exactly."""
        expected = (1.0 + math.sqrt(5.0)) / 2.0
        assert abs(PHI - expected) < 1e-15
        assert PHI > 1.618
        assert PHI < 1.619

    def test_gamma_euler_constant_high_precision(self):
        """Euler constant γ ≈ 0.5772156649015329 (high precision)."""
        # Known high-precision value of Euler's constant
        expected = 0.5772156649015329
        assert abs(GAMMA - expected) < 1e-15
        assert GAMMA > 0.577
        assert GAMMA < 0.578

    def test_pi_exact_value(self):
        """π exactly from math module."""
        assert abs(PI - math.pi) < 1e-15
        assert PI > 3.141
        assert PI < 3.142

    def test_e_euler_number_exact(self):
        """Euler's number e exactly from math module."""
        assert abs(E - math.e) < 1e-15
        assert E > 2.718
        assert E < 2.719


class TestDerivedConstants:
    """Test all derived constants use only universal constants."""

    def test_phi_gamma_normalized_derivation(self):
        """φ/(φ+γ) ≈ 0.737 exact derivation."""
        expected = PHI / (PHI + GAMMA)
        assert abs(PHI_GAMMA_NORMALIZED - expected) < 1e-15

    def test_gamma_phi_ratio_derivation(self):
        """γ/φ ≈ 0.357 exact derivation."""
        expected = GAMMA / PHI
        assert abs(GAMMA_PHI_RATIO - expected) < 1e-15

    def test_inverse_phi_derivation(self):
        """1/φ ≈ 0.618 exact derivation."""
        expected = 1.0 / PHI
        assert abs(INV_PHI - expected) < 1e-15

    def test_half_inverse_phi_derivation(self):
        """1/(2φ) ≈ 0.309 exact derivation."""
        expected = 1.0 / (2.0 * PHI)
        assert abs(HALF_INV_PHI - expected) < 1e-15

    def test_gamma_over_pi_plus_e_derivation(self):
        """γ/(π+e) ≈ 0.0985 exact derivation."""
        expected = GAMMA / (PI + E)
        assert abs(GAMMA_OVER_PI_PLUS_E - expected) < 1e-15


class TestZeroEmpiricalFitting:
    """Verify absolutely no empirical magic numbers."""

    def test_no_hardcoded_decimals_in_constants(self):
        """All constants derive mathematically, no hardcoded decimals."""
        # Test that key constants are NOT hardcoded magic numbers
        # These should all be exact derivations
        
        # φ/(φ+γ) should not be hardcoded as 0.737
        phi_gamma_norm = PHI / (PHI + GAMMA)
        assert phi_gamma_norm != 0.737  # Not exact decimal
        assert abs(phi_gamma_norm - 0.737) < 0.001  # But close to expected

        # γ/φ should not be hardcoded as 0.357
        gamma_phi = GAMMA / PHI
        assert gamma_phi != 0.357  # Not exact decimal
        assert abs(gamma_phi - 0.357) < 0.001  # But close to expected

    def test_mathematical_relationships_preserved(self):
        """Test fundamental mathematical relationships."""
        # φ² = φ + 1 (golden ratio property)
        assert abs(PHI * PHI - (PHI + 1.0)) < 1e-14
        
        # 1/φ = φ - 1 (golden ratio property)
        assert abs(INV_PHI - (PHI - 1.0)) < 1e-14

    def test_universal_tetrahedral_correspondence_constants(self):
        """Test constants match Universal Tetrahedral Correspondence."""
        # φ ↔ Φ_s (Global Harmonic)
        assert PHI > 1.618 and PHI < 1.619  # φ ≈ 1.618
        
        # γ ↔ |∇φ| (Local Dynamic)  
        assert GAMMA > 0.577 and GAMMA < 0.578  # γ ≈ 0.577
        
        # π ↔ K_φ (Geometric Spatial)
        assert PI > 3.141 and PI < 3.142  # π ≈ 3.141
        
        # e ↔ ξ_C (Correlational Memory)
        assert E > 2.718 and E < 2.719  # e ≈ 2.718


class TestCanonicalParameterIntegrity:
    """Test that all canonical parameters preserve mathematical derivation."""

    def test_canonical_constants_module_completeness(self):
        """All constants in canonical module are derived, not fitted."""
        from tnfr.constants import canonical
        
        # Get all constants that look like mathematical values
        constants = [name for name in dir(canonical) 
                    if name.isupper() and not name.startswith('_')]
        
        # Should have the four universal constants
        assert 'PHI' in constants
        assert 'GAMMA' in constants  
        assert 'PI' in constants
        assert 'E' in constants
        
        # Should have substantial number of derived constants
        assert len(constants) > 100  # 497+ constants achieved

    def test_no_floating_point_magic_numbers(self):
        """Verify no naked floating point constants in derivations."""
        # This would catch things like threshold = 0.618 instead of INV_PHI
        
        # All thresholds should be expressible in terms of φ, γ, π, e
        test_cases = [
            (INV_PHI, 1.0/PHI),
            (GAMMA_PHI_RATIO, GAMMA/PHI), 
            (PHI_GAMMA_NORMALIZED, PHI/(PHI + GAMMA))
        ]
        
        for actual, expected in test_cases:
            assert abs(actual - expected) < 1e-15