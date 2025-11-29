"""Test runner for Mathematical Purity Era test suite.

This is the new main test configuration for TNFR after achieving
497+ canonical constants and removing domain extensions.
"""
from __future__ import annotations


def test_mathematical_purity_foundation():
    """Smoke test that core mathematical purity components work."""
    # Test canonical constants are accessible
    from tnfr.constants.canonical import PHI, GAMMA, PI, E
    
    # Universal constants should have expected values
    assert 1.618 < PHI < 1.619
    assert 0.577 < GAMMA < 0.578
    assert 3.141 < PI < 3.142
    assert 2.718 < E < 2.719
    
    # Test core physics fields are available
    try:
        from tnfr.physics.fields import (
            compute_structural_potential,
            compute_phase_gradient,
        )
        physics_available = True
    except ImportError:
        physics_available = False
        
    assert physics_available, "Core TNFR physics should be available"


def test_pure_tnfr_engine_scope():
    """Test that pure TNFR engine scope is maintained."""
    import tnfr
    
    # Should have core modules
    core_modules = [
        'tnfr.constants',
        'tnfr.physics', 
        'tnfr.operators',
        'tnfr.mathematics',
    ]
    
    for module in core_modules:
        try:
            __import__(module)
        except ImportError as e:
            pytest.fail(f"Core module {module} should be available: {e}")
            
    # Should NOT have extensions (they were removed)
    extension_modules = [
        'tnfr.extensions.medical',
        'tnfr.extensions.business',
    ]
    
    for module in extension_modules:
        try:
            __import__(module)
            pytest.fail(f"Extension module {module} should not be available")
        except ImportError:
            pass  # Expected - extensions removed


def test_canonical_constants_mathematical_derivation():
    """Test that constants are mathematically derived, not empirically fitted."""
    from tnfr.constants.canonical import (
        PHI, GAMMA, PI, E,
        INV_PHI,
        PHI_GAMMA_NORMALIZED,
        GAMMA_PHI_RATIO,
    )
    
    # Test mathematical relationships
    assert abs(PHI * PHI - (PHI + 1.0)) < 1e-14  # φ² = φ + 1
    assert abs(INV_PHI - (1.0 / PHI)) < 1e-14     # 1/φ exact
    assert abs(PI - 3.141592653589793) < 1e-14    # π exact
    assert abs(E - 2.718281828459045) < 1e-14     # e exact
    
    # Test derived constants are exact derivations
    assert abs(PHI_GAMMA_NORMALIZED - (PHI / (PHI + GAMMA))) < 1e-15
    assert abs(GAMMA_PHI_RATIO - (GAMMA / PHI)) < 1e-15


if __name__ == "__main__":
    import pytest
    pytest.main([__file__])