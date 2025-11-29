"""TIER 2: CORE PHYSICS Tests

Tests the fundamental physics of TNFR dynamics:

1. **Nodal Equation**: ∂EPI/∂t = νf · ΔNFR(t)
2. **Structural Triad**: EPI, νf, phase properties  
3. **ΔNFR Dynamics**: Structural pressure physics

These tests validate the mathematical heart of TNFR theory.
"""

__all__ = [
    "test_nodal_equation",
    "test_structural_triad", 
    "test_delta_nfr"
]