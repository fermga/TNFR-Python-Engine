"""
TNFR-Based Primality Testing Package

This package provides a novel approach to primality testing based on
TNFR (Resonant Fractal Nature Theory) and arithmetic pressure equations.

Theoretical foundation: TNFR-Python-Engine/theory/TNFR_NUMBER_THEORY.md
  - Theorem (§4): n is prime ⟺ ΔNFR(n) = 0
  - Canonical constants (§5): All coefficients from φ, γ, π, e

Author: TNFR Research Team
License: MIT
Year: 2025
"""

from .core import (
    tnfr_is_prime,
    tnfr_delta_nfr,
    tnfr_component_breakdown,
    tnfr_structural_triad,
)
from .optimized import OptimizedTNFRPrimality

__version__ = "1.1.0"
__author__ = "F. F. Martinez Gamo"
__license__ = "MIT"
__doi__ = "10.5281/zenodo.17764749"

__all__ = [
    "tnfr_is_prime",
    "tnfr_delta_nfr",
    "tnfr_component_breakdown",
    "tnfr_structural_triad",
    "OptimizedTNFRPrimality",
]