"""
TNFR-Based Primality Testing Package

This package provides a novel approach to primality testing based on
TNFR (Resonant Fractal Nature Theory) and arithmetic pressure equations.

Author: TNFR Research Team
License: MIT
Year: 2025
"""

from .core import tnfr_is_prime, tnfr_delta_nfr
from .optimized import OptimizedTNFRPrimality

__version__ = "1.0.0"
__author__ = "F. F. Martinez Gamo"
__license__ = "MIT"
__doi__ = "10.5281/zenodo.17764749"

__all__ = [
    "tnfr_is_prime",
    "tnfr_delta_nfr",
    "OptimizedTNFRPrimality"
]