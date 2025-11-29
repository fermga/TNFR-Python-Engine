"""
TNFR Riemann Zeta Function Implementation
=========================================

This module provides the canonical implementation of the Riemann Zeta function
and related spectral functions for the TNFR engine.

It wraps high-precision arithmetic (mpmath) to ensure structural fidelity
during critical line exploration.

Functions:
- zeta(s): The Riemann Zeta function.
- chi_factor(s): The symmetry factor χ(s).
- structural_potential(s): Φ_s = log|ζ(s)|.
- structural_pressure(s): ΔNFR = |log|χ(s)||.
"""

from typing import Union, Any, Callable, Set, Optional
import numpy as np

# Import TNFR Cache Infrastructure
try:
    from ..utils.cache import cache_tnfr_computation, CacheLevel
    _CACHE_AVAILABLE = True
except ImportError:
    _CACHE_AVAILABLE = False

    # Fallback if cache not available
    def cache_tnfr_computation(level: Any, dependencies: Set[str]) -> Callable:
        def decorator(f: Callable) -> Callable:
            return f
        return decorator

    class CacheLevel:
        DERIVED_METRICS = "derived_metrics"
        GRAPH_STRUCTURE = "graph_structure"
        NODE_PROPERTIES = "node_properties"

# Import Unified Backend
try:
    from .backend import get_backend
    HAS_BACKEND = True
except ImportError:
    HAS_BACKEND = False

# We use mpmath for high precision (FFT-based arithmetic for large numbers)
try:
    from mpmath import mp, zeta as mp_zeta, gamma as mp_gamma, sin as mp_sin, pi as mp_pi, power as mp_power, log as mp_log, fabs as mp_fabs, zetazero as mp_zetazero
    HAS_MPMATH = True
except ImportError:
    HAS_MPMATH = False

# Default precision
if HAS_MPMATH:
    mp.dps = 25


def _ensure_complex(s: Any) -> Any:
    if HAS_MPMATH:
        return mp.mpc(s)
    return complex(s)


@cache_tnfr_computation(level=CacheLevel.DERIVED_METRICS, dependencies=set())
def zeta_zero(n: int) -> complex:
    """
    Return the n-th non-trivial zero of the Riemann Zeta function.
    """
    if HAS_MPMATH:
        return mp_zetazero(n)
    raise ImportError("mpmath required for zeta zeros.")


@cache_tnfr_computation(level=CacheLevel.DERIVED_METRICS, dependencies=set())
def zeta_function(s: Union[complex, float, Any]) -> Any:
    """
    Compute the Riemann Zeta function ζ(s).
    
    Integration:
    - Uses mpmath for high-precision FFT-based arithmetic.
    - Uses TNFR Cache for structural field memoization.
    - Compatible with Unified Backend for future vectorization.
    """
    if HAS_MPMATH:
        return mp_zeta(s)
    
    # Fallback to scipy if available, else raise
    try:
        from scipy.special import zeta as scipy_zeta
        return scipy_zeta(s)
    except ImportError:
        raise ImportError("mpmath or scipy required for zeta function.")


@cache_tnfr_computation(level=CacheLevel.DERIVED_METRICS, dependencies=set())
def chi_factor(s: Union[complex, float, Any]) -> Any:
    """
    Compute the Riemann xi factor χ(s).
    χ(s) = 2^s * π^(s-1) * sin(πs/2) * Γ(1-s)
    """
    if HAS_MPMATH:
        s_mp = mp.mpc(s)
        return mp_power(2, s_mp) * mp_power(mp_pi, s_mp - 1) * mp_sin(mp_pi * s_mp / 2) * mp_gamma(1 - s_mp)
    
    # Standard float implementation
    s_c = complex(s)
    return (2**s_c) * (np.pi**(s_c - 1)) * np.sin(np.pi * s_c / 2) * np.math.gamma(1 - s_c)  # type: ignore


@cache_tnfr_computation(level=CacheLevel.DERIVED_METRICS, dependencies=set())
def structural_potential(s: Union[complex, float, Any]) -> float:
    """
    Compute the Structural Potential Φ_s = log|ζ(s)|.
    """
    z = zeta_function(s)
    mag = abs(z)
    if HAS_MPMATH:
        return float(mp_log(mag + 1e-20))
    return float(np.log(mag + 1e-20))


@cache_tnfr_computation(level=CacheLevel.DERIVED_METRICS, dependencies=set())
def structural_pressure(s: Union[complex, float, Any]) -> float:
    """
    Compute the Structural Pressure ΔNFR = |log|χ(s)||.
    This is the derived form from symmetry breaking.
    """
    chi = chi_factor(s)
    mag = abs(chi)
    if HAS_MPMATH:
        return float(mp_fabs(mp_log(mag)))
    return float(abs(np.log(mag)))
    return float(abs(np.log(mag)))
