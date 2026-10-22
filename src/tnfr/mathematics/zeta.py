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

from typing import Any

# Import TNFR Cache Infrastructure
from .unified_cache import CacheLevel, cache_tnfr_computation
from .unified_numerical import np  # noqa: F401 - retain the module attribute

# Retain availability attributes for the supported dependency-complete package.
HAS_BACKEND = True
HAS_MPMATH = True

# We use mpmath for high precision (FFT-based arithmetic for large numbers)
from mpmath import mp as _mp

# Retain the declared 25-digit arithmetic without changing caller precision.
# Bind every function to this context rather than mpmath's shared singleton.
mp = _mp.clone()
mp.dps = 25
mp_fabs = mp.fabs
mp_gamma = mp.gamma
mp_log = mp.log
mp_pi = mp.pi
mp_power = mp.power
mp_sin = mp.sin
mp_zeta = mp.zeta
mp_zetazero = mp.zetazero


@cache_tnfr_computation(level=CacheLevel.DERIVED_METRICS, dependencies=set())
def zeta_zero(n: int) -> complex:
    """
    Return the n-th non-trivial zero of the Riemann Zeta function.
    """
    return mp_zetazero(n)


@cache_tnfr_computation(level=CacheLevel.DERIVED_METRICS, dependencies=set())
def zeta_function(s: complex | float | Any) -> Any:
    """
    Compute the Riemann Zeta function ζ(s).

    Integration:
    - Uses mpmath for high-precision FFT-based arithmetic.
    - Uses TNFR Cache for structural field memoization.
    - Compatible with Unified Backend for future vectorization.
    """
    return mp_zeta(s)


@cache_tnfr_computation(level=CacheLevel.DERIVED_METRICS, dependencies=set())
def chi_factor(s: complex | float | Any) -> Any:
    """
    Compute the Riemann xi factor χ(s).
    χ(s) = 2^s * π^(s-1) * sin(πs/2) * Γ(1-s)
    """
    s_mp = mp.mpc(s)
    return (
        mp_power(2, s_mp)
        * mp_power(mp_pi, s_mp - 1)
        * mp_sin(mp_pi * s_mp / 2)
        * mp_gamma(1 - s_mp)
    )


@cache_tnfr_computation(level=CacheLevel.DERIVED_METRICS, dependencies=set())
def structural_potential(s: complex | float | Any) -> float:
    """
    Compute the Structural Potential Φ_s = log|ζ(s)|.
    """
    z = zeta_function(s)
    mag = abs(z)
    return float(mp_log(mag + 1e-20))


@cache_tnfr_computation(level=CacheLevel.DERIVED_METRICS, dependencies=set())
def structural_pressure(s: complex | float | Any) -> float:
    """
    Compute the Structural Pressure ΔNFR = |log|χ(s)||.
    This is the derived form from symmetry breaking.
    """
    chi = chi_factor(s)
    mag = abs(chi)
    return float(mp_fabs(mp_log(mag)))
