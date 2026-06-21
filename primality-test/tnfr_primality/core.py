"""
Core TNFR Primality Testing Implementation

This module contains the fundamental TNFR-based primality testing algorithms
based on the arithmetic pressure equation ΔNFR(n).

Mathematical Foundation:
ΔNFR(n) = ζ·(Ω(n)−1) + η·(τ(n)−2) + θ·(σ(n)/n − (1+1/n))

Where:
- Ω(n) = prime factor count with multiplicity (big Omega)
- τ(n) = number of divisors
- σ(n) = sum of divisors
- ζ = φ×γ ≈ 0.9340  (factorization pressure, notational)
- η = (γ/φ)×π ≈ 1.1207  (divisor pressure, notational)
- θ = 1/φ ≈ 0.6180  (abundance pressure, notational)

These coefficients are (φ, γ, π, e) combinations chosen to approximate the
original empirical values (1.0, 0.8, 0.6) — notational, NOT derived (audit 2026).

Theorem: n is prime ⟺ ΔNFR(n) = 0

Dual-lever interpretation (experimental discovery, March 2026):
- ΔNFR is the pressure lever in the nodal equation ∂EPI/∂t = νf · ΔNFR(t)
- Primes are zero-pressure nodes (ΔNFR = 0): maximum structural coherence
- Composites carry positive pressure proportional to factorization complexity
- Φ_s responds linearly to ΔNFR perturbations (|r| = 1.000)
"""
from __future__ import annotations

import math
from typing import Tuple, Dict
from functools import lru_cache

from .constants import (
    ZETA_CANONICAL, ETA_CANONICAL, THETA_CANONICAL,
    ALPHA_EPI, BETA_EPI, GAMMA_EPI,
    NU_0, DELTA_FREQ, EPSILON_FREQ,
)


def _divisor_count(n: int) -> int:
    """Count the number of divisors of n."""
    if n <= 0:
        return 0
    count = 0
    i = 1
    while i * i <= n:
        if n % i == 0:
            count += 1
            if i != n // i:
                count += 1
        i += 1
    return count


def _divisor_sum(n: int) -> int:
    """Calculate the sum of all divisors of n."""
    if n <= 0:
        return 0
    total = 0
    i = 1
    while i * i <= n:
        if n % i == 0:
            total += i
            j = n // i
            if j != i:
                total += j
        i += 1
    return total


def _prime_factor_count(n: int) -> int:
    """Count prime factors of n WITH multiplicity (Ω, big Omega).

    Canonical TNFR uses Ω(n) = total prime factor count including
    repeated factors.  This gives stronger pressure signals for
    prime powers (e.g. Ω(8) = 3 vs ω(8) = 1).
    """
    if n <= 1:
        return 0
    count = 0
    d = 2
    temp_n = n

    while d * d <= temp_n:
        while temp_n % d == 0:
            count += 1
            temp_n //= d
        d += 1

    if temp_n > 1:
        count += 1

    return count


def _distinct_prime_factor_count(n: int) -> int:
    """Count distinct prime factors of n (ω, little omega).

    Legacy function kept for backward compatibility.
    """
    if n <= 1:
        return 0
    count = 0
    d = 2
    temp_n = n

    while d * d <= temp_n:
        if temp_n % d == 0:
            count += 1
            while temp_n % d == 0:
                temp_n //= d
        d += 1

    if temp_n > 1:
        count += 1

    return count


def tnfr_delta_nfr(
    n: int,
    *,
    zeta: float = ZETA_CANONICAL,
    eta: float = ETA_CANONICAL,
    theta: float = THETA_CANONICAL,
) -> float:
    """
    Calculate TNFR arithmetic pressure ΔNFR(n).

    The ΔNFR equation quantifies structural pressure in arithmetic systems.
    For prime numbers, this pressure is exactly zero due to their perfect
    structural coherence.

    Args:
        n: Integer to analyze
        zeta: Factorization pressure coefficient (default: φ×γ ≈ 0.9340)
        eta: Divisor pressure coefficient (default: (γ/φ)×π ≈ 1.1207)
        theta: Abundance pressure coefficient (default: 1/φ ≈ 0.6180)

    Returns:
        ΔNFR value. Zero indicates primality.

    Mathematical Derivation:
        - Factorization pressure: ζ·(Ω(n)−1)
        - Divisor pressure: η·(τ(n)−2)
        - Abundance pressure: θ·(σ(n)/n − (1+1/n))
    """
    if n < 2:
        return float('inf')  # Invalid input

    # Calculate arithmetic functions
    tau_n = _divisor_count(n)         # τ(n)
    sigma_n = _divisor_sum(n)         # σ(n)
    omega_n = _prime_factor_count(n)  # Ω(n) — with multiplicity

    # TNFR pressure components (pressure lever of nodal equation)
    factorization_pressure = zeta * (omega_n - 1)
    divisor_pressure = eta * (tau_n - 2)
    sigma_pressure = theta * (sigma_n / n - (1 + 1 / n))

    return factorization_pressure + divisor_pressure + sigma_pressure


def tnfr_is_prime(n: int, *, tolerance: float = 1e-10) -> Tuple[bool, float]:
    """
    TNFR-based primality test using arithmetic pressure analysis.

    This function determines primality by calculating the TNFR arithmetic
    pressure ΔNFR(n). Prime numbers exhibit perfect structural coherence
    with ΔNFR(p) = 0, while composite numbers show positive pressure.

    Args:
        n: Integer to test for primality
        tolerance: Numerical tolerance for zero detection (default: 1e-10)

    Returns:
        Tuple of (is_prime: bool, delta_nfr: float)

    Examples:
        >>> tnfr_is_prime(17)
        (True, 0.0)
        >>> tnfr_is_prime(982451653)
        (True, 0.0)

    Performance:
        - Time Complexity: O(√n)
        - Space Complexity: O(1)
        - Accuracy: 100% (deterministic)
    """
    delta_nfr = tnfr_delta_nfr(n)
    is_prime = abs(delta_nfr) < tolerance
    return (is_prime, delta_nfr)


def tnfr_component_breakdown(
    n: int,
    *,
    zeta: float = ZETA_CANONICAL,
    eta: float = ETA_CANONICAL,
    theta: float = THETA_CANONICAL,
) -> Dict[str, float]:
    """Return per-component ΔNFR breakdown for structural analysis.

    Exposes the three pressure terms individually so that consumers
    can inspect *which* structural axis drives a composite's pressure.

    Returns:
        Dictionary with keys:
            factorization_pressure, divisor_pressure, abundance_pressure,
            delta_nfr, omega, tau, sigma
    """
    if n < 2:
        return {
            'factorization_pressure': float('inf'),
            'divisor_pressure': float('inf'),
            'abundance_pressure': float('inf'),
            'delta_nfr': float('inf'),
            'omega': 0, 'tau': 0, 'sigma': 0,
        }

    tau_n = _divisor_count(n)
    sigma_n = _divisor_sum(n)
    omega_n = _prime_factor_count(n)

    fp = zeta * (omega_n - 1)
    dp = eta * (tau_n - 2)
    ap = theta * (sigma_n / n - (1 + 1 / n))

    return {
        'factorization_pressure': fp,
        'divisor_pressure': dp,
        'abundance_pressure': ap,
        'delta_nfr': fp + dp + ap,
        'omega': omega_n,
        'tau': tau_n,
        'sigma': sigma_n,
    }


def tnfr_structural_triad(
    n: int,
    *,
    zeta: float = ZETA_CANONICAL,
    eta: float = ETA_CANONICAL,
    theta: float = THETA_CANONICAL,
) -> Dict[str, float]:
    """Compute the full structural triad (EPI, νf, ΔNFR) for a number.

    The structural triad characterizes each number in the three
    fundamental dimensions of TNFR dynamics:
      - EPI (form): structural complexity profile
      - νf  (frequency): reorganization capacity
      - ΔNFR (pressure): structural coherence pressure

    This implements the dual-lever interpretation: νf is the capacity
    lever and ΔNFR is the pressure lever of ∂EPI/∂t = νf · ΔNFR(t).

    Returns:
        Dictionary with EPI, vf, delta_nfr, local_coherence, components.
    """
    if n < 2:
        return {
            'EPI': 0.0, 'vf': 0.0, 'delta_nfr': float('inf'),
            'local_coherence': 0.0, 'components': {},
        }

    tau_n = _divisor_count(n)
    sigma_n = _divisor_sum(n)
    omega_n = _prime_factor_count(n)
    log_n = math.log(max(n, 2))

    # EPI: structural form  (α·Ω + β·ln(τ) + γ·(σ/n − 1))
    epi = 1.0 + ALPHA_EPI * omega_n + BETA_EPI * math.log(max(tau_n, 1)) + GAMMA_EPI * (sigma_n / n - 1)

    # νf: structural frequency  (ν₀ · (1 + δ·τ/n + ε·Ω/ln(n)))
    vf = NU_0 * (1 + DELTA_FREQ * tau_n / n + EPSILON_FREQ * omega_n / log_n)

    # ΔNFR: structural pressure
    fp = zeta * (omega_n - 1)
    dp = eta * (tau_n - 2)
    ap = theta * (sigma_n / n - (1 + 1 / n))
    delta_nfr = fp + dp + ap

    # Local coherence: 1/(1 + |ΔNFR|)
    local_coherence = 1.0 / (1.0 + abs(delta_nfr))

    return {
        'EPI': epi,
        'vf': vf,
        'delta_nfr': delta_nfr,
        'local_coherence': local_coherence,
        'components': {
            'factorization_pressure': fp,
            'divisor_pressure': dp,
            'abundance_pressure': ap,
        },
    }


# Cached versions for performance
@lru_cache(maxsize=10000)
def _divisor_count_cached(n: int) -> int:
    """Cached version of divisor count."""
    return _divisor_count(n)


@lru_cache(maxsize=10000)
def _divisor_sum_cached(n: int) -> int:
    """Cached version of divisor sum.""" 
    return _divisor_sum(n)


@lru_cache(maxsize=10000)
def _prime_factor_count_cached(n: int) -> int:
    """Cached version of prime factor count (with multiplicity)."""
    return _prime_factor_count(n)


@lru_cache(maxsize=5000)
def tnfr_delta_nfr_cached(
    n: int,
    zeta: float = ZETA_CANONICAL,
    eta: float = ETA_CANONICAL,
    theta: float = THETA_CANONICAL,
) -> float:
    """
    Cached version of TNFR ΔNFR computation for enhanced performance.

    Uses LRU caching to avoid recomputing expensive arithmetic functions
    for previously analyzed numbers.
    """
    if n < 2:
        return float('inf')

    tau_n = _divisor_count_cached(n)
    sigma_n = _divisor_sum_cached(n)
    omega_n = _prime_factor_count_cached(n)

    factorization_pressure = zeta * (omega_n - 1)
    divisor_pressure = eta * (tau_n - 2)
    sigma_pressure = theta * (sigma_n / n - (1 + 1 / n))

    return factorization_pressure + divisor_pressure + sigma_pressure


def tnfr_is_prime_cached(n: int, *, tolerance: float = 1e-10) -> Tuple[bool, float]:
    """
    Cached version of TNFR primality test for improved performance.

    Uses LRU caching to store results of expensive arithmetic computations.
    Recommended for applications testing many numbers or repeated queries.
    """
    delta_nfr = tnfr_delta_nfr_cached(n)
    is_prime = abs(delta_nfr) < tolerance
    return (is_prime, delta_nfr)


def validate_tnfr_theory(test_range: int = 1000) -> dict:
    """
    Validate TNFR primality theory against known results.
    
    This function tests the TNFR primality criterion against all numbers
    in a given range and compares with traditional primality testing.
    
    Args:
        test_range: Test numbers from 2 to test_range
        
    Returns:
        Dictionary with validation statistics
    """
    def is_prime_traditional(n):
        """Traditional primality test for comparison."""
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(n**0.5) + 1, 2):
            if n % i == 0:
                return False
        return True
    
    correct = 0
    false_positives = 0
    false_negatives = 0
    tested = 0
    
    for n in range(2, test_range + 1):
        tnfr_result, _ = tnfr_is_prime(n)
        traditional_result = is_prime_traditional(n)
        
        tested += 1
        if tnfr_result == traditional_result:
            correct += 1
        elif tnfr_result and not traditional_result:
            false_positives += 1
        elif not tnfr_result and traditional_result:
            false_negatives += 1
    
    return {
        'tested': tested,
        'correct': correct,
        'accuracy': correct / tested if tested > 0 else 0,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'error_rate': (false_positives + false_negatives) / tested if tested > 0 else 0
    }