"""Arithmetic-pressure diagnostics and primality predicates.

Divisor enumeration and trial factorization supply the statistics. With
positive coefficients their exact-real pressure vanishes precisely at primes
n>=2. The implementation uses floats and configurable numerical tolerance;
this characterization is not a derivation of nodal dynamics or a universal
floating-point correctness/performance certificate."""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Dict, Tuple

from .constants import (
    ALPHA_EPI,
    BETA_EPI,
    DELTA_FREQ,
    EPSILON_FREQ,
    ETA_CANONICAL,
    GAMMA_EPI,
    NU_0,
    THETA_CANONICAL,
    ZETA_CANONICAL,
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
    """Compute a weighted arithmetic pressure from divisor/factor statistics.

    Args:
        n: Integer to analyze; values below 2 return positive infinity.
        zeta: Weight of Omega(n)-1; default 1.0.
        eta: Weight of tau(n)-2; default 1.0.
        theta: Weight of sigma(n)/n-(1+1/n); default 1.0.

    Returns:
        Floating pressure. For positive coefficients and exact arithmetic,
        its zero set on n>=2 is precisely the primes. Coefficients are not
        validated here; arbitrary nonpositive choices need not preserve that
        result. The defaults are a selected normalization.

    The routine computes divisors and prime-factor multiplicity first. It
    observes no phase and integrates no nodal equation."""
    if n < 2:
        return float("inf")  # Invalid input

    # Calculate arithmetic functions
    tau_n = _divisor_count(n)  # τ(n)
    sigma_n = _divisor_sum(n)  # σ(n)
    omega_n = _prime_factor_count(n)  # Ω(n) — with multiplicity

    # TNFR pressure components (pressure lever of nodal equation)
    factorization_pressure = zeta * (omega_n - 1)
    divisor_pressure = eta * (tau_n - 2)
    sigma_pressure = theta * (sigma_n / n - (1 + 1 / n))

    return factorization_pressure + divisor_pressure + sigma_pressure


def tnfr_is_prime(n: int, *, tolerance: float = 1e-10) -> Tuple[bool, float]:
    """Return a numerical zero-pressure predicate and its arithmetic pressure.

    Args:
        n: Integer to test; the default predicate rejects values below 2.
        tolerance: Strict absolute zero tolerance; default 1e-10.

    Returns:
        Tuple ``(abs(pressure) < tolerance, pressure)``. Custom tolerances
        change the predicate; it is not an unconditional accuracy certificate.

    The basic path enumerates divisors through sqrt(n) and trial-factors the
    input. O(sqrt(n)) describes the arithmetic-loop scale, not bit complexity
    or a performance advantage over established primality algorithms.

    Examples:
        >>> tnfr_is_prime(17)
        (True, 0.0)"""
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
            "factorization_pressure": float("inf"),
            "divisor_pressure": float("inf"),
            "abundance_pressure": float("inf"),
            "delta_nfr": float("inf"),
            "omega": 0,
            "tau": 0,
            "sigma": 0,
        }

    tau_n = _divisor_count(n)
    sigma_n = _divisor_sum(n)
    omega_n = _prime_factor_count(n)

    fp = zeta * (omega_n - 1)
    dp = eta * (tau_n - 2)
    ap = theta * (sigma_n / n - (1 + 1 / n))

    return {
        "factorization_pressure": fp,
        "divisor_pressure": dp,
        "abundance_pressure": ap,
        "delta_nfr": fp + dp + ap,
        "omega": omega_n,
        "tau": tau_n,
        "sigma": sigma_n,
    }


def tnfr_structural_triad(
    n: int,
    *,
    zeta: float = ZETA_CANONICAL,
    eta: float = ETA_CANONICAL,
    theta: float = THETA_CANONICAL,
) -> Dict[str, float]:
    """Return a compatibility bundle of static arithmetic diagnostics.

    The retained name refers to ``EPI``, ``vf`` and ``delta_nfr`` fields, with
    ``local_coherence`` and ``components``. It is not the canonical nodal triad
    (EPI, capacity, phase): no phase is returned. EPI and capacity are selected
    functions of supplied Omega, tau and sigma, not a derived autonomous law.

    Args:
        n: Integer; values below 2 use the documented inactive sentinel bundle.
        zeta: Pressure multiplicity weight, default 1.0.
        eta: Pressure divisor weight, default 1.0.
        theta: Pressure abundance weight, default 1.0.

    The function does not evolve n, EPI, capacity or graph support."""
    if n < 2:
        return {
            "EPI": 0.0,
            "vf": 0.0,
            "delta_nfr": float("inf"),
            "local_coherence": 0.0,
            "components": {},
        }

    tau_n = _divisor_count(n)
    sigma_n = _divisor_sum(n)
    omega_n = _prime_factor_count(n)
    log_n = math.log(max(n, 2))

    # EPI: structural form  (α·Ω + β·ln(τ) + γ·(σ/n − 1))
    epi = (
        1.0
        + ALPHA_EPI * omega_n
        + BETA_EPI * math.log(max(tau_n, 1))
        + GAMMA_EPI * (sigma_n / n - 1)
    )

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
        "EPI": epi,
        "vf": vf,
        "delta_nfr": delta_nfr,
        "local_coherence": local_coherence,
        "components": {
            "factorization_pressure": fp,
            "divisor_pressure": dp,
            "abundance_pressure": ap,
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
        return float("inf")

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
    """Compare this numerical predicate with trial division on a finite range.

    This is an implementation consistency check, not validation of the entire
    TNFR framework or independent evidence for emergent physical dynamics.

    Args:
        test_range: Inclusive maximum integer; tests begin at 2.

    Returns:
        Counts and accuracy statistics for the checked range."""

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
        "tested": tested,
        "correct": correct,
        "accuracy": correct / tested if tested > 0 else 0,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "error_rate": (false_positives + false_negatives) / tested if tested > 0 else 0,
    }
