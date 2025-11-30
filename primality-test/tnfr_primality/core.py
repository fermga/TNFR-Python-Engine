"""
Core TNFR Primality Testing Implementation

This module contains the fundamental TNFR-based primality testing algorithms
based on the arithmetic pressure equation ΔNFR(n).

Mathematical Foundation:
ΔNFR(n) = ζ·(ω(n)−1) + η·(τ(n)−2) + θ·(σ(n)/n − (1+1/n))

Where:
- ω(n) = number of distinct prime factors  
- τ(n) = number of divisors
- σ(n) = sum of divisors
- ζ=1.0, η=0.8, θ=0.6 = TNFR structural constants

Theorem: n is prime ⟺ ΔNFR(n) = 0
"""
from __future__ import annotations

from typing import Tuple
from functools import lru_cache


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
    """Count distinct prime factors of n."""
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


def tnfr_delta_nfr(n: int, *, zeta: float = 1.0, eta: float = 0.8, theta: float = 0.6) -> float:
    """
    Calculate TNFR arithmetic pressure ΔNFR(n).
    
    The ΔNFR equation quantifies structural pressure in arithmetic systems.
    For prime numbers, this pressure is exactly zero due to their perfect
    structural coherence.
    
    Args:
        n: Integer to analyze
        zeta: Factorization pressure coefficient (default: 1.0)
        eta: Divisor pressure coefficient (default: 0.8) 
        theta: Abundance pressure coefficient (default: 0.6)
        
    Returns:
        ΔNFR value. Zero indicates primality.
        
    Mathematical Derivation:
        - Factorization pressure: ζ·(ω(n)−1)
        - Divisor pressure: η·(τ(n)−2)  
        - Abundance pressure: θ·(σ(n)/n − (1+1/n))
    """
    if n < 2:
        return float('inf')  # Invalid input
    
    # Calculate arithmetic functions
    tau_n = _divisor_count(n)        # τ(n) 
    sigma_n = _divisor_sum(n)        # σ(n)
    omega_n = _prime_factor_count(n)  # ω(n)
    
    # TNFR pressure components
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
        >>> tnfr_is_prime(15)  
        (False, 2.92)
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
    """Cached version of prime factor count."""
    return _prime_factor_count(n)


@lru_cache(maxsize=5000)
def tnfr_delta_nfr_cached(n: int, zeta: float = 1.0, eta: float = 0.8, theta: float = 0.6) -> float:
    """
    Cached version of TNFR ΔNFR computation for enhanced performance.
    
    Uses LRU caching to avoid recomputing expensive arithmetic functions
    for previously analyzed numbers. Especially effective when testing
    multiple numbers with overlapping factor patterns.
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