"""
Optimized TNFR primality checker as a console script entry.

Usage after install:
  tnfr-is-prime 17 99991 999983
  tnfr-is-prime --timing 17 997 9973    # With timing
  tnfr-is-prime --cached 17 97 197      # Force cached mode

A number is prime iff ΔNFR(n) == 0, using the TNFR arithmetic pressure equation.
This optimized version uses LRU caching for significant performance improvements.
"""
from __future__ import annotations

import argparse
import time
from typing import List, Tuple
from functools import lru_cache


# Cached arithmetic functions for optimal performance
@lru_cache(maxsize=10000)
def _divisor_count_cached(n: int) -> int:
    """Optimized divisor count with LRU caching."""
    cnt = 0
    i = 1
    while i * i <= n:
        if n % i == 0:
            cnt += 1
            if i != n // i:
                cnt += 1
        i += 1
    return cnt


@lru_cache(maxsize=10000)
def _divisor_sum_cached(n: int) -> int:
    """Optimized divisor sum with LRU caching."""
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


@lru_cache(maxsize=10000)
def _prime_factor_count_cached(n: int) -> int:
    """Optimized prime factor count (ω function) with LRU caching."""
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


# Non-cached versions for compatibility
def _divisor_count(n: int) -> int:
    """Basic divisor count (τ function)."""
    cnt = 0
    i = 1
    while i * i <= n:
        if n % i == 0:
            cnt += 1
            if i != n // i:
                cnt += 1
        i += 1
    return cnt


def _divisor_sum(n: int) -> int:
    """Basic divisor sum (σ function)."""
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
    """Basic prime factor count (ω function)."""
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


@lru_cache(maxsize=5000)
def tnfr_delta_nfr_cached(n: int, zeta: float = 1.0, eta: float = 0.8, theta: float = 0.6) -> float:
    """
    Cached TNFR ΔNFR computation for optimal performance.
    
    ΔNFR(n) = ζ·(ω(n)−1) + η·(τ(n)−2) + θ·(σ(n)/n − (1+1/n))
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


def tnfr_delta_nfr(n: int, *, zeta=1.0, eta=0.8, theta=0.6) -> float:
    """Basic TNFR ΔNFR computation."""
    if n < 2:
        return float('inf')
    tau_n = _divisor_count(n)
    sigma_n = _divisor_sum(n)
    omega_n = _prime_factor_count(n)
    factorization_pressure = zeta * (omega_n - 1)
    divisor_pressure = eta * (tau_n - 2)
    sigma_pressure = theta * (sigma_n / n - (1 + 1 / n))
    return factorization_pressure + divisor_pressure + sigma_pressure


def tnfr_is_prime(n: int, *, use_cached: bool = False) -> Tuple[bool, float]:
    """TNFR primality test with optional caching."""
    if use_cached:
        dnfr = tnfr_delta_nfr_cached(n)
    else:
        dnfr = tnfr_delta_nfr(n)
    return (abs(dnfr) == 0.0, dnfr)


def benchmark_basic(max_n: int = 10000, sample_size: int = 100) -> dict:
    """Simple benchmark of TNFR primality testing."""
    import random
    
    # Generate test numbers
    test_numbers = list(range(2, 30))  # Small numbers
    test_numbers.extend([97, 997, 9973, 99991])  # Known primes
    
    # Add random samples
    random.seed(42)
    if max_n > 100:
        test_numbers.extend(random.sample(range(100, min(max_n, 10000)), min(sample_size, max_n - 100)))
    
    # Test basic implementation
    start = time.perf_counter()
    basic_results = []
    for n in test_numbers:
        is_prime, delta_nfr = tnfr_is_prime(n, use_cached=False)
        basic_results.append((n, is_prime, delta_nfr))
    basic_time = time.perf_counter() - start
    
    # Test cached implementation  
    start = time.perf_counter()
    cached_results = []
    for n in test_numbers:
        is_prime, delta_nfr = tnfr_is_prime(n, use_cached=True)
        cached_results.append((n, is_prime, delta_nfr))
    cached_time = time.perf_counter() - start
    
    # Second run to test cache effectiveness
    start = time.perf_counter()
    for n in test_numbers:
        tnfr_is_prime(n, use_cached=True)
    cached_time_2nd = time.perf_counter() - start
    
    prime_count = sum(1 for _, is_prime, _ in basic_results if is_prime)
    
    return {
        'total_numbers': len(test_numbers),
        'primes_found': prime_count,
        'prime_ratio': prime_count / len(test_numbers),
        'basic_time_ms': basic_time * 1000,
        'cached_time_ms': cached_time * 1000,
        'cached_2nd_time_ms': cached_time_2nd * 1000,
        'speedup_1st_run': basic_time / cached_time if cached_time > 0 else 0,
        'speedup_2nd_run': basic_time / cached_time_2nd if cached_time_2nd > 0 else 0,
        'cache_info': {
            'delta_nfr': tnfr_delta_nfr_cached.cache_info()._asdict(),
            'divisor_count': _divisor_count_cached.cache_info()._asdict(),
            'divisor_sum': _divisor_sum_cached.cache_info()._asdict(),
            'prime_factor_count': _prime_factor_count_cached.cache_info()._asdict(),
        }
    }


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Optimized TNFR primality check using ΔNFR equations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=\"\"\"
Examples:
  tnfr-is-prime 17 99991 999983          # Basic usage
  tnfr-is-prime --cached 982451653       # Use caching
  tnfr-is-prime --benchmark 10000        # Benchmark mode  
  tnfr-is-prime --timing 17 97           # With timing
        \"\"\"
    )
    
    parser.add_argument("numbers", nargs="*", type=int, help="Integers to check")
    parser.add_argument("--cached", action="store_true", help="Use cached implementation")
    parser.add_argument("--timing", action="store_true", help="Show timing information")
    parser.add_argument("--benchmark", type=int, metavar="N", help="Run benchmark up to N")
    parser.add_argument("--stats", action="store_true", help="Show cache statistics")
    
    args = parser.parse_args(argv)
    
    # Benchmark mode
    if args.benchmark:
        print(f"Running TNFR primality benchmark up to {args.benchmark}...")
        results = benchmark_basic(args.benchmark)
        
        print("\\nBenchmark Results:")
        print("=" * 50)
        print(f"Numbers tested: {results['total_numbers']:,}")
        print(f"Primes found: {results['primes_found']:,}")
        print(f"Prime ratio: {results['prime_ratio']:.4f}")
        print(f"Basic implementation: {results['basic_time_ms']:.2f} ms")
        print(f"Cached (1st run): {results['cached_time_ms']:.2f} ms")
        print(f"Cached (2nd run): {results['cached_2nd_time_ms']:.2f} ms")
        print(f"Speedup (1st run): {results['speedup_1st_run']:.2f}x")
        print(f"Speedup (2nd run): {results['speedup_2nd_run']:.2f}x")
        
        if args.stats:
            print("\\nCache Statistics:")
            print("-" * 30)
            for func_name, cache_info in results['cache_info'].items():
                print(f"{func_name}:")
                print(f"  Hits: {cache_info['hits']:,}")
                print(f"  Misses: {cache_info['misses']:,}")
                print(f"  Hit rate: {cache_info['hits']/(cache_info['hits']+cache_info['misses']):.1%}")
        
        return 0
    
    if not args.numbers:
        parser.print_help()
        return 1
    
    # Standard mode
    if args.timing:
        header = f"{'n':>12}  {'TNFR_PRIME':>11}  {'ΔNFR':>14}  {'Time(μs)':>10}"
    else:
        header = f"{'n':>12}  {'TNFR_PRIME':>11}  {'ΔNFR':>14}"
    
    print(header)
    print("-" * len(header))
    
    total_time = 0
    for n in args.numbers:
        if args.timing:
            start = time.perf_counter()
        
        is_prime, delta_nfr = tnfr_is_prime(n, use_cached=args.cached)
        
        if args.timing:
            elapsed_us = (time.perf_counter() - start) * 1_000_000
            total_time += elapsed_us
            print(f"{n:12d}  {str(is_prime):>11}  {delta_nfr:14.6f}  {elapsed_us:10.2f}")
        else:
            print(f"{n:12d}  {str(is_prime):>11}  {delta_nfr:14.6f}")
    
    # Show cache statistics if requested
    if args.stats and args.cached:
        print("\\nCache Statistics:")
        print("-" * 25)
        cache_stats = {
            'delta_nfr': tnfr_delta_nfr_cached.cache_info(),
            'divisor_count': _divisor_count_cached.cache_info(),
            'divisor_sum': _divisor_sum_cached.cache_info(),
            'prime_factor_count': _prime_factor_count_cached.cache_info(),
        }
        
        total_hits = sum(info.hits for info in cache_stats.values())
        total_requests = sum(info.hits + info.misses for info in cache_stats.values())
        overall_hit_rate = total_hits / total_requests if total_requests > 0 else 0
        
        print(f"Overall hit rate: {overall_hit_rate:.1%}")
        for func_name, cache_info in cache_stats.items():
            requests = cache_info.hits + cache_info.misses
            if requests > 0:
                hit_rate = cache_info.hits / requests
                print(f"{func_name}: {cache_info.hits}/{requests} ({hit_rate:.1%})")
    
    if args.timing and len(args.numbers) > 1:
        avg_time = total_time / len(args.numbers)
        print(f"\\nAverage time: {avg_time:.2f} μs/number")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())