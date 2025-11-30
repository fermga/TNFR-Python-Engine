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

# Try to import optimized modules
try:
    from tnfr.mathematics.optimized_primality import OptimizedTNFRPrimality
    HAS_OPTIMIZED = True
except ImportError:
    HAS_OPTIMIZED = False
    OptimizedTNFRPrimality = None

# Cache for arithmetic functions to improve performance
@lru_cache(maxsize=10000)
def _divisor_count_cached(n: int) -> int:
    """Cached version of divisor count for performance."""
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
    """Cached version of divisor sum for performance."""
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
    """Cached version of prime factor count for performance."""
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
    return _divisor_count_cached(n)

def _divisor_sum(n: int) -> int:
    return _divisor_sum_cached(n)

def _prime_factor_count(n: int) -> int:
    return _prime_factor_count_cached(n)

@lru_cache(maxsize=5000)
def tnfr_delta_nfr_cached(n: int, zeta: float = 1.0, eta: float = 0.8, theta: float = 0.6) -> float:
    """Cached TNFR ΔNFR computation for performance."""
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
    return tnfr_delta_nfr_cached(n, zeta, eta, theta)


def tnfr_is_prime(n: int, *, use_optimized: bool = False) -> Tuple[bool, float]:
    """TNFR primality test."""
    if use_optimized and HAS_OPTIMIZED:
        # Use optimized version if available
        optimizer = OptimizedTNFRPrimality()
        return optimizer.is_prime(n)
    else:
        # Use basic cached version
        dnfr = tnfr_delta_nfr(n)
        return (abs(dnfr) == 0.0, dnfr)


def _get_optimizer():
    """Get optimizer instance."""
    if HAS_OPTIMIZED:
        return OptimizedTNFRPrimality()
    return None


def benchmark_optimization(max_n: int = 10000) -> dict:
    """Benchmark performance."""
    import random
    
    # Test numbers
    test_numbers = list(range(2, 50))  # Small numbers
    test_numbers.extend([97, 997, 9973, 99991])  # Known primes
    
    if max_n > 100:
        random.seed(42)
        test_numbers.extend(random.sample(range(100, min(max_n, 5000)), min(100, max_n - 100)))
    
    # Basic implementation
    start = time.perf_counter()
    for n in test_numbers:
        tnfr_is_prime(n, use_optimized=False)
    basic_time = time.perf_counter() - start
    
    results = {
        'total_numbers': len(test_numbers),
        'basic_time_ms': basic_time * 1000,
    }
    
    # Optimized implementation if available
    if HAS_OPTIMIZED:
        start = time.perf_counter()
        for n in test_numbers:
            tnfr_is_prime(n, use_optimized=True)
        optimized_time = time.perf_counter() - start
        
        results['optimized_time_ms'] = optimized_time * 1000
        results['speedup'] = basic_time / optimized_time if optimized_time > 0 else 0
    
    return results


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Optimized TNFR primality check using ΔNFR equations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  tnfr-is-prime 17 99991 999983          # Basic usage
  tnfr-is-prime --optimized 982451653    # Use optimizations
  tnfr-is-prime --benchmark 100000       # Benchmark mode
  tnfr-is-prime --batch --stats 17 97    # Batch with statistics
        """
    )
    
    parser.add_argument("numbers", nargs="*", type=int, help="Integers to check")
    parser.add_argument("--optimized", action="store_true", help="Use optimized implementation")
    parser.add_argument("--batch", action="store_true", help="Batch processing mode")
    parser.add_argument("--stats", action="store_true", help="Show performance statistics")
    parser.add_argument("--benchmark", type=int, metavar="N", help="Run benchmark up to N")
    parser.add_argument("--timing", action="store_true", help="Show timing information")
    parser.add_argument("--no-optimize", action="store_true", help="Force basic implementation")
    
    args = parser.parse_args(argv)
    
    # Benchmark mode
    if args.benchmark:
        if not HAS_OPTIMIZED:
            print("Error: Optimized implementation not available for benchmarking")
            return 1
            
        print(f"Running TNFR primality benchmark up to {args.benchmark}...")
        results = benchmark_optimization(args.benchmark)
        
        print("\nBenchmark Results:")
        print("=" * 50)
        print(f"Numbers tested: {results['total_numbers_tested']:,}")
        print(f"Primes found: {results['primes_found']:,}")
        print(f"Prime ratio: {results['prime_ratio']:.4f}")
        print(f"Total time: {results['total_time_ms']:.2f} ms")
        print(f"Average time per number: {results['avg_time_per_number_ms']:.4f} ms")
        print(f"Throughput: {results['throughput_numbers_per_sec']:.0f} numbers/sec")
        print(f"Backend: {results['backend_used']}")
        print(f"Cache hit rate: {results['cache_statistics']['hit_rate']:.1%}")
        print(f"Largest number tested: {results['largest_number_tested']:,}")
        
        return 0
    
    if not args.numbers:
        parser.print_help()
        return 1
    
    # Determine which implementation to use
    use_optimized = (args.optimized or args.batch) and not args.no_optimize and HAS_OPTIMIZED
    
    # Initialize optimizer if needed
    optimizer = None
    if use_optimized:
        optimizer = _get_optimizer()
        if args.stats:
            print(f"Using optimized implementation: {optimizer.backend_name} backend")
            print(f"Sieve coverage: {optimizer.sieve_data['limit']:,} numbers")
            print(f"Primes in sieve: {len(optimizer.sieve_data['primes']):,}")
            print()
    
    # Batch mode with enhanced output
    if args.batch and use_optimized:
        start_time = time.perf_counter()
        results = optimizer.batch_test(args.numbers, include_metrics=args.stats)
        batch_time = time.perf_counter() - start_time
        
        # Display results
        if args.stats:
            header = f"{'n':>12}  {'PRIME':>6}  {'ΔNFR':>12}  {'Time(ms)':>9}  {'Method':>12}  {'Cache':>5}"
        else:
            header = f"{'n':>12}  {'PRIME':>6}  {'ΔNFR':>12}  {'Time(ms)':>9}"
        
        print(header)
        print("-" * len(header))
        
        for result in results:
            if args.stats:
                cache_str = "✓" if result.cache_hit else "✗"
                print(f"{result.n:12d}  {str(result.is_prime):>6}  {result.delta_nfr:12.6f}  "
                      f"{result.computation_time_ms:9.4f}  {result.method:>12}  {cache_str:>5}")
            else:
                print(f"{result.n:12d}  {str(result.is_prime):>6}  {result.delta_nfr:12.6f}  "
                      f"{result.computation_time_ms:9.4f}")
        
        if args.stats:
            print("\nBatch Statistics:")
            print("-" * 20)
            cache_hits = sum(1 for r in results if r.cache_hit)
            print(f"Total time: {batch_time * 1000:.2f} ms")
            print(f"Cache hit rate: {cache_hits / len(results):.1%}")
            print(f"Average per number: {batch_time * 1000 / len(results):.4f} ms")
            
            stats = optimizer.get_statistics()
            print(f"Total cache entries: {stats['cache_size'] + stats['arithmetic_cache_size']}")
    
    else:
        # Standard mode
        if args.timing:
            header = f"{'n':>12}  {'TNFR_PRIME':>11}  {'ΔNFR':>14}  {'Time(μs)':>10}"
        else:
            header = f"{'n':>12}  {'TNFR_PRIME':>11}  {'ΔNFR':>14}"
        
        print(header)
        print("-" * len(header))
        
        for n in args.numbers:
            if args.timing:
                start = time.perf_counter()
            
            isp, dnfr = tnfr_is_prime(n, use_optimized=use_optimized)
            
            if args.timing:
                elapsed_us = (time.perf_counter() - start) * 1_000_000
                print(f"{n:12d}  {str(isp):>11}  {dnfr:14.6f}  {elapsed_us:10.2f}")
            else:
                print(f"{n:12d}  {str(isp):>11}  {dnfr:14.6f}")
    
    # Show final statistics if requested
    if args.stats and use_optimized:
        print("\nOptimizer Statistics:")
        print("-" * 25)
        stats = optimizer.get_statistics()
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
