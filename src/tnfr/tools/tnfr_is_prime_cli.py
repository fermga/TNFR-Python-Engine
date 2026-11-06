"""Console entry point for the declared TNFR arithmetic primality models.

The positive arithmetic-pressure weights characterize primality; this is not a
derivation of physical nodal dynamics. The basic path uses bounded LRU reuse,
while optimized and batch modes delegate to the shared mathematics owner.
"""

from __future__ import annotations

import argparse
import time
from functools import lru_cache

from ..mathematics.arithmetic_pressure import big_omega, divisor_sum, num_divisors

try:
    from ..mathematics.optimized_primality import OptimizedTNFRPrimality
    from ..mathematics.optimized_primality import (
        benchmark_optimization as _benchmark_optimization,
    )

    HAS_OPTIMIZED = True
except ImportError:
    HAS_OPTIMIZED = False
    OptimizedTNFRPrimality = None
    _benchmark_optimization = None


# Reuse the arithmetic owner rather than maintaining another factorization.
_divisor_count = num_divisors
_divisor_sum = divisor_sum
_prime_factor_count = big_omega
_divisor_count_cached = lru_cache(maxsize=10000)(_divisor_count)
_divisor_sum_cached = lru_cache(maxsize=10000)(_divisor_sum)
_prime_factor_count_cached = lru_cache(maxsize=10000)(_prime_factor_count)


@lru_cache(maxsize=5000)
def tnfr_delta_nfr_cached(
    n: int, zeta: float = 1.0, eta: float = 0.8, theta: float = 0.6
) -> float:
    """Return the historical configured pressure, with bounded argument reuse."""
    if n < 2:
        return float("inf")
    tau_n = _divisor_count_cached(n)
    sigma_n = _divisor_sum_cached(n)
    omega_n = _prime_factor_count_cached(n)
    return (
        zeta * (omega_n - 1) + eta * (tau_n - 2) + theta * (sigma_n / n - (1 + 1 / n))
    )


def tnfr_delta_nfr(n: int, *, zeta=1.0, eta=0.8, theta=0.6) -> float:
    """Return the basic cached arithmetic pressure."""
    return tnfr_delta_nfr_cached(n, zeta, eta, theta)


@lru_cache(maxsize=1)
def _get_optimizer():
    """Reuse the actual optimizer, including its sieve and result caches."""
    if HAS_OPTIMIZED:
        return OptimizedTNFRPrimality()
    return None


def tnfr_is_prime(n: int, *, use_optimized: bool = False) -> tuple[bool, float]:
    """Return a decision and pressure from the requested available owner."""
    if use_optimized and HAS_OPTIMIZED:
        result = _get_optimizer().is_prime_optimized(n)
        return result.is_prime, result.delta_nfr
    pressure = tnfr_delta_nfr(n)
    return pressure == 0.0, pressure


def benchmark_optimization(max_n: int = 10000, sample_size: int = 1000) -> dict:
    """Delegate to the shared benchmark and preserve its report schema."""
    if max_n < 100:
        raise ValueError("benchmark maximum must be at least 100")
    if not HAS_OPTIMIZED:
        raise RuntimeError("Optimized implementation not available for benchmarking")
    return _benchmark_optimization(max_n, sample_size)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="TNFR primality check using configured arithmetic pressure",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  tnfr-is-prime 17 99991 999983          # Basic usage
  tnfr-is-prime --optimized 982451653    # Use optimizations
  tnfr-is-prime --benchmark 100000       # Benchmark mode
  tnfr-is-prime --batch --stats 17 97    # Batch with statistics
        """,
    )

    parser.add_argument("numbers", nargs="*", type=int, help="Integers to check")
    parser.add_argument(
        "--optimized", action="store_true", help="Use optimized implementation"
    )
    parser.add_argument(
        "--cached", action="store_true", help="Use the basic cached implementation"
    )
    parser.add_argument(
        "--batch", action="store_true", help="Process sorted unique inputs as a batch"
    )
    parser.add_argument(
        "--stats", action="store_true", help="Show performance statistics"
    )
    parser.add_argument(
        "--benchmark", type=int, metavar="N", help="Run benchmark up to N"
    )
    parser.add_argument("--timing", action="store_true", help="Show timing information")
    parser.add_argument(
        "--no-optimize", action="store_true", help="Force basic implementation"
    )

    args = parser.parse_args(argv)

    # Benchmark mode
    if args.benchmark is not None:
        if args.benchmark < 100:
            parser.error("--benchmark requires a maximum of at least 100")
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
    use_optimized = (
        (args.optimized or args.batch)
        and not (args.no_optimize or args.cached)
        and HAS_OPTIMIZED
    )

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
        # Performance statistics do not consume per-number structural telemetry.
        results = optimizer.batch_test(args.numbers)
        batch_time = time.perf_counter() - start_time

        # Display results
        if args.stats:
            header = f"{'n':>12}  {'PRIME':>6}  {'DeltaNFR':>12}  {'Time(ms)':>9}  {'Method':>12}  {'Cache':>5}"
        else:
            header = f"{'n':>12}  {'PRIME':>6}  {'DeltaNFR':>12}  {'Time(ms)':>9}"

        print(header)
        print("-" * len(header))

        for result in results:
            if args.stats:
                cache_str = "yes" if result.cache_hit else "no"
                print(
                    f"{result.n:12d}  {str(result.is_prime):>6}  {result.delta_nfr:12.6f}  "
                    f"{result.computation_time_ms:9.4f}  {result.method:>12}  {cache_str:>5}"
                )
            else:
                print(
                    f"{result.n:12d}  {str(result.is_prime):>6}  {result.delta_nfr:12.6f}  "
                    f"{result.computation_time_ms:9.4f}"
                )

        if args.stats:
            print("\nBatch Statistics:")
            print("-" * 20)
            cache_hits = sum(1 for r in results if r.cache_hit)
            print(f"Total time: {batch_time * 1000:.2f} ms")
            print(f"Cache hit rate: {cache_hits / len(results):.1%}")
            print(f"Average per number: {batch_time * 1000 / len(results):.4f} ms")

            stats = optimizer.get_statistics()
            print(
                f"Total cache entries: {stats['cache_size'] + stats['arithmetic_cache_size']}"
            )

    else:
        # Standard mode
        if args.timing:
            header = (
                f"{'n':>12}  {'TNFR_PRIME':>11}  {'DeltaNFR':>14}  {'Time(us)':>10}"
            )
        else:
            header = f"{'n':>12}  {'TNFR_PRIME':>11}  {'DeltaNFR':>14}"

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
