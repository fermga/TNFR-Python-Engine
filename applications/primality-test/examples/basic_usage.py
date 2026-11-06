"""Small static arithmetic-pressure examples, including cache observations.

The package computes divisor and factor statistics before evaluating pressure.
These examples do not evolve a nodal network or establish a general accuracy,
physical interpretation or algorithmic speedup.
"""

import os
import sys
import time

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tnfr_primality import (
    OptimizedTNFRPrimality,
    tnfr_component_breakdown,
    tnfr_is_prime,
)
from tnfr_primality.constants import ETA_CANONICAL, THETA_CANONICAL, ZETA_CANONICAL


def basic_example():
    """Demonstrate basic TNFR primality testing."""
    print("=== Basic TNFR Primality Testing ===")

    test_numbers = [2, 3, 4, 5, 15, 17, 25, 97, 99, 997]

    for n in test_numbers:
        is_prime, delta_nfr = tnfr_is_prime(n)
        print(
            f"{n:3d}: {'Prime' if is_prime else 'Composite':10} "
            f"(DeltaNFR = {delta_nfr:.6f})"
        )

    print("\nWith positive coefficients, exact arithmetic pressure vanishes at primes.")
    print("The implementation applies a numerical tolerance to floating values.")


def performance_example():
    """Demonstrate performance characteristics."""
    print("\n=== Performance Demonstration ===")

    large_numbers = [982451653, 2147483647, 4294967291]

    for n in large_numbers:
        start = time.perf_counter()
        is_prime, delta_nfr = tnfr_is_prime(n)
        elapsed_ms = (time.perf_counter() - start) * 1000

        print(
            f"{n:,}: {'Prime' if is_prime else 'Composite'} "
            f"(Time: {elapsed_ms:.2f} ms, DeltaNFR = {delta_nfr:.6f})"
        )


def optimized_example():
    """Demonstrate optimized implementation with caching."""
    print("\n=== Optimized Implementation with Caching ===")

    optimizer = OptimizedTNFRPrimality()

    # Test batch of numbers
    test_numbers = [97, 997, 9973, 99991, 982451653]

    print("First pass (cache building):")
    start = time.perf_counter()
    results = optimizer.batch_test(test_numbers)
    first_pass_time = time.perf_counter() - start

    for n, is_prime, delta_nfr in results:
        print(f"  {n:,}: {'Prime' if is_prime else 'Composite'}")

    print(f"First pass time: {first_pass_time*1000:.2f} ms")

    print("\nSecond pass (cached results):")
    start = time.perf_counter()
    for n in test_numbers:
        optimizer.is_prime(n)
    second_pass_time = time.perf_counter() - start

    print(f"Second pass time: {second_pass_time*1000:.2f} ms")
    if second_pass_time > 0:
        print(
            f"Observed first/cached time ratio: {first_pass_time/second_pass_time:.1f}"
        )
    else:
        print("Observed time ratio unavailable at this clock resolution")
    print("This compares repeated inputs in one process, not algorithmic complexity.")

    # Show statistics
    stats = optimizer.get_statistics()
    print(f"Cache hit rate: {stats['cache_hit_rate']:.2%}")


def theory_example():
    """Expose the actual default coefficients and arithmetic inputs."""
    print("\n=== Arithmetic-Pressure Components ===")

    print("DeltaNFR = zeta*(Omega-1) + eta*(tau-2) + theta*(sigma/n-(1+1/n))")
    print(
        "Omega counts prime factors WITH multiplicity; tau and sigma count/sum divisors."
    )
    print(
        f"Selected weights: zeta={ZETA_CANONICAL}, eta={ETA_CANONICAL}, "
        f"theta={THETA_CANONICAL}"
    )
    print()

    examples = [
        (7, "Prime example"),
        (15, "Composite example (3*5)"),
        (25, "Perfect square (5**2)"),
    ]

    for n, description in examples:
        components = tnfr_component_breakdown(n)
        is_prime, _ = tnfr_is_prime(n)

        print(f"{description}:")
        print(f"  n = {n}")
        print(f"  DeltaNFR({n}) = {components['delta_nfr']:.6f}")
        print(
            f"  Omega={components['omega']}, tau={components['tau']}, "
            f"sigma={components['sigma']}"
        )
        print(f"  Result: {'Prime' if is_prime else 'Composite'}")
        print()


def comparison_example():
    """State the computation's scope without an unmeasured comparison."""
    print("=== Interpretation ===")
    print("Divisor enumeration and trial factorization supply the inputs.")
    print("Positive weights preserve the exact prime zero set; they are not unique.")
    print("Finite checks and cached timings do not prove arbitrary-integer accuracy")
    print("or improve the complexity of established primality methods.")


def main():
    """Run all examples."""
    print("TNFR Primality Testing - Usage Examples")
    print("======================================")

    basic_example()
    performance_example()
    optimized_example()
    theory_example()
    comparison_example()

    print("\n" + "=" * 50)
    print("[OK] Examples completed")
    print("See applications/primality-test/README.md for usage and limits.")


if __name__ == "__main__":
    main()
