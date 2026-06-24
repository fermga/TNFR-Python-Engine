"""
Basic usage examples for TNFR primality testing.

This script demonstrates the core functionality of the TNFR-based
primality testing package with simple, clear examples.
"""

import os
import sys
import time

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tnfr_primality import OptimizedTNFRPrimality, tnfr_delta_nfr, tnfr_is_prime


def basic_example():
    """Demonstrate basic TNFR primality testing."""
    print("=== Basic TNFR Primality Testing ===")

    test_numbers = [2, 3, 4, 5, 15, 17, 25, 97, 99, 997]

    for n in test_numbers:
        is_prime, delta_nfr = tnfr_is_prime(n)
        print(
            f"{n:3d}: {'Prime' if is_prime else 'Composite':10} (ΔNFR = {delta_nfr:.6f})"
        )

    print("\nKey insight: Prime numbers have ΔNFR = 0 exactly!")
    print("Composite numbers have ΔNFR > 0 due to structural pressure.")


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
            f"(Time: {elapsed_ms:.2f} ms, ΔNFR = {delta_nfr:.6f})"
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
    print(f"Cache speedup: {first_pass_time/second_pass_time:.1f}x")

    # Show statistics
    stats = optimizer.get_statistics()
    print(f"Cache hit rate: {stats['cache_hit_rate']:.2%}")


def theory_example():
    """Demonstrate the underlying TNFR theory."""
    print("\n=== TNFR Theory Demonstration ===")

    print("TNFR Equation: ΔNFR(n) = ζ·(ω(n)−1) + η·(τ(n)−2) + θ·(σ(n)/n − (1+1/n))")
    print("Where: ω(n) = prime factors, τ(n) = divisors, σ(n) = sum of divisors")
    print("Constants: ζ=1.0, η=0.8, θ=0.6 (TNFR structural parameters)")
    print()

    examples = [
        (7, "Prime example"),
        (15, "Composite example (3×5)"),
        (25, "Perfect square (5²)"),
    ]

    for n, description in examples:
        delta_nfr = tnfr_delta_nfr(n)
        is_prime, _ = tnfr_is_prime(n)

        print(f"{description}:")
        print(f"  n = {n}")
        print(f"  ΔNFR({n}) = {delta_nfr:.6f}")
        print(f"  Result: {'Prime' if is_prime else 'Composite'}")
        print(
            f"  Explanation: {'Perfect coherence (ΔNFR=0)' if is_prime else 'Structural pressure (ΔNFR>0)'}"
        )
        print()


def comparison_example():
    """Compare TNFR with traditional methods conceptually."""
    print("=== TNFR vs Traditional Primality Testing ===")

    print("Traditional approaches:")
    print("  • Trial division: Test divisibility by all numbers up to √n")
    print("  • Miller-Rabin: Probabilistic testing based on Fermat's little theorem")
    print("  • AKS: Polynomial-time deterministic (but impractical)")
    print()

    print("TNFR approach:")
    print("  • Arithmetic pressure analysis: Measure structural coherence")
    print("  • Deterministic: Always gives correct answer (no probability)")
    print("  • Novel foundation: Based on resonant fractal nature theory")
    print("  • Competitive performance: O(√n) with caching optimizations")
    print()

    print("Key advantage: TNFR provides insight into WHY a number is prime")
    print("(perfect structural coherence) rather than just IF it is prime.")


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
    print("✅ All examples completed successfully!")
    print("📖 See README.md for more detailed documentation")
    print("🔬 Run benchmarks/comprehensive_benchmark.py for performance analysis")


if __name__ == "__main__":
    main()
