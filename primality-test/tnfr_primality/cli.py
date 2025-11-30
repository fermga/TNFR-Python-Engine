"""
Advanced Command Line Interface for TNFR Primality Testing

Provides access to both standard and advanced TNFR algorithms with repository integration.
Supports performance analysis, benchmarking, caching, and infrastructure diagnostics.

Usage:
    python -m tnfr_primality.cli 17 97 997 --timing
    python -m tnfr_primality.cli --benchmark 10000 --advanced
    python -m tnfr_primality.cli --validate 1000 --infrastructure-status
    python -m tnfr_primality.cli --batch 2 3 5 7 11 13 17 --cached
"""
from __future__ import annotations

import argparse
import time
from typing import List

# Import standard implementations
from .core import tnfr_is_prime, validate_tnfr_theory
from .optimized import OptimizedTNFRPrimality, performance_comparison

# Try to import advanced implementations
try:
    from .advanced_core import (
        tnfr_is_prime_advanced,
        cached_tnfr_is_prime_advanced,
        validate_tnfr_theory_advanced,
        get_infrastructure_status,
        get_system_info,
        HAS_TNFR_INFRASTRUCTURE
    )
    HAS_ADVANCED = True
except ImportError:
    HAS_ADVANCED = False


def main(argv: List[str] | None = None) -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="TNFR-based primality testing using arithmetic pressure equations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m tnfr_primality.cli 17 97 997 9973          # Test specific numbers
  python -m tnfr_primality.cli --timing 982451653      # With timing info
  python -m tnfr_primality.cli --benchmark 10000       # Performance benchmark  
  python -m tnfr_primality.cli --batch --optimized 2 3 5 7  # Batch mode
  python -m tnfr_primality.cli --validate 1000         # Validate theory

TNFR Theory:
  A number n is prime ⟺ ΔNFR(n) = 0, where:
  ΔNFR(n) = ζ·(ω(n)−1) + η·(τ(n)−2) + θ·(σ(n)/n − (1+1/n))
  
  This represents arithmetic pressure in structural coherence systems.
        """
    )
    
    parser.add_argument("numbers", nargs="*", type=int, help="Integers to test for primality")
    parser.add_argument("--timing", action="store_true", help="Show detailed timing information")
    parser.add_argument("--optimized", action="store_true", help="Use optimized implementation with caching")
    parser.add_argument("--batch", action="store_true", help="Batch processing mode for multiple numbers")
    parser.add_argument("--benchmark", type=int, metavar="N", help="Run performance benchmark up to N")
    parser.add_argument("--validate", type=int, metavar="N", help="Validate TNFR theory against traditional methods up to N")
    parser.add_argument("--compare", action="store_true", help="Compare basic vs optimized performance")
    parser.add_argument("--stats", action="store_true", help="Show detailed statistics")
    parser.add_argument("--sieve", type=int, metavar="N", help="Generate primes up to N using TNFR-verified sieve")
    
    args = parser.parse_args(argv)
    
    # Validation mode
    if args.validate:
        print(f"Validating TNFR theory against traditional primality testing up to {args.validate}...")
        print("=" * 70)
        
        start_time = time.perf_counter()
        results = validate_tnfr_theory(args.validate)
        elapsed = time.perf_counter() - start_time
        
        print(f"Numbers tested: {results['tested']:,}")
        print(f"Correct results: {results['correct']:,}")
        print(f"Accuracy: {results['accuracy']:.6f} ({results['accuracy']*100:.4f}%)")
        print(f"False positives: {results['false_positives']}")
        print(f"False negatives: {results['false_negatives']}")
        print(f"Error rate: {results['error_rate']:.8f}")
        print(f"Validation time: {elapsed*1000:.2f} ms")
        
        if results['accuracy'] == 1.0:
            print("✅ TNFR theory validation: PERFECT ACCURACY")
        else:
            print("❌ TNFR theory validation: ERRORS DETECTED")
            
        return 0
    
    # Benchmark mode  
    if args.benchmark:
        print(f"Running TNFR primality benchmark up to {args.benchmark:,}...")
        print("=" * 60)
        
        optimizer = OptimizedTNFRPrimality()
        results = optimizer.benchmark(args.benchmark)
        
        print("Benchmark Results:")
        print("-" * 30)
        print(f"Numbers tested: {results['total_numbers']:,}")
        print(f"Primes found: {results['primes_found']:,}")
        print(f"Composites found: {results['composites_found']:,}")
        print(f"Total time: {results['total_time_ms']:.2f} ms")
        print(f"Average time per number: {results['average_time_us']:.2f} μs")
        print(f"Processing rate: {results['numbers_per_second']:.0f} numbers/second")
        print(f"Cache hit rate: {results['cache_hit_rate']:.2%}")
        print(f"Optimization level: {results['optimization_effectiveness']}")
        
        return 0
    
    # Sieve mode
    if args.sieve:
        print(f"Generating primes up to {args.sieve:,} using TNFR-verified sieve...")
        
        optimizer = OptimizedTNFRPrimality()
        start_time = time.perf_counter()
        primes = optimizer.sieve_primes(args.sieve)
        elapsed = time.perf_counter() - start_time
        
        print(f"Found {len(primes):,} primes in {elapsed*1000:.2f} ms")
        
        if len(primes) <= 100:
            print("Primes found:", primes)
        else:
            print("First 20 primes:", primes[:20])
            print("Last 20 primes:", primes[-20:])
            
        return 0
    
    # Performance comparison mode
    if args.compare and args.numbers:
        print("Comparing basic vs optimized TNFR implementations...")
        print("=" * 60)
        
        results = performance_comparison(args.numbers)
        
        print("Performance Comparison Results:")
        print("-" * 35)
        print(f"Numbers tested: {results['numbers_tested']}")
        print(f"Basic implementation: {results['basic_time_ms']:.2f} ms")
        print(f"Optimized (1st pass): {results['optimized_time_ms']:.2f} ms")
        print(f"Optimized (cached): {results['optimized_cached_time_ms']:.2f} ms")
        print(f"Speedup (1st pass): {results['speedup_first_pass']:.2f}x")
        print(f"Speedup (cached): {results['speedup_cached_pass']:.2f}x")
        print(f"Cache effectiveness: {results['cache_effectiveness']:.2f}x")
        
        return 0
    
    # Regular testing mode
    if not args.numbers:
        print("Error: No numbers provided for testing")
        print("Use --help for usage information")
        return 1
    
    # Choose implementation
    if args.optimized or args.batch:
        optimizer = OptimizedTNFRPrimality()
        
        if args.batch:
            # Batch processing mode
            print("TNFR Batch Primality Testing (Optimized)")
            print("=" * 45)
            
            start_time = time.perf_counter()
            results = optimizer.batch_test(args.numbers)
            batch_time = time.perf_counter() - start_time
            
            # Display results
            if args.timing:
                print(f"{'Number':>12}  {'Prime':>8}  {'ΔNFR':>14}  {'Individual(μs)':>15}")
                print("-" * 65)
            else:
                print(f"{'Number':>12}  {'Prime':>8}  {'ΔNFR':>14}")
                print("-" * 40)
            
            for n, is_prime, delta_nfr in results:
                if args.timing:
                    # Individual timing approximation
                    individual_time = (batch_time / len(args.numbers)) * 1_000_000
                    print(f"{n:12d}  {str(is_prime):>8}  {delta_nfr:14.6f}  {individual_time:15.2f}")
                else:
                    print(f"{n:12d}  {str(is_prime):>8}  {delta_nfr:14.6f}")
            
            print(f"\nBatch Summary:")
            print(f"Total numbers: {len(args.numbers)}")
            print(f"Batch time: {batch_time*1000:.2f} ms") 
            print(f"Average per number: {(batch_time/len(args.numbers))*1_000_000:.2f} μs")
            
            if args.stats:
                print("\nOptimizer Statistics:")
                print("-" * 25)
                stats = optimizer.get_statistics()
                for key, value in stats.items():
                    if isinstance(value, float):
                        print(f"{key}: {value:.4f}")
                    else:
                        print(f"{key}: {value}")
        
        else:
            # Individual optimized testing
            print("TNFR Primality Testing (Optimized)")
            print("=" * 38)
            
            if args.timing:
                header = f"{'Number':>12}  {'Prime':>8}  {'ΔNFR':>14}  {'Time(μs)':>10}"
            else:
                header = f"{'Number':>12}  {'Prime':>8}  {'ΔNFR':>14}"
            
            print(header)
            print("-" * len(header))
            
            for n in args.numbers:
                if args.timing:
                    start = time.perf_counter()
                
                is_prime, delta_nfr = optimizer.is_prime(n)
                
                if args.timing:
                    elapsed_us = (time.perf_counter() - start) * 1_000_000
                    print(f"{n:12d}  {str(is_prime):>8}  {delta_nfr:14.6f}  {elapsed_us:10.2f}")
                else:
                    print(f"{n:12d}  {str(is_prime):>8}  {delta_nfr:14.6f}")
            
            if args.stats:
                print("\nOptimizer Statistics:")
                print("-" * 25)
                stats = optimizer.get_statistics()
                for key, value in stats.items():
                    if isinstance(value, float):
                        print(f"{key}: {value:.4f}")
                    else:
                        print(f"{key}: {value}")
    
    else:
        # Basic implementation mode
        print("TNFR Primality Testing (Basic)")
        print("=" * 32)
        
        if args.timing:
            header = f"{'Number':>12}  {'Prime':>8}  {'ΔNFR':>14}  {'Time(μs)':>10}"
        else:
            header = f"{'Number':>12}  {'Prime':>8}  {'ΔNFR':>14}"
        
        print(header)
        print("-" * len(header))
        
        total_time = 0
        for n in args.numbers:
            if args.timing:
                start = time.perf_counter()
            
            is_prime, delta_nfr = tnfr_is_prime(n)
            
            if args.timing:
                elapsed_us = (time.perf_counter() - start) * 1_000_000
                total_time += elapsed_us
                print(f"{n:12d}  {str(is_prime):>8}  {delta_nfr:14.6f}  {elapsed_us:10.2f}")
            else:
                print(f"{n:12d}  {str(is_prime):>8}  {delta_nfr:14.6f}")
        
        if args.timing and len(args.numbers) > 1:
            avg_time = total_time / len(args.numbers)
            print(f"\nAverage time per number: {avg_time:.2f} μs")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())