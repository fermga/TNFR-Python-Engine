"""
Advanced TNFR Primality Testing CLI with Repository Integration

Enhanced CLI providing access to both standard and advanced TNFR algorithms.
Includes infrastructure diagnostics, performance analytics, and caching support.

Author: F. F. Martinez Gamo
"""
from __future__ import annotations

import argparse
import time
import json
from typing import Dict, Any

# Import standard implementations
from .core import tnfr_is_prime, validate_tnfr_theory

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


def print_header():
    """Print enhanced CLI header with infrastructure status."""
    print("=" * 70)
    print("TNFR Advanced Primality Testing System")
    print("F. F. Martinez Gamo")
    print("DOI: 10.5281/zenodo.17764749")
    print("=" * 70)
    
    if HAS_ADVANCED:
        print(f"✓ Advanced TNFR infrastructure: {'AVAILABLE' if HAS_TNFR_INFRASTRUCTURE else 'LIMITED'}")
    else:
        print("! Advanced infrastructure: NOT AVAILABLE (using fallback algorithms)")
    print()


def test_single_number(n: int, use_advanced: bool = False, use_cached: bool = False, timing: bool = False) -> Dict[str, Any]:
    """Test a single number with enhanced reporting."""
    start_time = time.perf_counter()
    
    if use_advanced and HAS_ADVANCED:
        if use_cached:
            is_prime, delta_nfr = cached_tnfr_is_prime_advanced(n)
            method = "Advanced (Cached)"
        else:
            is_prime, delta_nfr = tnfr_is_prime_advanced(n)
            method = "Advanced"
    else:
        is_prime, delta_nfr = tnfr_is_prime(n)
        method = "Standard"
    
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    
    result = {
        'number': n,
        'is_prime': is_prime,
        'delta_nfr': delta_nfr,
        'method': method,
        'time_ms': elapsed_ms
    }
    
    # Enhanced output
    status = "PRIME" if is_prime else "COMPOSITE"
    print(f"{n:>10} | {status:<9} | {delta_nfr:>12.8f} | {method}")
    
    if timing:
        print(f"           | Time: {elapsed_ms:.3f} ms")
    
    return result


def run_benchmark(max_n: int, use_advanced: bool = False, use_cached: bool = False) -> Dict[str, Any]:
    """Run performance benchmark with advanced analytics."""
    print(f"Running benchmark up to {max_n}...")
    print(f"Algorithm: {'Advanced' if use_advanced and HAS_ADVANCED else 'Standard'}")
    print(f"Caching: {'Enabled' if use_cached else 'Disabled'}")
    print()
    
    start_time = time.perf_counter()
    results = []
    
    # Test a representative sample
    test_numbers = [
        2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,
        97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149,
        997, 1009, 1013, 1019, 1021, 1031, 1033, 1039, 1049, 1051
    ]
    
    test_numbers = [n for n in test_numbers if n <= max_n]
    
    print(f"{'Number':<10} | {'Status':<9} | {'ΔNFR Value':<12} | {'Method'}")
    print("-" * 50)
    
    for n in test_numbers:
        result = test_single_number(n, use_advanced, use_cached, timing=False)
        results.append(result)
    
    total_time = time.perf_counter() - start_time
    
    # Calculate statistics
    times = [r['time_ms'] for r in results]
    avg_time = sum(times) / len(times) if times else 0
    total_numbers = len(results)
    
    print()
    print("Benchmark Results:")
    print(f"  Numbers tested: {total_numbers}")
    print(f"  Total time: {total_time * 1000:.2f} ms")
    print(f"  Average per number: {avg_time:.3f} ms")
    print(f"  Numbers per second: {total_numbers / total_time:.1f}")
    
    return {
        'total_numbers': total_numbers,
        'total_time_ms': total_time * 1000,
        'average_time_ms': avg_time,
        'numbers_per_second': total_numbers / total_time if total_time > 0 else 0,
        'results': results
    }


def run_validation(max_n: int, use_advanced: bool = False) -> Dict[str, Any]:
    """Run theory validation with enhanced reporting."""
    print(f"Validating TNFR theory up to {max_n}...")
    print(f"Algorithm: {'Advanced' if use_advanced and HAS_ADVANCED else 'Standard'}")
    print()
    
    start_time = time.perf_counter()
    
    if use_advanced and HAS_ADVANCED:
        results = validate_tnfr_theory_advanced(max_n)
    else:
        results = validate_tnfr_theory(max_n)
    
    elapsed_time = time.perf_counter() - start_time
    
    print("Validation Results:")
    print(f"  Numbers tested: {results['tested_numbers']}")
    print(f"  Correct predictions: {results['correct_predictions']}")
    print(f"  False positives: {results['false_positives']}")
    print(f"  False negatives: {results['false_negatives']}")
    print(f"  Accuracy: {results['accuracy']:.6f} ({results['accuracy'] * 100:.4f}%)")
    
    if 'prime_mean_delta_nfr' in results:
        print(f"  Prime mean ΔNFR: {results['prime_mean_delta_nfr']:.8f}")
        print(f"  Composite mean ΔNFR: {results['composite_mean_delta_nfr']:.8f}")
    
    print(f"  Validation time: {elapsed_time * 1000:.2f} ms")
    print(f"  Numbers per second: {results['tested_numbers'] / elapsed_time:.1f}")
    
    return results


def show_infrastructure_status():
    """Display detailed infrastructure status."""
    print("TNFR Infrastructure Status")
    print("=" * 40)
    
    if HAS_ADVANCED:
        print(get_infrastructure_status())
        print()
        
        system_info = get_system_info()
        print("System Information:")
        print(f"  Python version: {system_info['python_version'].split()[0]}")
        print(f"  Advanced infrastructure: {system_info['infrastructure_available']}")
        print(f"  Cache available: {system_info['cache_available']}")
        
        print("\nTNFR Constants:")
        constants = system_info['constants']
        for name, value in constants.items():
            if isinstance(value, float):
                print(f"  {name}: {value:.12f}")
            else:
                print(f"  {name}: {value}")
        
    else:
        print("Advanced TNFR infrastructure not available.")
        print("Running in fallback mode with standard algorithms.")


def main() -> int:
    """Enhanced main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Advanced TNFR primality testing with repository integration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  tnfr-primality 17 97 997                    # Test specific numbers
  tnfr-primality --benchmark 10000 --advanced # Advanced benchmark
  tnfr-primality --validate 1000 --advanced   # Advanced validation
  tnfr-primality --infrastructure-status      # Check infrastructure
  tnfr-primality --batch 2 3 5 7 11 --cached # Cached batch processing

TNFR Theory:
  A number n is prime ⟺ ΔNFR(n) = 0, where:
  ΔNFR(n) = ζ·(ω(n)−1) + η·(τ(n)−2) + θ·(σ(n)/n − (1+1/n))
        """
    )
    
    # Positional arguments
    parser.add_argument("numbers", nargs="*", type=int,
                        help="Integers to test for primality")
    
    # Algorithm options
    parser.add_argument("--advanced", action="store_true",
                        help="Use advanced TNFR algorithms (requires infrastructure)")
    parser.add_argument("--cached", action="store_true",
                        help="Use cached computation (improves performance)")
    
    # Operation modes
    parser.add_argument("--benchmark", type=int, metavar="MAX_N",
                        help="Run performance benchmark up to MAX_N")
    parser.add_argument("--validate", type=int, metavar="MAX_N",
                        help="Validate TNFR theory up to MAX_N")
    parser.add_argument("--batch", action="store_true",
                        help="Process numbers in batch mode")
    
    # Information and diagnostics
    parser.add_argument("--infrastructure-status", action="store_true",
                        help="Show TNFR infrastructure status")
    parser.add_argument("--timing", action="store_true",
                        help="Show detailed timing information")
    parser.add_argument("--json-output", action="store_true",
                        help="Output results in JSON format")
    
    args = parser.parse_args()
    
    if not args.json_output:
        print_header()
    
    # Infrastructure status check
    if args.infrastructure_status:
        show_infrastructure_status()
        return 0
    
    # Benchmark mode
    if args.benchmark is not None:
        if args.json_output:
            results = run_benchmark(args.benchmark, args.advanced, args.cached)
            print(json.dumps(results, indent=2))
        else:
            run_benchmark(args.benchmark, args.advanced, args.cached)
        return 0
    
    # Validation mode 
    if args.validate is not None:
        if args.json_output:
            results = run_validation(args.validate, args.advanced)
            print(json.dumps(results, indent=2))
        else:
            run_validation(args.validate, args.advanced)
        return 0
    
    # Test specific numbers
    if args.numbers:
        if not args.json_output:
            print(f"{'Number':<10} | {'Status':<9} | {'ΔNFR Value':<12} | {'Method'}")
            print("-" * 50)
        
        results = []
        for n in args.numbers:
            result = test_single_number(n, args.advanced, args.cached, args.timing)
            results.append(result)
        
        if args.json_output:
            print(json.dumps(results, indent=2))
        
        return 0
    
    # No arguments provided
    parser.print_help()
    return 1


if __name__ == "__main__":
    exit(main())
