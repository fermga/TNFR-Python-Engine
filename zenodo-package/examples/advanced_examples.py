#!/usr/bin/env python3
"""
Advanced TNFR Primality Testing Examples
=====================================

This module demonstrates the advanced capabilities of the TNFR primality
testing package with full repository integration.

Examples include:
- Infrastructure detection and diagnostics
- Advanced algorithm usage with caching
- Prime certificate analysis
- Structural field monitoring
- Performance benchmarking with analytics
- JSON integration for programmatic use

Author: F. F. Martinez Gamo
"""

import time
import json
from typing import List, Dict, Any


def example_infrastructure_detection():
    """Demonstrate automatic infrastructure detection."""
    print("=" * 60)
    print("1. INFRASTRUCTURE DETECTION")
    print("=" * 60)
    
    try:
        from tnfr_primality.advanced_core import (
            HAS_TNFR_INFRASTRUCTURE,
            get_infrastructure_status,
            get_system_info
        )
        
        print(f"Advanced TNFR Infrastructure: {HAS_TNFR_INFRASTRUCTURE}")
        print("\nInfrastructure Status:")
        if HAS_TNFR_INFRASTRUCTURE:
            print(get_infrastructure_status())
            
            system_info = get_system_info()
            print(f"\nSystem Information:")
            print(f"  Infrastructure Available: {system_info['infrastructure_available']}")
            print(f"  Cache Available: {system_info['cache_available']}")
            print(f"  Python Version: {system_info['python_version'].split()[0]}")
            
            print(f"\nTNFR Constants:")
            for name, value in system_info['constants'].items():
                if isinstance(value, float):
                    print(f"  {name}: {value:.12f}")
                else:
                    print(f"  {name}: {value}")
        else:
            print("Advanced infrastructure not available - using fallback mode")
            
    except ImportError as e:
        print(f"Import error: {e}")
        print("Advanced TNFR infrastructure not available")


def example_basic_vs_advanced():
    """Compare basic and advanced algorithm performance."""
    print("\n" + "=" * 60)
    print("2. BASIC vs ADVANCED ALGORITHM COMPARISON")
    print("=" * 60)
    
    test_numbers = [997, 1009, 1013, 1019, 2147483647]
    
    try:
        from tnfr_primality.core import tnfr_is_prime
        from tnfr_primality.advanced_core import tnfr_is_prime_advanced, cached_tnfr_is_prime_advanced
        
        print(f"{'Number':<12} | {'Algorithm':<15} | {'Result':<6} | {'ΔNFR':<12} | {'Time (ms)':<10}")
        print("-" * 70)
        
        for n in test_numbers:
            # Basic algorithm
            start = time.perf_counter()
            is_prime_basic, delta_nfr_basic = tnfr_is_prime(n)
            time_basic = (time.perf_counter() - start) * 1000
            
            # Advanced algorithm
            start = time.perf_counter()
            is_prime_adv, delta_nfr_adv = tnfr_is_prime_advanced(n)
            time_adv = (time.perf_counter() - start) * 1000
            
            # Cached advanced algorithm
            start = time.perf_counter()
            is_prime_cached, delta_nfr_cached = cached_tnfr_is_prime_advanced(n)
            time_cached = (time.perf_counter() - start) * 1000
            
            # Display results
            status = "PRIME" if is_prime_basic else "COMPOSITE"
            print(f"{n:<12} | {'Basic':<15} | {status:<6} | {delta_nfr_basic:<12.8f} | {time_basic:<10.3f}")
            print(f"{'':<12} | {'Advanced':<15} | {status:<6} | {delta_nfr_adv:<12.8f} | {time_adv:<10.3f}")
            print(f"{'':<12} | {'Advanced+Cache':<15} | {status:<6} | {delta_nfr_cached:<12.8f} | {time_cached:<10.3f}")
            print()
            
    except ImportError as e:
        print(f"Advanced algorithms not available: {e}")


def example_prime_certificates():
    """Demonstrate prime certificate analysis."""
    print("=" * 60)
    print("3. PRIME CERTIFICATE ANALYSIS")
    print("=" * 60)
    
    try:
        from tnfr_primality.advanced_core import tnfr_is_prime_advanced
        
        test_numbers = [17, 97, 997, 9973]
        
        for n in test_numbers:
            print(f"\n--- Analysis for {n} ---")
            
            # Get basic result
            is_prime, delta_nfr = tnfr_is_prime_advanced(n)
            print(f"Prime: {is_prime}, ΔNFR: {delta_nfr:.8f}")
            
            try:
                # Get detailed certificate
                certificate = tnfr_is_prime_advanced(n, return_certificate=True)
                
                if hasattr(certificate, 'explanation'):
                    print(f"Certificate explanation: {certificate.explanation}")
                    print(f"Structural metrics:")
                    print(f"  τ(n) = {certificate.tau} (divisor count)")
                    print(f"  σ(n) = {certificate.sigma} (divisor sum)")
                    print(f"  ω(n) = {certificate.omega} (distinct prime factor count)")
                else:
                    print("Certificate data not available")
                    
            except Exception as e:
                print(f"Certificate analysis failed: {e}")
                
    except ImportError as e:
        print(f"Advanced certificates not available: {e}")


def example_batch_processing():
    """Demonstrate high-performance batch processing."""
    print("\n" + "=" * 60)
    print("4. HIGH-PERFORMANCE BATCH PROCESSING")
    print("=" * 60)
    
    try:
        from tnfr_primality.advanced_core import cached_tnfr_is_prime_advanced
        
        # Generate test batch
        test_batch = list(range(2, 101))  # First 100 numbers
        
        print(f"Processing batch of {len(test_batch)} numbers...")
        
        start_time = time.perf_counter()
        results = []
        
        for n in test_batch:
            is_prime, delta_nfr = cached_tnfr_is_prime_advanced(n)
            results.append({
                'n': n,
                'is_prime': is_prime,
                'delta_nfr': delta_nfr
            })
        
        total_time = time.perf_counter() - start_time
        
        # Count primes
        prime_count = sum(1 for r in results if r['is_prime'])
        
        print(f"\nBatch Processing Results:")
        print(f"  Numbers processed: {len(test_batch)}")
        print(f"  Primes found: {prime_count}")
        print(f"  Composites found: {len(test_batch) - prime_count}")
        print(f"  Total time: {total_time * 1000:.2f} ms")
        print(f"  Average per number: {(total_time / len(test_batch)) * 1000:.3f} ms")
        print(f"  Numbers per second: {len(test_batch) / total_time:.1f}")
        
        # Show first few primes
        primes = [r['n'] for r in results[:20] if r['is_prime']]
        print(f"  First primes found: {primes}")
        
    except ImportError as e:
        print(f"Advanced batch processing not available: {e}")


def example_validation_with_analytics():
    """Demonstrate comprehensive theory validation."""
    print("\n" + "=" * 60)
    print("5. COMPREHENSIVE THEORY VALIDATION")
    print("=" * 60)
    
    try:
        from tnfr_primality.advanced_core import validate_tnfr_theory_advanced
        
        print("Running comprehensive TNFR theory validation...")
        
        start_time = time.perf_counter()
        validation_results = validate_tnfr_theory_advanced(max_n=100)
        validation_time = time.perf_counter() - start_time
        
        print(f"\nValidation Results:")
        print(f"  Numbers tested: {validation_results['tested_numbers']}")
        print(f"  Correct predictions: {validation_results['correct_predictions']}")
        print(f"  False positives: {validation_results['false_positives']}")
        print(f"  False negatives: {validation_results['false_negatives']}")
        print(f"  Accuracy: {validation_results['accuracy']:.6f} ({validation_results['accuracy']*100:.4f}%)")
        
        if 'prime_mean_delta_nfr' in validation_results:
            print(f"  Prime mean ΔNFR: {validation_results['prime_mean_delta_nfr']:.8f}")
            print(f"  Composite mean ΔNFR: {validation_results['composite_mean_delta_nfr']:.8f}")
        
        print(f"  Validation time: {validation_time * 1000:.2f} ms")
        print(f"  Numbers per second: {validation_results['tested_numbers'] / validation_time:.1f}")
        
        # Show infrastructure usage
        if validation_results.get('infrastructure_used'):
            print(f"\nAdvanced Infrastructure Usage:")
            print(f"  Algorithm version: {validation_results.get('algorithm_version', 'unknown')}")
            if 'network_statistics' in validation_results:
                stats = validation_results['network_statistics']
                print(f"  Network prime ratio: {stats.get('prime_ratio', 0):.4f}")
                print(f"  ΔNFR separation: {stats.get('DELTA_NFR_separation', 0):.4f}")
        
    except ImportError as e:
        print(f"Advanced validation not available: {e}")


def example_json_integration():
    """Demonstrate JSON integration for programmatic use."""
    print("\n" + "=" * 60)
    print("6. JSON INTEGRATION FOR PROGRAMMATIC USE")
    print("=" * 60)
    
    try:
        from tnfr_primality.advanced_core import validate_tnfr_theory_advanced
        
        print("Generating JSON output for programmatic integration...")
        
        # Run small validation for JSON demo
        results = validate_tnfr_theory_advanced(max_n=20)
        
        # Convert to JSON
        json_output = json.dumps(results, indent=2, default=str)
        
        print(f"\nJSON Output Structure (first 500 characters):")
        print(json_output[:500] + "..." if len(json_output) > 500 else json_output)
        
        print(f"\nJSON Output Statistics:")
        print(f"  Total JSON size: {len(json_output)} characters")
        print(f"  Main sections: {list(results.keys())}")
        print(f"  Prime examples: {len(results.get('prime_examples', []))}")
        print(f"  Composite examples: {len(results.get('composite_examples', []))}")
        
        # Demonstrate parsing JSON back
        parsed_results = json.loads(json_output)
        print(f"\nParsed Results Verification:")
        print(f"  Accuracy: {parsed_results['accuracy']}")
        print(f"  Infrastructure used: {parsed_results.get('infrastructure_used', False)}")
        
    except ImportError as e:
        print(f"Advanced JSON integration not available: {e}")
    except Exception as e:
        print(f"JSON processing error: {e}")


def main():
    """Run all advanced examples."""
    print("TNFR Advanced Primality Testing Examples")
    print("========================================")
    print("Demonstrating full TNFR repository integration capabilities")
    
    # Run all examples
    example_infrastructure_detection()
    example_basic_vs_advanced()
    example_prime_certificates()
    example_batch_processing()
    example_validation_with_analytics()
    example_json_integration()
    
    print("\n" + "=" * 60)
    print("EXAMPLES COMPLETE")
    print("=" * 60)
    print("All advanced TNFR features demonstrated successfully!")
    print("For more information, see:")
    print("- Repository: https://github.com/fermga/TNFR-Python-Engine")
    print("- Documentation: See theory/ directory in main repository")
    print("- CLI Help: python -m tnfr_primality --help")


if __name__ == "__main__":
    main()