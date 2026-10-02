#!/usr/bin/env python3
"""
Installation test script for TNFR Primality Testing package.

This script validates that the package can be installed and used correctly
in a fresh environment.
"""
import sys
import time
import traceback


def _check_import():
    """Test that all modules can be imported."""
    print("Testing imports...")

    try:
        from tnfr_primality import OptimizedTNFRPrimality, tnfr_delta_nfr, tnfr_is_prime

        if not all(
            callable(api)
            for api in (OptimizedTNFRPrimality, tnfr_delta_nfr, tnfr_is_prime)
        ):
            print("[FAIL] Main API exports must be callable")
            return False
        print("[OK] Main imports successful")
    except ImportError as e:
        print(f"[FAIL] Import failed: {e}")
        return False

    try:
        from tnfr_primality.core import validate_tnfr_theory

        if not callable(validate_tnfr_theory):
            print("[FAIL] Arithmetic validation export must be callable")
            return False
        print("[OK] Core module imports successful")
    except ImportError as e:
        print(f"[FAIL] Core import failed: {e}")
        return False

    return True


def _check_basic_functionality():
    """Test basic primality testing functionality."""
    print("\nTesting basic functionality...")

    try:
        from tnfr_primality import tnfr_is_prime

        # Test known primes
        test_cases = [
            (2, True),
            (3, True),
            (5, True),
            (17, True),
            (97, True),
            (4, False),
            (6, False),
            (15, False),
            (25, False),
        ]

        for n, expected in test_cases:
            is_prime, delta_nfr = tnfr_is_prime(n)
            if is_prime != expected:
                print(f"[FAIL] Failed for n={n}: expected {expected}, got {is_prime}")
                return False

            # Verify DeltaNFR constraint
            if expected and abs(delta_nfr) > 1e-10:
                print(f"[FAIL] Prime {n} has non-zero DeltaNFR: {delta_nfr}")
                return False
            elif not expected and delta_nfr <= 0:
                print(f"[FAIL] Composite {n} has non-positive DeltaNFR: {delta_nfr}")
                return False

        print("[OK] Basic functionality tests passed")
        return True

    except Exception as e:
        print(f"[FAIL] Basic functionality test failed: {e}")
        traceback.print_exc()
        return False


def _check_optimized_functionality():
    """Test optimized implementation."""
    print("\nTesting optimized functionality...")

    try:
        from tnfr_primality import OptimizedTNFRPrimality

        optimizer = OptimizedTNFRPrimality()

        # Test individual calls
        is_prime, delta_nfr = optimizer.is_prime(97)
        if not is_prime or abs(delta_nfr) > 1e-10:
            print(f"[FAIL] Optimized test failed for 97: {is_prime}, {delta_nfr}")
            return False

        # Test batch processing
        numbers = [2, 3, 5, 7, 11, 13, 17, 19, 23]
        results = optimizer.batch_test(numbers)

        if len(results) != len(numbers):
            print(
                f"[FAIL] Batch test returned wrong count: {len(results)} vs {len(numbers)}"
            )
            return False

        for (n, is_prime, delta_nfr), expected_n in zip(results, numbers):
            if n != expected_n:
                print(f"[FAIL] Batch test number mismatch: {n} vs {expected_n}")
                return False
            if not is_prime:
                print(f"[FAIL] Batch test failed to identify prime {n}")
                return False

        # Test statistics
        stats = optimizer.get_statistics()
        if "tests_performed" not in stats:
            print("[FAIL] Statistics missing required fields")
            return False

        print("[OK] Optimized functionality tests passed")
        return True

    except Exception as e:
        print(f"[FAIL] Optimized functionality test failed: {e}")
        traceback.print_exc()
        return False


def _check_performance():
    """Test performance characteristics."""
    print("\nTesting performance...")

    try:
        from tnfr_primality import tnfr_is_prime

        # Test with larger numbers
        large_primes = [982451653, 2147483647]

        for n in large_primes:
            start = time.perf_counter()
            is_prime, delta_nfr = tnfr_is_prime(n)
            elapsed = time.perf_counter() - start

            if not is_prime:
                print(f"[FAIL] Failed to identify large prime {n}")
                return False

            if elapsed > 0.1:  # Should be much faster than 100ms
                print(f"[WARN]  Slow performance for {n}: {elapsed*1000:.2f} ms")
            else:
                print(f"[OK] Good performance for {n}: {elapsed*1000:.2f} ms")

        print("[OK] Performance tests passed")
        return True

    except Exception as e:
        print(f"[FAIL] Performance test failed: {e}")
        traceback.print_exc()
        return False


def _check_validation():
    """Test arithmetic validation."""
    print("\nTesting arithmetic validation...")

    try:
        from tnfr_primality.core import validate_tnfr_theory

        # Quick validation test
        results = validate_tnfr_theory(100)

        if results["accuracy"] != 1.0:
            print(f"[FAIL] Validation failed: accuracy = {results['accuracy']}")
            return False

        if results["false_positives"] > 0 or results["false_negatives"] > 0:
            print(
                f"[FAIL] Validation errors: FP={results['false_positives']}, FN={results['false_negatives']}"
            )
            return False

        print(f"[OK] Validation passed: {results['tested']} numbers, 100% accuracy")
        return True

    except Exception as e:
        print(f"[FAIL] Validation test failed: {e}")
        traceback.print_exc()
        return False


def test_import():
    """Expose import verification to pytest."""
    assert _check_import()


def test_basic_functionality():
    """Expose basic functional verification to pytest."""
    assert _check_basic_functionality()


def test_optimized_functionality():
    """Expose optimized functional verification to pytest."""
    assert _check_optimized_functionality()


def test_performance():
    """Expose the non-binding performance smoke check to pytest."""
    assert _check_performance()


def test_validation():
    """Expose arithmetic validation to pytest."""
    assert _check_validation()


def main():
    """Run all installation tests."""
    print("TNFR Primality Testing - Installation Verification")
    print("=" * 52)
    print(f"Python version: {sys.version}")
    print(f"Platform: {sys.platform}")
    print()

    tests = [
        ("Import functionality", _check_import),
        ("Basic functionality", _check_basic_functionality),
        ("Optimized functionality", _check_optimized_functionality),
        ("Performance characteristics", _check_performance),
        ("Arithmetic validation", _check_validation),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"Running {test_name}...")
        try:
            if test_func():
                passed += 1
            else:
                print(f"[FAIL] {test_name} FAILED")
        except Exception as e:
            print(f"[FAIL] {test_name} CRASHED: {e}")
        print()

    print("=" * 52)
    print("INSTALLATION TEST SUMMARY")
    print("=" * 52)
    print(f"Tests passed: {passed}/{total}")

    if passed == total:
        print("[OK] ALL TESTS PASSED - Installation successful!")
        print("[INFO] TNFR Primality Testing package is ready for use")
        return 0
    else:
        print("[FAIL] SOME TESTS FAILED - Installation needs attention")
        return 1


if __name__ == "__main__":
    sys.exit(main())
