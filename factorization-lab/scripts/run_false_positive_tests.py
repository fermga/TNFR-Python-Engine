"""
TNFR False-Positive Test Runner

Standalone runner for false-positive verifier tests with
comprehensive reporting and validation.
"""

import sys
from pathlib import Path

# Setup paths
LAB_PATH = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_PATH))

from tests.test_false_positive_verifier import run_false_positive_tests


def main():
    """Run false-positive verifier tests with detailed reporting."""

    print("TNFR FACTORIZATION LAB")
    print("False-Positive Verifier Test Suite")
    print("=" * 60)
    print()
    print("Testing verifier resistance to non-factor periodic patterns...")
    print("This comprehensive suite validates robustness against:")
    print("• Divisors of actual factors")
    print("• Close-to-factor candidates (±1, ±2)")
    print("• Harmonic multiples and submultiples")
    print("• Carmichael number partial products")
    print("• Prime-like deceptive patterns")
    print("• Fibonacci and sequence-based spurious patterns")
    print()

    success = run_false_positive_tests()

    print()
    print("=" * 60)
    if success:
        print("✅ FALSE-POSITIVE VERIFIER TESTS PASSED")
        print("The TNFR verifier demonstrates robust false-positive resistance.")
    else:
        print("❌ FALSE-POSITIVE VERIFIER TESTS FAILED")
        print("Review test output for specific failure details.")

    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
