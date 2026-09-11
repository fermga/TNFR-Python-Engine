"""
TNFR False-Positive Test Suite Runner

Comprehensive runner for all false-positive verifier tests.
Combines robustness testing, criteria validation, and comprehensive
false-positive resistance testing.
"""

import sys
from pathlib import Path

# Setup paths
LAB_PATH = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_PATH))

from tests.test_false_positive_verifier import run_false_positive_tests
from tests.test_verification_robustness import run_robustness_tests


def main():
    """Run comprehensive false-positive test suite."""

    print("TNFR FACTORIZATION LAB")
    print("COMPREHENSIVE FALSE-POSITIVE VERIFIER TEST SUITE")
    print("=" * 65)
    print()

    all_passed = True

    # Phase 1: Verification Criteria Robustness
    print("PHASE 1: VERIFICATION CRITERIA ROBUSTNESS")
    print("-" * 45)
    robustness_passed = run_robustness_tests()
    all_passed &= robustness_passed

    print()

    # Phase 2: Comprehensive False-Positive Resistance
    print("PHASE 2: COMPREHENSIVE FALSE-POSITIVE RESISTANCE")
    print("-" * 50)
    false_positive_passed = run_false_positive_tests()
    all_passed &= false_positive_passed

    print()
    print("=" * 65)
    print("COMPREHENSIVE TEST RESULTS:")
    print(
        f"• Verification Criteria Robustness: {'✓ PASS' if robustness_passed else '✗ FAIL'}"
    )
    print(
        f"• False-Positive Resistance: {'✓ PASS' if false_positive_passed else '✗ FAIL'}"
    )
    print()

    if all_passed:
        print("🎉 ALL FALSE-POSITIVE VERIFIER TESTS PASSED")
        print("The TNFR verification system demonstrates robust resistance")
        print("to false positives across all test categories and criteria.")
    else:
        print("❌ SOME FALSE-POSITIVE TESTS FAILED")
        print("Review individual test outputs for failure analysis.")
        print("Consider adjusting verification criteria if needed.")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
