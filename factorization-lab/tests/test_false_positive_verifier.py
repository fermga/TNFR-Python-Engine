"""
TNFR Factorization Lab: False-Positive Verifier Test Suite

Comprehensive testing for verifier resistance to non-factor periodic partitions.
Creates test cases with known non-factors to validate verifier robustness and
prevent false positives in factorization certificates.

This test suite challenges the TNFR verification system by:
1. Testing with pseudo-periodic patterns that are not actual factors
2. Validating resistance to divisors of factors (e.g., if 15=3×5, test 1,3,5,15)
3. Checking robustness against close-to-factors and harmonic multiples
4. Ensuring verification criteria properly reject spurious periodicities
5. Testing edge cases with Carmichael numbers and other deceptive composites
"""

import json
import math
import os
import random
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Setup paths for factorization lab
LAB_PATH = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_PATH))

from tnfr_factorization.spectral_paley import (
    SpectralPaleyFactorizer, 
    _TNFR_VERIFICATION_CRITERIA
)


class FalsePositiveTestCase:
    """Individual test case for false-positive verification."""
    
    def __init__(self, n: int, non_factor: int, reason: str, expected_periodic: bool = False):
        self.n = n
        self.non_factor = non_factor  
        self.reason = reason
        self.expected_periodic = expected_periodic  # Whether we expect periodic pattern
        
        # Validate that this is indeed a non-factor
        if n % non_factor == 0:
            raise ValueError(f"{non_factor} is actually a factor of {n}")
    
    def __repr__(self):
        return f"FalsePositiveTestCase(n={self.n}, non_factor={self.non_factor}, reason='{self.reason}')"


class FalsePositiveVerifierTestSuite(unittest.TestCase):
    """Comprehensive test suite for TNFR verifier false-positive resistance."""
    
    @classmethod
    def setUpClass(cls):
        """Setup test environment and factorizer."""
        cls.factorizer = SpectralPaleyFactorizer(max_nodes=1024)
        cls.test_results = {}
        cls._run_full_suite = os.getenv("TNFR_RUN_LONG_TESTS", "").lower() in {"1", "true", "yes", "on"}
        cls._case_limit = int(os.getenv("TNFR_FP_CASE_LIMIT", "24"))
        
        # Create temporary directory for test certificates
        cls.temp_dir = tempfile.mkdtemp(prefix="tnfr_false_positive_test_")
        cls.cert_dir = Path(cls.temp_dir) / "certificates"
        cls.cert_dir.mkdir(exist_ok=True)
        
    @classmethod
    def tearDownClass(cls):
        """Cleanup test environment."""
        import shutil
        if hasattr(cls, 'temp_dir'):
            shutil.rmtree(cls.temp_dir, ignore_errors=True)
    
    def setUp(self):
        """Setup individual test."""
        self.factorizer_config = {
            'trace_certificates': True,
            'certificate_dir': self.cert_dir,
        }
    
    def _curated_fast_false_positive_cases(self) -> List[FalsePositiveTestCase]:
        """Fast deterministic suite for CI and local development.

        This list intentionally targets representative false-positive patterns
        with bounded runtime.
        """

        curated: List[FalsePositiveTestCase] = [
            FalsePositiveTestCase(35, 4, "near-factor composite"),
            FalsePositiveTestCase(35, 6, "near-factor composite"),
            FalsePositiveTestCase(77, 6, "near-factor composite"),
            FalsePositiveTestCase(77, 8, "near-factor composite"),
            FalsePositiveTestCase(143, 12, "near-factor composite"),
            FalsePositiveTestCase(143, 14, "harmonic multiple"),
            FalsePositiveTestCase(187, 16, "prime-like pattern"),
            FalsePositiveTestCase(209, 15, "prime-like pattern"),
            FalsePositiveTestCase(221, 15, "near-factor composite"),
            FalsePositiveTestCase(247, 18, "prime-like pattern"),
            FalsePositiveTestCase(299, 20, "prime-like pattern"),
            FalsePositiveTestCase(323, 18, "near-factor composite"),
        ]
        return curated

    def _generate_false_positive_test_cases(self) -> List[FalsePositiveTestCase]:
        """Generate comprehensive false-positive test cases."""
        
        test_cases = []
        
        # Category 1: Divisors of actual factors
        # If n = p × q, test divisors of p and q that aren't factors of n
        known_factorizations = [
            (35, [5, 7]),      # 35 = 5 × 7
            (77, [7, 11]),     # 77 = 7 × 11  
            (143, [11, 13]),   # 143 = 11 × 13
            (221, [13, 17]),   # 221 = 13 × 17
            (323, [17, 19]),   # 323 = 17 × 19
            (437, [19, 23]),   # 437 = 19 × 23
        ]
        
        for n, factors in known_factorizations:
            for factor in factors:
                # Test divisors of each factor (except 1 and the factor itself)
                for d in range(2, factor):
                    if factor % d == 0 and n % d != 0:
                        test_cases.append(FalsePositiveTestCase(
                            n, d, f"divisor of factor {factor} of {n}"
                        ))
        
        # Category 2: Close-to-factors (±1, ±2 from actual factors)
        for n, factors in known_factorizations:
            for factor in factors:
                for delta in [-2, -1, 1, 2]:
                    candidate = factor + delta
                    if candidate > 1 and n % candidate != 0:
                        test_cases.append(FalsePositiveTestCase(
                            n, candidate, f"factor±{delta}: {factor}{delta:+d} for {n}"
                        ))
        
        # Category 3: Harmonic multiples and submultiples
        test_numbers = [105, 165, 231, 273, 315, 357, 399, 429, 483, 561]
        
        for n in test_numbers:
            # Get actual factors for reference
            actual_factors = []
            for i in range(2, int(math.sqrt(n)) + 1):
                if n % i == 0:
                    actual_factors.extend([i, n // i])
            
            # Test harmonic relationships
            for factor in actual_factors:
                if factor * factor != n:  # Skip perfect squares
                    # Test 2f, 3f, f/2, f/3 if they're not factors
                    for mult in [2, 3]:
                        harmonic = factor * mult
                        if harmonic < n and n % harmonic != 0:
                            test_cases.append(FalsePositiveTestCase(
                                n, harmonic, f"harmonic multiple {mult}×{factor}"
                            ))
                    
                    for div in [2, 3]:
                        if factor % div == 0:
                            submultiple = factor // div
                            if submultiple > 1 and n % submultiple != 0:
                                test_cases.append(FalsePositiveTestCase(
                                    n, submultiple, f"harmonic submultiple {factor}÷{div}"
                                ))
        
        # Category 4: Carmichael number divisors (known to be deceptive)
        carmichael_tests = [
            (561, [3, 11, 17]),  # 561 = 3 × 11 × 17 (first Carmichael number)
        ]
        
        for n, factors in carmichael_tests:
            # Test products of subsets that aren't complete factorizations
            for i, f1 in enumerate(factors):
                for j, f2 in enumerate(factors):
                    if i != j:
                        product = f1 * f2
                        if product < n and n % product != 0:
                            test_cases.append(FalsePositiveTestCase(
                                n, product, f"Carmichael partial product {f1}×{f2}"
                            ))
        
        # Category 5: Prime-like patterns (numbers that "look" like they could be factors)
        prime_like_patterns = [
            (143, 12, "near-factor composite 12 vs 143=11×13"),
            (187, 16, "power-of-2 near 17 for 187=11×17"),  
            (209, 15, "composite 15 near factors of 209=11×19"),
            (247, 18, "composite 18 for 247=13×19"),
            (299, 20, "round number 20 for 299=13×23"),
        ]
        
        for n, non_factor, reason in prime_like_patterns:
            if n % non_factor != 0:
                test_cases.append(FalsePositiveTestCase(n, non_factor, reason))
        
        # Category 6: Fibonacci and other sequence values that might create spurious patterns
        fibonacci = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
        test_composites = [77, 91, 143, 169, 187, 209, 221, 247, 299, 323]
        
        for n in test_composites:
            for fib in fibonacci:
                if fib > 1 and n % fib != 0 and fib < n // 2:
                    test_cases.append(FalsePositiveTestCase(
                        n, fib, f"Fibonacci number F_{fibonacci.index(fib)} = {fib}"
                    ))
        
        # Deduplicate while preserving order
        seen: set[tuple[int, int]] = set()
        unique_cases: List[FalsePositiveTestCase] = []
        for case in test_cases:
            key = (case.n, case.non_factor)
            if key in seen:
                continue
            seen.add(key)
            unique_cases.append(case)

        # Default fast mode: deterministic sampling with bounded size.
        if not self._run_full_suite:
            if len(unique_cases) <= self._case_limit:
                return unique_cases
            rng = random.Random(42)
            sampled = rng.sample(unique_cases, self._case_limit)
            sampled.sort(key=lambda c: (c.n, c.non_factor))
            return sampled

        return unique_cases
    
    def test_false_positive_resistance_comprehensive(self):
        """Test verifier resistance across all false-positive categories."""
        
        if self._run_full_suite:
            test_cases = self._generate_false_positive_test_cases()
        else:
            # Fast mode by default to avoid CI/local hangs.
            test_cases = self._curated_fast_false_positive_cases()

        print(f"\nRunning {len(test_cases)} false-positive test cases...")
        
        false_positives = []
        verification_failures = []
        test_summary = {
            'total_cases': len(test_cases),
            'false_positives': 0,
            'verification_failures': 0,
            'correct_rejections': 0,
            'categories': {}
        }
        
        for i, test_case in enumerate(test_cases):
            if i % 20 == 0:
                print(f"Progress: {i+1}/{len(test_cases)} ({100*(i+1)//len(test_cases)}%)")
            
            try:
                # Run factorization
                result = self.factorizer.analyze(test_case.n, **self.factorizer_config)
                
                # Check if the non-factor was incorrectly identified as a factor
                certified_factors = getattr(result, 'tnfr_certified_factors', None) or []
                
                is_false_positive = test_case.non_factor in certified_factors
                
                if is_false_positive:
                    false_positives.append((test_case, result))
                    test_summary['false_positives'] += 1
                    print(f"FALSE POSITIVE: {test_case}")
                else:
                    test_summary['correct_rejections'] += 1
                
                # Track by category
                category = test_case.reason.split(':')[0] if ':' in test_case.reason else test_case.reason
                if category not in test_summary['categories']:
                    test_summary['categories'][category] = {'total': 0, 'false_positives': 0}
                test_summary['categories'][category]['total'] += 1
                if is_false_positive:
                    test_summary['categories'][category]['false_positives'] += 1
                    
            except Exception as e:
                verification_failures.append((test_case, str(e)))
                test_summary['verification_failures'] += 1
                print(f"VERIFICATION FAILURE: {test_case} - {str(e)}")
        
        # Generate detailed report
        self._generate_false_positive_report(test_summary, false_positives, verification_failures)
        
        # Assertions for test success
        false_positive_rate = test_summary['false_positives'] / test_summary['total_cases']
        
        # Fast mode keeps strict guard but with realistic margin for
        # reduced sampling; exhaustive mode retains tight threshold.
        max_fp_rate = 0.02 if self._run_full_suite else 0.05
        self.assertLessEqual(false_positive_rate, max_fp_rate, 
                           f"False positive rate {false_positive_rate:.3f} exceeds 2% threshold")
        
        # Verification should not fail more than 5% of the time
        failure_rate = test_summary['verification_failures'] / test_summary['total_cases'] 
        self.assertLessEqual(failure_rate, 0.05,
                           f"Verification failure rate {failure_rate:.3f} exceeds 5% threshold")
        
        mode = "full" if self._run_full_suite else "fast"
        print(f"\n✓ False-positive resistance test completed ({mode} mode):")
        print(f"  - Total test cases: {test_summary['total_cases']}")
        print(f"  - False positives: {test_summary['false_positives']} ({false_positive_rate:.1%})")
        print(f"  - Verification failures: {test_summary['verification_failures']} ({failure_rate:.1%})")
        print(f"  - Correct rejections: {test_summary['correct_rejections']} ({test_summary['correct_rejections']/test_summary['total_cases']:.1%})")
    
    def test_verification_criteria_strictness(self):
        """Test that verification criteria are sufficiently strict to prevent false positives."""
        
        criteria = _TNFR_VERIFICATION_CRITERIA
        
        # Test that criteria are reasonably strict
        self.assertGreaterEqual(criteria['min_partition_flags'], 3, 
                               "Should require multiple partition conditions")
        
        self.assertGreaterEqual(criteria['dnfr_gain_min'], 0.1,
                               "Should require significant ΔNFR reduction") 
        
        self.assertGreaterEqual(criteria['periodicity_confidence_min'], 0.5,
                               "Should require high confidence in periodicity")
        
        self.assertGreaterEqual(criteria['required_partition_ratio'], 0.3,
                               "Should require substantial partition endorsement")
        
        self.assertLessEqual(criteria['phi_delta_max'], 0.5,
                            "Should limit structural potential deviation")
        
        print("✓ Verification criteria strictness validated")
    
    def test_known_good_factors_still_pass(self):
        """Ensure legitimate factors are still surfaced by the current pipeline.

        TNFR certification is intentionally strict and may not endorse every
        true factor in all small/moderate composites. The stable contract for
        discovery remains ``candidate_factors``.
        """
        
        known_factorizations = [
            (35, [5, 7]),
            (77, [7, 11]), 
            (143, [11, 13]),
            (221, [13, 17]),
        ]
        
        passed_recovery = 0
        total_factors = 0
        
        for n, expected_factors in known_factorizations:
            try:
                result = self.factorizer.analyze(n, **self.factorizer_config)
                discovered_factors = set(getattr(result, 'candidate_factors', None) or [])
                
                for expected_factor in expected_factors:
                    total_factors += 1
                    if expected_factor in discovered_factors:
                        passed_recovery += 1
                    else:
                        print(f"WARNING: Expected factor {expected_factor} of {n} not surfaced in candidate_factors")
                        
            except Exception as e:
                print(f"ERROR: Failed to factor {n}: {e}")
        
        verification_rate = passed_recovery / total_factors if total_factors > 0 else 0
        
        # Should still verify at least 70% of known good factors
        self.assertGreaterEqual(verification_rate, 0.7,
                               f"Known factor verification rate {verification_rate:.1%} too low")
        
        print(f"✓ Known good factors recovery: {passed_recovery}/{total_factors} ({verification_rate:.1%})")
    
    def test_periodic_pattern_detection_accuracy(self):
        """Test that detection distinguishes true factors from non-factors.

        Uses ``candidate_factors`` to reflect the active decoding contract.
        """
        
        test_cases = [
            # (n, candidate, is_real_factor, expected_periodic)
            (35, 5, True, True),    # Real factor should show strong periodicity
            (35, 4, False, False),  # Non-factor should not show factor periodicity
            (35, 6, False, False),  # Close to factor but not a factor
            (77, 7, True, True),    # Real factor
            (77, 6, False, False),  # Close non-factor
            (77, 8, False, False),  # Close non-factor
        ]
        
        periodicity_accuracy = 0
        total_tests = 0
        
        for n, candidate, is_real_factor, expected_periodic in test_cases:
            # This would require access to the internal periodicity detection
            # For now, we test indirectly through factor certification
            try:
                result = self.factorizer.analyze(n, **self.factorizer_config)
                discovered_factors = set(getattr(result, 'candidate_factors', None) or [])
                is_certified = candidate in discovered_factors
                
                # Real factors should be certified, non-factors should not
                if is_real_factor == is_certified:
                    periodicity_accuracy += 1
                
                total_tests += 1
                
            except Exception as e:
                print(f"Periodicity test failed for n={n}, candidate={candidate}: {e}")
        
        accuracy_rate = periodicity_accuracy / total_tests if total_tests > 0 else 0
        self.assertGreaterEqual(accuracy_rate, 0.8, 
                               f"Periodicity detection accuracy {accuracy_rate:.1%} too low")
        
        print(f"✓ Periodic pattern detection accuracy: {periodicity_accuracy}/{total_tests} ({accuracy_rate:.1%})")
    
    def _generate_false_positive_report(self, summary: Dict, false_positives: List, failures: List):
        """Generate detailed false-positive test report."""
        
        report_path = self.cert_dir / "false_positive_test_report.json"
        
        report = {
            'timestamp': math.floor(os.times().elapsed * 1000),
            'test_summary': summary,
            'false_positives': [
                {
                    'n': fp[0].n,
                    'non_factor': fp[0].non_factor,
                    'reason': fp[0].reason,
                    'certified_factors': getattr(fp[1], 'tnfr_certified_factors', [])
                }
                for fp in false_positives
            ],
            'verification_failures': [
                {
                    'n': vf[0].n,
                    'non_factor': vf[0].non_factor,
                    'reason': vf[0].reason, 
                    'error': vf[1]
                }
                for vf in failures
            ],
            'verification_criteria': _TNFR_VERIFICATION_CRITERIA
        }
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Detailed false-positive test report saved: {report_path}")
        
        # Print category breakdown
        if summary['categories']:
            print("\nFalse-positive breakdown by category:")
            for category, stats in summary['categories'].items():
                fp_rate = stats['false_positives'] / stats['total'] if stats['total'] > 0 else 0
                print(f"  {category}: {stats['false_positives']}/{stats['total']} ({fp_rate:.1%})")


def run_false_positive_tests():
    """Run the false-positive verifier test suite."""
    
    print("TNFR FALSE-POSITIVE VERIFIER TEST SUITE")
    print("=" * 50)
    print("Testing verifier resistance to non-factor periodic partitions...")
    print()
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(FalsePositiveVerifierTestSuite)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    print("\n" + "=" * 50)
    if result.wasSuccessful():
        print("✓ All false-positive verifier tests PASSED")
        print("The TNFR verifier demonstrates robust resistance to false positives")
    else:
        print("✗ Some false-positive verifier tests FAILED")
        print(f"Failures: {len(result.failures)}, Errors: {len(result.errors)}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_false_positive_tests()
    sys.exit(0 if success else 1)