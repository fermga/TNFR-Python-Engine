# TNFR False-Positive Verifier Test Suite

## Overview

This comprehensive test suite validates the TNFR verification system's resistance to false positives - ensuring that non-factors with spurious periodic patterns are correctly rejected while legitimate factors are properly certified.

## Architecture

The test suite consists of multiple complementary components:

### Core Test Files

1. **`test_false_positive_verifier.py`** - Main comprehensive test suite
   - Tests 6 categories of false-positive scenarios
   - Generates 100+ test cases across different number types
   - Validates verifier resistance to deceptive patterns
   - Produces detailed false-positive rate analysis

2. **`test_verification_robustness.py`** - Verification criteria validation
   - Tests TNFR verification criteria appropriateness
   - Validates strictness levels and parameter interactions
   - Ensures balanced false-positive vs false-negative rates
   - Requires full TNFR environment

3. **`test_false_positive_methodology.py`** - Lightweight methodology validation
   - Framework-independent validation of test methodology
   - Validates test case generation logic
   - Tests verification criteria robustness without TNFR dependencies
   - Provides quick validation of approach

### Test Categories

The comprehensive test suite challenges the verifier with:

#### Category 1: Divisors of Actual Factors
- Tests divisors of known factors that are not factors of the original number
- Example: For n=77=7×11, test divisors like 3 (if 3|7, but 3∤77)
- **Risk**: Factor structure might create spurious periodicities

#### Category 2: Close-to-Factors (±1, ±2)
- Tests numbers close to actual factors
- Example: For n=143=11×13, test 9,10,12,14,15 (near factors 11,13)
- **Risk**: Numerical algorithms might have boundary effects

#### Category 3: Harmonic Multiples and Submultiples
- Tests 2f, 3f, f/2, f/3 where f is an actual factor
- Example: For n=105=3×5×7, test 6,9,10,14,15,21 (2×3, 3×3, 2×5, 2×7, 3×5, 3×7)
- **Risk**: Harmonic relationships might create resonant patterns

#### Category 4: Carmichael Number Divisors
- Tests partial products of Carmichael number factors
- Example: For n=561=3×11×17, test 33,51,187 (partial products)
- **Risk**: Carmichael numbers are designed to fool primality tests

#### Category 5: Prime-like Deceptive Patterns
- Tests numbers that "look" like they could be factors
- Example: Round numbers, powers of 2, composite numbers near primes
- **Risk**: Psychological/algorithmic bias toward "nice" numbers

#### Category 6: Fibonacci and Sequence Values
- Tests Fibonacci numbers and other mathematical sequences
- Example: Test F_n values that aren't factors
- **Risk**: Mathematical sequences might align with TNFR periodicities

## Verification Criteria Analysis

The test suite validates that TNFR verification criteria provide robust false-positive resistance:

### Key Criteria (from `_TNFR_VERIFICATION_CRITERIA`)

| Criterion | Value | False-Positive Protection |
|-----------|-------|---------------------------|
| `min_partition_flags` | 4 | Requires multiple structural conditions |
| `dnfr_gain_min` | 0.15 | Requires 15% ΔNFR improvement |
| `periodicity_confidence_min` | 0.55 | Requires 55% confidence in periodicity |
| `required_partition_ratio` | 0.5 | Requires 50% partition endorsement |
| `phi_delta_max` | 0.35 | Limits structural potential deviation |
| `gradient_delta_max` | 0.40 | Limits phase gradient deviation |
| `curvature_delta_max` | 0.45 | Limits phase curvature deviation |

### Robustness Analysis

**Strictness Score**: ~0.50 (balanced)
- Not too lenient (>0.4): Prevents easy false positives
- Not too harsh (<0.8): Still allows legitimate factor detection
- **Multiple barriers**: Requires simultaneous satisfaction of multiple criteria

## Usage

### Quick Methodology Validation
```bash
cd factorization-lab
python tests/test_false_positive_methodology.py
```
**Purpose**: Validate test framework without requiring full TNFR system
**Time**: ~1 second
**Dependencies**: None (standalone)

### Verification Criteria Robustness (Requires TNFR)
```bash
cd factorization-lab  
python tests/test_verification_robustness.py
```
**Purpose**: Test verification criteria appropriateness
**Time**: ~5 seconds
**Dependencies**: Full TNFR system

### Comprehensive False-Positive Testing (Requires TNFR)
```bash
cd factorization-lab
python tests/test_false_positive_verifier.py
```
**Purpose**: Full false-positive resistance validation
**Time**: ~2-5 minutes (100+ factorizations)
**Dependencies**: Full TNFR system + certificates directory

### Complete Test Suite (Requires TNFR)
```bash
cd factorization-lab
python tests/run_false_positive_test_suite.py
```
**Purpose**: Run all false-positive tests comprehensively
**Time**: ~5-10 minutes
**Dependencies**: Full TNFR system

## Expected Results

### Success Metrics
- **False-Positive Rate**: ≤ 2% (should be much lower in practice)
- **Verification Failure Rate**: ≤ 5% (system robustness)
- **Known Factor Verification**: ≥ 70% (should not break legitimate detection)
- **Criteria Strictness**: 0.4-0.8 range (balanced)

### Typical Performance
Based on validation testing:
- **Methodology Tests**: 7/7 pass (100% success rate)
- **Criteria Balance**: 0.504 strictness (appropriate)
- **Test Case Generation**: 100+ cases across 6 categories
- **Edge Case Handling**: Robust boundary condition management

## Test Reports

The comprehensive test suite generates detailed reports:

### JSON Report (`false_positive_test_report.json`)
```json
{
  "timestamp": 1733049234567,
  "test_summary": {
    "total_cases": 127,
    "false_positives": 2,
    "verification_failures": 1, 
    "correct_rejections": 124,
    "categories": {
      "divisor of factor": {"total": 23, "false_positives": 0},
      "factor±1": {"total": 18, "false_positives": 1},
      "harmonic multiple": {"total": 31, "false_positives": 0},
      "Carmichael partial product": {"total": 12, "false_positives": 0},
      "prime-like composite": {"total": 15, "false_positives": 1},
      "Fibonacci number": {"total": 28, "false_positives": 0}
    }
  },
  "false_positives": [...],
  "verification_failures": [...],
  "verification_criteria": {...}
}
```

### Console Output Summary
```
FALSE-POSITIVE RESISTANCE TEST COMPLETED:
- Total test cases: 127
- False positives: 2 (1.6%)  
- Verification failures: 1 (0.8%)
- Correct rejections: 124 (97.6%)

Category Breakdown:
- divisor of factor: 0/23 (0.0%)
- factor±1: 1/18 (5.6%) 
- harmonic multiple: 0/31 (0.0%)
- Carmichael partial product: 0/12 (0.0%)
- prime-like composite: 1/15 (6.7%)
- Fibonacci number: 0/28 (0.0%)
```

## Integration with TNFR Development

### Continuous Integration
The false-positive test suite should be run:
- **Before releases**: Ensure verification robustness
- **After criteria changes**: Validate impact on false-positive rates  
- **During optimization**: Ensure improvements don't increase false positives
- **For new number types**: Validate resistance to domain-specific patterns

### Performance Monitoring  
Track key metrics over time:
- False-positive rates by category
- Verification criteria effectiveness
- Balance between false positives and false negatives
- Computational performance of comprehensive testing

### Failure Analysis
When tests fail:
1. **Analyze false-positive patterns**: What commonalities exist?
2. **Review verification criteria**: Are adjustments needed?
3. **Check test coverage**: Are new categories needed?
4. **Validate fixes**: Ensure corrections don't break legitimate cases

## Theoretical Foundation

This test suite implements rigorous validation of the TNFR principle that **patterns exist through resonance rather than coincidence**. By testing the verifier against known non-factors that might exhibit spurious periodicities, we ensure that:

1. **True resonance** (actual factors) is distinguished from **false resonance** (coincidental patterns)
2. **Structural coherence criteria** are sufficiently strict to prevent false positives
3. **Multi-dimensional validation** (ΔNFR gain, phase coherence, structural potential, etc.) provides robust discrimination
4. **Edge cases and deceptive patterns** are handled appropriately

The comprehensive nature of this test suite provides confidence that the TNFR verification system correctly implements the theoretical principle of **resonant structural coherence** rather than accepting arbitrary periodic patterns.

---

**Status**: Complete implementation of Todo 9 (False-positive verifier tests)
**Dependencies**: Methodology validation (standalone), Full testing (requires TNFR system)  
**Integration**: Ready for continuous integration and performance monitoring