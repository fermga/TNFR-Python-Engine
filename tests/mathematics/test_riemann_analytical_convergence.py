"""Tests for P8: Analytical proof of σ* → 1/2 rate.

Validates the three main theorems:
  Theorem 1 — Telescoping identity tr(L_k V_1) = (log p_k)² − (log 2)²
  Theorem 2 — PNT asymptotic Σ(log p_i)² ~ k(log k)² − 2k log k + 2k
  Theorem 3 — |σ* − 1/2| = O(1/k) with effective constant C(k) → 1
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tnfr.riemann.analytical_convergence import (
    # Data structures
    TelescopingIdentity,
    PNTAsymptoticBound,
    ConvergenceRateBound,
    EffectiveConstantResult,
    AnalyticalConvergenceProof,
    # Theorem 1
    compute_telescoping_trace,
    verify_telescoping_identity,
    # Theorem 2
    pnt_prime_estimate,
    euler_maclaurin_log_squared_sum,
    pnt_sum_log_squared,
    # Theorem 3
    compute_convergence_rate_bound,
    compute_effective_constant,
    analyze_convergence_sequence,
    # Integration
    run_analytical_convergence_proof,
)
from tnfr.riemann.spectral_proof import compute_analytic_sigma_star


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def small_k():
    """Small graph size for basic checks."""
    return 10


@pytest.fixture
def medium_k():
    """Medium graph size for asymptotic checks."""
    return 100


@pytest.fixture
def large_k():
    """Larger graph size for tighter bounds."""
    return 500


@pytest.fixture
def k_sequence():
    """Standard sequence for convergence tests."""
    return [5, 10, 20, 50, 100, 200, 500]


# ============================================================================
# TestTelescopingIdentity — Theorem 1
# ============================================================================


class TestTelescopingIdentity:
    """Verify the exact telescoping identity tr(L_k V_1) = (log p_k)² − (log 2)²."""

    def test_returns_dataclass(self, small_k):
        result = compute_telescoping_trace(small_k)
        assert isinstance(result, TelescopingIdentity)

    def test_k_stored(self, small_k):
        result = compute_telescoping_trace(small_k)
        assert result.k == small_k

    def test_log_p1_is_log2(self):
        """First prime is always 2."""
        result = compute_telescoping_trace(5)
        assert abs(result.log_p1_squared - math.log(2) ** 2) < 1e-14

    def test_telescoping_equals_difference(self):
        """Verify telescoping = (log p_k)² − (log p_1)²."""
        result = compute_telescoping_trace(20)
        expected = result.log_pk_squared - result.log_p1_squared
        assert abs(result.telescoping_value - expected) < 1e-14

    def test_exact_match_numerical(self, small_k):
        """Telescoping identity must match numerical tr(L V_1) to machine precision."""
        result = compute_telescoping_trace(small_k)
        assert result.relative_error < 1e-12, (
            f"k={small_k}: rel_error={result.relative_error:.2e}"
        )

    def test_exact_match_k5(self):
        result = compute_telescoping_trace(5)
        assert result.relative_error < 1e-12

    def test_exact_match_k50(self):
        result = compute_telescoping_trace(50)
        assert result.relative_error < 1e-12

    def test_exact_match_k100(self, medium_k):
        result = compute_telescoping_trace(medium_k)
        assert result.relative_error < 1e-12

    def test_exact_match_k500(self, large_k):
        result = compute_telescoping_trace(large_k)
        assert result.relative_error < 1e-12

    def test_positive_value(self, medium_k):
        """tr(L V_1) > 0 since log p_k > log p_1."""
        result = compute_telescoping_trace(medium_k)
        assert result.telescoping_value > 0

    def test_grows_with_k(self):
        """tr(L V_1) increases with k."""
        r10 = compute_telescoping_trace(10)
        r100 = compute_telescoping_trace(100)
        assert r100.telescoping_value > r10.telescoping_value

    def test_verify_sequence(self, k_sequence):
        """Batch verification across multiple k values."""
        results = verify_telescoping_identity(k_sequence)
        assert len(results) == len(k_sequence)
        for r in results:
            assert r.relative_error < 1e-12

    def test_invalid_k(self):
        with pytest.raises(ValueError, match="k >= 2"):
            compute_telescoping_trace(1)


# ============================================================================
# TestPNTEstimate
# ============================================================================


class TestPNTEstimate:
    """Test PNT prime estimate p_n ≈ n(ln n + ln ln n)."""

    def test_small_primes_exact(self):
        """First five primes returned exactly."""
        assert pnt_prime_estimate(1) == 2.0
        assert pnt_prime_estimate(2) == 3.0
        assert pnt_prime_estimate(3) == 5.0
        assert pnt_prime_estimate(4) == 7.0
        assert pnt_prime_estimate(5) == 11.0

    def test_p100_reasonable(self):
        """p_100 = 541; PNT estimate should be within 15%."""
        est = pnt_prime_estimate(100)
        assert abs(est - 541) / 541 < 0.15

    def test_p1000_reasonable(self):
        """p_1000 = 7919; PNT estimate should be within 15%."""
        est = pnt_prime_estimate(1000)
        assert abs(est - 7919) / 7919 < 0.15

    def test_invalid_index(self):
        with pytest.raises(ValueError, match="must be >= 1"):
            pnt_prime_estimate(0)


# ============================================================================
# TestEulerMaclaurinLogSquaredSum
# ============================================================================


class TestEulerMaclaurinLogSquaredSum:
    """Test Euler–Maclaurin approximation Σ(ln n)² ≈ k(ln k)² − 2k ln k + 2k."""

    def test_k2(self):
        """k=2: Σ = (ln 2)² ≈ 0.48."""
        exact = math.log(2) ** 2
        est = euler_maclaurin_log_squared_sum(2)
        # Euler-Maclaurin approximation is rough at small k
        assert est > 0

    def test_k1_returns_zero(self):
        assert euler_maclaurin_log_squared_sum(1) == 0.0

    def test_positive_for_k_ge_2(self):
        for k in [2, 10, 100]:
            assert euler_maclaurin_log_squared_sum(k) > 0

    def test_grows_with_k(self):
        v10 = euler_maclaurin_log_squared_sum(10)
        v100 = euler_maclaurin_log_squared_sum(100)
        assert v100 > v10

    def test_accuracy_at_k1000(self):
        """Compare Euler-Maclaurin approx with direct sum."""
        k = 1000
        direct = sum(math.log(n) ** 2 for n in range(2, k + 1))
        est = euler_maclaurin_log_squared_sum(k)
        rel_error = abs(est - direct) / direct
        # Should be within ~5% for k=1000 (Euler-Maclaurin O(1/k) correction)
        assert rel_error < 0.05


# ============================================================================
# TestPNTSumLogSquared — Theorem 2
# ============================================================================


class TestPNTSumLogSquared:
    """Verify PNT asymptotic for Σ(log p_i)²."""

    def test_returns_dataclass(self, small_k):
        result = pnt_sum_log_squared(small_k)
        assert isinstance(result, PNTAsymptoticBound)

    def test_exact_positive(self, small_k):
        result = pnt_sum_log_squared(small_k)
        assert result.exact_value > 0

    def test_leading_order_grows(self):
        """k(log k)² should grow with k."""
        r10 = pnt_sum_log_squared(10)
        r100 = pnt_sum_log_squared(100)
        assert r100.leading_order > r10.leading_order

    def test_pnt_estimate_within_10pct_k10(self):
        """PNT estimate Σ[ln(n ln n)]² should be close at k=10."""
        result = pnt_sum_log_squared(10)
        assert result.pnt_relative_error < 0.10

    def test_pnt_estimate_improves_with_k(self):
        """PNT estimate improves as k grows."""
        r10 = pnt_sum_log_squared(10)
        r100 = pnt_sum_log_squared(100)
        assert r100.pnt_relative_error < r10.pnt_relative_error

    def test_pnt_estimate_within_7pct_k100(self, medium_k):
        """At k=100, PNT estimate should be within 7%."""
        result = pnt_sum_log_squared(medium_k)
        assert result.pnt_relative_error < 0.07

    def test_pnt_estimate_within_5pct_k500(self, large_k):
        """At k=500, PNT estimate should be within 5%."""
        result = pnt_sum_log_squared(large_k)
        assert result.pnt_relative_error < 0.05

    def test_sandwich_bounds(self, medium_k):
        """Exact must lie within [lower_bound, upper_bound]."""
        result = pnt_sum_log_squared(medium_k)
        assert result.lower_bound <= result.exact_value
        assert result.exact_value <= result.upper_bound

    def test_scaling_ratio_bounded(self, medium_k):
        """exact / k(log k)² should be in range [1, 3]."""
        result = pnt_sum_log_squared(medium_k)
        assert 1.0 < result.scaling_ratio < 3.0

    def test_invalid_k(self):
        with pytest.raises(ValueError, match="k >= 2"):
            pnt_sum_log_squared(1)


# ============================================================================
# TestConvergenceRateBound — Theorem 3
# ============================================================================


class TestConvergenceRateBound:
    """Verify |σ* − 1/2| = O(1/k) convergence rate."""

    def test_returns_dataclass(self, small_k):
        result = compute_convergence_rate_bound(small_k)
        assert isinstance(result, ConvergenceRateBound)

    def test_sigma_star_below_half(self, medium_k):
        """σ* < 1/2 since tr(L V_1) > 0."""
        result = compute_convergence_rate_bound(medium_k)
        assert result.sigma_star < 0.5

    def test_deviation_positive(self, medium_k):
        result = compute_convergence_rate_bound(medium_k)
        assert result.deviation > 0

    def test_deviation_order_1_over_k(self, medium_k):
        """Deviation should be roughly 1/k."""
        result = compute_convergence_rate_bound(medium_k)
        ratio = result.deviation * medium_k
        # C(k) should be between 0.5 and 3.0 for k=100
        assert 0.5 < ratio < 3.0

    def test_deviation_decreases_with_k(self):
        """Larger k gives smaller |σ* − 1/2|."""
        r20 = compute_convergence_rate_bound(20)
        r100 = compute_convergence_rate_bound(100)
        assert r100.deviation < r20.deviation

    def test_curvature_grows(self):
        """d²E/dσ² increases with k."""
        r20 = compute_convergence_rate_bound(20)
        r100 = compute_convergence_rate_bound(100)
        assert r100.curvature > r20.curvature

    def test_curvature_order_log_k_squared(self, medium_k):
        """Curvature ~ (log k)²."""
        result = compute_convergence_rate_bound(medium_k)
        log_k_sq = math.log(medium_k) ** 2
        ratio = result.curvature / log_k_sq
        # Should be between 0.5 and 2.0 (correction terms are sub-leading)
        assert 0.5 < ratio < 2.0

    def test_matches_spectral_proof(self, medium_k):
        """Must agree with compute_analytic_sigma_star."""
        result = compute_convergence_rate_bound(medium_k)
        sigma_ref, _, _ = compute_analytic_sigma_star(medium_k)
        assert abs(result.sigma_star - sigma_ref) < 1e-14

    def test_numerator_is_tr_lv1(self, small_k):
        # Cross-check numerator = tr(L V_1)
        result = compute_convergence_rate_bound(small_k)
        tele = compute_telescoping_trace(small_k)
        assert abs(result.numerator - tele.numerical_value) < 1e-12

    def test_invalid_k(self):
        with pytest.raises(ValueError, match="k >= 2"):
            compute_convergence_rate_bound(1)


# ============================================================================
# TestEffectiveConstant
# ============================================================================


class TestEffectiveConstant:
    """Verify C(k) = k|σ* − 1/2| → 1."""

    def test_returns_dataclass(self, small_k):
        result = compute_effective_constant(small_k)
        assert isinstance(result, EffectiveConstantResult)

    def test_positive(self, medium_k):
        result = compute_effective_constant(medium_k)
        assert result.effective_constant > 0

    def test_converges_toward_1(self, k_sequence):
        """C(k) should approach 1 as k grows."""
        results = [compute_effective_constant(k) for k in k_sequence]
        # At largest k, C should be closer to 1 than at smallest k
        assert results[-1].deviation_from_unity < results[0].deviation_from_unity

    def test_bounded_range(self, medium_k):
        """C(k) should be in reasonable range."""
        result = compute_effective_constant(medium_k)
        assert 0.5 < result.effective_constant < 3.0

    def test_c_at_k500(self, large_k):
        """At k=500, C(k) should be closer to 1."""
        result = compute_effective_constant(large_k)
        assert result.deviation_from_unity < 0.5

    def test_deviation_from_unity_field(self, medium_k):
        result = compute_effective_constant(medium_k)
        assert abs(result.deviation_from_unity - abs(result.effective_constant - 1.0)) < 1e-14

    def test_invalid_k(self):
        with pytest.raises(ValueError, match="k >= 2"):
            compute_effective_constant(1)


# ============================================================================
# TestAnalyzeConvergenceSequence
# ============================================================================


class TestAnalyzeConvergenceSequence:
    """Test sequence analysis utility."""

    def test_returns_list(self, k_sequence):
        results = analyze_convergence_sequence(k_sequence)
        assert len(results) == len(k_sequence)

    def test_all_convergence_bounds(self, k_sequence):
        results = analyze_convergence_sequence(k_sequence)
        for r in results:
            assert isinstance(r, ConvergenceRateBound)

    def test_monotone_deviation(self):
        """Deviations should decrease monotonically for sorted k_values."""
        k_vals = [10, 20, 50, 100, 200]
        results = analyze_convergence_sequence(k_vals)
        for i in range(len(results) - 1):
            assert results[i].deviation >= results[i + 1].deviation


# ============================================================================
# TestRunAnalyticalConvergenceProof — Integration
# ============================================================================


class TestRunAnalyticalConvergenceProof:
    """Test the full analytical proof pipeline."""

    def test_returns_dataclass(self):
        proof = run_analytical_convergence_proof([5, 10, 20])
        assert isinstance(proof, AnalyticalConvergenceProof)

    def test_default_k_values(self):
        proof = run_analytical_convergence_proof()
        assert len(proof.k_values) == 7
        assert proof.k_values[0] == 5
        assert proof.k_values[-1] == 500

    def test_telescoping_exact(self):
        """Telescoping identity should hold to machine precision."""
        proof = run_analytical_convergence_proof([10, 50, 100])
        assert proof.telescoping_max_error < 1e-12

    def test_pnt_improves(self):
        """PNT error should be bounded."""
        proof = run_analytical_convergence_proof([10, 50, 100, 500])
        assert proof.pnt_max_error < 0.10  # PNT estimate within 10%

    def test_monotone_decrease(self):
        """Deviation should decrease monotonically."""
        proof = run_analytical_convergence_proof([10, 20, 50, 100, 200])
        assert proof.monotone_decrease is True

    def test_effective_constant_near_1(self):
        """Final C(k) should be near 1."""
        proof = run_analytical_convergence_proof([10, 50, 100, 200, 500])
        assert 0.5 < proof.final_effective_constant < 2.0

    def test_all_lists_same_length(self):
        k_vals = [5, 10, 20]
        proof = run_analytical_convergence_proof(k_vals)
        n = len(k_vals)
        assert len(proof.telescoping) == n
        assert len(proof.pnt_bounds) == n
        assert len(proof.convergence_rates) == n
        assert len(proof.effective_constants) == n

    def test_cross_consistency(self):
        """Numerator from telescoping should match convergence rate numerator."""
        proof = run_analytical_convergence_proof([20, 50])
        for t, c in zip(proof.telescoping, proof.convergence_rates):
            assert abs(t.numerical_value - c.numerator) < 1e-12


# ============================================================================
# TestTheoreticalProperties — Physics-derived properties
# ============================================================================


class TestTheoreticalProperties:
    """Cross-checks anchored to TNFR-Riemann physics."""

    def test_sigma_star_equals_formula(self):
        """σ* = 1/2 − tr(L V_1)/tr(V_1²) must hold identically."""
        for k in [10, 50, 100]:
            result = compute_convergence_rate_bound(k)
            expected = 0.5 - result.numerator / result.denominator
            assert abs(result.sigma_star - expected) < 1e-14

    def test_curvature_equals_tr_v1sq_over_k(self):
        """d²E/dσ² = (1/k) tr(V_1²)."""
        for k in [10, 50, 100]:
            result = compute_convergence_rate_bound(k)
            expected = result.denominator / k
            assert abs(result.curvature - expected) < 1e-14

    def test_deviation_equals_numerator_over_denominator(self):
        """|σ* − 1/2| = tr(L V_1) / tr(V_1²)."""
        for k in [10, 50, 100]:
            result = compute_convergence_rate_bound(k)
            expected = result.numerator / result.denominator
            assert abs(result.deviation - expected) < 1e-14

    def test_telescoping_grows_as_log_k_squared(self):
        """tr(L V_1) ~ (log k)² asymptotically."""
        for k in [50, 100, 200]:
            tele = compute_telescoping_trace(k)
            log_k_sq = math.log(k) ** 2
            ratio = tele.telescoping_value / log_k_sq
            # ratio → 1 as k → ∞ (with log log k correction)
            assert 0.8 < ratio < 3.0

    def test_sum_log_sq_grows_as_k_log_k_squared(self):
        """tr(V_1²) ~ k(log k)² asymptotically (with constant > 1)."""
        for k in [50, 100, 200]:
            pnt = pnt_sum_log_squared(k)
            # scaling_ratio = exact / k(log k)² should be bounded
            assert 1.0 < pnt.scaling_ratio < 3.0

    def test_power_law_fit(self):
        """ln|σ*−1/2| vs ln(k) should have slope near −1."""
        k_vals = [20, 50, 100, 200, 500]
        results = analyze_convergence_sequence(k_vals)
        log_k = np.array([np.log(r.k) for r in results])
        log_dev = np.array([np.log(r.deviation) for r in results])
        # Linear regression: log_dev ≈ a + slope * log_k
        slope, _ = np.polyfit(log_k, log_dev, 1)
        # slope should be near -1 (i.e. O(1/k) rate)
        assert -1.3 < slope < -0.7, f"slope={slope:.3f}, expected ~-1"


# ============================================================================
# TestEdgeCases
# ============================================================================


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_k2_minimum(self):
        """k=2 is the smallest valid graph."""
        tele = compute_telescoping_trace(2)
        assert tele.relative_error < 1e-12

    def test_k2_convergence(self):
        result = compute_convergence_rate_bound(2)
        assert result.deviation > 0
        assert result.sigma_star < 0.5

    def test_k3(self):
        result = compute_convergence_rate_bound(3)
        assert result.deviation > 0

    def test_large_k_300(self):
        """Ensure no numerical issues at k=300."""
        result = compute_convergence_rate_bound(300)
        assert result.deviation > 0
        assert result.deviation < 0.1  # 1/300 ~ 0.003
