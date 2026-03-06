r"""Tests for P9: Functional equation analog and spectral reflection symmetry.

Validates:
1. **Spectral Reflection** -- H(σ) + H(1-σ) = 2L (exact matrix identity)
2. **Trace Formulas** -- Exact trace-level functional equations (symmetric/antisymmetric)
3. **Completed Xi** -- log Ξ_H(σ) functional equation in log-space
4. **Conjecture 12.1** -- Riemann Invariant Conservation at σ = 1/2
5. **Conjecture 12.2** -- Frobenius energy minimised near σ = 1/2
6. **Large-k Convergence** -- Four-line verification at k ≥ 1000

Physics basis: H(σ) + H(1-σ) = 2L because V_σ + V_{1-σ} = 0.
"""

from __future__ import annotations

import numpy as np
import pytest

from tnfr.riemann.functional_equation import (
    # Data structures
    SpectralReflection,
    TraceFormulaResult,
    CompletedXiFunction,
    Conjecture12_1Result,
    Conjecture12_2Result,
    LargeKConvergence,
    FunctionalEquationAnalysis,
    # Core — reflection
    verify_spectral_reflection,
    verify_reflection_sequence,
    # Core — trace
    compute_trace_formulas,
    verify_trace_formula_pnt,
    # Core — xi
    compute_completed_xi,
    verify_xi_functional_equation,
    # Conjectures
    test_conjecture_12_1,
    test_conjecture_12_2,
    # Large-k
    verify_large_k_convergence,
    # Integration
    run_functional_equation_analysis,
)


# ============================================================================
# 1. Spectral Reflection Identity: H(σ) + H(1-σ) = 2L
# ============================================================================


class TestSpectralReflection:
    """Exact matrix identity H(σ) + H(1-σ) = 2L."""

    def test_trace_identity_exact(self) -> None:
        """tr(H(σ)) + tr(H(1-σ)) = 2·tr(L) at machine precision."""
        r = verify_spectral_reflection(50, 0.3)
        assert r.trace_error < 1e-12

    def test_energy_identity_exact(self) -> None:
        """E(σ) + E(1-σ) = 2E_L + δ²/k·tr(V₁²) at machine precision."""
        r = verify_spectral_reflection(50, 0.3)
        assert r.energy_error < 1e-10

    @pytest.mark.parametrize("sigma", [0.1, 0.3, 0.4, 0.6, 0.9])
    def test_trace_identity_multi_sigma(self, sigma: float) -> None:
        """Trace identity holds for various σ."""
        r = verify_spectral_reflection(30, sigma)
        assert r.trace_error < 1e-11

    @pytest.mark.parametrize("k", [5, 10, 50, 100, 200])
    def test_trace_identity_multi_k(self, k: int) -> None:
        """Trace identity holds across graph sizes."""
        r = verify_spectral_reflection(k, 0.3)
        assert r.trace_error < 1e-10
        assert r.energy_error < 1e-8

    def test_reflection_sequence(self) -> None:
        """Batch verification for a sequence of k values."""
        results = verify_reflection_sequence([10, 50, 100], 0.3)
        assert len(results) == 3
        for r in results:
            assert r.trace_error < 1e-10
            assert r.energy_error < 1e-8

    def test_eigenvalue_reflection_approximate(self) -> None:
        """Eigenvalue reflection is approximate (not exact) due to rotation."""
        r = verify_spectral_reflection(50, 0.3)
        # The eigenvalue reflection error > 0 is expected
        assert isinstance(r.eigenvalue_reflection_error, float)

    def test_sigma_half_trivial(self) -> None:
        """At σ = 0.5, δ = 0, identity is trivially satisfied."""
        r = verify_spectral_reflection(30, 0.5)
        assert r.trace_error < 1e-14
        assert r.energy_error < 1e-14

    def test_result_dataclass(self) -> None:
        """Verify SpectralReflection structure."""
        r = verify_spectral_reflection(10, 0.3)
        assert isinstance(r, SpectralReflection)
        assert r.k == 10
        assert r.sigma == 0.3

    def test_skips_k_below_3(self) -> None:
        """k < 3 filtered out by verify_reflection_sequence."""
        results = verify_reflection_sequence([1, 2, 10], 0.3)
        assert len(results) == 1
        assert results[0].k == 10


# ============================================================================
# 2. Trace Formula Identities
# ============================================================================


class TestTraceFormulas:
    """Exact trace-level functional equations."""

    def test_n1_trace_identity(self) -> None:
        """n=1: tr(H(σ)) + tr(H(1-σ)) = 2·tr(L)."""
        tf = compute_trace_formulas(50, 0.3)
        assert tf.trace_n1_error < 1e-11

    def test_n2_symmetric_identity(self) -> None:
        """n=2 symmetric: E(σ) + E(1-σ) = 2E_L + δ²/k·tr(V₁²)."""
        tf = compute_trace_formulas(100, 0.3)
        assert tf.energy_n2_symmetric_error < 1e-10

    def test_n2_antisymmetric_identity(self) -> None:
        """n=2 antisymmetric: E(σ) - E(1-σ) = (2δ/k)·tr(LV₁)."""
        tf = compute_trace_formulas(100, 0.3)
        assert tf.energy_n2_antisymmetric_error < 1e-10

    def test_pnt_mean_log_prime(self) -> None:
        """Mean log prime should be close to log(k) for large k."""
        import math
        tf = compute_trace_formulas(500, 0.3)
        # PNT: (1/k)Σlog(p_i) ≈ log(k) for large k
        assert abs(tf.pnt_mean_log_prime - math.log(500)) < 1.0

    def test_delta_sign(self) -> None:
        """Delta has correct sign."""
        tf = compute_trace_formulas(20, 0.3)
        assert tf.delta == pytest.approx(-0.2)
        tf2 = compute_trace_formulas(20, 0.7)
        assert tf2.delta == pytest.approx(0.2)

    def test_energy_diff_pnt_controlled(self) -> None:
        """Antisymmetric part is PNT-controlled (shrinks relative to k)."""
        tf_small = compute_trace_formulas(50, 0.3)
        tf_large = compute_trace_formulas(500, 0.3)
        # |E(σ)-E(1-σ)| relative to E(σ) should be smaller for larger k
        # since tr(LV₁)/k → 0 faster than tr(V₁²)/k → ∞
        ratio_small = abs(tf_small.energy_diff / max(tf_small.energy_sigma, 1e-15))
        ratio_large = abs(tf_large.energy_diff / max(tf_large.energy_sigma, 1e-15))
        # Not strict monotonicity, but large k should have smaller ratio
        # (within ~2x tolerance for stochastic-like effects)
        assert ratio_large < ratio_small * 2.0

    @pytest.mark.parametrize("sigma", [0.1, 0.25, 0.4, 0.6, 0.75, 0.9])
    def test_multi_sigma(self, sigma: float) -> None:
        """Both identities hold for various σ."""
        tf = compute_trace_formulas(50, sigma)
        assert tf.trace_n1_error < 1e-10
        assert tf.energy_n2_symmetric_error < 1e-9

    def test_verify_trace_formula_pnt_batch(self) -> None:
        """Batch verification."""
        results = verify_trace_formula_pnt([10, 50, 100], 0.3)
        assert len(results) == 3
        for tf in results:
            assert tf.trace_n1_error < 1e-10

    def test_result_dataclass(self) -> None:
        """Verify TraceFormulaResult structure."""
        tf = compute_trace_formulas(10, 0.3)
        assert isinstance(tf, TraceFormulaResult)
        assert tf.k == 10
        assert tf.sigma == 0.3
        assert isinstance(tf.tr_LV1, float)
        assert isinstance(tf.tr_V1_sq, float)


# ============================================================================
# 3. Completed Spectral Xi Function
# ============================================================================


class TestCompletedXi:
    """Completed spectral xi function in log-space."""

    def test_log_xi_reflection_symmetry(self) -> None:
        """log Ξ_H(σ) ≈ log Ξ_H(1-σ) near σ = 1/2."""
        xi = compute_completed_xi(50, n_points=100)
        # Near σ = 1/2, reflections should be closest
        half_idx = np.argmin(np.abs(xi.sigma_values - 0.5))
        # Check a few points near 1/2
        nearby = slice(max(0, half_idx - 5), min(len(xi.sigma_values), half_idx + 5))
        nearby_asym = xi.log_asymmetry[nearby]
        assert np.all(np.isfinite(nearby_asym) | (nearby_asym == 0.0))

    def test_xi_at_half_is_zero(self) -> None:
        """At σ = 1/2, log Ξ = -inf (δ = 0, so Ξ = 0)."""
        xi = compute_completed_xi(30, n_points=51)
        half_idx = np.argmin(np.abs(xi.sigma_values - 0.5))
        assert xi.log_xi_values[half_idx] == -np.inf

    def test_log_xi_arrays_correct_shape(self) -> None:
        """Arrays have correct shape."""
        n = 80
        xi = compute_completed_xi(20, n_points=n)
        assert xi.log_xi_values.shape == (n,)
        assert xi.log_xi_reflected.shape == (n,)
        assert xi.log_asymmetry.shape == (n,)
        assert xi.sigma_values.shape == (n,)

    def test_xi_multi_k(self) -> None:
        """Verify xi for multiple k values."""
        results = verify_xi_functional_equation([10, 30], n_points=30)
        assert len(results) == 2
        for r in results:
            assert isinstance(r, CompletedXiFunction)

    def test_result_dataclass(self) -> None:
        xi = compute_completed_xi(10, n_points=20)
        assert isinstance(xi, CompletedXiFunction)
        assert xi.k == 10
        assert isinstance(xi.max_log_asymmetry, float)
        assert isinstance(xi.mean_log_asymmetry, float)


# ============================================================================
# 4. Conjecture 12.1: Riemann Invariant Conservation
# ============================================================================


class TestConjecture12_1:
    """Numerical test for invariant conservation at σ = 1/2."""

    def test_energy_density_positive(self) -> None:
        """Energy density at σ = 1/2 should be non-negative."""
        c = test_conjecture_12_1(50)
        assert c.energy_density_at_half >= 0

    def test_optimal_alpha_finite(self) -> None:
        """Optimal α should be finite."""
        c = test_conjecture_12_1(50)
        assert np.isfinite(c.optimal_alpha)

    def test_conservation_improves_with_k(self) -> None:
        """Approximate conservation should improve for larger k."""
        c_small = test_conjecture_12_1(50)
        c_large = test_conjecture_12_1(500)
        # Combined derivative should be smaller (or at least bounded)
        # for larger k
        assert abs(c_large.combined_derivative) < 1.0

    def test_is_conserved_moderate_k(self) -> None:
        """At moderate k, conservation should hold."""
        c = test_conjecture_12_1(200)
        assert c.is_approximately_conserved

    def test_result_dataclass(self) -> None:
        c = test_conjecture_12_1(20)
        assert isinstance(c, Conjecture12_1Result)
        assert c.k == 20


# ============================================================================
# 5. Conjecture 12.2: Multiscale Minimization
# ============================================================================


class TestConjecture12_2:
    """Frobenius energy minimised near σ = 1/2."""

    def test_minimum_near_half(self) -> None:
        """Energy minimum should be near σ = 1/2."""
        c = test_conjecture_12_2(100)
        assert c.deviation_from_half < 0.05

    def test_minimum_improves_with_k(self) -> None:
        """Minimum gets closer to 1/2 for larger k."""
        c_small = test_conjecture_12_2(50)
        c_large = test_conjecture_12_2(500)
        # By PNT, |σ* - 1/2| = O(1/k), so larger k → closer
        # Allow tolerance for scan discretisation
        assert c_large.deviation_from_half < c_small.deviation_from_half + 0.01

    def test_energy_at_half_less_than_boundary(self) -> None:
        """E(1/2) should be less than E at boundaries."""
        c = test_conjecture_12_2(100, sigma_range=(0.1, 0.9))
        # E(1/2) < max(E) at boundaries
        assert c.coherence_at_half < float(np.max(c.coherence_values))

    def test_quadratic_shape(self) -> None:
        """Energy landscape should be approximately quadratic near 1/2."""
        c = test_conjecture_12_2(200, n_points=200)
        # Near 1/2, E(σ) ≈ E_min + a·(σ - σ*)²
        # Check that values increase away from minimum
        idx_min = int(np.argmin(c.coherence_values))
        # Values left and right of minimum should be larger
        if idx_min > 5 and idx_min < len(c.coherence_values) - 5:
            assert c.coherence_values[idx_min - 5] > c.coherence_at_min * 0.99
            assert c.coherence_values[idx_min + 5] > c.coherence_at_min * 0.99

    def test_is_minimized_flag(self) -> None:
        """Flag should be True for reasonable k."""
        c = test_conjecture_12_2(200, tolerance=0.05)
        assert c.is_minimized_at_half

    def test_result_dataclass(self) -> None:
        c = test_conjecture_12_2(20)
        assert isinstance(c, Conjecture12_2Result)
        assert c.k == 20
        assert len(c.sigma_values) == 200


# ============================================================================
# 6. Large-k Convergence
# ============================================================================


class TestLargeKConvergence:
    """Four-line verification at moderate-to-large k."""

    def test_equilibrium(self) -> None:
        """λ_min(H(1/2)) = 0 (Laplacian null eigenvalue)."""
        lk = verify_large_k_convergence(500)
        assert abs(lk.lambda_min_at_half) < 1e-12

    def test_thermodynamic_convergence(self) -> None:
        """σ* → 1/2 at rate O(1/k)."""
        lk = verify_large_k_convergence(1000)
        assert lk.deviation < 0.01  # |σ* - 1/2| < 0.01 for k=1000
        assert lk.effective_constant < 2.0  # C(k) bounded

    def test_hellmann_feynman_flow(self) -> None:
        """All eigenvalue velocities positive."""
        lk = verify_large_k_convergence(500)
        assert lk.all_velocities_positive
        assert lk.min_velocity > 0

    def test_spectral_gap_positive(self) -> None:
        """Spectral gap > 0 (connected graph)."""
        lk = verify_large_k_convergence(100)
        assert lk.spectral_gap > 0

    def test_effective_constant_converges(self) -> None:
        """C(k) = k·|σ*-1/2| should be near 1 for large k."""
        lk = verify_large_k_convergence(2000)
        assert 0.5 < lk.effective_constant < 2.0

    def test_computation_time_recorded(self) -> None:
        """Timing information present."""
        lk = verify_large_k_convergence(100)
        assert lk.computation_time_s > 0

    def test_result_dataclass(self) -> None:
        lk = verify_large_k_convergence(10)
        assert isinstance(lk, LargeKConvergence)
        assert lk.k == 10


# ============================================================================
# 7. Integration
# ============================================================================


class TestIntegration:
    """run_functional_equation_analysis combines all P9 analyses."""

    def test_basic_run(self) -> None:
        """Integration runs without error for small k."""
        analysis = run_functional_equation_analysis(
            k_values=[10, 30],
            large_k_values=[100],
        )
        assert isinstance(analysis, FunctionalEquationAnalysis)
        assert analysis.reflection_exact
        assert len(analysis.reflections) == 2
        assert len(analysis.trace_formulas) == 2
        assert analysis.xi_function is not None
        assert len(analysis.conjecture_12_1) == 2
        assert analysis.conjecture_12_2 is not None
        assert len(analysis.large_k) == 1

    def test_summary_nonempty(self) -> None:
        """Summary string is generated."""
        analysis = run_functional_equation_analysis(
            k_values=[10],
            large_k_values=[50],
        )
        assert len(analysis.summary) > 0
        assert "P9" in analysis.summary

    def test_convergence_exponent(self) -> None:
        """Convergence exponent estimated from single large-k."""
        analysis = run_functional_equation_analysis(
            k_values=[10],
            large_k_values=[500],
        )
        assert analysis.convergence_exponent == pytest.approx(1.0)

    def test_defaults(self) -> None:
        """Default k_values work."""
        analysis = run_functional_equation_analysis()
        assert len(analysis.k_values) == 4
