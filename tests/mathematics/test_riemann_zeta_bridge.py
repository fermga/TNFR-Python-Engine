r"""Tests for P11: Zeta bridge — explicit connection ζ_H ↔ ζ_R.

Validates:
1. **Weyl Asymptotic** -- N(λ) ~ A·λ^α eigenvalue counting law
2. **Heat Kernel Reflection** -- Θ_σ + Θ_{1-σ} = 2Θ_L + δ²·R(β)
3. **Spectral Zeta Reflection** -- regularised ζ_H(σ)+ζ_H(1-σ) = 2ζ_L + O(δ²)
4. **Prime Encoding** -- Σ Λ_H(j) = θ(p_k) trace identity
5. **Scaling Law** -- Conj 10.1 trend extraction (asymptotic result)
6. **Full Bridge Analysis** -- Integration of all components

Physics basis: The heat kernel and spectral zeta functional equations
express the σ ↔ 1-σ reflection symmetry of H(σ) at the level of
spectral invariants, creating the bridge from the discrete TNFR
operator to Riemann's analytic ζ.
"""

from __future__ import annotations

import numpy as np
import pytest

from tnfr.riemann.zeta_bridge import (
    # Data structures
    WeylAsymptotic,
    HeatKernelReflection,
    SpectralZetaReflection,
    ScalingLaw,
    PrimeEncoding,
    ZetaBridgeAnalysis,
    # Functions
    compute_weyl_asymptotic,
    compute_heat_kernel_reflection,
    compute_spectral_zeta_reflection,
    extract_scaling_law,
    compute_prime_encoding,
    run_zeta_bridge_analysis,
)


# ============================================================================
# 1. Weyl Eigenvalue Counting: N(λ) ~ A·λ^α
# ============================================================================


class TestWeylAsymptotic:
    """Weyl law for the Laplacian on the prime path graph."""

    @pytest.mark.parametrize("k", [20, 50, 100])
    def test_fit_quality(self, k: int) -> None:
        """R² of log-log fit should exceed 0.9."""
        w = compute_weyl_asymptotic(k)
        assert isinstance(w, WeylAsymptotic)
        assert w.r_squared > 0.9

    def test_alpha_around_half(self) -> None:
        """For 1D chains, α ≈ 1/2 (standard Weyl law)."""
        w = compute_weyl_asymptotic(100)
        assert 0.3 < w.alpha < 0.7

    def test_A_coeff_positive(self) -> None:
        w = compute_weyl_asymptotic(50)
        assert w.A_coeff > 0

    def test_eigenvalues_sorted(self) -> None:
        w = compute_weyl_asymptotic(30)
        diffs = np.diff(w.eigenvalues)
        assert np.all(diffs >= -1e-12)

    def test_k_stored(self) -> None:
        w = compute_weyl_asymptotic(42)
        assert w.k == 42


# ============================================================================
# 2. Heat Kernel Reflection: Θ_σ + Θ_{1-σ} = 2Θ_L + δ²·R(β)
# ============================================================================


class TestHeatKernelReflection:
    """Theorem A: heat kernel functional equation."""

    def test_basic_reflection(self) -> None:
        h = compute_heat_kernel_reflection(50, 0.6)
        assert isinstance(h, HeatKernelReflection)
        assert h.k == 50
        assert h.sigma == 0.6

    def test_leading_order_accuracy(self) -> None:
        """Relative error of δ²β²·tr(V₁²) prediction < 10%."""
        h = compute_heat_kernel_reflection(50, 0.6)
        assert h.relative_error_leading < 0.1

    @pytest.mark.parametrize("sigma", [0.55, 0.6, 0.7, 0.8])
    def test_multi_sigma(self, sigma: float) -> None:
        """Functional equation holds across σ."""
        h = compute_heat_kernel_reflection(50, sigma)
        assert h.relative_error_leading < 0.15

    @pytest.mark.parametrize("k", [30, 50, 100])
    def test_multi_k(self, k: int) -> None:
        h = compute_heat_kernel_reflection(k, 0.6)
        assert h.relative_error_leading < 0.15

    def test_delta_squared(self) -> None:
        """δ² = (σ-1/2)² stored correctly."""
        h = compute_heat_kernel_reflection(50, 0.7)
        assert abs(h.delta_sq - 0.04) < 1e-12

    def test_tr_V1_sq_positive(self) -> None:
        """tr(V₁²) > 0 (sum of squared log primes)."""
        h = compute_heat_kernel_reflection(50, 0.6)
        assert h.tr_V1_sq > 0

    def test_residual_proportional_to_delta_sq(self) -> None:
        """Residual at σ=0.8 larger than at σ=0.55 (δ² scaling)."""
        h_small = compute_heat_kernel_reflection(50, 0.55)
        h_large = compute_heat_kernel_reflection(50, 0.8)
        # Max residual scales with δ²
        r_small = np.max(np.abs(h_small.residual))
        r_large = np.max(np.abs(h_large.residual))
        assert r_large > r_small


# ============================================================================
# 3. Spectral Zeta Reflection (regularised)
# ============================================================================


class TestSpectralZetaReflection:
    """Regularised ζ_H(σ)+ζ_H(1-σ) = 2ζ_L + O(δ²)."""

    def test_basic(self) -> None:
        r = compute_spectral_zeta_reflection(50, 0.6)
        assert isinstance(r, SpectralZetaReflection)
        assert r.k == 50
        assert r.sigma == 0.6

    def test_delta_sq_scaling(self) -> None:
        """δ² scaling verified."""
        r = compute_spectral_zeta_reflection(50, 0.6)
        assert r.delta_sq_scaling

    @pytest.mark.parametrize("sigma", [0.55, 0.6, 0.7, 0.9])
    def test_multi_sigma(self, sigma: float) -> None:
        r = compute_spectral_zeta_reflection(50, sigma)
        assert r.delta_sq_scaling

    @pytest.mark.parametrize("k", [20, 50, 100])
    def test_multi_k(self, k: int) -> None:
        r = compute_spectral_zeta_reflection(k, 0.7)
        assert r.delta_sq_scaling

    def test_residual_shape(self) -> None:
        """Residual has same length as u_values."""
        u = np.linspace(2.0, 4.0, 10)
        r = compute_spectral_zeta_reflection(30, 0.7, u_values=u)
        assert r.residual.shape == (10,)

    def test_max_error_bounded(self) -> None:
        """Max error of median-ratio test < 0.5."""
        r = compute_spectral_zeta_reflection(50, 0.6)
        assert r.max_relative_error < 0.5

    def test_shift_is_canonical(self) -> None:
        """Regularisation buffer equals γ/π (canonical, not ad-hoc).

        The shift is composed as a = max(0, -min_eigenvalue) + γ/π,
        where γ/π = CRITICAL_EXPONENT comes from the Universal
        Tetrahedral Correspondence (γ ↔ |∇φ|). This replaces the
        previous ad-hoc 0.1 buffer with a first-principles constant.
        """
        from tnfr.constants.canonical import CRITICAL_EXPONENT
        from tnfr.riemann.spectral_proof import compute_eigensystem

        k, sigma = 50, 0.6
        evals_s, _ = compute_eigensystem(k, sigma)
        evals_r, _ = compute_eigensystem(k, 1.0 - sigma)
        evals_L, _ = compute_eigensystem(k, 0.5)
        all_min = min(evals_s.min(), evals_r.min(), evals_L.min())
        expected_shift = max(0.0, -all_min) + CRITICAL_EXPONENT

        r = compute_spectral_zeta_reflection(k, sigma)
        assert r.shift_canonical is True
        assert abs(r.shift_value - expected_shift) < 1e-12


# ============================================================================
# 4. Prime Encoding: Σ Λ_H(j) = θ(p_k)
# ============================================================================


class TestPrimeEncoding:
    """Spectral von Mangoldt function and trace identity."""

    @pytest.mark.parametrize("k", [10, 30, 50, 100])
    def test_trace_identity(self, k: int) -> None:
        """Trace identity error < 10⁻¹⁰."""
        pe = compute_prime_encoding(k)
        assert isinstance(pe, PrimeEncoding)
        assert pe.identity_error < 1e-10

    def test_lambda_H_length(self) -> None:
        pe = compute_prime_encoding(50)
        assert len(pe.lambda_H) == 50

    def test_chebyshev_positive(self) -> None:
        pe = compute_prime_encoding(50)
        assert pe.chebyshev_theta > 0

    def test_total_spectral_matches_theta(self) -> None:
        pe = compute_prime_encoding(50)
        assert abs(pe.total_spectral - pe.chebyshev_theta) < 1e-10

    def test_lambda_H_mostly_positive(self) -> None:
        """Most entries of Λ_H should be positive (log primes are positive)."""
        pe = compute_prime_encoding(50)
        pos_frac = np.sum(pe.lambda_H > 0) / len(pe.lambda_H)
        assert pos_frac > 0.5


# ============================================================================
# 5. Scaling Law: Conj 10.1 trend
# ============================================================================


class TestScalingLaw:
    """Conj 10.1 scaling (asymptotic — not expected to converge at small k)."""

    def test_returns_valid_structure(self) -> None:
        sl = extract_scaling_law([30, 50])
        assert isinstance(sl, ScalingLaw)
        assert len(sl.k_values) == 2

    def test_no_crash_on_single_k(self) -> None:
        sl = extract_scaling_law([50])
        assert len(sl.k_values) == 1

    def test_empty_on_failure(self) -> None:
        """If all k fail, return empty ScalingLaw."""
        sl = extract_scaling_law([2])
        # k=2 may fail; just check structure
        assert isinstance(sl, ScalingLaw)


# ============================================================================
# 6. Full Bridge Analysis
# ============================================================================


class TestZetaBridgeAnalysis:
    """Integration test: run_zeta_bridge_analysis."""

    def test_completes(self) -> None:
        result = run_zeta_bridge_analysis(k_values=[20, 30], sigma_test=0.6)
        assert isinstance(result, ZetaBridgeAnalysis)

    def test_weyl_present(self) -> None:
        result = run_zeta_bridge_analysis(k_values=[30], sigma_test=0.6)
        assert result.weyl.r_squared > 0.8

    def test_heat_kernel_present(self) -> None:
        result = run_zeta_bridge_analysis(k_values=[30], sigma_test=0.6)
        assert result.heat_kernel.relative_error_leading < 0.15

    def test_spectral_zeta_present(self) -> None:
        result = run_zeta_bridge_analysis(k_values=[30], sigma_test=0.6)
        assert result.spectral_zeta.delta_sq_scaling

    def test_prime_encoding_present(self) -> None:
        result = run_zeta_bridge_analysis(k_values=[30], sigma_test=0.6)
        assert result.prime_encoding.identity_error < 1e-10

    def test_summary_nonempty(self) -> None:
        result = run_zeta_bridge_analysis(k_values=[20, 30], sigma_test=0.6)
        assert len(result.summary) > 30

    def test_timing_recorded(self) -> None:
        result = run_zeta_bridge_analysis(k_values=[20], sigma_test=0.6)
        assert result.computation_time_s > 0
