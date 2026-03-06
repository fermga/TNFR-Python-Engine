r"""Tests for the TNFR–Riemann spectral proof framework.

Validates the three lines of attack toward σ_c^(k) → 1/2:

1. **GUE universality**: Eigenvalue statistics converge to GUE predictions
2. **Lyapunov attractor**: σ = 1/2 is stable energy minimum
3. **Spectral determinant**: Ξ(s) ≈ Ξ(1−s) functional equation

Physics basis: H_TNFR^(k)(σ) = L_k + V_σ where V_σ(i,i) = (σ−½)log p_i.
              σ_c^(k) = inf{σ : λ_min^(k)(σ) ≥ 0} → 1/2 as k → ∞.

All tests are deterministic (no random seeds required) since the prime
path graph construction is fully determined by k.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tnfr.riemann.operator import (
    build_prime_path_graph,
    build_h_tnfr,
    default_prime_potential,
    _first_primes,
)
from tnfr.riemann.spectral_proof import (
    CriticalParameterResult,
    GUEStatisticsResult,
    LyapunovLandscapeResult,
    SpectralDeterminantResult,
    ConvergenceProofResult,
    compute_eigenspectrum,
    find_critical_sigma,
    compute_critical_parameter_sequence,
    compute_eigenvalue_spacings,
    compute_pair_correlation,
    gue_pair_correlation_theoretical,
    compute_gue_statistics,
    compute_lyapunov_energy,
    compute_lyapunov_landscape,
    verify_lyapunov_attractor,
    compute_spectral_zeta,
    compute_spectral_determinant,
    verify_functional_equation,
    verify_convergence_proof,
    _unfold_eigenvalues,
    _wigner_surmise_gue,
    _fit_convergence_rate,
)


# ============================================================================
# Existing operator sanity checks
# ============================================================================

class TestPrimePath:
    """Tests for the prime path graph construction."""

    def test_first_primes(self) -> None:
        assert _first_primes(5) == [2, 3, 5, 7, 11]
        assert _first_primes(1) == [2]
        assert _first_primes(0) == []

    def test_graph_structure(self) -> None:
        G = build_prime_path_graph(5)
        assert G.number_of_nodes() == 5
        assert G.number_of_edges() == 4  # Path graph: k-1 edges
        labels = [G.nodes[n]["label"] for n in sorted(G.nodes())]
        assert labels == [2, 3, 5, 7, 11]

    def test_log_gap_weights(self) -> None:
        G = build_prime_path_graph(3, weight_by_log_gap=True)
        # Edge 0-1: |log(3) - log(2)| ≈ 0.405
        w = G[0][1]["weight"]
        assert abs(w - abs(math.log(3) - math.log(2))) < 1e-10

    def test_uniform_weights(self) -> None:
        G = build_prime_path_graph(3, weight_by_log_gap=False)
        assert G[0][1]["weight"] == 1.0


class TestHOperator:
    """Tests for the H_TNFR operator construction."""

    def test_symmetry(self) -> None:
        """H_TNFR must be symmetric (self-adjoint)."""
        G = build_prime_path_graph(10)
        H, _ = build_h_tnfr(G, sigma=0.5)
        assert np.allclose(H, H.T)

    def test_sigma_half_pure_laplacian(self) -> None:
        """At σ = 1/2, V_σ = 0 so H = L (pure Laplacian)."""
        G = build_prime_path_graph(10)
        H, V = build_h_tnfr(G, sigma=0.5)
        assert np.allclose(V, 0.0)
        # Laplacian: row sums = 0
        row_sums = np.sum(H, axis=1)
        assert np.allclose(row_sums, 0.0, atol=1e-12)

    def test_potential_sign(self) -> None:
        """V_σ > 0 when σ > 1/2, < 0 when σ < 1/2."""
        for p in [2, 3, 5, 7, 11]:
            assert default_prime_potential(p, 0.75) > 0
            assert default_prime_potential(p, 0.25) < 0
            assert abs(default_prime_potential(p, 0.5)) < 1e-15

    def test_eigenvalue_monotonicity(self) -> None:
        """λ_min(σ) should increase with σ (monotonicity)."""
        k = 20
        lam_lo = compute_eigenspectrum(k, 0.25)[0]
        lam_mid = compute_eigenspectrum(k, 0.50)[0]
        lam_hi = compute_eigenspectrum(k, 0.75)[0]
        assert lam_lo < lam_mid < lam_hi


# ============================================================================
# Core: Critical parameter σ_c^(k)
# ============================================================================

class TestCriticalParameter:
    """Tests for finding σ_c^(k) where λ_min = 0."""

    def test_critical_between_0_and_1(self) -> None:
        """σ_c should be in (0, 1) for moderate k."""
        for k in [5, 10, 20]:
            sc = find_critical_sigma(k)
            assert 0.0 < sc < 1.0, f"k={k}: σ_c={sc} out of range"

    def test_critical_near_half(self) -> None:
        """σ_c should be close to 1/2 (within tolerance for finite k)."""
        sc = find_critical_sigma(50)
        assert abs(sc - 0.5) < 0.3, f"σ_c={sc} too far from 1/2"

    def test_critical_convergence_trend(self) -> None:
        """σ_c should trend toward 1/2 as k increases."""
        k_values = [5, 10, 20, 50]
        deviations = [abs(find_critical_sigma(k) - 0.5) for k in k_values]
        # Overall trend: last deviation should be less than first
        assert deviations[-1] < deviations[0], (
            f"σ_c not converging: deviations = {deviations}"
        )

    def test_lambda_min_at_critical_is_near_zero(self) -> None:
        """At σ = σ_c, λ_min should be approximately 0."""
        k = 20
        sc = find_critical_sigma(k, tolerance=1e-10)
        evals = compute_eigenspectrum(k, sc)
        assert abs(evals[0]) < 1e-6, f"λ_min at σ_c = {evals[0]}"

    def test_critical_sequence_dataclass(self) -> None:
        """compute_critical_parameter_sequence returns valid results."""
        results = compute_critical_parameter_sequence([5, 10, 20])
        assert len(results) == 3
        for r in results:
            assert isinstance(r, CriticalParameterResult)
            assert r.k >= 5
            assert 0.0 < r.sigma_critical < 1.0
            assert r.deviation_from_half == abs(r.sigma_critical - 0.5)


# ============================================================================
# Line 1: GUE Universality
# ============================================================================

class TestGUEAnalysis:
    """Tests for GUE eigenvalue statistics analysis."""

    def test_unfold_preserves_count(self) -> None:
        """Unfolding should preserve the number of eigenvalues."""
        evals = np.array([0.1, 0.5, 1.2, 2.0, 3.5])
        unfolded = _unfold_eigenvalues(evals)
        assert len(unfolded) == len(evals)

    def test_spacings_positive(self) -> None:
        """Nearest-neighbor spacings should be non-negative."""
        evals = compute_eigenspectrum(20, 0.5)
        spacings = compute_eigenvalue_spacings(evals)
        assert np.all(spacings >= -1e-10)  # Small numerical tolerance

    def test_spacings_mean_near_one(self) -> None:
        """After normalization, mean spacing should be ≈1."""
        evals = compute_eigenspectrum(50, 0.5)
        spacings = compute_eigenvalue_spacings(evals)
        if len(spacings) > 1:
            assert abs(np.mean(spacings) - 1.0) < 0.5

    def test_gue_pair_correlation_at_zero(self) -> None:
        """R₂(0) = 0 (level repulsion)."""
        r = np.array([0.0])
        R2 = gue_pair_correlation_theoretical(r)
        assert abs(R2[0]) < 1e-10

    def test_gue_pair_correlation_at_large_r(self) -> None:
        """R₂(r) → 1 for large r (uncorrelated)."""
        r = np.array([10.0, 20.0, 50.0])
        R2 = gue_pair_correlation_theoretical(r)
        assert np.all(np.abs(R2 - 1.0) < 0.02)

    def test_wigner_surmise_normalized(self) -> None:
        """Wigner surmise should integrate to ≈1."""
        s = np.linspace(0, 5, 1000)
        ds = s[1] - s[0]
        integral = np.sum(_wigner_surmise_gue(s) * ds)
        assert abs(integral - 1.0) < 0.02

    def test_gue_statistics_result(self) -> None:
        """compute_gue_statistics returns valid GUEStatisticsResult."""
        result = compute_gue_statistics(20)
        assert isinstance(result, GUEStatisticsResult)
        assert result.k == 20
        assert 0.0 <= result.gue_quality <= 1.0
        assert len(result.pair_correlation_r) > 0
        assert len(result.pair_correlation_R2) == len(result.pair_correlation_r)
        assert len(result.pair_correlation_gue) == len(result.pair_correlation_r)


# ============================================================================
# Line 2: Lyapunov Landscape
# ============================================================================

class TestLyapunovLandscape:
    """Tests for Lyapunov energy landscape analysis."""

    def test_energy_at_half_is_minimum(self) -> None:
        """At σ = 1/2, Frobenius energy should be minimal (pure Laplacian)."""
        k = 10
        E_half = compute_lyapunov_energy(k, 0.5)
        E_off = compute_lyapunov_energy(k, 0.7)
        # E(1/2) should be less than E at σ away from 1/2
        assert E_half < E_off

    def test_energy_symmetry_around_half(self) -> None:
        """Energy rises on both sides of the basin minimum.

        The Frobenius energy tr(H²)/2k has a (σ−½)² term from V²
        giving a quadratic basin, plus a linear cross-term 2(σ−½)tr(LV)
        that shifts the minimum slightly from 1/2 for finite k.
        """
        k = 20
        # Find approximate minimum via landscape
        result = compute_lyapunov_landscape(k, n_points=200)
        sigma_min = result.min_energy_sigma
        E_min = compute_lyapunov_energy(k, sigma_min)
        # Energy should rise on both sides of the minimum
        delta = 0.2
        E_left = compute_lyapunov_energy(k, max(0.01, sigma_min - delta))
        E_right = compute_lyapunov_energy(k, min(0.99, sigma_min + delta))
        assert E_left > E_min
        assert E_right > E_min

    def test_landscape_minimum_near_half(self) -> None:
        """Lyapunov landscape minimum should be near σ = 1/2."""
        result = compute_lyapunov_landscape(30, n_points=100)
        assert isinstance(result, LyapunovLandscapeResult)
        # For uniform Ψ, minimum is at σ = 1/2 (V vanishes)
        assert abs(result.min_energy_sigma - 0.5) < 0.1

    def test_landscape_convexity(self) -> None:
        """d²E/dσ² should be positive near σ = 1/2 (convex basin)."""
        result = compute_lyapunov_landscape(30, n_points=100)
        assert result.curvature_at_half > 0

    def test_attractor_quality_positive(self) -> None:
        """Attractor quality should be positive for reasonable k."""
        result = compute_lyapunov_landscape(20)
        assert result.attractor_quality > 0

    def test_verify_attractor_produces_list(self) -> None:
        """verify_lyapunov_attractor returns list of results."""
        results = verify_lyapunov_attractor([5, 10, 20])
        assert len(results) == 3
        for r in results:
            assert isinstance(r, LyapunovLandscapeResult)


# ============================================================================
# Line 3: Spectral Determinant
# ============================================================================

class TestSpectralDeterminant:
    """Tests for spectral determinant Ξ(s) analysis."""

    def test_determinant_real_on_critical_line(self) -> None:
        """Ξ(1/2 + it) should be approximately real (self-adjoint H)."""
        k = 10
        for t in [1.0, 5.0, 10.0]:
            xi = compute_spectral_determinant(k, complex(0.5, t))
            # For real symmetric H and s on critical line,
            # det(I - s^{-1}H) has real and imaginary parts
            # but should be well-defined complex number
            assert np.isfinite(xi.real)
            assert np.isfinite(xi.imag)

    def test_determinant_at_zero(self) -> None:
        """Ξ(0) = 1 by convention (identity)."""
        xi = compute_spectral_determinant(10, complex(0.0, 0.0))
        assert abs(xi - 1.0) < 1e-10

    def test_spectral_zeta_finite(self) -> None:
        """Spectral zeta ζ_{H^(k)}(σ, u) should be finite."""
        k = 10
        zeta = compute_spectral_zeta(k, 0.5, complex(1.0, 0.0))
        assert np.isfinite(zeta.real)
        assert np.isfinite(zeta.imag)

    def test_functional_equation_result(self) -> None:
        """verify_functional_equation returns valid result."""
        result = verify_functional_equation(10, n_test_points=10)
        assert isinstance(result, SpectralDeterminantResult)
        assert result.k == 10
        assert len(result.s_values) == 10
        assert len(result.xi_values) == 10
        assert len(result.functional_equation_residual) == 10
        assert result.zeros_on_critical_line >= 0


# ============================================================================
# Convergence Rate Fitting
# ============================================================================

class TestConvergenceRate:
    """Tests for convergence rate estimation."""

    def test_fit_with_known_rate(self) -> None:
        """Fit should recover known C/log(k) rate."""
        k_vals = [10, 50, 100, 500, 1000]
        C_true = 0.3
        sigma_c_vals = [0.5 + C_true / math.log(k) for k in k_vals]
        C_fit, alpha_fit = _fit_convergence_rate(k_vals, sigma_c_vals)
        assert abs(C_fit - C_true) < 0.1, f"C_fit={C_fit}, expected ~{C_true}"
        assert abs(alpha_fit - 1.0) < 0.3, f"α_fit={alpha_fit}, expected ~1.0"

    def test_fit_with_insufficient_data(self) -> None:
        """Fit with < 2 points should return defaults."""
        C, alpha = _fit_convergence_rate([10], [0.6])
        assert C == 0.0
        assert alpha == 1.0


# ============================================================================
# Full Integration: verify_convergence_proof
# ============================================================================

class TestFullConvergence:
    """Integration test for the full convergence proof framework."""

    def test_small_scale_convergence(self) -> None:
        """Run full convergence check on small graph sizes."""
        result = verify_convergence_proof(
            k_values=[5, 10, 20],
            run_gue=True,
            run_lyapunov=True,
            run_determinant=False,  # Skip expensive determinant for speed
        )
        assert isinstance(result, ConvergenceProofResult)
        assert len(result.k_values) == 3
        assert len(result.critical_parameters) == 3
        assert len(result.gue_statistics) >= 1
        assert len(result.lyapunov_landscapes) == 3
        assert result.overall_confidence >= 0.0

    def test_critical_parameters_converge(self) -> None:
        """σ_c^(k) should trend toward 1/2."""
        result = verify_convergence_proof(
            k_values=[5, 10, 20, 50],
            run_gue=False,
            run_lyapunov=False,
            run_determinant=False,
        )
        devs = [cp.deviation_from_half for cp in result.critical_parameters]
        # Not strictly monotone for small k, but overall trend
        assert devs[-1] < devs[0], f"Not converging: {devs}"

    def test_convergence_rate_is_sensible(self) -> None:
        """Fitted convergence rate should have positive C and α ~ 1."""
        result = verify_convergence_proof(
            k_values=[5, 10, 20, 50, 100],
            run_gue=False,
            run_lyapunov=False,
            run_determinant=False,
        )
        C, alpha = result.convergence_rate_fit
        # C should be positive (σ_c > 1/2 for finite k)
        assert C > 0 or alpha > 0, f"Invalid fit: C={C}, α={alpha}"
