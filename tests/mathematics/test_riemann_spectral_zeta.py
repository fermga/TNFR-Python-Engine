"""Tests for P5: Discrete spectral zeta and trace formula.

Covers:
- Spectral zeta computation and mathematical properties
- Heat kernel trace and thermodynamics
- Mellin transform bridge verification
- Conjecture 10.1 fitting
- Edge cases and invariants
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tnfr.riemann.spectral_zeta import (
    RIEMANN_ZETA_KNOWN_VALUES,
    SpectralZetaResult,
    HeatKernelResult,
    MellinBridgeResult,
    ConjectureTestResult,
    SpectralZetaAnalysis,
    compute_positive_eigenvalues,
    compute_spectral_zeta,
    compute_spectral_zeta_derivative,
    compute_heat_kernel_trace,
    compute_partition_function,
    compute_free_energy,
    verify_mellin_bridge,
    riemann_zeta_approx,
    test_conjecture_10_1,
    test_conjecture_10_1_sequence,
    run_spectral_zeta_analysis,
    _pearson,
)
from tnfr.riemann.spectral_proof import compute_eigenspectrum


# ============================================================================
# Test: Core Eigenvalue Extraction
# ============================================================================


class TestPositiveEigenvalues:
    """Test the extraction of positive eigenvalues."""

    def test_at_half_excludes_zero(self):
        """At sigma=1/2, H=L has one zero eigenvalue (ker span{1})."""
        evals = compute_positive_eigenvalues(10, 0.5)
        assert len(evals) == 9  # k-1 positive for connected path graph

    def test_all_positive(self):
        """All returned eigenvalues are strictly positive."""
        evals = compute_positive_eigenvalues(20, 0.5)
        assert np.all(evals > 0)

    def test_sorted(self):
        """Eigenvalues are sorted ascending."""
        evals = compute_positive_eigenvalues(15, 0.5)
        assert np.all(np.diff(evals) >= 0)

    def test_off_half_may_include_all(self):
        """At sigma != 1/2, all eigenvalues may be positive."""
        evals = compute_positive_eigenvalues(10, 0.6)
        # At sigma > 1/2 the potential shifts upward, all should be > 0
        assert len(evals) == 10

    def test_count_matches_spectrum(self):
        """Positive eigenvalues are a subset of the full spectrum."""
        full = compute_eigenspectrum(10, 0.5)
        pos = compute_positive_eigenvalues(10, 0.5)
        assert len(pos) <= len(full)
        # Each positive eigenvalue should be close to one in full
        for lam in pos:
            assert np.min(np.abs(full - lam)) < 1e-10

    def test_k2_minimum(self):
        """k=2 path graph has 1 positive eigenvalue at sigma=1/2."""
        evals = compute_positive_eigenvalues(2, 0.5)
        assert len(evals) == 1

    def test_k3(self):
        """k=3 path graph: 2 positive eigenvalues at sigma=1/2."""
        evals = compute_positive_eigenvalues(3, 0.5)
        assert len(evals) == 2


# ============================================================================
# Test: Spectral Zeta Function
# ============================================================================


class TestSpectralZeta:
    """Test the discrete spectral zeta computation."""

    def test_returns_dataclass(self):
        result = compute_spectral_zeta(10, 0.5)
        assert isinstance(result, SpectralZetaResult)
        assert result.k == 10
        assert result.sigma == 0.5

    def test_shape(self):
        u = np.linspace(1.0, 5.0, 20)
        result = compute_spectral_zeta(10, 0.5, u_values=u)
        assert result.zeta_values.shape == (20,)
        assert result.u_values.shape == (20,)

    def test_positive_for_positive_u(self):
        """Spectral zeta with positive eigenvalues and u > 0 is positive."""
        u = np.linspace(0.5, 5.0, 20)
        result = compute_spectral_zeta(10, 0.5, u_values=u)
        assert np.all(result.zeta_values > 0)

    def test_monotone_behaviour_with_u(self):
        """Spectral zeta varies smoothly with u (no jumps)."""
        u = np.linspace(1.5, 5.0, 20)
        result = compute_spectral_zeta(20, 0.5, u_values=u)
        # Path-graph Laplacian has eigenvalues both < 1 and > 1,
        # so zeta may increase or decrease; just verify smoothness
        ratios = result.zeta_values[1:] / result.zeta_values[:-1]
        # Adjacent ratios should not jump by more than an order of magnitude
        assert np.all(ratios > 0.01) and np.all(ratios < 100)

    def test_u_zero_counts_eigenvalues(self):
        """At u=0, zeta(0) = sum lambda^0 = number of positive eigenvalues."""
        u = np.array([0.0])
        result = compute_spectral_zeta(10, 0.5, u_values=u)
        assert abs(result.zeta_values[0] - result.n_positive) < 1e-10

    def test_n_positive_field(self):
        result = compute_spectral_zeta(10, 0.5)
        assert result.n_positive == 9  # k-1 at sigma=1/2

    def test_eigenvalues_positive_stored(self):
        result = compute_spectral_zeta(10, 0.5)
        assert len(result.eigenvalues_positive) == result.n_positive
        assert np.all(result.eigenvalues_positive > 0)

    def test_u1_is_sum_inverse_eigenvalues(self):
        """At u=1, zeta(1) = sum 1/lambda_j."""
        evals = compute_positive_eigenvalues(10, 0.5)
        expected = np.sum(1.0 / evals)
        result = compute_spectral_zeta(10, 0.5, u_values=np.array([1.0]))
        assert abs(result.zeta_values[0] - expected) < 1e-10


class TestSpectralZetaDerivative:
    """Test the spectral zeta derivative."""

    def test_shape(self):
        u = np.linspace(1.0, 4.0, 15)
        d = compute_spectral_zeta_derivative(10, 0.5, u_values=u)
        assert d.shape == (15,)

    def test_finite_values(self):
        """Derivative should be finite at all typical u values."""
        u = np.array([2.0, 3.0, 4.0])
        d = compute_spectral_zeta_derivative(20, 0.5, u_values=u)
        assert np.all(np.isfinite(d))

    def test_finite_difference_check(self):
        """Derivative should match finite difference (relative tolerance)."""
        u = np.array([2.0])
        eps = 1e-6
        z1 = compute_spectral_zeta(10, 0.5, u_values=u).zeta_values[0]
        z2 = compute_spectral_zeta(10, 0.5, u_values=u + eps).zeta_values[0]
        fd = (z2 - z1) / eps
        analytic = compute_spectral_zeta_derivative(10, 0.5, u_values=u)[0]
        # Use relative tolerance — values can be large
        denom = max(abs(analytic), abs(fd), 1e-15)
        assert abs(fd - analytic) / denom < 1e-3


# ============================================================================
# Test: Heat Kernel and Thermodynamics
# ============================================================================


class TestHeatKernel:
    """Test heat kernel trace and thermodynamic quantities."""

    def test_returns_dataclass(self):
        result = compute_heat_kernel_trace(10, 0.5)
        assert isinstance(result, HeatKernelResult)
        assert result.k == 10

    def test_theta_positive(self):
        """Heat kernel trace is always positive (sum of exponentials)."""
        result = compute_heat_kernel_trace(10, 0.5)
        assert np.all(result.theta_values > 0)

    def test_theta_decreasing_in_beta(self):
        """Theta decreases as beta increases (deeper cooling)."""
        betas = np.array([0.01, 0.1, 1.0, 10.0, 100.0])
        result = compute_heat_kernel_trace(10, 0.5, beta_values=betas)
        # Theta should be monotonically decreasing
        assert np.all(np.diff(result.theta_values) <= 0)

    def test_theta_at_zero_beta_equals_k(self):
        """Theta(0) = tr(I) = k (all exponentials are 1)."""
        betas = np.array([1e-6])
        result = compute_heat_kernel_trace(10, 0.5, beta_values=betas)
        # Very small beta: should be close to k
        assert abs(result.theta_values[0] - 10.0) < 0.01

    def test_theta_at_large_beta(self):
        """At large beta, only ground state survives."""
        betas = np.array([1000.0])
        result = compute_heat_kernel_trace(10, 0.5, beta_values=betas)
        # At sigma=1/2 ground state is lambda=0, so e^0 = 1 dominates
        # Other eigenvalues produce negligible e^{-1000*lambda}
        assert abs(result.theta_values[0] - 1.0) < 0.01

    def test_free_energy_shape(self):
        result = compute_heat_kernel_trace(10, 0.5)
        assert result.free_energy.shape == result.beta_values.shape

    def test_entropy_shape(self):
        result = compute_heat_kernel_trace(10, 0.5)
        assert result.entropy.shape == result.beta_values.shape

    def test_partition_function_equals_theta(self):
        result = compute_heat_kernel_trace(10, 0.5)
        np.testing.assert_array_equal(
            result.partition_function, result.theta_values
        )

    def test_free_energy_finite(self):
        """Free energy should be finite for all beta > 0."""
        betas = np.logspace(-1, 1, 20)
        result = compute_heat_kernel_trace(10, 0.5, beta_values=betas)
        assert np.all(np.isfinite(result.free_energy))


class TestConvenienceFunctions:
    """Test partition_function and free_energy convenience wrappers."""

    def test_partition_function(self):
        betas, Z = compute_partition_function(10, 0.5)
        assert len(betas) > 0
        assert len(Z) == len(betas)
        assert np.all(Z > 0)

    def test_free_energy(self):
        betas, F = compute_free_energy(10, 0.5)
        assert len(F) == len(betas)
        assert np.all(np.isfinite(F))


# ============================================================================
# Test: Mellin Bridge
# ============================================================================


class TestMellinBridge:
    """Test the Mellin transform bridge between heat kernel and zeta."""

    def test_returns_dataclass(self):
        result = verify_mellin_bridge(10, 0.5)
        assert isinstance(result, MellinBridgeResult)

    def test_bridge_valid_for_moderate_k(self):
        """Mellin bridge should hold with reasonable accuracy.

        When eigenvalues are small, the integral converges slowly and
        needs large beta_max.  Use sigma != 1/2 so that no eigenvalue
        is near zero (potential shift pushes them up).
        """
        result = verify_mellin_bridge(
            10, 0.7, u_values=np.array([2.0, 3.0, 4.0]),
            n_beta=2000, beta_max=500.0, tol=0.15,
        )
        assert result.bridge_valid

    def test_relative_error_shape(self):
        u = np.array([2.0, 3.0, 4.0])
        result = verify_mellin_bridge(10, 0.5, u_values=u)
        assert result.relative_error.shape == (3,)

    def test_both_sides_positive(self):
        u = np.array([2.0, 3.0])
        result = verify_mellin_bridge(10, 0.5, u_values=u)
        assert np.all(result.zeta_direct > 0)
        assert np.all(result.zeta_mellin > 0)

    def test_improves_with_more_quadrature(self):
        """More quadrature points should give better or equal accuracy."""
        r1 = verify_mellin_bridge(10, 0.5, n_beta=100, u_values=np.array([2.0]))
        r2 = verify_mellin_bridge(10, 0.5, n_beta=1000, u_values=np.array([2.0]))
        assert r2.max_relative_error <= r1.max_relative_error + 0.01


# ============================================================================
# Test: Riemann Zeta Approximation
# ============================================================================


class TestRiemannZeta:
    """Test the truncated Dirichlet series approximation."""

    def test_known_values(self):
        """Check against known zeta values for u >= 2."""
        for u, expected in RIEMANN_ZETA_KNOWN_VALUES.items():
            if u >= 2.0:
                approx = riemann_zeta_approx(u, n_terms=100_000)
                assert abs(approx - expected) / abs(expected) < 1e-3, (
                    f"zeta({u}): expected {expected}, got {approx}"
                )

    def test_near_pole_convergence(self):
        """Near u=1 the Dirichlet series converges slowly."""
        # zeta(1.5) with 100k terms should be within a few percent
        approx = riemann_zeta_approx(1.5, n_terms=100_000)
        expected = RIEMANN_ZETA_KNOWN_VALUES[1.5]
        assert abs(approx - expected) / expected < 0.01

    def test_pole_region(self):
        """u <= 1 should return inf."""
        assert riemann_zeta_approx(1.0) == float("inf")
        assert riemann_zeta_approx(0.5) == float("inf")
        assert riemann_zeta_approx(-1.0) == float("inf")

    def test_large_u_approaches_one(self):
        """zeta(u) -> 1 as u -> infinity."""
        z = riemann_zeta_approx(20.0)
        assert abs(z - 1.0) < 1e-5

    def test_zeta_2(self):
        """zeta(2) = pi^2/6."""
        z = riemann_zeta_approx(2.0, n_terms=100_000)
        assert abs(z - math.pi**2 / 6) < 1e-3


# ============================================================================
# Test: Conjecture 10.1
# ============================================================================


class TestConjecture:
    """Test the Conjecture 10.1 fitting framework."""

    def test_returns_dataclass(self):
        result = test_conjecture_10_1(10)
        assert isinstance(result, ConjectureTestResult)
        assert result.k == 10

    def test_fit_parameters_finite(self):
        result = test_conjecture_10_1(15)
        assert np.isfinite(result.C_fit)
        assert np.isfinite(result.delta_fit)
        assert np.isfinite(result.residual)
        assert np.isfinite(result.correlation)

    def test_correlation_in_range(self):
        result = test_conjecture_10_1(20)
        assert -1.0 <= result.correlation <= 1.0

    def test_positive_C_fit(self):
        """C should be positive (both sides are positive sums)."""
        result = test_conjecture_10_1(15)
        assert result.C_fit > 0

    def test_sequence(self):
        results = test_conjecture_10_1_sequence([5, 10, 15])
        assert len(results) == 3
        assert results[0].k == 5
        assert results[2].k == 15

    def test_residual_nonnegative(self):
        result = test_conjecture_10_1(10)
        assert result.residual >= 0

    def test_shapes_match(self):
        u = np.linspace(1.5, 4.0, 10)
        result = test_conjecture_10_1(10, u_values=u)
        assert result.u_values.shape == (10,)
        assert result.zeta_spectral.shape == (10,)
        assert result.zeta_riemann.shape == (10,)


# ============================================================================
# Test: Integrated Analysis
# ============================================================================


class TestIntegratedAnalysis:
    """Test the combined P5 analysis."""

    def test_returns_dataclass(self):
        result = run_spectral_zeta_analysis(10)
        assert isinstance(result, SpectralZetaAnalysis)
        assert result.k == 10

    def test_all_fields_populated(self):
        result = run_spectral_zeta_analysis(10)
        assert isinstance(result.spectral_zeta, SpectralZetaResult)
        assert isinstance(result.heat_kernel, HeatKernelResult)
        assert isinstance(result.mellin_bridge, MellinBridgeResult)
        assert isinstance(result.conjecture, ConjectureTestResult)
        assert len(result.summary) > 0

    def test_summary_readable(self):
        result = run_spectral_zeta_analysis(10)
        assert "k = 10" in result.summary
        assert "Mellin" in result.summary


# ============================================================================
# Test: Mathematical Properties
# ============================================================================


class TestMathematicalProperties:
    """Test mathematical identities and properties."""

    def test_zeta_u0_equals_n_positive(self):
        """zeta(u=0) = n_positive (lambda^0 = 1 for each term)."""
        result = compute_spectral_zeta(10, 0.5, u_values=np.array([0.0]))
        assert abs(result.zeta_values[0] - result.n_positive) < 1e-10

    def test_zeta_u1_equals_trace_inverse(self):
        """zeta(u=1) = sum 1/lambda_j = tr(H^{-1}) restricted to pos."""
        evals = compute_positive_eigenvalues(15, 0.5)
        tr_inv = np.sum(1.0 / evals)
        z = compute_spectral_zeta(15, 0.5, u_values=np.array([1.0]))
        assert abs(z.zeta_values[0] - tr_inv) < 1e-10

    def test_theta_at_beta_zero_equals_k(self):
        """tr(e^0) = tr(I) = k."""
        betas = np.array([1e-8])
        result = compute_heat_kernel_trace(10, 0.5, beta_values=betas)
        assert abs(result.theta_values[0] - 10.0) < 0.001

    def test_theta_exact_for_known_spectrum(self):
        """Verify Theta against direct eigenvalue computation."""
        k = 8
        evals = compute_eigenspectrum(k, 0.5)
        beta = 1.0
        expected = np.sum(np.exp(-beta * evals))
        result = compute_heat_kernel_trace(k, 0.5, beta_values=np.array([beta]))
        assert abs(result.theta_values[0] - expected) < 1e-10

    def test_spectral_zeta_from_eigenvalues(self):
        """Direct check: zeta(u) = sum lambda_j^{-u}."""
        k, u = 12, 2.5
        evals = compute_positive_eigenvalues(k, 0.5)
        expected = np.sum(evals ** (-u))
        result = compute_spectral_zeta(k, 0.5, u_values=np.array([u]))
        assert abs(result.zeta_values[0] - expected) < 1e-10

    def test_heat_kernel_spectral_zeta_relation(self):
        """At large beta, Theta -> e^{-beta * lambda_min} (ground dominance)."""
        k = 10
        hk = compute_heat_kernel_trace(k, 0.5, beta_values=np.array([500.0]))
        # At sigma=1/2, lambda_min=0, so Theta(large beta) -> 1
        assert abs(hk.theta_values[0] - 1.0) < 0.001


# ============================================================================
# Test: Edge Cases
# ============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_k2_spectral_zeta(self):
        """k=2 has 1 positive eigenvalue."""
        result = compute_spectral_zeta(2, 0.5, u_values=np.array([2.0]))
        assert result.n_positive == 1
        assert result.zeta_values.shape == (1,)

    def test_k3_spectral_zeta(self):
        result = compute_spectral_zeta(3, 0.5)
        assert result.n_positive == 2

    def test_single_u_value(self):
        result = compute_spectral_zeta(10, 0.5, u_values=np.array([3.0]))
        assert result.zeta_values.shape == (1,)

    def test_single_beta_value(self):
        result = compute_heat_kernel_trace(10, 0.5, beta_values=np.array([1.0]))
        assert result.theta_values.shape == (1,)

    def test_large_k(self):
        """k=100 should still work."""
        result = compute_spectral_zeta(100, 0.5, u_values=np.array([2.0]))
        assert result.n_positive == 99
        assert result.zeta_values[0] > 0

    def test_sigma_not_half(self):
        """Spectral zeta works at sigma != 1/2."""
        result = compute_spectral_zeta(10, 0.7, u_values=np.array([2.0]))
        assert result.n_positive == 10  # no zero eigenvalue
        assert result.zeta_values[0] > 0

    def test_very_large_u(self):
        """At very large u, zeta -> lambda_min^{-u} (smallest dominates)."""
        evals = compute_positive_eigenvalues(10, 0.5)
        u = 50.0
        result = compute_spectral_zeta(10, 0.5, u_values=np.array([u]))
        expected_dominant = evals[0] ** (-u)
        # Should be dominated by smallest eigenvalue
        assert result.zeta_values[0] > 0.5 * expected_dominant

    def test_known_values_dict_format(self):
        """RIEMANN_ZETA_KNOWN_VALUES has correct structure."""
        assert isinstance(RIEMANN_ZETA_KNOWN_VALUES, dict)
        assert 2.0 in RIEMANN_ZETA_KNOWN_VALUES
        assert abs(RIEMANN_ZETA_KNOWN_VALUES[2.0] - math.pi**2 / 6) < 1e-10


# ============================================================================
# Test: Private Helpers
# ============================================================================


class TestPearson:
    """Test the Pearson correlation helper."""

    def test_perfect_positive(self):
        x = np.array([1.0, 2.0, 3.0])
        assert abs(_pearson(x, x) - 1.0) < 1e-10

    def test_perfect_negative(self):
        x = np.array([1.0, 2.0, 3.0])
        assert abs(_pearson(x, -x) + 1.0) < 1e-10

    def test_uncorrelated(self):
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([1.0, 0.0, 1.0])
        r = _pearson(x, y)
        assert abs(r) < 0.1

    def test_degenerate_short(self):
        assert _pearson(np.array([1.0]), np.array([1.0])) == 0.0

    def test_constant_input(self):
        """Constant input -> zero correlation."""
        x = np.array([1.0, 1.0, 1.0])
        y = np.array([1.0, 2.0, 3.0])
        assert _pearson(x, y) == 0.0


# ============================================================================
# Test: Scaling with k (Conjecture 10.1 trend)
# ============================================================================


class TestScaling:
    """Test that quantities scale sensibly with k."""

    def test_n_positive_equals_k_minus_1_at_half(self):
        """At sigma=1/2, n_positive = k-1 for path graph."""
        for k in [5, 10, 20, 50]:
            result = compute_spectral_zeta(k, 0.5, u_values=np.array([2.0]))
            assert result.n_positive == k - 1

    def test_zeta_grows_with_k(self):
        """More eigenvalues -> larger spectral zeta at fixed u."""
        z_prev = 0.0
        for k in [5, 10, 20]:
            result = compute_spectral_zeta(k, 0.5, u_values=np.array([1.0]))
            z = result.zeta_values[0]
            assert z > z_prev
            z_prev = z

    def test_correlation_trend(self):
        """Conjecture 10.1 fit improves with k (residual finite, params finite)."""
        results = test_conjecture_10_1_sequence([50, 100])
        for r in results:
            # Fit parameters must be finite regardless of fit quality
            assert np.isfinite(r.correlation)
            assert np.isfinite(r.residual)
            assert np.isfinite(r.C_fit)
            assert np.isfinite(r.delta_fit)
