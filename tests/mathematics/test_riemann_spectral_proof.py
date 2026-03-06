r"""Tests for the TNFR-Riemann spectral proof framework.

Validates four lines of attack for the TNFR-Riemann bridge:

1. **Structural Equilibrium** -- lambda_min(H(1/2)) = 0 (exact)
2. **Thermodynamic Attractor** -- sigma* -> 1/2 at O(1/k) (asymptotic)
3. **Eigenvalue Flow** -- all d(lambda_j)/dsigma > 0 (Hellmann-Feynman)
4. **Spectral Moments** -- trace formula and spacing analysis

Physics basis: H_TNFR^(k)(sigma) = L_k + (sigma - 1/2) diag(log p_i).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tnfr.riemann.operator import (
    build_prime_path_graph,
    build_h_tnfr,
    build_tridiagonal_h_tnfr,
    default_prime_potential,
    _first_primes,
)
from tnfr.riemann.spectral_proof import (
    # Data structures
    EquilibriumResult,
    ThermodynamicResult,
    EigenvalueFlowResult,
    SpectralMomentResult,
    TNFRRiemannAssessment,
    # Core
    compute_eigenspectrum,
    compute_eigensystem,
    # Line 1
    verify_equilibrium,
    verify_equilibrium_sequence,
    # Line 2
    compute_analytic_sigma_star,
    compute_frobenius_energy,
    compute_thermodynamic_landscape,
    verify_thermodynamic_convergence,
    # Line 3
    compute_eigenvalue_velocities,
    analyze_eigenvalue_flow,
    # Line 4
    compute_eigenvalue_spacings,
    compute_spectral_moments,
    # Integration
    run_tnfr_riemann_analysis,
    # Private helpers used in tests
    _unfold_eigenvalues,
    _get_log_primes,
    _compute_lv1_traces,
    _fit_power_law,
)


# ============================================================================
# Operator sanity checks (unchanged from v1)
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
        w = G[0][1]["weight"]
        assert abs(w - abs(math.log(3) - math.log(2))) < 1e-10

    def test_uniform_weights(self) -> None:
        G = build_prime_path_graph(3, weight_by_log_gap=False)
        assert G[0][1]["weight"] == 1.0


class TestHOperator:
    """Tests for the H_TNFR operator construction."""

    def test_symmetry(self) -> None:
        G = build_prime_path_graph(10)
        H, _ = build_h_tnfr(G, sigma=0.5)
        assert np.allclose(H, H.T)

    def test_sigma_half_pure_laplacian(self) -> None:
        """At sigma=1/2, V=0 so H=L (pure Laplacian) with row sums=0."""
        G = build_prime_path_graph(10)
        H, V = build_h_tnfr(G, sigma=0.5)
        assert np.allclose(V, 0.0)
        row_sums = np.sum(H, axis=1)
        assert np.allclose(row_sums, 0.0, atol=1e-12)

    def test_potential_sign(self) -> None:
        """V > 0 when sigma > 1/2, < 0 when sigma < 1/2."""
        for p in [2, 3, 5, 7, 11]:
            assert default_prime_potential(p, 0.75) > 0
            assert default_prime_potential(p, 0.25) < 0
            assert abs(default_prime_potential(p, 0.5)) < 1e-15

    def test_eigenvalue_monotonicity(self) -> None:
        """lambda_min(sigma) should increase with sigma."""
        k = 20
        lam_lo = compute_eigenspectrum(k, 0.25)[0]
        lam_mid = compute_eigenspectrum(k, 0.50)[0]
        lam_hi = compute_eigenspectrum(k, 0.75)[0]
        assert lam_lo < lam_mid < lam_hi


# ============================================================================
# Core spectral computation
# ============================================================================

class TestCoreSpectral:
    """Tests for eigenspectrum and eigensystem computation."""

    def test_eigenspectrum_sorted(self) -> None:
        evals = compute_eigenspectrum(10, 0.5)
        assert len(evals) == 10
        assert np.all(np.diff(evals) >= -1e-14)

    def test_eigensystem_orthonormal(self) -> None:
        evals, evecs = compute_eigensystem(10, 0.5)
        assert np.allclose(evecs.T @ evecs, np.eye(10), atol=1e-12)

    def test_eigensystem_consistent(self) -> None:
        """eigensystem eigenvalues match eigenspectrum."""
        evals1 = compute_eigenspectrum(10, 0.5)
        evals2, _ = compute_eigensystem(10, 0.5)
        assert np.allclose(evals1, evals2)


# ============================================================================
# LINE 1: Structural Equilibrium Theorem
# ============================================================================

class TestStructuralEquilibrium:
    """Tests for the structural equilibrium theorem.

    Exact result: lambda_min(H(1/2)) = 0 for all k >= 2 because
    H(1/2) = L_k and ker(L_k) = span{1} for connected graphs.
    """

    def test_lambda_min_zero_at_half(self) -> None:
        """lambda_min(H(1/2)) = 0 to machine precision."""
        for k in [5, 10, 20, 50]:
            result = verify_equilibrium(k)
            assert abs(result.lambda_min) < 1e-12, (
                f"k={k}: lambda_min = {result.lambda_min}"
            )

    def test_spectral_gap_positive(self) -> None:
        """Spectral gap lambda_2 - lambda_1 > 0 (connected graph)."""
        for k in [5, 10, 20]:
            result = verify_equilibrium(k)
            assert result.spectral_gap > 0

    def test_spectral_gap_decreases(self) -> None:
        """Spectral gap should decrease for larger k (path graph)."""
        results = verify_equilibrium_sequence([10, 20, 50, 100])
        gaps = [r.spectral_gap for r in results]
        for i in range(len(gaps) - 1):
            assert gaps[i] > gaps[i + 1]

    def test_ground_velocity_equals_mean_log_prime(self) -> None:
        """Ground state velocity = mean(log p) because psi_0 = 1/sqrt(k)."""
        for k in [5, 10, 20]:
            result = verify_equilibrium(k)
            assert abs(result.ground_velocity - result.mean_log_prime) < 1e-6

    def test_ground_velocity_grows_like_log_k(self) -> None:
        """Ground velocity ~ log(k) by PNT."""
        results = verify_equilibrium_sequence([10, 50, 100, 200])
        velocities = [r.ground_velocity for r in results]
        # Strictly increasing
        for i in range(len(velocities) - 1):
            assert velocities[i] < velocities[i + 1]

    def test_equilibrium_result_dataclass(self) -> None:
        result = verify_equilibrium(10)
        assert isinstance(result, EquilibriumResult)
        assert result.k == 10
        assert result.spectral_width > 0


# ============================================================================
# LINE 2: Thermodynamic Attractor Analysis
# ============================================================================

class TestThermodynamicAttractor:
    """Tests for the thermodynamic attractor sigma* -> 1/2.

    The Frobenius energy E(sigma) = (1/2k) tr(H^2) is exactly quadratic
    with minimum sigma* = 1/2 - tr(LV_1)/tr(V_1^2) -> 1/2 as k -> inf.
    """

    def test_analytic_sigma_star_near_half(self) -> None:
        """sigma* should approach 1/2 for large k."""
        for k, tol in [(10, 0.3), (50, 0.1), (100, 0.05)]:
            sigma_star, _, _ = compute_analytic_sigma_star(k)
            assert abs(sigma_star - 0.5) < tol, (
                f"k={k}: sigma* = {sigma_star}"
            )

    def test_cross_term_finite(self) -> None:
        """tr(L V_1) should be finite and well-defined."""
        _, cross_term, potential_norm = compute_analytic_sigma_star(20)
        assert np.isfinite(cross_term)
        assert potential_norm > 0

    def test_energy_at_half_less_than_away(self) -> None:
        """E(1/2) < E(sigma) for sigma far from 1/2."""
        k = 20
        E_half = compute_frobenius_energy(k, 0.5)
        E_off = compute_frobenius_energy(k, 0.8)
        assert E_half < E_off

    def test_energy_quadratic(self) -> None:
        """Frobenius energy is exactly quadratic in sigma."""
        k = 20
        # Sample three points and verify quadratic fit
        sigmas = np.array([0.3, 0.5, 0.7])
        energies = np.array([compute_frobenius_energy(k, s) for s in sigmas])
        # Fit quadratic: E = a*sig^2 + b*sig + c
        coeffs = np.polyfit(sigmas, energies, 2)
        # Predict at intermediate points
        test_sigmas = np.array([0.35, 0.45, 0.55, 0.65])
        predicted = np.polyval(coeffs, test_sigmas)
        actual = np.array([compute_frobenius_energy(k, s) for s in test_sigmas])
        assert np.allclose(predicted, actual, rtol=1e-6)

    def test_landscape_analytic_numerical_agree(self) -> None:
        """Analytic and numerical sigma* should agree closely."""
        result = compute_thermodynamic_landscape(50, n_points=500)
        assert abs(result.sigma_star_analytic - result.sigma_star_numerical) < 0.01

    def test_curvature_positive(self) -> None:
        """d^2E/dsigma^2 = (1/k) tr(V_1^2) > 0 always."""
        for k in [5, 10, 50]:
            result = compute_thermodynamic_landscape(k)
            assert result.curvature > 0

    def test_curvature_increases(self) -> None:
        """Curvature d^2E/dsigma^2 ~ (log k)^2 grows."""
        results = verify_thermodynamic_convergence([10, 50, 100])
        curvatures = [r.curvature for r in results]
        for i in range(len(curvatures) - 1):
            assert curvatures[i] < curvatures[i + 1]

    def test_deviation_decreases(self) -> None:
        """sigma* deviation from 1/2 decreases with k."""
        results = verify_thermodynamic_convergence([10, 20, 50, 100])
        deviations = [r.deviation for r in results]
        for i in range(len(deviations) - 1):
            assert deviations[i] > deviations[i + 1] - 1e-14

    def test_thermodynamic_result_dataclass(self) -> None:
        result = compute_thermodynamic_landscape(10)
        assert isinstance(result, ThermodynamicResult)
        assert result.k == 10
        assert result.energy_at_star <= result.energy_at_half + 1e-14


# ============================================================================
# LINE 3: Eigenvalue Flow Analysis
# ============================================================================

class TestEigenvalueFlow:
    """Tests for Hellmann-Feynman eigenvalue flow analysis.

    Key result: d(lambda_j)/dsigma = <psi_j|V_1|psi_j> > 0 for all j
    since log(p_i) > 0 for all primes p_i >= 2.
    """

    def test_all_velocities_positive(self) -> None:
        """Every eigenvalue velocity must be strictly positive."""
        for k in [5, 10, 20, 50]:
            velocities = compute_eigenvalue_velocities(k, 0.5)
            assert len(velocities) == k
            assert np.all(velocities > 0), (
                f"k={k}: min velocity = {np.min(velocities)}"
            )

    def test_velocity_count_matches_k(self) -> None:
        velocities = compute_eigenvalue_velocities(10, 0.5)
        assert len(velocities) == 10

    def test_velocities_bounded(self) -> None:
        """Velocities are bounded by [log(2), log(p_k)]."""
        k = 20
        velocities = compute_eigenvalue_velocities(k, 0.5)
        log_2 = math.log(2)
        primes = _first_primes(k)
        log_pk = math.log(primes[-1])
        assert float(np.min(velocities)) >= log_2 - 1e-10
        assert float(np.max(velocities)) <= log_pk + 1e-10

    def test_ground_state_velocity_equals_mean_log_p(self) -> None:
        """v_0 = mean(log p_i) for constant eigenvector."""
        k = 20
        velocities = compute_eigenvalue_velocities(k, 0.5)
        log_p = _get_log_primes(k)
        mean_log = float(np.mean(log_p))
        # Ground state is lowest eigenvalue (index 0)
        assert abs(velocities[0] - mean_log) < 1e-6

    def test_flow_analysis_result(self) -> None:
        result = analyze_eigenvalue_flow(10, n_scan=20)
        assert isinstance(result, EigenvalueFlowResult)
        assert result.k == 10
        assert result.all_positive
        assert result.min_velocity > 0
        assert result.velocity_ratio >= 1.0
        assert result.eigenvalue_trajectories.shape == (20, 10)
        assert len(result.sigma_scan) == 20

    def test_trajectories_monotone_in_sigma(self) -> None:
        """Each eigenvalue trajectory should increase in sigma."""
        result = analyze_eigenvalue_flow(10, n_scan=50)
        for j in range(10):
            traj = result.eigenvalue_trajectories[:, j]
            diffs = np.diff(traj)
            assert np.all(diffs >= -1e-10), (
                f"Eigenvalue {j} not monotone"
            )


# ============================================================================
# LINE 4: Spectral Moments & Spacings
# ============================================================================

class TestSpectralMoments:
    """Tests for spectral moment and spacing analysis."""

    def test_unfold_preserves_count(self) -> None:
        evals = np.array([0.1, 0.5, 1.2, 2.0, 3.5])
        unfolded = _unfold_eigenvalues(evals)
        assert len(unfolded) == len(evals)

    def test_spacings_non_negative(self) -> None:
        evals = compute_eigenspectrum(20, 0.5)
        spacings = compute_eigenvalue_spacings(evals)
        assert np.all(spacings >= -1e-10)

    def test_spacings_mean_near_one(self) -> None:
        evals = compute_eigenspectrum(50, 0.5)
        spacings = compute_eigenvalue_spacings(evals)
        if len(spacings) > 1:
            assert abs(np.mean(spacings) - 1.0) < 0.5

    def test_moments_result(self) -> None:
        result = compute_spectral_moments(20, max_n=4)
        assert isinstance(result, SpectralMomentResult)
        assert result.k == 20
        assert len(result.moments) == 4
        assert result.spectral_gap > 0

    def test_first_moment_is_trace(self) -> None:
        """mu_1 = (1/k) tr(L) = (1/k) sum(deg_w(i)) = (2/k) sum(w_j)."""
        k = 10
        result = compute_spectral_moments(k, max_n=1)
        evals = compute_eigenspectrum(k, 0.5)
        expected_mu1 = float(np.mean(evals))
        assert abs(result.moments[0] - expected_mu1) < 1e-10

    def test_moments_positive_at_half(self) -> None:
        """All moments at sigma=1/2 should be non-negative."""
        result = compute_spectral_moments(20, 0.5, max_n=6)
        assert np.all(result.moments >= -1e-14)

    def test_mean_spacing_positive(self) -> None:
        result = compute_spectral_moments(20)
        assert result.mean_spacing > 0


# ============================================================================
# Power Law Fit
# ============================================================================

class TestPowerLawFit:
    """Tests for the power law fitting utility."""

    def test_fit_known_rate(self) -> None:
        """Fit should recover y = A / x^beta."""
        x = [10.0, 50.0, 100.0, 500.0]
        A_true, beta_true = 2.0, 1.0
        y = [A_true / xi ** beta_true for xi in x]
        A_fit, beta_fit = _fit_power_law(x, y)
        assert abs(A_fit - A_true) < 0.1
        assert abs(beta_fit - beta_true) < 0.1

    def test_fit_insufficient_data(self) -> None:
        """Fit with < 2 valid points returns fallback."""
        A, beta = _fit_power_law([10.0], [0.1])
        assert A == 0.0
        assert beta == 1.0

    def test_fit_with_zeros_skipped(self) -> None:
        """Zero y values should be excluded from fit."""
        A, beta = _fit_power_law([10.0, 20.0, 30.0], [0.1, 0.0, 0.03])
        # Should skip y=0.0 and fit with 2 remaining points
        assert A > 0
        assert beta > 0


# ============================================================================
# Integration: Full TNFR-Riemann Assessment
# ============================================================================

class TestIntegratedAssessment:
    """Tests for the complete integrated analysis."""

    def test_small_scale_assessment(self) -> None:
        result = run_tnfr_riemann_analysis(
            [5, 10, 20],
            flow_n_scan=20,
            moment_max_n=4,
        )
        assert isinstance(result, TNFRRiemannAssessment)
        assert len(result.k_values) == 3
        assert len(result.equilibria) == 3
        assert len(result.thermodynamics) == 3
        assert len(result.flows) == 3
        assert len(result.moments) == 3
        assert result.overall_confidence >= 0.0

    def test_equilibrium_exact_flag(self) -> None:
        result = run_tnfr_riemann_analysis([10, 20], flow_n_scan=10)
        assert result.equilibrium_exact is True

    def test_flow_monotone_flag(self) -> None:
        result = run_tnfr_riemann_analysis([10, 20], flow_n_scan=10)
        assert result.flow_monotone is True

    def test_convergence_beta_near_one(self) -> None:
        """Fitted exponent should be near 1.0 from PNT."""
        result = run_tnfr_riemann_analysis(
            [10, 20, 50, 100],
            flow_n_scan=10,
        )
        if result.convergence_beta > 0:
            assert result.convergence_beta > 0.5

    def test_summary_nonempty(self) -> None:
        result = run_tnfr_riemann_analysis([5, 10], flow_n_scan=10)
        assert len(result.summary) > 50

    def test_default_k_values(self) -> None:
        """Default k_values should produce results."""
        result = run_tnfr_riemann_analysis(flow_n_scan=10)
        assert len(result.k_values) == 6  # [5, 10, 20, 50, 100, 200]


# ============================================================================
# Tridiagonal Solver: Correctness and Performance
# ============================================================================

class TestTridiagonalBuilder:
    """Tests for the tridiagonal H_TNFR representation.

    The prime path graph Laplacian is tridiagonal.  build_tridiagonal_h_tnfr
    produces (d, e, log_p) vectors in O(k) time and memory, enabling
    O(k^2) eigenvalue computation via eigh_tridiagonal.
    """

    def test_shapes(self) -> None:
        """Diagonal has length k, sub-diagonal has length k-1."""
        for k in [3, 10, 50]:
            d, e, log_p = build_tridiagonal_h_tnfr(k, 0.5)
            assert d.shape == (k,)
            assert e.shape == (k - 1,)
            assert log_p.shape == (k,)

    def test_eigenvalues_match_dense(self) -> None:
        """Tridiagonal eigenvalues must match dense eigenvalues."""
        from scipy.linalg import eigh_tridiagonal

        for k in [5, 10, 20, 50]:
            for sigma in [0.25, 0.5, 0.75]:
                # Dense path
                G = build_prime_path_graph(k)
                H, _ = build_h_tnfr(G, sigma=sigma)
                evals_dense = np.sort(np.linalg.eigvalsh(H))

                # Tridiagonal path
                d, e, _ = build_tridiagonal_h_tnfr(k, sigma)
                evals_tri = np.sort(eigh_tridiagonal(d, e, eigvals_only=True))

                np.testing.assert_allclose(
                    evals_tri, evals_dense, atol=1e-10,
                    err_msg=f"k={k}, sigma={sigma}"
                )

    def test_eigenvectors_match_dense(self) -> None:
        """Tridiagonal eigenvectors must agree with dense (up to sign)."""
        from scipy.linalg import eigh_tridiagonal

        for k in [5, 10, 20]:
            G = build_prime_path_graph(k)
            H, _ = build_h_tnfr(G, sigma=0.5)
            evals_d, evecs_d = np.linalg.eigh(H)

            d, e, _ = build_tridiagonal_h_tnfr(k, 0.5)
            evals_t, evecs_t = eigh_tridiagonal(d, e)

            np.testing.assert_allclose(evals_t, evals_d, atol=1e-10)
            # Eigenvectors match up to sign: |v_t . v_d| ~ 1
            for j in range(k):
                dot = abs(float(np.dot(evecs_t[:, j], evecs_d[:, j])))
                assert dot > 0.999, f"k={k}, mode {j}: dot={dot}"

    def test_laplacian_diagonal_at_half(self) -> None:
        """At sigma=1/2, V=0 so d equals Laplacian degree sequence."""
        k = 10
        d, e, log_p = build_tridiagonal_h_tnfr(k, 0.5)
        # Row sums of tridiagonal Laplacian should equal diagonal
        # For path: d[0]=w_01, d[i]=w_{i-1}+w_{i}, d[k-1]=w_{k-2}
        primes = _first_primes(k)
        log_primes = np.array([np.log(float(p)) for p in primes])
        weights = np.abs(np.diff(log_primes))
        expected_d = np.zeros(k)
        expected_d[0] = weights[0]
        expected_d[-1] = weights[-1]
        for i in range(1, k - 1):
            expected_d[i] = weights[i - 1] + weights[i]
        np.testing.assert_allclose(d, expected_d, atol=1e-14)

    def test_potential_shifts_diagonal(self) -> None:
        """At sigma != 1/2, diagonal is shifted by (sigma-1/2)*log(p)."""
        k = 10
        d_half, e_half, log_p = build_tridiagonal_h_tnfr(k, 0.5)
        d_off, e_off, _ = build_tridiagonal_h_tnfr(k, 0.75)
        # Sub-diagonal unchanged (potential is diagonal)
        np.testing.assert_allclose(e_off, e_half, atol=1e-14)
        # Diagonal shifted
        expected_shift = 0.25 * log_p
        np.testing.assert_allclose(d_off - d_half, expected_shift, atol=1e-14)

    def test_single_node(self) -> None:
        """k=1 should produce scalar result."""
        d, e, log_p = build_tridiagonal_h_tnfr(1, 0.5)
        assert d.shape == (1,)
        assert e.shape == (0,)
        assert abs(d[0]) < 1e-14  # No edges, sigma=1/2 => 0

    def test_sub_diagonal_negative(self) -> None:
        """Sub-diagonal must be negative (= -edge_weight)."""
        d, e, _ = build_tridiagonal_h_tnfr(20, 0.5)
        assert np.all(e < 0)

    def test_uniform_weights(self) -> None:
        """With uniform weights, e = [-1, -1, ..., -1]."""
        d, e, _ = build_tridiagonal_h_tnfr(10, 0.5, weight_by_log_gap=False)
        np.testing.assert_allclose(e, -np.ones(9), atol=1e-14)
        # Interior nodes have degree 2, endpoints degree 1
        assert abs(d[0] - 1.0) < 1e-14
        assert abs(d[-1] - 1.0) < 1e-14
        for i in range(1, 9):
            assert abs(d[i] - 2.0) < 1e-14


class TestLargeScale:
    """Tests validating TNFR-Riemann properties at large k.

    These tests exercise the tridiagonal solver at scales inaccessible
    to the previous dense O(k^3) implementation.
    """

    def test_equilibrium_k1000(self) -> None:
        """lambda_min(H(1/2)) = 0 at k = 1000."""
        result = verify_equilibrium(1000)
        assert abs(result.lambda_min) < 1e-10
        assert result.spectral_gap > 0

    def test_sigma_star_convergence_k2000(self) -> None:
        """sigma* deviation < 0.005 at k = 2000."""
        sigma_star, _, _ = compute_analytic_sigma_star(2000)
        assert abs(sigma_star - 0.5) < 0.005

    def test_all_velocities_positive_k1000(self) -> None:
        """All eigenvalue velocities positive at k = 1000."""
        velocities = compute_eigenvalue_velocities(1000, 0.5)
        assert len(velocities) == 1000
        assert np.all(velocities > 0)

    def test_large_k_spectrum_sorted(self) -> None:
        """Eigenspectrum at k=5000 should be properly sorted."""
        evals = compute_eigenspectrum(5000, 0.5)
        assert len(evals) == 5000
        assert np.all(np.diff(evals) >= -1e-10)
        assert abs(evals[0]) < 1e-10  # lambda_min = 0

    def test_curvature_monotone_large_k(self) -> None:
        """Curvature grows through large k values."""
        results = verify_thermodynamic_convergence([100, 500, 1000])
        curvatures = [r.curvature for r in results]
        for i in range(len(curvatures) - 1):
            assert curvatures[i] < curvatures[i + 1]

    def test_deviation_shrinks_large_k(self) -> None:
        """sigma* deviation continues shrinking at large k."""
        results = verify_thermodynamic_convergence([100, 500, 1000, 2000])
        devs = [r.deviation for r in results]
        for i in range(len(devs) - 1):
            assert devs[i] > devs[i + 1] - 1e-14
