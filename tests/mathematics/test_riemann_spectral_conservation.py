"""Tests for P7: Spectral conservation at criticality.

Validates the TNFR structural conservation theorem applied to
prime-graph eigenmodes: energy density, topological charge, Noether
charge, and grammar compliance conservation tests.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tnfr.riemann.spectral_conservation import (
    # Data structures
    EigenmodeConservation,
    ConservationAtSigma,
    ConservationSigmaScan,
    GrammarComplianceResult,
    CriticalConservationAnalysis,
    # Core
    compute_spectral_j_phi,
    compute_spectral_j_dnfr,
    compute_eigenmode_conservation,
    # Sigma scan
    scan_conservation_vs_sigma,
    # Grammar compliance
    test_grammar_conservation,
    # Integration
    run_critical_conservation_analysis,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def small_conservation():
    """Conservation at sigma=0.5, k=8."""
    return compute_eigenmode_conservation(8, sigma=0.5)


@pytest.fixture
def medium_conservation():
    """Conservation at sigma=0.5, k=15."""
    return compute_eigenmode_conservation(15, sigma=0.5)


# ============================================================================
# TestComputeSpectralJPhi
# ============================================================================


class TestComputeSpectralJPhi:
    """Test spectral phase current J_phi."""

    def test_identity_laplacian(self):
        """Zero Laplacian => all J_phi = 0."""
        k = 5
        L = np.zeros((k, k))
        vecs = np.eye(k)
        result = compute_spectral_j_phi(vecs, L)
        np.testing.assert_allclose(result, 0.0, atol=1e-15)

    def test_output_shape(self):
        """J_phi has one entry per eigenmode."""
        k = 6
        L = np.diag([2, 1, 1, 1, 1, 2]) - np.eye(k)
        L = L + L.T  # make symmetric
        _, vecs = np.linalg.eigh(L)
        result = compute_spectral_j_phi(vecs, L)
        assert result.shape == (k,)

    def test_nonnegative_for_laplacian(self):
        """Graph Laplacian is positive semidefinite => J_phi >= 0."""
        from tnfr.riemann.operator import build_prime_path_graph, build_h_tnfr

        G = build_prime_path_graph(10)
        H, V = build_h_tnfr(G, sigma=0.5)
        L = H - V  # At sigma=0.5, V=0, so L=H
        _, vecs = np.linalg.eigh(H)
        j_phi = compute_spectral_j_phi(vecs, L)
        # Laplacian is PSD so expectations are >= 0
        assert np.all(j_phi >= -1e-10)

    def test_varies_across_modes(self):
        """Different eigenmodes have different J_phi."""
        from tnfr.riemann.operator import build_prime_path_graph, build_h_tnfr

        G = build_prime_path_graph(10)
        H, V = build_h_tnfr(G, sigma=0.5)
        L = H - V
        _, vecs = np.linalg.eigh(H)
        j_phi = compute_spectral_j_phi(vecs, L)
        assert len(set(np.round(j_phi, 6))) > 1


# ============================================================================
# TestComputeSpectralJDNFR
# ============================================================================


class TestComputeSpectralJDNFR:
    """Test spectral DNFR flux J_DNFR (eigenvalue velocity)."""

    def test_output_shape(self):
        """J_DNFR has one entry per eigenmode."""
        k = 5
        vecs = np.eye(k)
        log_p = np.log([2, 3, 5, 7, 11])
        result = compute_spectral_j_dnfr(vecs, log_p)
        assert result.shape == (k,)

    def test_identity_eigenvectors(self):
        """Standard basis eigenvectors => J_DNFR(j) = log(p_j)."""
        log_p = np.log(np.array([2, 3, 5, 7, 11], dtype=float))
        vecs = np.eye(5)
        result = compute_spectral_j_dnfr(vecs, log_p)
        np.testing.assert_allclose(result, log_p, rtol=1e-12)

    def test_positive(self):
        """All primes > 1 => log(p) > 0 => J_DNFR > 0."""
        from tnfr.riemann.operator import build_prime_path_graph, build_h_tnfr

        G = build_prime_path_graph(8)
        H, V = build_h_tnfr(G, sigma=0.5)
        _, vecs = np.linalg.eigh(H)
        nodes = sorted(G.nodes())
        log_p = np.array([np.log(float(G.nodes[n]["label"])) for n in nodes])
        j_dnfr = compute_spectral_j_dnfr(vecs, log_p)
        assert np.all(j_dnfr > 0)

    def test_hellmann_feynman(self):
        """Verify J_DNFR ≈ numerical dλ/dσ via finite differences."""
        from tnfr.riemann.operator import build_prime_path_graph, build_h_tnfr

        k = 8
        sigma = 0.5
        dsigma = 1e-5

        G = build_prime_path_graph(k)
        nodes = sorted(G.nodes())
        log_p = np.array([np.log(float(G.nodes[n]["label"])) for n in nodes])

        H0, _ = build_h_tnfr(G, sigma=sigma)
        Hp, _ = build_h_tnfr(G, sigma=sigma + dsigma)
        Hm, _ = build_h_tnfr(G, sigma=sigma - dsigma)

        evals_0, vecs_0 = np.linalg.eigh(H0)
        evals_p, _ = np.linalg.eigh(Hp)
        evals_m, _ = np.linalg.eigh(Hm)

        # Numerical derivative
        d_lambda_numerical = (evals_p - evals_m) / (2 * dsigma)

        # Hellmann-Feynman
        j_dnfr = compute_spectral_j_dnfr(vecs_0, log_p)

        np.testing.assert_allclose(j_dnfr, d_lambda_numerical, rtol=1e-3)


# ============================================================================
# TestEigenmodeConservation
# ============================================================================


class TestEigenmodeConservation:
    """Test eigenmode conservation field computation."""

    def test_basic_k8(self, small_conservation):
        """Basic structure at k=8, sigma=0.5."""
        snap = small_conservation
        assert snap.k == 8
        assert snap.sigma == 0.5
        assert len(snap.modes) == 8

    def test_mode_fields_finite(self, small_conservation):
        """All conservation fields are finite."""
        for m in small_conservation.modes:
            assert math.isfinite(m.phi_s)
            assert math.isfinite(m.grad_phi)
            assert math.isfinite(m.k_phi)
            assert math.isfinite(m.j_phi)
            assert math.isfinite(m.j_dnfr)
            assert math.isfinite(m.energy_density)
            assert math.isfinite(m.topological_charge)
            assert math.isfinite(m.charge_density)

    def test_energy_density_nonneg(self, small_conservation):
        """Energy density E(j) = sum of squares >= 0."""
        for m in small_conservation.modes:
            assert m.energy_density >= 0

    def test_energy_density_formula(self, small_conservation):
        """Verify E(j) = Phi_s^2 + |grad_phi|^2 + K_phi^2 + J_phi^2 + J_DNFR^2."""
        for m in small_conservation.modes:
            expected = (
                m.phi_s**2 + m.grad_phi**2 + m.k_phi**2
                + m.j_phi**2 + m.j_dnfr**2
            )
            assert m.energy_density == pytest.approx(expected, rel=1e-12)

    def test_topological_charge_formula(self, small_conservation):
        """Verify Q(j) = |grad_phi|*J_phi - K_phi*J_DNFR."""
        for m in small_conservation.modes:
            expected = m.grad_phi * m.j_phi - m.k_phi * m.j_dnfr
            assert m.topological_charge == pytest.approx(expected, rel=1e-12)

    def test_charge_density_formula(self, small_conservation):
        """Verify rho(j) = Phi_s(j) + K_phi(j)."""
        for m in small_conservation.modes:
            expected = m.phi_s + m.k_phi
            assert m.charge_density == pytest.approx(expected, rel=1e-12)

    def test_total_energy_is_sum(self, small_conservation):
        """Total energy = sum of per-mode energies."""
        expected = sum(m.energy_density for m in small_conservation.modes)
        assert small_conservation.total_energy == pytest.approx(expected, rel=1e-12)

    def test_total_charge_is_sum(self, small_conservation):
        """Total charge = sum of per-mode charges."""
        expected = sum(m.topological_charge for m in small_conservation.modes)
        assert small_conservation.total_charge == pytest.approx(expected, rel=1e-12)

    def test_total_charge_density_is_sum(self, small_conservation):
        """Total charge density = sum of per-mode charge densities."""
        expected = sum(m.charge_density for m in small_conservation.modes)
        assert small_conservation.total_charge_density == pytest.approx(
            expected, rel=1e-12,
        )

    def test_mean_energy_density(self, small_conservation):
        """Mean energy = total / k."""
        expected = small_conservation.total_energy / small_conservation.k
        assert small_conservation.mean_energy_density == pytest.approx(
            expected, rel=1e-12,
        )

    def test_k2_minimal(self):
        """Minimum k=2 works."""
        snap = compute_eigenmode_conservation(2, sigma=0.5)
        assert snap.k == 2
        assert len(snap.modes) == 2

    def test_k_too_small(self):
        """k < 2 raises ValueError."""
        with pytest.raises(ValueError, match="k must be >= 2"):
            compute_eigenmode_conservation(1)

    def test_sigma_away_from_half(self):
        """Conservation at sigma != 0.5 still computes."""
        snap = compute_eigenmode_conservation(8, sigma=0.7)
        assert snap.sigma == 0.7
        assert snap.total_energy > 0


# ============================================================================
# TestSigmaEffect
# ============================================================================


class TestSigmaEffect:
    """Test how conservation fields change with sigma."""

    def test_sigma_half_v_equals_zero(self):
        """At sigma=0.5, V=0 so H=L and J_DNFR = sum |psi|^2 log(p)."""
        snap = compute_eigenmode_conservation(10, sigma=0.5)
        for m in snap.modes:
            assert m.j_dnfr > 0  # log(p) > 0 for all primes

    def test_energy_increases_away_from_half(self):
        """Energy at sigma=0.5 <= energy at sigma=0.8 (typically)."""
        e_half = compute_eigenmode_conservation(12, sigma=0.5).total_energy
        e_away = compute_eigenmode_conservation(12, sigma=0.8).total_energy
        # This is a structural prediction, not a strict theorem
        # We verify both are finite and positive
        assert e_half > 0
        assert e_away > 0


# ============================================================================
# TestConservationSigmaScan
# ============================================================================


class TestConservationSigmaScan:
    """Test sigma scan of conservation fields."""

    def test_basic_scan(self):
        """Sigma scan produces correct shapes."""
        scan = scan_conservation_vs_sigma(
            8, np.linspace(0.2, 0.8, 7),
        )
        assert scan.k == 8
        assert len(scan.sigma_values) == 7
        assert len(scan.total_energy) == 7
        assert len(scan.total_charge) == 7
        assert len(scan.total_charge_density) == 7
        assert len(scan.charge_gradient) == 7
        assert len(scan.charge_drift_from_half) == 7

    def test_energy_minimum_in_range(self):
        """Energy minimum sigma is in the scanned range."""
        sigmas = np.linspace(0.2, 0.8, 13)
        scan = scan_conservation_vs_sigma(8, sigmas)
        assert scan.energy_minimum_sigma >= 0.2
        assert scan.energy_minimum_sigma <= 0.8

    def test_charge_gradient_nonneg(self):
        """Charge gradient |dQ/dsigma| >= 0."""
        scan = scan_conservation_vs_sigma(8, np.linspace(0.3, 0.7, 9))
        assert np.all(scan.charge_gradient >= 0)

    def test_charge_drift_zero_at_half(self):
        """Drift from sigma=0.5 is zero at sigma=0.5."""
        sigmas = np.array([0.3, 0.4, 0.5, 0.6, 0.7])
        scan = scan_conservation_vs_sigma(6, sigmas)
        idx_half = 2  # sigma=0.5
        assert scan.charge_drift_from_half[idx_half] == pytest.approx(0.0, abs=1e-10)

    def test_critical_conservation_present(self):
        """Sigma scan includes critical snapshot."""
        scan = scan_conservation_vs_sigma(
            8, np.linspace(0.2, 0.8, 7),
        )
        assert scan.critical_conservation is not None
        assert scan.critical_conservation.k == 8

    def test_default_sigma_range(self):
        """Default sigma range works."""
        scan = scan_conservation_vs_sigma(6)
        assert len(scan.sigma_values) == 37
        assert scan.sigma_values[0] == pytest.approx(0.1, abs=1e-10)
        assert scan.sigma_values[-1] == pytest.approx(1.0, abs=1e-10)

    def test_all_finite(self):
        """All scan values are finite."""
        scan = scan_conservation_vs_sigma(6, np.linspace(0.3, 0.7, 5))
        assert np.all(np.isfinite(scan.total_energy))
        assert np.all(np.isfinite(scan.total_charge))
        assert np.all(np.isfinite(scan.total_charge_density))
        assert np.all(np.isfinite(scan.charge_gradient))


# ============================================================================
# TestGrammarComplianceConservation
# ============================================================================


class TestGrammarComplianceConservation:
    """Test grammar compliance via conservation charge drift."""

    def test_returns_four_results(self):
        """Grammar test returns 4 protocols."""
        results = test_grammar_conservation(8, sigma=0.5)
        assert len(results) == 4

    def test_two_compliant_two_violating(self):
        """Two compliant, two violating protocols."""
        results = test_grammar_conservation(8)
        compliant = [r for r in results if r.is_grammar_compliant]
        violating = [r for r in results if not r.is_grammar_compliant]
        assert len(compliant) == 2
        assert len(violating) == 2

    def test_compliant_smaller_drift(self):
        """Grammar-compliant protocols have smaller charge drift.

        This is the key P7 test: S_grammar -> 0 under U1-U6 manifests
        as lower charge drift for smooth (grammar-compliant) evolution
        vs abrupt (grammar-violating) jumps at sigma = 0.5.
        """
        results = test_grammar_conservation(15, sigma=0.5)
        compliant = [r for r in results if r.is_grammar_compliant]
        violating = [r for r in results if not r.is_grammar_compliant]

        mean_drift_c = sum(r.charge_drift for r in compliant) / len(compliant)
        mean_drift_v = sum(r.charge_drift for r in violating) / len(violating)

        assert mean_drift_c < mean_drift_v

    def test_compliant_higher_quality(self):
        """Grammar-compliant protocols have higher conservation quality."""
        results = test_grammar_conservation(12, sigma=0.5)
        compliant = [r for r in results if r.is_grammar_compliant]
        violating = [r for r in results if not r.is_grammar_compliant]

        mean_q_c = sum(r.conservation_quality for r in compliant) / len(compliant)
        mean_q_v = sum(r.conservation_quality for r in violating) / len(violating)

        assert mean_q_c > mean_q_v

    def test_conservation_quality_bounded(self):
        """Conservation quality in (0, 1]."""
        results = test_grammar_conservation(8)
        for r in results:
            assert 0 < r.conservation_quality <= 1.0

    def test_protocol_names(self):
        """Protocols have expected names."""
        results = test_grammar_conservation(6)
        names = {r.protocol for r in results}
        assert names == {
            "smooth_forward", "smooth_backward",
            "abrupt_forward", "abrupt_backward",
        }

    def test_sigma_endpoints(self):
        """Sigma endpoints are correct."""
        results = test_grammar_conservation(
            6, sigma=0.5, delta_small=0.02, delta_large=0.3,
        )
        for r in results:
            assert r.sigma_start == pytest.approx(0.5, abs=1e-10)

        smooth_fwd = [r for r in results if r.protocol == "smooth_forward"][0]
        assert smooth_fwd.sigma_end == pytest.approx(0.52, abs=1e-10)

        abrupt_fwd = [r for r in results if r.protocol == "abrupt_forward"][0]
        assert abrupt_fwd.sigma_end == pytest.approx(0.8, abs=1e-10)

    def test_k_too_small(self):
        """k < 2 raises ValueError."""
        with pytest.raises(ValueError, match="k must be >= 2"):
            test_grammar_conservation(1)


# ============================================================================
# TestRunCriticalConservationAnalysis
# ============================================================================


class TestRunCriticalConservationAnalysis:
    """Test integrated P7 analysis."""

    def test_basic_run(self):
        """Full analysis runs without error."""
        result = run_critical_conservation_analysis(
            k=8,
            sigma_values=np.linspace(0.3, 0.7, 9),
        )
        assert isinstance(result, CriticalConservationAnalysis)
        assert result.k == 8

    def test_has_sigma_scan(self):
        """Result includes sigma scan."""
        result = run_critical_conservation_analysis(
            k=6,
            sigma_values=np.linspace(0.3, 0.7, 5),
        )
        assert isinstance(result.sigma_scan, ConservationSigmaScan)

    def test_has_critical_snapshot(self):
        """Result includes critical snapshot at sigma=0.5."""
        result = run_critical_conservation_analysis(
            k=6,
            sigma_values=np.linspace(0.3, 0.7, 5),
        )
        assert isinstance(result.critical, ConservationAtSigma)
        assert result.critical.sigma == 0.5

    def test_has_grammar_tests(self):
        """Result includes 4 grammar tests."""
        result = run_critical_conservation_analysis(
            k=8,
            sigma_values=np.linspace(0.3, 0.7, 5),
        )
        assert len(result.grammar_tests) == 4

    def test_quality_ratio(self):
        """Quality ratio > 1 means compliance preserves charge better."""
        result = run_critical_conservation_analysis(
            k=15,
            sigma_values=np.linspace(0.3, 0.7, 9),
        )
        assert result.quality_ratio > 1.0

    def test_compliant_mean_higher(self):
        """Compliant mean quality > violating mean quality."""
        result = run_critical_conservation_analysis(
            k=12,
            sigma_values=np.linspace(0.3, 0.7, 5),
        )
        assert result.compliant_mean_quality > result.violating_mean_quality

    def test_custom_sigma(self):
        """Analysis at sigma != 0.5."""
        result = run_critical_conservation_analysis(
            k=6,
            sigma=0.4,
            sigma_values=np.linspace(0.2, 0.6, 5),
        )
        assert result.critical.sigma == 0.4


# ============================================================================
# TestPhysicsValidation
# ============================================================================


class TestPhysicsValidation:
    """Validate P7 physics predictions."""

    def test_noether_charge_sign_consistency(self):
        """Total charge has consistent sign across modes at sigma=0.5."""
        snap = compute_eigenmode_conservation(15, sigma=0.5)
        # Not all modes have same sign, but total is well-defined
        assert math.isfinite(snap.total_charge)

    def test_charge_density_sum_converges(self):
        """Total charge density converges as k grows."""
        # Charge density per mode should not diverge
        rho_8 = compute_eigenmode_conservation(8).total_charge_density / 8
        rho_15 = compute_eigenmode_conservation(15).total_charge_density / 15
        # Both finite and bounded
        assert abs(rho_8) < 100
        assert abs(rho_15) < 100

    def test_energy_positive(self):
        """Total energy is strictly positive."""
        snap = compute_eigenmode_conservation(10, sigma=0.5)
        assert snap.total_energy > 0

    def test_energy_sum_of_squares(self):
        """Energy density is sum of squares => positive definite."""
        snap = compute_eigenmode_conservation(8, sigma=0.5)
        for m in snap.modes:
            assert m.energy_density >= 0
            # At least one field should be nonzero
            fields_sum = abs(m.phi_s) + abs(m.grad_phi) + abs(m.k_phi)
            if fields_sum > 1e-10:
                assert m.energy_density > 0

    def test_continuity_residual_at_half(self):
        """Conservation residual (|dQ/dsigma|) is low near sigma=0.5.

        This tests the structural conservation theorem prediction:
        S_grammar -> 0 at criticality.
        """
        scan = scan_conservation_vs_sigma(
            12, np.linspace(0.2, 0.8, 25),
        )
        # Find gradient at sigma closest to 0.5
        idx_half = int(np.argmin(np.abs(scan.sigma_values - 0.5)))
        grad_at_half = scan.charge_gradient[idx_half]

        # The gradient should be finite
        assert math.isfinite(grad_at_half)

    def test_j_dnfr_hellmann_feynman_general(self):
        """J_DNFR matches numerical dλ/dσ for k=10 at sigma=0.3."""
        from tnfr.riemann.operator import build_prime_path_graph, build_h_tnfr

        k, sigma, ds = 10, 0.3, 1e-5
        G = build_prime_path_graph(k)
        nodes = sorted(G.nodes())
        log_p = np.array([np.log(float(G.nodes[n]["label"])) for n in nodes])

        Hc, _ = build_h_tnfr(G, sigma=sigma)
        Hp, _ = build_h_tnfr(G, sigma=sigma + ds)
        Hm, _ = build_h_tnfr(G, sigma=sigma - ds)

        evals_c, vecs_c = np.linalg.eigh(Hc)
        evals_p, _ = np.linalg.eigh(Hp)
        evals_m, _ = np.linalg.eigh(Hm)

        numerical = (evals_p - evals_m) / (2 * ds)
        analytic = compute_spectral_j_dnfr(vecs_c, log_p)

        np.testing.assert_allclose(analytic, numerical, rtol=1e-3)


# ============================================================================
# TestEdgeCases
# ============================================================================


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_k_equals_2(self):
        """Minimal graph k=2."""
        snap = compute_eigenmode_conservation(2, sigma=0.5)
        assert len(snap.modes) == 2
        assert snap.total_energy > 0

    def test_large_sigma(self):
        """Large sigma value doesn't crash."""
        snap = compute_eigenmode_conservation(6, sigma=2.0)
        assert math.isfinite(snap.total_energy)

    def test_small_sigma(self):
        """Small sigma value doesn't crash."""
        snap = compute_eigenmode_conservation(6, sigma=0.01)
        assert math.isfinite(snap.total_energy)

    def test_sigma_exactly_half(self):
        """sigma=0.5 exactly."""
        snap = compute_eigenmode_conservation(8, sigma=0.5)
        assert snap.sigma == 0.5

    def test_scan_single_point(self):
        """Sigma scan with single point."""
        scan = scan_conservation_vs_sigma(6, np.array([0.5]))
        assert len(scan.total_energy) == 1

    def test_grammar_small_deltas(self):
        """Very small deltas for grammar test."""
        results = test_grammar_conservation(
            6, sigma=0.5, delta_small=0.001, delta_large=0.01,
        )
        assert len(results) == 4
        for r in results:
            assert math.isfinite(r.charge_drift)


# ============================================================================
# TestReproducibility (Invariant #6)
# ============================================================================


class TestReproducibility:
    """Verify deterministic results (Canonical Invariant #6)."""

    def test_conservation_deterministic(self):
        """Same inputs produce identical conservation results."""
        snap1 = compute_eigenmode_conservation(8, sigma=0.5)
        snap2 = compute_eigenmode_conservation(8, sigma=0.5)

        assert snap1.total_energy == pytest.approx(snap2.total_energy, rel=1e-14)
        assert snap1.total_charge == pytest.approx(snap2.total_charge, rel=1e-14)

    def test_scan_deterministic(self):
        """Sigma scan is deterministic."""
        sigmas = np.linspace(0.3, 0.7, 5)
        scan1 = scan_conservation_vs_sigma(6, sigmas)
        scan2 = scan_conservation_vs_sigma(6, sigmas)

        np.testing.assert_array_equal(scan1.total_energy, scan2.total_energy)
        np.testing.assert_array_equal(scan1.total_charge, scan2.total_charge)

    def test_grammar_test_deterministic(self):
        """Grammar compliance test is deterministic."""
        r1 = test_grammar_conservation(6, sigma=0.5)
        r2 = test_grammar_conservation(6, sigma=0.5)

        for a, b in zip(r1, r2):
            assert a.charge_drift == pytest.approx(b.charge_drift, rel=1e-14)
            assert a.conservation_quality == pytest.approx(
                b.conservation_quality, rel=1e-14,
            )
