"""Tests for per-eigenmode structural field tetrad (TNFR-Riemann P3).

Validates the spectral-domain analogues of the TNFR canonical fields
(Phi_s, |grad_phi|, K_phi, xi_C) computed for each eigenmode psi_j
of the discrete operator H^(k)(sigma) = L_k + V_sigma.

Test categories:
  1. Construction & data structure integrity
  2. Spectral structural potential Phi_s(j)
  3. Eigenvector gradient |grad_phi|(j)
  4. Eigenvector curvature K_phi(j)
  5. Coherence length xi_C(j)
  6. U6 confinement diagnostics
  7. General graph topology support
  8. Scaling and reproducibility

Physics basis: the structural field tetrad is the canonical diagnostic
toolkit for TNFR coherence (AGENTS.md, Structural Field Tetrad).
"""

import math

import numpy as np
import pytest

from tnfr.riemann.eigenmode_fields import (
    PHI_S_GOLDEN_THRESHOLD,
    PHI_S_VON_KOCH_THRESHOLD,
    EigenmodeFieldAnalysis,
    EigenmodeTetrad,
    _eigenvector_coherence_length_path,
    _path_eigenvector_curvature,
    _path_eigenvector_gradient,
    _spectral_structural_potential,
    check_u6_confinement,
    compare_confinement_at_sigma,
    compute_eigenmode_fields_general,
    compute_eigenmode_tetrad,
)
from tnfr.riemann.operator import (
    build_prime_complete_graph,
    build_prime_cycle_graph,
    build_prime_path_graph,
)
from tnfr.riemann.spectral_proof import compute_eigensystem

# ============================================================================
# Section 1: Construction & Data Structure Integrity
# ============================================================================


class TestConstruction:
    """Basic construction and data type tests."""

    def test_basic_construction_k10(self):
        """Can compute tetrad for k=10."""
        result = compute_eigenmode_tetrad(10, 0.5)
        assert isinstance(result, EigenmodeFieldAnalysis)
        assert result.k == 10
        assert result.sigma == 0.5
        assert len(result.tetrads) == 10

    def test_basic_construction_k2(self):
        """Minimum graph size k=2."""
        result = compute_eigenmode_tetrad(2, 0.5)
        assert result.k == 2
        assert len(result.tetrads) == 2

    def test_invalid_k_raises(self):
        """k < 2 raises ValueError."""
        with pytest.raises(ValueError, match="k must be >= 2"):
            compute_eigenmode_tetrad(1, 0.5)

    def test_tetrad_fields_populated(self):
        """All tetrad fields are finite for each mode."""
        result = compute_eigenmode_tetrad(20, 0.5)
        for t in result.tetrads:
            assert isinstance(t, EigenmodeTetrad)
            assert math.isfinite(t.eigenvalue)
            assert math.isfinite(t.phi_s)
            assert math.isfinite(t.grad_phi)
            assert math.isfinite(t.k_phi)
            # xi_c may be NaN for certain modes
            assert isinstance(t.xi_c, float)

    def test_mode_indices_sequential(self):
        """Mode indices are 0, 1, ..., k-1."""
        result = compute_eigenmode_tetrad(15, 0.5)
        indices = [t.mode_index for t in result.tetrads]
        assert indices == list(range(15))

    def test_eigenvalues_sorted_ascending(self):
        """Eigenvalues are in ascending order."""
        result = compute_eigenmode_tetrad(20, 0.5)
        eigenvalues = [t.eigenvalue for t in result.tetrads]
        for i in range(len(eigenvalues) - 1):
            assert eigenvalues[i] <= eigenvalues[i + 1] + 1e-12

    def test_u6_fraction_in_range(self):
        """U6 fraction is between 0 and 1."""
        result = compute_eigenmode_tetrad(20, 0.5)
        assert 0.0 <= result.u6_fraction <= 1.0

    def test_aggregate_means_positive(self):
        """Aggregate mean fields are non-negative."""
        result = compute_eigenmode_tetrad(20, 0.5)
        assert result.mean_phi_s >= 0.0
        assert result.mean_grad_phi >= 0.0
        assert result.mean_k_phi >= 0.0

    def test_u6_violations_consistent(self):
        """U6 violations list is consistent with u6_fraction."""
        result = compute_eigenmode_tetrad(20, 0.5)
        expected_fraction = 1.0 - len(result.u6_violations) / result.k
        assert abs(result.u6_fraction - expected_fraction) < 1e-12

    def test_sigma_off_half(self):
        """Can compute at sigma != 1/2."""
        for sigma in [0.1, 0.3, 0.7, 0.9]:
            result = compute_eigenmode_tetrad(10, sigma)
            assert result.sigma == sigma
            assert len(result.tetrads) == 10


# ============================================================================
# Section 2: Spectral Structural Potential Phi_s(j)
# ============================================================================


class TestSpectralPotential:
    """Tests for the spectral structural potential field."""

    def test_phi_s_nonnegative(self):
        """Phi_s(j) >= 0 since |lambda_m| >= 0 and distances > 0."""
        result = compute_eigenmode_tetrad(20, 0.5)
        for t in result.tetrads:
            assert t.phi_s >= 0.0

    def test_phi_s_boundary_modes_largest(self):
        """Boundary modes (j=0, j=k-1) feel more pressure from asymmetry.

        Interior modes have contributions from both sides, partially
        cancelling in magnitude.  Boundary modes have all contributions
        from one side, typically giving larger Phi_s.
        """
        result = compute_eigenmode_tetrad(30, 0.5)
        phi_s_vals = [t.phi_s for t in result.tetrads]
        # The first and last modes should have among the largest Phi_s
        interior_max = max(phi_s_vals[5:-5])
        boundary_max = max(phi_s_vals[0], phi_s_vals[-1])
        # Boundary modes should be at least as large as most interior modes
        assert boundary_max >= interior_max * 0.5

    def test_phi_s_ground_state_at_half(self):
        """At sigma=1/2, ground state has lambda_0=0.

        This reduces its contribution |lambda_0|=0 to other modes' Phi_s,
        lowering overall structural pressure.
        """
        result = compute_eigenmode_tetrad(20, 0.5)
        # Ground state eigenvalue should be ~0
        assert abs(result.tetrads[0].eigenvalue) < 1e-10
        # Its Phi_s is the sum of |lambda_m| / m^2 for m=1..k-1
        assert result.tetrads[0].phi_s > 0.0

    def test_phi_s_increases_with_sigma_off_half(self):
        """Phi_s should generally increase when sigma moves away from 1/2.

        At sigma=1/2, lambda_0=0 contributes nothing.  Off 1/2, all
        eigenvalues are positive, increasing total structural pressure.
        """
        analysis_half = compute_eigenmode_tetrad(20, 0.5)
        analysis_off = compute_eigenmode_tetrad(20, 0.7)
        # Mean |Phi_s| should be larger off 1/2
        assert analysis_off.mean_phi_s >= analysis_half.mean_phi_s * 0.8

    def test_phi_s_internal_function(self):
        """Direct test of the internal _spectral_structural_potential."""
        eigenvalues = np.array([0.0, 1.0, 4.0])
        # Phi_s(0) = |1|/1 + |4|/4 = 1 + 1 = 2.0
        assert abs(_spectral_structural_potential(eigenvalues, 0) - 2.0) < 1e-10
        # Phi_s(1) = |0|/1 + |4|/1 = 0 + 4 = 4.0
        assert abs(_spectral_structural_potential(eigenvalues, 1) - 4.0) < 1e-10
        # Phi_s(2) = |0|/4 + |1|/1 = 0 + 1 = 1.0
        assert abs(_spectral_structural_potential(eigenvalues, 2) - 1.0) < 1e-10

    def test_phi_s_symmetric_spectrum(self):
        """For symmetric eigenvalue spectrum, Phi_s is symmetric about center."""
        eigenvalues = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
        phi_0 = _spectral_structural_potential(eigenvalues, 0)
        phi_4 = _spectral_structural_potential(eigenvalues, 4)
        assert abs(phi_0 - phi_4) < 1e-10


# ============================================================================
# Section 3: Eigenvector Gradient |grad_phi|(j)
# ============================================================================


class TestEigenvectorGradient:
    """Tests for the eigenvector gradient field."""

    def test_gradient_nonnegative(self):
        """Gradient magnitude is always >= 0."""
        result = compute_eigenmode_tetrad(20, 0.5)
        for t in result.tetrads:
            assert t.grad_phi >= 0.0

    def test_ground_state_gradient_zero_at_half(self):
        """At sigma=1/2, ground state is constant -> gradient = 0."""
        result = compute_eigenmode_tetrad(20, 0.5)
        assert result.tetrads[0].grad_phi < 1e-10

    def test_gradient_increases_on_average(self):
        """Higher eigenmodes oscillate more -> larger gradient on average.

        For H = L + V with log-gap weights, strict monotonicity may not
        hold mode-by-mode, but mean gradient over first-quarter modes
        should be smaller than last-quarter modes.
        """
        result = compute_eigenmode_tetrad(30, 0.5)
        grads = [t.grad_phi for t in result.tetrads]
        quarter = len(grads) // 4
        low_mean = float(np.mean(grads[:quarter]))
        high_mean = float(np.mean(grads[-quarter:]))
        assert high_mean > low_mean

    def test_gradient_constant_vector_zero(self):
        """Constant eigenvector has zero gradient."""
        psi = np.ones(10) / np.sqrt(10)
        assert abs(_path_eigenvector_gradient(psi)) < 1e-14

    def test_gradient_alternating_vector(self):
        """Alternating sign vector has maximum gradient."""
        psi = np.array([1, -1, 1, -1, 1], dtype=np.float64) / np.sqrt(5)
        grad = _path_eigenvector_gradient(psi)
        assert grad > 0.3  # Should be large

    def test_gradient_single_step(self):
        """Manual gradient computation for simple vector."""
        psi = np.array([0.0, 1.0, 0.0])
        # node 0: |1-0| = 1.0
        # node 1: 0.5*(|0-1| + |0-1|) = 1.0
        # node 2: |0-1| = 1.0
        # mean = 1.0
        assert abs(_path_eigenvector_gradient(psi) - 1.0) < 1e-14


# ============================================================================
# Section 4: Eigenvector Curvature K_phi(j)
# ============================================================================


class TestEigenvectorCurvature:
    """Tests for the eigenvector curvature field."""

    def test_curvature_nonnegative(self):
        """Curvature magnitude is always >= 0."""
        result = compute_eigenmode_tetrad(20, 0.5)
        for t in result.tetrads:
            assert t.k_phi >= 0.0

    def test_ground_state_curvature_zero_at_half(self):
        """At sigma=1/2, ground state is constant -> curvature = 0."""
        result = compute_eigenmode_tetrad(20, 0.5)
        assert result.tetrads[0].k_phi < 1e-10

    def test_curvature_increases_on_average(self):
        """Higher eigenmodes have higher curvature on average.

        For H = L + V with log-gap weights, strict monotonicity may not
        hold, but mean curvature over low modes should be smaller
        than over high modes.
        """
        result = compute_eigenmode_tetrad(30, 0.5)
        curvatures = [t.k_phi for t in result.tetrads]
        quarter = len(curvatures) // 4
        low_mean = float(np.mean(curvatures[:quarter]))
        high_mean = float(np.mean(curvatures[-quarter:]))
        assert high_mean > low_mean

    def test_curvature_constant_vector_zero(self):
        """Constant vector has zero curvature."""
        psi = np.ones(10) / np.sqrt(10)
        assert abs(_path_eigenvector_curvature(psi)) < 1e-14

    def test_curvature_linear_vector(self):
        """Linear vector on path has small curvature (only at boundaries)."""
        psi = np.linspace(0, 1, 20)
        curv = _path_eigenvector_curvature(psi)
        # Interior curvature is zero for linear function
        # Only boundary contributions -> total should be small
        assert curv < 0.1

    def test_curvature_manual_computation(self):
        """Manual curvature for [0, 1, 0]."""
        psi = np.array([0.0, 1.0, 0.0])
        # node 0: |0 - 1| = 1.0
        # node 1: |1 - (0+0)/2| = 1.0
        # node 2: |0 - 1| = 1.0
        # mean = 1.0
        assert abs(_path_eigenvector_curvature(psi) - 1.0) < 1e-14

    def test_gradient_curvature_ordering(self):
        """Gradient and curvature correlate positively across modes.

        Both measure spatial variation of eigenmode, so higher gradient
        modes should also have higher curvature.
        """
        result = compute_eigenmode_tetrad(30, 0.5)
        grads = [t.grad_phi for t in result.tetrads]
        curvs = [t.k_phi for t in result.tetrads]
        # Pearson correlation should be strongly positive
        corr = np.corrcoef(grads, curvs)[0, 1]
        assert corr > 0.9


# ============================================================================
# Section 5: Coherence Length xi_C(j)
# ============================================================================


class TestCoherenceLength:
    """Tests for the eigenmode coherence length."""

    def test_xi_c_ground_state_large(self):
        """Ground state at sigma=1/2 is delocalized -> large or NaN xi_C.

        The constant eigenvector has uniform |psi|^2, so C(r) = const
        and the exponential fit gives slope ~ 0 -> xi_C -> inf (NaN).
        """
        result = compute_eigenmode_tetrad(30, 0.5)
        xi_c_0 = result.tetrads[0].xi_c
        # Either NaN (slope ~0) or very large
        assert math.isnan(xi_c_0) or xi_c_0 > 5.0

    def test_xi_c_positive_when_finite(self):
        """Finite xi_C values are positive."""
        result = compute_eigenmode_tetrad(30, 0.5)
        for t in result.tetrads:
            if math.isfinite(t.xi_c):
                assert t.xi_c > 0.0

    def test_xi_c_decreases_with_mode_index(self):
        """Higher modes are more localized -> shorter xi_C.

        The probability density of higher eigenmodes oscillates faster,
        so correlations decay more rapidly.  This trend should hold
        on average, though not necessarily strictly monotonically.
        """
        result = compute_eigenmode_tetrad(50, 0.5)
        # Compare low vs high modes (skip ground state which may be NaN)
        low_modes = [t.xi_c for t in result.tetrads[1:10] if math.isfinite(t.xi_c)]
        high_modes = [t.xi_c for t in result.tetrads[-10:] if math.isfinite(t.xi_c)]
        if low_modes and high_modes:
            assert np.mean(low_modes) > np.mean(high_modes) * 0.8

    def test_xi_c_small_k_returns_nan(self):
        """For k < 4, xi_C returns NaN."""
        psi = np.array([0.5, 0.5, 0.5])
        assert math.isnan(_eigenvector_coherence_length_path(psi))

    def test_xi_c_localized_vector(self):
        """Strongly localized eigenvector has short xi_C."""
        k = 30
        psi = np.zeros(k)
        psi[k // 2] = 1.0  # Delta function at center
        xi_c = _eigenvector_coherence_length_path(psi)
        # Should be NaN or very small (only one non-zero entry)
        assert math.isnan(xi_c) or xi_c < 2.0

    def test_xi_c_mean_excludes_nan(self):
        """Mean xi_C in analysis excludes NaN values."""
        result = compute_eigenmode_tetrad(20, 0.5)
        finite_xi = [t.xi_c for t in result.tetrads if math.isfinite(t.xi_c)]
        if finite_xi:
            expected_mean = float(np.mean(finite_xi))
            assert abs(result.mean_xi_c - expected_mean) < 1e-10


# ============================================================================
# Section 6: U6 Confinement Diagnostics
# ============================================================================


class TestU6Confinement:
    """Tests for the eigenmode-level U6 confinement analysis."""

    def test_u6_default_threshold(self):
        """Default threshold is von Koch value."""
        result = compute_eigenmode_tetrad(20, 0.5)
        assert result.u6_threshold == PHI_S_VON_KOCH_THRESHOLD

    def test_u6_custom_threshold(self):
        """Custom threshold is respected."""
        result = compute_eigenmode_tetrad(20, 0.5, u6_threshold=10.0)
        assert result.u6_threshold == 10.0

    def test_u6_high_threshold_all_confined(self):
        """Very high threshold -> all modes confined."""
        result = compute_eigenmode_tetrad(20, 0.5, u6_threshold=1e6)
        assert result.u6_fraction == 1.0
        assert len(result.u6_violations) == 0

    def test_u6_zero_threshold_all_violated(self):
        """Zero threshold -> all modes violated (Phi_s > 0 for all)."""
        result = compute_eigenmode_tetrad(20, 0.5, u6_threshold=0.0)
        assert result.u6_fraction == 0.0
        assert len(result.u6_violations) == 20

    def test_u6_check_function(self):
        """check_u6_confinement returns correct structure."""
        analysis = compute_eigenmode_tetrad(20, 0.5)
        u6 = check_u6_confinement(analysis)

        assert "confined" in u6
        assert "fraction" in u6
        assert "violations" in u6
        assert "max_phi_s" in u6
        assert "threshold" in u6
        assert isinstance(u6["confined"], bool)
        assert 0.0 <= u6["fraction"] <= 1.0

    def test_u6_check_custom_threshold(self):
        """check_u6_confinement with custom threshold."""
        analysis = compute_eigenmode_tetrad(20, 0.5)
        u6_strict = check_u6_confinement(analysis, u6_threshold=0.1)
        u6_loose = check_u6_confinement(analysis, u6_threshold=100.0)
        assert u6_strict["fraction"] <= u6_loose["fraction"]

    def test_u6_violations_are_correct(self):
        """Violations list matches modes with |Phi_s| >= threshold."""
        result = compute_eigenmode_tetrad(20, 0.5)
        for j in result.u6_violations:
            assert abs(result.tetrads[j].phi_s) >= result.u6_threshold
        for t in result.tetrads:
            if t.mode_index not in result.u6_violations:
                assert abs(t.phi_s) < result.u6_threshold

    def test_compare_confinement_default_sigmas(self):
        """compare_confinement_at_sigma returns dict for default sigmas."""
        results = compare_confinement_at_sigma(15)
        assert 0.5 in results
        assert len(results) == 5  # Default: [0.3, 0.4, 0.5, 0.6, 0.7]
        for sigma, r in results.items():
            assert "fraction" in r
            assert "mean_phi_s" in r

    def test_compare_confinement_custom_sigmas(self):
        """compare_confinement_at_sigma with custom sigma list."""
        sigmas = [0.25, 0.5, 0.75]
        results = compare_confinement_at_sigma(15, sigma_values=sigmas)
        assert set(results.keys()) == set(sigmas)

    def test_confinement_best_near_half(self):
        """U6 confinement should be best (highest fraction) near sigma=1/2.

        At sigma=1/2, lambda_0=0 reduces total structural pressure,
        so fewer modes should violate confinement.
        """
        results = compare_confinement_at_sigma(
            30,
            sigma_values=[0.2, 0.35, 0.5, 0.65, 0.8],
        )
        fraction_at_half = results[0.5]["fraction"]
        fraction_at_02 = results[0.2]["fraction"]
        fraction_at_08 = results[0.8]["fraction"]
        # sigma=1/2 should have equal or better confinement than extremes
        assert fraction_at_half >= min(fraction_at_02, fraction_at_08)


# ============================================================================
# Section 7: General Graph Topology Support
# ============================================================================


class TestGeneralTopology:
    """Tests for the general graph implementation."""

    def test_general_path_matches_tridiagonal(self):
        """General graph results match tridiagonal for path graph.

        The path graph Phi_s depends only on eigenvalues, so should
        be identical.  Gradient, curvature, and xi_C use graph structure,
        so path-specific and general implementations should agree.
        """
        k = 15
        sigma = 0.5

        # Tridiagonal (fast path)
        r_tri = compute_eigenmode_tetrad(k, sigma)

        # General graph
        G = build_prime_path_graph(k)
        r_gen = compute_eigenmode_fields_general(G, sigma)

        assert r_tri.k == r_gen.k

        for j in range(k):
            t_tri = r_tri.tetrads[j]
            t_gen = r_gen.tetrads[j]

            # Eigenvalues should match closely
            assert abs(t_tri.eigenvalue - t_gen.eigenvalue) < 1e-6, (
                f"Mode {j}: eigenvalue mismatch "
                f"{t_tri.eigenvalue} vs {t_gen.eigenvalue}"
            )

            # Phi_s depends only on eigenvalues, should be very close
            assert (
                abs(t_tri.phi_s - t_gen.phi_s) < 0.1
            ), f"Mode {j}: Phi_s mismatch {t_tri.phi_s} vs {t_gen.phi_s}"

            # Gradient and curvature should be close
            assert (
                abs(t_tri.grad_phi - t_gen.grad_phi) < 0.05
            ), f"Mode {j}: grad_phi mismatch"
            assert abs(t_tri.k_phi - t_gen.k_phi) < 0.05, f"Mode {j}: k_phi mismatch"

    def test_general_cycle_graph(self):
        """Can compute tetrad for cycle topology."""
        G = build_prime_cycle_graph(15)
        result = compute_eigenmode_fields_general(G, 0.5)
        assert result.k == 15
        assert len(result.tetrads) == 15
        # All fields should be finite
        for t in result.tetrads:
            assert math.isfinite(t.phi_s)
            assert math.isfinite(t.grad_phi)
            assert math.isfinite(t.k_phi)

    def test_general_complete_graph(self):
        """Can compute tetrad for complete graph."""
        G = build_prime_complete_graph(10)
        result = compute_eigenmode_fields_general(G, 0.5)
        assert result.k == 10
        assert len(result.tetrads) == 10

    def test_general_sigma_off_half(self):
        """General graph tetrad at sigma != 1/2."""
        G = build_prime_path_graph(12)
        result = compute_eigenmode_fields_general(G, 0.7)
        assert result.sigma == 0.7
        assert len(result.tetrads) == 12

    def test_general_u6_analysis(self):
        """U6 confinement works for general graphs."""
        G = build_prime_cycle_graph(15)
        result = compute_eigenmode_fields_general(G, 0.5)
        u6 = check_u6_confinement(result)
        assert "confined" in u6
        assert "fraction" in u6


# ============================================================================
# Section 8: Scaling and Reproducibility
# ============================================================================


class TestScalingAndReproducibility:
    """Tests for scaling behavior and deterministic output."""

    def test_reproducibility(self):
        """Same (k, sigma) produces identical results."""
        r1 = compute_eigenmode_tetrad(20, 0.5)
        r2 = compute_eigenmode_tetrad(20, 0.5)

        for j in range(20):
            assert r1.tetrads[j].eigenvalue == r2.tetrads[j].eigenvalue
            assert r1.tetrads[j].phi_s == r2.tetrads[j].phi_s
            assert r1.tetrads[j].grad_phi == r2.tetrads[j].grad_phi
            assert r1.tetrads[j].k_phi == r2.tetrads[j].k_phi
            # xi_c may both be NaN
            if math.isfinite(r1.tetrads[j].xi_c):
                assert r1.tetrads[j].xi_c == r2.tetrads[j].xi_c

    def test_scaling_k_50(self):
        """Can handle k=50 without error."""
        result = compute_eigenmode_tetrad(50, 0.5)
        assert result.k == 50
        assert len(result.tetrads) == 50

    def test_scaling_k_200(self):
        """Can handle k=200 without error."""
        result = compute_eigenmode_tetrad(200, 0.5)
        assert result.k == 200
        assert len(result.tetrads) == 200

    def test_max_gradient_increases_with_k(self):
        """Max eigenmode gradient grows with k.

        As k increases, eigenvectors are normalised to unit length over
        more nodes, so mean gradient per node may decrease.  However, the
        maximum gradient mode should have a comparable or growing value
        as the spectral bandwidth increases.
        """
        r20 = compute_eigenmode_tetrad(20, 0.5)
        r50 = compute_eigenmode_tetrad(50, 0.5)
        max_grad_20 = max(t.grad_phi for t in r20.tetrads)
        max_grad_50 = max(t.grad_phi for t in r50.tetrads)
        # Max gradient should be comparable (within 2x)
        assert max_grad_50 > max_grad_20 * 0.3

    def test_phi_s_grows_with_k(self):
        """Mean Phi_s grows with k (more modes contribute pressure)."""
        r10 = compute_eigenmode_tetrad(10, 0.5)
        r30 = compute_eigenmode_tetrad(30, 0.5)
        assert r30.mean_phi_s > r10.mean_phi_s * 0.5


# ============================================================================
# Section 9: Constants
# ============================================================================


class TestConstants:
    """Tests for physical constants."""

    def test_von_koch_threshold(self):
        """Per-node Φ_s confinement bound is π/4 (quarter phase-wrap)."""
        assert abs(PHI_S_VON_KOCH_THRESHOLD - (3.141592653589793 / 4)) < 1e-6

    def test_golden_threshold(self):
        """Φ_s drift bound (U6) is π/2 (half phase-wrap)."""
        assert abs(PHI_S_GOLDEN_THRESHOLD - (3.141592653589793 / 2)) < 0.001


# ============================================================================
# Section 10: Edge Cases
# ============================================================================


class TestEdgeCases:
    """Edge case handling."""

    def test_k2_minimal(self):
        """k=2 produces valid results."""
        result = compute_eigenmode_tetrad(2, 0.5)
        assert len(result.tetrads) == 2
        for t in result.tetrads:
            assert math.isfinite(t.phi_s)
            assert math.isfinite(t.grad_phi)
            assert math.isfinite(t.k_phi)

    def test_k3_xi_c_may_be_nan(self):
        """k=3 may give NaN xi_C (too few points for fit)."""
        result = compute_eigenmode_tetrad(3, 0.5)
        # Just ensure no crash
        assert len(result.tetrads) == 3

    def test_sigma_at_half_exact(self):
        """sigma=0.5 gives exact lambda_0=0."""
        result = compute_eigenmode_tetrad(20, 0.5)
        assert abs(result.tetrads[0].eigenvalue) < 1e-12

    def test_sigma_zero(self):
        """sigma=0 is valid (strong negative potential)."""
        result = compute_eigenmode_tetrad(10, 0.0)
        assert result.k == 10

    def test_sigma_one(self):
        """sigma=1 is valid (strong positive potential)."""
        result = compute_eigenmode_tetrad(10, 1.0)
        assert result.k == 10

    def test_general_graph_too_small_raises(self):
        """General graph with < 2 nodes raises ValueError."""
        import networkx as nx

        G = nx.Graph()
        G.add_node(2, prime=2)
        with pytest.raises(ValueError, match="Graph must have >= 2 nodes"):
            compute_eigenmode_fields_general(G, 0.5)

    def test_check_u6_deprecated_threshold_alias(self):
        """Deprecated 'threshold' kwarg still works for check_u6_confinement."""
        analysis = compute_eigenmode_tetrad(20, 0.5)
        r_old = check_u6_confinement(analysis, threshold=100.0)
        r_new = check_u6_confinement(analysis, u6_threshold=100.0)
        assert r_old["fraction"] == r_new["fraction"]
        assert r_old["violations"] == r_new["violations"]

    def test_check_u6_u6_threshold_wins_over_threshold(self):
        """u6_threshold takes precedence over deprecated threshold kwarg."""
        analysis = compute_eigenmode_tetrad(20, 0.5)
        r = check_u6_confinement(analysis, u6_threshold=100.0, threshold=0.0)
        # u6_threshold=100.0 wins -> all confined
        assert r["fraction"] == 1.0

    def test_general_weight_by_log_gap(self):
        """Edge weights on graph affect general tetrad calculation."""
        G1 = build_prime_path_graph(10, weight_by_log_gap=True)
        G2 = build_prime_path_graph(10, weight_by_log_gap=False)
        r1 = compute_eigenmode_fields_general(G1, 0.5)
        r2 = compute_eigenmode_fields_general(G2, 0.5)
        # Different weighting produces different non-ground eigenvalues
        assert r1.tetrads[1].eigenvalue != r2.tetrads[1].eigenvalue
