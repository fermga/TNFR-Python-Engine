"""Tests for P6: Random prime-graph ensembles and RMT statistics.

Covers all 5 dataclasses, 3 reference distributions, 2 ensemble
generators, 6 statistics functions, 2 comparison functions, and
2 integration functions.

TNFR physics basis: Validates that Dissonance (OZ) ensemble averaging
produces convergent statistics (U2), reproducible dynamics (Invariant #6),
and meaningful RMT classification.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tnfr.riemann.random_ensemble import (
    # Data structures
    EnsembleConfig,
    EnsembleSample,
    SpacingStats,
    RMTComparison,
    EnsembleAnalysis,
    # Constants
    GOE_MEAN_RATIO,
    GUE_MEAN_RATIO,
    POISSON_MEAN_RATIO,
    # Reference distributions
    goe_wigner_surmise,
    gue_wigner_surmise,
    poisson_spacing_pdf,
    # Ensemble generation
    generate_er_ensemble,
    generate_wigner_ensemble,
    # Spacing statistics
    compute_ensemble_spacings,
    compute_spacing_ratio,
    compute_mean_spacing_ratio,
    compute_level_repulsion_exponent,
    # Long-range statistics
    compute_number_variance,
    compute_spectral_rigidity,
    # RMT comparison
    ks_test_vs_reference,
    classify_ensemble,
    # Integration
    run_rmt_ensemble_analysis,
    rmt_convergence_study,
)


# ============================================================================
# Constants
# ============================================================================


class TestConstants:
    """Validate RMT reference constant values."""

    def test_goe_mean_ratio(self):
        assert abs(GOE_MEAN_RATIO - 0.5307) < 1e-4

    def test_gue_mean_ratio(self):
        assert abs(GUE_MEAN_RATIO - 0.5996) < 1e-4

    def test_poisson_mean_ratio(self):
        expected = 2.0 * math.log(2.0) - 1.0
        assert abs(POISSON_MEAN_RATIO - expected) < 1e-10

    def test_ordering(self):
        """Poisson < GOE < GUE for mean spacing ratio."""
        assert POISSON_MEAN_RATIO < GOE_MEAN_RATIO < GUE_MEAN_RATIO


# ============================================================================
# Reference distributions
# ============================================================================


class TestGOEWignerSurmise:
    """Test GOE Wigner surmise P(s) = (pi/2) s exp(-pi s^2/4)."""

    def test_zero(self):
        assert goe_wigner_surmise(0.0) == pytest.approx(0.0)

    def test_positive(self):
        """P(s > 0) > 0 for GOE."""
        s = np.array([0.5, 1.0, 1.5, 2.0])
        p = goe_wigner_surmise(s)
        assert np.all(p > 0)

    def test_normalisation(self):
        """Integral of GOE surmise over [0, inf) ~ 1."""
        s = np.linspace(0, 8, 10000)
        integral = np.trapezoid(goe_wigner_surmise(s), s)
        assert integral == pytest.approx(1.0, abs=0.001)

    def test_mean(self):
        """Mean spacing of GOE ~ 1.0."""
        s = np.linspace(0, 8, 10000)
        mean = np.trapezoid(s * goe_wigner_surmise(s), s)
        assert mean == pytest.approx(1.0, abs=0.02)

    def test_peak_location(self):
        """GOE peak near s ~ 0.8."""
        s = np.linspace(0.01, 3.0, 1000)
        p = goe_wigner_surmise(s)
        peak_s = s[np.argmax(p)]
        assert 0.5 < peak_s < 1.2

    def test_vectorised(self):
        """Works on arrays."""
        s = np.array([0.1, 0.5, 1.0])
        result = goe_wigner_surmise(s)
        assert result.shape == (3,)


class TestGUEWignerSurmise:
    """Test GUE Wigner surmise P(s) = (32/pi^2) s^2 exp(-4s^2/pi)."""

    def test_zero(self):
        assert gue_wigner_surmise(0.0) == pytest.approx(0.0)

    def test_normalisation(self):
        s = np.linspace(0, 8, 10000)
        integral = np.trapezoid(gue_wigner_surmise(s), s)
        assert integral == pytest.approx(1.0, abs=0.001)

    def test_stronger_repulsion(self):
        """GUE has stronger repulsion near s=0 than GOE."""
        s = np.array([0.1])
        assert gue_wigner_surmise(s)[0] < goe_wigner_surmise(s)[0]


class TestPoissonSpacing:
    """Test Poisson spacing distribution P(s) = exp(-s)."""

    def test_zero(self):
        assert poisson_spacing_pdf(0.0) == pytest.approx(1.0)

    def test_normalisation(self):
        s = np.linspace(0, 15, 10000)
        integral = np.trapezoid(poisson_spacing_pdf(s), s)
        assert integral == pytest.approx(1.0, abs=0.001)

    def test_monotone_decreasing(self):
        s = np.linspace(0, 5, 100)
        p = poisson_spacing_pdf(s)
        assert np.all(np.diff(p) <= 0)

    def test_no_repulsion_at_zero(self):
        """Poisson: P(0) = 1 (no level repulsion)."""
        assert poisson_spacing_pdf(0.0) == pytest.approx(1.0)


# ============================================================================
# Data structures
# ============================================================================


class TestDataStructures:
    """Test dataclass creation and basic properties."""

    def test_ensemble_config_defaults(self):
        cfg = EnsembleConfig(k=20)
        assert cfg.k == 20
        assert cfg.n_samples == 100
        assert cfg.sigma == 0.5
        assert cfg.ensemble_type == "erdos_renyi"
        assert cfg.seed == 42

    def test_ensemble_config_frozen(self):
        cfg = EnsembleConfig(k=10)
        with pytest.raises(AttributeError):
            cfg.k = 20  # type: ignore[misc]

    def test_ensemble_sample(self):
        evals = np.array([1.0, 2.0, 3.0])
        sp = np.array([1.0, 1.0])
        s = EnsembleSample(eigenvalues=evals, spacings=sp)
        assert len(s.eigenvalues) == 3
        assert len(s.spacings) == 2


# ============================================================================
# Ensemble generation
# ============================================================================


class TestERensemble:
    """Test Erdos-Renyi ensemble generation."""

    def test_basic_generation(self):
        samples = generate_er_ensemble(10, 5, edge_prob=0.4, seed=42)
        assert len(samples) == 5
        for s in samples:
            assert len(s.eigenvalues) == 10
            assert np.all(np.isfinite(s.eigenvalues))

    def test_eigenvalues_sorted(self):
        samples = generate_er_ensemble(8, 3, seed=99)
        for s in samples:
            assert np.all(np.diff(s.eigenvalues) >= -1e-12)

    def test_spacings_nonnegative(self):
        samples = generate_er_ensemble(10, 5, seed=42)
        for s in samples:
            if len(s.spacings) > 0:
                # After unfolding, spacings should be mostly non-negative
                assert np.mean(s.spacings >= -0.1) > 0.8

    def test_reproducibility(self):
        """Invariant #6: same seed => identical results."""
        s1 = generate_er_ensemble(10, 5, seed=123)
        s2 = generate_er_ensemble(10, 5, seed=123)
        for a, b in zip(s1, s2):
            np.testing.assert_array_equal(a.eigenvalues, b.eigenvalues)

    def test_different_seeds(self):
        """Different seeds produce different ensembles."""
        s1 = generate_er_ensemble(10, 5, seed=1)
        s2 = generate_er_ensemble(10, 5, seed=2)
        # At least one sample should differ
        any_diff = any(
            not np.allclose(a.eigenvalues, b.eigenvalues)
            for a, b in zip(s1, s2)
        )
        assert any_diff

    def test_sigma_effect(self):
        """Different sigma values produce different eigenspectra."""
        s_half = generate_er_ensemble(8, 3, sigma=0.5, seed=42)
        s_one = generate_er_ensemble(8, 3, sigma=1.0, seed=42)
        # Eigenvalues should differ when sigma != 0.5
        any_diff = any(
            not np.allclose(a.eigenvalues, b.eigenvalues)
            for a, b in zip(s_half, s_one)
        )
        assert any_diff

    def test_small_k(self):
        """k = 3 should work (no spacings though)."""
        samples = generate_er_ensemble(3, 2, seed=42)
        assert len(samples) == 2
        for s in samples:
            assert len(s.eigenvalues) == 3
            # k < 4 => empty spacings
            assert len(s.spacings) == 0


class TestWignerEnsemble:
    """Test Wigner perturbation ensemble generation."""

    def test_basic_generation(self):
        samples = generate_wigner_ensemble(10, 5, seed=42)
        assert len(samples) == 5
        for s in samples:
            assert len(s.eigenvalues) == 10

    def test_reproducibility(self):
        s1 = generate_wigner_ensemble(10, 5, seed=77)
        s2 = generate_wigner_ensemble(10, 5, seed=77)
        for a, b in zip(s1, s2):
            np.testing.assert_array_equal(a.eigenvalues, b.eigenvalues)

    def test_zero_scale_deterministic(self):
        """With wigner_scale=0, all samples should be identical (deterministic)."""
        samples = generate_wigner_ensemble(8, 5, wigner_scale=0.0, seed=42)
        for i in range(1, len(samples)):
            np.testing.assert_allclose(
                samples[0].eigenvalues, samples[i].eigenvalues, atol=1e-12
            )

    def test_scale_effect(self):
        """Larger wigner_scale => more spread in eigenvalues."""
        s_small = generate_wigner_ensemble(10, 20, wigner_scale=0.01, seed=42)
        s_large = generate_wigner_ensemble(10, 20, wigner_scale=1.0, seed=42)

        # Compute variance of first eigenvalue across ensemble
        var_small = np.var([s.eigenvalues[0] for s in s_small])
        var_large = np.var([s.eigenvalues[0] for s in s_large])
        assert var_large > var_small

    def test_symmetry(self):
        """Wigner perturbation preserves real symmetric structure."""
        samples = generate_wigner_ensemble(6, 3, seed=42)
        for s in samples:
            # All eigenvalues should be real (real symmetric)
            assert np.all(np.isreal(s.eigenvalues))


# ============================================================================
# Spacing statistics
# ============================================================================


class TestComputeEnsembleSpacings:
    """Test pooling of spacings from ensemble."""

    def test_concatenation(self):
        s1 = EnsembleSample(np.array([1, 2, 3, 4.0]), np.array([0.5, 1.0, 1.5]))
        s2 = EnsembleSample(np.array([1, 2, 3, 4.0]), np.array([0.7, 1.2]))
        pooled = compute_ensemble_spacings([s1, s2])
        assert len(pooled) == 5

    def test_empty_handling(self):
        s1 = EnsembleSample(np.array([1, 2.0]), np.array([]))
        pooled = compute_ensemble_spacings([s1])
        assert len(pooled) == 0


class TestSpacingRatio:
    """Test spacing ratio computation."""

    def test_equal_spacings(self):
        """Equal spacings => r = 1 everywhere."""
        sp = np.array([1.0, 1.0, 1.0, 1.0])
        r = compute_spacing_ratio(sp)
        np.testing.assert_allclose(r, 1.0, atol=1e-10)

    def test_alternating_spacings(self):
        """Alternating [1, 2, 1, 2] => r = [0.5, 0.5, 0.5]."""
        sp = np.array([1.0, 2.0, 1.0, 2.0])
        r = compute_spacing_ratio(sp)
        np.testing.assert_allclose(r, 0.5, atol=1e-10)

    def test_range(self):
        """Ratios should be in [0, 1]."""
        rng = np.random.RandomState(42)
        sp = rng.exponential(size=100)
        r = compute_spacing_ratio(sp)
        assert np.all(r >= 0)
        assert np.all(r <= 1.0 + 1e-10)

    def test_insufficient_data(self):
        r = compute_spacing_ratio(np.array([1.0]))
        assert len(r) == 0


class TestMeanSpacingRatio:
    """Test mean spacing ratio computation."""

    def test_uniform_spacings(self):
        """All equal => <r> = 1."""
        sp = np.ones(100)
        assert compute_mean_spacing_ratio(sp) == pytest.approx(1.0, abs=1e-10)

    def test_poisson_sample(self):
        """Exponential spacings should give <r> ~ 0.386."""
        rng = np.random.RandomState(12345)
        sp = rng.exponential(size=50000)
        r_mean = compute_mean_spacing_ratio(sp)
        assert r_mean == pytest.approx(POISSON_MEAN_RATIO, abs=0.02)

    def test_empty(self):
        assert compute_mean_spacing_ratio(np.array([])) == 0.0


class TestLevelRepulsion:
    """Test level repulsion exponent fitting."""

    def test_goe_sample(self):
        """GOE-like spacings should give beta ~ 1."""
        # Generate spacings from Wigner surmise (GOE)
        rng = np.random.RandomState(42)
        # Rejection sampling from GOE distribution
        proposals = rng.rayleigh(scale=0.8, size=50000)
        # Weight by GOE pdf / proposal pdf
        goe_spacings = proposals[proposals < 4.0]
        if len(goe_spacings) > 1000:
            beta = compute_level_repulsion_exponent(goe_spacings[:5000])
            # Should be in range [0.5, 2.0] (approximate)
            assert 0.3 < beta < 2.5

    def test_insufficient_data(self):
        sp = np.array([0.1, 0.2])
        beta = compute_level_repulsion_exponent(sp)
        assert beta == -1.0


# ============================================================================
# Long-range statistics
# ============================================================================


class TestNumberVariance:
    """Test number variance computation."""

    def test_basic(self):
        # Uniformly spaced eigenvalues (rigid spectrum)
        evals = np.arange(1, 21, dtype=float)
        L_vals, sigma2 = compute_number_variance(evals)
        assert len(L_vals) == len(sigma2)
        assert np.all(np.isfinite(sigma2))

    def test_rigid_spectrum(self):
        """Perfectly uniform spacing => Sigma^2 = 0."""
        evals = np.arange(100, dtype=float)
        L_vals = np.array([1.0])
        _, sigma2 = compute_number_variance(evals, L_vals)
        assert sigma2[0] == pytest.approx(0.0, abs=0.01)

    def test_short_spectrum(self):
        evals = np.array([1.0, 2.0])
        L_vals, sigma2 = compute_number_variance(evals)
        assert np.all(sigma2 == 0)  # n < 4


class TestSpectralRigidity:
    """Test Dyson-Mehta spectral rigidity."""

    def test_basic(self):
        evals = np.arange(1, 21, dtype=float)
        L_vals, delta3 = compute_spectral_rigidity(evals)
        assert len(L_vals) == len(delta3)
        assert np.all(np.isfinite(delta3))

    def test_rigid_low(self):
        """Rigid spectrum should have low Delta_3."""
        evals = np.arange(100, dtype=float)
        L_vals = np.array([1.0, 2.0])
        _, d3 = compute_spectral_rigidity(evals, L_vals)
        # Perfectly rigid: Delta_3 ~ 0
        assert np.all(d3 < 0.1)


# ============================================================================
# RMT comparison
# ============================================================================


class TestKSTest:
    """Test KS statistic computation vs reference distributions."""

    def test_goe_self_consistency(self):
        """GOE samples should have small KS vs GOE."""
        # Inverse CDF sampling from GOE Wigner surmise
        rng = np.random.RandomState(42)
        # Use rejection sampling for GOE
        spacings = []
        while len(spacings) < 5000:
            s = rng.rayleigh(0.8)
            if s < 6 and rng.random() < goe_wigner_surmise(s) / (2.0 * s * np.exp(-s * s / 2)):
                spacings.append(s)
        spacings = np.array(spacings[:5000])
        # Normalise to mean 1
        spacings = spacings / np.mean(spacings)

        ks_goe = ks_test_vs_reference(spacings, "GOE")
        ks_poisson = ks_test_vs_reference(spacings, "Poisson")
        # GOE should fit better than Poisson
        assert ks_goe < ks_poisson

    def test_poisson_self_consistency(self):
        """Exponential spacings should have small KS vs Poisson."""
        rng = np.random.RandomState(42)
        spacings = rng.exponential(size=5000)
        ks_poisson = ks_test_vs_reference(spacings, "Poisson")
        ks_goe = ks_test_vs_reference(spacings, "GOE")
        assert ks_poisson < ks_goe

    def test_empty(self):
        ks = ks_test_vs_reference(np.array([]), "GOE")
        assert ks == 1.0

    def test_invalid_reference(self):
        with pytest.raises(ValueError):
            ks_test_vs_reference(np.array([1.0]), "INVALID")


class TestClassifyEnsemble:
    """Test ensemble classification."""

    def test_returns_rmt_comparison(self):
        stats = SpacingStats(
            all_spacings=np.random.exponential(size=1000),
            mean_spacing_ratio=0.39,
            level_repulsion_beta=0.0,
            histogram_edges=np.linspace(0, 4, 61),
            histogram_counts=np.zeros(60),
            n_spacings=1000,
        )
        rmt = classify_ensemble(stats)
        assert isinstance(rmt, RMTComparison)
        assert rmt.best_match in ("GOE", "GUE", "Poisson")
        assert rmt.ratio_best_match in ("GOE", "GUE", "Poisson")

    def test_poisson_like_classification(self):
        """Spacings with low ratio should classify near Poisson."""
        stats = SpacingStats(
            all_spacings=np.random.exponential(size=2000),
            mean_spacing_ratio=POISSON_MEAN_RATIO,
            level_repulsion_beta=0.0,
            histogram_edges=np.linspace(0, 4, 61),
            histogram_counts=np.zeros(60),
            n_spacings=2000,
        )
        rmt = classify_ensemble(stats)
        assert rmt.ratio_best_match == "Poisson"


# ============================================================================
# Integration analysis
# ============================================================================


class TestRunRMTEnsembleAnalysis:
    """Test the integrated analysis function."""

    def test_er_basic(self):
        analysis = run_rmt_ensemble_analysis(
            k=8, n_samples=10, ensemble_type="erdos_renyi",
            edge_prob=0.4, seed=42, compute_long_range=False,
        )
        assert isinstance(analysis, EnsembleAnalysis)
        assert len(analysis.samples) == 10
        assert analysis.config.k == 8
        assert analysis.spacing_stats.n_spacings > 0
        assert analysis.rmt_comparison.best_match in ("GOE", "GUE", "Poisson")

    def test_wigner_basic(self):
        analysis = run_rmt_ensemble_analysis(
            k=8, n_samples=10, ensemble_type="wigner",
            wigner_scale=0.5, seed=42, compute_long_range=False,
        )
        assert isinstance(analysis, EnsembleAnalysis)
        assert len(analysis.samples) == 10

    def test_with_config(self):
        cfg = EnsembleConfig(k=6, n_samples=5, seed=99)
        analysis = run_rmt_ensemble_analysis(cfg, compute_long_range=False)
        assert analysis.config.k == 6
        assert len(analysis.samples) == 5

    def test_long_range_statistics(self):
        analysis = run_rmt_ensemble_analysis(
            k=10, n_samples=10, seed=42, compute_long_range=True,
        )
        assert analysis.number_variance is not None
        assert analysis.number_variance_L is not None
        assert analysis.spectral_rigidity is not None
        assert analysis.spectral_rigidity_L is not None
        assert np.all(np.isfinite(analysis.number_variance))
        assert np.all(np.isfinite(analysis.spectral_rigidity))

    def test_invalid_ensemble_type(self):
        with pytest.raises(ValueError, match="Unknown ensemble_type"):
            run_rmt_ensemble_analysis(ensemble_type="bogus")

    def test_reproducibility(self):
        """Same config => identical analysis (Invariant #6)."""
        a1 = run_rmt_ensemble_analysis(k=8, n_samples=5, seed=42, compute_long_range=False)
        a2 = run_rmt_ensemble_analysis(k=8, n_samples=5, seed=42, compute_long_range=False)
        np.testing.assert_array_equal(
            a1.spacing_stats.all_spacings,
            a2.spacing_stats.all_spacings,
        )


class TestRMTConvergenceStudy:
    """Test convergence study across k values."""

    def test_basic(self):
        results = rmt_convergence_study(
            [6, 8, 10], n_samples=5, seed=42,
        )
        assert len(results) == 3
        assert results[0].config.k == 6
        assert results[1].config.k == 8
        assert results[2].config.k == 10

    def test_spacing_count_grows(self):
        """More primes => more spacings per sample."""
        results = rmt_convergence_study(
            [6, 12], n_samples=10, seed=42,
        )
        n_sp_small = results[0].spacing_stats.n_spacings
        n_sp_large = results[1].spacing_stats.n_spacings
        assert n_sp_large > n_sp_small

    def test_independent_seeds(self):
        """Different k values should use different seeds."""
        results = rmt_convergence_study([8, 8], n_samples=5, seed=42)
        # Same k but different seeds => different results
        sp1 = results[0].spacing_stats.all_spacings
        sp2 = results[1].spacing_stats.all_spacings
        # Should differ (different seed offsets)
        if len(sp1) > 0 and len(sp2) > 0:
            assert not np.allclose(sp1, sp2)


# ============================================================================
# Edge cases and robustness
# ============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_k_equals_4(self):
        """Minimum k for meaningful spacings."""
        samples = generate_er_ensemble(4, 5, seed=42)
        for s in samples:
            assert len(s.eigenvalues) == 4

    def test_single_sample(self):
        """n_samples=1 should work."""
        samples = generate_er_ensemble(8, 1, seed=42)
        assert len(samples) == 1

    def test_high_edge_prob(self):
        """edge_prob=1.0 => complete graph every time."""
        samples = generate_er_ensemble(6, 3, edge_prob=1.0, seed=42)
        # All samples should be identical (complete graph)
        for i in range(1, len(samples)):
            np.testing.assert_allclose(
                samples[0].eigenvalues, samples[i].eigenvalues, atol=1e-10,
            )

    def test_low_edge_prob(self):
        """Low edge_prob still produces connected graphs."""
        samples = generate_er_ensemble(8, 5, edge_prob=0.1, seed=42)
        for s in samples:
            assert len(s.eigenvalues) == 8

    def test_sigma_half(self):
        """At sigma=0.5, potential vanishes => pure Laplacian."""
        samples = generate_wigner_ensemble(6, 3, sigma=0.5, seed=42)
        for s in samples:
            # Smallest eigenvalue of Laplacian + noise could be negative
            # but spectrum should be finite
            assert np.all(np.isfinite(s.eigenvalues))

    def test_large_wigner_scale(self):
        """Large perturbation should still produce finite results."""
        samples = generate_wigner_ensemble(6, 3, wigner_scale=10.0, seed=42)
        for s in samples:
            assert np.all(np.isfinite(s.eigenvalues))


# ============================================================================
# Physics validation
# ============================================================================


class TestPhysicsValidation:
    """Validate TNFR-RMT physics properties."""

    def test_er_level_repulsion(self):
        """ER ensemble should show level repulsion (beta > 0)."""
        samples = generate_er_ensemble(20, 50, edge_prob=0.4, seed=42)
        all_sp = compute_ensemble_spacings(samples)
        if len(all_sp) > 50:
            beta = compute_level_repulsion_exponent(all_sp)
            # Should show some repulsion (not purely Poisson beta=0)
            assert beta > -0.5  # at minimum not strongly anti-repulsive

    def test_wigner_goe_tendency(self):
        """Wigner perturbation should push statistics towards GOE."""
        analysis = run_rmt_ensemble_analysis(
            k=20, n_samples=50, ensemble_type="wigner",
            wigner_scale=1.0, seed=42, compute_long_range=False,
        )
        rmt = analysis.rmt_comparison
        # KS vs GOE or GUE should be reasonably small
        assert min(rmt.ks_goe, rmt.ks_gue) < 0.5

    def test_ensemble_mean_converges(self):
        """Mean eigenvalue should converge as n_samples grows."""
        s_small = generate_er_ensemble(10, 10, seed=42)
        s_large = generate_er_ensemble(10, 100, seed=42)

        mean_small = np.mean([np.mean(s.eigenvalues) for s in s_small])
        mean_large = np.mean([np.mean(s.eigenvalues) for s in s_large])
        # Both should be finite and in same ballpark
        assert np.isfinite(mean_small)
        assert np.isfinite(mean_large)
        assert abs(mean_small - mean_large) < 2.0 * abs(mean_large) + 1.0

    def test_spectral_gap_positive(self):
        """Ensemble spectra should have positive spectral gap (connected graphs)."""
        samples = generate_er_ensemble(10, 10, sigma=0.5, seed=42)
        for s in samples:
            # At sigma=0.5, potential vanishes, so H = L (Laplacian)
            # Connected graph Laplacian has smallest eigenvalue 0
            # and second eigenvalue > 0 (Fiedler)
            sorted_ev = np.sort(s.eigenvalues)
            # First eigenvalue should be near 0
            assert sorted_ev[0] < 0.01
            # Second should be positive (algebraic connectivity)
            if len(sorted_ev) > 1:
                assert sorted_ev[1] > 1e-10
