r"""Tests for P10: Formal convergence proof that σ*(k) → 1/2.

Validates:
1. **Dusart Bounds** -- Rigorous two-sided prime inequalities
2. **Bilinear Decomposition** -- σ* = 1/2 − tr(LV₁)/tr(V₁²)
3. **Telescoping Identity** -- tr(LV₁) = (log p_k)² − (log 2)²
4. **Sum Lower Bound** -- tr(V₁²) = Ω(k · (log k)²)
5. **Convergence Rate** -- |σ* − 1/2| = O(1/k)
6. **Explicit Bound** -- |σ* − 1/2| ≤ A/k for certified A
7. **Curvature Divergence** -- d²E/dσ² → ∞
8. **C(k) Asymptotics** -- C(k) = 1 + a₁/ln k + O(1/ln²k)
9. **Full Proof Chain** -- All steps verified together

Physics basis: σ* minimises Frobenius energy E(σ) = (1/2k)Σ λ_j²;
convergence to 1/2 proves H(1/2) = L_k is the unique thermodynamic
attractor.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tnfr.riemann.convergence_proof import (  # Data structures; Dusart bounds; Proof steps; Explicit bound; C(k) asymptotics; Integration
    CKAsymptoticFit,
    CurvatureGrowthResult,
    DusartVerification,
    ExplicitBoundResult,
    FormalConvergenceProof,
    ProofStep,
    compute_explicit_bound_constant,
    dusart_lower_bound,
    dusart_upper_bound,
    fit_ck_asymptotics,
    prove_bilinear_decomposition,
    prove_convergence_rate,
    prove_curvature_divergence,
    prove_explicit_bound,
    prove_sum_lower_bound,
    prove_telescoping,
    run_formal_convergence_proof,
    scan_effective_constant,
    verify_dusart_bounds,
)

# ============================================================================
# 1. Dusart Prime Bounds
# ============================================================================


class TestDusartBounds:
    """Two-sided analytic bounds on p_n (Dusart 2010)."""

    @pytest.mark.parametrize("n", [2, 3, 5, 10, 50, 100])
    def test_lower_bound_positive(self, n: int) -> None:
        """Lower bound is positive for n ≥ 2."""
        lb = dusart_lower_bound(n)
        assert lb > 0

    @pytest.mark.parametrize("n", [6, 10, 50, 100, 500])
    def test_upper_greater_than_lower(self, n: int) -> None:
        """Upper bound exceeds lower bound for n ≥ 6."""
        assert dusart_upper_bound(n) > dusart_lower_bound(n)

    @pytest.mark.parametrize("k", [10, 50, 100])
    def test_verify_all_within_bounds(self, k: int) -> None:
        """All k primes lie within Dusart bounds."""
        v = verify_dusart_bounds(k)
        assert isinstance(v, DusartVerification)
        assert v.k == k
        assert v.all_within_bounds

    def test_relative_gaps_bounded(self) -> None:
        """Relative gaps are small (< 40% for k=100)."""
        v = verify_dusart_bounds(100)
        assert v.max_relative_gap_lower < 0.4
        assert v.max_relative_gap_upper < 0.4


# ============================================================================
# 2. Bilinear Decomposition: σ* = 1/2 − tr(LV₁)/tr(V₁²)
# ============================================================================


class TestBilinearDecomposition:
    """Lemma 1: exact algebra for the optimal σ*."""

    @pytest.mark.parametrize("k", [10, 30, 50])
    def test_verified(self, k: int) -> None:
        step = prove_bilinear_decomposition(k)
        assert isinstance(step, ProofStep)
        assert "Lemma 1" in step.name
        assert step.verified

    def test_sigma_star_near_half(self) -> None:
        """σ* should be close to 1/2 for moderate k."""
        step = prove_bilinear_decomposition(50)
        sigma_star = step.certificate["sigma_star"]
        assert abs(sigma_star - 0.5) < 0.1

    def test_certificate_keys(self) -> None:
        """Certificate contains expected keys."""
        step = prove_bilinear_decomposition(50)
        assert "sigma_star" in step.certificate
        assert "tr_LV1" in step.certificate
        assert "tr_V1_sq" in step.certificate


# ============================================================================
# 3. Telescoping Identity: tr(LV₁) = (log p_k)² − (log 2)²
# ============================================================================


class TestTelescopingIdentity:
    """Lemma 2: exact sum by telescoping."""

    @pytest.mark.parametrize("k", [5, 20, 100])
    def test_verified(self, k: int) -> None:
        step = prove_telescoping(k)
        assert "Lemma 2" in step.name
        assert step.verified

    def test_machine_precision(self) -> None:
        """Relative error should be at machine precision."""
        step = prove_telescoping(50)
        assert step.certificate["relative_error"] < 1e-12

    def test_certificate_values(self) -> None:
        """Check that certificate contains expected keys."""
        step = prove_telescoping(30)
        assert "numerical" in step.certificate
        assert "telescoping" in step.certificate


# ============================================================================
# 4. Sum Lower Bound: tr(V₁²) ≥ ⌊k/2⌋ · (log p_{⌈k/2⌉})²
# ============================================================================


class TestSumLowerBound:
    """Lemma 3: lower bound on the denominator."""

    @pytest.mark.parametrize("k", [10, 50, 200])
    def test_verified(self, k: int) -> None:
        step = prove_sum_lower_bound(k)
        assert "Lemma 3" in step.name
        assert step.verified

    def test_ratio_greater_than_one(self) -> None:
        """tr(V₁²) / lower_bound > 1."""
        step = prove_sum_lower_bound(50)
        assert step.certificate["ratio"] > 1.0


# ============================================================================
# 5. Convergence Rate: |σ* − 1/2| = O(1/k)
# ============================================================================


class TestConvergenceRate:
    """Theorem 1: O(1/k) rate via Lemmas 1-3."""

    @pytest.mark.parametrize("k", [10, 50, 100])
    def test_verified(self, k: int) -> None:
        step = prove_convergence_rate(k)
        assert "Theorem 1" in step.name
        assert step.verified

    def test_effective_constant_finite(self) -> None:
        """C(k) = k|σ*-1/2| should be finite and positive."""
        step = prove_convergence_rate(50)
        ck = step.certificate["C_k"]
        assert 0 < ck < 100


# ============================================================================
# 6. Explicit Bound: |σ*(k) − 1/2| ≤ A/k
# ============================================================================


class TestExplicitBound:
    """Theorem 2: explicit constant A."""

    def test_verified_with_known_A(self) -> None:
        """Explicit bound holds for A ≈ 3.33."""
        step = prove_explicit_bound(50, A=4.0)
        assert "Theorem 2" in step.name
        assert step.verified

    def test_fails_for_too_small_A(self) -> None:
        """Bound should fail if A is too small."""
        step = prove_explicit_bound(50, A=0.1)
        assert not step.verified

    def test_scan_effective_constant(self) -> None:
        """Scanning C(k) for k up to 500 should not crash."""
        k_arr, c_arr = scan_effective_constant(500)
        assert len(k_arr) == len(c_arr)
        assert len(k_arr) > 0
        assert np.all(c_arr > 0)

    def test_compute_explicit_bound_constant(self) -> None:
        """Compute A constant for moderate k_max."""
        result = compute_explicit_bound_constant(500)
        assert isinstance(result, ExplicitBoundResult)
        assert result.A > 0
        assert result.bound_holds_all


# ============================================================================
# 7. Curvature Divergence: d²E/dσ² → ∞
# ============================================================================


class TestCurvatureDivergence:
    """Theorem 3: Frobenius energy well deepens unboundedly."""

    def test_proof_step_verified(self) -> None:
        step = prove_curvature_divergence(50)
        assert "Theorem 3" in step.name
        assert step.verified

    def test_curvature_grows(self) -> None:
        """Curvature at k=100 > curvature at k=10."""
        s10 = prove_curvature_divergence(10)
        s100 = prove_curvature_divergence(100)
        assert s100.certificate["curvature"] > s10.certificate["curvature"]


# ============================================================================
# 8. C(k) Asymptotics: C(k) = 1 + a₁/ln k + …
# ============================================================================


class TestCKAsymptotics:
    """Second-order fit of the effective constant."""

    def test_fit_returns_valid_result(self) -> None:
        ks = [20, 50, 100, 200, 500]
        result = fit_ck_asymptotics(ks)
        assert isinstance(result, CKAsymptoticFit)
        assert result.r_squared > 0.95

    def test_a1_positive(self) -> None:
        """Leading correction a₁ > 0 (C approaches 1 from above)."""
        result = fit_ck_asymptotics([20, 50, 100, 200, 500])
        assert result.a1 > 0


# ============================================================================
# 9. Full Proof Chain
# ============================================================================


class TestFullProof:
    """Integration test: run_formal_convergence_proof."""

    def test_proof_completes(self) -> None:
        """Full proof chain finishes without error."""
        proof = run_formal_convergence_proof(k_values=[10, 50], k_max_scan=200)
        assert isinstance(proof, FormalConvergenceProof)
        assert proof.all_verified
        assert proof.explicit_A > 0

    def test_proof_summary_nonempty(self) -> None:
        proof = run_formal_convergence_proof(k_values=[10, 30], k_max_scan=100)
        assert len(proof.summary) > 50

    def test_proof_steps_count(self) -> None:
        """Should have at least 6 proof steps (Lemmas + Theorems + Corollary)."""
        proof = run_formal_convergence_proof(k_values=[10, 30], k_max_scan=100)
        assert len(proof.proof_steps) >= 6

    def test_all_steps_verified(self) -> None:
        proof = run_formal_convergence_proof(k_values=[10, 50], k_max_scan=200)
        for step in proof.proof_steps:
            assert step.verified, f"{step.name} failed verification"

    def test_explicit_A_reasonable(self) -> None:
        """A should be somewhere in [1, 10] based on empirical data."""
        proof = run_formal_convergence_proof(k_values=[10, 50], k_max_scan=500)
        assert 1.0 < proof.explicit_A < 10.0

    def test_curvature_divergence_in_proof(self) -> None:
        proof = run_formal_convergence_proof(k_values=[10, 50], k_max_scan=200)
        assert proof.curvature.all_exceed_bound
        assert proof.curvature.growth_unbounded

    def test_dusart_in_proof(self) -> None:
        proof = run_formal_convergence_proof(k_values=[10, 50], k_max_scan=200)
        assert proof.dusart.all_within_bounds
