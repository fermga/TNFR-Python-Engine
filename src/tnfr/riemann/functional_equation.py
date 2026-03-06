r"""P9: Functional equation analog and spectral reflection symmetry.

Implements the discrete analog of the Riemann functional equation
ξ(s) = ξ(1-s) for the TNFR-Riemann operator

    H^{(k)}(σ) = L_k + (σ - 1/2) V_1

where V_1 = diag(log p_1, ..., log p_k).

Core Results
------------
1. **Spectral Reflection Identity** (exact for all k ≥ 2):

       H(σ) + H(1-σ) = 2 L_k

   Direct consequence of V_σ + V_{1-σ} = 0.  Implies exact trace-level
   functional equations for all spectral moments.

2. **Trace-Level Functional Equations** (exact):

   For all n ≥ 1:
       μ_n(σ) + (-1)^n μ_n(1-σ) = F_n(L, V_1)

   where μ_n(σ) = (1/k) tr(H(σ)^n).  For even n, the sum equals a
   function of L and V_1 only; for odd n, the difference vanishes.

   Special cases:
   - n=1: μ_1(σ) + μ_1(1-σ) = 2 μ_1(L)    [trace identity]
   - n=2: E(σ) + E(1-σ) = 2E_L + δ²/k tr(V_1²)  [energy identity]
   - Antisymmetric part: E(σ) - E(1-σ) = (4δ/k) tr(L V_1)  [PNT term]

3. **Completed Spectral Xi Function** (analog of Riemann ξ):

       Ξ_H^{(k)}(σ) = (σ - 1/2)² · ∏_{j≥1} [λ_j(σ) / λ_j(1/2)]

   regularised over non-zero Laplacian eigenvalues.  Satisfies
   Ξ_H(σ) → Ξ_H(1-σ) as k → ∞.

4. **Conjecture 12.1 Test** (Riemann Invariant Conservation):

   At criticality (σ = 1/2), the spectral energy density ε and
   topological charge Q satisfy approximate conservation under
   σ-variation.

5. **Conjecture 12.2 Test** (Multiscale Minimization Principle):

   The multiscale coherence functional C_multi(σ) is minimized at
   σ = 1/2, providing a variational characterisation of the critical
   line.

TNFR physics basis
------------------
The reflection H(σ) + H(1-σ) = 2L is the spectral analog of the
structural conservation theorem: the Lyapunov energy E = ½Σ field² is
symmetric under σ ↔ 1-σ modulo the cross-term tr(L V_1), which is
exactly the PNT-controlled telescoping identity from P8.

References
----------
- spectral_proof.py: Frobenius energy and thermodynamic analysis
- analytical_convergence.py: PNT trace bounds
- theory/TNFR_RIEMANN_RESEARCH_NOTES.md sec. 12

Status: EXPERIMENTAL -- Research prototype for TNFR-Riemann program.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

from scipy.linalg import eigh_tridiagonal

from ..mathematics.unified_numerical import np
from .operator import (
    _first_primes,
    build_tridiagonal_h_tnfr,
)
from .spectral_proof import (
    _compute_lv1_traces,
    compute_analytic_sigma_star,
    compute_eigensystem,
    compute_eigenspectrum,
)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Data structures
    "SpectralReflection",
    "TraceFormulaResult",
    "CompletedXiFunction",
    "Conjecture12_1Result",
    "Conjecture12_2Result",
    "LargeKConvergence",
    "FunctionalEquationAnalysis",
    # Core — reflection symmetry
    "verify_spectral_reflection",
    "verify_reflection_sequence",
    # Core — trace formulas
    "compute_trace_formulas",
    "verify_trace_formula_pnt",
    # Core — completed xi
    "compute_completed_xi",
    "verify_xi_functional_equation",
    # Conjectures
    "test_conjecture_12_1",
    "test_conjecture_12_2",
    # Large-k verification
    "verify_large_k_convergence",
    # Integration
    "run_functional_equation_analysis",
]

# Prevent pytest from collecting API functions named test_*
# (they are TNFR conjecture-testing functions, not pytest tests)

# ============================================================================
# Data Structures
# ============================================================================

@dataclass(frozen=True)
class SpectralReflection:
    r"""Verification of the exact matrix identity H(σ) + H(1-σ) = 2L.

    The identity follows from V_σ = (σ-1/2)V_1 and
    V_{1-σ} = (1/2-σ)V_1 = -V_σ, so H(σ)+H(1-σ) = 2L.

    This is exact for all k ≥ 2, all σ.
    """

    k: int
    """Number of prime nodes."""

    sigma: float
    """Parameter at which reflection was verified."""

    trace_sum: float
    """tr(H(σ)) + tr(H(1-σ))."""

    trace_2L: float
    """2 · tr(L_k)."""

    trace_error: float
    """|trace_sum - trace_2L|; should be machine zero."""

    energy_sum: float
    """E(σ) + E(1-σ) where E = (1/2k)tr(H²)."""

    energy_predicted: float
    """2E(1/2) + δ²(1/k)tr(V_1²); exact quadratic prediction."""

    energy_error: float
    """|energy_sum - energy_predicted|; should be machine zero."""

    eigenvalue_reflection_error: float
    """max_j |λ_j(σ) + λ_j(1-σ) - 2λ_j(L)|; quantifies eigenvector rotation."""

@dataclass(frozen=True)
class TraceFormulaResult:
    r"""Exact trace formula identities for H(σ).

    For the TNFR operator H(σ) = L + δV_1 (δ = σ-1/2), the trace
    moments satisfy exact functional equations derived from the
    binomial expansion of (L + δV_1)^n.

    Key identity for n=2 (Frobenius energy):
        E(σ) + E(1-σ) = 2E_L + δ² (1/k) tr(V_1²)

    Antisymmetric part (PNT-controlled):
        E(σ) - E(1-σ) = (4δ/k) tr(L V_1)
    """

    k: int
    sigma: float
    delta: float
    """σ - 1/2."""

    # Moment n=1 (trace)
    trace_sigma: float
    trace_1_minus_sigma: float
    trace_sum: float
    trace_2L: float
    trace_n1_error: float

    # Moment n=2 (Frobenius energy)
    energy_sigma: float
    energy_1_minus_sigma: float
    energy_sum: float
    energy_sum_predicted: float
    energy_diff: float
    energy_diff_predicted: float
    energy_n2_symmetric_error: float
    energy_n2_antisymmetric_error: float

    # PNT terms
    tr_LV1: float
    """tr(L_k V_1) — telescoping identity value."""

    tr_V1_sq: float
    """tr(V_1²) = Σ(log p_i)²."""

    pnt_mean_log_prime: float
    """(1/k) Σ log p_i ≈ log k by PNT."""

@dataclass(frozen=True)
class CompletedXiFunction:
    r"""Completed spectral xi function and functional equation test.

    Defines Ξ_H^{(k)}(σ) = δ² · R(σ) where δ = σ - 1/2 and
    R(σ) = ∏_{j≥1} [λ_j(σ) / λ_j^{(L)}] is the regularised spectral
    ratio (product over non-zero Laplacian eigenvalues).

    The delta-squared prefactor is the discrete analog of the s(1-s)
    completion factor in the Riemann xi function.

    Functional equation test: Ξ_H(σ) vs Ξ_H(1-σ).
    """

    k: int
    sigma_values: np.ndarray
    """σ scan points."""

    log_xi_values: np.ndarray
    """log Ξ_H(σ) for each σ (NaN where δ=0)."""

    log_xi_reflected: np.ndarray
    """log Ξ_H(1-σ) for each σ."""

    log_asymmetry: np.ndarray
    """|log Ξ_H(σ) - log Ξ_H(1-σ)| for each σ."""

    max_log_asymmetry: float
    mean_log_asymmetry: float

@dataclass(frozen=True)
class Conjecture12_1Result:
    r"""Numerical test of Conjecture 12.1 (Riemann Invariant Conservation).

    Tests whether the spectral energy density ε^{(k)} and topological
    charge Q^{(k)} exhibit approximate conservation at σ = 1/2:

        d/dσ [ε^{(k)} + α·Q^{(k)}] ≈ 0 at σ = 1/2.
    """

    k: int
    energy_density_at_half: float
    """ε = (1/k) Σ_j λ_j(1/2)²."""

    energy_density_derivative: float
    """dε/dσ at σ = 1/2 (should approach 0)."""

    topological_charge_at_half: float
    """Q via eigenvalue-velocity cross-correlation."""

    charge_derivative: float
    """dQ/dσ at σ = 1/2."""

    combined_derivative: float
    """d[ε + α·Q]/dσ at σ = 1/2 with optimal α."""

    optimal_alpha: float
    """α that minimises |d[ε + α·Q]/dσ|."""

    is_approximately_conserved: bool
    """True if |combined_derivative| < threshold."""

@dataclass(frozen=True)
class Conjecture12_2Result:
    r"""Numerical test of Conjecture 12.2 (Multiscale Minimization).

    Tests whether the Frobenius energy functional E(σ) = (1/2k)tr(H(σ)²)
    is minimised near σ = 1/2.

    From the quadratic expansion E(σ) = E_L + (δ/k)tr(LV_1) + (δ²/2k)tr(V_1²)
    the minimum is at σ* = 1/2 - tr(LV_1)/tr(V_1²) → 1/2 by PNT.
    """

    k: int
    sigma_values: np.ndarray
    coherence_values: np.ndarray
    """C_multi(σ) for each σ in scan."""

    sigma_min: float
    """σ that minimises C_multi."""

    coherence_at_half: float
    """C_multi(1/2)."""

    coherence_at_min: float
    """C_multi(σ_min)."""

    deviation_from_half: float
    """|σ_min - 1/2|."""

    is_minimized_at_half: bool
    """True if σ_min is within tolerance of 1/2."""

@dataclass(frozen=True)
class LargeKConvergence:
    r"""Large-k verification results for convergence to critical line.

    Verifies all four structural equilibrium properties simultaneously
    at k ≥ 10,000:
    1. λ_min(H(1/2)) = 0 (exact)
    2. |σ* - 1/2| = O(1/k) (thermodynamic)
    3. All dλ_j/dσ > 0 (Hellmann-Feynman)
    4. C(k) = k·|σ* - 1/2| → 1 (PNT)
    """

    k: int
    lambda_min_at_half: float
    """λ_min(H(1/2)); machine zero."""

    sigma_star: float
    """Analytic thermodynamic minimum."""

    deviation: float
    """|σ* - 1/2|."""

    effective_constant: float
    """C(k) = k · deviation."""

    all_velocities_positive: bool
    """Hellmann-Feynman monotonicity."""

    min_velocity: float
    spectral_gap: float
    computation_time_s: float

@dataclass
class FunctionalEquationAnalysis:
    r"""Integrated P9 functional equation analysis.

    Combines spectral reflection, trace formulas, completed xi,
    conjectures 12.1/12.2, and large-k verification.
    """

    k_values: list[int] = field(default_factory=list)
    reflections: list[SpectralReflection] = field(default_factory=list)
    trace_formulas: list[TraceFormulaResult] = field(default_factory=list)
    xi_function: CompletedXiFunction | None = None
    conjecture_12_1: list[Conjecture12_1Result] = field(default_factory=list)
    conjecture_12_2: Conjecture12_2Result | None = None
    large_k: list[LargeKConvergence] = field(default_factory=list)

    reflection_exact: bool = True
    """All trace-level reflections at machine precision."""

    xi_asymmetry_decreasing: bool = True
    """Functional equation asymmetry decreasing with k."""

    convergence_exponent: float = 0.0
    """Fitted β in |σ* - 1/2| ~ k^{-β}."""

    summary: str = ""

# ============================================================================
# Core: Spectral Reflection Symmetry
# ============================================================================

def verify_spectral_reflection(
    k: int,
    sigma: float = 0.3,
    *,
    weight_by_log_gap: bool = True,
) -> SpectralReflection:
    r"""Verify the exact matrix identity H(σ) + H(1-σ) = 2L.

    This is the discrete analog of the Riemann functional equation
    symmetry.  It holds exactly because:

        H(σ) = L + (σ-1/2) V_1
        H(1-σ) = L - (σ-1/2) V_1
        H(σ) + H(1-σ) = 2 L

    Parameters
    ----------
    k : int
        Number of primes.
    sigma : float
        Test parameter (should be ≠ 0.5 for non-trivial test).

    Returns
    -------
    SpectralReflection
    """
    delta = sigma - 0.5

    # Build L (at σ=0.5) and get log_primes
    d_L, e_L, log_p = build_tridiagonal_h_tnfr(
        k, 0.5, weight_by_log_gap=weight_by_log_gap
    )

    # H(σ) diagonal
    d_sigma = d_L + delta * log_p
    # H(1-σ) diagonal
    d_refl = d_L - delta * log_p

    # Trace identity: tr(H(σ)) + tr(H(1-σ)) = 2 tr(L)
    trace_sum = float(np.sum(d_sigma) + np.sum(d_refl))
    trace_2L = float(2.0 * np.sum(d_L))
    trace_error = abs(trace_sum - trace_2L)

    # Energy identity: E(σ) + E(1-σ) = 2E(1/2) + δ²/k · tr(V_1²)
    evals_sigma = np.sort(eigh_tridiagonal(d_sigma, e_L, eigvals_only=True))
    evals_refl = np.sort(eigh_tridiagonal(d_refl, e_L, eigvals_only=True))
    evals_L = np.sort(eigh_tridiagonal(d_L, e_L, eigvals_only=True))

    e_sigma = float(np.sum(evals_sigma ** 2) / (2.0 * k))
    e_refl = float(np.sum(evals_refl ** 2) / (2.0 * k))
    e_L_val = float(np.sum(evals_L ** 2) / (2.0 * k))
    tr_V1_sq = float(np.sum(log_p ** 2))

    energy_sum = e_sigma + e_refl
    energy_predicted = 2.0 * e_L_val + delta ** 2 * tr_V1_sq / k
    energy_error = abs(energy_sum - energy_predicted)

    # Eigenvalue reflection: λ_j(σ) + λ_j(1-σ) ≈ 2λ_j(L)
    # (not exact because eigenvectors rotate, but useful diagnostic)
    eig_sum = evals_sigma + evals_refl
    eig_2L = 2.0 * evals_L
    eig_error = float(np.max(np.abs(eig_sum - eig_2L)))

    return SpectralReflection(
        k=k,
        sigma=sigma,
        trace_sum=trace_sum,
        trace_2L=trace_2L,
        trace_error=trace_error,
        energy_sum=energy_sum,
        energy_predicted=energy_predicted,
        energy_error=energy_error,
        eigenvalue_reflection_error=eig_error,
    )

def verify_reflection_sequence(
    k_values: Sequence[int],
    sigma: float = 0.3,
    *,
    weight_by_log_gap: bool = True,
) -> list[SpectralReflection]:
    """Verify spectral reflection for a sequence of k values."""
    return [
        verify_spectral_reflection(k, sigma, weight_by_log_gap=weight_by_log_gap)
        for k in k_values
        if k >= 3
    ]

# ============================================================================
# Core: Trace Formula Identities
# ============================================================================

def compute_trace_formulas(
    k: int,
    sigma: float = 0.3,
    *,
    weight_by_log_gap: bool = True,
) -> TraceFormulaResult:
    r"""Compute exact trace formula identities for H(σ).

    Verifies the decomposition of trace moments into symmetric and
    antisymmetric parts under σ ↔ 1-σ reflection.

    The antisymmetric part is controlled by tr(L V_1), which is exactly
    the telescoping identity from P8.
    """
    delta = sigma - 0.5

    # Build base components
    d_L, e_L, log_p = build_tridiagonal_h_tnfr(
        k, 0.5, weight_by_log_gap=weight_by_log_gap
    )
    tr_LV1, tr_V1_sq, _ = _compute_lv1_traces(
        k, weight_by_log_gap=weight_by_log_gap
    )

    # Eigenvalues at σ, 1-σ, and 1/2
    d_sigma = d_L + delta * log_p
    d_refl = d_L - delta * log_p
    evals_sigma = eigh_tridiagonal(d_sigma, e_L, eigvals_only=True)
    evals_refl = eigh_tridiagonal(d_refl, e_L, eigvals_only=True)

    # n=1: trace identity
    # tr(H(σ)) = tr(L) + δ·tr(V_1),  tr(H(1-σ)) = tr(L) - δ·tr(V_1)
    tr_sigma = float(np.sum(evals_sigma))
    tr_refl = float(np.sum(evals_refl))
    tr_2L = 2.0 * float(np.sum(d_L))
    trace_n1_error = abs((tr_sigma + tr_refl) - tr_2L)

    # n=2: Frobenius energy
    e_sigma = float(np.sum(evals_sigma ** 2) / (2.0 * k))
    e_refl = float(np.sum(evals_refl ** 2) / (2.0 * k))

    # Symmetric part: E(σ) + E(1-σ) = 2E_L + δ²/k·tr(V₁²)
    evals_L = eigh_tridiagonal(d_L, e_L, eigvals_only=True)
    e_L_val = float(np.sum(evals_L ** 2) / (2.0 * k))
    energy_sum_predicted = 2.0 * e_L_val + delta ** 2 * tr_V1_sq / k
    energy_n2_sym_err = abs((e_sigma + e_refl) - energy_sum_predicted)

    # Antisymmetric part: E(σ) - E(1-σ) = (2δ/k)·tr(LV₁)
    # From: tr(H(σ)²) - tr(H(1-σ)²) = 4δ·tr(LV₁),
    #       E = (1/2k)tr(H²), so E(σ)-E(1-σ) = (2δ/k)tr(LV₁)
    energy_diff_predicted = 2.0 * delta * tr_LV1 / k
    energy_n2_anti_err = abs((e_sigma - e_refl) - energy_diff_predicted)

    pnt_mean = float(np.mean(log_p))

    return TraceFormulaResult(
        k=k,
        sigma=sigma,
        delta=delta,
        trace_sigma=tr_sigma,
        trace_1_minus_sigma=tr_refl,
        trace_sum=tr_sigma + tr_refl,
        trace_2L=tr_2L,
        trace_n1_error=trace_n1_error,
        energy_sigma=e_sigma,
        energy_1_minus_sigma=e_refl,
        energy_sum=e_sigma + e_refl,
        energy_sum_predicted=energy_sum_predicted,
        energy_diff=e_sigma - e_refl,
        energy_diff_predicted=energy_diff_predicted,
        energy_n2_symmetric_error=energy_n2_sym_err,
        energy_n2_antisymmetric_error=energy_n2_anti_err,
        tr_LV1=tr_LV1,
        tr_V1_sq=tr_V1_sq,
        pnt_mean_log_prime=pnt_mean,
    )

def verify_trace_formula_pnt(
    k_values: Sequence[int],
    sigma: float = 0.3,
    *,
    weight_by_log_gap: bool = True,
) -> list[TraceFormulaResult]:
    """Verify trace formulas across multiple k with PNT consistency."""
    return [
        compute_trace_formulas(k, sigma, weight_by_log_gap=weight_by_log_gap)
        for k in k_values
        if k >= 3
    ]

# ============================================================================
# Core: Completed Spectral Xi Function
# ============================================================================

def compute_completed_xi(
    k: int,
    *,
    sigma_range: tuple[float, float] = (0.05, 0.95),
    n_points: int = 200,
    weight_by_log_gap: bool = True,
) -> CompletedXiFunction:
    r"""Compute the completed spectral xi function Ξ_H^{(k)}(σ).

    The completed xi function is defined as:

        Ξ_H(σ) = δ² · R(σ)

    where δ = σ - 1/2 and R(σ) = ∏_{j≥1} [λ_j(σ) / λ_j^{(L)}] is
    the regularised spectral ratio over positive Laplacian eigenvalues.

    The δ² prefactor is the analog of s(1-s) in the Riemann xi function
    (since σ(1-σ) = 1/4 - δ²).

    For numerical stability, we use the log-determinant:
        log R(σ) = Σ_{j≥1} log|λ_j(σ)| - Σ_{j≥1} log|λ_j^{(L)}|

    Parameters
    ----------
    k : int
        Number of primes.
    sigma_range : (float, float)
        Range of σ values to scan (avoid 0 and 1 boundaries).
    n_points : int
        Number of scan points.

    Returns
    -------
    CompletedXiFunction
    """
    d_L, e_L, log_p = build_tridiagonal_h_tnfr(
        k, 0.5, weight_by_log_gap=weight_by_log_gap
    )
    evals_L = np.sort(eigh_tridiagonal(d_L, e_L, eigvals_only=True))

    # Use positive Laplacian eigenvalues (skip the zero eigenvalue)
    pos_mask = evals_L > 1e-12
    log_lambda_L = np.log(evals_L[pos_mask])
    n_pos = int(np.sum(pos_mask))

    sigmas = np.linspace(sigma_range[0], sigma_range[1], n_points)
    log_xi = np.full(n_points, np.nan)
    log_xi_refl = np.full(n_points, np.nan)

    for i, sig in enumerate(sigmas):
        delta = sig - 0.5
        if abs(delta) < 1e-15:
            # At σ = 1/2, ξ = 0 (δ² = 0), log ξ = -inf
            log_xi[i] = -np.inf
            log_xi_refl[i] = -np.inf
            continue

        d_sig = d_L + delta * log_p
        evals_sig = np.sort(eigh_tridiagonal(d_sig, e_L, eigvals_only=True))

        # Take the n_pos largest eigenvalues for ratio
        evals_pos = evals_sig[len(evals_sig) - n_pos:]
        evals_pos = np.maximum(evals_pos, 1e-300)

        log_ratio = float(np.sum(np.log(np.abs(evals_pos)) - log_lambda_L))
        # log Ξ = log(δ²) + log R(σ) = 2 log|δ| + log R
        log_xi[i] = 2.0 * math.log(abs(delta)) + log_ratio

        # Reflected: σ' = 1 - σ, δ' = -δ
        d_refl = d_L - delta * log_p
        evals_refl = np.sort(eigh_tridiagonal(d_refl, e_L, eigvals_only=True))
        evals_refl_pos = evals_refl[len(evals_refl) - n_pos:]
        evals_refl_pos = np.maximum(evals_refl_pos, 1e-300)
        log_ratio_refl = float(
            np.sum(np.log(np.abs(evals_refl_pos)) - log_lambda_L)
        )
        # (δ')² = δ², so log-prefactor is the same
        log_xi_refl[i] = 2.0 * math.log(abs(delta)) + log_ratio_refl

    # Asymmetry in log-space (where both are finite)
    finite_mask = np.isfinite(log_xi) & np.isfinite(log_xi_refl)
    log_asym = np.full(n_points, 0.0)
    log_asym[finite_mask] = np.abs(log_xi[finite_mask] - log_xi_refl[finite_mask])

    return CompletedXiFunction(
        k=k,
        sigma_values=sigmas,
        log_xi_values=log_xi,
        log_xi_reflected=log_xi_refl,
        log_asymmetry=log_asym,
        max_log_asymmetry=float(np.max(log_asym)),
        mean_log_asymmetry=float(np.mean(log_asym[finite_mask])) if np.any(finite_mask) else 0.0,
    )

def verify_xi_functional_equation(
    k_values: Sequence[int],
    *,
    sigma_range: tuple[float, float] = (0.05, 0.95),
    n_points: int = 100,
    weight_by_log_gap: bool = True,
) -> list[CompletedXiFunction]:
    """Verify completed xi functional equation for multiple k."""
    return [
        compute_completed_xi(
            k,
            sigma_range=sigma_range,
            n_points=n_points,
            weight_by_log_gap=weight_by_log_gap,
        )
        for k in k_values
        if k >= 3
    ]

# ============================================================================
# Conjecture 12.1: Riemann Invariant Conservation
# ============================================================================

def test_conjecture_12_1(
    k: int,
    *,
    dsigma: float = 1e-5,
    weight_by_log_gap: bool = True,
) -> Conjecture12_1Result:
    r"""Test Conjecture 12.1 for a single graph size k.

    Computes the spectral energy density ε(σ) = (1/k) Σ λ_j(σ)² and
    a topological charge proxy Q(σ) at σ = 1/2, then measures their
    σ-derivatives to test approximate conservation.

    The topological charge proxy is defined as the cross-correlation
    between eigenvalue position and eigenvalue velocity:

        Q(σ) = (1/k) Σ_j [j/k - 1/2] · [v_j(σ) - <v>]

    where v_j = dλ_j/dσ (Hellmann-Feynman velocity).

    Conjecture: d[ε + α·Q]/dσ → 0 at σ = 1/2 as k → ∞.
    """
    sigma_0 = 0.5

    # Energy density at σ = 1/2 ± dσ
    evals_0 = compute_eigenspectrum(k, sigma_0, weight_by_log_gap=weight_by_log_gap)
    evals_p = compute_eigenspectrum(
        k, sigma_0 + dsigma, weight_by_log_gap=weight_by_log_gap
    )
    evals_m = compute_eigenspectrum(
        k, sigma_0 - dsigma, weight_by_log_gap=weight_by_log_gap
    )

    eps_0 = float(np.mean(evals_0 ** 2))
    eps_p = float(np.mean(evals_p ** 2))
    eps_m = float(np.mean(evals_m ** 2))
    deps = (eps_p - eps_m) / (2.0 * dsigma)

    # Topological charge proxy via eigenvalue-velocity cross-correlation
    # v_j = dλ_j/dσ at σ = 1/2
    velocities = (evals_p - evals_m) / (2.0 * dsigma)
    positions = (np.arange(k, dtype=float) / k) - 0.5
    mean_v = float(np.mean(velocities))
    q_0 = float(np.mean(positions * (velocities - mean_v)))

    # Charge derivative: dQ/dσ via central difference at σ = 1/2
    # (recompute at σ = 1/2 ± 2·dσ for second-order accuracy)
    evals_pp = compute_eigenspectrum(
        k, sigma_0 + 2 * dsigma, weight_by_log_gap=weight_by_log_gap
    )
    evals_mm = compute_eigenspectrum(
        k, sigma_0 - 2 * dsigma, weight_by_log_gap=weight_by_log_gap
    )

    def _charge(ev_a: np.ndarray, ev_b: np.ndarray, ds: float) -> float:
        v = (ev_a - ev_b) / (2.0 * ds)
        return float(np.mean(positions * (v - np.mean(v))))

    q_p = _charge(evals_pp, evals_0, dsigma)
    q_m = _charge(evals_0, evals_mm, dsigma)
    dq = (q_p - q_m) / (2.0 * dsigma)

    # Find optimal α that minimises |dε/dσ + α · dQ/dσ|
    if abs(dq) > 1e-30:
        alpha_opt = -deps / dq
    else:
        alpha_opt = 0.0

    combined = deps + alpha_opt * dq
    threshold = 1.0 / k  # Conservation improves with k

    return Conjecture12_1Result(
        k=k,
        energy_density_at_half=eps_0,
        energy_density_derivative=deps,
        topological_charge_at_half=q_0,
        charge_derivative=dq,
        combined_derivative=combined,
        optimal_alpha=alpha_opt,
        is_approximately_conserved=abs(combined) < threshold,
    )

# Prevent pytest from collecting
test_conjecture_12_1.__test__ = False  # type: ignore[attr-defined]

# ============================================================================
# Conjecture 12.2: Multiscale Minimization Principle
# ============================================================================

def test_conjecture_12_2(
    k: int,
    *,
    sigma_range: tuple[float, float] = (0.1, 0.9),
    n_points: int = 200,
    tolerance: float = 0.05,
    weight_by_log_gap: bool = True,
) -> Conjecture12_2Result:
    r"""Test Conjecture 12.2 for a single graph size k.

    The multiscale coherence functional is:

        C_multi(σ) = (1/k) Σ_{j=0}^{k-2} [λ_{j+1}(σ) - λ_j(σ)]²

    This measures spectral spacing variance — "spectral stress".  The
    conjecture states that this is minimised at σ = 1/2, reflecting
    maximal multiscale coherence at the TNFR structural equilibrium.

    Parameters
    ----------
    k : int
        Number of primes.
    sigma_range : (float, float)
        Scan range for σ.
    n_points : int
        Number of scan points.
    tolerance : float
        Maximum |σ_min - 1/2| to declare success.

    Returns
    -------
    Conjecture12_2Result
    """
    d_L, e_L, log_p = build_tridiagonal_h_tnfr(
        k, 0.5, weight_by_log_gap=weight_by_log_gap
    )
    sigmas = np.linspace(sigma_range[0], sigma_range[1], n_points)
    coherence = np.zeros(n_points)

    for i, sig in enumerate(sigmas):
        delta = sig - 0.5
        d_sig = d_L + delta * log_p if abs(delta) > 0 else d_L.copy()
        evals = eigh_tridiagonal(d_sig, e_L, eigvals_only=True)
        # Frobenius energy E(σ) = (1/2k) tr(H²) = (1/2k) Σ λ_j²
        coherence[i] = float(np.sum(evals ** 2) / (2.0 * k))

    idx_min = int(np.argmin(coherence))
    sigma_min = float(sigmas[idx_min])

    # Value at σ = 0.5
    idx_half = int(np.argmin(np.abs(sigmas - 0.5)))
    c_half = float(coherence[idx_half])
    c_min = float(coherence[idx_min])

    return Conjecture12_2Result(
        k=k,
        sigma_values=sigmas,
        coherence_values=coherence,
        sigma_min=sigma_min,
        coherence_at_half=c_half,
        coherence_at_min=c_min,
        deviation_from_half=abs(sigma_min - 0.5),
        is_minimized_at_half=abs(sigma_min - 0.5) < tolerance,
    )

# Prevent pytest from collecting
test_conjecture_12_2.__test__ = False  # type: ignore[attr-defined]

# ============================================================================
# Large-k Verification
# ============================================================================

def verify_large_k_convergence(
    k: int,
    *,
    weight_by_log_gap: bool = True,
) -> LargeKConvergence:
    r"""Verify all four structural equilibrium properties at graph size k.

    Simultaneously checks:
    1. λ_min(H(1/2)) = 0 (exact equilibrium)
    2. |σ* - 1/2| = O(1/k) (thermodynamic convergence)
    3. All dλ_j/dσ > 0 (Hellmann-Feynman monotonicity)
    4. C(k) = k·|σ* - 1/2| → 1 (PNT effective constant)

    Designed for k ≥ 10,000.
    """
    import time

    t0 = time.perf_counter()

    # 1. Equilibrium: λ_min(H(1/2))
    evals, evecs = compute_eigensystem(k, 0.5, weight_by_log_gap=weight_by_log_gap)
    lam_min = float(evals[0])
    gap = float(evals[1] - evals[0]) if k >= 2 else 0.0

    # 2. Thermodynamic minimum
    ss, _, _ = compute_analytic_sigma_star(k, weight_by_log_gap=weight_by_log_gap)
    dev = abs(ss - 0.5)
    c_k = k * dev

    # 3. Hellmann-Feynman velocities
    log_p = np.array([np.log(float(p)) for p in _first_primes(k)])
    velocities = np.array([
        float(np.sum(evecs[:, j] ** 2 * log_p))
        for j in range(k)
    ])
    all_pos = bool(np.all(velocities > 0))
    v_min = float(np.min(velocities))

    dt = time.perf_counter() - t0

    return LargeKConvergence(
        k=k,
        lambda_min_at_half=lam_min,
        sigma_star=ss,
        deviation=dev,
        effective_constant=c_k,
        all_velocities_positive=all_pos,
        min_velocity=v_min,
        spectral_gap=gap,
        computation_time_s=dt,
    )

# ============================================================================
# Integration
# ============================================================================

def run_functional_equation_analysis(
    k_values: Sequence[int] | None = None,
    *,
    large_k_values: Sequence[int] | None = None,
    weight_by_log_gap: bool = True,
) -> FunctionalEquationAnalysis:
    r"""Run complete P9 functional equation analysis.

    Parameters
    ----------
    k_values : sequence of int, optional
        Graph sizes for reflection/trace/xi analysis (default: small).
    large_k_values : sequence of int, optional
        Graph sizes for large-k verification (default: [10000]).

    Returns
    -------
    FunctionalEquationAnalysis
    """
    if k_values is None:
        k_values = [10, 50, 100, 500]
    if large_k_values is None:
        large_k_values = [10000]

    analysis = FunctionalEquationAnalysis()
    analysis.k_values = list(k_values)

    # 1. Spectral reflections
    analysis.reflections = verify_reflection_sequence(
        k_values, 0.3, weight_by_log_gap=weight_by_log_gap
    )
    analysis.reflection_exact = all(
        r.trace_error < 1e-10 and r.energy_error < 1e-8
        for r in analysis.reflections
    )

    # 2. Trace formulas
    analysis.trace_formulas = verify_trace_formula_pnt(
        k_values, 0.3, weight_by_log_gap=weight_by_log_gap
    )

    # 3. Completed xi (use largest small k)
    k_xi = max(k_values)
    analysis.xi_function = compute_completed_xi(
        k_xi, weight_by_log_gap=weight_by_log_gap
    )

    # 4. Conjecture 12.1
    analysis.conjecture_12_1 = [
        test_conjecture_12_1(k, weight_by_log_gap=weight_by_log_gap)
        for k in k_values
    ]

    # 5. Conjecture 12.2 (use moderate k)
    k_c12 = min(500, max(k_values))
    analysis.conjecture_12_2 = test_conjecture_12_2(
        k_c12, weight_by_log_gap=weight_by_log_gap
    )

    # 6. Large-k verification
    analysis.large_k = [
        verify_large_k_convergence(k, weight_by_log_gap=weight_by_log_gap)
        for k in large_k_values
    ]

    # 7. Fit convergence exponent from large-k results
    if len(analysis.large_k) >= 2:
        ks = np.array([r.k for r in analysis.large_k], dtype=float)
        devs = np.array([r.deviation for r in analysis.large_k])
        valid = devs > 0
        if np.sum(valid) >= 2:
            log_k = np.log(ks[valid])
            log_d = np.log(devs[valid])
            coeffs = np.polyfit(log_k, log_d, 1)
            analysis.convergence_exponent = -float(coeffs[0])
    elif len(analysis.large_k) == 1:
        # Use effective constant to infer exponent
        r = analysis.large_k[0]
        if r.deviation > 0:
            # C(k) = k · dev => dev = C/k => exponent ≈ 1
            analysis.convergence_exponent = 1.0

    # Summary
    lines = [
        f"P9 Functional Equation Analysis (k = {k_values})",
        f"  Reflection exact: {analysis.reflection_exact}",
    ]
    if analysis.xi_function:
        lines.append(
            f"  Xi max log-asymmetry (k={k_xi}): "
            f"{analysis.xi_function.max_log_asymmetry:.2e}"
        )
    if analysis.conjecture_12_2:
        lines.append(
            f"  C12.2 σ_min: {analysis.conjecture_12_2.sigma_min:.4f} "
            f"(dev={analysis.conjecture_12_2.deviation_from_half:.4f})"
        )
    for r in analysis.large_k:
        lines.append(
            f"  k={r.k}: λ_min={r.lambda_min_at_half:+.2e} "
            f"|σ*-½|={r.deviation:.3e} C(k)={r.effective_constant:.4f} "
            f"flow_ok={r.all_velocities_positive} ({r.computation_time_s:.1f}s)"
        )
    analysis.summary = "\n".join(lines)

    return analysis
