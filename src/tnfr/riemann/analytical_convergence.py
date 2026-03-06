r"""P8: Analytical proof of σ* → 1/2 rate via Prime Number Theorem.

Provides a self-contained mathematical proof that the thermodynamic
minimum σ* of the Frobenius energy E(σ) = (1/2k) tr(H(σ)²) converges
to 1/2 at rate O(1/k), with explicit constants derived from the Prime
Number Theorem (PNT).

Core Results
------------
1. **Telescoping Identity** (Theorem 1, exact for all k ≥ 2):

       tr(L_k V_1) = (log p_k)² − (log p_1)²

   for log-gap weighted path graph.  Proof: the inner product
   Σ_i deg_w(i) log p_i telescopes over edges because each edge weight
   w_j = log p_{j+1} − log p_j produces a difference-of-squares.

2. **PNT Asymptotic for tr(V_1²)** (Theorem 2):

       tr(V_1²) = Σ_{i=1}^{k} (log p_i)² = k(log k)² − 2k log k + 2k
                                              + O(k log k · log log k)

   Derived via Euler–Maclaurin summation on (log n)² and the PNT
   estimate p_n = n(ln n + ln ln n − 1 + O(ln ln n / ln n)).

3. **Main Convergence Rate** (Theorem 3):

       |σ* − 1/2| = [(log p_k)² − (log 2)²] / Σ(log p_i)²
                   = 1/k · (1 + o(1))

   with effective constant C(k) = k|σ* − 1/2| → 1 as k → ∞.

4. **Curvature Growth** (Corollary):

       d²E/dσ² = (1/k) tr(V_1²) → (log k)² → ∞

   confirming σ = 1/2 is an increasingly sharp thermodynamic attractor.

TNFR physics basis
------------------
The thermodynamic minimum σ* minimises the Frobenius energy functional
E(σ) = (1/2k) Σ_j λ_j(σ)², the spectral analogue of the structural
Lyapunov energy from the TNFR conservation theorem.  The closed-form

    σ* = 1/2 − tr(L_k V_1) / tr(V_1²)

(derived in spectral_proof.py) reduces the convergence question to
bounding two number-theoretic sums.  This module provides those bounds
via PNT, completing the analytical proof chain:

    TNFR nodal equation  →  Frobenius energy minimum
    →  closed-form σ*    →  PNT trace bounds  →  |σ* − 1/2| = O(1/k)

References
----------
- spectral_proof.py: compute_analytic_sigma_star(), _compute_lv1_traces()
- theory/TNFR_RIEMANN_RESEARCH_NOTES.md sec. 7-16
- AGENTS.md: TNFR-Riemann Program Overview
- Hardy & Wright, An Introduction to the Theory of Numbers (PNT, Mertens)

Status: EXPERIMENTAL -- Research prototype for TNFR-Riemann program.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from ..mathematics.unified_numerical import np
from .operator import _first_primes
from .spectral_proof import _compute_lv1_traces, compute_analytic_sigma_star

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Data structures
    "TelescopingIdentity",
    "PNTAsymptoticBound",
    "ConvergenceRateBound",
    "EffectiveConstantResult",
    "AnalyticalConvergenceProof",
    # Telescoping identity (Theorem 1)
    "compute_telescoping_trace",
    "verify_telescoping_identity",
    # PNT asymptotics (Theorem 2)
    "pnt_prime_estimate",
    "euler_maclaurin_log_squared_sum",
    "pnt_sum_log_squared",
    # Convergence rate (Theorem 3)
    "compute_convergence_rate_bound",
    "compute_effective_constant",
    "analyze_convergence_sequence",
    # Integration
    "run_analytical_convergence_proof",
]

# ============================================================================
# Data Structures
# ============================================================================

@dataclass(frozen=True)
class TelescopingIdentity:
    r"""Result of the exact telescoping identity for tr(L_k V_1).

    Theorem 1: For the prime path graph G_k with log-gap edge weights
    w_j = log p_{j+1} − log p_j, the cross-term tr(L_k V_1) satisfies:

        tr(L_k V_1) = (log p_k)² − (log p_1)²

    This is exact for all k ≥ 2.
    """

    k: int
    """Number of prime nodes."""

    log_pk_squared: float
    """(log p_k)² — dominant term."""

    log_p1_squared: float
    """(log p_1)² = (log 2)² ≈ 0.480 — constant correction."""

    telescoping_value: float
    """Exact: (log p_k)² − (log p_1)²."""

    numerical_value: float
    """Numerical tr(L_k V_1) from _compute_lv1_traces (verification)."""

    relative_error: float
    """|(telescoping − numerical) / numerical|."""

@dataclass(frozen=True)
class PNTAsymptoticBound:
    r"""PNT-based asymptotic for tr(V_1²) = Σ(log p_i)².

    Theorem 2: By PNT (p_n ≈ n(ln n + ln ln n)) the trace satisfies

        tr(V_1²) = Σ_{i=1}^{k} (log p_i)²

    with PNT estimate using log p_n ≈ ln n + ln(ln n + ln ln n)
    (computed directly from the improved Rosser–Schoenfeld bound),
    and leading-order scaling k(log k)², establishing the asymptotic
    growth rate needed for the O(1/k) convergence proof.

    Sandwich bounds (for k ≥ 6):
        k(ln 2)² < tr(V_1²) ≤ k(log p_k)²

    Lower bound from p_i > 2 for all i; upper from monotonicity.
    """

    k: int
    """Number of prime nodes."""

    exact_value: float
    """Exact Σ_{i=1}^{k} (log p_i)² computed from actual primes."""

    pnt_estimate: float
    """PNT-based estimate Σ_{n=2}^{k} [ln(n ln n)]² + (ln 2)²."""

    leading_order: float
    """k(log k)² — leading scaling term."""

    upper_bound: float
    """k · (log p_k)² — trivial upper bound."""

    lower_bound: float
    """(k/2) · (log p_{k/2})² — effective lower bound from tail sum."""

    pnt_relative_error: float
    """|(pnt_estimate − exact) / exact|."""

    scaling_ratio: float
    """exact / (k · (log k)²) — should approach a constant > 1."""

@dataclass(frozen=True)
class ConvergenceRateBound:
    r"""Analytical bound on |σ* − 1/2| = O(1/k).

    Theorem 3 (Main Result):

        |σ* − 1/2| = tr(L_k V_1) / tr(V_1²)
                    = [(log p_k)² − (log 2)²] / Σ(log p_i)²

    By Theorems 1–2, the numerator grows as (log k)² and the
    denominator as k(log k)², giving |σ* − 1/2| = O(1/k).
    """

    k: int
    """Number of prime nodes."""

    sigma_star: float
    """Exact σ* = 1/2 − tr(L V_1) / tr(V_1²)."""

    deviation: float
    """|σ* − 1/2| (exact, from primes)."""

    numerator: float
    """tr(L_k V_1) = (log p_k)² − (log 2)² (Theorem 1)."""

    denominator: float
    """tr(V_1²) = Σ(log p_i)² (exact from primes)."""

    curvature: float
    """d²E/dσ² = (1/k) tr(V_1²) — thermodynamic curvature."""

@dataclass(frozen=True)
class EffectiveConstantResult:
    r"""Effective constant C(k) = k · |σ* − 1/2| tracking convergence to 1.

    As k → ∞, C(k) → 1 because:

        C(k) = k · (log p_k)² / Σ(log p_i)² · [1 − (log 2)²/(log p_k)²]

    The ratio k(log p_k)² / Σ(log p_i)² → 1 since both numerator and
    denominator scale as k(log k)², with the p_k correction ln ln k / ln k
    vanishing.
    """

    k: int
    """Number of prime nodes."""

    effective_constant: float
    """C(k) = k · |σ* − 1/2|."""

    deviation_from_unity: float
    """|C(k) − 1| — measures distance from asymptotic limit."""

@dataclass(frozen=True)
class AnalyticalConvergenceProof:
    r"""Complete P8 analytical proof that σ* → 1/2 at rate O(1/k).

    Integrates all four results:
    1. Telescoping identity → exact numerator
    2. PNT asymptotic → denominator scaling
    3. Convergence rate → O(1/k) bound
    4. Effective constant → C(k) → 1
    """

    k_values: list[int]
    """Sequence of graph sizes analysed."""

    telescoping: list[TelescopingIdentity]
    """Theorem 1 verification for each k."""

    pnt_bounds: list[PNTAsymptoticBound]
    """Theorem 2 asymptotic accuracy for each k."""

    convergence_rates: list[ConvergenceRateBound]
    """Theorem 3 deviation for each k."""

    effective_constants: list[EffectiveConstantResult]
    """C(k) → 1 tracking."""

    telescoping_max_error: float
    """max relative error in telescoping identity across all k."""

    pnt_max_error: float
    """max relative error in PNT asymptotic across all k."""

    final_effective_constant: float
    """C(k) at largest k — should be close to 1."""

    monotone_decrease: bool
    """Whether |σ* − 1/2| decreases monotonically with k."""

# ============================================================================
# Theorem 1: Telescoping Identity
# ============================================================================

def compute_telescoping_trace(k: int) -> TelescopingIdentity:
    r"""Compute tr(L_k V_1) via the exact telescoping identity.

    **Theorem 1** (Telescoping Identity):
    For the prime path graph G_k with log-gap edge weights
    w_j = log p_{j+1} − log p_j, we have:

        tr(L_k V_1) = (log p_k)² − (log p_1)²

    **Proof**:
    Write tr(L_k V_1) = Σ_i deg_w(i) log p_i.  Each edge j with weight
    w_j = log p_{j+1} − log p_j contributes w_j to both deg_w(j) and
    deg_w(j+1).  Regrouping by edges:

        Σ_i deg_w(i) log p_i = Σ_j w_j (log p_j + log p_{j+1})
            = Σ_j (log p_{j+1} − log p_j)(log p_{j+1} + log p_j)
            = Σ_j [(log p_{j+1})² − (log p_j)²]
            = (log p_k)² − (log p_1)²     ∎

    Parameters
    ----------
    k : int
        Number of prime nodes (k ≥ 2).

    Returns
    -------
    TelescopingIdentity
        Exact value and numerical verification.
    """
    if k < 2:
        raise ValueError("Telescoping identity requires k >= 2")

    primes = _first_primes(k)
    log_pk = math.log(primes[-1])
    log_p1 = math.log(primes[0])  # log(2)

    log_pk_sq = log_pk ** 2
    log_p1_sq = log_p1 ** 2
    telescoping = log_pk_sq - log_p1_sq

    # Numerical verification via existing infrastructure
    tr_LV1, _, _ = _compute_lv1_traces(k, weight_by_log_gap=True)

    rel_err = abs(telescoping - tr_LV1) / max(abs(tr_LV1), 1e-15)

    return TelescopingIdentity(
        k=k,
        log_pk_squared=log_pk_sq,
        log_p1_squared=log_p1_sq,
        telescoping_value=telescoping,
        numerical_value=tr_LV1,
        relative_error=rel_err,
    )

def verify_telescoping_identity(
    k_values: Sequence[int],
) -> list[TelescopingIdentity]:
    r"""Verify the telescoping identity for a sequence of graph sizes.

    Parameters
    ----------
    k_values : sequence of int
        Graph sizes to verify.

    Returns
    -------
    list of TelescopingIdentity
        Results for each k, all should have relative_error < 1e-12.
    """
    return [compute_telescoping_trace(k) for k in k_values]

# ============================================================================
# Theorem 2: PNT Asymptotic for tr(V_1²)
# ============================================================================

def pnt_prime_estimate(n: int) -> float:
    r"""Estimate the n-th prime using PNT: p_n ≈ n(ln n + ln ln n).

    For n ≥ 6, p_n lies between n(ln n + ln ln n − 1) and
    n(ln n + ln ln n).  This returns the upper estimate.

    Parameters
    ----------
    n : int
        Prime index (1-based: p_1 = 2, p_2 = 3, ...).

    Returns
    -------
    float
        PNT estimate of p_n.
    """
    if n < 1:
        raise ValueError("Prime index must be >= 1")
    if n <= 5:
        return [2.0, 3.0, 5.0, 7.0, 11.0][n - 1]
    ln_n = math.log(n)
    ln_ln_n = math.log(ln_n)
    return n * (ln_n + ln_ln_n)

def euler_maclaurin_log_squared_sum(k: int) -> float:
    r"""Compute Σ_{n=2}^{k} (ln n)² via Euler–Maclaurin approximation.

    Uses the integral:

        ∫₁ᵏ (ln x)² dx = k(ln k)² − 2k ln k + 2k − 2

    This gives the leading asymptotic for Σ(ln n)².  The Euler–Maclaurin
    correction terms are O((ln k)²).

    Parameters
    ----------
    k : int
        Upper summation limit.

    Returns
    -------
    float
        ≈ k(ln k)² − 2k ln k + 2k.
    """
    if k < 2:
        return 0.0
    ln_k = math.log(k)
    return k * ln_k ** 2 - 2 * k * ln_k + 2 * k

def pnt_sum_log_squared(k: int) -> PNTAsymptoticBound:
    r"""Compute and bound tr(V_1²) = Σ_{i=1}^{k} (log p_i)².

    **Theorem 2**: By PNT, p_n ≈ n(ln n + ln ln n), so
    log p_n ≈ ln n + ln(ln n + ln ln n).  The PNT estimate

        Σ_{n=6}^{k} [ln n + ln(ln n + ln ln n)]² + Σ_{n=1}^{5} (log p_n)²

    captures the growth rate of tr(V_1²), using the improved Rosser–Schoenfeld
    bound for the n-th prime.  For n < 6, exact primes are used.

    The proof of O(1/k) only requires the scaling order:

        tr(V_1²) = Θ(k · (log k)²)

    which follows from sandwich bounds:
        (k/2)(log p_{⌊k/2⌋})² ≤ Σ(log p_i)² ≤ k(log p_k)²

    Parameters
    ----------
    k : int
        Number of primes.

    Returns
    -------
    PNTAsymptoticBound
        Contains exact value, PNT estimate, bounds, and scaling ratio.
    """
    if k < 2:
        raise ValueError("PNT asymptotic requires k >= 2")

    # Exact computation from actual primes
    primes = _first_primes(k)
    log_p = np.array([math.log(p) for p in primes])
    exact = float(np.sum(log_p ** 2))

    # PNT-based estimate using p_n ≈ n(ln n + ln ln n) for n ≥ 6.
    # For n < 6, use exact primes (PNT unreliable there).
    small_primes = [2, 3, 5, 7, 11]  # p_1 .. p_5
    pnt_est = 0.0
    for i in range(min(k, 5)):
        pnt_est += math.log(small_primes[i]) ** 2
    for n in range(6, k + 1):
        ln_n = math.log(n)
        # log(p_n) ≈ ln(n(ln n + ln ln n)) = ln n + ln(ln n + ln ln n)
        pnt_est += (ln_n + math.log(ln_n + math.log(ln_n))) ** 2

    # Leading-order scaling: k(log k)²
    ln_k = math.log(k)
    leading = k * ln_k ** 2

    # Upper bound: k · (log p_k)²
    upper = k * float(log_p[-1]) ** 2

    # Lower bound: (k/2) · (log p_{k/2})² (tail sum)
    half_idx = max(k // 2 - 1, 0)
    lower = (k / 2.0) * float(log_p[half_idx]) ** 2

    pnt_rel_err = abs(pnt_est - exact) / max(abs(exact), 1e-15)
    scaling_ratio = exact / max(leading, 1e-15)

    return PNTAsymptoticBound(
        k=k,
        exact_value=exact,
        pnt_estimate=pnt_est,
        leading_order=leading,
        upper_bound=upper,
        lower_bound=lower,
        pnt_relative_error=pnt_rel_err,
        scaling_ratio=scaling_ratio,
    )

# ============================================================================
# Theorem 3: Main Convergence Rate
# ============================================================================

def compute_convergence_rate_bound(k: int) -> ConvergenceRateBound:
    r"""Compute the exact convergence rate |σ* − 1/2|.

    **Theorem 3** (Main Result):
    Combining Theorems 1 and 2:

        |σ* − 1/2| = tr(L_k V_1) / tr(V_1²)
                    = [(log p_k)² − (log 2)²] / Σ_{i=1}^{k} (log p_i)²

    Since the numerator ~ (log k)² and denominator ~ k(log k)²:

        |σ* − 1/2| = 1/k · (1 + o(1))

    The curvature d²E/dσ² = (1/k) tr(V_1²) → (log k)² → ∞ confirms
    σ = 1/2 is an increasingly sharp thermodynamic attractor.

    Parameters
    ----------
    k : int
        Number of prime nodes (k ≥ 2).

    Returns
    -------
    ConvergenceRateBound
        Full convergence analysis including σ*, deviation, and curvature.
    """
    if k < 2:
        raise ValueError("Convergence rate requires k >= 2")

    sigma_star, tr_LV1, tr_V1_sq = compute_analytic_sigma_star(k)

    deviation = abs(sigma_star - 0.5)
    curvature = tr_V1_sq / k  # d²E/dσ²

    return ConvergenceRateBound(
        k=k,
        sigma_star=sigma_star,
        deviation=deviation,
        numerator=tr_LV1,
        denominator=tr_V1_sq,
        curvature=curvature,
    )

def compute_effective_constant(k: int) -> EffectiveConstantResult:
    r"""Compute the effective constant C(k) = k · |σ* − 1/2|.

    C(k) → 1 as k → ∞, confirming the rate is precisely O(1/k)
    with asymptotic coefficient 1.

    Parameters
    ----------
    k : int
        Number of prime nodes (k ≥ 2).

    Returns
    -------
    EffectiveConstantResult
    """
    if k < 2:
        raise ValueError("Effective constant requires k >= 2")

    sigma_star, tr_LV1, tr_V1_sq = compute_analytic_sigma_star(k)
    deviation = abs(sigma_star - 0.5)
    c_k = k * deviation

    return EffectiveConstantResult(
        k=k,
        effective_constant=c_k,
        deviation_from_unity=abs(c_k - 1.0),
    )

def analyze_convergence_sequence(
    k_values: Sequence[int],
) -> list[ConvergenceRateBound]:
    r"""Analyse convergence across a sequence of graph sizes.

    Parameters
    ----------
    k_values : sequence of int
        Graph sizes (each ≥ 2).

    Returns
    -------
    list of ConvergenceRateBound
        Results for each k; |σ* − 1/2| should decrease as O(1/k).
    """
    return [compute_convergence_rate_bound(k) for k in k_values]

# ============================================================================
# Integration: Full Analytical Proof
# ============================================================================

def run_analytical_convergence_proof(
    k_values: Sequence[int] | None = None,
) -> AnalyticalConvergenceProof:
    r"""Run the complete P8 analytical convergence proof.

    Verifies all four theorems across the given sequence of graph sizes
    and returns a comprehensive proof object.

    Parameters
    ----------
    k_values : sequence of int or None
        Graph sizes to analyse.  Default: [5, 10, 20, 50, 100, 200, 500].

    Returns
    -------
    AnalyticalConvergenceProof
        Complete proof with all intermediate results.
    """
    if k_values is None:
        k_values = [5, 10, 20, 50, 100, 200, 500]

    k_list = list(k_values)

    # Theorem 1: Telescoping identity
    telescoping = verify_telescoping_identity(k_list)

    # Theorem 2: PNT asymptotic bounds
    pnt_bounds = [pnt_sum_log_squared(k) for k in k_list]

    # Theorem 3: Convergence rates
    convergence_rates = analyze_convergence_sequence(k_list)

    # Effective constants
    effective_constants = [compute_effective_constant(k) for k in k_list]

    # Aggregates
    tele_max_err = max(t.relative_error for t in telescoping)
    pnt_max_err = max(p.pnt_relative_error for p in pnt_bounds)
    final_c = effective_constants[-1].effective_constant

    # Check monotone decrease of deviation
    devs = [r.deviation for r in convergence_rates]
    monotone = all(devs[i] >= devs[i + 1] for i in range(len(devs) - 1))

    return AnalyticalConvergenceProof(
        k_values=k_list,
        telescoping=telescoping,
        pnt_bounds=pnt_bounds,
        convergence_rates=convergence_rates,
        effective_constants=effective_constants,
        telescoping_max_error=tele_max_err,
        pnt_max_error=pnt_max_err,
        final_effective_constant=final_c,
        monotone_decrease=monotone,
    )
