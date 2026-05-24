r"""P5: Discrete spectral zeta and trace formula.

Computes the **discrete spectral zeta function** of the TNFR operator

    zeta_{H^(k)}(sigma, u) = sum_{j: lambda_j > 0} lambda_j(sigma)^{-u}

and the **heat kernel trace**

    Theta(beta) = tr(e^{-beta H(sigma)}) = sum_j e^{-beta lambda_j}

for a range of inverse temperatures beta.

Main analyses
-------------
1. **Spectral zeta**: Analytic continuation in u for fixed k, sigma.
2. **Heat kernel trace**: Partition function Z(beta) and thermodynamics.
3. **Mellin bridge**: Numerical verification that
       zeta_H(u) = (1/Gamma(u)) int_0^infty beta^{u-1} Theta(beta) dbeta.
4. **Conjecture 10.1**: Does zeta_{H^(k)}(1/2, u) -> C * zeta_R(u + delta)
   as k -> infinity, for some normalisation C and shift delta?
5. **Free energy & entropy**: F(beta) = -(1/beta) ln Z(beta),
   S(beta) = beta^2 dF/dbeta.

TNFR physics basis
-------------------
The spectral zeta connects the eigenvalue spectrum to partition
functions and thermodynamic state variables within the structural
conservation framework.  The Mellin transform bridges heat-kernel
dynamics (nodal-equation time evolution e^{-beta H}) to zeta values,
establishing a direct link between TNFR structural thermodynamics
and the Riemann zeta function.

At sigma = 1/2 the operator reduces to the graph Laplacian, so
zeta_{H^(k)}(1/2, u) = sum_{j>=2} lambda_j(L_k)^{-u}  (excluding
the zero eigenvalue from ker(L_k)).  The scaling behaviour with k
encodes the prime distribution, and Conjecture 10.1 asks whether
this spectral zeta converges to a rescaled Riemann zeta in the
large-k limit.

Status: EXPERIMENTAL -- Research prototype for TNFR-Riemann P5 program.

References
----------
- AGENTS.md: TNFR-Riemann Program Overview
- theory/TNFR_RIEMANN_RESEARCH_NOTES.md sec. 7-16
- src/tnfr/riemann/spectral_proof.py: P1 thermodynamic attractor
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import scipy.special
from scipy.linalg import eigh_tridiagonal

from ..mathematics.unified_numerical import np
from .operator import (
    build_tridiagonal_h_tnfr,
)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Data structures
    "SpectralZetaResult",
    "HeatKernelResult",
    "MellinBridgeResult",
    "ConjectureTestResult",
    "SpectralZetaAnalysis",
    # Constants
    "RIEMANN_ZETA_KNOWN_VALUES",
    # Core computation
    "compute_positive_eigenvalues",
    "compute_spectral_zeta",
    "compute_spectral_zeta_derivative",
    # Heat kernel
    "compute_heat_kernel_trace",
    "compute_partition_function",
    "compute_free_energy",
    # Mellin bridge
    "verify_mellin_bridge",
    # Conjecture 10.1
    "riemann_zeta_approx",
    "test_conjecture_10_1",
    "test_conjecture_10_1_sequence",
    # Integration
    "run_spectral_zeta_analysis",
]

# ---------------------------------------------------------------------------
# Reference values: Riemann zeta at selected points.
# Used for Conjecture 10.1 comparison.  Source: standard mathematical tables.
# ---------------------------------------------------------------------------

RIEMANN_ZETA_KNOWN_VALUES: dict[float, float] = {
    # zeta(u) for u > 1 (convergent region)
    2.0: math.pi**2 / 6,                   # 1.6449...
    3.0: 1.2020569031595942,                # Apery's constant
    4.0: math.pi**4 / 90,                   # 1.0823...
    5.0: 1.0369277551433699,
    6.0: math.pi**6 / 945,                  # 1.0173...
    8.0: math.pi**8 / 9450,                 # 1.00408...
    10.0: 1.0009945751278181,
    # Negative integers (via analytic continuation / Bernoulli)
    0.0: -0.5,
    -1.0: -1.0 / 12,
    # Near-pole region (u close to 1, finite approximations)
    1.5: 2.612375348685488,
    1.1: 10.584448464950809,
}

# ============================================================================
# Data Structures
# ============================================================================

@dataclass(frozen=True)
class SpectralZetaResult:
    r"""Result of computing the discrete spectral zeta.

    zeta_{H^(k)}(sigma, u) = sum_{j: lambda_j > 0} lambda_j(sigma)^{-u}

    Attributes
    ----------
    k : int
        Number of primes (graph size).
    sigma : float
        Structural parameter.
    u_values : np.ndarray
        Values of u where zeta was evaluated.
    zeta_values : np.ndarray
        zeta_{H^(k)}(sigma, u) for each u.
    n_positive : int
        Number of positive eigenvalues used (excludes lambda ~ 0).
    eigenvalues_positive : np.ndarray
        The positive eigenvalues lambda_j > 0.
    """

    k: int
    sigma: float
    u_values: np.ndarray
    zeta_values: np.ndarray
    n_positive: int
    eigenvalues_positive: np.ndarray

@dataclass(frozen=True)
class HeatKernelResult:
    r"""Result of heat kernel trace computation.

    Theta(beta) = tr(e^{-beta H}) = sum_j e^{-beta lambda_j}

    Attributes
    ----------
    k : int
        Number of primes.
    sigma : float
        Structural parameter.
    beta_values : np.ndarray
        Inverse temperature values.
    theta_values : np.ndarray
        Heat kernel trace Theta(beta) for each beta.
    partition_function : np.ndarray
        Z(beta) = Theta(beta) (same as theta for these operators).
    free_energy : np.ndarray
        F(beta) = -(1/beta) ln Z(beta).
    entropy : np.ndarray
        S(beta) = beta^2 dF/dbeta (numerical differentiation).
    eigenvalues : np.ndarray
        All eigenvalues (sorted).
    """

    k: int
    sigma: float
    beta_values: np.ndarray
    theta_values: np.ndarray
    partition_function: np.ndarray
    free_energy: np.ndarray
    entropy: np.ndarray
    eigenvalues: np.ndarray

@dataclass(frozen=True)
class MellinBridgeResult:
    r"""Verification of the Mellin transform relation.

    zeta_H(u) = (1/Gamma(u)) int_0^infty beta^{u-1} Theta(beta) dbeta

    Attributes
    ----------
    k : int
        Number of primes.
    sigma : float
        Structural parameter.
    u_values : np.ndarray
        Values of u tested.
    zeta_direct : np.ndarray
        Direct spectral zeta computation.
    zeta_mellin : np.ndarray
        Mellin transform of heat kernel.
    relative_error : np.ndarray
        |zeta_direct - zeta_mellin| / |zeta_direct| for each u.
    max_relative_error : float
        Worst-case error across all u.
    bridge_valid : bool
        True if max relative error < tolerance.
    """

    k: int
    sigma: float
    u_values: np.ndarray
    zeta_direct: np.ndarray
    zeta_mellin: np.ndarray
    relative_error: np.ndarray
    max_relative_error: float
    bridge_valid: bool

@dataclass(frozen=True)
class ConjectureTestResult:
    r"""Test result for Conjecture 10.1.

    Does zeta_{H^(k)}(1/2, u) -> C * zeta_R(u + delta) as k -> infty?

    Attributes
    ----------
    k : int
        Graph size tested.
    u_values : np.ndarray
        Values of u used for fitting.
    zeta_spectral : np.ndarray
        zeta_{H^(k)}(1/2, u).
    zeta_riemann : np.ndarray
        zeta_R(u + delta_fit) for comparison.
    C_fit : float
        Best-fit normalisation constant.
    delta_fit : float
        Best-fit shift parameter.
    residual : float
        RMS residual of the fit (normalised).
    correlation : float
        Pearson correlation between spectral and fitted Riemann.
    """

    k: int
    u_values: np.ndarray
    zeta_spectral: np.ndarray
    zeta_riemann: np.ndarray
    C_fit: float
    delta_fit: float
    residual: float
    correlation: float

@dataclass(frozen=True)
class SpectralZetaAnalysis:
    r"""Integrated P5 analysis combining all components.

    Attributes
    ----------
    k : int
        Graph size.
    spectral_zeta : SpectralZetaResult
        Core spectral zeta.
    heat_kernel : HeatKernelResult
        Heat kernel and thermodynamics.
    mellin_bridge : MellinBridgeResult
        Mellin transform verification.
    conjecture : ConjectureTestResult
        Conjecture 10.1 test.
    summary : str
        Human-readable summary.
    """

    k: int
    spectral_zeta: SpectralZetaResult
    heat_kernel: HeatKernelResult
    mellin_bridge: MellinBridgeResult
    conjecture: ConjectureTestResult
    summary: str

# ============================================================================
# Core Eigenvalue Computation
# ============================================================================

def compute_positive_eigenvalues(
    k: int,
    sigma: float = 0.5,
    *,
    tol: float = 1e-10,
) -> np.ndarray:
    """Return strictly positive eigenvalues of H^(k)(sigma).

    The spectral zeta sums over lambda_j > 0, excluding the kernel.
    At sigma = 1/2 the smallest eigenvalue is zero (structural
    equilibrium); at other sigma all eigenvalues are typically positive.

    Parameters
    ----------
    k : int
        Number of primes.
    sigma : float
        Structural parameter.
    tol : float
        Eigenvalues below this threshold are treated as zero.

    Returns
    -------
    np.ndarray
        Sorted positive eigenvalues.
    """
    d, e, _ = build_tridiagonal_h_tnfr(k, sigma)
    if len(e) == 0:
        evals = np.sort(d)
    else:
        evals = np.sort(eigh_tridiagonal(d, e, eigvals_only=True))
    return evals[evals > tol]

# ============================================================================
# Spectral Zeta Function
# ============================================================================

def compute_spectral_zeta(
    k: int,
    sigma: float = 0.5,
    *,
    u_values: np.ndarray | None = None,
    tol: float = 1e-10,
) -> SpectralZetaResult:
    r"""Compute the discrete spectral zeta of H^(k)(sigma).

    zeta_{H^(k)}(sigma, u) = sum_{j: lambda_j > tol} lambda_j(sigma)^{-u}

    Parameters
    ----------
    k : int
        Number of primes.
    sigma : float
        Structural parameter.
    u_values : np.ndarray, optional
        Values of u (default: linspace(0.5, 6, 50)).
    tol : float
        Threshold for zero eigenvalues.

    Returns
    -------
    SpectralZetaResult
    """
    if u_values is None:
        u_values = np.linspace(0.5, 6.0, 50)
    u_values = np.asarray(u_values, dtype=float)

    evals_pos = compute_positive_eigenvalues(k, sigma, tol=tol)
    n_pos = len(evals_pos)

    # zeta(u) = sum lambda_j^{-u} = sum exp(-u * log(lambda_j))
    log_evals = np.log(evals_pos)  # shape (n_pos,)
    # Vectorised: (n_u,) outer with (n_pos,) -> (n_u, n_pos)
    zeta_vals = np.sum(
        np.exp(-u_values[:, None] * log_evals[None, :]),
        axis=1,
    )

    return SpectralZetaResult(
        k=k,
        sigma=sigma,
        u_values=u_values,
        zeta_values=zeta_vals,
        n_positive=n_pos,
        eigenvalues_positive=evals_pos,
    )

def compute_spectral_zeta_derivative(
    k: int,
    sigma: float = 0.5,
    *,
    u_values: np.ndarray | None = None,
    tol: float = 1e-10,
) -> np.ndarray:
    r"""Compute d/du zeta_{H^(k)}(sigma, u).

    d/du zeta(u) = -sum_{j: lambda_j > 0} log(lambda_j) * lambda_j^{-u}

    Returns
    -------
    np.ndarray
        Derivative values at each u.
    """
    if u_values is None:
        u_values = np.linspace(0.5, 6.0, 50)
    u_values = np.asarray(u_values, dtype=float)

    evals_pos = compute_positive_eigenvalues(k, sigma, tol=tol)
    log_evals = np.log(evals_pos)

    deriv = -np.sum(
        log_evals[None, :] * np.exp(-u_values[:, None] * log_evals[None, :]),
        axis=1,
    )
    return deriv

# ============================================================================
# Heat Kernel and Thermodynamics
# ============================================================================

def compute_heat_kernel_trace(
    k: int,
    sigma: float = 0.5,
    *,
    beta_values: np.ndarray | None = None,
) -> HeatKernelResult:
    r"""Compute the heat kernel trace and derived thermodynamics.

    Theta(beta) = tr(e^{-beta H}) = sum_j e^{-beta lambda_j}
    Z(beta)     = Theta(beta)
    F(beta)     = -(1/beta) ln Z(beta)
    S(beta)     = beta^2 dF/dbeta  (numerical)

    Parameters
    ----------
    k : int
        Number of primes.
    sigma : float
        Structural parameter.
    beta_values : np.ndarray, optional
        Inverse temperatures (default: logspace(-2, 2, 60)).

    Returns
    -------
    HeatKernelResult
    """
    if beta_values is None:
        beta_values = np.logspace(-2.0, 2.0, 60)
    beta_values = np.asarray(beta_values, dtype=float)

    d, e, _ = build_tridiagonal_h_tnfr(k, sigma)
    if len(e) == 0:
        evals = np.sort(d)
    else:
        evals = np.sort(eigh_tridiagonal(d, e, eigvals_only=True))

    # Theta(beta) = sum_j exp(-beta * lambda_j)
    # Shape: (n_beta, k) -> sum over j axis
    theta = np.sum(
        np.exp(-beta_values[:, None] * evals[None, :]),
        axis=1,
    )

    # Partition function is the same as theta for this operator
    Z = theta.copy()

    # Free energy: F = -(1/beta) * ln(Z),  guard Z > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        safe_Z = np.maximum(Z, 1e-300)
        F = -(1.0 / beta_values) * np.log(safe_Z)

    # Entropy via numerical differentiation: S = beta^2 * dF/dbeta
    if len(beta_values) >= 3:
        dF_dbeta = np.gradient(F, beta_values)
        S = beta_values**2 * dF_dbeta
    else:
        S = np.zeros_like(beta_values)

    return HeatKernelResult(
        k=k,
        sigma=sigma,
        beta_values=beta_values,
        theta_values=theta,
        partition_function=Z,
        free_energy=F,
        entropy=S,
        eigenvalues=evals,
    )

def compute_partition_function(
    k: int,
    sigma: float = 0.5,
    *,
    beta_values: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Convenience: return (beta_values, Z(beta)) only."""
    result = compute_heat_kernel_trace(k, sigma, beta_values=beta_values)
    return result.beta_values, result.partition_function

def compute_free_energy(
    k: int,
    sigma: float = 0.5,
    *,
    beta_values: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Convenience: return (beta_values, F(beta)) only."""
    result = compute_heat_kernel_trace(k, sigma, beta_values=beta_values)
    return result.beta_values, result.free_energy

# ============================================================================
# Mellin Transform Bridge
# ============================================================================
#
# The identity connecting heat kernel to spectral zeta is:
#
#   zeta_H(u) = (1/Gamma(u)) * integral_0^infty beta^{u-1} Theta(beta) dbeta
#
# where Theta(beta) = tr(e^{-beta H}) and zeta_H(u) = sum lambda_j^{-u}.
#
# Proof (for positive eigenvalues):
#   integral_0^infty beta^{u-1} e^{-beta lambda} dbeta = Gamma(u) lambda^{-u}
# Summing over j:
#   integral_0^infty beta^{u-1} Theta(beta) dbeta = Gamma(u) zeta_H(u)
#
# We verify this numerically by computing both sides independently.
# ============================================================================

def verify_mellin_bridge(
    k: int,
    sigma: float = 0.5,
    *,
    u_values: np.ndarray | None = None,
    n_beta: int = 500,
    beta_max: float = 200.0,
    tol: float = 0.05,
) -> MellinBridgeResult:
    r"""Verify the Mellin transform relation numerically.

    Computes both sides of:
        zeta_H(u) = (1/Gamma(u)) * int_0^infty beta^{u-1} Theta(beta) dbeta

    Parameters
    ----------
    k : int
        Number of primes.
    sigma : float
        Structural parameter.
    u_values : np.ndarray, optional
        u values for comparison (default: [1.5, 2, 3, 4, 5]).
    n_beta : int
        Number of beta quadrature points.
    beta_max : float
        Upper integration limit (should be large enough for convergence).
    tol : float
        Relative error tolerance for bridge validity.

    Returns
    -------
    MellinBridgeResult
    """
    if u_values is None:
        u_values = np.array([1.5, 2.0, 3.0, 4.0, 5.0])
    u_values = np.asarray(u_values, dtype=float)

    # Direct computation
    sz = compute_spectral_zeta(k, sigma, u_values=u_values)
    zeta_direct = sz.zeta_values

    # Mellin integral via trapezoidal quadrature on log-spaced grid
    # Use log-spacing for better coverage of both small and large beta
    beta_grid = np.logspace(-3, np.log10(beta_max), n_beta)

    # Compute Theta(beta) on the grid
    evals_pos = sz.eigenvalues_positive
    # theta_grid shape: (n_beta,)
    theta_grid = np.sum(
        np.exp(-beta_grid[:, None] * evals_pos[None, :]),
        axis=1,
    )

    # Mellin integral for each u:
    # int beta^{u-1} Theta(beta) dbeta  via trapezoidal rule
    zeta_mellin = np.empty_like(u_values)
    for i, u in enumerate(u_values):
        integrand = beta_grid ** (u - 1) * theta_grid
        integral = np.trapezoid(integrand, beta_grid)
        gamma_u = float(scipy.special.gamma(u))
        zeta_mellin[i] = integral / gamma_u

    # Relative error
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_err = np.abs(zeta_direct - zeta_mellin) / np.maximum(
            np.abs(zeta_direct), 1e-15
        )

    max_err = float(np.max(rel_err))

    return MellinBridgeResult(
        k=k,
        sigma=sigma,
        u_values=u_values,
        zeta_direct=zeta_direct,
        zeta_mellin=zeta_mellin,
        relative_error=rel_err,
        max_relative_error=max_err,
        bridge_valid=max_err < tol,
    )

# ============================================================================
# Riemann Zeta Approximation
# ============================================================================

def riemann_zeta_approx(u: float, *, n_terms: int = 10_000) -> float:
    r"""Compute zeta_R(u) via truncated Dirichlet series for u > 1.

    zeta_R(u) = sum_{n=1}^{N} n^{-u}

    For u <= 1, returns np.inf (pole region, not summable).

    Parameters
    ----------
    u : float
        Argument (must be > 1 for convergence).
    n_terms : int
        Number of terms in the truncated series.

    Returns
    -------
    float
        Approximate zeta_R(u).
    """
    if u <= 1.0:
        return float("inf")
    ns = np.arange(1, n_terms + 1, dtype=float)
    return float(np.sum(ns ** (-u)))

# ============================================================================
# Conjecture 10.1
# ============================================================================
#
# Conjecture 10.1:
#   zeta_{H^(k)}(1/2, u) -> C * zeta_R(u + delta) as k -> infty,
#   for some normalisation C(k) and shift delta(k).
#
# Method: For each k, fit C and delta by least-squares minimisation of
#   || zeta_spectral(u) - C * zeta_R(u + delta) ||^2
# over a set of u values in the convergent region u > 1.
# ============================================================================

def test_conjecture_10_1(
    k: int,
    *,
    u_values: np.ndarray | None = None,
    delta_range: tuple[float, float] = (-2.0, 2.0),
    n_delta: int = 200,
) -> ConjectureTestResult:  # noqa: N802 – name is API
    r"""Test Conjecture 10.1 for a single graph size k.

    Finds best-fit C and delta such that
        zeta_{H^(k)}(1/2, u) ~ C * zeta_R(u + delta)

    .. warning::
        **Negative numerical result (May 2026).**
        Systematic tests for k in {10, 20, 50, 100, 200, 500, 1000}
        show that the simple affine fit does NOT converge:

        - Normalised residual *rises* with k (2.43 at k=10 → 4.80 at k=1000).
        - Pearson correlation is *negative* at all tested k (≈ −0.41 … −0.21).
        - delta is pinned at the search-range boundary (delta=2.0) for all k,
          indicating no interior minimum.
        - C(k) diverges (≈ 2e7 at k=10 → 9e37 at k=1000), signalling a
          missing spectral renormalisation.

        The bridge between the TNFR spectral zeta and the classical Riemann
        zeta function is therefore **not numerically closed** in its current
        form.  Six identified missing pieces are documented in
        ``theory/TNFR_RIEMANN_RESEARCH_NOTES.md`` § 7.4.

        The sigma_c(k) → 1/2 result from ``spectral_proof.py`` is independent
        of this fit and remains valid.  This function is kept for further
        research into corrected renormalisation schemes.

    Parameters
    ----------
    k : int
        Number of primes.
    u_values : np.ndarray, optional
        u > 1 values for fitting (default: linspace(1.5, 5, 30)).
    delta_range : (float, float)
        Search range for shift delta.
    n_delta : int
        Number of delta candidates to scan.

    Returns
    -------
    ConjectureTestResult
    """
    if u_values is None:
        u_values = np.linspace(1.5, 5.0, 30)
    u_values = np.asarray(u_values, dtype=float)

    # Spectral zeta at sigma = 1/2
    sz = compute_spectral_zeta(k, 0.5, u_values=u_values)
    zeta_spec = sz.zeta_values

    # Scan over delta to find best fit
    deltas = np.linspace(delta_range[0], delta_range[1], n_delta)
    best_residual = float("inf")
    best_C = 1.0
    best_delta = 0.0
    best_zeta_r = np.ones_like(u_values)

    for delta in deltas:
        # Compute zeta_R(u + delta) for each u
        zr = np.array([riemann_zeta_approx(float(u + delta)) for u in u_values])

        # Skip if any inf (u + delta <= 1)
        if not np.all(np.isfinite(zr)):
            continue

        # Optimal C via least-squares: C = <spec, zr> / <zr, zr>
        denom = float(np.dot(zr, zr))
        if denom < 1e-30:
            continue
        C = float(np.dot(zeta_spec, zr)) / denom

        # Residual
        diff = zeta_spec - C * zr
        rms = float(np.sqrt(np.mean(diff**2)))
        # Normalise by mean spectral zeta
        mean_spec = float(np.mean(np.abs(zeta_spec)))
        if mean_spec > 1e-15:
            rms_norm = rms / mean_spec
        else:
            rms_norm = rms

        if rms_norm < best_residual:
            best_residual = rms_norm
            best_C = C
            best_delta = float(delta)
            best_zeta_r = zr.copy()

    # Pearson correlation
    corr = _pearson(zeta_spec, best_C * best_zeta_r)

    return ConjectureTestResult(
        k=k,
        u_values=u_values,
        zeta_spectral=zeta_spec,
        zeta_riemann=best_C * best_zeta_r,
        C_fit=best_C,
        delta_fit=best_delta,
        residual=best_residual,
        correlation=corr,
    )

def test_conjecture_10_1_sequence(
    k_values: Sequence[int],
    *,
    u_values: np.ndarray | None = None,
) -> list[ConjectureTestResult]:
    """Test Conjecture 10.1 for a sequence of k values."""
    return [
        test_conjecture_10_1(k, u_values=u_values)
        for k in k_values
    ]

# Prevent pytest from collecting API functions as test cases
test_conjecture_10_1.__test__ = False  # type: ignore[attr-defined]
test_conjecture_10_1_sequence.__test__ = False  # type: ignore[attr-defined]

# ============================================================================
# Integrated Analysis
# ============================================================================

def run_spectral_zeta_analysis(
    k: int,
    *,
    sigma: float = 0.5,
    u_values: np.ndarray | None = None,
    beta_values: np.ndarray | None = None,
) -> SpectralZetaAnalysis:
    r"""Run integrated P5 spectral zeta analysis.

    Combines:
    - Spectral zeta computation
    - Heat kernel and thermodynamics
    - Mellin bridge verification
    - Conjecture 10.1 test (at sigma = 1/2)

    Parameters
    ----------
    k : int
        Number of primes.
    sigma : float
        Structural parameter.
    u_values : np.ndarray, optional
        Values of u for spectral zeta.
    beta_values : np.ndarray, optional
        Inverse temperatures for heat kernel.

    Returns
    -------
    SpectralZetaAnalysis
    """
    if u_values is None:
        u_values = np.linspace(1.5, 5.0, 30)

    sz = compute_spectral_zeta(k, sigma, u_values=u_values)
    hk = compute_heat_kernel_trace(k, sigma, beta_values=beta_values)
    mb = verify_mellin_bridge(k, sigma, u_values=u_values)
    cj = test_conjecture_10_1(k, u_values=u_values)

    # Build summary
    lines = [
        f"P5 Spectral Zeta Analysis: k = {k}, sigma = {sigma}",
        f"  Positive eigenvalues: {sz.n_positive} / {k}",
        f"  Spectral zeta range: [{float(np.min(sz.zeta_values)):.4f}, "
        f"{float(np.max(sz.zeta_values)):.4f}]",
        f"  Heat kernel: Theta(beta=0.01) = {float(hk.theta_values[0]):.4f}, "
        f"Theta(beta=100) = {float(hk.theta_values[-1]):.6e}",
        f"  Mellin bridge: max rel error = {mb.max_relative_error:.4e} "
        f"({'VALID' if mb.bridge_valid else 'FAILED'})",
        f"  Conjecture 10.1: C = {cj.C_fit:.4e}, delta = {cj.delta_fit:.4f}, "
        f"r = {cj.correlation:.4f}, residual = {cj.residual:.4e}",
    ]
    summary = "\n".join(lines)

    return SpectralZetaAnalysis(
        k=k,
        spectral_zeta=sz,
        heat_kernel=hk,
        mellin_bridge=mb,
        conjecture=cj,
        summary=summary,
    )

# ============================================================================
# Private Helpers
# ============================================================================

def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation coefficient, returns 0.0 on degenerate input."""
    if len(x) < 2:
        return 0.0
    mx = np.mean(x)
    my = np.mean(y)
    dx = x - mx
    dy = y - my
    num = float(np.dot(dx, dy))
    denom = float(np.sqrt(np.dot(dx, dx) * np.dot(dy, dy)))
    if denom < 1e-30:
        return 0.0
    return num / denom
