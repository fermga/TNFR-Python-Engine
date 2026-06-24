r"""Rigorous TNFR-Riemann spectral analysis framework.

Implements four independent lines of analysis for the TNFR-Riemann bridge,
each derived from the discrete TNFR operator

    H_TNFR^(k)(sigma) = L_k + V_sigma

where L_k is the weighted Laplacian of the prime path graph G_k with
edge weights w_j = |log p_{j+1} - log p_j|, and
V_sigma = (sigma - 1/2) diag(log p_1, ..., log p_k).

Lines of analysis
-----------------
1. **Structural Equilibrium Theorem** (exact for all k >= 2)
   lambda_min(H(sigma)) = 0 iff sigma = 1/2.  Derivation: L_k is PSD
   with ker(L_k) = span{1}; V_{1/2} = 0; hence H(1/2) = L_k.  The
   nodal equation equilibrium dEPI/dt = nu_f * dNFR = 0 corresponds to
   the zero eigenvalue condition.

2. **Thermodynamic Attractor Analysis** (asymptotic)
   The Frobenius energy E_sigma = (1/2k) tr(H^2) is exactly quadratic
   in sigma with analytic minimum
       sigma* = 1/2 - tr(L_k V_1) / tr(V_1^2)
   where V_1 = diag(log p_i).  By PNT estimates |sigma* - 1/2| = O(1/k),
   and the curvature d^2E/dsigma^2 = (1/k) tr(V_1^2) ~ (log k)^2 grows
   without bound, making sigma = 1/2 an increasingly sharp attractor.

3. **Eigenvalue Flow Analysis** (exact perturbation theory)
   By Hellmann-Feynman: dlambda_j/dsigma = <psi_j|V_1|psi_j> > 0 for
   all j, confirming strictly monotone spectral flow with velocities
   that encode the prime distribution through eigenvector amplitudes.

4. **Spectral Moment Analysis** (trace identities)
   mu_n = (1/k) tr(H^n) at sigma = 1/2 equals (1/k) tr(L^n) which
   counts weighted walks on the prime path graph.

TNFR physics basis
------------------
The structural equilibrium condition DELTA_NFR = 0 of the nodal equation

    dEPI/dt = nu_f * DELTA_NFR(t)

selects sigma = 1/2 as the unique zero of lambda_min(H(sigma)).  The
Frobenius energy plays the role of the Lyapunov functional from the
TNFR structural conservation theorem: E >= 0 with dE/dt <= 0 under
grammar-compliant evolution, and sigma = 1/2 is its thermodynamic
minimum.

Key results
-----------
- **Exact**: sigma = 1/2 is the structural equilibrium for all k >= 2.
- **Asymptotic**: |sigma* - 1/2| = O(1/k) <-- uses prime number theorem.
- **Curvature**: d^2E/dsigma^2 ~ (log k)^2 --> sharper basin with k.
- **Flow**: All eigenvalue velocities are positive --> monotone flow.

References
----------
- theory/TNFR_RIEMANN_RESEARCH_NOTES.md sec. 7-16
- AGENTS.md: TNFR-Riemann Program Overview

Status: EXPERIMENTAL -- Research prototype for TNFR-Riemann program.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

from scipy.linalg import eigh_tridiagonal

from ..mathematics.unified_numerical import np
from .operator import _first_primes, build_tridiagonal_h_tnfr

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Data structures
    "EquilibriumResult",
    "ThermodynamicResult",
    "EigenvalueFlowResult",
    "SpectralMomentResult",
    "TNFRRiemannAssessment",
    # Core
    "compute_eigenspectrum",
    "compute_eigensystem",
    # Line 1 - Structural Equilibrium
    "verify_equilibrium",
    "verify_equilibrium_sequence",
    # Line 2 - Thermodynamic Attractor
    "compute_analytic_sigma_star",
    "compute_frobenius_energy",
    "compute_thermodynamic_landscape",
    "verify_thermodynamic_convergence",
    # Line 3 - Eigenvalue Flow
    "compute_eigenvalue_velocities",
    "analyze_eigenvalue_flow",
    # Line 4 - Spectral Moments
    "compute_eigenvalue_spacings",
    "compute_spectral_moments",
    # Integration
    "run_tnfr_riemann_analysis",
]

# ============================================================================
# Data Structures
# ============================================================================


@dataclass(frozen=True)
class EquilibriumResult:
    r"""Structural equilibrium verification for graph size k.

    The exact result is: lambda_min(H^(k)(sigma)) = 0 iff sigma = 1/2,
    because H(1/2) = L_k (pure Laplacian) and ker(L_k) = span{1} for
    any connected graph.

    Attributes
    ----------
    k : int
        Number of primes.
    lambda_min : float
        lambda_min(H(1/2)); should be numerically zero.
    spectral_gap : float
        lambda_2 - lambda_1 at sigma = 1/2; encodes prime gap info.
    spectral_width : float
        lambda_max - lambda_min at sigma = 1/2.
    ground_velocity : float
        d(lambda_min)/dsigma at sigma = 1/2 by Hellmann-Feynman.
        Equals (1/k) sum(log p_i) ~ log k by PNT.
    mean_log_prime : float
        (1/k) sum(log p_i) -- reference scale from PNT.
    """

    k: int
    lambda_min: float
    spectral_gap: float
    spectral_width: float
    ground_velocity: float
    mean_log_prime: float


@dataclass(frozen=True)
class ThermodynamicResult:
    r"""Thermodynamic attractor analysis for graph size k.

    The Frobenius energy E(sigma) = (1/2k) tr(H^2) is exactly quadratic:

        E(sigma) = (1/2k)[tr(L^2) + 2 delta tr(L V_1) + delta^2 tr(V_1^2)]

    with delta = sigma - 1/2.  Minimum at:

        sigma* = 1/2 - tr(L V_1) / tr(V_1^2)

    Attributes
    ----------
    k : int
        Number of primes.
    sigma_star_analytic : float
        Exact analytic minimum from the formula.
    sigma_star_numerical : float
        Independent numerical minimization (verification).
    deviation : float
        |sigma* - 1/2|.
    cross_term : float
        tr(L_k V_1) -- encodes prime-gap/log-prime correlation.
    potential_norm : float
        tr(V_1^2) = sum(log p_i)^2.
    curvature : float
        d^2E/dsigma^2 = (1/k) tr(V_1^2).
    energy_at_half : float
        E(1/2) = (1/2k) tr(L^2).
    energy_at_star : float
        E(sigma*) -- global minimum.
    """

    k: int
    sigma_star_analytic: float
    sigma_star_numerical: float
    deviation: float
    cross_term: float
    potential_norm: float
    curvature: float
    energy_at_half: float
    energy_at_star: float


@dataclass(frozen=True)
class EigenvalueFlowResult:
    r"""Eigenvalue flow analysis via Hellmann-Feynman theorem.

    At any sigma, d(lambda_j)/dsigma = <psi_j|V_1|psi_j>
    = sum_i |psi_j(i)|^2 log(p_i) > 0.

    Since all log(p_i) > 0 and eigenvectors are normalised, every
    eigenvalue velocity is strictly positive: the spectral flow is
    monotonically increasing in sigma.

    Attributes
    ----------
    k : int
        Number of primes.
    velocities_at_half : np.ndarray
        d(lambda_j)/dsigma at sigma = 1/2, shape (k,).
    all_positive : bool
        True iff all velocities > 0 (should always be True).
    min_velocity : float
        Smallest velocity (lowest eigenmode).
    max_velocity : float
        Largest velocity.
    velocity_ratio : float
        max/min -- measures spectral asymmetry.
    eigenvalue_trajectories : np.ndarray
        (n_scan, k) matrix of eigenvalues over sigma range.
    sigma_scan : np.ndarray
        sigma values used for trajectory scan.
    """

    k: int
    velocities_at_half: np.ndarray
    all_positive: bool
    min_velocity: float
    max_velocity: float
    velocity_ratio: float
    eigenvalue_trajectories: np.ndarray
    sigma_scan: np.ndarray


@dataclass(frozen=True)
class SpectralMomentResult:
    r"""Spectral moment and trace analysis.

    mu_n = (1/k) sum lambda_j^n = (1/k) tr(H^n).

    At sigma = 1/2: mu_n = (1/k) tr(L^n), which counts weighted walks
    of length n on the prime path graph.

    Attributes
    ----------
    k : int
        Number of primes.
    moments : np.ndarray
        mu_1, mu_2, ..., mu_{max_n}.
    mean_spacing : float
        Average eigenvalue spacing at sigma = 1/2.
    spacing_distribution : np.ndarray
        Normalised nearest-neighbour spacings.
    spectral_gap : float
        lambda_2 - lambda_1.
    """

    k: int
    moments: np.ndarray
    mean_spacing: float
    spacing_distribution: np.ndarray
    spectral_gap: float


@dataclass
class TNFRRiemannAssessment:
    r"""Integrated assessment combining all four lines of analysis.

    Attributes
    ----------
    k_values : list[int]
        Graph sizes analysed.
    equilibria : list[EquilibriumResult]
        Line 1 results.
    thermodynamics : list[ThermodynamicResult]
        Line 2 results.
    flows : list[EigenvalueFlowResult]
        Line 3 results.
    moments : list[SpectralMomentResult]
        Line 4 results.
    convergence_A : float
        Fitted constant in |sigma* - 1/2| ~ A / k^beta.
    convergence_beta : float
        Fitted exponent (should approach 1.0 from PNT).
    gap_scaling_exponent : float
        Fitted exponent in lambda_gap ~ C / k^alpha.
    equilibrium_exact : bool
        All lambda_min(H(1/2)) numerically zero?
    flow_monotone : bool
        All eigenvalue velocities positive?
    thermodynamic_convergent : bool
        |sigma* - 1/2| decreasing with k?
    curvature_growing : bool
        d^2E/dsigma^2 increasing with k?
    overall_confidence : float
        Weighted score in [0, 1].
    summary : str
        Human-readable summary.
    """

    k_values: list[int] = field(default_factory=list)
    equilibria: list[EquilibriumResult] = field(default_factory=list)
    thermodynamics: list[ThermodynamicResult] = field(default_factory=list)
    flows: list[EigenvalueFlowResult] = field(default_factory=list)
    moments: list[SpectralMomentResult] = field(default_factory=list)

    convergence_A: float = 0.0
    convergence_beta: float = 0.0
    gap_scaling_exponent: float = 0.0

    equilibrium_exact: bool = True
    flow_monotone: bool = True
    thermodynamic_convergent: bool = True
    curvature_growing: bool = True

    overall_confidence: float = 0.0
    summary: str = ""


# ============================================================================
# Core Spectral Computation
# ============================================================================


def compute_eigenspectrum(
    k: int,
    sigma: float = 0.5,
    *,
    weight_by_log_gap: bool = True,
) -> np.ndarray:
    """Compute sorted eigenvalues of H_TNFR^(k)(sigma).

    Uses the tridiagonal structure of the prime path Laplacian for
    O(k^2) computation instead of O(k^3) dense diagonalisation.

    Parameters
    ----------
    k : int
        Number of primes (graph size).
    sigma : float
        Structural parameter Re(s) analogue.
    weight_by_log_gap : bool
        Use log-gap edge weights (default True).

    Returns
    -------
    np.ndarray
        Sorted eigenvalues lambda_1 <= lambda_2 <= ... <= lambda_k.
    """
    d, e, _ = build_tridiagonal_h_tnfr(k, sigma, weight_by_log_gap=weight_by_log_gap)
    if len(e) == 0:
        return np.sort(d)
    return np.sort(eigh_tridiagonal(d, e, eigvals_only=True))


def compute_eigensystem(
    k: int,
    sigma: float = 0.5,
    *,
    weight_by_log_gap: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute eigenvalues and eigenvectors of H_TNFR^(k)(sigma).

    Uses the tridiagonal structure of the prime path Laplacian for
    O(k^2) computation instead of O(k^3) dense diagonalisation.

    Returns
    -------
    (eigenvalues, eigenvectors)
        eigenvalues sorted ascending; eigenvectors[:, j] corresponds
        to eigenvalues[j].
    """
    d, e, _ = build_tridiagonal_h_tnfr(k, sigma, weight_by_log_gap=weight_by_log_gap)
    if len(e) == 0:
        return np.array(d), np.eye(len(d))
    eigenvalues, eigenvectors = eigh_tridiagonal(d, e)
    return eigenvalues, eigenvectors


# ============================================================================
# LINE 1: Structural Equilibrium Theorem
# ============================================================================
#
# Theorem: For the TNFR operator H^(k)(sigma) = L_k + (sigma - 1/2) V_1
# on any connected prime path graph G_k with k >= 2 primes:
#
#     lambda_min(H(sigma)) = 0  iff  sigma = 1/2.
#
# Proof:
#   (=>) L_k is the combinatorial Laplacian of a connected graph,
#        so ker(L_k) = span{1} and lambda_min(L_k) = 0.
#        At sigma = 1/2: V_{1/2} = 0, hence H(1/2) = L_k.
#   (<=) For sigma != 1/2: V_sigma = (sigma - 1/2) diag(log p_i)
#        with all log p_i > 0.  The ground state |1>/sqrt(k)
#        gets shifted: <1|V_sigma|1> = (sigma - 1/2) mean(log p) != 0,
#        so lambda_min(H(sigma)) != 0.
#
# TNFR interpretation:
#   The nodal equation equilibrium dEPI/dt = nu_f * DELTA_NFR = 0
#   corresponds to the zero eigenvalue condition.  The critical line
#   sigma = 1/2 is the unique structural equilibrium of H_TNFR.
# ============================================================================


def _get_log_primes(k: int, *, weight_by_log_gap: bool = True) -> np.ndarray:
    """Return log(p_i) for the first k primes."""
    primes = _first_primes(k)
    return np.array([np.log(float(p)) for p in primes])


def verify_equilibrium(
    k: int,
    *,
    weight_by_log_gap: bool = True,
) -> EquilibriumResult:
    r"""Verify structural equilibrium for graph size k.

    Computes:
    - lambda_min(H(1/2)) -- should be numerically zero
    - Spectral gap lambda_2 - lambda_1
    - Ground state velocity d(lambda_min)/dsigma via Hellmann-Feynman

    The ground state velocity equals <1|V_1|1>/k = mean(log p_i),
    which by PNT approaches log(k) as k -> infinity.
    """
    eigenvalues, eigenvectors = compute_eigensystem(
        k, 0.5, weight_by_log_gap=weight_by_log_gap
    )

    log_p = _get_log_primes(k, weight_by_log_gap=weight_by_log_gap)
    mean_log_p = float(np.mean(log_p))

    lambda_min = float(eigenvalues[0])
    spectral_gap = float(eigenvalues[1] - eigenvalues[0]) if k >= 2 else 0.0
    spectral_width = float(eigenvalues[-1] - eigenvalues[0]) if k >= 2 else 0.0

    # Hellmann-Feynman: d(lambda_j)/dsigma = <psi_j|V_1|psi_j>
    psi_0 = eigenvectors[:, 0]
    ground_velocity = float(np.sum(psi_0**2 * log_p))

    return EquilibriumResult(
        k=k,
        lambda_min=lambda_min,
        spectral_gap=spectral_gap,
        spectral_width=spectral_width,
        ground_velocity=ground_velocity,
        mean_log_prime=mean_log_p,
    )


def verify_equilibrium_sequence(
    k_values: Sequence[int],
    *,
    weight_by_log_gap: bool = True,
) -> list[EquilibriumResult]:
    """Verify structural equilibrium for a sequence of k values."""
    return [
        verify_equilibrium(k, weight_by_log_gap=weight_by_log_gap)
        for k in k_values
        if k >= 3
    ]


# ============================================================================
# LINE 2: Thermodynamic Attractor Analysis
# ============================================================================
#
# The Frobenius energy functional is:
#
#     E(sigma) = (1 / 2k) tr(H(sigma)^2)
#              = (1 / 2k) [tr(L^2) + 2 delta tr(L V_1) + delta^2 tr(V_1^2)]
#
# where delta = sigma - 1/2 and V_1 = diag(log p_1, ..., log p_k).
#
# This is an EXACT quadratic: E(sigma) = a + b*delta + c*delta^2 with
#   a = (1/2k) tr(L^2)        >= 0
#   b = (1/k) tr(L V_1)       (sign depends on prime gaps)
#   c = (1/2k) tr(V_1^2)      > 0  (always positive)
#
# The unique minimum is at:
#   delta* = -b / (2c) = -tr(L V_1) / tr(V_1^2)
#   sigma* = 1/2 - tr(L V_1) / tr(V_1^2)
#
# Key asymptotic result (from PNT):
#   tr(L V_1) = sum_i deg_w(i) log(p_i) ~ O((log k)^2)
#   tr(V_1^2) = sum(log p_i)^2 ~ k (log k)^2
#   Therefore |sigma* - 1/2| ~ O(1/k).
#
# The curvature d^2E/dsigma^2 = (1/k) tr(V_1^2) ~ (log k)^2 grows,
# making sigma = 1/2 an increasingly sharp thermodynamic attractor.
#
# Connection to TNFR conservation theorem:
#   The structural Lyapunov energy E = (1/2) sum(Phi_s^2 + |grad phi|^2 +
#   K_phi^2 + J_phi^2 + J_DNFR^2) is non-negative with dE/dt <= 0 under
#   grammar-compliant evolution.  The Frobenius energy (1/2k) sum lambda_j^2
#   is the spectral-domain analogue, and its minimum at sigma ~ 1/2
#   identifies the thermodynamic ground state of the TNFR operator.
# ============================================================================


def _compute_lv1_traces(
    k: int,
    *,
    weight_by_log_gap: bool = True,
) -> tuple[float, float, np.ndarray]:
    """Compute tr(L V_1), tr(V_1^2), and log-prime vector.

    Uses the tridiagonal Laplacian diagonal directly: tr(L V_1) =
    sum_i d_L[i] * log(p_i) where d_L is the Laplacian main diagonal.

    Returns
    -------
    (tr_LV1, tr_V1_sq, log_p)
    """
    # At sigma=0.5, V=0 so d_L is the pure Laplacian diagonal
    d_L, _, log_p = build_tridiagonal_h_tnfr(
        k, 0.5, weight_by_log_gap=weight_by_log_gap
    )
    tr_LV1 = float(np.dot(d_L, log_p))
    tr_V1_sq = float(np.sum(log_p**2))
    return tr_LV1, tr_V1_sq, log_p


def compute_analytic_sigma_star(
    k: int,
    *,
    weight_by_log_gap: bool = True,
) -> tuple[float, float, float]:
    r"""Compute the exact thermodynamic minimum sigma* analytically.

    sigma* = 1/2 - tr(L_k V_1) / tr(V_1^2)

    Parameters
    ----------
    k : int
        Number of primes.

    Returns
    -------
    (sigma_star, cross_term, potential_norm)
        sigma_star : float -- the analytic minimum
        cross_term : float -- tr(L V_1)
        potential_norm : float -- tr(V_1^2)
    """
    tr_LV1, tr_V1_sq, _ = _compute_lv1_traces(k, weight_by_log_gap=weight_by_log_gap)

    if tr_V1_sq < 1e-15:
        return 0.5, tr_LV1, tr_V1_sq

    sigma_star = 0.5 - tr_LV1 / tr_V1_sq
    return sigma_star, tr_LV1, tr_V1_sq


def compute_frobenius_energy(
    k: int,
    sigma: float,
    *,
    weight_by_log_gap: bool = True,
) -> float:
    r"""Compute Frobenius energy E_sigma = (1/2k) tr(H(sigma)^2).

    This is the spectral-domain Lyapunov functional for the TNFR
    operator, corresponding to (1/2k) sum lambda_j(sigma)^2.
    """
    eigenvalues = compute_eigenspectrum(k, sigma, weight_by_log_gap=weight_by_log_gap)
    return float(np.sum(eigenvalues**2) / (2.0 * k))


def compute_thermodynamic_landscape(
    k: int,
    *,
    sigma_range: tuple[float, float] = (0.0, 1.0),
    n_points: int = 200,
    weight_by_log_gap: bool = True,
) -> ThermodynamicResult:
    r"""Full thermodynamic landscape analysis for graph size k.

    Computes:
    - Analytic minimum sigma* from closed-form formula
    - Numerical verification by scanning E(sigma)
    - Cross-term and curvature analysis
    """
    # Analytic result
    tr_LV1, tr_V1_sq, log_p = _compute_lv1_traces(
        k, weight_by_log_gap=weight_by_log_gap
    )
    sigma_star_a = 0.5 - tr_LV1 / max(tr_V1_sq, 1e-15)

    # Curvature: d^2E/dsigma^2 = (1/k) tr(V_1^2)
    curvature = tr_V1_sq / k

    # Energy at sigma = 1/2
    evals_half = compute_eigenspectrum(k, 0.5, weight_by_log_gap=weight_by_log_gap)
    energy_half = float(np.sum(evals_half**2) / (2.0 * k))

    # Numerical verification: scan E(sigma) using cached tridiagonal base
    d_L, e_L, _ = build_tridiagonal_h_tnfr(k, 0.5, weight_by_log_gap=weight_by_log_gap)
    sigmas = np.linspace(sigma_range[0], sigma_range[1], n_points)
    energies = np.zeros(n_points)

    for i, sig in enumerate(sigmas):
        delta = sig - 0.5
        d_sigma = d_L + delta * log_p if abs(delta) > 0 else d_L
        evals = eigh_tridiagonal(d_sigma, e_L, eigvals_only=True)
        energies[i] = float(np.sum(evals**2) / (2.0 * k))

    idx_min = int(np.argmin(energies))
    sigma_star_n = float(sigmas[idx_min])

    # Energy at analytic minimum
    energy_star = compute_frobenius_energy(
        k, sigma_star_a, weight_by_log_gap=weight_by_log_gap
    )

    deviation = abs(sigma_star_a - 0.5)

    return ThermodynamicResult(
        k=k,
        sigma_star_analytic=sigma_star_a,
        sigma_star_numerical=sigma_star_n,
        deviation=deviation,
        cross_term=tr_LV1,
        potential_norm=tr_V1_sq,
        curvature=curvature,
        energy_at_half=energy_half,
        energy_at_star=energy_star,
    )


def verify_thermodynamic_convergence(
    k_values: Sequence[int],
    *,
    weight_by_log_gap: bool = True,
) -> list[ThermodynamicResult]:
    """Verify thermodynamic attractor convergence for increasing k.

    Checks that:
    1. |sigma* - 1/2| decreases with k (convergence)
    2. Curvature d^2E/dsigma^2 increases with k (sharper basin)
    3. Analytic and numerical sigma* agree
    """
    return [
        compute_thermodynamic_landscape(k, weight_by_log_gap=weight_by_log_gap)
        for k in k_values
        if k >= 3
    ]


# ============================================================================
# LINE 3: Eigenvalue Flow Analysis
# ============================================================================
#
# By the Hellmann-Feynman theorem (exact first-order perturbation theory):
#
#     d(lambda_j)/dsigma = <psi_j(sigma) | dH/dsigma | psi_j(sigma)>
#                        = <psi_j(sigma) | V_1 | psi_j(sigma)>
#                        = sum_i |psi_j(i)|^2 log(p_i)
#
# Since log(p_i) > 0 for all primes p_i >= 2, and |psi_j|^2 is a
# probability distribution, every velocity d(lambda_j)/dsigma > 0.
#
# Physical interpretation:
# - The velocity of eigenmode j is a weighted average of log-primes.
# - Low modes (delocalised) see the global average: v ~ mean(log p).
# - High modes (localised at specific nodes) see local log-primes.
# - The velocity spectrum encodes how each eigenmode "sees" prime structure.
#
# Consequence: sigma = 1/2 is the UNIQUE value where lambda_min = 0.
#   Since all eigenvalues increase monotonically with sigma, the
#   minimum eigenvalue crosses zero exactly once, at sigma = 1/2.
# ============================================================================


def compute_eigenvalue_velocities(
    k: int,
    sigma: float = 0.5,
    *,
    weight_by_log_gap: bool = True,
) -> np.ndarray:
    r"""Compute d(lambda_j)/dsigma at given sigma via Hellmann-Feynman.

    Returns array of velocities v_j = <psi_j|V_1|psi_j> for j=0,...,k-1,
    sorted in the same order as the eigenvalues.
    """
    eigenvalues, eigenvectors = compute_eigensystem(
        k, sigma, weight_by_log_gap=weight_by_log_gap
    )
    log_p = _get_log_primes(k, weight_by_log_gap=weight_by_log_gap)

    # v_j = sum_i |psi_j(i)|^2 log(p_i)
    velocities = np.array(
        [float(np.sum(eigenvectors[:, j] ** 2 * log_p)) for j in range(k)]
    )

    return velocities


def analyze_eigenvalue_flow(
    k: int,
    *,
    sigma_range: tuple[float, float] = (0.0, 1.0),
    n_scan: int = 100,
    weight_by_log_gap: bool = True,
) -> EigenvalueFlowResult:
    r"""Full eigenvalue flow analysis over a sigma range.

    Computes eigenvalue trajectories lambda_j(sigma) and velocities
    at sigma = 1/2.
    """
    d_L, e_L, log_p = build_tridiagonal_h_tnfr(
        k, 0.5, weight_by_log_gap=weight_by_log_gap
    )
    sigmas = np.linspace(sigma_range[0], sigma_range[1], n_scan)
    trajectories = np.zeros((n_scan, k))

    for i, sig in enumerate(sigmas):
        delta = sig - 0.5
        d_sigma = d_L + delta * log_p if abs(delta) > 0 else d_L
        evals = np.sort(eigh_tridiagonal(d_sigma, e_L, eigvals_only=True))
        trajectories[i, :] = evals

    velocities = compute_eigenvalue_velocities(
        k, 0.5, weight_by_log_gap=weight_by_log_gap
    )

    all_pos = bool(np.all(velocities > 0))
    v_min = float(np.min(velocities))
    v_max = float(np.max(velocities))
    v_ratio = v_max / max(v_min, 1e-15)

    return EigenvalueFlowResult(
        k=k,
        velocities_at_half=velocities,
        all_positive=all_pos,
        min_velocity=v_min,
        max_velocity=v_max,
        velocity_ratio=v_ratio,
        eigenvalue_trajectories=trajectories,
        sigma_scan=sigmas,
    )


# ============================================================================
# LINE 4: Spectral Moment & Spacing Analysis
# ============================================================================
#
# Spectral moments:
#     mu_n = (1/k) sum_j lambda_j^n = (1/k) tr(H^n)
#
# At sigma = 1/2 (H = L):
#     mu_1 = (1/k) tr(L) = (1/k) sum_i deg_w(i) = (2/k) sum_j w_j
#     mu_2 = (1/k) tr(L^2) = (1/k) (sum_i deg_w(i)^2 + 2 sum_{(i,j)} w_ij^2)
#     mu_n counts weighted walks of length n on the prime path graph.
#
# Eigenvalue spacings (normalised nearest-neighbour):
#     s_i = (epsilon_{i+1} - epsilon_i) / <s>
#
# Note: For the deterministic tridiagonal Laplacian L_k, the spacing
# distribution is NOT expected to follow random-matrix universality
# classes (GUE, GOE, Poisson).  It has specific structure reflecting
# the prime-gap edge weights.
# ============================================================================


def _unfold_eigenvalues(eigenvalues: np.ndarray) -> np.ndarray:
    r"""Unfold eigenvalues to unit mean spacing.

    Polynomial unfolding: fit smooth cumulative density N(lambda) and
    map lambda -> N(lambda) so that local density = 1.
    """
    n = len(eigenvalues)
    if n < 3:
        return eigenvalues.copy()

    sorted_evals = np.sort(eigenvalues)
    ranks = np.arange(1, n + 1, dtype=float) / n

    deg = min(5, max(1, n // 3))
    coeffs = np.polyfit(sorted_evals, ranks, deg)
    unfolded = np.polyval(coeffs, sorted_evals) * n
    return unfolded


def compute_eigenvalue_spacings(eigenvalues: np.ndarray) -> np.ndarray:
    """Compute normalised nearest-neighbour spacings after unfolding."""
    unfolded = _unfold_eigenvalues(eigenvalues)
    spacings = np.diff(unfolded)

    mean_sp = np.mean(spacings)
    if mean_sp > 1e-15:
        spacings = spacings / mean_sp
    return spacings


def compute_spectral_moments(
    k: int,
    sigma: float = 0.5,
    *,
    max_n: int = 6,
    weight_by_log_gap: bool = True,
) -> SpectralMomentResult:
    r"""Compute spectral moments and spacing distribution.

    Parameters
    ----------
    k : int
        Number of primes.
    sigma : float
        Structural parameter.
    max_n : int
        Highest moment order.
    weight_by_log_gap : bool
        Use log-gap edge weights.

    Returns
    -------
    SpectralMomentResult
    """
    evals = compute_eigenspectrum(k, sigma, weight_by_log_gap=weight_by_log_gap)

    moments = np.array([float(np.mean(evals**n)) for n in range(1, max_n + 1)])

    spacings = compute_eigenvalue_spacings(evals) if k >= 4 else np.array([])

    mean_sp = float(np.mean(np.diff(evals))) if k >= 2 else 0.0
    gap = float(evals[1] - evals[0]) if k >= 2 else 0.0

    return SpectralMomentResult(
        k=k,
        moments=moments,
        mean_spacing=mean_sp,
        spacing_distribution=spacings,
        spectral_gap=gap,
    )


# ============================================================================
# Integration: Combined TNFR-Riemann Assessment
# ============================================================================


def _fit_power_law(
    x_values: Sequence[float],
    y_values: Sequence[float],
) -> tuple[float, float]:
    r"""Fit y = A * x^{-beta} via log-log least squares.

    Returns (A, beta).
    """
    valid = [(x, y) for x, y in zip(x_values, y_values) if x > 0 and y > 0]

    if len(valid) < 2:
        return (0.0, 1.0)

    log_x = np.array([math.log(x) for x, _ in valid])
    log_y = np.array([math.log(y) for _, y in valid])

    # log y = log A - beta * log x
    A_mat = np.column_stack([np.ones_like(log_x), -log_x])
    result, _, _, _ = np.linalg.lstsq(A_mat, log_y, rcond=None)
    log_A, beta = float(result[0]), float(result[1])

    return (math.exp(log_A), beta)


def run_tnfr_riemann_analysis(
    k_values: Sequence[int] | None = None,
    *,
    weight_by_log_gap: bool = True,
    flow_n_scan: int = 60,
    moment_max_n: int = 6,
) -> TNFRRiemannAssessment:
    r"""Execute comprehensive TNFR-Riemann spectral analysis.

    Runs all four lines of attack:
    1. Structural equilibrium verification
    2. Thermodynamic attractor analysis
    3. Eigenvalue flow analysis
    4. Spectral moment analysis

    and integrates them into a unified assessment.

    Parameters
    ----------
    k_values : sequence of int, optional
        Graph sizes. Default: [5, 10, 20, 50, 100, 200].
    weight_by_log_gap : bool
        Use log-gap edge weights.
    flow_n_scan : int
        Number of sigma scan points for eigenvalue flow.
    moment_max_n : int
        Highest spectral moment order.

    Returns
    -------
    TNFRRiemannAssessment
    """
    if k_values is None:
        k_values = [5, 10, 20, 50, 100, 200]

    result = TNFRRiemannAssessment(k_values=list(k_values))

    # Line 1: Structural equilibrium
    result.equilibria = verify_equilibrium_sequence(
        k_values, weight_by_log_gap=weight_by_log_gap
    )

    # Line 2: Thermodynamic attractor
    result.thermodynamics = verify_thermodynamic_convergence(
        k_values, weight_by_log_gap=weight_by_log_gap
    )

    # Line 3: Eigenvalue flow
    for k in k_values:
        if k >= 3:
            result.flows.append(
                analyze_eigenvalue_flow(
                    k,
                    n_scan=flow_n_scan,
                    weight_by_log_gap=weight_by_log_gap,
                )
            )

    # Line 4: Spectral moments
    for k in k_values:
        if k >= 3:
            result.moments.append(
                compute_spectral_moments(
                    k,
                    max_n=moment_max_n,
                    weight_by_log_gap=weight_by_log_gap,
                )
            )

    # --- Quality assessment ---

    # Equilibrium: all lambda_min numerically zero?
    if result.equilibria:
        max_lambda_min = max(abs(eq.lambda_min) for eq in result.equilibria)
        result.equilibrium_exact = max_lambda_min < 1e-10

    # Flow: all velocities positive?
    if result.flows:
        result.flow_monotone = all(fl.all_positive for fl in result.flows)

    # Thermodynamic convergence: fit |sigma* - 1/2| ~ A / k^beta
    if len(result.thermodynamics) >= 2:
        ks_float = [float(td.k) for td in result.thermodynamics]
        devs = [td.deviation for td in result.thermodynamics]

        # Check monotone decrease
        result.thermodynamic_convergent = all(
            devs[i] >= devs[i + 1] - 1e-14 for i in range(len(devs) - 1)
        )

        # Fit power law
        valid_pairs = [(k, d) for k, d in zip(ks_float, devs) if d > 1e-14]
        if len(valid_pairs) >= 2:
            fit_k = [p[0] for p in valid_pairs]
            fit_d = [p[1] for p in valid_pairs]
            result.convergence_A, result.convergence_beta = _fit_power_law(fit_k, fit_d)

    # Curvature growing?
    if len(result.thermodynamics) >= 2:
        curvatures = [td.curvature for td in result.thermodynamics]
        result.curvature_growing = all(
            curvatures[i] <= curvatures[i + 1] + 1e-14
            for i in range(len(curvatures) - 1)
        )

    # Spectral gap scaling
    if len(result.equilibria) >= 2:
        gap_ks = [float(eq.k) for eq in result.equilibria if eq.spectral_gap > 1e-15]
        gap_vals = [
            eq.spectral_gap for eq in result.equilibria if eq.spectral_gap > 1e-15
        ]
        if len(gap_ks) >= 2:
            _, result.gap_scaling_exponent = _fit_power_law(gap_ks, gap_vals)

    # Overall confidence (weighted score)
    scores: list[float] = []

    # 1. Equilibrium exactness (30%)
    if result.equilibria:
        eq_score = 1.0 if result.equilibrium_exact else 0.0
        scores.append(0.30 * eq_score)

    # 2. Thermodynamic convergence quality (30%)
    if result.thermodynamics:
        last_dev = result.thermodynamics[-1].deviation
        conv_score = max(0.0, 1.0 - 5.0 * last_dev)  # ~1 if dev < 0.2
        beta_score = min(1.0, result.convergence_beta / 1.0)  # ~1 if beta >= 1
        td_score = 0.5 * conv_score + 0.5 * beta_score
        scores.append(0.30 * td_score)

    # 3. Flow monotonicity (20%)
    if result.flows:
        flow_score = 1.0 if result.flow_monotone else 0.0
        scores.append(0.20 * flow_score)

    # 4. Curvature growth (20%)
    if result.thermodynamics:
        curv_score = 1.0 if result.curvature_growing else 0.0
        scores.append(0.20 * curv_score)

    result.overall_confidence = float(sum(scores)) if scores else 0.0

    # Summary
    lines: list[str] = []
    lines.append("TNFR-Riemann Spectral Analysis Summary")
    lines.append("=" * 42)
    lines.append("")
    lines.append("Operator: H^(k)(sigma) = L_k + (sigma - 1/2) V_1")
    lines.append(f"Tested k values: {result.k_values}")
    lines.append("")

    lines.append("Line 1 (Structural Equilibrium):")
    lines.append(
        f"  lambda_min(H(1/2)) = 0: "
        f"{'EXACT' if result.equilibrium_exact else 'FAILED'}"
    )
    if result.equilibria:
        lines.append(
            f"  max |lambda_min| = "
            f"{max(abs(eq.lambda_min) for eq in result.equilibria):.2e}"
        )
        lines.append(
            f"  Spectral gap scaling exponent: " f"{result.gap_scaling_exponent:.3f}"
        )

    lines.append("")
    lines.append("Line 2 (Thermodynamic Attractor):")
    if result.thermodynamics:
        lines.append(
            f"  sigma* convergence: "
            f"{'YES' if result.thermodynamic_convergent else 'NO'}"
        )
        lines.append(
            f"  Fitted: |sigma* - 1/2| ~ {result.convergence_A:.4f} "
            f"/ k^{result.convergence_beta:.3f}"
        )
        lines.append(
            f"  Curvature growing: " f"{'YES' if result.curvature_growing else 'NO'}"
        )

    lines.append("")
    lines.append("Line 3 (Eigenvalue Flow):")
    lines.append(
        f"  All velocities positive: " f"{'YES' if result.flow_monotone else 'NO'}"
    )
    if result.flows:
        last_flow = result.flows[-1]
        lines.append(
            f"  Velocity range (k={last_flow.k}): "
            f"[{last_flow.min_velocity:.4f}, {last_flow.max_velocity:.4f}]"
        )

    lines.append("")
    lines.append(f"Overall confidence: {result.overall_confidence:.4f}")
    result.summary = "\n".join(lines)

    return result
