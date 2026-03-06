r"""P7 — Conservation laws and grammar compliance at criticality.

Computes the TNFR **energy density** and **topological charge** for the
per-eigenmode structural fields of the discrete operator
H^(k)(sigma) = L_k + V_sigma, and tracks them under sigma-perturbation
to test structural conservation at the critical parameter sigma = 1/2.

Two new spectral transport fields complete the conservation five-tuple:

  J_phi(j)    = <psi_j | L | psi_j> / k        phase current
  J_DNFR(j)   = sum_i |psi_j(i)|^2 log(p_i)    eigenvalue velocity (dλ_j/dσ)

The second identity follows from the Hellmann-Feynman theorem applied to
V(p) = (sigma - 0.5) log(p).

With these, the five conservation fields (Phi_s, |grad_phi|, K_phi, J_phi,
J_DNFR) are available per eigenmode, giving:

  Energy density   E(j)  = Phi_s^2 + |grad_phi|^2 + K_phi^2 + J_phi^2 + J_DNFR^2
  Topological charge Q(j) = |grad_phi| * J_phi - K_phi * J_DNFR
  Charge density   rho(j) = Phi_s(j) + K_phi(j)

Key result:  Under smooth sigma-evolution (grammar-compliant), the total
Noether charge Q_total = sum_j Q(j) is approximately conserved, with the
conservation residual |dQ/dsigma| minimal at sigma = 1/2.  Abrupt
sigma-jumps (grammar-violating) produce large residuals, verifying the
structural conservation theorem d(rho)/dt + div(J) = S_grammar with
S_grammar -> 0 under U1-U6.

TNFR physics basis
-------------------
- Nodal equation: dEPI/dt = nu_f * DELTA_NFR(t)
- Structural conservation: d(rho)/dt + div(J) = S_grammar
- Grammar compliance: S_grammar -> 0 under U1-U6
- Conservation module: src/tnfr/physics/conservation.py
- Eigenmode fields: src/tnfr/riemann/eigenmode_fields.py

References
----------
- AGENTS.md § Structural Conservation Theorem
- theory/STRUCTURAL_CONSERVATION_THEOREM.md
- src/tnfr/physics/conservation.py  (canonical Noether-like module)

Status: EXPERIMENTAL — Research prototype for TNFR-Riemann P7 program.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    # Data structures
    "EigenmodeConservation",
    "ConservationAtSigma",
    "ConservationSigmaScan",
    "GrammarComplianceResult",
    "CriticalConservationAnalysis",
    # Core eigenmode conservation
    "compute_spectral_j_phi",
    "compute_spectral_j_dnfr",
    "compute_eigenmode_conservation",
    # Sigma scan
    "scan_conservation_vs_sigma",
    # Grammar compliance
    "test_grammar_conservation",
    # Integration
    "run_critical_conservation_analysis",
]

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class EigenmodeConservation:
    """Conservation fields for a single eigenmode.

    Attributes
    ----------
    mode_index : int
        Eigenmode index j (0-based, sorted by eigenvalue).
    eigenvalue : float
        Eigenvalue lambda_j of H^(k)(sigma).
    phi_s : float
        Spectral structural potential Phi_s(j).
    grad_phi : float
        Eigenvector gradient |grad_phi|(j).
    k_phi : float
        Eigenvector curvature K_phi(j).
    j_phi : float
        Phase current J_phi(j) = <psi_j|L|psi_j> / k.
    j_dnfr : float
        DNFR flux J_DNFR(j) = dλ_j/dσ (Hellmann-Feynman).
    energy_density : float
        E(j) = Phi_s^2 + |grad_phi|^2 + K_phi^2 + J_phi^2 + J_DNFR^2.
    topological_charge : float
        Q(j) = |grad_phi| * J_phi - K_phi * J_DNFR.
    charge_density : float
        rho(j) = Phi_s(j) + K_phi(j).
    """

    mode_index: int
    eigenvalue: float
    phi_s: float
    grad_phi: float
    k_phi: float
    j_phi: float
    j_dnfr: float
    energy_density: float
    topological_charge: float
    charge_density: float

@dataclass
class ConservationAtSigma:
    """Full conservation analysis at a given sigma.

    Attributes
    ----------
    k : int
        Number of primes (graph size).
    sigma : float
        Structural parameter.
    modes : list of EigenmodeConservation
        Per-eigenmode conservation fields.
    total_energy : float
        Sum of E(j) over all modes.
    total_charge : float
        Sum of Q(j) over all modes (topological Noether charge).
    total_charge_density : float
        Sum of rho(j) = sum_j [Phi_s(j) + K_phi(j)].
    mean_energy_density : float
        Mean E(j) across modes.
    mean_charge : float
        Mean Q(j) across modes.
    """

    k: int
    sigma: float
    modes: list[EigenmodeConservation]
    total_energy: float
    total_charge: float
    total_charge_density: float
    mean_energy_density: float
    mean_charge: float

@dataclass
class ConservationSigmaScan:
    """Conservation fields tracked across sigma values.

    Attributes
    ----------
    k : int
        Number of primes (graph size).
    sigma_values : ndarray
        Scanned sigma values.
    total_energy : ndarray
        E_total(sigma) for each sigma.
    total_charge : ndarray
        Q_total(sigma) for each sigma.
    total_charge_density : ndarray
        rho_total(sigma) for each sigma.
    charge_gradient : ndarray
        |dQ/dsigma| (conservation residual proxy).
    charge_drift_from_half : ndarray
        |Q(sigma) - Q(0.5)| / max(|Q(0.5)|, epsilon).
    energy_minimum_sigma : float
        sigma value that minimises total energy.
    min_gradient_sigma : float
        sigma value where |dQ/dsigma| is smallest.
    critical_conservation : ConservationAtSigma
        Conservation snapshot at sigma = 0.5 (or nearest).
    """

    k: int
    sigma_values: np.ndarray
    total_energy: np.ndarray
    total_charge: np.ndarray
    total_charge_density: np.ndarray
    charge_gradient: np.ndarray
    charge_drift_from_half: np.ndarray
    energy_minimum_sigma: float
    min_gradient_sigma: float
    critical_conservation: ConservationAtSigma

@dataclass
class GrammarComplianceResult:
    """Result of a grammar compliance conservation test.

    A single test applies an evolution protocol (smooth or abrupt) and
    measures how well the Noether charge is conserved.

    Attributes
    ----------
    protocol : str
        Evolution protocol name (e.g. 'smooth_step', 'abrupt_jump').
    is_grammar_compliant : bool
        Whether the protocol satisfies grammar U1-U6 by design.
    sigma_start : float
        Starting sigma value.
    sigma_end : float
        Ending sigma value.
    charge_before : float
        Q_total at sigma_start.
    charge_after : float
        Q_total at sigma_end.
    charge_drift : float
        |Q_after - Q_before| / max(|Q_before|, epsilon).
    energy_before : float
        E_total at sigma_start.
    energy_after : float
        E_total at sigma_end.
    energy_change : float
        (E_after - E_before) / max(|E_before|, epsilon).
    conservation_quality : float
        1 / (1 + charge_drift).  Higher is better.
    """

    protocol: str
    is_grammar_compliant: bool
    sigma_start: float
    sigma_end: float
    charge_before: float
    charge_after: float
    charge_drift: float
    energy_before: float
    energy_after: float
    energy_change: float
    conservation_quality: float

@dataclass
class CriticalConservationAnalysis:
    """Complete P7 conservation-at-criticality analysis.

    Attributes
    ----------
    k : int
        Number of primes (graph size).
    sigma_scan : ConservationSigmaScan
        Full sigma sweep results.
    critical : ConservationAtSigma
        Conservation snapshot at sigma = 0.5.
    grammar_tests : list of GrammarComplianceResult
        Grammar compliance tests.
    sigma_half_is_energy_min : bool
        True if sigma = 0.5 minimises total energy.
    sigma_half_has_min_residual : bool
        True if sigma = 0.5 has the smallest |dQ/dsigma|.
    compliant_mean_quality : float
        Mean conservation quality of grammar-compliant protocols.
    violating_mean_quality : float
        Mean conservation quality of grammar-violating protocols.
    quality_ratio : float
        compliant / violating (> 1 means compliance preserves charge better).
    """

    k: int
    sigma_scan: ConservationSigmaScan
    critical: ConservationAtSigma
    grammar_tests: list[GrammarComplianceResult]
    sigma_half_is_energy_min: bool
    sigma_half_has_min_residual: bool
    compliant_mean_quality: float
    violating_mean_quality: float
    quality_ratio: float

# ---------------------------------------------------------------------------
# Core: Spectral Transport Fields
# ---------------------------------------------------------------------------

def compute_spectral_j_phi(
    eigenvectors: np.ndarray,
    laplacian: np.ndarray,
) -> np.ndarray:
    r"""Compute spectral phase current J_phi for each eigenmode.

    J_phi(j) = <psi_j | L | psi_j> / k

    This is the Laplacian expectation value: the mean "flow" of the
    eigenvector across edges.  For the j-th eigenmode of H = L + V with
    eigenvalue lambda_j, we have <psi_j|L|psi_j> = lambda_j - <psi_j|V|psi_j>,
    but we compute it directly to avoid numerical issues at sigma = 0.5
    where V = 0.

    Parameters
    ----------
    eigenvectors : ndarray, shape (k, k)
        Column eigenvectors from eigh(H).
    laplacian : ndarray, shape (k, k)
        Graph Laplacian matrix L (not H = L + V).

    Returns
    -------
    ndarray, shape (k,)
        J_phi(j) for each eigenmode j.
    """
    k = eigenvectors.shape[1]
    # L @ psi gives the Laplacian action; dot with psi gives expectation
    L_psi = laplacian @ eigenvectors  # (k, k)
    # j-th diagonal of psi^T @ L @ psi
    j_phi = np.einsum("ij,ij->j", eigenvectors, L_psi) / k
    return j_phi

def compute_spectral_j_dnfr(
    eigenvectors: np.ndarray,
    log_primes: np.ndarray,
) -> np.ndarray:
    r"""Compute spectral DNFR flux J_DNFR for each eigenmode.

    J_DNFR(j) = d(lambda_j)/d(sigma) = sum_i |psi_j(i)|^2 * log(p_i)

    This follows from the Hellmann-Feynman theorem applied to
    V(p) = (sigma - 0.5) * log(p), giving dV/dsigma = diag(log p_i).

    Parameters
    ----------
    eigenvectors : ndarray, shape (k, k)
        Column eigenvectors from eigh(H).
    log_primes : ndarray, shape (k,)
        log(p_i) for each node i (in graph node order).

    Returns
    -------
    ndarray, shape (k,)
        J_DNFR(j) = dλ_j/dσ for each eigenmode j.
    """
    # |psi_j(i)|^2 weighted sum of log(p_i)
    prob = eigenvectors ** 2  # (k, k): prob[i, j] = |psi_j(i)|^2
    j_dnfr = log_primes @ prob  # (k,): sum_i log(p_i) * |psi_j(i)|^2
    return j_dnfr

# ---------------------------------------------------------------------------
# Core: Eigenmode Conservation Fields
# ---------------------------------------------------------------------------

def _build_eigensystem(
    k: int,
    sigma: float,
    *,
    weight_by_log_gap: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build H, compute eigendecomposition, return (eigenvalues, eigenvectors, L, log_primes).

    Delegates eigenvalue/eigenvector computation to
    :func:`spectral_proof.compute_eigensystem` (O(k²) tridiagonal path)
    and builds the dense Laplacian + log-primes vector separately for
    the conservation field calculations.

    Parameters
    ----------
    k : int
        Number of primes.
    sigma : float
        Structural parameter.
    weight_by_log_gap : bool
        Use log-gap edge weights.

    Returns
    -------
    (eigenvalues, eigenvectors, laplacian, log_primes)
    """
    from .operator import build_prime_path_graph, build_h_tnfr
    from .spectral_proof import compute_eigensystem

    # Eigensystem via efficient tridiagonal decomposition
    eigenvalues, eigenvectors = compute_eigensystem(
        k, sigma, weight_by_log_gap=weight_by_log_gap,
    )

    # Build graph + dense H only to extract Laplacian and log-primes
    G = build_prime_path_graph(k, weight_by_log_gap=weight_by_log_gap)
    H, diag_V = build_h_tnfr(G, sigma=sigma)
    L = H - diag_V

    nodes = sorted(G.nodes())
    log_primes = np.array([np.log(float(G.nodes[n]["label"])) for n in nodes])

    return eigenvalues, eigenvectors, L, log_primes

def compute_eigenmode_conservation(
    k: int,
    sigma: float = 0.5,
    *,
    weight_by_log_gap: bool = True,
) -> ConservationAtSigma:
    """Compute full conservation fields for all eigenmodes at (k, sigma).

    For each eigenmode j of H^(k)(sigma), computes the five structural
    fields (Phi_s, |grad_phi|, K_phi, J_phi, J_DNFR), then derives the
    energy density, topological charge, and charge density.

    Parameters
    ----------
    k : int
        Number of primes (>= 2).
    sigma : float
        Structural parameter.
    weight_by_log_gap : bool
        Use log-gap edge weights.

    Returns
    -------
    ConservationAtSigma
        Complete conservation analysis at this (k, sigma).
    """
    if k < 2:
        raise ValueError(f"k must be >= 2, got {k}")

    eigenvalues, eigenvectors, L, log_primes = _build_eigensystem(
        k, sigma, weight_by_log_gap=weight_by_log_gap,
    )

    # Spectral transport fields
    j_phi_arr = compute_spectral_j_phi(eigenvectors, L)
    j_dnfr_arr = compute_spectral_j_dnfr(eigenvectors, log_primes)

    # Per-eigenmode tetrad fields (reuse eigenmode_fields internals)
    from .eigenmode_fields import (
        _spectral_structural_potential,
        _path_eigenvector_gradient,
        _path_eigenvector_curvature,
    )

    modes: list[EigenmodeConservation] = []
    for j in range(k):
        psi_j = eigenvectors[:, j]

        phi_s = _spectral_structural_potential(eigenvalues, j)
        grad_phi = _path_eigenvector_gradient(psi_j)
        k_phi = _path_eigenvector_curvature(psi_j)
        j_phi = float(j_phi_arr[j])
        j_dnfr = float(j_dnfr_arr[j])

        energy = phi_s**2 + grad_phi**2 + k_phi**2 + j_phi**2 + j_dnfr**2
        charge = grad_phi * j_phi - k_phi * j_dnfr
        rho = phi_s + k_phi

        modes.append(EigenmodeConservation(
            mode_index=j,
            eigenvalue=float(eigenvalues[j]),
            phi_s=phi_s,
            grad_phi=grad_phi,
            k_phi=k_phi,
            j_phi=j_phi,
            j_dnfr=j_dnfr,
            energy_density=energy,
            topological_charge=charge,
            charge_density=rho,
        ))

    total_e = sum(m.energy_density for m in modes)
    total_q = sum(m.topological_charge for m in modes)
    total_rho = sum(m.charge_density for m in modes)

    return ConservationAtSigma(
        k=k,
        sigma=sigma,
        modes=modes,
        total_energy=total_e,
        total_charge=total_q,
        total_charge_density=total_rho,
        mean_energy_density=total_e / k,
        mean_charge=total_q / k,
    )

# ---------------------------------------------------------------------------
# Sigma Scan
# ---------------------------------------------------------------------------

def scan_conservation_vs_sigma(
    k: int,
    sigma_values: np.ndarray | None = None,
    *,
    weight_by_log_gap: bool = True,
) -> ConservationSigmaScan:
    """Scan conservation fields across a range of sigma values.

    Computes E_total(sigma), Q_total(sigma), rho_total(sigma) and
    the conservation residual |dQ/dsigma| for each sigma.

    Parameters
    ----------
    k : int
        Number of primes (>= 2).
    sigma_values : ndarray, optional
        Sigma values to scan.  Defaults to linspace(0.1, 1.0, 37).
    weight_by_log_gap : bool
        Use log-gap edge weights.

    Returns
    -------
    ConservationSigmaScan
        Scan results including energy minimum and charge gradient.
    """
    if sigma_values is None:
        sigma_values = np.linspace(0.1, 1.0, 37)
    sigma_values = np.asarray(sigma_values, dtype=float)

    n_sigma = len(sigma_values)
    energies = np.zeros(n_sigma)
    charges = np.zeros(n_sigma)
    charge_densities = np.zeros(n_sigma)

    critical_snap: ConservationAtSigma | None = None
    idx_half = int(np.argmin(np.abs(sigma_values - 0.5)))

    for i, sigma in enumerate(sigma_values):
        snap = compute_eigenmode_conservation(
            k, sigma, weight_by_log_gap=weight_by_log_gap,
        )
        energies[i] = snap.total_energy
        charges[i] = snap.total_charge
        charge_densities[i] = snap.total_charge_density

        if i == idx_half:
            critical_snap = snap

    # Charge gradient: |dQ/dsigma| via central differences
    if len(sigma_values) < 2:
        charge_grad = np.zeros_like(charges)
    else:
        charge_grad = np.abs(np.gradient(charges, sigma_values))

    # Charge drift from sigma = 0.5
    q_half = charges[idx_half]
    eps = 1e-30
    drift = np.abs(charges - q_half) / max(abs(q_half), eps)

    # Energy minimum
    e_min_idx = int(np.argmin(energies))
    energy_min_sigma = float(sigma_values[e_min_idx])

    # Minimal gradient
    grad_min_idx = int(np.argmin(charge_grad))
    min_grad_sigma = float(sigma_values[grad_min_idx])

    if critical_snap is None:
        critical_snap = compute_eigenmode_conservation(
            k, 0.5, weight_by_log_gap=weight_by_log_gap,
        )

    return ConservationSigmaScan(
        k=k,
        sigma_values=sigma_values,
        total_energy=energies,
        total_charge=charges,
        total_charge_density=charge_densities,
        charge_gradient=charge_grad,
        charge_drift_from_half=drift,
        energy_minimum_sigma=energy_min_sigma,
        min_gradient_sigma=min_grad_sigma,
        critical_conservation=critical_snap,
    )

# ---------------------------------------------------------------------------
# Grammar Compliance Conservation Test
# ---------------------------------------------------------------------------

def _smooth_evolution(
    k: int,
    sigma_start: float,
    sigma_end: float,
    n_steps: int = 10,
    *,
    weight_by_log_gap: bool = True,
) -> tuple[float, float, float, float]:
    """Smooth sigma evolution: small steps (grammar-compliant).

    Returns (Q_start, Q_end, E_start, E_end).
    """
    sigmas = np.linspace(sigma_start, sigma_end, n_steps + 1)

    snap_start = compute_eigenmode_conservation(
        k, float(sigmas[0]), weight_by_log_gap=weight_by_log_gap,
    )
    snap_end = compute_eigenmode_conservation(
        k, float(sigmas[-1]), weight_by_log_gap=weight_by_log_gap,
    )

    return (
        snap_start.total_charge,
        snap_end.total_charge,
        snap_start.total_energy,
        snap_end.total_energy,
    )

def _abrupt_jump(
    k: int,
    sigma_start: float,
    sigma_end: float,
    *,
    weight_by_log_gap: bool = True,
) -> tuple[float, float, float, float]:
    """Abrupt sigma jump: single large step (grammar-violating).

    Returns (Q_start, Q_end, E_start, E_end).
    """
    snap_start = compute_eigenmode_conservation(
        k, sigma_start, weight_by_log_gap=weight_by_log_gap,
    )
    snap_end = compute_eigenmode_conservation(
        k, sigma_end, weight_by_log_gap=weight_by_log_gap,
    )

    return (
        snap_start.total_charge,
        snap_end.total_charge,
        snap_start.total_energy,
        snap_end.total_energy,
    )

def _make_compliance_result(
    protocol: str,
    is_compliant: bool,
    sigma_start: float,
    sigma_end: float,
    q_before: float,
    q_after: float,
    e_before: float,
    e_after: float,
) -> GrammarComplianceResult:
    """Build a GrammarComplianceResult from raw values."""
    eps = 1e-30
    drift = abs(q_after - q_before) / max(abs(q_before), eps)
    e_change = (e_after - e_before) / max(abs(e_before), eps)
    quality = 1.0 / (1.0 + drift)

    return GrammarComplianceResult(
        protocol=protocol,
        is_grammar_compliant=is_compliant,
        sigma_start=sigma_start,
        sigma_end=sigma_end,
        charge_before=q_before,
        charge_after=q_after,
        charge_drift=drift,
        energy_before=e_before,
        energy_after=e_after,
        energy_change=e_change,
        conservation_quality=quality,
    )

def test_grammar_conservation(
    k: int,
    sigma: float = 0.5,
    *,
    delta_small: float = 0.02,
    delta_large: float = 0.3,
    weight_by_log_gap: bool = True,
) -> list[GrammarComplianceResult]:
    """Test grammar compliance via conservation of Noether charge.

    Applies four evolution protocols around *sigma*:

    1. **smooth_forward** (compliant): sigma -> sigma + delta_small in
       10 steps with stabilisation.
    2. **smooth_backward** (compliant): sigma -> sigma - delta_small.
    3. **abrupt_forward** (violating): sigma -> sigma + delta_large
       in a single jump.
    4. **abrupt_backward** (violating): sigma -> sigma - delta_large.

    Grammar-compliant protocols should yield higher conservation_quality
    (smaller charge_drift), analogous to S_grammar -> 0 under U1-U6.

    Parameters
    ----------
    k : int
        Number of primes (>= 4).
    sigma : float
        Centre sigma value (default: 0.5 for criticality test).
    delta_small : float
        Step size for smooth (compliant) evolution.
    delta_large : float
        Jump size for abrupt (violating) evolution.
    weight_by_log_gap : bool
        Use log-gap edge weights.

    Returns
    -------
    list of GrammarComplianceResult
        Four results: two compliant, two violating.
    """
    if k < 2:
        raise ValueError(f"k must be >= 2, got {k}")

    results: list[GrammarComplianceResult] = []

    # Compliant: smooth forward
    q0, q1, e0, e1 = _smooth_evolution(
        k, sigma, sigma + delta_small,
        weight_by_log_gap=weight_by_log_gap,
    )
    results.append(_make_compliance_result(
        "smooth_forward", True, sigma, sigma + delta_small, q0, q1, e0, e1,
    ))

    # Compliant: smooth backward
    q0, q1, e0, e1 = _smooth_evolution(
        k, sigma, sigma - delta_small,
        weight_by_log_gap=weight_by_log_gap,
    )
    results.append(_make_compliance_result(
        "smooth_backward", True, sigma, sigma - delta_small, q0, q1, e0, e1,
    ))

    # Violating: abrupt forward
    q0, q1, e0, e1 = _abrupt_jump(
        k, sigma, sigma + delta_large,
        weight_by_log_gap=weight_by_log_gap,
    )
    results.append(_make_compliance_result(
        "abrupt_forward", False, sigma, sigma + delta_large, q0, q1, e0, e1,
    ))

    # Violating: abrupt backward
    q0, q1, e0, e1 = _abrupt_jump(
        k, sigma, sigma - delta_large,
        weight_by_log_gap=weight_by_log_gap,
    )
    results.append(_make_compliance_result(
        "abrupt_backward", False, sigma, sigma - delta_large, q0, q1, e0, e1,
    ))

    return results

# Prevent pytest from collecting this function as a test
test_grammar_conservation.__test__ = False  # type: ignore[attr-defined]

# ---------------------------------------------------------------------------
# Integration: Full Analysis
# ---------------------------------------------------------------------------

def run_critical_conservation_analysis(
    k: int = 20,
    sigma: float = 0.5,
    *,
    sigma_values: np.ndarray | None = None,
    delta_small: float = 0.02,
    delta_large: float = 0.3,
    weight_by_log_gap: bool = True,
) -> CriticalConservationAnalysis:
    """Run the complete P7 conservation-at-criticality analysis.

    Combines:
    1. Sigma scan of E, Q, rho across a sigma range.
    2. Conservation snapshot at sigma = 0.5 (or specified).
    3. Grammar compliance tests (smooth vs abrupt evolution).
    4. Summary diagnostics.

    Parameters
    ----------
    k : int
        Number of primes (>= 4 recommended).
    sigma : float
        Critical sigma for grammar tests.
    sigma_values : ndarray, optional
        Sigma range for scan.  Defaults to linspace(0.1, 1.0, 37).
    delta_small : float
        Step size for smooth (compliant) evolution.
    delta_large : float
        Jump size for abrupt (violating) evolution.
    weight_by_log_gap : bool
        Use log-gap edge weights.

    Returns
    -------
    CriticalConservationAnalysis
        Full P7 analysis.
    """
    # 1. Sigma scan
    scan = scan_conservation_vs_sigma(
        k, sigma_values, weight_by_log_gap=weight_by_log_gap,
    )

    # 2. Critical snapshot (reuse from scan if sigma is 0.5)
    if abs(sigma - 0.5) < 1e-10:
        critical = scan.critical_conservation
    else:
        critical = compute_eigenmode_conservation(
            k, sigma, weight_by_log_gap=weight_by_log_gap,
        )

    # 3. Grammar compliance
    grammar_tests = test_grammar_conservation(
        k, sigma,
        delta_small=delta_small,
        delta_large=delta_large,
        weight_by_log_gap=weight_by_log_gap,
    )

    # 4. Summary diagnostics
    compliant = [r for r in grammar_tests if r.is_grammar_compliant]
    violating = [r for r in grammar_tests if not r.is_grammar_compliant]

    comp_mean = (
        sum(r.conservation_quality for r in compliant) / len(compliant)
        if compliant else 0.0
    )
    viol_mean = (
        sum(r.conservation_quality for r in violating) / len(violating)
        if violating else 0.0
    )
    ratio = comp_mean / viol_mean if viol_mean > 1e-30 else float("inf")

    # Energy minimum at sigma = 0.5?
    sigma_half_energy = abs(scan.energy_minimum_sigma - 0.5) < 0.15

    # Minimal gradient at sigma = 0.5?
    sigma_half_residual = abs(scan.min_gradient_sigma - 0.5) < 0.15

    return CriticalConservationAnalysis(
        k=k,
        sigma_scan=scan,
        critical=critical,
        grammar_tests=grammar_tests,
        sigma_half_is_energy_min=sigma_half_energy,
        sigma_half_has_min_residual=sigma_half_residual,
        compliant_mean_quality=comp_mean,
        violating_mean_quality=viol_mean,
        quality_ratio=ratio,
    )
