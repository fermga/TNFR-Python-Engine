r"""Per-eigenmode structural field tetrad for TNFR-Riemann spectral analysis.

Bridges the TNFR canonical measurement infrastructure (Phi_s, |grad_phi|,
K_phi, xi_C) to the spectral domain of the discrete operator
H^(k)(sigma) = L_k + V_sigma.

For each eigenmode psi_j with eigenvalue lambda_j, the four tetrad fields:

  Phi_s(j) = sum_{m != j} |lambda_m| / |j - m|^2
      Spectral structural potential: pressure felt by eigenmode j
      from all other eigenmodes, weighted by inverse spectral distance.

  |grad_phi|(j) = (1/k) sum_i mean_{n in N(i)} |psi_j(n) - psi_j(i)|
      Eigenvector gradient: spatial variation rate of psi_j on the graph.

  K_phi(j) = (1/k) sum_i |psi_j(i) - mean_{n in N(i)} psi_j(n)|
      Eigenvector curvature: discrete Laplacian magnitude of psi_j.

  xi_C(j) from exponential fit of C(r) = <|psi_j(i)|^2 |psi_j(i')|^2>
      Coherence length of eigenmode probability density.

Key results:
  - U6 analogue: test |Phi_s(j)| < phi_threshold at eigenmode level
  - Confinement transitions: eigenmodes violating U6 away from sigma=1/2
  - Spectral-structural bridge: eigenvalue structure -> TNFR fields

TNFR physics basis: the structural field tetrad is the canonical
diagnostic toolkit for TNFR coherence (AGENTS.md, Structural Field
Tetrad).  Extending it to eigenmodes of H^(k)(sigma) reveals how
spectral structure maps to structural confinement -- the core of the
TNFR-Riemann bridge.

Status: EXPERIMENTAL -- Research prototype for TNFR-Riemann P3 program.

References
----------
- AGENTS.md: Structural Field Tetrad, U6 Structural Potential Confinement
- src/tnfr/physics/canonical.py: Canonical field implementations
- theory/TNFR_RIEMANN_RESEARCH_NOTES.md: TNFR-Riemann program
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..constants.canonical import PHI_S_VON_KOCH_THRESHOLD
from ..constants.canonical import (
    U6_STRUCTURAL_POTENTIAL_LIMIT as PHI_S_GOLDEN_THRESHOLD,
)
from .operator import build_h_tnfr
from .spectral_proof import compute_eigensystem

# ============================================================================
# Data Structures
# ============================================================================


@dataclass
class EigenmodeTetrad:
    """Structural field tetrad for a single eigenmode psi_j.

    Each field is a scalar summary of the per-node field values for
    eigenmode j, providing a single number characterising that mode's
    structural properties in the four canonical dimensions.

    Attributes
    ----------
    mode_index : int
        Index j in sorted eigenvalue order (0-based).
    eigenvalue : float
        Eigenvalue lambda_j of H^(k)(sigma).
    phi_s : float
        Spectral structural potential Phi_s(j).
    grad_phi : float
        Eigenvector gradient magnitude |grad_phi|(j).
    k_phi : float
        Eigenvector curvature |K_phi(j)|.
    xi_c : float
        Coherence length xi_C(j) from probability density autocorrelation.
    """

    mode_index: int
    eigenvalue: float
    phi_s: float
    grad_phi: float
    k_phi: float
    xi_c: float


@dataclass
class EigenmodeFieldAnalysis:
    """Complete per-eigenmode tetrad analysis at given (k, sigma).

    Attributes
    ----------
    k : int
        Number of primes (graph size).
    sigma : float
        Structural parameter.
    tetrads : list of EigenmodeTetrad
        One tetrad per eigenmode, sorted by eigenvalue.
    u6_violations : list of int
        Indices of eigenmodes violating |Phi_s| >= threshold.
    u6_threshold : float
        Threshold used for U6 confinement test.
    u6_fraction : float
        Fraction of modes satisfying U6 confinement.
    mean_phi_s : float
        Mean |Phi_s| across all eigenmodes.
    mean_grad_phi : float
        Mean |grad_phi| across all eigenmodes.
    mean_k_phi : float
        Mean K_phi across all eigenmodes.
    mean_xi_c : float
        Mean xi_C across non-NaN eigenmodes.
    """

    k: int
    sigma: float
    tetrads: list[EigenmodeTetrad]
    u6_violations: list[int]
    u6_threshold: float
    u6_fraction: float
    mean_phi_s: float
    mean_grad_phi: float
    mean_k_phi: float
    mean_xi_c: float


# ============================================================================
# Public API
# ============================================================================

__all__ = [
    # Data structures
    "EigenmodeTetrad",
    "EigenmodeFieldAnalysis",
    # Constants
    "PHI_S_VON_KOCH_THRESHOLD",
    "PHI_S_GOLDEN_THRESHOLD",
    # Core
    "compute_eigenmode_tetrad",
    "compute_eigenmode_fields_general",
    # Diagnostics
    "check_u6_confinement",
    "compare_confinement_at_sigma",
]

# ============================================================================
# Core: Path Graph (Tridiagonal) -- O(k^2) via compute_eigensystem
# ============================================================================


def compute_eigenmode_tetrad(
    k: int,
    sigma: float = 0.5,
    *,
    u6_threshold: float = PHI_S_VON_KOCH_THRESHOLD,
    weight_by_log_gap: bool = True,
) -> EigenmodeFieldAnalysis:
    """Compute structural field tetrad for all eigenmodes of H^(k)(sigma).

    Uses the tridiagonal path graph structure for efficient computation.

    Parameters
    ----------
    k : int
        Number of primes (graph size).  Must be >= 2.
    sigma : float
        Structural parameter Re(s) analogue.
    u6_threshold : float
        Threshold for U6 confinement test on |Phi_s(j)|.
    weight_by_log_gap : bool
        Use log-gap edge weights (default True).

    Returns
    -------
    EigenmodeFieldAnalysis
        Complete tetrad analysis for all eigenmodes.
    """
    if k < 2:
        raise ValueError(f"k must be >= 2, got {k}")

    eigenvalues, eigenvectors = compute_eigensystem(
        k,
        sigma,
        weight_by_log_gap=weight_by_log_gap,
    )

    tetrads = []
    for j in range(k):
        psi_j = eigenvectors[:, j]

        phi_s = _spectral_structural_potential(eigenvalues, j)
        grad_phi = _path_eigenvector_gradient(psi_j)
        k_phi = _path_eigenvector_curvature(psi_j)
        xi_c = _eigenvector_coherence_length_path(psi_j)

        tetrads.append(
            EigenmodeTetrad(
                mode_index=j,
                eigenvalue=float(eigenvalues[j]),
                phi_s=phi_s,
                grad_phi=grad_phi,
                k_phi=k_phi,
                xi_c=xi_c,
            )
        )

    return _build_analysis(k, sigma, tetrads, u6_threshold)


# ============================================================================
# Core: General Graph -- dense eigensystem via scipy.linalg.eigh
# ============================================================================


def compute_eigenmode_fields_general(
    G,
    sigma: float = 0.5,
    *,
    u6_threshold: float = PHI_S_VON_KOCH_THRESHOLD,
) -> EigenmodeFieldAnalysis:
    """Compute per-eigenmode tetrad for any graph topology.

    Uses the dense H_TNFR matrix from build_h_tnfr(G, sigma) and
    scipy.linalg.eigh for O(k^3) diagonalisation.

    Parameters
    ----------
    G : nx.Graph
        Prime-labelled NetworkX graph with TNFR node attributes.
        Edge weights (e.g. from log-gap weighting) are read directly
        from the graph; use the appropriate builder to control them.
    sigma : float
        Structural parameter.
    u6_threshold : float
        Threshold for U6 confinement test.

    Returns
    -------
    EigenmodeFieldAnalysis
        Complete tetrad analysis for all eigenmodes.

    Raises
    ------
    ValueError
        If the graph has fewer than 2 nodes.
    """
    import scipy.linalg as sla

    if len(G) < 2:
        raise ValueError(f"Graph must have >= 2 nodes, got {len(G)}")

    H, _diag_V = build_h_tnfr(G, sigma=sigma)
    k = H.shape[0]
    eigenvalues, eigenvectors = sla.eigh(H)

    nodes = list(G.nodes())
    node_to_idx = {n: i for i, n in enumerate(nodes)}

    adjacency: list[list[int]] = []
    for node in nodes:
        adjacency.append(
            [node_to_idx[n] for n in G.neighbors(node) if n in node_to_idx]
        )

    dist_matrix = _compute_distance_matrix(G, nodes, node_to_idx)

    tetrads = []
    for j in range(k):
        psi_j = eigenvectors[:, j]

        phi_s = _spectral_structural_potential(eigenvalues, j)
        grad_phi = _general_eigenvector_gradient(psi_j, adjacency)
        k_phi = _general_eigenvector_curvature(psi_j, adjacency)
        xi_c = _eigenvector_coherence_length_general(psi_j, dist_matrix)

        tetrads.append(
            EigenmodeTetrad(
                mode_index=j,
                eigenvalue=float(eigenvalues[j]),
                phi_s=phi_s,
                grad_phi=grad_phi,
                k_phi=k_phi,
                xi_c=xi_c,
            )
        )

    return _build_analysis(k, sigma, tetrads, u6_threshold)


# ============================================================================
# Diagnostics
# ============================================================================


def check_u6_confinement(
    analysis: EigenmodeFieldAnalysis,
    u6_threshold: float | None = None,
    *,
    threshold: float | None = None,
) -> dict:
    """Check U6 structural confinement at eigenmode level.

    Parameters
    ----------
    analysis : EigenmodeFieldAnalysis
        Tetrad analysis to check.
    u6_threshold : float or None
        Confinement threshold.  Falls back to *analysis.u6_threshold*.
    threshold : float or None
        **Deprecated** alias for *u6_threshold* (kept for backwards
        compatibility).  If both are given, *u6_threshold* wins.

    Returns
    -------
    dict
        confined : bool -- True if all modes satisfy |Phi_s| < threshold.
        fraction : float -- fraction of confined modes.
        violations : list[int] -- mode indices violating confinement.
        max_phi_s : float -- maximum |Phi_s| across modes.
        threshold : float -- threshold used.
    """
    effective = (
        u6_threshold
        if u6_threshold is not None
        else threshold if threshold is not None else analysis.u6_threshold
    )
    t = effective
    violations = [
        tetrad.mode_index for tetrad in analysis.tetrads if abs(tetrad.phi_s) >= t
    ]
    phi_s_abs = [abs(tetrad.phi_s) for tetrad in analysis.tetrads]

    n = len(analysis.tetrads)
    return {
        "confined": len(violations) == 0,
        "fraction": 1.0 - len(violations) / n if n > 0 else 1.0,
        "violations": violations,
        "max_phi_s": max(phi_s_abs) if phi_s_abs else 0.0,
        "threshold": t,
    }


def compare_confinement_at_sigma(
    k: int,
    sigma_values: list[float] | None = None,
    *,
    u6_threshold: float = PHI_S_VON_KOCH_THRESHOLD,
    weight_by_log_gap: bool = True,
) -> dict[float, dict]:
    """Compare U6 confinement across multiple sigma values.

    Tests the hypothesis that eigenmodes are better confined at
    sigma = 1/2 than away from it.

    Parameters
    ----------
    k : int
        Graph size.
    sigma_values : list of float or None
        Sigma values to test.  Defaults to [0.3, 0.4, 0.5, 0.6, 0.7].
    u6_threshold : float
        Confinement threshold.

    Returns
    -------
    dict mapping sigma -> U6 test result dict (augmented with mean fields).
    """
    if sigma_values is None:
        sigma_values = [0.3, 0.4, 0.5, 0.6, 0.7]

    results: dict[float, dict] = {}
    for sigma in sigma_values:
        analysis = compute_eigenmode_tetrad(
            k,
            sigma,
            u6_threshold=u6_threshold,
            weight_by_log_gap=weight_by_log_gap,
        )
        r = check_u6_confinement(analysis, u6_threshold)
        r["mean_phi_s"] = analysis.mean_phi_s
        r["mean_grad_phi"] = analysis.mean_grad_phi
        r["mean_k_phi"] = analysis.mean_k_phi
        r["mean_xi_c"] = analysis.mean_xi_c
        results[sigma] = r

    return results


# ============================================================================
# Internal: Spectral Structural Potential Phi_s(j)
# ============================================================================


def _spectral_structural_potential(eigenvalues: np.ndarray, j: int) -> float:
    r"""Compute spectral structural potential for eigenmode j.

    Phi_s(j) = sum_{m != j} |lambda_m| / |j - m|^2

    Analogous to canonical Phi_s(i) = sum_{n != i} DELTA_NFR_n / d(i,n)^alpha
    where |lambda_m| plays the role of structural pressure and the
    eigenmode index distance |j - m| plays the role of graph distance.
    """
    k = len(eigenvalues)
    indices = np.arange(k, dtype=np.float64)
    distances_sq = (indices - j) ** 2
    distances_sq[j] = 1.0  # avoid division by zero (zeroed below)

    contributions = np.abs(eigenvalues) / distances_sq
    contributions[j] = 0.0

    return float(np.sum(contributions))


# ============================================================================
# Internal: Path Graph Field Computations
# ============================================================================


def _path_eigenvector_gradient(psi: np.ndarray) -> float:
    r"""Compute mean eigenvector gradient |nabla psi| on path graph.

    For the path graph, node i connects to i +/- 1.
    |nabla psi|(j,i) = mean of |psi(n) - psi(i)| over neighbors n.

    Returns the graph-averaged gradient: (1/k) sum_i |nabla psi|(j,i).
    """
    k = len(psi)
    if k < 2:
        return 0.0

    fwd = np.abs(np.diff(psi))  # |psi(i+1) - psi(i)|, length k-1

    node_grads = np.empty(k)
    node_grads[0] = fwd[0]
    node_grads[-1] = fwd[-1]
    if k > 2:
        node_grads[1:-1] = 0.5 * (fwd[:-1] + fwd[1:])

    return float(np.mean(node_grads))


def _path_eigenvector_curvature(psi: np.ndarray) -> float:
    r"""Compute mean eigenvector curvature K_phi on path graph.

    K_phi(j,i) = |psi(i) - mean(neighbors)|.
    For path: mean(neighbors) = (psi(i-1) + psi(i+1))/2 interior,
    or psi(i +/- 1) for boundaries.

    Returns graph-averaged curvature: (1/k) sum_i |K_phi(j,i)|.
    """
    k = len(psi)
    if k < 2:
        return 0.0

    curvatures = np.empty(k)
    curvatures[0] = abs(psi[0] - psi[1])
    curvatures[-1] = abs(psi[-1] - psi[-2])

    if k > 2:
        neighbor_mean = 0.5 * (psi[:-2] + psi[2:])
        curvatures[1:-1] = np.abs(psi[1:-1] - neighbor_mean)

    return float(np.mean(curvatures))


def _eigenvector_coherence_length_path(psi: np.ndarray) -> float:
    r"""Compute coherence length xi_C from |psi|^2 autocorrelation on path graph.

    Uses d(i, m) = |i - m| on the path graph.
    Computes C(r) = mean_{|i-m|=r} rho(i) rho(m), rho = |psi|^2.
    Fits exponential decay C(r) ~ exp(-r / xi_C).
    """
    k = len(psi)
    if k < 4:
        return float("nan")

    rho = psi**2  # probability density (real eigenvectors)
    max_r = min(k - 1, k // 2)

    distances = []
    correlations = []

    for r in range(1, max_r + 1):
        corr = float(np.mean(rho[: k - r] * rho[r:]))
        if corr > 1e-15:
            distances.append(r)
            correlations.append(corr)

    if len(distances) < 2:
        return float("nan")

    distances_arr = np.array(distances, dtype=np.float64)
    log_corr = np.log(np.array(correlations))

    try:
        slope, _ = np.polyfit(distances_arr, log_corr, 1)
        if slope >= 0:
            return float("nan")
        xi_c = -1.0 / slope
        return float(xi_c) if xi_c > 0 else float("nan")
    except (np.linalg.LinAlgError, ValueError):
        return float("nan")


# ============================================================================
# Internal: General Graph Field Computations
# ============================================================================


def _general_eigenvector_gradient(
    psi: np.ndarray,
    adjacency: list[list[int]],
) -> float:
    r"""Compute mean eigenvector gradient on general graph.

    |nabla psi|(j,i) = mean_{n in N(i)} |psi(n) - psi(i)|
    """
    k = len(psi)
    if k < 2:
        return 0.0

    total = 0.0
    for i in range(k):
        neighbors = adjacency[i]
        if not neighbors:
            continue
        diff_sum = sum(abs(psi[n] - psi[i]) for n in neighbors)
        total += diff_sum / len(neighbors)

    return total / k


def _general_eigenvector_curvature(
    psi: np.ndarray,
    adjacency: list[list[int]],
) -> float:
    r"""Compute mean eigenvector curvature on general graph.

    K_phi(j,i) = |psi(i) - mean_{n in N(i)} psi(n)|
    """
    k = len(psi)
    if k < 2:
        return 0.0

    total = 0.0
    for i in range(k):
        neighbors = adjacency[i]
        if not neighbors:
            continue
        neighbor_mean = sum(psi[n] for n in neighbors) / len(neighbors)
        total += abs(psi[i] - neighbor_mean)

    return total / k


def _eigenvector_coherence_length_general(
    psi: np.ndarray,
    dist_matrix: np.ndarray,
) -> float:
    r"""Compute coherence length from |psi|^2 autocorrelation on general graph.

    Groups node pairs by graph distance, computes mean product of
    rho = |psi|^2, and fits exponential decay.
    """
    k = len(psi)
    if k < 4:
        return float("nan")

    rho = psi**2
    finite_mask = dist_matrix < np.inf
    np.fill_diagonal(finite_mask, False)
    max_dist = int(np.max(dist_matrix[finite_mask])) if np.any(finite_mask) else 0

    rho_outer = np.outer(rho, rho)

    distances = []
    correlations = []

    for r in range(1, min(max_dist + 1, k)):
        mask = dist_matrix == r
        if not np.any(mask):
            continue
        mean_corr = float(np.mean(rho_outer[mask]))
        if mean_corr > 1e-15:
            distances.append(r)
            correlations.append(mean_corr)

    if len(distances) < 2:
        return float("nan")

    distances_arr = np.array(distances, dtype=np.float64)
    log_corr = np.log(np.array(correlations))

    try:
        slope, _ = np.polyfit(distances_arr, log_corr, 1)
        if slope >= 0:
            return float("nan")
        xi_c = -1.0 / slope
        return float(xi_c) if xi_c > 0 else float("nan")
    except (np.linalg.LinAlgError, ValueError):
        return float("nan")


def _compute_distance_matrix(G, nodes, node_to_idx) -> np.ndarray:
    """Compute all-pairs shortest path distance matrix."""
    import networkx as nx

    k = len(nodes)
    dist_matrix = np.full((k, k), np.inf)
    np.fill_diagonal(dist_matrix, 0)

    lengths = dict(nx.all_pairs_shortest_path_length(G))
    for src in lengths:
        src_idx = node_to_idx[src]
        for dst, dist in lengths[src].items():
            dst_idx = node_to_idx[dst]
            dist_matrix[src_idx, dst_idx] = dist

    return dist_matrix


# ============================================================================
# Internal: Analysis Builder
# ============================================================================


def _build_analysis(
    k: int,
    sigma: float,
    tetrads: list[EigenmodeTetrad],
    u6_threshold: float,
) -> EigenmodeFieldAnalysis:
    """Construct EigenmodeFieldAnalysis from tetrad list."""
    u6_violations = [t.mode_index for t in tetrads if abs(t.phi_s) >= u6_threshold]
    u6_fraction = 1.0 - len(u6_violations) / k if k > 0 else 1.0

    phi_s_vals = [abs(t.phi_s) for t in tetrads]
    grad_vals = [t.grad_phi for t in tetrads]
    k_phi_vals = [t.k_phi for t in tetrads]
    xi_c_vals = [t.xi_c for t in tetrads if not math.isnan(t.xi_c)]

    return EigenmodeFieldAnalysis(
        k=k,
        sigma=sigma,
        tetrads=tetrads,
        u6_violations=u6_violations,
        u6_threshold=u6_threshold,
        u6_fraction=u6_fraction,
        mean_phi_s=float(np.mean(phi_s_vals)) if phi_s_vals else 0.0,
        mean_grad_phi=float(np.mean(grad_vals)) if grad_vals else 0.0,
        mean_k_phi=float(np.mean(k_phi_vals)) if k_phi_vals else 0.0,
        mean_xi_c=float(np.mean(xi_c_vals)) if xi_c_vals else float("nan"),
    )
