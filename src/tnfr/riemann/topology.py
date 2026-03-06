r"""Alternative topology analysis for the TNFR-Riemann operator.

Investigates whether the structural equilibrium sigma = 1/2 and the
thermodynamic convergence sigma* -> 1/2 are universal across graph
topologies or specific to the prime path graph.

Key insight: the equilibrium lambda_min(H(1/2)) = 0 is trivially exact
for ANY connected graph (L is PSD with ker = span{1}).  The prime
structure enters through edge weights and eigenvector statistics,
which affect:
  - sigma* via tr(L V_1) (weighted degree * log-prime correlation)
  - Spectral gap (graph connectivity / expansion)
  - Eigenvalue velocities (eigenvector localisation on primes)
  - Convergence rate |sigma* - 1/2| ~ O(1/k^beta)

The curvature d^2E/dsigma^2 = (1/k) tr(V_1^2) depends only on node
labels, not topology, so it is identical across topologies for the same
set of primes.

TNFR physics basis: different topologies test different coupling regimes
of the phase-gated coupling condition |phi_i - phi_j| <= Delta_phi_max
(grammar rule U3), from nearest-neighbour (path) to all-to-all (K_k).

Status: EXPERIMENTAL -- Research prototype for TNFR-Riemann P2 program.

References
----------
- AGENTS.md: Canonical Invariant #2 (Phase-Coherent Coupling)
- theory/TNFR_RIEMANN_RESEARCH_NOTES.md sec. 15.2 (universality)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Sequence

import scipy.linalg

from ..mathematics.unified_numerical import np
import networkx as nx

# ---------------------------------------------------------------------------
# Convergence exponent classification
# ---------------------------------------------------------------------------
_CONVERGENCE_SPREAD_NARROW = 0.3

from .operator import (
    build_h_tnfr,
    build_prime_path_graph,
    build_prime_cycle_graph,
    build_prime_star_graph,
    build_prime_complete_graph,
    build_prime_tree_graph,
    build_prime_random_graph,
)

# ============================================================================
# Public API
# ============================================================================

__all__ = [
    # Data structures
    "TopologyResult",
    "TopologyConvergenceResult",
    # Builders registry
    "TOPOLOGY_BUILDERS",
    # Analysis
    "analyze_graph_topology",
    "compare_topologies",
    "topology_convergence_study",
]

# ============================================================================
# Topology Builder Registry
# ============================================================================

# Each builder: (count, *, weight_by_log_gap=True) -> nx.Graph
# The random builder needs a seed keyword, handled separately.
GraphBuilder = Callable[..., nx.Graph]

TOPOLOGY_BUILDERS: dict[str, GraphBuilder] = {
    "path": build_prime_path_graph,
    "cycle": build_prime_cycle_graph,
    "star": build_prime_star_graph,
    "complete": build_prime_complete_graph,
    "tree": build_prime_tree_graph,
    "random": build_prime_random_graph,
}

# ============================================================================
# Data Structures
# ============================================================================

@dataclass(frozen=True)
class TopologyResult:
    r"""Analysis result for a single topology at a single k.

    Attributes
    ----------
    topology : str
        Topology name (path, cycle, star, complete, tree, random).
    k : int
        Number of prime-labeled nodes.
    n_edges : int
        Number of edges in the graph.
    lambda_min : float
        lambda_min(H(1/2)) -- should be ~0 for any connected graph.
    spectral_gap : float
        lambda_2 - lambda_1 at sigma = 1/2.
    spectral_width : float
        lambda_max - lambda_min at sigma = 1/2.
    sigma_star : float
        Analytic thermodynamic attractor: 1/2 - tr(L V_1) / tr(V_1^2).
    deviation : float
        |sigma* - 1/2|.
    cross_term : float
        tr(L V_1) -- topology-dependent prime-gap correlation.
    potential_norm : float
        tr(V_1^2) -- topology-independent (depends only on labels).
    curvature : float
        d^2E/dsigma^2 = (1/k) tr(V_1^2) -- topology-independent.
    all_velocities_positive : bool
        True iff all d(lambda_j)/dsigma > 0 at sigma = 1/2.
    min_velocity : float
        Smallest eigenvalue velocity.
    max_velocity : float
        Largest eigenvalue velocity.
    """

    topology: str
    k: int
    n_edges: int
    lambda_min: float
    spectral_gap: float
    spectral_width: float
    sigma_star: float
    deviation: float
    cross_term: float
    potential_norm: float
    curvature: float
    all_velocities_positive: bool
    min_velocity: float
    max_velocity: float

@dataclass
class TopologyConvergenceResult:
    r"""Convergence comparison across topologies and k values.

    Attributes
    ----------
    k_values : list[int]
        Graph sizes tested.
    topologies : list[str]
        Topology names.
    results : dict[str, list[TopologyResult]]
        topology name -> list of results (one per k).
    convergence_rates : dict[str, tuple[float, float]]
        topology name -> (A, beta) from |sigma* - 1/2| ~ A / k^beta.
    gap_exponents : dict[str, float]
        topology name -> alpha from spectral_gap ~ C / k^alpha.
    summary : str
        Human-readable comparison table.
    """

    k_values: list[int] = field(default_factory=list)
    topologies: list[str] = field(default_factory=list)
    results: dict[str, list[TopologyResult]] = field(default_factory=dict)
    convergence_rates: dict[str, tuple[float, float]] = field(default_factory=dict)
    gap_exponents: dict[str, float] = field(default_factory=dict)
    summary: str = ""

# ============================================================================
# Core Analysis: Works with Any Connected Graph
# ============================================================================

def _build_graph(
    name: str,
    k: int,
    *,
    weight_by_log_gap: bool = True,
    seed: int = 42,
) -> nx.Graph:
    """Build a graph by topology name from the registry."""
    builder = TOPOLOGY_BUILDERS[name]
    if name == "random":
        return builder(k, seed=seed, weight_by_log_gap=weight_by_log_gap)
    return builder(k, weight_by_log_gap=weight_by_log_gap)

def analyze_graph_topology(
    G: nx.Graph,
    name: str,
    *,
    weight_by_log_gap: bool = True,
) -> TopologyResult:
    r"""Run the four lines of attack on an arbitrary prime-labeled graph.

    Uses the dense eigensolver (scipy.linalg.eigh) for generality.
    For path-specific analysis with tridiagonal acceleration, use the
    functions in spectral_proof.py.

    Parameters
    ----------
    G : nx.Graph
        Connected graph with integer ``label`` attributes on nodes.
    name : str
        Topology name for identification.
    weight_by_log_gap : bool
        Whether log-gap edge weights are used (for reference).

    Returns
    -------
    TopologyResult
    """
    k = G.number_of_nodes()
    n_edges = G.number_of_edges()

    # Build dense H at sigma = 1/2 (pure Laplacian)
    H_half, _ = build_h_tnfr(G, sigma=0.5)
    eigenvalues, eigenvectors = scipy.linalg.eigh(H_half)

    # Line 1: Structural equilibrium
    lambda_min = float(eigenvalues[0])
    spectral_gap = float(eigenvalues[1] - eigenvalues[0]) if k >= 2 else 0.0
    spectral_width = float(eigenvalues[-1] - eigenvalues[0]) if k >= 2 else 0.0

    # V_1 = diag(log p_i) -- extract from labels
    nodes = sorted(G.nodes())
    labels = [G.nodes[n]["label"] for n in nodes]
    log_p = np.array([np.log(float(lbl)) for lbl in labels])

    # Line 2: Thermodynamic attractor (analytic sigma*)
    # tr(L V_1) = sum_i L_ii * log(p_i) where L_ii = weighted degree
    diag_L = np.diag(H_half)  # At sigma=1/2, H = L
    tr_LV1 = float(np.dot(diag_L, log_p))
    tr_V1_sq = float(np.sum(log_p ** 2))

    if tr_V1_sq > 1e-15:
        sigma_star = 0.5 - tr_LV1 / tr_V1_sq
    else:
        sigma_star = 0.5

    deviation = abs(sigma_star - 0.5)
    curvature = tr_V1_sq / k

    # Line 3: Eigenvalue flow (Hellmann-Feynman)
    # v_j = <psi_j|V_1|psi_j> = sum_i |psi_j(i)|^2 log(p_i)
    velocities = np.array([
        float(np.sum(eigenvectors[:, j] ** 2 * log_p))
        for j in range(k)
    ])
    all_pos = bool(np.all(velocities > 0))
    v_min = float(np.min(velocities))
    v_max = float(np.max(velocities))

    return TopologyResult(
        topology=name,
        k=k,
        n_edges=n_edges,
        lambda_min=lambda_min,
        spectral_gap=spectral_gap,
        spectral_width=spectral_width,
        sigma_star=sigma_star,
        deviation=deviation,
        cross_term=tr_LV1,
        potential_norm=tr_V1_sq,
        curvature=curvature,
        all_velocities_positive=all_pos,
        min_velocity=v_min,
        max_velocity=v_max,
    )

def compare_topologies(
    k: int,
    *,
    topologies: Sequence[str] | None = None,
    weight_by_log_gap: bool = True,
    seed: int = 42,
) -> dict[str, TopologyResult]:
    r"""Compare all topology types at a given graph size k.

    Parameters
    ----------
    k : int
        Number of prime-labeled nodes.
    topologies : sequence of str, optional
        Topology names to compare. Default: all registered topologies.
    weight_by_log_gap : bool
        Use log-gap edge weights.
    seed : int
        Random seed for stochastic topologies.

    Returns
    -------
    dict mapping topology name to TopologyResult.
    """
    if topologies is None:
        topologies = list(TOPOLOGY_BUILDERS.keys())

    results: dict[str, TopologyResult] = {}
    for name in topologies:
        G = _build_graph(name, k, weight_by_log_gap=weight_by_log_gap, seed=seed)
        results[name] = analyze_graph_topology(
            G, name, weight_by_log_gap=weight_by_log_gap
        )
    return results

def _fit_power_law(
    x_values: Sequence[float],
    y_values: Sequence[float],
) -> tuple[float, float]:
    r"""Fit y = A * x^{-beta} via log-log least squares."""
    valid = [(x, y) for x, y in zip(x_values, y_values) if x > 0 and y > 0]
    if len(valid) < 2:
        return (0.0, 1.0)

    log_x = np.array([math.log(x) for x, _ in valid])
    log_y = np.array([math.log(y) for _, y in valid])
    A_mat = np.column_stack([np.ones_like(log_x), -log_x])
    result, _, _, _ = np.linalg.lstsq(A_mat, log_y, rcond=None)
    return (math.exp(float(result[0])), float(result[1]))

def topology_convergence_study(
    k_values: Sequence[int] | None = None,
    *,
    topologies: Sequence[str] | None = None,
    weight_by_log_gap: bool = True,
    seed: int = 42,
) -> TopologyConvergenceResult:
    r"""Study sigma* convergence rate across topologies and k values.

    The central question: does |sigma* - 1/2| ~ A / k^beta have the
    same exponent beta for all topologies (graph-universal), or does
    it vary (topology-specific)?

    If beta ~ 1 universally, the O(1/k) convergence is a property of
    the prime labels through the PNT, not of graph structure.  If beta
    varies, the topology has structural significance beyond labels.

    Parameters
    ----------
    k_values : sequence of int, optional
        Graph sizes. Default: [10, 20, 50, 100, 200, 500].
    topologies : sequence of str, optional
        Topology names. Default: all registered.
    weight_by_log_gap : bool
        Use log-gap edge weights.
    seed : int
        Random seed for stochastic topologies.

    Returns
    -------
    TopologyConvergenceResult
    """
    if k_values is None:
        k_values = [10, 20, 50, 100, 200, 500]
    if topologies is None:
        topologies = list(TOPOLOGY_BUILDERS.keys())

    result = TopologyConvergenceResult(
        k_values=list(k_values),
        topologies=list(topologies),
    )

    # Collect results for each topology
    for name in topologies:
        topo_results: list[TopologyResult] = []
        for k in k_values:
            if k < 3:
                continue
            G = _build_graph(
                name, k, weight_by_log_gap=weight_by_log_gap, seed=seed
            )
            tr = analyze_graph_topology(
                G, name, weight_by_log_gap=weight_by_log_gap
            )
            topo_results.append(tr)
        result.results[name] = topo_results

    # Fit convergence rates |sigma* - 1/2| ~ A / k^beta
    for name, topo_results in result.results.items():
        ks = [float(r.k) for r in topo_results]
        devs = [r.deviation for r in topo_results]
        A, beta = _fit_power_law(ks, devs)
        result.convergence_rates[name] = (A, beta)

    # Fit spectral gap scaling gap ~ C / k^alpha
    for name, topo_results in result.results.items():
        ks = [float(r.k) for r in topo_results if r.spectral_gap > 1e-15]
        gaps = [r.spectral_gap for r in topo_results if r.spectral_gap > 1e-15]
        if len(ks) >= 2:
            _, alpha = _fit_power_law(ks, gaps)
            result.gap_exponents[name] = alpha
        else:
            result.gap_exponents[name] = 0.0

    # Build summary table
    lines: list[str] = []
    lines.append("TNFR-Riemann Topology Comparison")
    lines.append("=" * 50)
    lines.append("")
    lines.append(f"k values: {result.k_values}")
    lines.append("")

    # Convergence rate table
    lines.append("Convergence: |sigma* - 1/2| ~ A / k^beta")
    lines.append(f"  {'Topology':>10}  {'A':>8}  {'beta':>8}  "
                 f"{'gap_exp':>8}  {'universal?':>10}")
    lines.append(f"  {'─' * 10}  {'─' * 8}  {'─' * 8}  "
                 f"{'─' * 8}  {'─' * 10}")

    for name in topologies:
        A, beta = result.convergence_rates.get(name, (0.0, 0.0))
        alpha = result.gap_exponents.get(name, 0.0)
        # beta ~ 1 indicates universal O(1/k) convergence
        universal = "YES" if 0.7 < beta < 1.5 else "NO"
        lines.append(f"  {name:>10}  {A:8.3f}  {beta:8.3f}  "
                     f"{alpha:8.3f}  {universal:>10}")

    lines.append("")

    # Detail table for largest k
    if result.results:
        largest_k_results = {
            name: topo_results[-1]
            for name, topo_results in result.results.items()
            if topo_results
        }
        if largest_k_results:
            k_max = next(iter(largest_k_results.values())).k
            lines.append(f"Detail at k = {k_max}:")
            lines.append(f"  {'Topology':>10}  {'edges':>6}  {'|lambda_min|':>12}  "
                         f"{'gap':>10}  {'|dev|':>10}  "
                         f"{'v_min':>8}  {'v_max':>8}  {'v>0':>4}")
            lines.append(f"  {'─' * 10}  {'─' * 6}  {'─' * 12}  "
                         f"{'─' * 10}  {'─' * 10}  "
                         f"{'─' * 8}  {'─' * 8}  {'─' * 4}")
            for name in topologies:
                r = largest_k_results.get(name)
                if r:
                    lines.append(
                        f"  {r.topology:>10}  {r.n_edges:6d}  "
                        f"{abs(r.lambda_min):12.2e}  "
                        f"{r.spectral_gap:10.6f}  "
                        f"{r.deviation:10.6f}  "
                        f"{r.min_velocity:8.3f}  "
                        f"{r.max_velocity:8.3f}  "
                        f"{'Y' if r.all_velocities_positive else 'N':>4}"
                    )

    lines.append("")

    # Physics interpretation
    betas = [b for _, b in result.convergence_rates.values()]
    if betas:
        mean_beta = sum(betas) / len(betas)
        spread = max(betas) - min(betas)
        if spread < _CONVERGENCE_SPREAD_NARROW:
            lines.append(f"  Convergence exponent spread: {spread:.3f} (NARROW)")
            lines.append(f"  Mean beta = {mean_beta:.3f}")
            lines.append("  --> sigma* -> 1/2 is GRAPH-UNIVERSAL at O(1/k)")
            lines.append("  --> Prime labels (PNT) drive convergence, not topology")
        else:
            lines.append(f"  Convergence exponent spread: {spread:.3f} (WIDE)")
            lines.append(f"  Beta range: [{min(betas):.3f}, {max(betas):.3f}]")
            lines.append("  --> Convergence rate is TOPOLOGY-DEPENDENT")
            lines.append("  --> Graph structure has structural significance")

    result.summary = "\n".join(lines)
    return result
