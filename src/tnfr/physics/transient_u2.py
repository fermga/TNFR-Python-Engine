r"""Transient U2/U6 certificate for directed non-normal dynamics.

A stable spectrum (``spectral_abscissa(−L) ≤ 0``) does not, on its own, bound the
**transient** of a non-normal diffusion generator.  This module bundles the
transient reading of the two structural safety rules into a single, honest
certificate on the non-consensus subspace ``{y : πᵀ y = 0}`` (the ``L``-invariant
range of the consensus projection ``Q = I − 1 πᵀ``):

* **U2 (bounded reorganization).**  The peak amplification ``sup_s ‖e^{−sL}‖`` on
  the non-consensus subspace is certified both directly (``peak_gain``, an upper
  scan) and from the resolvent (``kreiss_lower_bound``, the Kreiss lower bound);
  ``kreiss ≤ peak`` by the Kreiss matrix theorem.  The integrated reorganization
    ``J`` and its closed bound ``M‖LQ‖‖x₀‖/ω`` come from the structural-time
    layer.
* **U6 (structural-potential confinement).** The structural potential along the
    sampled relaxation window is ``Φ_s(s) = −B L e^{−sL} Q x₀`` with the
    inverse-square aggregation ``B[i][j] = d(i,j)^{−2}``; its sampled peak is
    compared with the canonical drift limit ``π/2``. No unobserved tail is
    certified by this module.

**Honest scope.**  The certificate *reports* the transient in a declared norm; it
does **not** decide the canonical U2 metric (``NT-P09b/c`` OPEN) and does **not**
modify U2/U6 in [AGENTS.md](../../../AGENTS.md).  It is restricted to the linear
EPI channel with a scalar ``ν_f`` on a fixed graph (heterogeneous nodal ``ν_f``
requires separate analysis). No complexity / crypto / Millennium claim.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..constants.canonical import U6_STRUCTURAL_POTENTIAL_LIMIT
from .directed_diffusion import (
    consensus_projection,
    directed_rw_laplacian,
    stationary_distribution,
    sustained_gain,
    total_variation_bound,
)
from .spectral_projectors import (
    commutator_norm,
    derived_tolerance,
    matrix_exponential,
    pseudospectral_bound,
    spectral_abscissa,
)

__all__ = [
    "nonconsensus_basis",
    "restricted_generator",
    "symmetric_part_min_eig",
    "peak_transient_gain",
    "kreiss_lower_bound",
    "potential_operator",
    "potential_operator_from_graph",
    "structural_potential_peak",
    "TransientU2Certificate",
    "certify_transient_u2",
]


def nonconsensus_basis(adjacency) -> np.ndarray:
    r"""Orthonormal basis ``V`` (``n × (n−1)``) of the non-consensus subspace
    ``{y : πᵀ y = 0}`` — the ``L``-invariant complement of the consensus mode.

    ``L`` preserves this subspace (``πᵀ L = 0``), so ``V`` diagonalises the
    transient dynamics away from the conserved consensus direction.
    """
    pi = stationary_distribution(adjacency)
    n = len(pi)
    # nullspace of the 1 x n row pi^T = last n-1 right singular vectors
    _, _, vh = np.linalg.svd(pi.reshape(1, n))
    return vh[1:].T.copy()  # columns span {pi^T y = 0}, orthonormal


def restricted_generator(adjacency) -> np.ndarray:
    r"""``L_sub = Vᵀ L V`` — the diffusion generator restricted to the
    non-consensus subspace (orthonormal coordinates)."""
    laplacian = directed_rw_laplacian(adjacency)
    v = nonconsensus_basis(adjacency)
    return v.T @ laplacian @ v


def symmetric_part_min_eig(adjacency) -> float:
    r"""``λ_min½(L_sub + L_subᵀ)`` — the min eigenvalue of the symmetric part of
    the non-consensus generator.

    ``>= 0`` implies Euclidean non-consensus contraction for this generator.
    It is not universal for weighted directed graphs: the fixed six-node
    counterexample has a negative value. Stationary ``L2(pi)`` contraction is
    a separate valid statement.
    """
    lsub = restricted_generator(adjacency)
    return float(np.min(np.linalg.eigvalsh((lsub + lsub.T) / 2.0)))


def peak_transient_gain(
    adjacency, *, t_max: float = 40.0, samples: int = 400
) -> tuple[float, float]:
    r"""Finite-window ``(M_window, s*)`` on the non-consensus subspace.

    ``M_window = max_{0 <= s <= t_max} ||e^{-s L_sub}||₂``. It is a measured
    scan and must not be read as a supremum over the full trajectory.
    """
    if not np.isfinite(t_max) or t_max < 0.0:
        raise ValueError("t_max must be finite and nonnegative")
    if samples < 2:
        raise ValueError("samples must be at least 2")
    lsub = restricted_generator(adjacency)
    best, s_star = 0.0, 0.0
    for s in np.linspace(0.0, t_max, samples):
        g = float(np.linalg.norm(matrix_exponential(-lsub * s), 2))
        if g > best:
            best, s_star = g, float(s)
    return best, s_star


def kreiss_lower_bound(adjacency) -> float:
    r"""Kreiss lower bound ``sup_{Re z>0} Re(z)‖(zI + L_sub)⁻¹‖₂`` on the
    non-consensus peak gain — a resolvent certificate of transient amplification
    (``> 1`` proves it without exponentials)."""
    return pseudospectral_bound(-restricted_generator(adjacency))


def potential_operator(adjacency, *, alpha: float = 2.0) -> np.ndarray:
    r"""Auxiliary unweighted-hop potential kernel for matrix fixtures.

    This legacy helper symmetrizes adjacency support and is intentionally not
    the weighted directed graph-field implementation.

    Applied to the reorganization pressure ``ΔNFR = −L x`` it reproduces the
    canonical field ``Φ_s(i) = Σ_{j≠i} ΔNFR_j / d(i,j)^α`` (α = 2, inverse-square)
    of :func:`tnfr.physics.canonical.compute_structural_potential`.
    """
    w = np.asarray(adjacency, dtype=float)
    n = w.shape[0]
    support = (w != 0) | (w.T != 0)
    dist = np.where(support, 1.0, np.inf)
    np.fill_diagonal(dist, 0.0)
    for k in range(n):  # Floyd-Warshall (small n)
        dist = np.minimum(dist, dist[:, k, None] + dist[None, k, :])
    with np.errstate(divide="ignore"):
        b = np.where(dist > 0, dist ** (-alpha), 0.0)
    b[~np.isfinite(b)] = 0.0
    return b


def potential_operator_from_graph(
    graph, *, alpha: float = 2.0, weight: str | None = "weight",
    directed: bool | None = None,
) -> tuple[list, np.ndarray]:
    r"""Build the canonical distance kernel for a graph-owned readout.

    Distances follow outgoing arcs for directed graphs and weighted shortest
    paths when ``weight`` is supplied, matching the canonical field contract.
    """
    import networkx as nx

    nodes = list(graph)
    use_directed = graph.is_directed() if directed is None else directed
    source = graph if use_directed else graph.to_undirected()
    distances = dict(nx.all_pairs_dijkstra_path_length(source, weight=weight))
    kernel = np.zeros((len(nodes), len(nodes)), dtype=float)
    positions = {node: index for index, node in enumerate(nodes)}
    for source_node, lengths in distances.items():
        i = positions[source_node]
        for target, distance in lengths.items():
            if target == source_node or distance <= 0.0:
                continue
            kernel[i, positions[target]] = float(distance) ** (-alpha)
    return nodes, kernel


def structural_potential_peak(
    adjacency, x0, *, alpha: float = 2.0, t_max: float = 40.0, samples: int = 400
) -> tuple[float, float]:
    r"""Finite-window ``(peak, bound)`` for the structural potential.

    Both values are sampled on ``0 <= s <= t_max``. The second value is a
    finite-scan operator comparison, not a continuous-time or tail bound.
    """
    if not np.isfinite(t_max) or t_max < 0.0:
        raise ValueError("t_max must be finite and nonnegative")
    if samples < 2:
        raise ValueError("samples must be at least 2")
    laplacian = directed_rw_laplacian(adjacency)
    q = consensus_projection(adjacency)
    b = potential_operator(adjacency, alpha=alpha)
    x0v = np.asarray(x0, dtype=float)
    bl = b @ laplacian
    peak, sup_op = 0.0, 0.0
    for s in np.linspace(0.0, t_max, samples):
        prop = matrix_exponential(-laplacian * s) @ q
        phi = -bl @ (prop @ x0v)
        peak = max(peak, float(np.max(np.abs(phi))))
        sup_op = max(sup_op, float(np.linalg.norm(bl @ prop, 2)))
    return peak, sup_op * float(np.linalg.norm(x0v, 2))


@dataclass(frozen=True)
class TransientU2Certificate:
    """Transient reading of U2/U6 on the non-consensus subspace.

    ``peak_gain`` is a finite-window reading of the restricted Euclidean
    dynamics. It is not a universal contraction theorem for weighted graphs;
    weighted directed graphs can have a negative symmetric-part eigenvalue.
    The ambient ``ambient_oblique_gain > 1`` can additionally include the
    oblique consensus-projection factor ``‖Q‖``.
    """

    norm_kind: str
    consensus_projection_residual: float  # ‖LQ − L‖ ≈ 0 (Q commutes with L)
    consensus_projection_norm: float      # ‖Q‖₂ ≥ 1 (the oblique factor)
    spectral_abscissa: float              # α(−L) ≤ 0 => spectrally stable
    normality_residual: float             # ‖[L, Lᵀ]‖ = 0 => normal
    symmetric_part_min_eig: float         # sampled/derived symmetric-part readout
    peak_gain: float                      # sup_s ‖e^{−s L_sub}‖ (per-node energy)
    peak_time_structural: float           # s* achieving the peak
    kreiss_lower_bound: float             # ≤ peak_gain (Kreiss theorem)
    ambient_oblique_gain: float           # sup_s ‖e^{−sL} Q‖ (= ‖Q‖ artifact)
    integrated_reorganization: float      # J
    integrated_reorganization_bound: float
    peak_structural_potential: float      # max_{s,i} |Φ_s(s)[i]|
    structural_potential_bound: float
    u6_confined: bool                     # sampled peak Φ_s < π/2
    no_transient_amplification: bool      # measured peak_gain ≤ 1 in this window
    tolerance: float
    bounds_hold: bool
    claim_status: str
    observation_window_structural: float
    tail_status: str
    continuous_u6_status: str


def certify_transient_u2(
    adjacency, x0, *, norm_kind: str = "euclidean_nonconsensus"
) -> TransientU2Certificate:
    r"""Bundle the transient U2/U6 readings for a digraph and initial ``x₀``.

    ``bounds_hold`` is the conjunction of inequalities checked on the sampled
    windows (``kreiss ≤ peak``, ``J_window ≤ bound``, ``peak_window Φ_s ≤
    operator comparison``); it does not assert a canonical U2 decision or an
    infinite-horizon tail bound.
    """
    laplacian = directed_rw_laplacian(adjacency)
    q = consensus_projection(adjacency)
    tol = derived_tolerance(laplacian)
    proj_residual = float(np.linalg.norm(laplacian @ q - laplacian, 2))
    q_norm = float(np.linalg.norm(q, 2))
    abscissa = spectral_abscissa(-laplacian)
    normality = commutator_norm(laplacian)
    sym_min = symmetric_part_min_eig(adjacency)
    peak, s_star = peak_transient_gain(adjacency)
    kreiss = kreiss_lower_bound(adjacency)
    ambient = sustained_gain(adjacency)
    j, j_bound, j_holds = total_variation_bound(adjacency, x0)
    phi_peak, phi_bound = structural_potential_peak(adjacency, x0)
    u6_confined = phi_peak < U6_STRUCTURAL_POTENTIAL_LIMIT

    kreiss_le_peak = kreiss <= peak * (1.0 + 1e-6) + tol
    phi_le_bound = phi_peak <= phi_bound * (1.0 + 1e-6) + tol
    no_amp = peak <= 1.0 + max(tol, 1e-6)
    bounds_hold = kreiss_le_peak and j_holds and phi_le_bound

    status = (
        "MEASURED (linear EPI channel, scalar nu_f): restricted Euclidean "
        f"peak={peak:.6g} on the sampled window; ambient projection gain is "
        "reported separately. Universal weighted-graph Euclidean contraction "
        "is refuted by a fixed exact counterexample; stationary L2(pi) "
        "contraction and canonical U2 metric remain OPEN and separately "
        "scoped; "
        "U2/U6 unmodified"
    )
    return TransientU2Certificate(
        norm_kind=norm_kind,
        consensus_projection_residual=proj_residual,
        consensus_projection_norm=q_norm,
        spectral_abscissa=abscissa,
        normality_residual=normality,
        symmetric_part_min_eig=sym_min,
        peak_gain=peak,
        peak_time_structural=s_star,
        kreiss_lower_bound=kreiss,
        ambient_oblique_gain=ambient,
        integrated_reorganization=j,
        integrated_reorganization_bound=j_bound,
        peak_structural_potential=phi_peak,
        structural_potential_bound=phi_bound,
        u6_confined=u6_confined,
        no_transient_amplification=no_amp,
        tolerance=tol,
        bounds_hold=bounds_hold,
        claim_status=status,
        observation_window_structural=40.0,
        tail_status="UNASSESSED_FINITE_WINDOW",
        continuous_u6_status="INCONCLUSIVE_NO_INTERVAL_OR_TAIL_BOUND",
    )
