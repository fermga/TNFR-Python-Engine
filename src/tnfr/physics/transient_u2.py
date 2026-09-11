r"""Transient U2 and structural-potential diagnostics for directed dynamics.

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
* **Structural-potential magnitude.** The structural potential along the
  sampled relaxation window is ``Φ_s(s) = −B L e^{−sL} Q x₀`` with the
  inverse-square aggregation ``B[i][j] = d(i,j)^{−2}``. Its sampled absolute
  peak and an operator-norm bound are reported. The optional comparison with
  ``π/2`` is magnitude metadata only: canonical U6 compares potential *drift*
  between two declared graph-state snapshots, which this routine does not have.

**Honest scope.**  The certificate *reports* the transient in a declared norm; it
does **not** decide the canonical U2 metric (``NT-P09b/c`` OPEN) and does **not**
modify U2 or U6 in [AGENTS.md](../../../AGENTS.md). It is restricted to the linear
EPI channel with a scalar ``ν_f`` on a fixed graph (heterogeneous nodal ``ν_f``
requires separate analysis). No complexity / crypto / Millennium claim.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral, Real

import numpy as np

from ..constants.canonical import U6_STRUCTURAL_POTENTIAL_LIMIT
from ._edge_semantics import structural_path_weight
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


def _positive_alpha(alpha: float) -> float:
    """Return a finite positive distance exponent."""
    if isinstance(alpha, bool) or not isinstance(alpha, Real):
        raise ValueError("alpha must be a finite positive real number")
    try:
        value = float(alpha)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("alpha must be a finite positive real number") from exc
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("alpha must be a finite positive real number")
    return value


def _scan_parameters(t_max: float, samples: int) -> tuple[float, int]:
    """Validate one finite sampling window without boolean coercions."""
    if isinstance(t_max, bool) or not isinstance(t_max, Real):
        raise ValueError("t_max must be finite and nonnegative")
    try:
        window = float(t_max)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("t_max must be finite and nonnegative") from exc
    if not math.isfinite(window) or window < 0.0:
        raise ValueError("t_max must be finite and nonnegative")
    if (
        isinstance(samples, bool)
        or not isinstance(samples, Integral)
        or samples < 2
    ):
        raise ValueError("samples must be an integer of at least 2")
    return window, int(samples)


def _state_vector(x0, size: int) -> np.ndarray:
    """Return one finite state vector aligned with the adjacency order."""
    try:
        vector = np.asarray(x0, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("x0 must be a finite vector aligned with adjacency") from exc
    if vector.shape != (size,) or not np.all(np.isfinite(vector)):
        raise ValueError("x0 must be a finite vector aligned with adjacency")
    return vector


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
    if lsub.size == 0:
        return 0.0
    return float(np.min(np.linalg.eigvalsh((lsub + lsub.T) / 2.0)))


def peak_transient_gain(
    adjacency, *, t_max: float = 40.0, samples: int = 400
) -> tuple[float, float]:
    r"""Finite-window ``(M_window, s*)`` on the non-consensus subspace.

    ``M_window = max_{0 <= s <= t_max} ||e^{-s L_sub}||₂``. It is a measured
    scan and must not be read as a supremum over the full trajectory.
    """
    t_max, samples = _scan_parameters(t_max, samples)
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
    restricted = restricted_generator(adjacency)
    if restricted.size == 0:
        return 0.0
    return pseudospectral_bound(-restricted)


def potential_operator(adjacency, *, alpha: float = 2.0) -> np.ndarray:
    r"""Unweighted-hop potential kernel for adjacency-matrix fixtures.

    Non-symmetric support follows outgoing arcs, matching the directed graph
    field convention. Edge magnitudes remain transport conductances and do not
    become path lengths in this matrix-only helper.

    Applied to the reorganization pressure ``ΔNFR = −L x`` it reproduces the
    canonical field ``Φ_s(i) = Σ_{j≠i} ΔNFR_j / d(i,j)^α`` (α = 2, inverse-square)
    of :func:`tnfr.physics.canonical.compute_structural_potential`.
    """
    alpha = _positive_alpha(alpha)
    # Reuse the diffusion validator so malformed or signed conductance cannot
    # silently become a different undirected support graph.
    directed_rw_laplacian(adjacency)
    w = np.asarray(adjacency, dtype=float)
    n = w.shape[0]
    support = w != 0
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

    Distances follow outgoing arcs for directed graphs.  The default public
    ``weight='weight'`` spelling now resolves through the shared structural
    path channel: explicit ``length`` first, legacy ``weight`` second, then
    unit length. Passing ``None`` or a different attribute remains an explicit
    NetworkX distance override.
    """
    import networkx as nx

    alpha = _positive_alpha(alpha)
    nodes = list(graph)
    use_directed = graph.is_directed() if directed is None else directed
    source = graph if use_directed else graph.to_undirected()
    path_weight = structural_path_weight(source) if weight == "weight" else weight
    distances = dict(
        nx.all_pairs_dijkstra_path_length(source, weight=path_weight)
    )
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
    alpha = _positive_alpha(alpha)
    t_max, samples = _scan_parameters(t_max, samples)
    laplacian = directed_rw_laplacian(adjacency)
    q = consensus_projection(adjacency)
    b = potential_operator(adjacency, alpha=alpha)
    x0v = _state_vector(x0, laplacian.shape[0])
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
    """Transient U2 reading plus potential-magnitude diagnostics.

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
    peak_structural_potential: float  # max_{s,i} |Φ_s(s)[i]|
    structural_potential_bound: float
    u6_confined: bool
    no_transient_amplification: bool      # measured peak_gain ≤ 1 in this window
    tolerance: float
    bounds_hold: bool
    claim_status: str
    observation_window_structural: float
    tail_status: str
    continuous_u6_status: str
    u6_drift_assessed: bool

    @property
    def peak_structural_potential_magnitude(self) -> float:
        """Honest name for the legacy sampled-potential field."""
        return self.peak_structural_potential

    @property
    def structural_potential_magnitude_bound(self) -> float:
        """Honest name for the legacy finite-window comparison field."""
        return self.structural_potential_bound

    @property
    def potential_magnitude_below_pi_scale(self) -> bool:
        """Honest name for the legacy field; this is not a U6 verdict."""
        return self.u6_confined


def certify_transient_u2(
    adjacency, x0, *, norm_kind: str = "euclidean_nonconsensus"
) -> TransientU2Certificate:
    r"""Bundle transient U2 and potential readings for a digraph and ``x₀``.

    ``bounds_hold`` is the conjunction of inequalities checked on the sampled
    windows (``kreiss ≤ peak``, ``J_window ≤ bound``, ``peak_window Φ_s ≤
    operator comparison``); it does not assert a canonical U2 decision or an
    infinite-horizon tail bound.
    """
    if norm_kind != "euclidean_nonconsensus":
        raise ValueError(
            "norm_kind must be 'euclidean_nonconsensus'; other metrics are "
            "not implemented"
        )
    laplacian = directed_rw_laplacian(adjacency)
    x0v = _state_vector(x0, laplacian.shape[0])
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
    if laplacian.shape == (1, 1):
        j, j_bound, j_holds = 0.0, 0.0, True
    else:
        j, j_bound, j_holds = total_variation_bound(adjacency, x0v)
    phi_peak, phi_bound = structural_potential_peak(adjacency, x0v)
    magnitude_below_pi_scale = phi_peak < U6_STRUCTURAL_POTENTIAL_LIMIT

    kreiss_le_peak = kreiss <= peak * (1.0 + 1e-6) + tol
    phi_le_bound = phi_peak <= phi_bound * (1.0 + 1e-6) + tol
    no_amp = peak <= 1.0 + max(tol, 1e-6)
    bounds_hold = kreiss_le_peak and j_holds and phi_le_bound

    status = (
        "MEASURED (linear EPI channel, scalar nu_f): restricted Euclidean "
        f"peak={peak:.6g} on the sampled window; ambient projection gain is "
        "reported separately. Universal weighted-graph Euclidean contraction "
        "is refuted by a fixed exact counterexample; stationary L2(pi) "
        "contraction remains separately scoped; canonical U2 metric OPEN; "
        "U6 drift not assessed"
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
        u6_confined=magnitude_below_pi_scale,
        no_transient_amplification=no_amp,
        tolerance=tol,
        bounds_hold=bounds_hold,
        claim_status=status,
        observation_window_structural=40.0,
        tail_status="UNASSESSED_FINITE_WINDOW",
        continuous_u6_status="NOT_ASSESSED_NO_REFERENCE_STATE_TRAJECTORY",
        u6_drift_assessed=False,
    )
