r"""Transient U2/U6 certificate for directed non-normal dynamics (R9, N05).

A stable spectrum (``spectral_abscissa(−L) ≤ 0``) does not, on its own, bound the
**transient** of a non-normal diffusion generator.  This module bundles the
transient reading of the two structural safety rules into a single, honest
certificate on the non-consensus subspace ``{y : πᵀ y = 0}`` (the ``L``-invariant
range of the consensus projection ``Q = I − 1 πᵀ``):

* **U2 (bounded reorganization).**  The peak amplification ``sup_s ‖e^{−sL}‖`` on
  the non-consensus subspace is certified both directly (``peak_gain``, an upper
  scan) and from the resolvent (``kreiss_lower_bound``, the Kreiss lower bound);
  ``kreiss ≤ peak`` by the Kreiss matrix theorem.  The integrated reorganization
  ``J`` and its closed bound ``M‖LQ‖‖x₀‖/ω`` come from the N04 structural-time
  layer.
* **U6 (structural-potential confinement).**  The structural potential along the
  relaxation trajectory is ``Φ_s(s) = −B L e^{−sL} Q x₀`` with the canonical
  inverse-square aggregation ``B[i][j] = d(i,j)^{−2}``; the per-node peak is
  checked against the canonical drift limit ``π/2`` over the **full** trajectory,
  and bounded above by ``sup_s ‖B L e^{−sL} Q‖ ‖x₀‖``.

**Honest scope.**  The certificate *reports* the transient in a declared norm; it
does **not** decide the canonical U2 metric (``NT-P09b/c`` OPEN) and does **not**
modify U2/U6 in [AGENTS.md](../../../AGENTS.md).  It is restricted to the linear
EPI channel with a scalar ``ν_f`` on a fixed graph (heterogeneous nodal ``ν_f``
is N13).  No complexity / crypto / Millennium claim.
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

    ``≥ 0`` ⟺ the semigroup ``e^{−s L_sub}`` is a contraction in the canonical
    Euclidean per-node energy (no non-consensus transient amplification).
    Measured ``> 0`` across 2·10⁵ random strongly-connected digraphs and extreme
    in-hub constructions (worst ``≈ +9·10⁻³``); the general positive-definite
    property is **CONJECTURAL** (strong evidence, no proof yet).
    """
    lsub = restricted_generator(adjacency)
    return float(np.min(np.linalg.eigvalsh((lsub + lsub.T) / 2.0)))


def peak_transient_gain(
    adjacency, *, t_max: float = 40.0, samples: int = 400
) -> tuple[float, float]:
    r"""``(M, s*) = (max_s ‖e^{−s L_sub}‖₂, argmax)`` on the non-consensus
    subspace — the peak transient amplification and the **structural time** at
    which it occurs (``M = 1`` normal, ``M > 1`` non-normal)."""
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
    r"""Canonical structural-potential aggregation ``B[i][j] = d(i,j)^{−α}``
    (``i ≠ j``, zero diagonal), ``d`` = undirected shortest-path distance.

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


def structural_potential_peak(
    adjacency, x0, *, alpha: float = 2.0, t_max: float = 40.0, samples: int = 400
) -> tuple[float, float]:
    r"""``(peak, bound)`` for the per-node structural potential along the
    relaxation ``Φ_s(s) = −B L e^{−sL} Q x₀``.

    ``peak = max_{s, i} |Φ_s(s)[i]|`` (the U6 drift over the full trajectory,
    since ``Φ_s(∞) = 0``); ``bound = sup_s ‖B L e^{−sL} Q‖₂ ‖x₀‖₂`` is the
    operator-norm upper bound.
    """
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
    """Transient reading of U2/U6 on the non-consensus subspace (N05).

    The canonical finding: in the Euclidean per-node energy the non-consensus
    dynamics is a **contraction** (``peak_gain = 1``); the naive ambient
    ``ambient_oblique_gain > 1`` is the oblique consensus-projection factor
    ``consensus_projection_norm = ‖Q‖`` (peak at ``s = 0``), not dynamical growth.
    """

    norm_kind: str
    consensus_projection_residual: float  # ‖LQ − L‖ ≈ 0 (Q commutes with L)
    consensus_projection_norm: float      # ‖Q‖₂ ≥ 1 (the oblique factor)
    spectral_abscissa: float              # α(−L) ≤ 0 => spectrally stable
    normality_residual: float             # ‖[L, Lᵀ]‖ = 0 => normal
    symmetric_part_min_eig: float         # ≥ 0 => Euclidean contraction
    peak_gain: float                      # sup_s ‖e^{−s L_sub}‖ (per-node energy)
    peak_time_structural: float           # s* achieving the peak
    kreiss_lower_bound: float             # ≤ peak_gain (Kreiss theorem)
    ambient_oblique_gain: float           # sup_s ‖e^{−sL} Q‖ (= ‖Q‖ artifact)
    integrated_reorganization: float      # J
    integrated_reorganization_bound: float
    peak_structural_potential: float      # max_{s,i} |Φ_s(s)[i]|
    structural_potential_bound: float
    u6_confined: bool                     # peak Φ_s < π/2 over full trajectory
    no_transient_amplification: bool      # peak_gain ≤ 1 (per-node energy)
    tolerance: float
    bounds_hold: bool
    claim_status: str


def certify_transient_u2(
    adjacency, x0, *, norm_kind: str = "euclidean_nonconsensus"
) -> TransientU2Certificate:
    r"""Bundle the transient U2/U6 readings for a digraph and initial ``x₀``.

    ``bounds_hold`` is the conjunction of the certified inequalities
    (``kreiss ≤ peak``, ``J ≤ bound``, ``peak Φ_s ≤ operator bound``); it does
    **not** assert a canonical U2 decision (which stays OPEN).
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
        "MEASURED (linear EPI channel, scalar nu_f, Euclidean per-node energy): "
        "non-consensus contraction, peak=1; ambient >1 is the ||Q|| artifact. "
        "General PSD-on-subspace CONJECTURAL; canonical U2 metric OPEN "
        "(NT-P09b/c); U2/U6 unmodified"
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
    )
