r"""Directed non-normal structural dynamics (R9).

The canonical structural-diffusion operator ``L_rw = I − D_out⁻¹ W`` is symmetric
(hence **normal**) only for undirected or vertex-transitive graphs.  A general
**directed** graph makes ``L_rw`` non-normal, and then the naive spectral reading
breaks down:

* ``eigh`` is invalid (the operator is not symmetric);
* a stable spectrum (``spectral_abscissa(−L) ≤ 0``) does **not** exclude
  **transient amplification** of ``‖e^{−tL}‖``;
* right eigenvectors are not orthogonal, so ``Q Qᴴ`` is not the spectral
  projector — the correct object is the Schur / Riesz form.

Directed **circulant** Cayley graphs (the residue digraphs of R2) are a benign
special case: they are normal, so their transient gain is ``1`` and the R2/C4
spectral machinery applies unchanged.  This module separates the two regimes and
certifies the non-normal one with numpy-only measures (transient gain,
pseudospectral / Kreiss bound), gating the Schur residual on SciPy.

**U2 scope (honest).**  The reading ``r_c = ν_f λ₂`` describes **asymptotic**
relaxation only.  For non-normal operators a positive transient gain means
``C(t)`` can dip before it relaxes; the U2 integral convergence still holds
asymptotically, but no generalized U2 bound is claimed until the transient
contribution is derived (``NT-P09`` OPEN).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from .spectral_projectors import (
    commutator_norm,
    derived_tolerance,
    is_normal,
    matrix_exponential,
    pseudospectral_bound,
    schur_residual,
    scipy_available,
    spectral_abscissa,
    transient_gain,
)

__all__ = [
    "directed_rw_laplacian",
    "directed_cayley_adjacency",
    "is_directed_circulant",
    "DirectedDynamicsCertificate",
    "certify_directed_dynamics",
    "NormKind",
    "stationary_distribution",
    "state_norm",
    "induced_operator_norm",
    "transient_gain_in_norm",
    "stationary_transient_gain",
    "is_stationary_contraction",
    "net_reorganization",
    "total_reorganization",
    "U2IntegralReadings",
    "u2_integral_readings",
]


def directed_rw_laplacian(adjacency) -> np.ndarray:
    r"""Directed random-walk Laplacian ``L_rw = I − D_out⁻¹ W`` (OUTGOING
    convention, C1).

    ``W[i][j] = 1`` means an edge ``i → j``; the row is normalised by the
    out-degree ``D_out[i]``.  A sink (out-degree 0) contributes a zero row (it
    neither relaxes nor drives), which keeps ``L_rw`` well defined.
    """
    w = np.asarray(adjacency, dtype=float)
    n = w.shape[0]
    out_degree = w.sum(axis=1)
    inv = np.divide(
        1.0, out_degree, out=np.zeros_like(out_degree), where=out_degree > 0
    )
    transition = (w.T * inv).T  # D_out^{-1} W
    laplacian = np.eye(n) - transition
    # zero-out sink rows so isolated sinks do not spuriously self-relax
    laplacian[out_degree == 0] = 0.0
    return laplacian


def directed_cayley_adjacency(n: int, connection) -> np.ndarray:
    r"""Directed circulant Cayley adjacency on ``ℤ/nℤ`` (edge ``i → i + s``).

    A circulant is **normal**, so its directed diffusion has no transient growth
    — the benign special case (R2 residue digraphs).
    """
    conn = {int(s) % n for s in connection}
    w = np.zeros((n, n), dtype=float)
    for i in range(n):
        for s in conn:
            w[i][(i + s) % n] = 1.0
    return w


def is_directed_circulant(adjacency, *, tol: float | None = None) -> bool:
    r"""Whether the adjacency is circulant (each row a cyclic shift of row 0)."""
    w = np.asarray(adjacency, dtype=float)
    n = w.shape[0]
    if tol is None:
        tol = derived_tolerance(w)
    first = w[0]
    for i in range(1, n):
        if not np.allclose(w[i], np.roll(first, i), atol=max(tol, 1e-12)):
            return False
    return True


@dataclass(frozen=True)
class DirectedDynamicsCertificate:
    """Spectral / transient certificate for a directed diffusion generator ``−L``.

    Attributes
    ----------
    normal:
        Whether ``L`` is normal (``L Lᴴ = Lᴴ L``).
    commutator:
        ``‖L Lᴴ − Lᴴ L‖₂`` (zero iff normal).
    abscissa:
        Spectral abscissa ``α(−L) = max Re λ(−L)``; ``≤ 0`` ⇒ asymptotically
        stable.
    asymptotically_stable:
        ``abscissa ≤ tol``.
    transient_gain:
        Peak ``‖e^{−tL}‖₂``; ``> 1`` ⇒ transient amplification.
    has_transient_amplification:
        ``transient_gain > 1 + tol``.
    pseudospectral_bound:
        Kreiss lower bound on the transient gain from the resolvent.
    schur_residual:
        ``‖L − Q T Qᴴ‖₂`` if SciPy is available, else ``None`` (gated).
    """

    normal: bool
    commutator: float
    abscissa: float
    asymptotically_stable: bool
    transient_gain: float
    has_transient_amplification: bool
    pseudospectral_bound: float
    schur_residual: float | None


def certify_directed_dynamics(
    adjacency, *, t_max: float = 10.0, samples: int = 200
) -> DirectedDynamicsCertificate:
    r"""Certify the directed diffusion generator ``−L_rw`` of a digraph.

    Measures normality, asymptotic stability (spectral abscissa), and transient
    behaviour (gain + pseudospectral bound) — the transient measures are what a
    stable spectrum alone cannot provide for a non-normal operator.
    """
    laplacian = directed_rw_laplacian(adjacency)
    generator = -laplacian  # the diffusion flow ∂x/∂t = −L x
    tol = derived_tolerance(laplacian)
    commutator = commutator_norm(laplacian)
    normal = is_normal(laplacian, tol=tol)
    abscissa = spectral_abscissa(generator)
    gain = transient_gain(generator, t_max=t_max, samples=samples)
    kreiss = pseudospectral_bound(generator)
    schur = schur_residual(laplacian) if scipy_available() else None
    return DirectedDynamicsCertificate(
        normal=normal,
        commutator=commutator,
        abscissa=abscissa,
        asymptotically_stable=abscissa <= tol,
        transient_gain=gain,
        has_transient_amplification=gain > 1.0 + tol,
        pseudospectral_bound=kreiss,
        schur_residual=schur,
    )


# ════════════════════════════════════════════════════════════════════════════
# N03 — metric layer and U2 integral semantics
# The Euclidean transient gain of a non-normal generator can exceed 1 (R9). But
# the physical reading of U2 depends on the NORM, and the report shows the growth
# may be a metric artefact: in the stationary-weighted norm L²(π) the diffusion
# semigroup is a contraction (Jensen), so its transient gain is ≤ 1. This layer
# exposes both norms and both integral readings WITHOUT deciding U2 — that
# decision (which norm is canonical) stays OPEN until the gate in AGENTS.md §6 is
# met.
# ════════════════════════════════════════════════════════════════════════════
class NormKind(Enum):
    """The norm in which a state / operator gain is measured.

    ``EUCLIDEAN`` is the unweighted per-node ℓ² norm (the canonical TNFR
    per-node structural energy corresponds to this unweighted sum).
    ``STATIONARY`` is the ``L²(π)`` norm weighted by the stationary distribution
    ``π`` of the random walk, in which the diffusion semigroup is a contraction.
    """

    EUCLIDEAN = "euclidean"
    STATIONARY = "stationary"


def stationary_distribution(adjacency, *, tol: float | None = None) -> np.ndarray:
    r"""Stationary distribution ``π`` of the random walk ``P = D_out⁻¹ W``.

    ``π`` is the left Perron eigenvector (``πᵀ P = πᵀ``, ``π ≥ 0``, ``Σ π = 1``).
    Raises ``ValueError`` unless ``π`` is strictly positive (the graph must be
    strongly connected for the ``L²(π)`` norm to be non-degenerate).
    """
    w = np.asarray(adjacency, dtype=float)
    n = w.shape[0]
    out_degree = w.sum(axis=1)
    inv = np.divide(
        1.0, out_degree, out=np.zeros_like(out_degree), where=out_degree > 0
    )
    transition = (w.T * inv).T
    vals, vecs = np.linalg.eig(transition.T)
    idx = int(np.argmin(np.abs(vals - 1.0)))
    pi = np.real(vecs[:, idx])
    total = pi.sum()
    if abs(total) < np.finfo(float).eps:
        raise ValueError("degenerate stationary vector (graph not irreducible)")
    pi = pi / total
    if tol is None:
        tol = derived_tolerance(transition)
    if np.min(pi) <= tol:
        raise ValueError(
            "stationary distribution is not strictly positive; the L²(π) norm "
            "requires a strongly connected graph"
        )
    return pi


def state_norm(
    vec, *, kind: NormKind = NormKind.EUCLIDEAN, pi: np.ndarray | None = None
) -> float:
    r"""Norm of a state vector in the chosen metric.

    ``EUCLIDEAN``: ``‖v‖₂``.  ``STATIONARY``: ``‖v‖_{2,π} = √(Σ π_i v_i²)``.
    """
    v = np.asarray(vec, dtype=float)
    if kind is NormKind.EUCLIDEAN:
        return float(np.linalg.norm(v, 2))
    if pi is None:
        raise ValueError("the stationary norm requires the distribution pi")
    p = np.asarray(pi, dtype=float)
    return float(np.sqrt(np.sum(p * v * v)))


def induced_operator_norm(
    matrix, *, kind: NormKind = NormKind.EUCLIDEAN, pi: np.ndarray | None = None
) -> float:
    r"""Operator norm induced by the chosen state norm.

    ``EUCLIDEAN``: spectral norm ``‖M‖₂``.  ``STATIONARY``:
    ``‖M‖_{2,π} = ‖S M S⁻¹‖₂`` with ``S = diag(√π)`` (the similarity that turns
    the π-weighted inner product into the standard one).
    """
    m = np.asarray(matrix, dtype=float)
    if kind is NormKind.EUCLIDEAN:
        return float(np.linalg.norm(m, 2))
    if pi is None:
        raise ValueError("the stationary norm requires the distribution pi")
    s = np.sqrt(np.asarray(pi, dtype=float))
    return float(np.linalg.norm((s[:, None] * m) / s[None, :], 2))


def transient_gain_in_norm(
    generator,
    *,
    kind: NormKind = NormKind.EUCLIDEAN,
    pi: np.ndarray | None = None,
    t_max: float = 12.0,
    samples: int = 180,
) -> float:
    r"""``max_{t∈[0, t_max]} ‖e^{tA}‖`` in the chosen norm.

    In the ``EUCLIDEAN`` norm a non-normal stable ``A`` can give ``> 1``; in the
    ``STATIONARY`` norm the diffusion semigroup is a contraction so it is ``≤ 1``.
    """
    a = np.asarray(generator, dtype=float)
    gain = 0.0
    for t in np.linspace(0.0, t_max, samples):
        gain = max(
            gain,
            induced_operator_norm(
                matrix_exponential(a * t), kind=kind, pi=pi
            ),
        )
    return gain


def stationary_transient_gain(
    adjacency, *, t_max: float = 12.0, samples: int = 180
) -> float:
    r"""Peak ``‖e^{−tL}‖_{2,π}`` of the diffusion semigroup (should be ``≤ 1``)."""
    pi = stationary_distribution(adjacency)
    generator = -directed_rw_laplacian(adjacency)
    return transient_gain_in_norm(
        generator, kind=NormKind.STATIONARY, pi=pi, t_max=t_max, samples=samples
    )


def is_stationary_contraction(
    adjacency, *, t_max: float = 12.0, samples: int = 180
) -> bool:
    r"""Whether the diffusion semigroup contracts in ``L²(π)`` (gain ``≤ 1``).

    True for every (strongly connected) directed graph — the Euclidean transient
    amplification does not appear in the stationary-weighted energy.
    """
    pi = stationary_distribution(adjacency)
    generator = -directed_rw_laplacian(adjacency)
    tol = derived_tolerance(generator)
    gain = transient_gain_in_norm(
        generator, kind=NormKind.STATIONARY, pi=pi, t_max=t_max, samples=samples
    )
    return gain <= 1.0 + tol


def net_reorganization(
    generator, x0, *, kind: NormKind = NormKind.EUCLIDEAN,
    pi: np.ndarray | None = None, t_infinity: float = 60.0,
) -> float:
    r"""Net displacement ``‖x(∞) − x(0)‖`` — the norm of the **signed** integral
    ``∫₀^∞ ẋ dt = x(∞) − x(0)`` (reorganizations may cancel)."""
    a = np.asarray(generator, dtype=float)
    x0 = np.asarray(x0, dtype=float)
    x_inf = matrix_exponential(a * t_infinity) @ x0
    return state_norm(x_inf - x0, kind=kind, pi=pi)


def total_reorganization(
    generator, x0, *, kind: NormKind = NormKind.EUCLIDEAN,
    pi: np.ndarray | None = None, t_max: float = 60.0, samples: int = 400,
) -> float:
    r"""Total structural variation ``∫₀^∞ ‖ẋ(s)‖ ds = ∫ ‖A e^{sA} x₀‖ ds`` — the
    accumulated reorganization with **no cancellation** (``≥`` net)."""
    a = np.asarray(generator, dtype=float)
    x0 = np.asarray(x0, dtype=float)
    ts = np.linspace(0.0, t_max, samples)
    speeds = [
        state_norm(a @ (matrix_exponential(a * t) @ x0), kind=kind, pi=pi)
        for t in ts
    ]
    return float(np.trapezoid(speeds, ts))


@dataclass(frozen=True)
class U2IntegralReadings:
    """The two candidate meanings of the U2 convergence integral.

    ``net`` is the signed integral ``‖x(∞) − x(0)‖`` (cancellation allowed);
    ``total`` is the total structural variation ``∫ ‖ẋ‖`` (no cancellation).
    Which one U2 canonically means is **OPEN** (report §16.2); ``net ≤ total``
    always.
    """

    norm_kind: str
    net: float
    total: float

    @property
    def net_le_total(self) -> bool:
        return self.net <= self.total + 1e-9


def u2_integral_readings(
    adjacency, x0, *, kind: NormKind = NormKind.EUCLIDEAN,
    t_max: float = 60.0, samples: int = 400,
) -> U2IntegralReadings:
    r"""Both U2 integral readings for the diffusion of ``x0`` on the digraph.

    Exposes signed displacement vs total variation without deciding which is the
    canonical U2 quantity (the decision is gated in AGENTS.md §6).
    """
    generator = -directed_rw_laplacian(adjacency)
    pi = (
        stationary_distribution(adjacency)
        if kind is NormKind.STATIONARY else None
    )
    net = net_reorganization(
        generator, x0, kind=kind, pi=pi, t_infinity=t_max
    )
    total = total_reorganization(
        generator, x0, kind=kind, pi=pi, t_max=t_max, samples=samples
    )
    return U2IntegralReadings(kind.value, net, total)
