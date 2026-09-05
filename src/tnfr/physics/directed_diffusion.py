r"""Directed non-normal structural dynamics (R9).

The structural-diffusion operator ``L_rw = I − D_out⁻¹ W`` uses outgoing
conductance. Even for an undirected graph its matrix need not be symmetric:
on positive-strength vertices, symmetric W makes it similar through D^(1/2)
to the symmetric normalized Laplacian. Raw Euclidean normality is a separate
property. On a non-normal operator the naive orthogonal spectral reading
does not apply:

* ``eigh`` requires a symmetric/Hermitian representation;
* a stable spectrum (``spectral_abscissa(−L) ≤ 0``) does **not** exclude
  **transient amplification** of ``‖e^{−tL}‖``;
* right eigenvectors are not orthogonal, so ``Q Qᴴ`` is not the spectral
  projector — the correct object is the Schur / Riesz form.

Directed **circulant** Cayley graphs (the residue digraphs of R2) are a benign
special case: they are normal, so their transient gain is ``1`` and the R2/C4
spectral machinery applies unchanged.  This module separates the two regimes and
certifies the non-normal one with numpy-only measures (transient gain,
pseudospectral / Kreiss bound), gating the Schur residual on SciPy.

**U2 scope.** These matrix diagnostics do not prove grammar sufficiency or
convergence for arbitrary pressure laws. Semigroup norms depend on the chosen
metric, and a diffusion generator retains stationary modes. Its nonpositive
spectral abscissa does not imply decay of the full state to zero. The canonical
U2 metric and generalized bound remain open.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable

import numpy as np

from ..mathematics._weight_normalization import normalize_weights
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
    "consensus_projection",
    "nonconsensus_abscissa",
    "sustained_gain",
    "structural_time",
    "clock_change_residual",
    "reorganization_time_invariance_residual",
    "total_variation_bound",
    "StructuralTimeCertificate",
    "certify_structural_time",
]


def _row_normalized_transition(adjacency) -> np.ndarray:
    """Validated outgoing transition with absorbing zero-strength rows.

    Divide rows directly instead of forming reciprocal strengths, which can
    overflow for subnormal conductance. Scaling each nonzero row by its largest
    entry first also avoids overflow in a sum of finite conductances. Returned
    storage is independent of the input matrix.
    """
    try:
        if np.iscomplexobj(adjacency):
            raise ValueError("Adjacency must contain real conductance")
        weights = np.asarray(adjacency, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Adjacency must be a square real matrix") from exc
    if weights.ndim != 2 or weights.shape[0] != weights.shape[1]:
        raise ValueError("Adjacency must be a square real matrix")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("Adjacency requires finite nonnegative conductance")

    transition, row_scale, _ = normalize_weights(weights)
    sinks = np.flatnonzero(row_scale == 0.0)
    transition[sinks, sinks] = 1.0
    return transition


def directed_rw_laplacian(adjacency) -> np.ndarray:
    r"""Outgoing random-walk Laplacian ``L_rw = I − P`` (convention C1).

    Input is a square matrix of finite nonnegative conductance. Positive rows
    use ``P_ij = W_ij / sum_j W_ij``. Zero-strength rows are absorbing in P and
    zero in L: the sink's own state is fixed, but it can still influence nodes
    with arcs pointing to it. Self-loops contribute once to row strength.
    """
    transition = _row_normalized_transition(adjacency)
    return np.eye(len(transition)) - transition


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
        Spectral abscissa ``α(−L) = max Re λ(−L)``. Diffusion retains stationary
        modes; a nonpositive value does not imply full-state decay to zero.
    asymptotically_stable:
        Legacy flag for ``abscissa ≤ tol`` (no positive spectral growth),
        not a claim that the stationary subspace decays.
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

    Measures normality, spectral growth (abscissa), and transient
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
    r"""Strictly positive stationary ``π`` for the outgoing walk ``P = I−L``.

    Uses the same validated transition as :func:`directed_rw_laplacian`,
    including absorbing zero-strength rows. Verifies ``π P = π`` and
    ``sum(π)=1``. The stationary-norm API retains its strict positivity gate;
    it does not return an arbitrary sink-supported measure of a reducible
    graph. A strongly connected finite walk has a unique positive stationary
    distribution; the one-node absorbing walk is also supported.
    """
    transition = _row_normalized_transition(adjacency)
    if not len(transition):
        raise ValueError("A stationary distribution requires at least one node")
    if tol is None:
        tol = derived_tolerance(transition)
    if not np.isfinite(tol) or tol < 0.0:
        raise ValueError("Stationary tolerance must be finite and nonnegative")
    vals, vecs = np.linalg.eig(transition.T)
    idx = int(np.argmin(np.abs(vals - 1.0)))
    pi = np.real(vecs[:, idx])
    total = pi.sum()
    if abs(total) < np.finfo(float).eps:
        raise ValueError("degenerate stationary vector (graph not irreducible)")
    pi = pi / total
    if np.min(pi) <= tol:
        raise ValueError(
            "stationary distribution is not strictly positive; the L²(π) norm "
            "requires positive stationary weights"
        )
    residual_tolerance = max(tol, len(transition) * np.finfo(float).eps)
    if (
        not np.all(np.isfinite(pi))
        or abs(vals[idx] - 1.0) > residual_tolerance
        or np.max(np.abs(pi @ transition - pi)) > residual_tolerance
        or abs(float(pi.sum()) - 1.0) > residual_tolerance
    ):
        raise ValueError("Stationary eigenvector does not satisfy the walk balance")
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


# ════════════════════════════════════════════════════════════════════════════
# N04 — structural-time theorem (scalar common frequency)
# For a scalar frequency ν_f(t) ≥ 0 the linear EPI transport ẋ = −ν_f(t) L x has
# the exact solution x(t) = e^{−s(t) L} x₀ with s(t) = ∫₀^t ν_f (all ν_f(τ) L
# commute, L constant). ν_f is a CLOCK CHANGE (mobility), not a mass: it rescales
# the speed along a FIXED state-space trajectory. On the non-consensus subspace
# Q = I − 1πᵀ the semigroup decays (‖e^{−sL}Q‖ ≤ M e^{−ωs}), so the total
# reorganization is finite: J ≤ M‖LQ‖‖x₀‖/ω. Linear EPI channel, fixed graph.
# ════════════════════════════════════════════════════════════════════════════
def consensus_projection(adjacency) -> np.ndarray:
    r"""Projection ``Q = I − 1 πᵀ`` out of the stationary consensus mode.

    ``Q`` annihilates the consensus (``Q·1 = 0``, ``πᵀQ = 0``), is idempotent,
    and commutes with ``L`` (``L·1 = 0`` and ``πᵀL = 0`` give ``LQ = QL = L``).
    """
    pi = stationary_distribution(adjacency)
    n = len(pi)
    return np.eye(n) - np.outer(np.ones(n), pi)


def nonconsensus_abscissa(adjacency, *, tol: float | None = None) -> float:
    r"""``ω = min_{λ≠0} Re λ(L)`` — the relaxation rate on the non-consensus
    subspace (the spectral gap of the diffusion generator)."""
    laplacian = directed_rw_laplacian(adjacency)
    if tol is None:
        tol = derived_tolerance(laplacian)
    eig = np.linalg.eigvals(laplacian)
    nonzero = eig[np.abs(eig) > tol]
    if nonzero.size == 0:
        return 0.0
    return float(np.min(nonzero.real))


def sustained_gain(
    adjacency, *, t_max: float = 40.0, samples: int = 200
) -> float:
    r"""``M = sup_{s≥0} ‖e^{−sL} Q‖₂`` — the sustained non-consensus gain.

    ``M = 1`` for a normal generator (contraction); ``M > 1`` for a non-normal
    one (the transient cost that a stable spectrum alone does not show).
    """
    laplacian = directed_rw_laplacian(adjacency)
    q = consensus_projection(adjacency)
    gain = 0.0
    for s in np.linspace(0.0, t_max, samples):
        gain = max(
            gain, float(np.linalg.norm(matrix_exponential(-laplacian * s) @ q,
                                       2))
        )
    return gain


def structural_time(vf: Callable[[float], float], t_grid) -> np.ndarray:
    r"""Structural time ``s(t) = ∫₀^t ν_f(τ) dτ`` (cumulative trapezoid)."""
    t = np.asarray(t_grid, dtype=float)
    vals = np.array([float(vf(ti)) for ti in t], dtype=float)
    ds = (vals[1:] + vals[:-1]) / 2.0 * np.diff(t)
    return np.concatenate([[0.0], np.cumsum(ds)])


def _rk4_transport(laplacian, x0, vf, t_grid) -> np.ndarray:
    r"""RK4 integration of ``ẋ = −ν_f(t) L x`` on ``t_grid`` (returns x(t))."""
    x = np.asarray(x0, dtype=float).copy()
    out = [x.copy()]
    for i in range(len(t_grid) - 1):
        t, h = t_grid[i], t_grid[i + 1] - t_grid[i]

        def f(tt, xx):
            return -float(vf(tt)) * (laplacian @ xx)

        k1 = f(t, x)
        k2 = f(t + h / 2, x + h / 2 * k1)
        k3 = f(t + h / 2, x + h / 2 * k2)
        k4 = f(t + h, x + h * k3)
        x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        out.append(x.copy())
    return np.array(out)


def clock_change_residual(
    adjacency, x0, vf: Callable[[float], float], t_grid
) -> float:
    r"""``max_t ‖x_RK4(t) − e^{−s(t)L} x₀‖`` for ``ẋ = −ν_f(t) L x``.

    A numerical (non-exponential) integrator of the time-varying ODE converges to
    the clock-changed exponential ``e^{−s(t)L} x₀`` — the structural-time theorem.
    """
    laplacian = directed_rw_laplacian(adjacency)
    t = np.asarray(t_grid, dtype=float)
    x_rk4 = _rk4_transport(laplacian, x0, vf, t)
    s = structural_time(vf, t)
    x0v = np.asarray(x0, dtype=float)
    x_exact = np.array(
        [matrix_exponential(-laplacian * si) @ x0v for si in s]
    )
    return float(np.max(np.abs(x_rk4 - x_exact)))


def reorganization_time_invariance_residual(
    adjacency, x0, vf: Callable[[float], float], t_grid
) -> float:
    r"""Relative residual of the change-of-variables identity

    ``∫ ν_f(t) ‖L e^{−s(t)L} Q x₀‖ dt = ∫ ‖L e^{−sL} Q x₀‖ ds``,

    i.e. the total reorganization is **invariant** under the ``ν_f`` clock change
    (it depends on the trajectory, not the speed).
    """
    laplacian = directed_rw_laplacian(adjacency)
    q = consensus_projection(adjacency)
    x0v = np.asarray(x0, dtype=float)
    t = np.asarray(t_grid, dtype=float)
    s = structural_time(vf, t)

    def speed(si):
        return float(np.linalg.norm(
            laplacian @ (matrix_exponential(-laplacian * si) @ q @ x0v)
        ))

    lhs = float(np.trapezoid([float(vf(ti)) * speed(si)
                              for ti, si in zip(t, s)], t))
    s_grid = np.linspace(0.0, s[-1], len(t))
    rhs = float(np.trapezoid([speed(si) for si in s_grid], s_grid))
    return abs(lhs - rhs) / max(rhs, np.finfo(float).eps)


def total_variation_bound(
    adjacency, x0, *, t_max: float = 60.0, samples: int = 600
) -> tuple[float, float, bool]:
    r"""``(J, bound, J ≤ bound)`` for the total reorganization on the
    non-consensus subspace.

    ``J = ∫₀^∞ ‖L e^{−sL} Q x₀‖ ds`` and ``bound = M ‖LQ‖ ‖x₀‖ / ω`` with
    ``M = sustained_gain``, ``ω = nonconsensus_abscissa``. Finiteness of ``J`` is
    the linear-EPI-channel form of U2 integral convergence.
    """
    laplacian = directed_rw_laplacian(adjacency)
    q = consensus_projection(adjacency)
    x0v = np.asarray(x0, dtype=float)
    ts = np.linspace(0.0, t_max, samples)
    speeds = [
        float(np.linalg.norm(
            laplacian @ (matrix_exponential(-laplacian * s) @ q @ x0v)))
        for s in ts
    ]
    j = float(np.trapezoid(speeds, ts))
    m = sustained_gain(adjacency)
    omega = nonconsensus_abscissa(adjacency)
    lq_norm = float(np.linalg.norm(laplacian @ q, 2))
    x0_norm = float(np.linalg.norm(x0v, 2))
    bound = m * lq_norm * x0_norm / omega if omega > 0 else float("inf")
    return j, bound, j <= bound * (1.0 + 1e-6)


@dataclass(frozen=True)
class StructuralTimeCertificate:
    """Certificate of the scalar-frequency structural-time theorem (N04)."""

    nonconsensus_abscissa: float
    sustained_gain: float  # M
    clock_change_residual: float
    reorganization_invariance_residual: float
    total_reorganization: float
    total_reorganization_bound: float
    bound_holds: bool


def certify_structural_time(
    adjacency, x0, vf: Callable[[float], float], t_grid
) -> StructuralTimeCertificate:
    r"""Bundle the structural-time theorem checks for a digraph and ``ν_f(t)``."""
    omega = nonconsensus_abscissa(adjacency)
    m = sustained_gain(adjacency)
    clock = clock_change_residual(adjacency, x0, vf, t_grid)
    invariance = reorganization_time_invariance_residual(
        adjacency, x0, vf, t_grid
    )
    j, bound, holds = total_variation_bound(adjacency, x0)
    return StructuralTimeCertificate(
        nonconsensus_abscissa=omega,
        sustained_gain=m,
        clock_change_residual=clock,
        reorganization_invariance_residual=invariance,
        total_reorganization=j,
        total_reorganization_bound=bound,
        bound_holds=holds,
    )
