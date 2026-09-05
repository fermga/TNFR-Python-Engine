r"""Heterogeneous nodal frequency — where the clock-change theorem stops (R9b, N13).

The scalar-frequency structural-time theorem (N04) needs a **common** ``ν_f(t)``:
then ``ẋ = −ν_f(t) L x`` integrates to the clock change ``x(t) = e^{−s(t)L} x₀``
because all the generators ``ν_f(τ)L`` commute.  With a **heterogeneous** nodal
frequency ``D_{ν_f}(t) = diag(ν_f_1(t), …, ν_f_n(t))`` the transport is

    ``ẋ = −D_{ν_f}(t) · L · x``,

and the generators ``D_{ν_f}(t)·L`` at different times **do not commute**, so there
is **no clock change**: ``e^{−s(t)L}x₀`` is not the solution.  This module measures
that boundary and the associated stability questions (R9-T05), so the N04 theorem
is not over-extended.

**Honest scope.**  The failure of the scalar-time ansatz is DERIVED (the
commutator is non-zero) and MEASURED.  A *fixed* positive ``D_{ν_f}`` keeps
``−D_{ν_f}L`` stable (consensus preserved, spectral abscissa ``≤ 0``); the
**uniform** stability of the time-varying / switched flow is only MEASURED on the
tested schedules — a general bound is **OPEN** (`NT-P09` heterogeneous).  U2/U6 in
[AGENTS.md](../../../AGENTS.md) are **not** modified.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .directed_diffusion import consensus_projection, directed_rw_laplacian
from .spectral_projectors import matrix_exponential, spectral_abscissa

__all__ = [
    "heterogeneous_generator",
    "generator_commutator_norm",
    "scalar_schedule",
    "heterogeneous_schedule",
    "structural_time_mean",
    "scalar_time_ansatz_residual",
    "fixed_generator_abscissa",
    "nonconsensus_transient_gain",
    "HeterogeneousVfCertificate",
    "certify_heterogeneous_vf",
]

VfSchedule = Callable[[float], np.ndarray]


def heterogeneous_generator(vf_diag, adjacency) -> np.ndarray:
    r"""The time-frozen generator ``D_{ν_f} · L`` for a nodal frequency vector."""
    return np.diag(np.asarray(vf_diag, dtype=float)) @ directed_rw_laplacian(
        adjacency)


def generator_commutator_norm(vf1, vf2, adjacency) -> float:
    r"""``‖[D₁L, D₂L]‖₂`` — zero iff the two generators share an eigenbasis.

    For **scalar** (all-equal) ``ν_f`` both are multiples of ``L`` and commute;
    a **heterogeneous** ``ν_f`` makes it non-zero, so no clock change exists.
    """
    a = heterogeneous_generator(vf1, adjacency)
    b = heterogeneous_generator(vf2, adjacency)
    return float(np.linalg.norm(a @ b - b @ a, 2))


def scalar_schedule(n: int, base: float = 1.0, amp: float = 0.4,
                    freq: float = 1.0) -> VfSchedule:
    r"""A **common** frequency schedule ``ν_f(t) = base·(1 + amp·sin(freq·t))·1``
    (identical on every node — a genuine clock change)."""

    def vf(t: float) -> np.ndarray:
        return base * (1.0 + amp * np.sin(freq * t)) * np.ones(n)
    return vf


def heterogeneous_schedule(n: int, base: float = 1.0,
                           amp: float = 0.6) -> VfSchedule:
    r"""A **per-node** frequency schedule ``ν_f_i(t) = base·(1 + amp·sin(t + i))``
    (a different phase per node — no clock change)."""
    idx = np.arange(n)

    def vf(t: float) -> np.ndarray:
        return base * (1.0 + amp * np.sin(t + idx))
    return vf


def structural_time_mean(vf: VfSchedule, t_grid) -> np.ndarray:
    r"""Cumulative mean structural time ``s̄(t) = ∫ mean_i ν_f_i`` (the best
    scalar surrogate for a heterogeneous schedule)."""
    t = np.asarray(t_grid, dtype=float)
    means = np.array([float(np.mean(vf(ti))) for ti in t])
    ds = (means[1:] + means[:-1]) / 2.0 * np.diff(t)
    return np.concatenate([[0.0], np.cumsum(ds)])


def _rk4_heterogeneous(laplacian, x0, vf: VfSchedule, t_grid) -> np.ndarray:
    r"""RK4 of ``ẋ = −D_{ν_f}(t) L x`` on ``t_grid``."""
    x = np.asarray(x0, dtype=float).copy()
    out = [x.copy()]
    for i in range(len(t_grid) - 1):
        t, h = t_grid[i], t_grid[i + 1] - t_grid[i]

        def f(tt, xx):
            return -(np.asarray(vf(tt)) * (laplacian @ xx))

        k1 = f(t, x)
        k2 = f(t + h / 2, x + h / 2 * k1)
        k3 = f(t + h / 2, x + h / 2 * k2)
        k4 = f(t + h, x + h * k3)
        x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        out.append(x.copy())
    return np.array(out)


def scalar_time_ansatz_residual(adjacency, x0, vf: VfSchedule, t_grid) -> float:
    r"""``max_t ‖x_RK4(t) − e^{−s̄(t)L} x₀‖`` for ``ẋ = −D_{ν_f}(t) L x``.

    The scalar clock-change ansatz uses the mean frequency ``s̄ = ∫ mean ν_f``; it
    is exact (``≈ 0``) for a common schedule and **fails** (large) for a
    heterogeneous one — the N04 theorem does not extend.
    """
    laplacian = directed_rw_laplacian(adjacency)
    t = np.asarray(t_grid, dtype=float)
    x_rk4 = _rk4_heterogeneous(laplacian, x0, vf, t)
    s = structural_time_mean(vf, t)
    x0v = np.asarray(x0, dtype=float)
    resid = 0.0
    for i in range(len(t)):
        ansatz = matrix_exponential(-laplacian * s[i]) @ x0v
        resid = max(resid, float(np.linalg.norm(x_rk4[i] - ansatz)))
    return resid


def fixed_generator_abscissa(vf_diag, adjacency) -> float:
    r"""``α(−D_{ν_f} L)`` for a **fixed** positive nodal frequency.

    ``≤ 0`` means the frozen heterogeneous generator is stable (it preserves
    consensus, ``L·1 = 0 ⇒ D_{ν_f} L·1 = 0``, and the rest decays)."""
    return spectral_abscissa(-heterogeneous_generator(vf_diag, adjacency))


def nonconsensus_transient_gain(adjacency, x0, vf: VfSchedule, t_grid) -> float:
    r"""``sup_t ‖Q x(t)‖ / ‖Q x₀‖`` — heterogeneity-induced amplification of the
    non-consensus (reorganizing) component under ``ẋ = −D_{ν_f}(t) L x``."""
    laplacian = directed_rw_laplacian(adjacency)
    q = consensus_projection(adjacency)
    x0v = np.asarray(x0, dtype=float)
    traj = _rk4_heterogeneous(laplacian, x0v, vf, t_grid)
    base = float(np.linalg.norm(q @ x0v))
    if base == 0.0:
        return 0.0
    return max(float(np.linalg.norm(q @ x)) / base for x in traj)


@dataclass(frozen=True)
class HeterogeneousVfCertificate:
    """Where the scalar clock-change theorem stops (R9b, N13)."""

    commutator_scalar: float           # ≈ 0 (common ν_f commutes)
    commutator_heterogeneous: float    # > 0 (no clock change)
    scalar_time_residual_common: float       # ≈ 0 (clock change holds)
    scalar_time_residual_heterogeneous: float  # large (theorem fails)
    scalar_time_theorem_extends: bool  # False for heterogeneous ν_f
    fixed_generator_abscissa: float    # ≤ 0 (frozen D is stable)
    fixed_generator_stable: bool
    heterogeneity_transient_gain: float
    tolerance: float
    claim_status: str


def certify_heterogeneous_vf(adjacency, x0, t_grid, *,
                             tol: float = 1e-6) -> HeterogeneousVfCertificate:
    r"""Contrast a common vs a heterogeneous ``ν_f`` schedule on ``ẋ = −D_{ν_f}L x``.

    Documents that the N04 clock-change theorem holds for a common schedule and
    **fails** for a heterogeneous one, while a frozen positive ``D_{ν_f}`` is still
    stable.  Does **not** modify U2/U6.
    """
    n = len(np.asarray(x0))
    common = scalar_schedule(n)
    hetero = heterogeneous_schedule(n)
    t0 = float(t_grid[0])
    comm_scalar = generator_commutator_norm(common(t0), common(t0 + 1.0),
                                            adjacency)
    comm_hetero = generator_commutator_norm(hetero(t0), hetero(t0 + 1.0),
                                            adjacency)
    res_common = scalar_time_ansatz_residual(adjacency, x0, common, t_grid)
    res_hetero = scalar_time_ansatz_residual(adjacency, x0, hetero, t_grid)
    abscissa = fixed_generator_abscissa(hetero(t0), adjacency)
    gain = nonconsensus_transient_gain(adjacency, x0, hetero, t_grid)
    extends = res_hetero < max(tol, 1e-6)
    return HeterogeneousVfCertificate(
        commutator_scalar=comm_scalar,
        commutator_heterogeneous=comm_hetero,
        scalar_time_residual_common=res_common,
        scalar_time_residual_heterogeneous=res_hetero,
        scalar_time_theorem_extends=extends,
        fixed_generator_abscissa=abscissa,
        fixed_generator_stable=(abscissa <= max(tol, 1e-6)),
        heterogeneity_transient_gain=gain,
        tolerance=tol,
        claim_status=(
            "scalar clock-change theorem does NOT extend to heterogeneous nu_f "
            "DERIVED (commutator != 0) + MEASURED (scalar-time residual large); "
            "fixed-D stability MEASURED; uniform time-varying stability OPEN "
            "(NT-P09 heterogeneous); U2/U6 unmodified"
        ),
    )
