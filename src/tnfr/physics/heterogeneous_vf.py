r"""Heterogeneous nodal frequency — where the clock-change theorem stops.

The scalar-frequency structural-time theorem (N04) needs a **common** ``ν_f(t)``:
then ``ẋ = −ν_f(t) L x`` integrates to the clock change ``x(t) = e^{−s(t)L} x₀``
because all the generators ``ν_f(τ)L`` commute.  With a **heterogeneous** nodal
frequency ``D_{ν_f}(t) = diag(ν_f_1(t), …, ν_f_n(t))`` the transport is

    ``ẋ = −D_{ν_f}(t) · L · x``,

and the generators ``D_{ν_f}(t)·L`` at different times **do not commute**, so there
is **no clock change**: ``e^{−s(t)L}x₀`` is not the solution.  This module measures
that boundary and its associated stability questions, so the scalar theorem is
not over-extended.

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

from .directed_diffusion import (
    consensus_projection,
    directed_rw_laplacian,
    stationary_distribution,
)
from .spectral_projectors import matrix_exponential, spectral_abscissa

__all__ = [
    "heterogeneous_generator",
    "heterogeneous_stationary_distribution",
    "FixedMeasureComparison",
    "compare_fixed_stationary_measures",
    "ConvexHullGeneratorCertificate",
    "certify_convex_hull_generator",
    "generator_commutator_norm",
    "scalar_schedule",
    "heterogeneous_schedule",
    "structural_time_mean",
    "scalar_time_ansatz_residual",
    "fixed_generator_abscissa",
    "nonconsensus_transient_gain",
    "piecewise_propagator",
    "within_initial_convex_hull",
    "TimeDependentDirichletBalance",
    "DirichletConvergenceCertificate",
    "time_dependent_dirichlet_balance",
    "certify_dirichlet_convergence",
    "FiniteScheduleReadout",
    "finite_schedule_readout",
    "HeterogeneousVfCertificate",
    "certify_heterogeneous_vf",
]

VfSchedule = Callable[[float], np.ndarray]


def _symmetric_conductance(
    adjacency,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    weights = np.asarray(adjacency, dtype=float)
    if (
        weights.ndim != 2
        or weights.shape[0] != weights.shape[1]
        or not np.all(np.isfinite(weights))
        or np.any(weights < 0.0)
    ):
        raise ValueError("adjacency must be square, finite and nonnegative")
    if not np.allclose(weights, weights.T, atol=1e-12, rtol=1e-10):
        raise ValueError("Dirichlet identity requires symmetric conductance")
    degree = weights.sum(axis=1)
    laplacian = np.diag(degree) - weights
    return weights, degree, laplacian


@dataclass(frozen=True)
class TimeDependentDirichletBalance:
    """Instantaneous fixed-topology energy identity for ``M(t)``."""

    energy: float
    gradient: np.ndarray
    mobility: np.ndarray
    state_rate: np.ndarray
    energy_rate: float
    dissipation: float
    identity_residual: float
    topology_term: float
    scope: str


@dataclass(frozen=True)
class FixedMeasureComparison:
    """Original walk measure versus the fixed-capacity invariant measure."""

    stationary_distribution: tuple[float, ...]
    capacity_weighted_distribution: tuple[float, ...]
    mismatch_norm: float
    left_residual: float


def compare_fixed_stationary_measures(
    vf_diag, adjacency
) -> FixedMeasureComparison:
    """Expose the change from ``pi`` to normalized ``pi/nu_f``."""
    pi = stationary_distribution(adjacency)
    eta = heterogeneous_stationary_distribution(vf_diag, adjacency)
    generator = heterogeneous_generator(vf_diag, adjacency)
    return FixedMeasureComparison(
        tuple(float(value) for value in pi),
        tuple(float(value) for value in eta),
        float(np.linalg.norm(pi - eta, 1)),
        float(np.linalg.norm(eta @ generator, np.inf)),
    )


@dataclass(frozen=True)
class ConvexHullGeneratorCertificate:
    """Metzler/zero-row-sum certificate for scalar interval preservation."""

    is_metzler: bool
    zero_row_sums: bool
    preserves_convex_hull: bool
    minimum_off_diagonal: float
    maximum_row_sum_residual: float
    scope: str


def certify_convex_hull_generator(
    vf_diag, adjacency, *, tolerance: float = 1e-12
) -> ConvexHullGeneratorCertificate:
    """Certify the frozen generator ``-diag(nu_f)L`` as Markov-positive.

    A finite Metzler matrix with zero row sums generates a nonnegative
    row-stochastic semigroup, hence preserves every scalar initial interval.
    Zero capacities are admitted and produce frozen rows.
    """
    vf = np.asarray(vf_diag, dtype=float)
    if vf.ndim != 1 or not np.all(np.isfinite(vf)) or np.any(vf < 0.0):
        raise ValueError("capacity must be finite and nonnegative")
    generator = -heterogeneous_generator(vf, adjacency)
    off_diagonal = generator.copy()
    np.fill_diagonal(off_diagonal, 0.0)
    minimum = float(np.min(off_diagonal)) if off_diagonal.size else 0.0
    row_residual = (
        float(np.max(np.abs(generator.sum(axis=1)))) if len(vf) else 0.0
    )
    metzler = minimum >= -tolerance
    zero_rows = row_residual <= tolerance
    return ConvexHullGeneratorCertificate(
        metzler,
        zero_rows,
        metzler and zero_rows,
        minimum,
        row_residual,
        "finite frozen scalar EPI generator; products preserve the interval",
    )


def time_dependent_dirichlet_balance(
    adjacency, state, capacity
) -> TimeDependentDirichletBalance:
    r"""Evaluate ``E' = -(Bx)^T M(t)(Bx)`` on fixed symmetric conductance.

    ``M_ii(t)=nu_i(t)/d_i`` on positive-strength vertices and zero at
    isolates.  Time variation of nonnegative capacity changes mobility but
    introduces no ``M_dot`` term because ``E_D`` depends on ``B`` and ``x``,
    not on the metric.  The topology term is zero because ``B`` is fixed.
    """
    _, degree, b = _symmetric_conductance(adjacency)
    x = np.asarray(state, dtype=float)
    vf = np.asarray(capacity, dtype=float)
    if x.shape != degree.shape or vf.shape != degree.shape:
        raise ValueError("state and capacity must match adjacency")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(vf)):
        raise ValueError("state and capacity must be finite")
    if np.any(vf < 0.0):
        raise ValueError("capacity must be nonnegative")
    mobility = np.divide(vf, degree, out=np.zeros_like(vf), where=degree > 0.0)
    gradient = b @ x
    state_rate = -mobility * gradient
    energy = float(0.5 * x @ gradient)
    energy_rate = float(gradient @ state_rate)
    dissipation = float(gradient @ (mobility * gradient))
    return TimeDependentDirichletBalance(
        energy,
        gradient.copy(),
        mobility.copy(),
        state_rate.copy(),
        energy_rate,
        dissipation,
        abs(energy_rate + dissipation),
        0.0,
        "instantaneous fixed symmetric graph; "
        "time-varying nonnegative mobility",
    )


@dataclass(frozen=True)
class DirichletConvergenceCertificate:
    """Sufficient fixed-graph energy hierarchy under bounded capacities."""

    algebraic_connectivity: float
    capacity_lower_bound: float
    capacity_upper_bound: float
    integrated_min_mobility: float
    initial_energy: float
    finite_horizon_energy_bound: float
    convex_hull_preserved: bool
    dirichlet_nonincreasing: bool
    asymptotic_consensus_guaranteed: bool
    total_variation_status: str
    total_variation_bound: float | None
    scope: str


def certify_dirichlet_convergence(
    adjacency,
    state,
    *,
    capacity_lower_bound: float,
    capacity_upper_bound: float,
    integrated_min_mobility: float,
    tail_integral_diverges: bool,
) -> DirichletConvergenceCertificate:
    r"""Apply ``E(t)<=E(0) exp(-2 lambda_2 int mu_min)``.

    The energy and total-variation bounds are sufficient upper bounds under
    the stated exponential energy-decay hypotheses; tightness and necessity
    are not claimed.  A finite mobility clock or zero lower bound preserves
    the convex-hull statement but does not imply asymptotic consensus.  No
    finite total-variation bound is inferred without an additional integrable
    rate estimate.
    """
    _, degree, b = _symmetric_conductance(adjacency)
    x = np.asarray(state, dtype=float)
    bounds = (capacity_lower_bound, capacity_upper_bound,
              integrated_min_mobility)
    if x.shape != degree.shape or not np.all(np.isfinite(x)):
        raise ValueError("state must be finite and match adjacency")
    if not all(np.isfinite(value) and value >= 0.0 for value in bounds):
        raise ValueError(
            "capacity and mobility bounds must be finite and nonnegative"
        )
    if capacity_lower_bound > capacity_upper_bound:
        raise ValueError("capacity lower bound cannot exceed upper bound")
    eigenvalues = np.linalg.eigvalsh(b)
    positive = eigenvalues[eigenvalues > 1e-12]
    gap = float(positive[0]) if positive.size else 0.0
    energy = float(0.5 * x @ (b @ x))
    energy_bound = float(
        energy * np.exp(-2.0 * gap * integrated_min_mobility)
    )
    connected = bool(len(x) <= 1 or np.sum(eigenvalues <= 1e-12) == 1)
    consensus = bool(
        connected and capacity_lower_bound > 0.0 and tail_integral_diverges
    )
    positive_degree = degree[degree > 0.0]
    minimum_mobility = (
        float(capacity_lower_bound / np.max(positive_degree))
        if positive_degree.size else 0.0
    )
    maximum_mobility = (
        float(capacity_upper_bound / np.min(positive_degree))
        if positive_degree.size else 0.0
    )
    maximum_eigenvalue = float(eigenvalues[-1]) if eigenvalues.size else 0.0
    variation_bound = None
    if consensus and gap > 0.0 and minimum_mobility > 0.0:
        variation_bound = float(
            maximum_mobility * np.sqrt(2.0 * maximum_eigenvalue * energy)
            / (gap * minimum_mobility)
        )
    variation = (
        "CONDITIONAL_FINITE_FROM_EXPONENTIAL_ENERGY_DECAY"
        if consensus
        else "NOT_IMPLIED_BY_FINITE_OR_ZERO_MOBILITY_CLOCK"
    )
    return DirichletConvergenceCertificate(
        gap,
        float(capacity_lower_bound),
        float(capacity_upper_bound),
        float(integrated_min_mobility),
        energy,
        energy_bound,
        True,
        True,
        consensus,
        variation,
        variation_bound,
        "fixed symmetric conductance; scalar EPI channel; sufficient bound",
    )


@dataclass(frozen=True)
class FiniteScheduleReadout:
    """Endpoint norm and exact-kernel U6 readings for a finite schedule."""

    endpoint_states: tuple[tuple[float, ...], ...]
    euclidean_deviations: tuple[float, ...]
    instantaneous_weighted_deviations: tuple[float, ...]
    metric_capacity_context: tuple[str, ...]
    mean_absolute_u6_drifts: tuple[float, ...]
    maximum_u6_drift: float
    u6_confined_at_endpoints: bool
    u6_kernel: str
    time_coverage: str
    metric_derivative_status: str


def finite_schedule_readout(graph, state, segments) -> FiniteScheduleReadout:
    """Measure finite segment endpoints without claiming interval safety.

    Pressure is the exact isolated-channel value ``-L_rw x`` and potential is
    evaluated with the canonical directed weighted shortest-path kernel.
    The capacity-dependent stationary norm is an instantaneous readout only;
    no smooth metric-energy identity is asserted across segment switches.
    """
    import networkx as nx

    from ..constants.canonical import U6_STRUCTURAL_POTENTIAL_LIMIT
    from .transient_u2 import potential_operator_from_graph

    nodes, kernel = potential_operator_from_graph(graph)
    adjacency = nx.to_numpy_array(graph, nodelist=nodes, weight="weight")
    laplacian = directed_rw_laplacian(adjacency)
    x = np.asarray(state, dtype=float)
    if x.shape != (len(nodes),) or not np.all(np.isfinite(x)):
        raise ValueError("state must be finite and match graph nodes")
    states = [x.copy()]
    capacities = []
    for duration, capacity in segments:
        vf = np.asarray(capacity, dtype=float)
        propagator = piecewise_propagator(adjacency, [(duration, vf)])
        x = propagator @ x
        states.append(x.copy())
        capacities.append(vf)

    initial_pressure = -laplacian @ states[0]
    initial_potential = kernel @ initial_pressure
    euclidean = []
    weighted = []
    metric_context = []
    u6 = []
    for index, current in enumerate(states):
        centered = current - np.mean(current)
        euclidean.append(float(np.linalg.norm(centered)))
        if capacities:
            vf = capacities[max(0, min(index - 1, len(capacities) - 1))]
            metric_context.append(
                "first_segment" if index == 0 else f"preceding_segment_{index}"
            )
        else:
            vf = np.ones(len(nodes))
            metric_context.append("unit_default")
        if np.all(vf > 0.0):
            eta = heterogeneous_stationary_distribution(vf, adjacency)
            mean = float(eta @ current)
            weighted.append(float(np.sqrt(eta @ ((current - mean) ** 2))))
        else:
            weighted.append(float("nan"))
        pressure = -laplacian @ current
        potential = kernel @ pressure
        u6.append(float(np.mean(np.abs(potential - initial_potential))))
    return FiniteScheduleReadout(
        tuple(tuple(float(value) for value in current) for current in states),
        tuple(euclidean),
        tuple(weighted),
        tuple(metric_context),
        tuple(u6),
        max(u6, default=0.0),
        all(value < U6_STRUCTURAL_POTENTIAL_LIMIT for value in u6),
        "canonical_directed_weighted_shortest_path_inverse_square",
        "segment_endpoints_only",
        "NOT_USED_AS_ENERGY; SWITCH_JUMPS_REPORTED_ONLY_AS_READOUTS",
    )


def heterogeneous_generator(vf_diag, adjacency) -> np.ndarray:
    r"""The time-frozen generator ``D_{ν_f} · L`` for a nodal frequency vector."""
    return np.diag(np.asarray(vf_diag, dtype=float)) @ directed_rw_laplacian(
        adjacency)


def heterogeneous_stationary_distribution(vf_diag, adjacency) -> np.ndarray:
    r"""Left invariant probability for fixed positive heterogeneous capacity.

    If ``pi.T @ L = 0`` and ``A = diag(nu_f) @ L``, then the normalized vector
    ``eta_i proportional to pi_i / nu_f_i`` satisfies ``eta.T @ A = 0``.
    This is an identity for the fixed linear EPI channel, not a canonical U2
    metric or a statement about time-varying capacities.
    """
    vf = np.asarray(vf_diag, dtype=float)
    if vf.ndim != 1 or not np.all(np.isfinite(vf)) or np.any(vf <= 0.0):
        raise ValueError("fixed heterogeneous capacity must be finite and positive")
    pi = stationary_distribution(adjacency)
    if len(vf) != len(pi):
        raise ValueError("capacity length must match adjacency")
    eta = pi / vf
    return eta / eta.sum()


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
    heterogeneous one — the scalar theorem does not extend.
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


def piecewise_propagator(adjacency, segments) -> np.ndarray:
    r"""Time-ordered propagator for fixed-capacity segments.

    Each segment is ``(duration, capacity_vector)`` and evolves with
    ``exp(-duration * diag(capacity) * L)``. Later segments multiply on the
    left, so reversing a noncommuting schedule is intentionally distinct.
    """
    laplacian = directed_rw_laplacian(adjacency)
    propagator = np.eye(len(laplacian))
    for duration, capacity in segments:
        if not np.isfinite(duration) or duration < 0.0:
            raise ValueError("segment duration must be finite and nonnegative")
        generator = heterogeneous_generator(capacity, adjacency)
        if generator.shape != laplacian.shape:
            raise ValueError("capacity length must match adjacency")
        propagator = matrix_exponential(-float(duration) * generator) @ propagator
    return propagator


def within_initial_convex_hull(propagator, x0, *, tol: float = 1e-10) -> bool:
    r"""Whether a measured scalar propagator preserves an initial interval.

    This numerical predicate checks nonnegative row weights summing to one and
    the resulting state interval. It is suitable for finite piecewise controls;
    it does not certify arbitrary switching or nonlinear pressure dynamics.
    """
    p = np.asarray(propagator, dtype=float)
    x = np.asarray(x0, dtype=float)
    if p.ndim != 2 or p.shape[0] != p.shape[1] or p.shape[1] != len(x):
        raise ValueError("propagator and state dimensions must agree")
    if not np.isfinite(tol) or tol < 0.0:
        raise ValueError("tol must be finite and nonnegative")
    result = p @ x
    return bool(
        np.all(p >= -tol)
        and np.allclose(p.sum(axis=1), 1.0, atol=tol, rtol=0.0)
        and np.all(result >= np.min(x) - tol)
        and np.all(result <= np.max(x) + tol)
    )


@dataclass(frozen=True)
class HeterogeneousVfCertificate:
    """Where the scalar clock-change theorem stops."""

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

    Documents that the scalar clock-change theorem holds for a common schedule and
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
