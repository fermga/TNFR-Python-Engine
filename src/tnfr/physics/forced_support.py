"""Exact relative profiles for fixed-support forced nodal diffusion.

The forcing is an explicit held coefficient vector, not inferred from stored
pressure. Phase/capacity/support realization and causal execution remain
separate obligations. All operations below act on detached rational data.
"""

from collections.abc import Mapping, Set
from dataclasses import dataclass, replace
from fractions import Fraction

from .._exact_time import exact_or_represented_real
from ._cycle_algebra import Vector, dot, ordered_vector
from ._exact_linear_algebra import exact_matrix_inverse
from .support_transport import (
    SupportTransportEuler, SupportTransportReset, SupportTransportSnapshot,
    _energy, _laplacian, _rebuild,
    observe_support_transport_euler, observe_support_transport_reset,
)

__all__ = [
    "ForcedSupportBalance", "ForcedSupportState", "ForcedSupportStep",
    "derive_forced_support_balance", "observe_forced_support_state",
    "observe_forced_support_step",
    "ForcedSupportPattern", "ForcedSupportResetEnergy", "ForcedSupportReset",
    "observe_forced_support_pattern", "observe_forced_support_reset",
]


@dataclass(frozen=True)
class ForcedSupportBalance:
    """Held exact model and its H-centered relative profile; no causal seal."""

    source: SupportTransportSnapshot
    epi_weight: Fraction
    forcing: Vector
    strengths: Vector
    metric_weights: Vector
    compatibility_residual: Fraction
    mean_drift: Fraction
    relative_profile: Vector
    profile_residual: Vector
    profile_center_residual: Fraction
    has_zero_pressure_equilibrium: bool
    max_convex_step: Fraction


def derive_forced_support_balance(
    snapshot, *, epi_weight, forcing,
) -> ForcedSupportBalance:
    """Solve e*B*z=D*F-vbar*(d/nu), fixing the gauge sum(d*z/nu)=0.

    The nodal model is xdot=diag(nu)*(-e*D^-1*B*x+F). It requires connected
    positive conductance, positive strengths/capacity and e>0. Support-only
    zero-weight links do not establish this connectivity. Effective weights
    follow the supplied snapshot; rational inputs remain exact.

    The rank-one term h*h^T makes the gauge system nonsingular. It is only
    a linear-algebra device and never a new term in the nodal pressure.
    Exponential relative relaxation assumes these coefficients remain fixed
    in an unrestricted scalar chart. Finite clipping or later operators
    require separate observations; a stored pressure is not a forcing law.
    """
    source = _rebuild(snapshot)
    e = exact_or_represented_real(epi_weight, "epi_weight")
    f = ordered_vector(forcing, "forcing")
    size = len(source.nodes)
    if not size or len(f) != size:
        raise ValueError("forcing must match a nonempty node order")
    if e <= 0 or any(nu <= 0 for nu in source.capacity):
        raise ValueError("EPI weight and every capacity must be positive")
    strengths = [Fraction(0) for _ in range(size)]
    matrix = [[Fraction(0) for _ in range(size)] for _ in range(size)]
    neighbors = [set() for _ in range(size)]
    for i, j, w in source.conductance:
        strengths[i] += w
        matrix[i][i] += w
        matrix[i][j] -= w
        neighbors[i].add(j)
    visited, pending = {0}, [0]
    while pending:
        i = pending.pop()
        new = neighbors[i] - visited
        visited.update(new)
        pending.extend(new)
    if len(visited) != size or any(d <= 0 for d in strengths):
        raise ValueError("positive conductance must be connected with positive strengths")
    d = tuple(strengths)
    metric = tuple(di / nu for di, nu in zip(d, source.capacity))
    mass = sum(metric, Fraction(0))
    compatibility = dot(d, f)
    drift = compatibility / mass
    rhs = tuple(di * fi - drift * hi for di, fi, hi in zip(d, f, metric))
    gauge_system = tuple(
        tuple(e * value + metric[i] * metric[j] for j, value in enumerate(row))
        for i, row in enumerate(matrix)
    )
    inverse = exact_matrix_inverse(gauge_system)
    profile = tuple(dot(row, rhs) for row in inverse)
    residual = tuple(
        e * value - target
        for value, target in zip(_laplacian(source.conductance, profile), rhs)
    )
    center_residual = dot(metric, profile)
    if any(residual) or center_residual:
        raise RuntimeError("exact relative profile lost its Poisson or gauge identity")
    return ForcedSupportBalance(
        source, e, f, d, metric, compatibility, drift, profile, residual,
        center_residual, compatibility == 0, 1 / (e * max(source.capacity)),
    )


def _reference(value):
    if type(value) is not ForcedSupportBalance:
        raise TypeError("reference must be a ForcedSupportBalance")
    # Public cached fields are not provenance or trusted mathematical results.
    return derive_forced_support_balance(
        value.source, epi_weight=value.epi_weight, forcing=value.forcing,
    )


@dataclass(frozen=True)
class ForcedSupportPattern:
    """EPI compared with one fixed profile and its original metrics only."""

    nodes: tuple
    epi: Vector
    mean: Fraction
    relative_error: Vector
    error_variance: Fraction
    error_dirichlet_energy: Fraction


def _pattern(reference, epi):
    mean = dot(reference.metric_weights, epi) / sum(reference.metric_weights)
    error = tuple(x - mean - z for x, z in zip(epi, reference.relative_profile))
    return ForcedSupportPattern(
        reference.source.nodes, epi, mean, error,
        dot(reference.metric_weights, tuple(u**2 for u in error)) / 2,
        _energy(reference.source.conductance, error),
    )


def observe_forced_support_pattern(reference, *, nodes, epi) -> ForcedSupportPattern:
    """Retain an original pattern readout when the actual dynamics changes.

    Only the ordered node IDs and EPI are supplied. The original H, B and z
    remain fixed, so this comparison cannot certify current pressure, a
    preserved evolution law or recovered physical identity. It intentionally
    differs from observe_forced_support_state's held-model validation.
    """
    ref = _reference(reference)
    if isinstance(nodes, (str, bytes, bytearray, Mapping, Set)):
        raise TypeError("nodes must be an ordered sequence")
    if tuple(nodes) != ref.source.nodes:
        raise ValueError("pattern comparison requires the original node order")
    x = ordered_vector(epi, "epi")
    if len(x) != len(ref.source.nodes):
        raise ValueError("EPI must match the original node order")
    return _pattern(ref, x)


@dataclass(frozen=True)
class ForcedSupportState:
    """Mean, relative error and pressure defect of one detached state."""

    snapshot: SupportTransportSnapshot
    mean: Fraction
    relative_error: Vector
    error_variance: Fraction
    error_dirichlet_energy: Fraction
    modeled_pressure: Vector
    pressure_defect: Vector


def _state(reference, snapshot):
    value = _rebuild(snapshot)
    source = reference.source
    if any(getattr(value, field) != getattr(source, field) for field in (
        "nodes", "conductance", "support_neighbors", "capacity",
    )):
        raise ValueError("observation requires the reference support and capacity")
    pattern = _pattern(reference, value.epi)
    pressure = tuple(reference.epi_weight * g + f
                     for g, f in zip(value.epi_gradient, reference.forcing))
    defect = tuple(p - expected for p, expected in zip(value.stored_pressure, pressure))
    return ForcedSupportState(
        value, pattern.mean, pattern.relative_error,
        pattern.error_variance, pattern.error_dirichlet_energy, pressure, defect,
    )


def observe_forced_support_state(reference, snapshot) -> ForcedSupportState:
    """Compare a state with a rebuilt exact reference without graph writes.

    The phase and forcing constancy are caller assumptions. A pressure defect
    can represent numeric realization, stale pressure or another model; the
    observer never identifies these possibilities from stored pressure alone.
    """
    return _state(_reference(reference), snapshot)


@dataclass(frozen=True)
class ForcedSupportStep:
    """Exact mean and relative-coordinate accounting for one observed step."""

    reference: ForcedSupportBalance
    before: ForcedSupportState
    after: ForcedSupportState
    dt: Fraction
    support_budget: SupportTransportEuler
    relative_energy_budget: SupportTransportEuler
    mean_change: Fraction
    mean_model_change: Fraction
    mean_pressure_defect: Fraction
    mean_step_defect: Fraction
    mean_identity_residual: Fraction
    relative_recurrence_residual: Vector
    convex_step_admissible: bool


def observe_forced_support_step(
    reference, before, after, dt,
) -> ForcedSupportStep:
    """Separate frozen-model drift, pressure realization and endpoint defects.

    With delta=x_after-x_before-h*nu*p_stored and pressure defect epsilon,
    the H-mean residual equals h*sum(d*epsilon)/sum(H)+sum(H*delta)/sum(H).
    Centering h*nu*epsilon+delta gives the forcing of the relative error.
    The existing support Euler observer accounts for its Dirichlet energy
    on detached error coordinates; these are not graph state assignments.
    Its convex-step flag concerns only the ideal unforced error map.
    """
    ref = _reference(reference)
    before_state, after_state = _state(ref, before), _state(ref, after)
    h = exact_or_represented_real(dt, "dt")
    if h < 0:
        raise ValueError("dt must be nonnegative")
    support_budget = observe_support_transport_euler(
        before_state.snapshot, after_state.snapshot, h,
    )
    mass = sum(ref.metric_weights)
    mean_change = after_state.mean - before_state.mean
    mean_model = h * ref.mean_drift
    mean_pressure = h * dot(ref.strengths, before_state.pressure_defect) / mass
    mean_step = dot(ref.metric_weights, support_budget.state_defect) / mass
    mean_residual = mean_change - mean_model - mean_pressure - mean_step

    bu = _laplacian(ref.source.conductance, before_state.relative_error)
    error_pressure = tuple(-ref.epi_weight * value / d
                           for value, d in zip(bu, ref.strengths))
    error_before = replace(before_state.snapshot, epi=before_state.relative_error,
                           stored_pressure=error_pressure)
    error_after = replace(after_state.snapshot, epi=after_state.relative_error)
    error_budget = observe_support_transport_euler(error_before, error_after, h)
    combined_defect = tuple(
        h * nu * pressure_error + step_error
        for nu, pressure_error, step_error in zip(
            ref.source.capacity, before_state.pressure_defect,
            support_budget.state_defect,
        )
    )
    defect_mean = dot(ref.metric_weights, combined_defect) / mass
    recurrence_residual = tuple(
        observed - (error - defect_mean)
        for observed, error in zip(error_budget.state_defect, combined_defect)
    )
    if mean_residual or any(recurrence_residual):
        raise RuntimeError("exact mean or centered error recurrence lost its identity")
    return ForcedSupportStep(
        ref, before_state, after_state, h, support_budget, error_budget,
        mean_change, mean_model, mean_pressure, mean_step, mean_residual,
        recurrence_residual, h <= ref.max_convex_step,
    )


@dataclass(frozen=True)
class ForcedSupportResetEnergy:
    """Signed metric-first, then reference-shift decomposition of one energy."""

    metric_term: Fraction
    reference_cross_term: Fraction
    reference_quadratic_term: Fraction
    energy_change: Fraction
    identity_residual: Fraction


def _reset_energy(metric, cross, quadratic, change):
    residual = change - metric - cross - quadratic
    if residual:
        raise RuntimeError("exact change-of-reference energy identity was lost")
    return ForcedSupportResetEnergy(metric, cross, quadratic, change, residual)


@dataclass(frozen=True)
class ForcedSupportReset:
    """Same-EPI event comparison under two separately validated references."""

    before_reference: ForcedSupportBalance
    after_reference: ForcedSupportBalance
    before: ForcedSupportState
    after: ForcedSupportState
    raw_support_reset: SupportTransportReset
    error_support_reset: SupportTransportReset
    mean_reweighting: Fraction
    profile_shift: Vector
    error_shift: Vector
    drift_change: Fraction
    variance_budget: ForcedSupportResetEnergy
    dirichlet_budget: ForcedSupportResetEnergy


def observe_forced_support_reset(
    before_reference, after_reference, before, after,
) -> ForcedSupportReset:
    """Account for a new profile/metric without mistaking it for EPI recovery.

    References may have been derived earlier; actual event snapshots are
    therefore mandatory. Each snapshot must match its own held reference,
    and node order and EPI must agree across the reset. No pressure refresh,
    operator admission or causal link between these data is inferred.

    For u_j=x-mean_Hj(x)-z_j, du=u_1-u_0, the H-energy change is
    .5*u_0^T*(H_1-H_0)*u_0 + u_0^T*H_1*du + .5*du^T*H_1*du.
    Its B analogue reuses the shared conductance reset on detached u_0.
    Mean reweighting is a coordinate jump at unchanged EPI, not evolution.
    Terms depend on the declared metric-first decomposition and need not
    be negative. The fixed original-pattern observation itself is unchanged.
    """
    ref0, ref1 = _reference(before_reference), _reference(after_reference)
    state0, state1 = _state(ref0, before), _state(ref1, after)
    raw_reset = observe_support_transport_reset(state0.snapshot, state1.snapshot)
    u0, u1 = state0.relative_error, state1.relative_error
    du = tuple(right - left for left, right in zip(u0, u1))
    mean_jump = state1.mean - state0.mean
    dz = tuple(right - left for left, right in zip(
        ref0.relative_profile, ref1.relative_profile,
    ))
    if any(shift + mean_jump + profile_shift for shift, profile_shift in zip(du, dz)):
        raise RuntimeError("exact same-EPI reference shift lost its identity")
    h0, h1 = ref0.metric_weights, ref1.metric_weights
    variance = _reset_energy(
        dot(tuple(new - old for old, new in zip(h0, h1)), tuple(u**2 for u in u0)) / 2,
        dot(h1, tuple(u * shift for u, shift in zip(u0, du))),
        dot(h1, tuple(shift**2 for shift in du)) / 2,
        state1.error_variance - state0.error_variance,
    )
    error_reset = observe_support_transport_reset(
        replace(state0.snapshot, epi=u0), replace(state1.snapshot, epi=u0),
    )
    dirichlet = _reset_energy(
        error_reset.energy_change,
        dot(_laplacian(state1.snapshot.conductance, u0), du),
        _energy(state1.snapshot.conductance, du),
        state1.error_dirichlet_energy - state0.error_dirichlet_energy,
    )
    return ForcedSupportReset(
        ref0, ref1, state0, state1, raw_reset, error_reset, mean_jump, dz, du,
        ref1.mean_drift - ref0.mean_drift, variance, dirichlet,
    )
