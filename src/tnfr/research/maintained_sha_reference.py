"""Fixed-chart initial moving-reference audit for the retained SHA source.

This research consumer installs no dynamics and chooses no identity threshold
or response horizon. Source association and primitive admission belong to the
existing pair-budget owner. All calculations require a prospective instrument
freeze; an initial passing condition supplies no whole-window certificate.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from ..mathematics._rational_interval import (
    INTERVAL_METHOD,
    I,
    cos,
    pi_interval,
    sin,
    sqrt,
)
from .maintained_sha_motion import (
    PairBoundaryBudget,
    _bound_pair_budget,
    prepare_sha_pair_budget,
)

_ROOT_ITERATIONS = 80
_ROOT_WIDTH = Q(1, 2**70)


@dataclass(frozen=True)
class ReferenceGate:
    """A strict initial inequality; overlap means unavailable, never passing."""

    margin: I | None
    status: str
    reason: str | None = None


def _gate(margin, reason=None):
    if margin is None:
        return ReferenceGate(None, "unavailable", reason)
    status = (
        "pass" if margin.lo > 0 else "excluded" if margin.hi <= 0 else "unavailable"
    )
    return ReferenceGate(margin, status, reason if status == "unavailable" else None)


@dataclass(frozen=True)
class ReferenceRoot:
    """Radian root enclosure with a retained, bounded refinement history.

    Endpoint residuals refer to the original fixed chart. Refinement may use
    a derivative-based enclosing contraction at an unresolved midpoint; the
    final endpoints need not themselves have strictly resolved signs.
    """

    bracket: I | None
    original_endpoint_residuals: tuple[I, I]
    iterations: int
    status: str
    reason: str | None
    target_width: Q = _ROOT_WIDTH
    maximum_iterations: int = _ROOT_ITERATIONS


def _refine_root(force, curvature_lower):
    """Enclose one increasing force root; callers prove the derivative bound."""
    if not isinstance(curvature_lower, Q) or curvature_lower <= 0:
        raise ValueError("root contraction requires a positive exact curvature bound")
    left, right = Q(-1), Q(1)
    signs = force(left), force(right)
    if not signs[0].hi < 0 < signs[1].lo:
        return ReferenceRoot(None, signs, 0, "unavailable", "endpoint_signs_unresolved")
    for iteration in range(1, _ROOT_ITERATIONS + 1):
        middle = (left + right) / 2
        value = force(middle)
        old_width = right - left
        if value.hi < 0:
            left = middle
        elif value.lo > 0:
            right = middle
        else:
            # |q*-mid| <= |F(mid)|/m. Overlap never chooses a root side.
            radius = value.abs_max / curvature_lower
            left, right = max(left, middle - radius), min(right, middle + radius)
        bracket = I(left, right)
        if bracket.width <= _ROOT_WIDTH:
            return ReferenceRoot(bracket, signs, iteration, "available", None)
        if right - left >= old_width:
            return ReferenceRoot(
                bracket, signs, iteration, "unavailable", "root_refinement_stalled"
            )
    return ReferenceRoot(
        I(left, right),
        signs,
        _ROOT_ITERATIONS,
        "unavailable",
        "root_resolution_budget_exhausted",
    )


@dataclass(frozen=True)
class ReferenceChart:
    gamma: Q
    kappa: Q
    form_center: Q
    phase_center: Q
    external_form_coefficient: Q
    cosine_coefficient: I
    sine_coefficient: I
    form_reference: Q
    endpoint_force_margin: Q
    curvature_lower: Q
    curvature_upper: Q
    relative_curvature_lower: Q
    relative_curvature_upper: Q
    actual_phase_in_chart: bool
    status: str
    reasons: tuple[str, ...]
    root: ReferenceRoot | None
    chart: tuple[Q, Q] = (Q(-1), Q(1))


@dataclass(frozen=True)
class ReferenceState:
    """Initial reference geometry, retaining both actual and coarse clearance."""

    raw_relative_storage: I
    coercive_storage_enclosure: I
    relative_storage: I
    radial_excitation: I
    reference_radius_squared: I
    error_radius_squared: I
    reference_radius: I
    error_radius: I
    actual_reference_clearance: I
    coarse_reference_clearance: I
    geometric_offset_gate: ReferenceGate
    fixed_metric_offset_gate: ReferenceGate
    coarse_chart_gate: ReferenceGate
    actual_initial_clearance_gate: ReferenceGate
    zero_work_error_lower: I
    zero_work_error_upper: I
    zero_work_original_energy: I
    original_energy_lower_available: bool


@dataclass(frozen=True)
class ReferenceRates:
    """Initial rates, not future bounds; source rows retain their 64-bit widths."""

    center_form_rate: I
    center_phase_rate: I
    external_form_coefficient_rate: I
    cosine_coefficient_rate: I
    sine_coefficient_rate: I
    form_reference_rate: I
    phase_reference_rate: I
    work_norm: I


@dataclass(frozen=True)
class ReferenceWorkBudget:
    """Optimistic necessary budgets, not a selected horizon or rotation verdict.

    With the actual initial coefficient norm G these margins can exclude a
    uniform g_star >= G(t) and horizon for the registered Cauchy-work recipe.
    They do not exclude a sharper bound on the actual radial derivative or
    another work estimator. With a chosen global upper bound, they assess
    only that chosen bound package.
    """

    offset_gate: ReferenceGate
    signal_slack: I
    energy_slack: I
    offset_ratio: I | None
    optimistic_rotation_rate: I | None
    minimum_turn_time: I | None
    signal_work_gate: ReferenceGate
    energy_work_gate: ReferenceGate


@dataclass(frozen=True)
class ReferenceGlobalEnvelope:
    """Registered upper bounds, never lower bounds on actual offset or work."""

    form_coefficient_rate_bound: I
    phase_coefficient_rate_bound: I
    work_norm_bound: I
    reference_radius_squared_bound: I
    reference_radius_bound: I
    work_budget: ReferenceWorkBudget


@dataclass(frozen=True)
class ShaReferenceAudit:
    pair_budget: PairBoundaryBudget
    chart: ReferenceChart
    state: ReferenceState | None
    rates: ReferenceRates | None
    initial_work_budget: ReferenceWorkBudget | None
    global_envelope: ReferenceGlobalEnvelope | None
    unavailable_reasons: tuple[str, ...]
    arithmetic_method: str = INTERVAL_METHOD
    scope: tuple[str, ...] = (
        "fresh_primitive_post_sha_association_and_complete_pair_budget_law",
        "fixed_radian_chart_minus_one_to_one_universal_neighbor_count_curvature",
        "all_full_state_coordinates_means_and_original_clock_retained",
        "initial_passing_gates_do_not_bound_future_work_offset_or_retention",
        "chosen_upper_envelopes_are_not_actual_rate_or_offset_lower_bounds",
        "necessary_work_obstruction_requires_uniform_bound_on_registered_cauchy_norm_G",
        "no_alternate_chart_parameter_scan_trajectory_event_or_pressure_refresh",
    )


def _work_budget(*, excitation, offset, work, lower, upper, rotation):
    root_lower, root_upper = sqrt(I(lower)), sqrt(I(upper))
    offset_gate = _gate(
        lower * excitation - upper * root_upper * offset,
        "q_envelope_offset_sign_unresolved",
    )
    signal = excitation - upper * root_upper * offset / lower
    energy = root_lower * (2 - offset) - excitation
    ratio = root_upper * offset / excitation if excitation.lo > 0 else None
    omega = turn_time = None
    if offset_gate.status == "pass" and ratio is not None:
        candidate = rotation * (lower - upper * ratio) / (1 + ratio) ** 2
        if candidate.lo > 0:
            omega = candidate
            turn_time = 2 * pi_interval() / omega
    reason = "positive_optimistic_turn_rate_unavailable"
    return ReferenceWorkBudget(
        offset_gate,
        signal,
        energy,
        ratio,
        omega,
        turn_time,
        _gate(signal - work * turn_time if turn_time is not None else None, reason),
        _gate(energy - work * turn_time if turn_time is not None else None, reason),
    )


def _reference_from_fresh_budget(budget):
    """Internal calculation only: both entry points rebuild the budget first."""
    pair = budget.connected
    source = pair.source
    a, c = pair.parent_index, pair.child_index
    na = source.capacity[a] / source.degrees[a]
    nc = source.capacity[c] / source.degrees[c]
    gamma = na / (na + nc)  # Cancel pi exactly before interval arithmetic.
    mu = I(na * nc / (na + nc)) / pi_interval()
    w, sigma, n = (
        pair.reference_conductance,
        budget.external_conductance_sum,
        budget.external_degree,
    )
    r, delta = pair.form_difference, pair.phase_difference
    X, theta = source.epi[a] + gamma * r, source.phase[a] + gamma * delta
    kappa = w + gamma**2 * sigma
    positions = {node: i for i, node in enumerate(source.nodes)}
    strengths = [Q(0) for _ in source.nodes]
    external = {}
    for (left, right), weight in zip(source.edges, source.conductances, strict=True):
        i, j = positions[left], positions[right]
        strengths[i] += weight
        strengths[j] += weight
        if i == a and j != c:
            external[j] = weight
        elif j == a and i != c:
            external[i] = weight
    external = tuple((j, external[j]) for j in pair.external_indices)
    V0 = sum((weight * (X - source.epi[j]) for j, weight in external), Q(0))
    alpha = tuple(source.phase[j] - theta for j, _ in external)
    cosine, sine = tuple(cos(I(value)) for value in alpha), tuple(
        sin(I(value)) for value in alpha
    )
    C, S = sum(cosine, I(0)), sum(sine, I(0))
    rstar = gamma * V0 / kappa
    sin_one, cos_one = sin(I(1)), cos(I(1))
    eta, m, M = sin_one.lo - gamma * n, cos_one.lo - gamma**2 * n, 1 + gamma**2 * n
    lower, upper = min(m, kappa / w), max(M, kappa / w)
    reasons = []
    if eta <= 0:
        reasons.append("universal_endpoint_margin_not_positive")
    if m <= 0:
        reasons.append("universal_chart_curvature_not_positive")

    def force(q):
        return sin(I.coerce(q)) + gamma * (
            C * sin(gamma * I.coerce(q)) + S * cos(gamma * I.coerce(q))
        )

    def potential(q):
        q = I.coerce(q)
        return 1 - cos(q) + C * (1 - cos(gamma * q)) + S * sin(gamma * q)

    root = None if reasons else _refine_root(force, m)
    if root is not None and root.status != "available":
        reasons.append(root.reason)
    if not -1 <= delta <= 1:
        reasons.append("actual_initial_phase_outside_fixed_chart")
    chart = ReferenceChart(
        gamma,
        kappa,
        X,
        theta,
        V0,
        C,
        S,
        rstar,
        eta,
        m,
        M,
        lower,
        upper,
        -1 <= delta <= 1,
        "available" if not reasons else "unavailable",
        tuple(reasons),
        root,
    )
    if reasons:
        return ShaReferenceAudit(budget, chart, None, None, None, None, tuple(reasons))

    qstar = root.bracket
    u, v = r - rstar, I(delta) - qstar
    raw = I(kappa * u**2 / 2) + potential(I(delta)) - potential(qstar)
    coercive = I(kappa * u**2 / 2) + I(m, M) * v**2 / 2
    lo, hi = max(raw.lo, coercive.lo), min(raw.hi, coercive.hi)
    if lo > hi:
        raise ArithmeticError("independent relative-storage enclosures are disjoint")
    storage = I(lo, hi)
    excitation = sqrt(2 * storage)
    z_squared = qstar**2 + w * rstar**2
    e_squared = v**2 + w * u**2
    B, error = sqrt(z_squared), sqrt(e_squared)
    clearance, coarse_clearance = 1 - abs(qstar), I(eta / M)
    root_lower, root_upper = sqrt(I(lower)), sqrt(I(upper))
    e_min, e_max = excitation / root_upper, excitation / root_lower
    c_energy = (2 * sin(I(Q(1, 2)))) ** 2
    energy_upper = ((e_max + B) ** 2 / 2).hi
    energy_lower_available = (e_min - B).lo > 0
    energy_lower = (
        (c_energy * (e_min - B) ** 2 / 2).lo if energy_lower_available else Q(0)
    )
    state = ReferenceState(
        raw,
        coercive,
        storage,
        excitation,
        z_squared,
        e_squared,
        B,
        error,
        clearance,
        coarse_clearance,
        _gate(e_squared - z_squared, "geometric_offset_sign_unresolved"),
        _gate(
            lower**2 * e_squared - upper**2 * z_squared,
            "fixed_metric_offset_sign_unresolved",
        ),
        _gate(
            sqrt(I(m)) * coarse_clearance - excitation, "coarse_chart_margin_unresolved"
        ),
        _gate(
            sqrt(I(m)) * clearance - excitation, "initial_clearance_margin_unresolved"
        ),
        e_min,
        e_max,
        I(energy_lower, energy_upper),
        energy_lower_available,
    )
    Xdot, theta_dot = mu * pair.external_sine_sum, mu * pair.external_form_gradient
    Vdot = sigma * Xdot - sum(
        (weight * source.form_rates[j] for j, weight in external), I(0)
    )
    alpha_dot = tuple(source.phase_rates[j] - theta_dot for j, _ in external)
    Cdot = -sum((s * rate for s, rate in zip(sine, alpha_dot, strict=True)), I(0))
    Sdot = sum((c * rate for c, rate in zip(cosine, alpha_dot, strict=True)), I(0))
    hstar_raw = cos(qstar) + gamma**2 * (
        C * cos(gamma * qstar) - S * sin(gamma * qstar)
    )
    hstar = I(max(m, hstar_raw.lo), min(M, hstar_raw.hi))
    rstar_dot = gamma * Vdot / kappa
    qstar_dot = -gamma * (Cdot * sin(gamma * qstar) + Sdot * cos(gamma * qstar)) / hstar
    work_norm = gamma * sqrt(Vdot**2 / kappa + (Cdot**2 + Sdot**2) / m)
    rates = ReferenceRates(
        Xdot, theta_dot, Vdot, Cdot, Sdot, rstar_dot, qstar_dot, work_norm
    )
    work = _work_budget(
        excitation=excitation,
        offset=B,
        work=work_norm,
        lower=lower,
        upper=upper,
        rotation=pair.internal_rotation_coefficient,
    )
    mobilities = tuple(
        I(nu / degree) / pi_interval()
        for nu, degree in zip(source.capacity, source.degrees, strict=True)
    )
    P_V = sigma * mu * n + sum(
        (weight * mobilities[j] * source.degrees[j] for j, weight in external), I(0)
    )
    P_phi = sqrt(2 * source.storage) * (
        sum((mobilities[j] * sqrt(I(strengths[j])) for j, _ in external), I(0))
        + n * mu * sqrt(I(sigma))
    )
    global_work = gamma * sqrt(P_V**2 / kappa + P_phi**2 / m)
    global_B_squared = (
        gamma * n / sin_one
    ) ** 2 + 2 * source.storage * gamma**2 * sigma / kappa
    global_B = sqrt(global_B_squared)
    envelope = ReferenceGlobalEnvelope(
        P_V,
        P_phi,
        global_work,
        global_B_squared,
        global_B,
        _work_budget(
            excitation=excitation,
            offset=global_B,
            work=global_work,
            lower=lower,
            upper=upper,
            rotation=pair.internal_rotation_coefficient,
        ),
    )
    return ShaReferenceAudit(budget, chart, state, rates, work, envelope, ())


def _bound_reference(graph, *, parent, child, reference_model):
    """Manufactured fixed controls rebuild their complete primitive budget."""
    return _reference_from_fresh_budget(
        _bound_pair_budget(
            graph, parent=parent, child=child, reference_model=reference_model
        )
    )


def prepare_sha_reference(admitted):
    """Re-admit the actual retained SHA source and audit only its initial state.

    No supplied cached budget, reference, root or verdict is accepted. The
    shared source wrapper validates persistent association and exact clock.
    Calling this is one static scientific evaluation requiring its own freeze.
    """
    return _reference_from_fresh_budget(prepare_sha_pair_budget(admitted))
