"""Retrospective signed-transport bounds on the unchanged retained SHA cells.

The retained reader owns byte association, the complete held law, cell joins,
Picard inclusion and recorded Taylor arithmetic. Its frozen derivative and
remainder provenance is preserved, not independently regenerated here. This
consumer observes the single fixed half-unit window and installs no dynamics.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from fractions import Fraction as Q

from ..mathematics._interval_taylor import Jet, sinc
from ..mathematics._rational_interval import (
    INTERVAL_METHOD,
    I,
    cos,
    pi_interval,
    sin,
    sqrt,
)
from ..physics.relational_observations import _ordered
from .local_capacity_history import MODEL
from .maintained_parent_thol import _signature
from .maintained_principal_retention import HORIZON, STEP, STEP_COUNT
from .maintained_sha_box_budget import BoxCoefficientRows, _coefficient_rows
from .maintained_sha_reference import (
    ReferenceGate,
    ShaReferenceAudit,
    _bound_reference,
    _gate,
)
from .maintained_sha_retention_evidence import ShaRetentionEvidence, load_sha_retention


@dataclass(frozen=True)
class SmoothDividedDifferences:
    h: I
    a: I
    b: I


@dataclass(frozen=True)
class CoefficientObservation:
    V0: I
    C: I
    S: I
    Vdot: I
    Vddot: I
    Cdot: I
    Sdot: I
    Cddot: I
    Sddot: I
    alpha_rates: tuple[I, ...]
    alpha_accelerations: tuple[I, ...]


@dataclass(frozen=True)
class InitialSignedTransport:
    coefficients: CoefficientObservation
    divided_differences: SmoothDividedDifferences | None
    P0: I | None
    C0: I | None
    corrected_storage_gate: ReferenceGate
    hidden_storage: I | None
    storage_identity_residual: I | None
    storage_identity_consistent: bool | None
    unavailable_reasons: tuple[str, ...]


@dataclass(frozen=True)
class SignedTransportCell:
    index: int
    time: Q
    duration: Q
    V0: I
    C: I
    S: I
    Vdot: I
    Vddot: I
    Cdot: I
    Sdot: I
    Cddot: I
    Sddot: I
    alpha_rates: tuple[I, ...]
    alpha_accelerations: tuple[I, ...]
    V1: Q
    V2: Q
    R0: Q
    R1: Q
    R2: Q
    Pdom: Q
    epsilon: I | None
    eta_R: I | None
    quotient_bounds: tuple[tuple[str, Q], ...]
    delta: I
    actual_phase_in_chart: bool
    form_mean: I
    phase_mean: I
    means_consistent: tuple[bool, bool]
    form_coefficient_displacement: Q
    phase_coefficient_displacement: Q
    unavailable_reasons: tuple[str, ...]


@dataclass(frozen=True)
class ExcitationEnvelope:
    C0: I
    epsilon: Q
    remainder_integral: Q
    radicand: I
    upper: I
    D0: I
    lower: I
    floor_gate: ReferenceGate


@dataclass(frozen=True)
class SignedResponseBounds:
    reference_phase_displacement: I | None
    reference_displacement: I | None
    reference_radius_upper: Q | None
    error_radius_lower: Q | None
    error_radius_upper: Q | None
    original_radius_lower: I | None
    original_energy: I | None
    original_energy_lower_available: bool
    offset_ratio: I | None
    rotation_rate: I | None
    chart_gate: ReferenceGate
    signal_gate: ReferenceGate
    offset_gate: ReferenceGate
    energy_upper_gate: ReferenceGate
    turn_gate: ReferenceGate
    whole_cell_chart_admitted: bool
    registered_conditions_met: bool
    unavailable_reasons: tuple[str, ...]


@dataclass(frozen=True)
class SignedTransportAggregate:
    full_window: bool
    accepted_cells: int
    accepted_duration: Q
    epsilon: Q | None
    remainder_integral: Q | None
    form_displacement: Q | None
    phase_displacement: Q | None
    envelope: ExcitationEnvelope | None
    response: SignedResponseBounds | None
    unavailable_reasons: tuple[str, ...]


@dataclass(frozen=True)
class ShaSignedTransportAudit:
    reference: ShaReferenceAudit
    rows: BoxCoefficientRows
    initial: InitialSignedTransport
    cells: tuple[SignedTransportCell, ...]
    aggregate: SignedTransportAggregate
    source_receipts: tuple[dict, ...]
    retained_receipts: tuple[dict, ...]
    retained_summary: dict
    persistent_source_unchanged: bool
    arithmetic_method: str = INTERVAL_METHOD
    scope: tuple[str, ...] = (
        "retrospective_observation_of_one_unchanged_retained_256_cell_half_unit_window",
        "shared_reader_rebuilds_source_association_Picard_inclusion_and_recorded_arithmetic",
        "retained_Taylor_derivatives_and_remainders_keep_frozen_producer_provenance",
        "new_coefficient_derivatives_use_complete_signed_rows_and_shared128bit_intervals",
        "initial_root_and_excitation_use_existing_reference_owner_and_its_inherited_widths",
        "both_weighted_means_checked_without_narrowing_or_projecting_retained_boxes",
        "uniform_correction_and_integrated_remainder_follow_the_registered_section13_recipe",
        "no_raw_division_by_relative_phase_error_or_positive_corrected_storage_assumption",
        "no_prefix_horizon_substitution_identity_threshold_selection_or_matched_cut_claim",
        "no_new_trajectory_solver_call_event_pressure_refresh_or_source_replacement",
    )


def _nonnegative_constant(value, label):
    if type(value) is not int and not isinstance(value, Q):
        raise TypeError(f"{label} requires an exact rational constant")
    value = Q(value)
    if value < 0:
        raise ValueError(f"{label} must be nonnegative")
    return value


def _intersection(value, lower, upper, label):
    lo, hi = max(value.lo, lower), min(value.hi, upper)
    if lo > hi:
        raise ArithmeticError(f"{label} has disjoint independently proved enclosures")
    return I(lo, hi)


def _smooth_divided_differences(delta, root, *, gamma, C, S, Cdot, Sdot, lower, upper):
    """Enclose h,a,b by shared smooth sinc, including every q=0 interval.

    The caller proves criticality of root and the uniform curvature bounds.
    Intersecting h with those bounds retains that theorem; an inconsistent
    intersection raises rather than manufacturing a usable denominator.
    """
    delta, root = I.coerce(delta), I.coerce(root)
    gamma = _nonnegative_constant(gamma, "gamma")
    lower = _nonnegative_constant(lower, "curvature lower bound")
    upper = _nonnegative_constant(upper, "curvature upper bound")
    if not 0 < gamma < 1 or not 0 < lower <= upper:
        raise ValueError("require 0<gamma<1 and positive ordered curvature bounds")
    if any(value.lo < -1 or value.hi > 1 for value in (delta, root)):
        raise ValueError("both phases must be enclosed in the fixed [-1,1] chart")
    C, S, Cdot, Sdot = map(I.coerce, (C, S, Cdot, Sdot))
    q, middle = delta - root, (delta + root) / 2

    def continuous_sinc(value):
        return sinc(Jet.constant(value, 0)).coeffs[0]

    small = continuous_sinc(gamma * q / 2)
    h_raw = (
        cos(middle) * continuous_sinc(q / 2)
        + gamma**2 * (C * cos(gamma * middle) - S * sin(gamma * middle)) * small
    )
    h = _intersection(h_raw, lower, upper, "divided curvature")
    a = gamma * (Cdot * sin(gamma * middle) + Sdot * cos(gamma * middle)) * small
    return SmoothDividedDifferences(h, a, a / h)


def _complete_box(source, values):
    values = _ordered(
        values, "complete retained state", limit=2 * len(source.nodes) + 1
    )
    if len(values) != 2 * len(source.nodes):
        raise ValueError("state must retain every form and primitive phase coordinate")
    return tuple(map(I.coerce, values))


def _dot(coefficients, values):
    return sum(
        (
            coefficient * value
            for coefficient, value in zip(coefficients, values, strict=True)
        ),
        I(0),
    )


def _linear_coefficients(reference, rows):
    pair, chart = reference.pair_budget.connected, reference.chart
    a, c = pair.parent_index, pair.child_index
    ell = list(rows.external_port)
    transfer = chart.gamma * reference.pair_budget.external_conductance_sum
    ell[a] -= transfer
    ell[c] += transfer
    error = [-chart.gamma * value / chart.kappa for value in ell]
    error[a] -= 1
    error[c] += 1
    return tuple(ell), tuple(error)


def _coefficient_observation(reference, rows, box):
    """Complete instantaneous/whole-box derivatives, with no field callback.

    Rational signed incidence coefficients are combined before multiplication
    by interval sine, cosine or the common mathematical-pi divisor. This
    retains the internal-edge cancellation and every remote environmental row.
    """
    pair, chart = reference.pair_budget.connected, reference.chart
    source, gamma = pair.source, chart.gamma
    values = _complete_box(source, box)
    size = len(source.nodes)
    forms, phases = values[:size], values[size:]
    a, c = pair.parent_index, pair.child_index
    ell, _ = _linear_coefficients(reference, rows)
    positions = {node: index for index, node in enumerate(source.nodes)}
    edges = tuple((positions[left], positions[right]) for left, right in source.edges)
    gaps = tuple(phases[i] - phases[j] for i, j in edges)
    sines, cosines = tuple(map(sin, gaps)), tuple(map(cos, gaps))
    pi = pi_interval()
    g = tuple(rows.beta[i] * ell[i] - rows.beta[j] * ell[j] for i, j in edges)
    phase_edge_rows = tuple(
        tuple(
            rows.beta[i] * left - rows.beta[j] * right
            for left, right in zip(
                rows.weighted_laplacian[i], rows.weighted_laplacian[j], strict=True
            )
        )
        for i, j in edges
    )
    Vdot = -_dot(g, sines) / pi
    Vddot = (
        -sum(
            (
                coefficient * cosine * _dot(edge_row, forms)
                for coefficient, cosine, edge_row in zip(
                    g, cosines, phase_edge_rows, strict=True
                )
            ),
            I(0),
        )
        / pi**2
    )
    alpha = tuple(
        phases[j] - ((1 - gamma) * phases[a] + gamma * phases[c])
        for j in rows.external_indices
    )
    phase_cosines, phase_sines = tuple(map(cos, alpha)), tuple(map(sin, alpha))
    alpha_rates = tuple(_dot(row, forms) / pi for row in rows.phase_rate_numerators)
    alpha_accelerations = tuple(
        -_dot(
            tuple(row[i] * rows.beta[i] - row[j] * rows.beta[j] for i, j in edges),
            sines,
        )
        / pi**2
        for row in rows.phase_rate_numerators
    )
    Cdot = -sum(
        (sine * rate for sine, rate in zip(phase_sines, alpha_rates, strict=True)), I(0)
    )
    Sdot = sum(
        (
            cosine * rate
            for cosine, rate in zip(phase_cosines, alpha_rates, strict=True)
        ),
        I(0),
    )
    Cddot = -sum(
        (
            cosine * rate**2 + sine * acceleration
            for cosine, sine, rate, acceleration in zip(
                phase_cosines,
                phase_sines,
                alpha_rates,
                alpha_accelerations,
                strict=True,
            )
        ),
        I(0),
    )
    Sddot = sum(
        (
            -sine * rate**2 + cosine * acceleration
            for cosine, sine, rate, acceleration in zip(
                phase_cosines,
                phase_sines,
                alpha_rates,
                alpha_accelerations,
                strict=True,
            )
        ),
        I(0),
    )
    return CoefficientObservation(
        _dot(ell, forms),
        sum(phase_cosines, I(0)),
        sum(phase_sines, I(0)),
        Vdot,
        Vddot,
        Cdot,
        Sdot,
        Cddot,
        Sddot,
        alpha_rates,
        alpha_accelerations,
    )


def _initial_transport(reference, rows):
    pair, chart = reference.pair_budget.connected, reference.chart
    source = pair.source
    observation = _coefficient_observation(reference, rows, source.epi + source.phase)
    reasons = list(reference.unavailable_reasons)
    if not chart.actual_phase_in_chart:
        reasons.append("initial_phase_outside_fixed_chart")
    if reference.state is None or chart.root is None or chart.root.bracket is None:
        reasons.append("initial_reference_state_unavailable")
    if reasons:
        return InitialSignedTransport(
            observation,
            None,
            None,
            None,
            _gate(None, "initial_signed_correction_unavailable"),
            None,
            None,
            None,
            tuple(dict.fromkeys(reasons)),
        )
    root = chart.root.bracket
    divided = _smooth_divided_differences(
        pair.phase_difference,
        root,
        gamma=chart.gamma,
        C=observation.C,
        S=observation.S,
        Cdot=observation.Cdot,
        Sdot=observation.Sdot,
        lower=chart.curvature_lower,
        upper=chart.curvature_upper,
    )
    k = I(rows.beta[pair.parent_index] + rows.beta[pair.child_index]) / pi_interval()
    rstar_rate = chart.gamma * observation.Vdot / chart.kappa
    q, p = I(pair.phase_difference) - root, pair.form_difference - chart.form_reference
    P0 = -(q * rstar_rate + p * divided.b) / k
    C0 = 2 * (reference.state.relative_storage - P0)

    # Independent algebraic storage consistency only. It is not used to
    # improve an unavailable correction, excitation or remainder estimate.
    positions = {node: index for index, node in enumerate(source.nodes)}
    a, c = pair.parent_index, pair.child_index
    hidden = I(0)
    variance = Q(0)
    sigma = reference.pair_budget.external_conductance_sum
    for (left, right), weight in zip(source.edges, source.conductances, strict=True):
        i, j = positions[left], positions[right]
        if a not in (i, j) and c not in (i, j):
            hidden += (
                I(weight * (source.epi[i] - source.epi[j]) ** 2 / 2)
                + 1
                - cos(I(source.phase[i] - source.phase[j]))
            )
        elif a in (i, j) and c not in (i, j):
            other = j if i == a else i
            variance += (
                weight
                * (
                    chart.form_center
                    - source.epi[other]
                    - chart.external_form_coefficient / sigma
                )
                ** 2
                / 2
            )
    hidden += I(variance)
    Wstar = (
        1
        - cos(root)
        + observation.C * (1 - cos(chart.gamma * root))
        + observation.S * sin(chart.gamma * root)
    )
    psi = (
        I(
            pair.reference_conductance
            * chart.external_form_coefficient**2
            / (2 * sigma * chart.kappa)
        )
        + reference.pair_budget.external_degree
        - observation.C
        + Wstar
    )
    residual = reference.state.relative_storage - (source.storage - hidden - psi)
    return InitialSignedTransport(
        observation,
        divided,
        P0,
        C0,
        _gate(C0, "initial_corrected_storage_sign_unresolved"),
        hidden,
        residual,
        residual.contains(0),
        (),
    )


def _bound_cell(reference, rows, tube, *, index=0, time=Q(0), duration=STEP):
    """Observe one unchanged full-state enclosure; no chart/root search.

    The global n-based chart admits the reference throughout this box. A box
    not contained in the fixed phase chart, or inconsistent with a conserved
    mean, leaves its correction/remainder constants unavailable. No box is
    narrowed to a mean leaf and no shorter horizon is selected.
    """
    if type(index) is not int or index < 0:
        raise ValueError("cell index must be a nonnegative ordinary integer")
    time = _nonnegative_constant(time, "cell time")
    duration = _nonnegative_constant(duration, "cell duration")
    if duration <= 0:
        raise ValueError("cell duration must be positive")
    pair, chart = reference.pair_budget.connected, reference.chart
    source = pair.source
    values = _complete_box(source, tube)
    size = len(source.nodes)
    forms, phases = values[:size], values[size:]
    observation = _coefficient_observation(reference, rows, values)
    ell, error = _linear_coefficients(reference, rows)
    delta = phases[pair.child_index] - phases[pair.parent_index]
    in_chart = delta.lo >= -1 and delta.hi <= 1
    weights = tuple(
        Q(degree) / capacity
        for degree, capacity in zip(source.degrees, source.capacity, strict=True)
    )
    weight_sum = sum(weights, Q(0))
    form_mean, phase_mean = (
        _dot(weights, forms) / weight_sum,
        _dot(weights, phases) / weight_sum,
    )
    consistent = (
        form_mean.contains(pair.form_mean),
        phase_mean.contains(pair.phase_lift_mean),
    )
    displacement_V = _dot(
        ell,
        tuple(
            value - initial for value, initial in zip(forms, source.epi, strict=True)
        ),
    ).abs_max
    displacement_CS = sqrt(
        (observation.C - chart.cosine_coefficient) ** 2
        + (observation.S - chart.sine_coefficient) ** 2
    ).hi
    V1, V2 = observation.Vdot.abs_max, observation.Vddot.abs_max
    R0 = Q(reference.pair_budget.external_degree)
    R1 = sqrt(observation.Cdot**2 + observation.Sdot**2).hi
    R2 = sqrt(observation.Cddot**2 + observation.Sddot**2).hi
    Pdom = _dot(error, forms).abs_max
    reasons = []
    if chart.endpoint_force_margin <= 0 or chart.curvature_lower <= 0:
        reasons.append("uniform_reference_chart_unavailable")
    if not in_chart:
        reasons.append("cell_phase_not_enclosed_in_fixed_chart")
    if not all(consistent):
        reasons.append("cell_does_not_enclose_both_conserved_means")
    epsilon = eta_R = None
    constants = ()
    if not reasons:
        gamma, kappa, m = chart.gamma, chart.kappa, chart.curvature_lower
        k = (
            I(rows.beta[pair.parent_index] + rows.beta[pair.child_index])
            / pi_interval()
        )
        Ddelta = max((k * kappa * Pdom).hi, gamma * R1 / m)
        H2 = 1 + gamma**3 * R0
        H1 = H2 * Ddelta + gamma**2 * R1
        A1 = gamma**2 * R1 * Ddelta + gamma * R2
        Bb = gamma * R1 / m
        Db = gamma**2 * R1 / (2 * m) + gamma * R1 * H2 / (2 * m**2)
        B1 = A1 / m + gamma * R1 * H1 / m**2
        epsilon = sqrt(I(gamma**2 * V1**2 / (kappa**2 * m) + Bb**2 / kappa)) / k
        eta_R = (
            sqrt(I(gamma**2 * (V2 + V1 * Db) ** 2 / (kappa**2 * m) + B1**2 / kappa)) / k
        )
        constants = tuple(
            zip(
                ("Ddelta", "H2", "H1", "A1", "Bb", "Db", "B1"),
                (Ddelta, H2, H1, A1, Bb, Db, B1),
            )
        )
    return SignedTransportCell(
        index,
        time,
        duration,
        observation.V0,
        observation.C,
        observation.S,
        observation.Vdot,
        observation.Vddot,
        observation.Cdot,
        observation.Sdot,
        observation.Cddot,
        observation.Sddot,
        observation.alpha_rates,
        observation.alpha_accelerations,
        V1,
        V2,
        R0,
        R1,
        R2,
        Pdom,
        epsilon,
        eta_R,
        constants,
        delta,
        in_chart,
        form_mean,
        phase_mean,
        consistent,
        displacement_V,
        displacement_CS,
        tuple(reasons),
    )


def _excitation_envelope(Q0_squared, P0, *, epsilon, remainder_integral):
    """Quadratic transport envelope, retaining the signed initial correction.

    The caller proves |P0|<=epsilon*Q0 through the same domain Cauchy bound.
    Evaluate the upper radicand as one exact rational before enclosing it.
    Its nonnegativity follows from that premise; no clipped interval or
    positive-corrected-storage assumption replaces the proof. The lower
    root uses its stable rationalized expression when its exact margin is
    positive, while D0 retains both interval endpoints for status reporting.
    """
    Q0_squared, P0 = I.coerce(Q0_squared), I.coerce(P0)
    epsilon = _nonnegative_constant(epsilon, "uniform correction bound")
    remainder_integral = _nonnegative_constant(remainder_integral, "remainder integral")
    if Q0_squared.lo < 0:
        raise ValueError("initial excitation squared must be nonnegative")
    C0 = Q0_squared - 2 * P0
    A = epsilon + remainder_integral
    upper_radicand = A**2 + C0.hi
    if upper_radicand < 0:
        raise ArithmeticError(
            "initial Cauchy premise and quadratic radicand are inconsistent"
        )
    radicand = I(upper_radicand)
    upper = A + sqrt(radicand)
    spent = 2 * remainder_integral * upper.hi
    Dlo = C0.lo - spent
    D0 = I(Dlo, C0.hi - spent)
    lower = I(Dlo) / (sqrt(I(epsilon**2 + Dlo)) + epsilon) if Dlo > 0 else I(0)
    return ExcitationEnvelope(
        C0,
        epsilon,
        remainder_integral,
        radicand,
        upper,
        D0,
        lower,
        _gate(D0, "strict_positive_excitation_floor_unresolved"),
    )


def _unavailable_response(reason):
    gate = _gate(None, reason)
    return SignedResponseBounds(
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        False,
        None,
        None,
        gate,
        gate,
        gate,
        gate,
        gate,
        False,
        False,
        (reason,),
    )


def _response_bounds(
    reference,
    envelope,
    *,
    form_displacement,
    phase_displacement,
    full_window,
    actual_chart_admitted=False,
):
    if type(full_window) is not bool:
        raise TypeError("full_window must be Boolean")
    if type(actual_chart_admitted) is not bool:
        raise TypeError("actual_chart_admitted must be Boolean")
    if not full_window:
        return _unavailable_response("fixed_full_window_unavailable")
    if reference.state is None:
        return _unavailable_response("initial_reference_state_unavailable")
    DV = _nonnegative_constant(form_displacement, "form coefficient displacement")
    DCS = _nonnegative_constant(phase_displacement, "phase coefficient displacement")
    chart, state, pair = (
        reference.chart,
        reference.state,
        reference.pair_budget.connected,
    )
    gamma, kappa, m = chart.gamma, chart.kappa, chart.curvature_lower
    lower, upper = chart.relative_curvature_lower, chart.relative_curvature_upper
    if min(m, lower, upper) <= 0:
        return _unavailable_response("positive_reference_metric_unavailable")
    dtheta = I(gamma * DCS / m)
    dz = gamma * sqrt(I(pair.reference_conductance * DV**2 / kappa**2 + DCS**2 / m**2))
    B = (state.reference_radius + dz).hi
    emin = (envelope.lower / sqrt(I(upper))).lo
    emax = (envelope.upper / sqrt(I(lower))).hi
    radius = I(emin - B)
    chart_gate = _gate(
        sqrt(I(m)) * (state.actual_reference_clearance - dtheta) - envelope.upper.hi,
        "signed_chart_margin_unresolved",
    )
    signal_gate = _gate(radius, "original_signal_margin_unresolved")
    offset_gate = _gate(
        I(lower * emin - upper * B), "original_offset_margin_unresolved"
    )
    energy_upper = (I(emax + B) ** 2 / 2).hi
    energy_lower_available = radius.lo > 0
    c_energy = (2 * sin(I(Q(1, 2)))) ** 2
    energy_lower = (c_energy * radius**2 / 2).lo if energy_lower_available else Q(0)
    energy = I(energy_lower, energy_upper)
    energy_gate = _gate(I(2 - energy_upper), "subseparatrix_upper_margin_unresolved")
    ratio = I(B) / emin if emin > 0 else None
    rotation = None
    if offset_gate.status == "pass" and ratio is not None:
        candidate = (
            pair.internal_rotation_coefficient
            * (lower - upper * ratio)
            / (1 + ratio) ** 2
        )
        if candidate.lo > 0:
            rotation = candidate
    turn_gate = _gate(
        HORIZON * rotation - 2 * pi_interval() if rotation is not None else None,
        "positive_original_rotation_bound_unavailable",
    )
    # The unchanged whole cells already prove their actual chart premise.
    # The displacement/Q chart test and E<2 are separate diagnostics; neither
    # is an independently selected identity threshold or a necessary turn gate.
    gates = (envelope.floor_gate, signal_gate, offset_gate, turn_gate)
    if not actual_chart_admitted:
        gates += (chart_gate,)
    conditions = all(gate.status == "pass" for gate in gates)
    reasons = tuple(gate.reason for gate in gates if gate.reason is not None)
    return SignedResponseBounds(
        dtheta,
        dz,
        B,
        emin,
        emax,
        radius,
        energy,
        energy_lower_available,
        ratio,
        rotation,
        chart_gate,
        signal_gate,
        offset_gate,
        energy_gate,
        turn_gate,
        actual_chart_admitted,
        conditions,
        reasons,
    )


def _aggregate(reference, initial, cells, *, full_window):
    if type(full_window) is not bool:
        raise TypeError("full_window must be Boolean")
    cells = tuple(cells)
    duration = sum((cell.duration for cell in cells), Q(0))
    full_window = full_window and len(cells) == STEP_COUNT and duration == HORIZON
    reasons = []
    if not full_window:
        reasons.append("fixed_256_cell_half_unit_window_unavailable")
    if initial.P0 is None or reference.state is None:
        reasons.append("initial_signed_correction_unavailable")
    if initial.storage_identity_consistent is False:
        reasons.append("initial_storage_identity_enclosures_disjoint")
    if any(cell.unavailable_reasons for cell in cells):
        reasons.append("one_or_more_cell_domains_unavailable")
    if reasons:
        return SignedTransportAggregate(
            full_window,
            len(cells),
            duration,
            None,
            None,
            None,
            None,
            None,
            None,
            tuple(reasons),
        )
    epsilon = max(cell.epsilon.hi for cell in cells)
    remainder = sum((cell.duration * cell.eta_R.hi for cell in cells), Q(0))
    DV = max(cell.form_coefficient_displacement for cell in cells)
    DCS = max(cell.phase_coefficient_displacement for cell in cells)
    envelope = _excitation_envelope(
        2 * reference.state.relative_storage,
        initial.P0,
        epsilon=epsilon,
        remainder_integral=remainder,
    )
    response = _response_bounds(
        reference,
        envelope,
        form_displacement=DV,
        phase_displacement=DCS,
        full_window=True,
        actual_chart_admitted=all(cell.actual_phase_in_chart for cell in cells),
    )
    return SignedTransportAggregate(
        True, len(cells), duration, epsilon, remainder, DV, DCS, envelope, response, ()
    )


def _audit_from_fresh_retention(evidence):
    """Internal only: the public boundary reloads the retained primitive files."""
    if type(evidence) is not ShaRetentionEvidence:
        raise TypeError("fresh shared retained SHA evidence required")
    graph = evidence.admitted.graph
    signature = _signature(graph)
    reference = _bound_reference(
        graph, parent=2, child="2_sub_0", reference_model=MODEL
    )
    source = reference.pair_budget.connected.source
    expected_initial = tuple(map(I, source.epi + source.phase))
    if (
        evidence.initial != expected_initial
        or evidence.prediction.nodes != source.nodes
    ):
        raise ValueError(
            "retained initial point or node association differs from fresh source"
        )
    rows = _coefficient_rows(reference)
    initial = _initial_transport(reference, rows)
    cells, carried = [], expected_initial
    for index, step in enumerate(evidence.steps):
        if (
            step.index != index
            or step.time != index * STEP
            or step.duration != STEP
            or step.initial != carried
        ):
            raise ValueError(
                "retained chronological endpoint carry changed after admission"
            )
        cells.append(
            _bound_cell(
                reference,
                rows,
                step.tube,
                index=index,
                time=step.time,
                duration=step.duration,
            )
        )
        carried = step.endpoint
    full_window = len(cells) == STEP_COUNT
    aggregate = _aggregate(reference, initial, cells, full_window=full_window)
    if _signature(graph) != signature:
        raise ValueError("retrospective observation changed persistent source state")
    return ShaSignedTransportAudit(
        reference,
        rows,
        initial,
        tuple(cells),
        aggregate,
        deepcopy(evidence.admitted.receipts),
        deepcopy(evidence.receipts),
        deepcopy(evidence.summary),
        True,
    )


def audit_sha_signed_transport(evidence_dir, *, repository_root):
    """Re-admit the fixed retained attempt once, then observe its unchanged cells.

    Calling this requires its own frozen retrospective protocol and instrument.
    Supplied cached evidence, roots and pass flags never replace the reader.
    No source replay, post-SHA solver, event or continuation is invoked.
    """
    return _audit_from_fresh_retention(
        load_sha_retention(evidence_dir, repository_root=repository_root)
    )
