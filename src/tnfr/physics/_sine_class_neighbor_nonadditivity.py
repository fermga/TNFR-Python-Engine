"""Conditional mixed interventions at distinct donor and receiver ports.

The mediator observation compares four histories with one identical complete
initial state. Structural onset, source accounting and simultaneous event work
are response-free. An onset coefficient alone never supplies a finite sign.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from ..mathematics._rational_interval import I
from ._sine_class_collective_interface import (
    _GAMMA_UPPER,
    _MAX_INPUT_VARIATION,
    _bound_collective_interface,
    _CollectiveInterfaceBound,
)
from ._sine_class_contrast import _contrast_decision, _ContrastDecision
from .relational_sine_class_cubic_response import _cubic_parameters
from .relational_sine_class_mediation import _EDGES
from .relational_sine_class_memory import _normalized_laplacian
from .relational_sine_class_superposition import (
    _admit_probe_primitives,
    _ClassProbeEventLedger,
    _probe_event_ledger,
)


@dataclass(frozen=True)
class _NeighborOnset:
    """Time-four cubic factors rebuilt from the actual normalized edge rows.

    Entries in each channel are the coefficients of a**2*b and a*b**2 before
    multiplication by its target cosine and the common gamma**4. The factors
    are exact rationals; trigonometric coefficients remain outward bounds of
    the named law, not a rational replacement of that law.
    """

    mediator_class: int
    gamma_bounds: I
    channel_cosine_bounds: tuple[I, ...]
    mixed_channel_factors: tuple[tuple[Q, Q], ...]
    degrees: tuple[int, ...]
    donor_phase_velocity_column: tuple[Q, ...]
    receiver_phase_velocity_column: tuple[Q, ...]
    donor_phase_curvature_column: tuple[Q, ...]
    receiver_phase_curvature_column: tuple[Q, ...]
    channel_order: tuple[str, ...] = (
        "donor_internal",
        "mediator_internal",
        "receiver_internal",
        "contacts",
    )
    intervention_nodes: tuple[int, int] = (4, 22)
    readout_node: int = 13
    time_degree: int = 4
    amplitude_degree: int = 3


def _derive_neighbor_onset(mediator_class) -> _NeighborOnset:
    """Use first phase velocities and oriented cubic edge currents, not a solver."""
    parameters = _cubic_parameters(mediator_class)
    form = _normalized_laplacian(_EDGES, parameters.degrees)
    donor, receiver = tuple(row[4] for row in form), tuple(row[22] for row in form)
    channels = [[Q(0), Q(0)] for _ in range(4)]
    for index, (i, j) in enumerate(_EDGES):
        left = donor[j] - donor[i]
        right = receiver[j] - receiver[i]
        current = Q(int(i == 13) - int(j == 13), parameters.degrees[13])
        channel = min(index // 9, 3)
        # The sine cubic contributes -1/6, the mixed monomial multiplicity
        # is three, and integration of t**3 supplies the separate factor 1/4.
        channels[channel][0] -= current * left**2 * right / 8
        channels[channel][1] -= current * left * right**2 / 8
    return _NeighborOnset(
        mediator_class=mediator_class,
        gamma_bounds=parameters.gamma,
        channel_cosine_bounds=tuple(parameters.edge_cosines[9 * i] for i in range(3))
        + (I(1),),
        mixed_channel_factors=tuple(map(tuple, channels)),
        degrees=parameters.degrees,
        donor_phase_velocity_column=donor,
        receiver_phase_velocity_column=receiver,
        donor_phase_curvature_column=tuple(
            sum((x * y for x, y in zip(row, donor)), Q(0)) for row in form
        ),
        receiver_phase_curvature_column=tuple(
            sum((x * y for x, y in zip(row, receiver)), Q(0)) for row in form
        ),
    )


def _formal_onset_bounds(onset, donor, receiver, horizon):
    """Retain exact rational products below the absolute interval grid."""
    lower = upper = Q(0)
    for (first, second), cosine in zip(
        onset.mixed_channel_factors, onset.channel_cosine_bounds
    ):
        factor = first * donor**2 * receiver + second * donor * receiver**2
        products = factor * cosine.lo, factor * cosine.hi
        lower += min(products)
        upper += max(products)
    products = tuple(
        g**4 * horizon**4 * value
        for g in (onset.gamma_bounds.lo, onset.gamma_bounds.hi)
        for value in (lower, upper)
    )
    return min(products), max(products)


@dataclass(frozen=True)
class _NeighborStaticCone:
    """Static tangent-phase cones and a heat-transported direct cubic enclosure.

    The two exact columns A*e and A**2*e are geometric derivatives, not a
    time-series execution. The complete tangent-phase remainder is bounded
    analytically throughout the window. Every internal cosine is enclosed by
    [1/6,1], while the two contact cosines are exactly one.
    The direct cubic heat term does not include the separately bounded
    quadratic feedback and cubic phase recoupling correction.
    """

    tangent_phase_remainder_bound: Q
    donor_edge_phase_cones: tuple[I, ...]
    receiver_edge_phase_cones: tuple[I, ...]
    normalized_mixed_force_bounds: tuple[I, ...]
    minimum_force_lower_bound: Q
    mediator_force_upper_bound: Q
    competing_force_upper_bound: Q
    normalized_heat_shape_bounds: tuple[Q, Q]
    direct_cubic_heat_bounds: tuple[Q, Q]
    complete_cubic_correction_upper_bound: Q


def _neighbor_static_cone(onset, amplitude, horizon):
    """Bound equal positive end-port inputs on the proved short window."""
    if amplitude <= 0 or not 0 < horizon <= Q(1, 8):
        raise ValueError("the static cone requires equal positive inputs and 0<H<=1/8")
    g, h, m = _GAMMA_UPPER, horizon, amplitude
    d = 1 - 2 * g**2 * h**2
    error = Q(4, 3) * h**2 * (1 + g**2 / d)
    if any(c.lo < Q(1, 6) or c.hi > 1 for c in onset.channel_cosine_bounds[:3]):
        raise ArithmeticError("fixed internal target cosine enclosure failed")

    def edge_cones(velocity, curvature):
        bounds = []
        for i, j in _EDGES:
            slope = velocity[j] - velocity[i]
            correction = -h * (curvature[j] - curvature[i]) / 2
            bounds.append(
                I(
                    slope + min(0, correction) - error,
                    slope + max(0, correction) + error,
                )
            )
        return tuple(bounds)

    donor = edge_cones(
        onset.donor_phase_velocity_column, onset.donor_phase_curvature_column
    )
    receiver = edge_cones(
        onset.receiver_phase_velocity_column, onset.receiver_phase_curvature_column
    )
    force = [I(0)] * 27
    for index, ((i, j), u, v) in enumerate(zip(_EDGES, donor, receiver)):
        cosine = I(Q(1, 6), 1) if index < 27 else I(1)
        current = -cosine * u * v * (u + v) / 2
        force[i] += current / onset.degrees[i]
        force[j] -= current / onset.degrees[j]
    lower = min(row.lo for row in force)
    central = force[13].hi
    competing = max(Q(0), *(row.hi for row in force))
    shape = lower / 4, central / 4 + (competing - central) * h / 20
    products = tuple(
        gamma**4 * m**3 * h**4 * value
        for gamma in (onset.gamma_bounds.lo, onset.gamma_bounds.hi)
        for value in shape
    )
    # All three nonzero histories have ||f||infinity and ||A*f||infinity<=m,
    # including f=m*(e4+e22). This is not the total variation 2m used by R5.
    correction = (
        3 * m**3 * (Q(4, 15) * g**6 * h**6 / d**4 + Q(1, 63) * g**8 * h**8 / d**5)
    )
    return _NeighborStaticCone(
        tangent_phase_remainder_bound=error,
        donor_edge_phase_cones=donor,
        receiver_edge_phase_cones=receiver,
        normalized_mixed_force_bounds=tuple(force),
        minimum_force_lower_bound=lower,
        mediator_force_upper_bound=central,
        competing_force_upper_bound=competing,
        normalized_heat_shape_bounds=shape,
        direct_cubic_heat_bounds=(min(products), max(products)),
        complete_cubic_correction_upper_bound=correction,
    )


@dataclass(frozen=True)
class _NeighborNonadditivity:
    """Same-source mixed observation with separate work and error obligations.

    The conditional source has per-component Euclidean form/phase residual
    bounds and the inherited zero sums about declared origins. In all four
    histories, the full initial state is identical. Its linear evolution
    cancels in the mixed observation, but its nonlinear defect does not.
    """

    mediator_class: int
    donor_amplitude: Q
    receiver_amplitude: Q
    horizon: Q
    endpoint_radius: Q
    readout_error_bound: Q
    radius: Q
    contact_work_allowance: Q
    donor_work_allowance: Q
    receiver_work_allowance: Q
    onset: _NeighborOnset
    formal_onset_bounds: tuple[Q, Q]
    per_history_fidelity: tuple[_CollectiveInterfaceBound, ...]
    higher_amplitude_error_upper_bound: Q
    nonlinear_source_error_upper_bound: Q
    event_ledger: _ClassProbeEventLedger
    pre_receiver_laplacian_absolute_bounds: tuple[Q, ...]
    static_cone: _NeighborStaticCone | None
    complete_cubic_correction_upper_bound: Q | None
    decision: _ContrastDecision | None
    exact_mixed_zero: bool
    response_bound_available: bool
    all_work_within_allowances: bool
    all_identities_certified: bool
    conditional_protocol_sufficient: bool
    status: str
    unavailable_reasons: tuple[str, ...]
    intervention_nodes: tuple[int, int] = (4, 22)
    readout_node: int = 13
    history_order: tuple[str, ...] = ("neither", "donor_only", "receiver_only", "both")
    reading_count: int = 4
    additive_comparator_rule: str = "F_add(a,b;z0)=F(a,0;z0)+F(0,b;z0)-F(0,0;z0)"
    scope: tuple[str, ...] = (
        "both_signed_form_interventions_are_simultaneous_at_distinct_end_ports",
        "same_complete_initial_source_is_retained_in_all_four_histories",
        "the_exact_common_linear_source_cancels_but_four_nonlinear_source_defects_are_charged",
        "hidden_quadratic_feedback_is_part_of_the_complete_cubic_hierarchy",
        "formal_time_onset_is_not_a_finite_response_or_sign_certificate",
        "finite_static_cone_requires_equal_positive_impulses_and_horizon_at_most_one_eighth",
        "complete_cubic_correction_includes_quadratic_hidden_feedback_and_phase_recoupling",
        "four_reading_errors_and_separately_noisy_additive_null_eight_errors_are_distinct",
        "commuting_event_bookkeeping_reuses_the_ledger_with_receiver_pressure_unchanged_by_donor",
        "identity_and_work_are_conditional_on_original_component_source_norms_and_zero_sums",
        "status_describes_observation_only_work_and_identity_remain_separate_conditional_checks",
        "no_response_time_coefficient_source_acquisition_or_frozen_evidence_is_evaluated",
    )


def _bound_neighbor_nonadditivity(
    *,
    mediator_class,
    donor_amplitude,
    receiver_amplitude,
    horizon,
    endpoint_radius,
    readout_error_bound,
    radius,
    contact_work_allowance,
    donor_work_allowance,
    receiver_work_allowance,
) -> _NeighborNonadditivity:
    """Admit the mixed protocol before constructing any structural coefficient.

    Equal positive impulses on 0<H<=1/8 admit a finite static-cone enclosure.
    Other nonzero protocols retain only their onset and independent source
    and event certificates. Exact zero identities need no time approximation.
    """
    values = _admit_probe_primitives(
        mediator_class,
        dict(
            first_probe_amplitude=donor_amplitude,
            second_probe_amplitude=receiver_amplitude,
            delay=Q(0),
            total_duration=horizon,
            endpoint_radius=endpoint_radius,
            readout_error_bound=readout_error_bound,
            radius=radius,
            contact_work_allowance=contact_work_allowance,
            first_probe_work_allowance=donor_work_allowance,
            second_probe_work_allowance=receiver_work_allowance,
        ),
        Q(2),
    )
    a, b, h, eps, delta = (
        values[key]
        for key in (
            "first_probe_amplitude",
            "second_probe_amplitude",
            "total_duration",
            "endpoint_radius",
            "readout_error_bound",
        )
    )
    if abs(a) + abs(b) > _MAX_INPUT_VARIATION:
        raise ValueError("total input variation must not exceed 7/5000")
    onset = _derive_neighbor_onset(mediator_class)
    fidelity = tuple(
        _bound_collective_interface(
            total_input_variation=value,
            horizon=h,
            endpoint_radius=eps,
        )
        for value in (Q(0), abs(a), abs(b), abs(a) + abs(b))
    )
    pressure = (6 * eps,) * 4
    ledger = _probe_event_ledger(
        values,
        onset.gamma_bounds.hi,
        (None,) * 4,
        pre_second_laplacian_abs_bounds=pressure,
    )
    zero = a == 0 or b == 0 or h == 0
    fifth = sum((row.nominal_fifth_order_error_upper_bound for row in fidelity), Q(0))
    source = sum(
        (row.nonlinear_initialization_error_upper_bound for row in fidelity), Q(0)
    )
    cone = (
        _neighbor_static_cone(onset, a, h)
        if a == b and a > 0 and 0 < h <= Q(1, 8)
        else None
    )
    if zero:
        correction = Q(0)
        decision = _contrast_decision(
            (Q(0), Q(0)), Q(0), delta, exact_zero=True, reading_count=4
        )
    elif cone is not None:
        correction = cone.complete_cubic_correction_upper_bound
        decision = _contrast_decision(
            cone.direct_cubic_heat_bounds,
            correction + fifth + source,
            delta,
            reading_count=4,
        )
    else:
        correction, decision = None, None
    work = all(row.work_within_allowances for row in ledger.histories)
    identity = all(row.identity_certified for row in ledger.histories)
    return _NeighborNonadditivity(
        mediator_class=mediator_class,
        donor_amplitude=a,
        receiver_amplitude=b,
        horizon=h,
        endpoint_radius=eps,
        readout_error_bound=delta,
        radius=values["radius"],
        contact_work_allowance=values["contact_work_allowance"],
        donor_work_allowance=values["first_probe_work_allowance"],
        receiver_work_allowance=values["second_probe_work_allowance"],
        onset=onset,
        formal_onset_bounds=_formal_onset_bounds(onset, a, b, h),
        per_history_fidelity=fidelity,
        higher_amplitude_error_upper_bound=fifth,
        nonlinear_source_error_upper_bound=source,
        event_ledger=ledger,
        pre_receiver_laplacian_absolute_bounds=pressure,
        static_cone=cone,
        complete_cubic_correction_upper_bound=correction,
        decision=decision,
        exact_mixed_zero=zero,
        response_bound_available=decision is not None,
        all_work_within_allowances=work,
        all_identities_certified=identity,
        conditional_protocol_sufficient=bool(
            decision is not None and decision.null_excluded and work and identity
        ),
        status=(
            decision.status if decision is not None else "finite_remainder_unavailable"
        ),
        unavailable_reasons=(
            ()
            if decision is not None
            else ("finite_cone_requires_equal_positive_impulses_and_0<H<=1/8",)
        ),
    )
