"""Uniform isolated waiting and automatic handoff to the circulation task.

The finite maturity radius is derived from the actual common UM image and
its intact invariant chart. No endpoint, chosen age or mature radius is an
input. Source, elapsed evolution, actuator and sensor association remain
conditional premises, not authenticated runtime facts.
"""

from dataclasses import dataclass
from fractions import Fraction as Q

from ._mixed_reciprocal_partner import _nonnegative, _nonnegative_pair
from ._mixed_two_channel_circulation import (
    _assess_mixed_two_channel_circulation,
    _MixedTwoChannelCirculation,
)
from ._sine_coupling_successor import (
    _assess_sine_coupling_successor,
    _CouplingSuccessorGeometry,
    _derive_coupling_successor_geometry,
    _SineCouplingSuccessor,
)
from ._sine_lyapunov import _sine_common_quadratic_coefficients

_SOURCE_CAP = Q(1, 10**10)
_DELTA = Q(1, 1024)
_ETA = Q(20, 10**70)
_RADIUS = Q(1, 32)
_PHASE_SCALE = Q(6, 5)
_AUXILIARY_CAPACITY = Q(4, 9)
_SCALED_GAMMA = (Q(1, 2750), Q(1, 2500))
_SCALED_COSINE = Q(5, 72)
_GAP = Q(3)
_WAIT = Q(10**10)
_DECAY = Q(1, 2**39)
_PREF_LIMIT = Q(1, 2**17)
_SQUARED_TARGET = Q(1, 2**56)
_HANDOFF_RADIUS = Q(1, 2**28)
_OLD_MATURE_LIMIT = Q(1, 10**8)
_RECORD_FLOOR = Q(1503, 400000000)


@dataclass(frozen=True)
class _MixedCirculationReadiness:
    """Candidate bounds, source obligations and downstream handoff separately.

    The waiting duration starts after the associated common UM image and
    covers every later uninterrupted isolated age. Isolation retains its
    connected internal support. Physical capacities stay one; the auxiliary
    coefficient only rewrites the degree-nine rows for a shared proof kernel.
    A valid handoff removes the independent maturity assumption from the
    composed theorem without authenticating a graph, event or actual timer.
    """

    form_error_bounds: tuple[Q, Q]
    phase_error_bounds: tuple[Q, Q]
    phase_jump_error_bound: Q
    readout_error_bound: Q
    sources: tuple[_SineCouplingSuccessor, ...]
    source_geometry: _CouplingSuccessorGeometry
    source_certified: tuple[bool, bool]
    initial_inclusion_certified: tuple[bool, bool]
    all_time_source_identity_certified: tuple[bool, bool]
    metric_lower_bound: Q
    metric_upper_bound: Q
    metric_decay_rate: Q
    initial_norm_squared_upper_bounds: tuple[Q, Q]
    initial_scaled_norm_squared_upper_bounds: tuple[Q, Q]
    initial_metric_upper_bounds: tuple[Q, Q]
    physical_norm_prefactors: tuple[Q, Q]
    prefactor_margins: tuple[Q, Q]
    exponent_lower_bound: Q
    exponent_margin: Q
    decay_comparison_certified: bool
    returned_norm_squared_upper_bounds: tuple[Q, Q]
    readiness_squared_margins: tuple[Q, Q]
    readiness_certified: tuple[bool, bool]
    later_isolated_age_guarantee_certified: bool
    handoff_norm_upper_bounds: tuple[Q, Q] | None
    circulation: _MixedTwoChannelCirculation | None
    handoff_association_certified: bool
    composed_response_certified: bool
    certified: bool
    status: str
    unresolved_conditions: tuple[str, ...]
    source_error_cap: Q = _SOURCE_CAP
    phase_scale: Q = _PHASE_SCALE
    auxiliary_metric_capacity: Q = _AUXILIARY_CAPACITY
    scaled_gamma_bounds: tuple[Q, Q] = _SCALED_GAMMA
    scaled_cosine_lower_bound: Q = _SCALED_COSINE
    scaled_stiffness_upper_ratio: Q = Q(25, 27)
    weighted_gap_lower_bound: Q = _GAP
    waiting_duration: Q = _WAIT
    original_clock_waiting_duration: Q = Q(1024, 1023) * _WAIT
    decay_upper_bound: Q = _DECAY
    physical_prefactor_limit: Q = _PREF_LIMIT
    readiness_squared_target: Q = _SQUARED_TARGET
    derived_mature_norm_upper_bound: Q = _HANDOFF_RADIUS
    circulation_mature_norm_limit: Q = _OLD_MATURE_LIMIT
    conservative_record_separation_floor: Q = _RECORD_FLOOR
    physical_capacities: tuple[Q, ...] = (Q(1),) * 18
    caller_maturity_required: bool = False
    runtime_source_authenticated: bool = False
    attained_waiting_age_authenticated: bool = False
    source_trajectory_evaluated: bool = False
    minimum_waiting_time_claimed: bool = False
    clock: str = "tau=(1023/1024)*t; waiting starts after the actual common UM image"
    scope: tuple[str, ...] = (
        "actual_common_degree_loss_source_and_intact_internal_support_during_wait",
        "zero_form_mean_and_actual_conserved_phase_gauge_no_endpoint_reset",
        "phase_rescaling_and_auxiliary_metric_do_not_change_physical_capacities",
        "noncommuting_time_varying_bare_Hessian_admitted_against_weighted_transport",
        "initial_metric_upper_bound_includes_the_unknown_signed_mixed_product",
        "uniform_sufficient_wait_and_all_later_isolated_ages_not_a_minimal_time",
        "derived_positive_norm_automatically_supplies_unchanged_response_protocol",
        "no_runtime_event_timer_sensor_authentication_or_physical_validation",
    )


def _assess_mixed_circulation_readiness(
    *,
    form_error_bounds,
    phase_error_bounds,
    phase_jump_error_bound,
    readout_error_bound,
):
    """Derive maturity from original source primitives, then rebuild its consumer."""
    forms = _nonnegative_pair(form_error_bounds, "form_error_bounds")
    phases = _nonnegative_pair(phase_error_bounds, "phase_error_bounds")
    jump_error = _nonnegative(phase_jump_error_bound, "phase_jump_error_bound")
    readout = _nonnegative(readout_error_bound, "readout_error_bound")
    sources = tuple(
        _assess_sine_coupling_successor(form_error_bound=rx, phase_error_bound=rt)
        for rx, rt in zip(forms, phases, strict=True)
    )
    geometry = _derive_coupling_successor_geometry(completion="common_sense")
    geometry_ok = (
        geometry.completion == "common_sense"
        and geometry.nodes == tuple(range(18))
        and geometry.degree == 9
        and geometry.degrees == (9,) * 18
        and geometry.components == (tuple(range(18)),)
        and geometry.held_capacities == (Q(1),) * 18
        and len(geometry.internal_edges) == 18
        and all(w == 1 for _, _, w in geometry.internal_edges)
        and len(geometry.new_edge_pairs) == 63
        and len(geometry.edge_pairs) == 81
        and geometry.support_certified
        and geometry.reference_stationary_certified
        and geometry.spectral_gap_certified
        and geometry.weighted_gap_lower_bound >= _GAP
        and geometry.quotient_coercivity_lower_bound >= Q(1, 5)
    )
    source_ok = tuple(
        max(rx, rt) <= _SOURCE_CAP
        and source.form_error_bound == rx
        and source.phase_error_bound == rt
        and source.source_certified
        and source.isolated_waiting_certified
        and source.geometry_certified
        and source.conditional_image_certified
        and source.phase_radius == _RADIUS
        and source.relative_phase_gap == _DELTA
        and source.waiting_radius == _ETA
        and source.phase_reset_error_bound == 3 * _ETA
        and 0 <= source.gauge_phase_deviation_upper_bound <= 6 * _ETA
        and source.phase_cosine_lower_bound >= Q(1, 10)
        and source.phase_curvature_bounds.lo >= Q(1, 10)
        and source.phase_curvature_margin > 0
        and Q(1, 3300) <= source.gamma_range[0] <= source.gamma_range[1] <= Q(1, 3000)
        and source.new_weight_range == (Q(3, 4), Q(1))
        and source.precontact_reference.initial_form_means == (Q(0), Q(0))
        and source.completion_reports[0].geometry == geometry
        and source.completion_reports[0].degree_form_limit_bounds == ((Q(0), Q(0)),)
        and geometry_ok
        for rx, rt, source in zip(forms, phases, sources, strict=True)
    )
    common = tuple(source.completion_reports[0] for source in sources)
    expected_initial = Q(9, 2) * _DELTA**2 + 54 * _DELTA * _ETA + 164 * _ETA**2
    initial = tuple(row.initial_quotient_norm_squared_upper_bound for row in common)
    inclusion = tuple(
        good
        and value == expected_initial
        and 0 < value < 5 * _DELTA**2
        and row.initial_phase_norm_squared_upper_bound == value - 2 * _ETA**2
        for good, value, row in zip(source_ok, initial, common, strict=True)
    )
    trapped = tuple(
        good
        and row.constituent_identity_certified
        and row.initial_quotient_radius_margin == _RADIUS**2 - value > 0
        and row.storage_barrier == geometry.quotient_coercivity_lower_bound * _RADIUS**2
        and 0 <= row.initial_energy_upper_bound < row.storage_barrier
        and row.storage_margin
        == row.storage_barrier - row.initial_energy_upper_bound
        > 0
        for good, row, value in zip(inclusion, common, initial, strict=True)
    )
    lower, upper, rate = _sine_common_quadratic_coefficients(
        gamma_lower=_SCALED_GAMMA[0],
        gamma_upper=_SCALED_GAMMA[1],
        cosine_lower=_SCALED_COSINE,
        gap_lower=_GAP,
        capacity_lower=_AUXILIARY_CAPACITY,
        capacity_upper=_AUXILIARY_CAPACITY,
    )
    scaled_initial = tuple(_PHASE_SCALE**2 * value for value in initial)
    metric_initial = tuple(upper * value for value in scaled_initial)
    prefactors = tuple(value / lower for value in metric_initial)
    prefactor_margins = tuple(_PREF_LIMIT - value for value in prefactors)
    exponent = rate * _WAIT
    exponent_margin = exponent - 30
    # e > 5/2 and 5**10 > 2**23 imply e**30 > 2**39. All quantities
    # below are rational; no new exponential or square-root call is needed.
    decay_ok = exponent_margin > 0 and 5**10 > 2**23
    returned = tuple(value * _DECAY for value in prefactors)
    readiness_margins = tuple(_SQUARED_TARGET - value for value in returned)
    ready = tuple(
        held
        and prefactor_margin > 0
        and decay_ok
        and margin > 0
        and _HANDOFF_RADIUS**2 == _SQUARED_TARGET
        and 0 < _HANDOFF_RADIUS < _OLD_MATURE_LIMIT
        for held, prefactor_margin, margin in zip(
            trapped, prefactor_margins, readiness_margins, strict=True
        )
    )
    handoff = (_HANDOFF_RADIUS, _HANDOFF_RADIUS) if all(ready) else None
    circulation = (
        _assess_mixed_two_channel_circulation(
            form_error_bounds=forms,
            phase_error_bounds=phases,
            mature_norm_upper_bounds=handoff,
            phase_jump_error_bound=jump_error,
            readout_error_bound=readout,
        )
        if handoff is not None
        else None
    )
    association = (
        circulation is not None
        and circulation.form_error_bounds == forms
        and circulation.phase_error_bounds == phases
        and circulation.mature_norm_upper_bounds == handoff
        and circulation.phase_jump_error_bound == jump_error
        and circulation.readout_error_bound == readout
        and circulation.source_geometry == geometry
        and circulation.source_association_certified
        and circulation.charge_amplitude == 0
        and circulation.charge_duration == circulation.prefix_leak_duration == Q(1, 32)
        and circulation.phase_jump_amplitude == Q(1, 8192)
        and circulation.contact_duration == 512
        and circulation.readout_error_cap == Q(1, 10**6)
        and circulation.classifier_threshold == Q(1, 400000)
        and circulation.contact_pairs == tuple((j, 18 + j) for j in range(18))
        and circulation.full_state_dimension == 72
        and circulation.independently_observed_channels == 1
        and all(
            source.form_error_bound == rx
            and source.phase_error_bound == rt
            and source.completion_reports[0].geometry == geometry
            and source.completion_reports[0].initial_quotient_norm_squared_upper_bound
            == value
            and source.new_weight_range == original.new_weight_range
            and source.edge_weight_deviation_upper_bound
            == original.edge_weight_deviation_upper_bound
            and source.gamma_range == original.gamma_range
            and source.phase_reset_error_bound == original.phase_reset_error_bound
            and source.gauge_phase_deviation_upper_bound
            == original.gauge_phase_deviation_upper_bound
            and source.waiting_radius == original.waiting_radius
            and source.precontact_reference.initial_form_means
            == original.precontact_reference.initial_form_means
            and source.completion_reports[0].initial_registration_bounds
            == original.completion_reports[0].initial_registration_bounds
            and source.completion_reports[0].degree_form_limit_bounds
            == original.completion_reports[0].degree_form_limit_bounds
            for source, original, rx, rt, value in zip(
                circulation.sources, sources, forms, phases, initial, strict=True
            )
        )
    )
    composed = (
        association
        and circulation.certified
        and circulation.finite_identity_certified
        and circulation.work_certified
        and circulation.independent_hypotheses_separated
        and circulation.record_separation_margin >= _RECORD_FLOOR
        and min(
            circulation.full_threshold_margin, circulation.comparator_threshold_margin
        )
        > 0
    )
    unresolved = tuple(
        label
        for good, label in (
            (all(source_ok), "actual_common_degree_source_unavailable"),
            (all(inclusion), "original_full_image_norm_inclusion_unavailable"),
            (all(trapped), "all_time_original_chart_or_identity_unavailable"),
            (decay_ok, "fixed_rational_decay_comparison_unavailable"),
            (all(ready), "uniform_positive_maturity_unavailable"),
            (
                association,
                "derived_radius_or_response_protocol_association_unavailable",
            ),
            (composed, "unchanged_circulation_discriminator_unavailable"),
        )
        if not good
    )
    return _MixedCirculationReadiness(
        form_error_bounds=forms,
        phase_error_bounds=phases,
        phase_jump_error_bound=jump_error,
        readout_error_bound=readout,
        sources=sources,
        source_geometry=geometry,
        source_certified=source_ok,
        initial_inclusion_certified=inclusion,
        all_time_source_identity_certified=trapped,
        metric_lower_bound=lower,
        metric_upper_bound=upper,
        metric_decay_rate=rate,
        initial_norm_squared_upper_bounds=initial,
        initial_scaled_norm_squared_upper_bounds=scaled_initial,
        initial_metric_upper_bounds=metric_initial,
        physical_norm_prefactors=prefactors,
        prefactor_margins=prefactor_margins,
        exponent_lower_bound=exponent,
        exponent_margin=exponent_margin,
        decay_comparison_certified=decay_ok,
        returned_norm_squared_upper_bounds=returned,
        readiness_squared_margins=readiness_margins,
        readiness_certified=ready,
        later_isolated_age_guarantee_certified=all(ready),
        handoff_norm_upper_bounds=handoff,
        circulation=circulation,
        handoff_association_certified=association,
        composed_response_certified=composed,
        certified=not unresolved,
        status=(
            "conditional_finite_readiness_and_circulation"
            if not unresolved
            else "unavailable"
        ),
        unresolved_conditions=unresolved,
    )
