"""A registered first section with nonzero centered resource in one UM source.

The isolated source reaches a specified mean phase gap before its established
maturity deadline. A comparison with the original nominal geometry bounds
every marked gap; the actual source is never replaced by that comparison.
Resource and orientation are certified at the section, not after arbitrary
additional waiting or as a successful preparation of a finite partner.
"""

from dataclasses import dataclass
from fractions import Fraction as Q

from ..mathematics._rational_interval import I, cos, pi_interval
from ._mixed_circulation_readiness import (
    _assess_mixed_circulation_readiness,
    _MixedCirculationReadiness,
)
from ._mixed_reciprocal_partner import _nonnegative
from ._mixed_registered_phase_probe import _derive_registered_phase_probe_geometry
from ._sine_coupling_successor import _CouplingSuccessorGeometry, _SineCouplingSuccessor
from ._sine_lyapunov import _sine_common_quadratic_coefficients

_SOURCE_CAP = Q(1, 10**10)
_DELTA = Q(1, 1024)
_ETA = Q(20, 10**70)
_WEIGHT_DEVIATION = Q(2, 10**60)
_RADIUS = Q(1, 32)
_REGISTRATION = _DELTA / 4
_AMPLITUDE = _DELTA / 8
_SECTION = 3 * _AMPLITUDE / 2
_WAIT = Q(10**10)
_MATURE = Q(1, 2**28)
_GAMMA_LOWER = Q(1, 3300)
_GAMMA_UPPER = Q(1, 3000)
_SCALE = Q(6, 5)


@dataclass(frozen=True)
class _MixedRegisteredResource:
    """Conditional source admission at its own finite first section.

    The repeated pair in ``readiness_reference`` is only a way to reuse a
    uniform one-source bound. Its second source and composed circulation
    verdict are not premises of this theorem. No physical second copy,
    prefix, phase jump, response or runtime stopping action is introduced.
    Candidate bounds require their respective certification flags.
    """

    form_error_bound: Q
    phase_error_bound: Q
    readiness_reference: _MixedCirculationReadiness
    source: _SineCouplingSuccessor
    source_geometry: _CouplingSuccessorGeometry
    cross_incidence: tuple[tuple[int, ...], ...]
    source_association_certified: bool
    deadline_certified: bool
    gamma_bounds: I
    regional_sine_coefficient_bounds: I
    nominal_coupling_bounds: I
    nominal_form_norm_upper_bound: Q
    nominal_chart_norm_upper_bound: Q
    nominal_wedge_margin: Q
    nominal_reference_certified: bool
    metric_lower_bound: Q
    metric_upper_bound: Q
    metric_decay_rate: Q
    metric_growth_squared_margin: Q
    initial_comparison_norm_squared_upper_bound: Q
    initial_comparison_norm_upper_bound: Q
    initial_comparison_margin: Q
    matrix_deviation_upper_bound: Q
    comparison_forcing_upper_bound: Q
    comparison_forcing_margin: Q
    comparison_error_upper_bound: Q
    comparison_error_margin: Q
    comparison_certified: bool
    initial_mean_gap_lower_bound: Q
    initial_section_margin: Q
    deadline_mean_gap_upper_bound: Q
    deadline_section_margin: Q
    first_unit_mean_gap_lower_bound: Q
    first_unit_section_margin: Q
    section_time_bounds: tuple[Q, Q]
    section_exists_certified: bool
    section_marked_gap_bounds: tuple[Q, Q]
    predeadline_registered_gap_upper_bound: Q
    postdeadline_registered_gap_upper_bound: Q
    registered_gap_upper_bound: Q
    registration_margin: Q
    registration_retained_after_section_certified: bool
    section_phase_norm_squared_lower_bound: Q
    section_excess_storage_lower_bound: Q
    excess_storage_upper_bound: Q
    comparison_phase_floor_upper_bound: Q
    resource_floor_margin: Q
    section_resource_certified: bool
    last_unit_nominal_gap_margin: Q
    nominal_regional_form_upper_bound: Q
    regional_form_contrast_upper_bound: Q
    form_orientation_margin: Q
    nominal_mean_gap_rate_upper_bound: Q
    phase_rate_error_upper_bound: Q
    section_mean_gap_rate_upper_bound: Q
    rate_orientation_margin: Q
    section_orientation_certified: bool
    certified: bool
    status: str
    unresolved_conditions: tuple[str, ...]
    source_error_cap: Q = _SOURCE_CAP
    original_relative_phase_gap: Q = _DELTA
    original_supplied_phase_origins: tuple[Q, Q] = (-_DELTA / 2, _DELTA / 2)
    section_mean_gap: Q = _SECTION
    comparison_target_gap_amplitude: Q = _AMPLITUDE
    registration_gap_limit: Q = _REGISTRATION
    phase_radius: Q = _RADIUS
    original_waiting_radius: Q = _ETA
    actual_weight_deviation_upper_bound: Q = _WEIGHT_DEVIATION
    isolated_deadline: Q = _WAIT
    derived_mature_radius: Q = _MATURE
    phase_scale: Q = _SCALE
    auxiliary_metric_capacity: Q = Q(4, 9)
    phase_coercivity: Q = Q(1, 5)
    mean_gap_covector_norm_upper_bound: Q = Q(1, 2)
    gap_minus_mean_covector_norm: Q = Q(4, 3)
    physical_capacities: tuple[Q, ...] = (Q(1),) * 18
    physical_source_count: int = 1
    full_state_dimension: int = 36
    section_age_computed: bool = False
    runtime_stopping_rule_executed: bool = False
    runtime_source_authenticated: bool = False
    source_trajectory_evaluated: bool = False
    new_external_input_supplied: bool = False
    new_phase_jump_performed: bool = False
    charge_or_leak_prefix_applied: bool = False
    resource_lower_bound_at_section_only: bool = True
    resource_lower_bound_for_arbitrary_later_age: bool = False
    clock: str = "tau=(1023/1024)*t; unchanged isolated common UM degree-nine law"
    scope: tuple[str, ...] = (
        "one_actual_conditional_common_UM_source_with_original_acquisition_premises",
        "first_mean_gap_section_is_supplied_preparation_criterion_not_event_selector",
        "original_nominal_weights_are_comparison_only_not_actual_carrier_substitution",
        "full_actual_reset_gauge_and_both_weighted_pressure_rows_are_retained",
        "all_marked_gaps_and_constituent_identity_retained_after_the_section",
        "positive_centered_resource_and_orientation_hold_at_the_section_only",
        "exceeding_a_necessary_phase_floor_is_not_mutual_target_reachability",
        "no_second_physical_copy_prefix_input_phase_jump_or_circulation_consumption",
        "no_authenticated_endpoint_timer_autonomous_contact_or_physical_energy_claim",
    )


def _assess_mixed_registered_resource(*, form_error_bound, phase_error_bound):
    """Rebuild the original source and certify a registered resource section."""
    rx = _nonnegative(form_error_bound, "form_error_bound")
    rt = _nonnegative(phase_error_bound, "phase_error_bound")
    readiness = _assess_mixed_circulation_readiness(
        form_error_bounds=(rx, rx),
        phase_error_bounds=(rt, rt),
        phase_jump_error_bound=Q(1, 10**8),
        readout_error_bound=Q(1, 10**6),
    )
    registered = _derive_registered_phase_probe_geometry()
    geometry = registered.successor
    source = readiness.sources[0]
    common = source.completion_reports[0]
    expected_initial = Q(9, 2) * _DELTA**2 + 54 * _DELTA * _ETA + 164 * _ETA**2
    expected_energy = Q(27, 2) * _DELTA**2 + 486 * _ETA + 18 * _ETA**2
    source_ok = (
        max(rx, rt) <= _SOURCE_CAP
        and readiness.form_error_bounds == (rx, rx)
        and readiness.phase_error_bounds == (rt, rt)
        and readiness.source_geometry == geometry
        and readiness.source_certified[0]
        and readiness.initial_inclusion_certified[0]
        and readiness.all_time_source_identity_certified[0]
        and readiness.initial_norm_squared_upper_bounds[0] == expected_initial
        and registered.cross_symmetry_certified
        and geometry.completion == "common_sense"
        and geometry.nodes == tuple(range(18))
        and geometry.degrees == (9,) * 18
        and geometry.components == (tuple(range(18)),)
        and geometry.held_capacities == (Q(1),) * 18
        and len(geometry.internal_edges) == 18
        and all(weight == 1 for _, _, weight in geometry.internal_edges)
        and len(geometry.new_edge_pairs) == 63
        and len(geometry.edge_pairs) == 81
        and geometry.support_certified
        and geometry.reference_stationary_certified
        and geometry.spectral_gap_certified
        and geometry.weighted_gap_lower_bound >= 3
        and geometry.bare_gap_lower_bound >= 4
        and geometry.quotient_coercivity_lower_bound >= Q(1, 5)
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
        and _GAMMA_LOWER
        <= source.gamma_range[0]
        <= source.gamma_range[1]
        <= _GAMMA_UPPER
        and source.new_weight_range == (Q(3, 4), Q(1))
        and 0 <= source.edge_weight_deviation_upper_bound <= _WEIGHT_DEVIATION
        and source.precontact_reference.relative_phase_gap == _DELTA
        and source.precontact_reference.supplied_phase_origins
        == (-_DELTA / 2, _DELTA / 2)
        and source.precontact_reference.initial_form_means == (Q(0), Q(0))
        and common.geometry == geometry
        and common.degree_form_limit_bounds == ((Q(0), Q(0)),)
        and common.constituent_identity_certified
        and common.initial_registration_bounds == (_DELTA - 6 * _ETA, _DELTA + 6 * _ETA)
        and common.initial_quotient_norm_squared_upper_bound == expected_initial
        and common.initial_phase_norm_squared_upper_bound
        == expected_initial - 2 * _ETA**2
        and common.initial_quotient_radius_margin == _RADIUS**2 - expected_initial > 0
        and common.initial_energy_upper_bound == expected_energy
        and common.storage_barrier
        == geometry.quotient_coercivity_lower_bound * _RADIUS**2
        and common.storage_margin == common.storage_barrier - expected_energy > 0
    )
    lower, upper, rate = _sine_common_quadratic_coefficients(
        gamma_lower=Q(1, 2750),
        gamma_upper=Q(1, 2500),
        cosine_lower=Q(5, 72),
        gap_lower=Q(3),
        capacity_lower=Q(4, 9),
        capacity_upper=Q(4, 9),
    )
    metric_ratio = _SCALE**2 * upper / lower
    growth_margin = Q(9, 4) - metric_ratio
    deadline_ok = (
        source_ok
        and readiness.metric_lower_bound == lower
        and readiness.metric_upper_bound == upper
        and readiness.metric_decay_rate == rate
        and readiness.phase_scale == _SCALE
        and readiness.auxiliary_metric_capacity == Q(4, 9)
        and readiness.waiting_duration == _WAIT
        and readiness.derived_mature_norm_upper_bound == _MATURE
        and readiness.readiness_certified[0]
        and readiness.decay_comparison_certified
        and readiness.physical_norm_prefactors[0] == metric_ratio * expected_initial
        and readiness.decay_upper_bound == Q(1, 2**39)
        and readiness.returned_norm_squared_upper_bounds[0]
        == metric_ratio * expected_initial / 2**39
        < _MATURE**2
        and rate * _WAIT > 30
        and 5**10 > 2**23
        and growth_margin > 0
    )
    pi = pi_interval()
    gamma = 1 / (1023 * pi)
    coefficient = 1 + 2 * cos(pi / 9)
    coupling = Q(13, 27) - _DELTA / (18 * pi)
    nominal_form = 3 * _GAMMA_UPPER * _DELTA
    nominal_norm = 3 * (1 + _GAMMA_UPPER) * _DELTA
    wedge_margin = 2 * coupling.lo / 3 - _GAMMA_UPPER**2
    nominal_ok = (
        source_ok
        and _GAMMA_LOWER <= gamma.lo <= gamma.hi <= _GAMMA_UPPER
        and 2 < coefficient.lo <= coefficient.hi < 3
        and Q(4, 9) < coupling.lo <= coupling.hi < Q(1, 2)
        and wedge_margin > 0
        and nominal_form < Q(1, 1024)
        and nominal_norm < _RADIUS
    )
    initial_squared = 164 * _ETA**2
    initial_norm = 14 * _ETA
    initial_margin = initial_norm**2 - initial_squared
    matrix_deviation = 14 * _WEIGHT_DEVIATION
    forcing = 2 * _WEIGHT_DEVIATION
    # Both original rows of the matrix mismatch are bounded before the
    # phase scaling; its norm factor is already in metric_ratio.
    forcing_margin = forcing - 2 * matrix_deviation * nominal_form / 9
    comparison = Q(3, 2) * (initial_norm + _WAIT * forcing)
    comparison_margin = Q(1, 10**40) - comparison
    comparison_ok = (
        nominal_ok
        and deadline_ok
        and initial_margin > 0
        and forcing_margin > 0
        and comparison_margin > 0
    )
    initial_gap = _DELTA - 6 * _ETA
    initial_section_margin = initial_gap - _SECTION
    deadline_gap = _MATURE / 2
    deadline_section_margin = _SECTION - deadline_gap
    early_gap = _DELTA * (1 - _GAMMA_UPPER**2) - comparison
    early_margin = early_gap - _SECTION
    section = (
        comparison_ok
        and initial_section_margin > 0
        and deadline_section_margin > 0
        and early_margin > 0
        and 1 < _WAIT
    )
    section_gaps = (_SECTION - Q(4, 3) * comparison, _SECTION + Q(4, 3) * comparison)
    predeadline_gap = _SECTION + 3 * comparison
    postdeadline_gap = Q(3, 2) * _MATURE
    registered_gap = max(predeadline_gap, postdeadline_gap)
    registration_margin = _REGISTRATION - registered_gap
    identity = section and section_gaps[0] > 0 and registration_margin > 0
    phase_squared = Q(9, 2) * _SECTION**2
    energy_lower = phase_squared / 5
    phase_floor = Q(9, 5) * _AMPLITUDE**2
    resource_margin = energy_lower - phase_floor
    resource = identity and resource_margin > 0 and energy_lower < expected_energy
    last_unit_margin = _SECTION / 2 - comparison / 2
    nominal_form_upper = -_GAMMA_LOWER * _SECTION / 36
    form_upper = 2 * nominal_form_upper + comparison
    form_margin = -_GAMMA_LOWER * _SECTION / 36 - form_upper
    nominal_rate_upper = -4 * _GAMMA_LOWER**2 * _SECTION / 81
    rate_error = 2 * _GAMMA_UPPER * (comparison + _WEIGHT_DEVIATION)
    rate_upper = nominal_rate_upper + rate_error
    rate_margin = -rate_upper
    orientation = (
        section and last_unit_margin > 0 and form_margin > 0 and rate_margin > 0
    )
    unresolved = tuple(
        label
        for good, label in (
            (
                source_ok,
                "original_actual_source_identity_or_nominal_association_unavailable",
            ),
            (deadline_ok, "one_source_uniform_deadline_unavailable"),
            (nominal_ok, "original_nominal_unforced_reference_unavailable"),
            (comparison_ok, "full_actual_to_nominal_comparison_unavailable"),
            (section, "finite_first_mean_gap_section_unavailable"),
            (identity, "all_gap_registration_retention_unavailable"),
            (resource, "section_only_centered_resource_lower_bound_unavailable"),
            (orientation, "section_centered_mode_orientation_unavailable"),
        )
        if not good
    )
    return _MixedRegisteredResource(
        form_error_bound=rx,
        phase_error_bound=rt,
        readiness_reference=readiness,
        source=source,
        source_geometry=geometry,
        cross_incidence=registered.cross_incidence,
        source_association_certified=source_ok,
        deadline_certified=deadline_ok,
        gamma_bounds=gamma,
        regional_sine_coefficient_bounds=coefficient,
        nominal_coupling_bounds=coupling,
        nominal_form_norm_upper_bound=nominal_form,
        nominal_chart_norm_upper_bound=nominal_norm,
        nominal_wedge_margin=wedge_margin,
        nominal_reference_certified=nominal_ok,
        metric_lower_bound=lower,
        metric_upper_bound=upper,
        metric_decay_rate=rate,
        metric_growth_squared_margin=growth_margin,
        initial_comparison_norm_squared_upper_bound=initial_squared,
        initial_comparison_norm_upper_bound=initial_norm,
        initial_comparison_margin=initial_margin,
        matrix_deviation_upper_bound=matrix_deviation,
        comparison_forcing_upper_bound=forcing,
        comparison_forcing_margin=forcing_margin,
        comparison_error_upper_bound=comparison,
        comparison_error_margin=comparison_margin,
        comparison_certified=comparison_ok,
        initial_mean_gap_lower_bound=initial_gap,
        initial_section_margin=initial_section_margin,
        deadline_mean_gap_upper_bound=deadline_gap,
        deadline_section_margin=deadline_section_margin,
        first_unit_mean_gap_lower_bound=early_gap,
        first_unit_section_margin=early_margin,
        section_time_bounds=(Q(1), _WAIT),
        section_exists_certified=section,
        section_marked_gap_bounds=section_gaps,
        predeadline_registered_gap_upper_bound=predeadline_gap,
        postdeadline_registered_gap_upper_bound=postdeadline_gap,
        registered_gap_upper_bound=registered_gap,
        registration_margin=registration_margin,
        registration_retained_after_section_certified=identity,
        section_phase_norm_squared_lower_bound=phase_squared,
        section_excess_storage_lower_bound=energy_lower,
        excess_storage_upper_bound=expected_energy,
        comparison_phase_floor_upper_bound=phase_floor,
        resource_floor_margin=resource_margin,
        section_resource_certified=resource,
        last_unit_nominal_gap_margin=last_unit_margin,
        nominal_regional_form_upper_bound=nominal_form_upper,
        regional_form_contrast_upper_bound=form_upper,
        form_orientation_margin=form_margin,
        nominal_mean_gap_rate_upper_bound=nominal_rate_upper,
        phase_rate_error_upper_bound=rate_error,
        section_mean_gap_rate_upper_bound=rate_upper,
        rate_orientation_margin=rate_margin,
        section_orientation_certified=orientation,
        certified=not unresolved,
        status=(
            "conditional_registered_resource_section"
            if not unresolved
            else "unavailable"
        ),
        unresolved_conditions=unresolved,
    )
