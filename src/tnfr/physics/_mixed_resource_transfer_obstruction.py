"""A resource-bearing donor need not reach the fixed opposite-phase target.

Actual acquired sources evolve under the existing reciprocal full-slot form
port. A comparison keeps the donor's inherited form/phase correlation and
the recipient's genuine maturity error. No trajectory, new field, waiting
endpoint or phase-only substitute is evaluated by this conditional reader.
"""

from dataclasses import dataclass
from fractions import Fraction as Q

from ..mathematics._rational_interval import cos, pi_interval
from ._mixed_circulation_readiness import (
    _assess_mixed_circulation_readiness,
    _MixedCirculationReadiness,
)
from ._mixed_reciprocal_partner import _nonnegative_pair
from ._mixed_registered_phase_probe import _derive_registered_phase_probe_geometry
from ._mixed_registered_resource import (
    _assess_mixed_registered_resource,
    _MixedRegisteredResource,
)
from ._mixed_two_channel_circulation import _MixedTwoChannelCirculation
from ._sine_coupling_successor import _CouplingSuccessorGeometry, _SineCouplingSuccessor
from ._sine_lyapunov import (
    _sine_common_quadratic_coefficients,
    _sine_common_quadratic_exchange_coefficients,
)

_SOURCE_CAP = Q(1, 10**10)
_DELTA = Q(1, 1024)
_ETA = Q(20, 10**70)
_WEIGHT_DEVIATION = Q(2, 10**60)
_RADIUS = Q(1, 32)
_MATURE = Q(1, 2**28)
_WAIT = Q(10**10)
_AMPLITUDE = _DELTA / 8
_SECTION = 3 * _AMPLITUDE / 2
_SIGMA = Q(1, 10**8)
_OLD_READOUT = Q(1, 10**6)
_TARGET_BASE = Q(3, 2) * _MATURE
_SCALE = Q(6, 5)
_GAMMA_LOWER = Q(1, 3300)
_GAMMA_UPPER = Q(1, 3000)
_INITIAL = Q(9, 2) * _DELTA**2 + 54 * _DELTA * _ETA + 164 * _ETA**2
_ENERGY = Q(27, 2) * _DELTA**2 + 486 * _ETA + 18 * _ETA**2
_SECTION_ERROR = Q(3, 2) * (14 * _ETA + 2 * _WAIT * _WEIGHT_DEVIATION)


@dataclass(frozen=True)
class _MixedResourceTransferObstruction:
    """Candidate bounds with separate source, identity and target obligations.

    The donor is at its actual first section, with its nonzero velocity
    inherited from isolation. The recipient alone is mature. Synchronizing
    those actual states is a supplied association, not a runtime event.
    The recipient reader's repeated second slot is theorem reuse only.
    Its old response, prefix and jump are not performed here.
    The target is bound to the original owner's immutable protocol defaults;
    no downstream circulation report or verdict is consumed.
    ``nominal_velocity_factor_upper_bound`` is a coefficient envelope,
    not permission to replace the inherited gamma-correlated velocity
    by the wider independent interval obtained from that envelope.
    """

    form_error_bounds: tuple[Q, Q]
    phase_error_bounds: tuple[Q, Q]
    donor: _MixedRegisteredResource
    recipient_readiness: _MixedCirculationReadiness
    sources: tuple[_SineCouplingSuccessor, _SineCouplingSuccessor]
    source_geometry: _CouplingSuccessorGeometry
    donor_source_certified: bool
    recipient_source_certified: bool
    source_association_certified: bool
    new_edge_incident_counts: tuple[int, ...]
    transport_half_difference_upper_bound: Q
    dissipation_margin: Q
    initial_excess_storage_upper_bound: Q
    initial_norm_squared_upper_bound: Q
    full_norm_squared_upper_bound: Q
    full_norm_upper_bound: Q
    trap_norm_margin: Q
    acute_chart_margin: Q
    actual_trap_certified: bool
    nominal_q0_bounds: tuple[Q, Q]
    nominal_velocity_factor_upper_bound: Q
    nominal_wedge_margin: Q
    nominal_wedges_certified: bool
    base_metric_lower_bound: Q
    base_metric_upper_bound: Q
    base_metric_decay_rate: Q
    metric_lower_bound: Q
    metric_upper_bound: Q
    metric_decay_rate: Q
    conservative_metric_decay_rate: Q
    metric_growth_squared_margin: Q
    metric_rate_margin: Q
    metric_base_ratio_margin: Q
    metric_certified: bool
    initial_comparison_norm_upper_bound: Q
    comparison_matrix_deviation_upper_bound: Q
    comparison_forcing_upper_bound: Q
    comparison_forcing_margin: Q
    uniform_comparison_bound: Q
    uniform_comparison_upper_bound: Q
    uniform_comparison_margin: Q
    comparison_certified: bool
    marked_gap_upper_bound: Q
    registration_margin: Q
    all_time_identity_certified: bool
    actual_gap_contrast_upper_bound: Q
    target_gap_contrast_lower_bound: Q
    target_separation_margin: Q
    target_domain_certified: bool
    donor_resource_lower_bound: Q
    comparison_phase_floor_upper_bound: Q
    resource_floor_margin: Q
    positive_donor_resource_certified: bool
    connected_target_unreachable: bool
    disconnected_target_unreachable: bool
    certified: bool
    status: str
    unresolved_conditions: tuple[str, ...]
    source_error_cap: Q = _SOURCE_CAP
    original_relative_phase_gap: Q = _DELTA
    original_supplied_phase_origins: tuple[Q, Q] = (-_DELTA / 2, _DELTA / 2)
    section_mean_gap: Q = _SECTION
    donor_section_comparison_error: Q = _SECTION_ERROR
    recipient_mature_radius: Q = _MATURE
    recipient_isolated_deadline: Q = _WAIT
    actual_weight_deviation_upper_bound: Q = _WEIGHT_DEVIATION
    phase_scale: Q = _SCALE
    auxiliary_metric_capacity: Q = Q(4, 9)
    exchange_norm_upper_bound: Q = Q(2)
    nominal_transport_gap_lower_bound: Q = Q(3)
    physical_norm_growth_upper_bound: Q = Q(4)
    phase_radius: Q = _RADIUS
    registration_gap_limit: Q = _DELTA / 4
    target_gap_amplitude: Q = _AMPLITUDE
    target_gap_error_bound: Q = _SIGMA
    target_base_radius: Q = _TARGET_BASE
    mean_gap_contrast_covector_norm: Q = Q(2, 3)
    contact_pairs: tuple[tuple[int, int], ...] = tuple((j, 18 + j) for j in range(18))
    physical_capacities: tuple[Q, ...] = (Q(1),) * 36
    initial_form_means: tuple[Q, Q] = (Q(0), Q(0))
    full_state_dimension: int = 72
    all_contact_durations_covered: bool = True
    donor_section_age_computed: bool = False
    source_synchronization_authenticated: bool = False
    donor_dwell_applied: bool = False
    charge_or_leak_prefix_applied: bool = False
    phase_jump_performed: bool = False
    inherited_donor_velocity_retained: bool = True
    inherited_nominal_velocity_relation: str = (
        "-gamma**2*q0 <= w0 <= 0 at the same isolated age"
    )
    actual_transport_commutation_assumed: bool = False
    zero_interaction_claimed: bool = False
    response_or_observational_equivalence_claimed: bool = False
    source_trajectory_evaluated: bool = False
    runtime_source_authenticated: bool = False
    clock: str = "tau=(1023/1024)*t; complete internal rows and reciprocal form inputs"
    scope: tuple[str, ...] = (
        "actual_first_section_donor_and_independently_mature_recipient",
        "same_original_nominal_weights_compare_but_do_not_replace_actual_sources",
        "inherited_nominal_velocity_is_correlated_with_its_phase_gap",
        "actual_unequal_transport_enters_both_error_rows_on_actual_form",
        "commuting_nominal_exchange_metric_on_each_mean_zero_leaf_only",
        "positive_source_resource_does_not_imply_fixed_pattern_accessibility",
        "all_time_target_exclusion_for_full_and_cut_ports_not_zero_interaction",
        "fixed_old_target_domain_without_performing_its_prefix_or_jump",
        "no_runtime_source_authentication_event_selector_or_general_no_go_claim",
    )


def _original_source_matches(source, geometry, form_error, phase_error):
    """Check consumed original-image and nominal-association premises."""
    common = source.completion_reports[0]
    return (
        max(form_error, phase_error) <= _SOURCE_CAP
        and source.form_error_bound == form_error
        and source.phase_error_bound == phase_error
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
        and common.initial_quotient_norm_squared_upper_bound == _INITIAL
        and common.initial_phase_norm_squared_upper_bound == _INITIAL - 2 * _ETA**2
        and common.initial_quotient_radius_margin == _RADIUS**2 - _INITIAL > 0
        and common.initial_energy_upper_bound == _ENERGY
        and common.storage_barrier
        == geometry.quotient_coercivity_lower_bound * _RADIUS**2
        and common.storage_margin == common.storage_barrier - _ENERGY > 0
    )


def _assess_mixed_resource_transfer_obstruction(
    *, form_error_bounds, phase_error_bounds
):
    """Rebuild source associations and bound the entire reacting pair history."""
    forms = _nonnegative_pair(form_error_bounds, "form_error_bounds")
    phases = _nonnegative_pair(phase_error_bounds, "phase_error_bounds")
    donor = _assess_mixed_registered_resource(
        form_error_bound=forms[0], phase_error_bound=phases[0]
    )
    recipient = _assess_mixed_circulation_readiness(
        form_error_bounds=(forms[1], forms[1]),
        phase_error_bounds=(phases[1], phases[1]),
        phase_jump_error_bound=_SIGMA,
        readout_error_bound=_OLD_READOUT,
    )
    registered = _derive_registered_phase_probe_geometry()
    geometry = registered.successor
    sources = (donor.source, recipient.sources[0])
    incident = tuple(
        sum(i in edge for edge in geometry.new_edge_pairs) for i in geometry.nodes
    )
    geometry_ok = (
        registered.cross_symmetry_certified
        and donor.source_geometry == recipient.source_geometry == geometry
        and donor.cross_incidence == registered.cross_incidence
        and geometry.completion == "common_sense"
        and geometry.nodes == tuple(range(18))
        and geometry.degrees == (9,) * 18
        and geometry.components == (tuple(range(18)),)
        and geometry.held_capacities == (Q(1),) * 18
        and len(geometry.internal_edges) == 18
        and all(w == 1 for _, _, w in geometry.internal_edges)
        and len(geometry.new_edge_pairs) == 63
        and len(geometry.edge_pairs) == 81
        and incident == (7,) * 18
        and geometry.support_certified
        and geometry.reference_stationary_certified
        and geometry.spectral_gap_certified
        and geometry.weighted_gap_lower_bound >= 3
        and geometry.bare_gap_lower_bound >= 4
        and geometry.quotient_coercivity_lower_bound >= Q(1, 5)
    )
    base_lower, base_upper, base_rate = _sine_common_quadratic_coefficients(
        gamma_lower=Q(1, 2750),
        gamma_upper=Q(1, 2500),
        cosine_lower=Q(5, 72),
        gap_lower=Q(3),
        capacity_lower=Q(4, 9),
        capacity_upper=Q(4, 9),
    )
    lower, upper, rate = _sine_common_quadratic_exchange_coefficients(
        gamma_lower=Q(1, 2750),
        gamma_upper=Q(1, 2500),
        cosine_lower=Q(5, 72),
        gap_lower=Q(3),
        capacity=Q(4, 9),
        exchange_norm_upper=Q(2),
    )
    donor_ok = (
        geometry_ok
        and donor.form_error_bound == forms[0]
        and donor.phase_error_bound == phases[0]
        and _original_source_matches(sources[0], geometry, forms[0], phases[0])
        and donor.source_association_certified
        and donor.deadline_certified
        and donor.nominal_reference_certified
        and donor.comparison_certified
        and donor.section_exists_certified
        and donor.section_orientation_certified
        and donor.registration_retained_after_section_certified
        and donor.section_mean_gap == _SECTION
        and donor.section_time_bounds == (Q(1), _WAIT)
        and donor.comparison_error_upper_bound == _SECTION_ERROR
        and donor.matrix_deviation_upper_bound == 14 * _WEIGHT_DEVIATION
        and donor.initial_comparison_norm_upper_bound == 14 * _ETA
        and donor.excess_storage_upper_bound == _ENERGY
        and donor.original_relative_phase_gap == _DELTA
        and donor.original_supplied_phase_origins == (-_DELTA / 2, _DELTA / 2)
        and donor.metric_lower_bound == base_lower
        and donor.metric_upper_bound == base_upper
        and donor.metric_decay_rate == base_rate
        and not donor.charge_or_leak_prefix_applied
        and not donor.new_external_input_supplied
        and not donor.new_phase_jump_performed
    )
    prefactor = base_upper * _SCALE**2 * _INITIAL / base_lower
    recipient_ok = (
        geometry_ok
        and recipient.form_error_bounds == (forms[1], forms[1])
        and recipient.phase_error_bounds == (phases[1], phases[1])
        and _original_source_matches(sources[1], geometry, forms[1], phases[1])
        and recipient.source_certified[0]
        and recipient.initial_inclusion_certified[0]
        and recipient.all_time_source_identity_certified[0]
        and recipient.readiness_certified[0]
        and recipient.initial_norm_squared_upper_bounds[0] == _INITIAL
        and recipient.metric_lower_bound == base_lower
        and recipient.metric_upper_bound == base_upper
        and recipient.metric_decay_rate == base_rate
        and recipient.phase_scale == _SCALE
        and recipient.auxiliary_metric_capacity == Q(4, 9)
        and recipient.waiting_duration == _WAIT
        and recipient.derived_mature_norm_upper_bound == _MATURE
        and recipient.physical_norm_prefactors[0] == prefactor
        and recipient.decay_upper_bound == Q(1, 2**39)
        and recipient.returned_norm_squared_upper_bounds[0]
        == prefactor / 2**39
        < _MATURE**2
        and recipient.decay_comparison_certified
        and base_rate * _WAIT > 30
        and 5**10 > 2**23
    )
    source_ok = donor_ok and recipient_ok
    half_difference = 14 * _WEIGHT_DEVIATION
    dissipation = 1 - half_difference
    initial_storage = _ENERGY + 9 * _MATURE**2
    initial_norm = 5 * _ENERGY + _MATURE**2
    full_squared = 5 * initial_storage
    full_norm = Q(1, 100)
    trap_margin = full_norm**2 - full_squared
    chart_margin = _RADIUS - full_norm
    trap = (
        source_ok
        and dissipation > 0
        and initial_norm < _RADIUS**2
        and initial_storage < Q(1, 65536)
        and trap_margin > 0
        and chart_margin > 0
    )
    pi = pi_interval()
    gamma = 1 / (1023 * pi)
    coupling = Q(13, 27) - _DELTA / (18 * pi)
    sine_coefficient = 1 + 2 * cos(pi / 9)
    q0 = (_SECTION - _SECTION_ERROR / 2, _SECTION + _SECTION_ERROR / 2)
    wedge_margin = 2 * coupling.lo / 3 - _GAMMA_UPPER**2
    wedges = (
        donor_ok
        and donor.gamma_bounds == gamma
        and donor.nominal_coupling_bounds == coupling
        and donor.regional_sine_coefficient_bounds == sine_coefficient
        and _GAMMA_LOWER <= gamma.lo <= gamma.hi <= _GAMMA_UPPER
        and Q(4, 9) < coupling.lo <= coupling.hi < Q(1, 2)
        and 2 < sine_coefficient.lo <= sine_coefficient.hi < 3
        and wedge_margin > 0
        and 0 < q0[0] <= q0[1] < _DELTA / 4
    )
    growth_margin = 16 - _SCALE**2 * upper / lower
    conservative_rate = base_rate / 8
    rate_margin = rate - conservative_rate
    extra = upper - base_upper
    base_ratio_margin = 8 - (1 + extra / base_lower)
    metric = (
        source_ok
        and trap
        and wedges
        and lower == base_lower > 0
        and extra == Q(27, 2)
        and growth_margin > 0
        and rate_margin > 0
        and base_ratio_margin > 0
    )
    initial_error = _MATURE + _SECTION_ERROR
    matrix_deviation = 14 * _WEIGHT_DEVIATION
    forcing = 2 * _WEIGHT_DEVIATION
    forcing_margin = forcing - 2 * matrix_deviation * full_norm / 9
    # The metric rate is a squared-norm rate. Integrating the norm decay
    # therefore uses 2/(base_rate/8), retaining both mismatch rows.
    comparison = 4 * (initial_error + 2 * forcing / conservative_rate)
    comparison_upper = 8 * _MATURE
    comparison_margin = comparison_upper - comparison
    compared = metric and forcing_margin > 0 and comparison_margin > 0
    marked_gap = q0[1] + 12 * _MATURE
    registration_margin = _DELTA / 4 - marked_gap
    identity = compared and registration_margin > 0
    actual_contrast = q0[1] + comparison_upper
    target_contrast = 2 * (_AMPLITUDE - _SIGMA) - _TARGET_BASE
    separation = target_contrast - actual_contrast
    # Only the fixed target definition is reused. A repeated recipient slot
    # or an unrelated old response verdict cannot remove that definition.
    target = _MixedTwoChannelCirculation
    target_ok = (
        geometry_ok
        and target.phase_jump_amplitude == _AMPLITUDE
        and target.phase_jump_error_cap == _SIGMA
        and _TARGET_BASE == Q(3, 2) * _MATURE
        and target.phase_radius == _RADIUS
        and target.registration_gap_bound == _DELTA / 4
        and target.contact_pairs == tuple((j, 18 + j) for j in range(18))
        and target.full_state_dimension == 72
        and 3 * (_AMPLITUDE + _SIGMA) + _TARGET_BASE < _RADIUS
    )
    resource_lower = Q(9, 10) * _SECTION**2
    phase_floor = Q(9, 5) * _AMPLITUDE**2
    resource_margin = resource_lower - phase_floor
    resource_ok = (
        donor_ok
        and donor.section_resource_certified
        and donor.section_excess_storage_lower_bound == resource_lower
        and donor.comparison_phase_floor_upper_bound == phase_floor
        and resource_margin > 0
    )
    unreachable = identity and target_ok and target_contrast > 0 and separation > 0
    unresolved = tuple(
        label
        for good, label in (
            (donor_ok, "actual_first_section_donor_association_unavailable"),
            (recipient_ok, "independent_mature_recipient_association_unavailable"),
            (trap, "actual_unequal_weight_all_time_trap_unavailable"),
            (wedges, "inherited_velocity_and_paired_nonlinear_wedges_unavailable"),
            (metric, "nominal_commuting_exchange_metric_unavailable"),
            (compared, "uniform_both_row_actual_to_nominal_comparison_unavailable"),
            (identity, "all_time_registration_or_constituent_identity_unavailable"),
            (target_ok, "unchanged_opposite_gap_target_domain_unavailable"),
            (unreachable, "all_time_target_contrast_separation_unavailable"),
            (resource_ok, "positive_donor_section_resource_comparison_unavailable"),
        )
        if not good
    )
    return _MixedResourceTransferObstruction(
        form_error_bounds=forms,
        phase_error_bounds=phases,
        donor=donor,
        recipient_readiness=recipient,
        sources=sources,
        source_geometry=geometry,
        donor_source_certified=donor_ok,
        recipient_source_certified=recipient_ok,
        source_association_certified=source_ok,
        new_edge_incident_counts=incident,
        transport_half_difference_upper_bound=half_difference,
        dissipation_margin=dissipation,
        initial_excess_storage_upper_bound=initial_storage,
        initial_norm_squared_upper_bound=initial_norm,
        full_norm_squared_upper_bound=full_squared,
        full_norm_upper_bound=full_norm,
        trap_norm_margin=trap_margin,
        acute_chart_margin=chart_margin,
        actual_trap_certified=trap,
        nominal_q0_bounds=q0,
        nominal_velocity_factor_upper_bound=_GAMMA_UPPER**2,
        nominal_wedge_margin=wedge_margin,
        nominal_wedges_certified=wedges,
        base_metric_lower_bound=base_lower,
        base_metric_upper_bound=base_upper,
        base_metric_decay_rate=base_rate,
        metric_lower_bound=lower,
        metric_upper_bound=upper,
        metric_decay_rate=rate,
        conservative_metric_decay_rate=conservative_rate,
        metric_growth_squared_margin=growth_margin,
        metric_rate_margin=rate_margin,
        metric_base_ratio_margin=base_ratio_margin,
        metric_certified=metric,
        initial_comparison_norm_upper_bound=initial_error,
        comparison_matrix_deviation_upper_bound=matrix_deviation,
        comparison_forcing_upper_bound=forcing,
        comparison_forcing_margin=forcing_margin,
        uniform_comparison_bound=comparison,
        uniform_comparison_upper_bound=comparison_upper,
        uniform_comparison_margin=comparison_margin,
        comparison_certified=compared,
        marked_gap_upper_bound=marked_gap,
        registration_margin=registration_margin,
        all_time_identity_certified=identity,
        actual_gap_contrast_upper_bound=actual_contrast,
        target_gap_contrast_lower_bound=target_contrast,
        target_separation_margin=separation,
        target_domain_certified=target_ok,
        donor_resource_lower_bound=resource_lower,
        comparison_phase_floor_upper_bound=phase_floor,
        resource_floor_margin=resource_margin,
        positive_donor_resource_certified=resource_ok,
        connected_target_unreachable=unreachable,
        disconnected_target_unreachable=unreachable,
        certified=not unresolved,
        status=(
            "conditional_all_time_resource_transfer_obstruction"
            if not unresolved
            else "unavailable"
        ),
        unresolved_conditions=unresolved,
    )
