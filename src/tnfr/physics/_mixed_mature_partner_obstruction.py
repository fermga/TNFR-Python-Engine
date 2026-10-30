"""An all-time resource obstruction for two reacting mature UM objects.

The existing full matching-slot field evolves both objects and both phase
rows. Its actual-weight excess storage bounds the entire reachable family.
The old circulation reader supplies only a fixed target domain; no phase
jump, response or control history is claimed to occur in this preparation.
"""

from dataclasses import dataclass
from fractions import Fraction as Q

from ._mixed_circulation_readiness import (
    _assess_mixed_circulation_readiness,
    _MixedCirculationReadiness,
)
from ._mixed_reciprocal_partner import _nonnegative_pair
from ._mixed_two_channel_circulation import _MixedTwoChannelCirculation
from ._sine_coupling_successor import _derive_coupling_successor_geometry

_SOURCE_CAP = Q(1, 10**10)
_MATURE = Q(1, 2**28)
_PREFIX_TIME = Q(1, 32)
_PREFIX = Q(3, 2) * _MATURE
_AMPLITUDE = Q(1, 8192)
_TARGET_ERROR = Q(1, 10**8)
_OLD_READOUT = Q(1, 10**6)
_RADIUS = Q(1, 32)
_REGISTRATION = Q(1, 4096)
_DELTA = Q(1, 1024)
_WAITING_RADIUS = Q(20, 10**70)
_WEIGHT_DEVIATION = Q(2, 10**60)
_COERCIVITY = Q(1, 5)
_STORAGE_UPPER = Q(9)


@dataclass(frozen=True)
class _MixedMaturePartnerObstruction:
    """Conditional reachable-set and target separation, not a response verdict.

    Excess storage is measured above the formed winding patterns. Their
    nonzero baseline storage is not treated as freely available. Independent
    actual source errors, common forms and phase gauges are retained.
    ``transport_difference_upper_bound`` bounds the half difference
    ``||(H1-H2)/2||``; the full difference has twice that allowance.
    """

    form_error_bounds: tuple[Q, Q]
    phase_error_bounds: tuple[Q, Q]
    readiness: _MixedCirculationReadiness
    target_reference: _MixedTwoChannelCirculation | None
    source_association_certified: bool
    target_domain_certified: bool
    prefix_norm_upper_bound: Q
    common_form_norm_upper_bound: Q
    new_edge_incident_counts: tuple[int, ...]
    transport_difference_upper_bound: Q
    dissipation_margin: Q
    storage_balance_certified: bool
    initial_excess_storage_upper_bound: Q
    full_norm_squared_upper_bound: Q
    full_norm_upper_bound: Q
    norm_squared_margin: Q
    acute_chart_margin: Q
    marked_gap_upper_bound: Q
    marked_gap_squared_margin: Q
    registration_margin: Q
    all_time_identity_certified: bool
    target_radius_upper_bound: Q
    target_marked_gap_lower_bound: Q
    target_gap_separation_margin: Q
    target_phase_norm_lower_bound: Q
    target_excess_storage_lower_bound: Q
    resource_shortfall_lower_bound: Q
    geometric_obstruction_certified: bool
    resource_obstruction_certified: bool
    connected_target_unreachable: bool
    disconnected_target_unreachable: bool
    certified: bool
    status: str
    unresolved_conditions: tuple[str, ...]
    source_error_cap: Q = _SOURCE_CAP
    derived_source_radius: Q = _MATURE
    isolated_readiness_duration: Q = Q(10**10)
    prefix_charge_duration: Q = _PREFIX_TIME
    prefix_charge_amplitude: Q = Q(0)
    prefix_leak_duration: Q = _PREFIX_TIME
    target_gap_amplitude: Q = _AMPLITUDE
    target_gap_error_bound: Q = _TARGET_ERROR
    target_reference_readout_error_bound: Q = _OLD_READOUT
    actual_weight_deviation_upper_bound: Q = _WEIGHT_DEVIATION
    full_transport_difference_upper_bound: Q = 28 * _WEIGHT_DEVIATION
    weighted_gap_lower_bound: Q = Q(3)
    bare_gap_lower_bound: Q = Q(4)
    transport_norm_upper_bound: Q = Q(18)
    bare_hessian_norm_upper_bound: Q = Q(18)
    phase_cosine_lower_bound: Q = Q(1, 10)
    excess_storage_coercivity: Q = _COERCIVITY
    excess_storage_upper_coefficient: Q = _STORAGE_UPPER
    exchange_strength: Q = Q(1)
    common_form_difference_decay_rate: Q = Q(2)
    contact_pairs: tuple[tuple[int, int], ...] = tuple((j, 18 + j) for j in range(18))
    physical_capacities: tuple[Q, ...] = (Q(1),) * 36
    full_state_dimension: int = 72
    all_contact_durations_covered: bool = True
    common_form_reset: bool = False
    contact_work_nonpositive_claimed: bool = False
    zero_interaction_claimed: bool = False
    arbitrary_common_offsets_admitted_as_sources: bool = False
    phase_jump_performed: bool = False
    source_trajectory_evaluated: bool = False
    runtime_source_authenticated: bool = False
    clock: str = "tau=(1023/1024)*t; both internal rows and reciprocal form inputs"
    scope: tuple[str, ...] = (
        "actual_independent_mature_common_UM_sources_after_unchanged_prefix",
        "existing_full_matching_slot_field_reacts_on_both_sources",
        "actual_weighted_form_and_bare_sine_storage_not_a_quarter_law_transfer",
        "unequal_weight_contact_work_is_bounded_not_assumed_nonpositive",
        "common_forms_contract_separately_and_each_phase_gauge_is_conserved",
        "all_time_obstruction_for_this_source_port_and_target_only",
        "target_reference_is_not_a_reached_response_or_performed_jump",
        "nonzero_winding_storage_is_not_available_excess_resource",
        "no_universal_partner_observation_or_physical_impossibility_claim",
    )


def _assess_mixed_mature_partner_obstruction(*, form_error_bounds, phase_error_bounds):
    """Rebuild actual sources and the fixed target before bounding reachability."""
    forms = _nonnegative_pair(form_error_bounds, "form_error_bounds")
    phases = _nonnegative_pair(phase_error_bounds, "phase_error_bounds")
    readiness = _assess_mixed_circulation_readiness(
        form_error_bounds=forms,
        phase_error_bounds=phases,
        phase_jump_error_bound=_TARGET_ERROR,
        readout_error_bound=_OLD_READOUT,
    )
    geometry = _derive_coupling_successor_geometry(completion="common_sense")
    target = readiness.circulation
    incident = tuple(
        sum(i in edge for edge in geometry.new_edge_pairs) for i in geometry.nodes
    )
    geometry_ok = (
        readiness.source_geometry == geometry
        and geometry.completion == "common_sense"
        and geometry.nodes == tuple(range(18))
        and geometry.degrees == (9,) * 18
        and geometry.components == (tuple(range(18)),)
        and geometry.held_capacities == (Q(1),) * 18
        and len(geometry.internal_edges) == 18
        and all(weight == 1 for _, _, weight in geometry.internal_edges)
        and len(geometry.new_edge_pairs) == 63
        and incident == (7,) * 18
        and geometry.support_certified
        and geometry.reference_stationary_certified
        and geometry.spectral_gap_certified
        and geometry.weighted_gap_lower_bound >= 3
        and geometry.bare_gap_lower_bound >= 4
        and geometry.quotient_coercivity_lower_bound >= _COERCIVITY
    )
    source_ok = (
        geometry_ok
        and readiness.form_error_bounds == forms
        and readiness.phase_error_bounds == phases
        and readiness.phase_jump_error_bound == _TARGET_ERROR
        and readiness.readout_error_bound == _OLD_READOUT
        and all(readiness.source_certified)
        and all(readiness.initial_inclusion_certified)
        and all(readiness.all_time_source_identity_certified)
        and all(readiness.readiness_certified)
        and readiness.later_isolated_age_guarantee_certified
        and readiness.derived_mature_norm_upper_bound == _MATURE
        and readiness.handoff_norm_upper_bounds == (_MATURE, _MATURE)
        and readiness.waiting_duration == 10**10
        and all(
            0 <= value < _MATURE**2
            for value in readiness.returned_norm_squared_upper_bounds
        )
        and all(
            source.form_error_bound == rx
            and source.phase_error_bound == rt
            and max(rx, rt) <= _SOURCE_CAP
            and source.source_certified
            and source.isolated_waiting_certified
            and source.geometry_certified
            and source.conditional_image_certified
            and source.completion_reports[0].geometry == geometry
            and source.phase_radius == _RADIUS
            and source.relative_phase_gap == _DELTA
            and source.waiting_radius == _WAITING_RADIUS
            and source.phase_reset_error_bound == 3 * _WAITING_RADIUS
            and 0 <= source.gauge_phase_deviation_upper_bound <= 6 * _WAITING_RADIUS
            and source.phase_cosine_lower_bound >= Q(1, 10)
            and source.phase_curvature_bounds.lo >= Q(1, 10)
            and source.phase_curvature_margin > 0
            and Q(1, 3300)
            <= source.gamma_range[0]
            <= source.gamma_range[1]
            <= Q(1, 3000)
            and source.new_weight_range == (Q(3, 4), Q(1))
            and 0 <= source.edge_weight_deviation_upper_bound <= _WEIGHT_DEVIATION
            and source.precontact_reference.relative_phase_gap == _DELTA
            and source.precontact_reference.supplied_phase_origins
            == (-_DELTA / 2, _DELTA / 2)
            and source.precontact_reference.initial_form_means == (Q(0), Q(0))
            and source.completion_reports[0].degree_form_limit_bounds == ((Q(0), Q(0)),)
            for source, rx, rt in zip(readiness.sources, forms, phases, strict=True)
        )
    )
    prefix_ok = (
        source_ok
        and target is not None
        and readiness.handoff_association_certified
        and target.form_error_bounds == forms
        and target.phase_error_bounds == phases
        and target.source_geometry == geometry
        and target.source_association_certified
        and target.mature_norm_upper_bounds == (_MATURE, _MATURE)
        and target.charge_amplitude == 0
        and target.charge_duration == target.prefix_leak_duration == _PREFIX_TIME
        and target.prefix_norm_upper_bound == _PREFIX
        and target.prefix_growth_margin > 0
        and all(
            child.form_error_bound == original.form_error_bound == rx
            and child.phase_error_bound == original.phase_error_bound == rt
            and child.source_certified
            and child.isolated_waiting_certified
            and child.geometry_certified
            and child.conditional_image_certified
            and child.completion_reports[0].geometry
            == original.completion_reports[0].geometry
            == geometry
            and child.new_weight_range == original.new_weight_range
            and child.edge_weight_deviation_upper_bound
            == original.edge_weight_deviation_upper_bound
            and child.gamma_range == original.gamma_range
            and child.relative_phase_gap == original.relative_phase_gap
            and child.waiting_radius == original.waiting_radius
            and child.phase_reset_error_bound == original.phase_reset_error_bound
            and child.gauge_phase_deviation_upper_bound
            == original.gauge_phase_deviation_upper_bound
            and child.precontact_reference.initial_form_means
            == original.precontact_reference.initial_form_means
            and child.precontact_reference.relative_phase_gap
            == original.precontact_reference.relative_phase_gap
            and child.precontact_reference.supplied_phase_origins
            == original.precontact_reference.supplied_phase_origins
            for child, original, rx, rt in zip(
                target.sources, readiness.sources, forms, phases, strict=True
            )
        )
    )
    target_radius = 3 * (_AMPLITUDE + _TARGET_ERROR) + _PREFIX
    target_ok = (
        prefix_ok
        and target.phase_jump_amplitude == _AMPLITUDE
        and target.phase_jump_error_bound == _TARGET_ERROR
        and target.readout_error_bound == _OLD_READOUT
        and target.contact_pairs == tuple((j, 18 + j) for j in range(18))
        and target.full_state_dimension == 72
        and target.phase_radius == _RADIUS
        and target.registration_gap_bound == _REGISTRATION
        and target.edge_weight_deviation_upper_bound == _WEIGHT_DEVIATION
        and target_radius < _RADIUS
    )
    difference = 14 * _WEIGHT_DEVIATION
    dissipation = 1 - difference
    balance_ok = prefix_ok and dissipation > 0
    initial_storage = _STORAGE_UPPER * _PREFIX**2
    # Centered N(t)^2 <= 45*N(0)^2 and the common-form norm contracts.
    # Their original orthogonal sum is <= prefix^2, so the full norm is
    # <= 45*prefix^2 without resetting or adding an independent mean budget.
    full_squared = initial_storage / _COERCIVITY
    full_norm = 7 * _PREFIX
    norm_margin = full_norm**2 - full_squared
    chart_margin = _RADIUS - full_norm
    marked_gap = 10 * _PREFIX
    marked_margin = marked_gap**2 - 2 * full_squared
    registration_margin = _REGISTRATION - marked_gap
    retention = (
        balance_ok
        and norm_margin > 0
        and chart_margin > 0
        and marked_margin > 0
        and registration_margin > 0
    )
    target_gap = _AMPLITUDE - _TARGET_ERROR - Q(3, 2) * _PREFIX
    gap_separation = target_gap - marked_gap
    target_phase = 3 * (_AMPLITUDE - _TARGET_ERROR) - _PREFIX
    target_storage = _COERCIVITY * target_phase**2
    resource_gap = target_storage - initial_storage
    geometric = retention and target_ok and target_gap > 0 and gap_separation > 0
    resource = retention and target_ok and target_phase > 0 and resource_gap > 0
    unreachable = geometric and resource
    unresolved = tuple(
        label
        for good, label in (
            (source_ok, "actual_source_readiness_or_weight_association_unavailable"),
            (prefix_ok, "unchanged_prefix_or_common_mode_association_unavailable"),
            (target_ok, "fixed_full_target_domain_unavailable"),
            (balance_ok, "actual_weight_excess_storage_dissipation_unavailable"),
            (retention, "all_time_full_state_or_identity_bound_unavailable"),
            (geometric, "marked_gap_reachable_target_separation_unavailable"),
            (resource, "necessary_target_excess_resource_gap_unavailable"),
        )
        if not good
    )
    return _MixedMaturePartnerObstruction(
        form_error_bounds=forms,
        phase_error_bounds=phases,
        readiness=readiness,
        target_reference=target,
        source_association_certified=prefix_ok,
        target_domain_certified=target_ok,
        prefix_norm_upper_bound=_PREFIX,
        common_form_norm_upper_bound=_PREFIX_TIME * _PREFIX,
        new_edge_incident_counts=incident,
        transport_difference_upper_bound=difference,
        dissipation_margin=dissipation,
        storage_balance_certified=balance_ok,
        initial_excess_storage_upper_bound=initial_storage,
        full_norm_squared_upper_bound=full_squared,
        full_norm_upper_bound=full_norm,
        norm_squared_margin=norm_margin,
        acute_chart_margin=chart_margin,
        marked_gap_upper_bound=marked_gap,
        marked_gap_squared_margin=marked_margin,
        registration_margin=registration_margin,
        all_time_identity_certified=retention,
        target_radius_upper_bound=target_radius,
        target_marked_gap_lower_bound=target_gap,
        target_gap_separation_margin=gap_separation,
        target_phase_norm_lower_bound=target_phase,
        target_excess_storage_lower_bound=target_storage,
        resource_shortfall_lower_bound=resource_gap,
        geometric_obstruction_certified=geometric,
        resource_obstruction_certified=resource,
        connected_target_unreachable=unreachable,
        disconnected_target_unreachable=unreachable,
        certified=not unresolved,
        status=(
            "conditional_all_time_target_obstruction"
            if not unresolved
            else "unavailable"
        ),
        unresolved_conditions=unresolved,
    )
