"""Finite form-only preparation of the actual common UM circulation source.

The constant regional bias is specified from a target family and the held law,
not from an observed response. Phase remains live. The endpoint is admitted to
the existing contact estimates by a domain lemma, not by claiming that their
old instantaneous phase jump or preparation history occurred.
"""

from dataclasses import dataclass
from fractions import Fraction as Q

from .._exact_time import exact_or_represented_real as _exact
from ..mathematics._rational_interval import I, cos, pi_interval, sin
from ._mixed_circulation_readiness import (
    _assess_mixed_circulation_readiness,
    _MixedCirculationReadiness,
)
from ._mixed_reciprocal_partner import _nonnegative, _nonnegative_pair
from ._mixed_registered_phase_probe import _derive_registered_phase_probe_geometry
from ._mixed_two_channel_circulation import _MixedTwoChannelCirculation
from ._sine_coupling_successor import _coupling_successor_field
from ._sine_lyapunov import _sine_common_quadratic_coefficients

_AMPLITUDE = Q(1, 8192)
_TARGET_ERROR_CAP = Q(1, 10**8)
_READOUT_CAP = Q(1, 10**6)
_SOURCE_CAP = Q(1, 10**10)
_RADIUS = Q(1, 32)
_REGISTRATION = Q(1, 4096)
_MATURE = Q(1, 2**28)
_TIME = Q(10**10)
_PREFIX_TIME = Q(1, 32)
_GAMMA_UPPER = Q(1, 3000)
_WEIGHT_DEVIATION = Q(2, 10**60)
_SCALE = Q(6, 5)
_DECAY = Q(1, 2**39)
_ENDPOINT = Q(1, 2**30)
_RECORD_FLOOR = Q(1503, 400000000)


def _bias_constants():
    """Enclose ideal model constants without replacing their values by midpoints."""
    pi = pi_interval()
    return 1 / (1023 * pi), 1 + 2 * cos(pi / 9)


def _mixed_phase_bias_field(*, new_edge_conductances, target_phase_gap):
    """Add the fixed balanced form bias; retain all 36 internal evolution rows.

    The signed target belongs to the frozen positive or negative amplitude
    band. It selects an ideal time-constant command, not arbitrary actuator
    noise. Actual source association is not established by these generic
    admitted weights or by evaluating a constructed state.
    """
    target = _exact(target_phase_gap, "target_phase_gap")
    if (
        not _AMPLITUDE - _TARGET_ERROR_CAP
        <= abs(target)
        <= _AMPLITUDE + _TARGET_ERROR_CAP
    ):
        raise ValueError(
            "target_phase_gap must lie in the frozen signed amplitude bands"
        )
    internal = _coupling_successor_field(
        completion="common_sense",
        new_edge_conductances=new_edge_conductances,
        loss_normalization="bare_degree",
    )
    gamma, coefficient = _bias_constants()
    first_input = -gamma * coefficient * sin(I(target)) / 9

    def flow(values):
        rates = internal(values)
        return (
            tuple(
                value + (first_input if i < 9 else -first_input)
                for i, value in enumerate(rates[:18])
            )
            + rates[18:]
        )

    return flow


@dataclass(frozen=True)
class _MixedFormPhasePreparation:
    """Original readiness, controlled endpoint and contact algebra kept separate.

    The contact reference retains its historical meaning. Only its admitted
    contact-domain bounds are used here, after establishing endpoint inclusion.
    Its old phase-jump work is not part of this preparation's resource ledger.
    All elapsed source/control/measurement associations remain conditional.
    """

    form_error_bounds: tuple[Q, Q]
    phase_error_bounds: tuple[Q, Q]
    target_gap_error_bound: Q
    readout_error_bound: Q
    readiness: _MixedCirculationReadiness
    contact_reference: _MixedTwoChannelCirculation | None
    source_association_certified: bool
    target_gap_bounds: tuple[Q, Q]
    sine_coefficient_bounds: tuple[Q, Q]
    gamma_bounds: tuple[Q, Q]
    nominal_coupling_bounds: tuple[Q, Q]
    target_family_certified: bool
    prefix_norm_upper_bound: Q
    carried_common_form_norm_upper_bound: Q
    target_norm_upper_bound: Q
    initial_centered_norm_upper_bound: Q
    initial_norm_margin: Q
    metric_lower_bound: Q
    metric_upper_bound: Q
    metric_decay_rate: Q
    physical_metric_ratio: Q
    norm_growth_squared_margin: Q
    centered_norm_upper_bound: Q
    original_norm_upper_bound: Q
    shifted_chart_margin: Q
    metric_domain_certified: bool
    nominal_form_norm_upper_bound: Q
    nominal_wedge_margin: Q
    weight_matrix_deviation_upper_bound: Q
    actual_reference_error_upper_bound: Q
    actual_reference_error_margin: Q
    preparation_marked_gap_upper_bound: Q
    preparation_registration_margin: Q
    preparation_identity_certified: bool
    endpoint_prefactor: Q
    endpoint_prefactor_margin: Q
    decay_exponent_margin: Q
    centered_endpoint_error_squared_upper_bound: Q
    centered_endpoint_error_bound: Q
    endpoint_error_squared_margin: Q
    endpoint_base_norm_upper_bound: Q
    endpoint_domain_margin: Q
    endpoint_certified: bool
    contact_domain_certified: bool
    contact_response_certified: bool
    bias_node_amplitude_upper_bound: Q
    bias_net_work_absolute_upper_bound: Q
    bias_absolute_work_upper_bound: Q
    bias_absolute_form_amount_upper_bound: Q
    prefix_storage_work_upper_bound: Q
    contact_storage_work_upper_bound: Q
    work_certified: bool
    certified: bool
    status: str
    unresolved_conditions: tuple[str, ...]
    source_error_cap: Q = _SOURCE_CAP
    target_gap_error_cap: Q = _TARGET_ERROR_CAP
    readout_error_cap: Q = _READOUT_CAP
    target_gap_amplitude: Q = _AMPLITUDE
    bias_duration: Q = _TIME
    original_clock_bias_duration: Q = Q(1024, 1023) * _TIME
    prefix_charge_duration: Q = _PREFIX_TIME
    prefix_charge_amplitude: Q = Q(0)
    prefix_leak_duration: Q = _PREFIX_TIME
    derived_source_radius: Q = _MATURE
    phase_scale: Q = _SCALE
    auxiliary_metric_capacity: Q = Q(4, 9)
    decay_upper_bound: Q = _DECAY
    actual_weight_deviation_upper_bound: Q = _WEIGHT_DEVIATION
    conservative_record_separation_floor: Q = _RECORD_FLOOR
    full_preparation_state_dimension: int = 72
    distinct_bias_commands: int = 2
    actuated_form_coordinates: int = 36
    phase_coordinates_directly_actuated: int = 0
    phase_jump_performed: bool = False
    old_jump_work_included: bool = False
    exact_endpoint_claimed: bool = False
    common_form_reset: bool = False
    comparator_prepares_source: bool = False
    runtime_source_authenticated: bool = False
    source_trajectory_evaluated: bool = False
    actuator_family: str = (
        "time-constant calibrated regional bias indexed by independent latent target gaps"
    )
    clock: str = "tau=(1023/1024)*t; both rows and external form input use tau"
    scope: tuple[str, ...] = (
        "actual_readiness_then_original_zero_charge_and_leak_then_constant_bias",
        "common_form_from_leak_and_actual_phase_gauges_are_retained",
        "nominal_regional_wedge_is_a_comparison_not_the_actual_weighted_carrier",
        "endpoint_base_is_an_auxiliary_representation_not_the_old_prefix_endpoint",
        "old_contact_estimates_apply_by_domain_inclusion_without_a_phase_jump",
        "net_work_absolute_work_and_supplied_form_amount_are_distinct",
        "no_arbitrary_actuator_noise_exact_collapse_or_autonomous_occurrence_claim",
        "conditional_model_certificate_not_executed_preparation_or_physical_validation",
    )


def _assess_mixed_form_phase_preparation(
    *,
    form_error_bounds,
    phase_error_bounds,
    target_gap_error_bound,
    readout_error_bound,
):
    """Rebuild source and fixed-control bounds before admitting contact reuse."""
    forms = _nonnegative_pair(form_error_bounds, "form_error_bounds")
    phases = _nonnegative_pair(phase_error_bounds, "phase_error_bounds")
    error = _nonnegative(target_gap_error_bound, "target_gap_error_bound")
    readout = _nonnegative(readout_error_bound, "readout_error_bound")
    readiness = _assess_mixed_circulation_readiness(
        form_error_bounds=forms,
        phase_error_bounds=phases,
        phase_jump_error_bound=error,
        readout_error_bound=readout,
    )
    registered = _derive_registered_phase_probe_geometry()
    geometry = registered.successor
    contact = readiness.circulation
    source_ok = (
        readiness.form_error_bounds == forms
        and readiness.phase_error_bounds == phases
        and readiness.phase_jump_error_bound == error
        and readiness.readout_error_bound == readout
        and readiness.source_geometry == geometry
        and all(readiness.source_certified)
        and all(readiness.initial_inclusion_certified)
        and all(readiness.all_time_source_identity_certified)
        and all(readiness.readiness_certified)
        and readiness.later_isolated_age_guarantee_certified
        and readiness.handoff_norm_upper_bounds == (_MATURE, _MATURE)
        and readiness.derived_mature_norm_upper_bound == _MATURE
        and readiness.waiting_duration == _TIME
        and all(
            0 <= value < _MATURE**2
            for value in readiness.returned_norm_squared_upper_bounds
        )
        and registered.cross_symmetry_certified
        and geometry.completion == "common_sense"
        and geometry.nodes == tuple(range(18))
        and geometry.degrees == (9,) * 18
        and geometry.held_capacities == (Q(1),) * 18
        and geometry.weighted_gap_lower_bound >= 3
        and all(
            source.form_error_bound == rx
            and source.phase_error_bound == rt
            and max(rx, rt) <= _SOURCE_CAP
            and source.completion_reports[0].geometry == geometry
            and source.phase_curvature_bounds.lo >= Q(1, 10)
            and source.new_weight_range == (Q(3, 4), Q(1))
            and 0 <= source.edge_weight_deviation_upper_bound <= _WEIGHT_DEVIATION
            and source.precontact_reference.initial_form_means == (Q(0), Q(0))
            for source, rx, rt in zip(readiness.sources, forms, phases, strict=True)
        )
    )
    gamma, coefficient = _bias_constants()
    coupling = Q(13, 27) - Q(1, 1024) / (18 * pi_interval())
    target_min, target_max = _AMPLITUDE - error, _AMPLITUDE + error
    target_ok = (
        error <= _TARGET_ERROR_CAP
        and 0 < target_min <= target_max < Q(1, 8000)
        and 2 < coefficient.lo <= coefficient.hi < 3
        and Q(1, 3300) <= gamma.lo <= gamma.hi <= _GAMMA_UPPER
        and Q(4, 9) < coupling.lo <= coupling.hi < Q(1, 2)
    )
    lower, upper, rate = _sine_common_quadratic_coefficients(
        gamma_lower=Q(1, 2750),
        gamma_upper=Q(1, 2500),
        cosine_lower=Q(5, 72),
        gap_lower=Q(3),
        capacity_lower=Q(4, 9),
        capacity_upper=Q(4, 9),
    )
    prefix = Q(3, 2) * _MATURE
    common_form = _PREFIX_TIME * prefix
    target_norm = 3 * target_max
    initial = prefix + target_norm
    initial_margin = Q(1, 2048) - initial
    ratio = _SCALE**2 * upper / lower
    growth_margin = Q(9, 4) - ratio
    centered_norm = Q(3, 2) * initial
    original_norm = centered_norm + target_norm + common_form
    chart_margin = _RADIUS - original_norm
    metric_ok = (
        source_ok
        and target_ok
        and initial_margin > 0
        and ratio < 2
        and growth_margin > 0
        and rate > Q(3, 10**9)
        and centered_norm < Q(1, 1024)
        and chart_margin > 0
    )
    nominal_form = Q(27, 8) * _GAMMA_UPPER * target_max
    wedge_margin = 2 * coupling.lo / 3 - _GAMMA_UPPER**2
    matrix_deviation = 14 * _WEIGHT_DEVIATION
    comparison = Q(3, 2) * (prefix + 2 * _TIME * _WEIGHT_DEVIATION)
    comparison_margin = 2 * prefix - comparison
    marked_gap = target_max + 3 * prefix
    registration_margin = _REGISTRATION - marked_gap
    identity = (
        metric_ok
        and nominal_form < Q(1, 1024)
        and wedge_margin > 0
        and comparison_margin > 0
        and registration_margin > 0
    )
    prefactor = ratio * initial**2
    prefactor_margin = Q(1, 2**21) - prefactor
    exponent_margin = rate * _TIME - 30
    endpoint_square = prefactor * _DECAY
    endpoint_margin = _ENDPOINT**2 - endpoint_square
    endpoint_base = _ENDPOINT + common_form
    domain_margin = prefix - endpoint_base
    endpoint_ok = (
        identity
        and prefactor_margin > 0
        and exponent_margin > 0
        and 5**10 > 2**23
        and endpoint_margin > 0
        and _ENDPOINT == _MATURE / 4
        and domain_margin > 0
    )
    contact_ok = (
        endpoint_ok
        and contact is not None
        and readiness.handoff_association_certified
        and contact.form_error_bounds == forms
        and contact.phase_error_bounds == phases
        and contact.mature_norm_upper_bounds == (_MATURE, _MATURE)
        and contact.phase_jump_error_bound == error
        and contact.readout_error_bound == readout
        and contact.source_geometry == geometry
        and contact.cross_incidence == registered.cross_incidence
        and contact.source_association_certified
        and contact.prefix_norm_upper_bound == prefix
        and contact.charge_amplitude == 0
        and contact.charge_duration == contact.prefix_leak_duration == _PREFIX_TIME
        and contact.phase_jump_amplitude == _AMPLITUDE
        and contact.contact_duration == 512
        and contact.classifier_threshold == Q(1, 400000)
        and contact.contact_pairs == tuple((j, 18 + j) for j in range(18))
        and contact.full_state_dimension == 72
        and contact.independently_observed_channels == 1
        and contact.gamma_bounds == (Q(1, 3300), _GAMMA_UPPER)
        and contact.cosine_lower_bound == Q(1, 10)
        and contact.edge_weight_deviation_upper_bound == _WEIGHT_DEVIATION
        and all(
            child.form_error_bound == original.form_error_bound == rx
            and child.phase_error_bound == original.phase_error_bound == rt
            and child.completion_reports[0].geometry
            == original.completion_reports[0].geometry
            == geometry
            and child.new_weight_range == original.new_weight_range
            and child.edge_weight_deviation_upper_bound
            == original.edge_weight_deviation_upper_bound
            and child.phase_reset_error_bound == original.phase_reset_error_bound
            and child.gauge_phase_deviation_upper_bound
            == original.gauge_phase_deviation_upper_bound
            and child.precontact_reference.initial_form_means
            == original.precontact_reference.initial_form_means
            and child.gamma_range == original.gamma_range
            for child, original, rx, rt in zip(
                contact.sources, readiness.sources, forms, phases, strict=True
            )
        )
    )
    response_ok = (
        contact_ok
        and readout <= _READOUT_CAP
        and contact.finite_identity_certified
        and contact.circulation_certified
        and contact.independent_hypotheses_separated
        and contact.record_separation_margin >= _RECORD_FLOOR
        and min(contact.full_threshold_margin, contact.comparator_threshold_margin) > 0
    )
    node_amplitude = _GAMMA_UPPER * target_max / 3
    net_work = 18 * target_max * (target_norm + prefix + _ENDPOINT)
    absolute_work = 108 * _GAMMA_UPPER * target_max / (2048 * rate)
    form_amount = 12 * _TIME * _GAMMA_UPPER * target_max
    prefix_work = 18 * _PREFIX_TIME * prefix**2
    contact_work = 36 * 512 * Q(1, 1024) ** 2
    work_ok = (
        endpoint_ok
        and net_work < Q(1, 10**6)
        and absolute_work < 1
        and form_amount < 5000
    )
    unresolved = tuple(
        label
        for good, label in (
            (
                source_ok,
                "original_source_readiness_or_actual_weight_association_unavailable",
            ),
            (target_ok, "constant_calibrated_target_family_unavailable"),
            (metric_ok, "shifted_full_state_metric_or_bootstrap_unavailable"),
            (identity, "whole_preparation_winding_or_marked_registration_unavailable"),
            (endpoint_ok, "finite_full_endpoint_inclusion_unavailable"),
            (contact_ok, "contact_domain_or_original_protocol_association_unavailable"),
            (response_ok, "independent_contact_record_separation_unavailable"),
            (work_ok, "separate_input_resource_budgets_unavailable"),
        )
        if not good
    )
    return _MixedFormPhasePreparation(
        form_error_bounds=forms,
        phase_error_bounds=phases,
        target_gap_error_bound=error,
        readout_error_bound=readout,
        readiness=readiness,
        contact_reference=contact,
        source_association_certified=source_ok,
        target_gap_bounds=(target_min, target_max),
        sine_coefficient_bounds=(coefficient.lo, coefficient.hi),
        gamma_bounds=(gamma.lo, gamma.hi),
        nominal_coupling_bounds=(coupling.lo, coupling.hi),
        target_family_certified=target_ok,
        prefix_norm_upper_bound=prefix,
        carried_common_form_norm_upper_bound=common_form,
        target_norm_upper_bound=target_norm,
        initial_centered_norm_upper_bound=initial,
        initial_norm_margin=initial_margin,
        metric_lower_bound=lower,
        metric_upper_bound=upper,
        metric_decay_rate=rate,
        physical_metric_ratio=ratio,
        norm_growth_squared_margin=growth_margin,
        centered_norm_upper_bound=centered_norm,
        original_norm_upper_bound=original_norm,
        shifted_chart_margin=chart_margin,
        metric_domain_certified=metric_ok,
        nominal_form_norm_upper_bound=nominal_form,
        nominal_wedge_margin=wedge_margin,
        weight_matrix_deviation_upper_bound=matrix_deviation,
        actual_reference_error_upper_bound=comparison,
        actual_reference_error_margin=comparison_margin,
        preparation_marked_gap_upper_bound=marked_gap,
        preparation_registration_margin=registration_margin,
        preparation_identity_certified=identity,
        endpoint_prefactor=prefactor,
        endpoint_prefactor_margin=prefactor_margin,
        decay_exponent_margin=exponent_margin,
        centered_endpoint_error_squared_upper_bound=endpoint_square,
        centered_endpoint_error_bound=_ENDPOINT,
        endpoint_error_squared_margin=endpoint_margin,
        endpoint_base_norm_upper_bound=endpoint_base,
        endpoint_domain_margin=domain_margin,
        endpoint_certified=endpoint_ok,
        contact_domain_certified=contact_ok,
        contact_response_certified=response_ok,
        bias_node_amplitude_upper_bound=node_amplitude,
        bias_net_work_absolute_upper_bound=net_work,
        bias_absolute_work_upper_bound=absolute_work,
        bias_absolute_form_amount_upper_bound=form_amount,
        prefix_storage_work_upper_bound=prefix_work,
        contact_storage_work_upper_bound=contact_work,
        work_certified=work_ok,
        certified=not unresolved,
        status=(
            "conditional_form_preparation_and_circulation"
            if not unresolved
            else "unavailable"
        ),
        unresolved_conditions=unresolved,
    )
