"""Common-form storage, conjugate work and fine-law inheritance.

The selected quadratic extension leaves every existing evolution row intact.
It separates nominal coefficient freedom, exact fixed-port total dissipation
and finite errors relative to actual regional projections and a carried
reference. No previously predicted form or phase record is consumed.
"""

from dataclasses import dataclass
from fractions import Fraction as Q

from ..mathematics._rational_interval import I, pi_interval
from ._mixed_positive_interface import (
    _assess_mixed_positive_interface,
    _MixedPositiveInterface,
)
from ._mixed_reciprocal_partner import _nonnegative_pair
from ._mixed_registered_phase_probe import _derive_registered_phase_probe_geometry
from ._mixed_resource_transfer_obstruction import _original_source_matches
from ._sine_lyapunov import _sine_noncommuting_port_coefficients

_DELTA = Q(1, 1024)
_WEIGHT = Q(2, 10**60)
_MATURE = Q(1, 2**28)
_ETA = Q(2, 10**69)
_SECTION = 3 * _DELTA / 16
_SECTION_ERROR = Q(3, 2) * (14 * _ETA + 2 * 10**10 * _WEIGHT)
_GAMMA = Q(1, 3000)
_ERROR = 8 * _MATURE
_FORCING = 2 * _WEIGHT
_CUT = Q(10000)
_HORIZON = Q(1, 2)


@dataclass(frozen=True)
class _MixedCommonFormStorage:
    """A conditional structural storage, not a physical energy assignment.

    Storage and work remainders first compare with the ACTUAL projected
    coordinates. The separate reference allowance is needed before replacing
    them by the carried nominal family. The effort error concerns the regional
    projection of effort, not all fine port-slot efforts.
    """

    form_error_bounds: tuple[Q, Q]
    phase_error_bounds: tuple[Q, Q]
    source: _MixedPositiveInterface
    source_association_certified: bool
    storage_domain_certified: bool
    nominal_classification_certified: bool
    hidden_dissipation_certified: bool
    actual_dissipation_certified: bool
    fine_port_only_obstruction_certified: bool
    finite_inheritance_certified: bool
    coupling_bounds: I
    cross_strength_bounds: I
    nominal_total_mu_margins: tuple[Q, Q]
    hidden_bracket: Q
    hidden_coercivity_lower_bound: Q | None
    nominal_dissipation_lower_bound: Q
    actual_weight_perturbation_upper_bound: Q
    actual_dissipation_lower_bound: Q
    form_error_upper_bound: Q
    phase_error_upper_bound: Q
    norm_upper_bound: Q
    storage_remainder_upper_bound: Q
    projected_effort_error_upper_bound: Q
    port_power_remainder_upper_bound: Q
    accumulated_port_work_remainder_upper_bound: Q
    projected_reference_storage_error_upper_bound: Q
    total_reference_storage_error_upper_bound: Q
    certified: bool
    status: str
    unresolved_conditions: tuple[str, ...]
    selected_mu: Q = Q(1)
    nominal_total_mu_examples: tuple[Q, Q] = (Q(1, 2), Q(2))
    horizon: Q = _HORIZON
    storage_coercivity_lower_bound: Q = Q(1, 5)
    storage_norm_upper_bound: Q = Q(9)
    regional_dissipation_lower_bound: Q = Q(2)
    transport_gap_lower_bound: Q = Q(3)
    port_norm_upper_bound: Q = Q(2)
    loss_divisor: Q = Q(9)
    fixed_port_strength: Q = Q(1)
    whole_form_mean: Q = Q(0)
    mean_projector_rank: int = 1
    independently_observed_aggregates: int = 0
    numerical_response_allowance: Q = Q(0)
    nominal_port_mu_unique_within_ansatz: bool = True
    fixed_unit_nominal_total_storage_nonunique: bool = True
    fine_port_only_passivity_claimed: bool = False
    all_port_strength_total_passivity_claimed: bool = False
    canonical_storage_selected: bool = False
    phase_potential_selected: bool = False
    physical_energy_identified: bool = False
    old_form_or_phase_record_consumed: bool = False
    extra_reference_price_included: bool = True
    state_reset_performed: bool = False
    untracked_reservoir_introduced: bool = False
    source_trajectory_evaluated: bool = False
    runtime_event_executed: bool = False
    new_all_time_joint_registration_claimed: bool = False
    empirical_validation: bool = False
    storage_comparison_target: str = "actual_regional_projection"
    effort_comparison_target: str = "regional_projection_of_fine_form_effort"
    work_comparison_target: str = "regional_port_work_at_actual_projected_coordinates"
    reference_comparison_target: str = "same_carried_correlated_nominal_family"
    origin_reference: str = "conserved_whole_form_mean; P_rel=P_copymean-P_wholemean"
    clock: str = "tau=(1023/1024)*t; structural storage and conjugate form work"


def _assess_mixed_common_form_storage(*, form_error_bounds, phase_error_bounds):
    """Rebuild the fixed source and its structural storage/work bounds."""
    forms = _nonnegative_pair(form_error_bounds, "form_error_bounds")
    phases = _nonnegative_pair(phase_error_bounds, "phase_error_bounds")
    source = _assess_mixed_positive_interface(
        form_error_bounds=forms,
        phase_error_bounds=phases,
        observation_error_bound=Q(0),
    )
    handoff = source.handoff
    prefix = handoff.prefix
    reference = prefix.reference
    registered = _derive_registered_phase_probe_geometry()
    geometry = registered.successor
    coupling = Q(13, 27) - _DELTA / (18 * pi_interval())
    strength = 9 * coupling
    q0 = (_SECTION - _SECTION_ERROR / 2, _SECTION + _SECTION_ERROR / 2)
    source_ok = (
        source.form_error_bounds == handoff.form_error_bounds == forms
        and source.phase_error_bounds == handoff.phase_error_bounds == phases
        and source.source_association_certified
        and handoff.source_association_certified
        and reference.source_association_certified
        and reference.donor_source_certified
        and reference.recipient_source_certified
        and registered.cross_symmetry_certified
        and geometry == prefix.source_geometry == reference.source_geometry
        and reference.donor.cross_incidence == registered.cross_incidence
        and reference.sources
        == (reference.donor.source, reference.recipient_readiness.sources[0])
        and all(
            _original_source_matches(row, geometry, rx, rt)
            for row, rx, rt in zip(reference.sources, forms, phases, strict=True)
        )
        and reference.original_relative_phase_gap == _DELTA
        and reference.original_supplied_phase_origins == (-_DELTA / 2, _DELTA / 2)
        and reference.donor.section_mean_gap == reference.section_mean_gap == _SECTION
        and reference.donor_section_comparison_error == _SECTION_ERROR
        and reference.donor.comparison_error_upper_bound == _SECTION_ERROR
        and reference.actual_weight_deviation_upper_bound == _WEIGHT
        and reference.recipient_mature_radius == _MATURE
        and reference.recipient_isolated_deadline == Q(10**10)
        and reference.recipient_readiness.waiting_duration == Q(10**10)
        and reference.recipient_readiness.derived_mature_norm_upper_bound == _MATURE
        and reference.initial_form_means == handoff.retained_form_means == (Q(0), Q(0))
        and source.initial_donor_form_total == source.whole_form_mean == 0
        and source.exact_zero_entry_total_from_prior_laws
        and source.physical_capacities == reference.physical_capacities == (Q(1),) * 36
        and source.full_state_dimension == reference.full_state_dimension == 72
        and source.contact_pairs == tuple((j, 18 + j) for j in range(9))
        and source.prefix_full_duration == handoff.prefix_duration == Q(2000000)
        and source.prefix_cut_duration == handoff.continuation_duration == _CUT
        and source.handoff_time == handoff.total_duration == Q(2010000)
        and source.positive_duration == _HORIZON
        and source.total_duration == Q(2010000) + _HORIZON
        and source.original_clock_horizon == (Q(2010000) + _HORIZON) * Q(1024, 1023)
        and source.clock
        == "tau=(1023/1024)*t; complete internal rows and reciprocal A form port"
        and handoff.switch_word == (1, 0)
        and source.observation_error_bound == handoff.observation_error_bound == 0
        and source.handoff_observations == handoff.handoff_observations == 0
        and source.relative_common_form_mode_retained
        and source.inherited_form_velocity_retained
        and source.original_phase_gauges_and_branches_retained
        and handoff.reference_family_carried
        and not source.state_reset_performed
        and not handoff.state_reset_performed
    )
    domain = (
        source_ok
        and source.projected_rows_certified
        and source.new_trap_certified
        and source.identity_certified
        and geometry.nodes == tuple(range(18))
        and geometry.degrees == (9,) * 18
        and geometry.held_capacities == (Q(1),) * 18
        and geometry.support_certified
        and geometry.reference_stationary_certified
        and geometry.weighted_gap_lower_bound >= 3
        and geometry.quotient_coercivity_lower_bound >= Q(1, 5)
        and reference.donor.nominal_coupling_bounds == coupling
        and Q(4, 9) < coupling.lo <= coupling.hi < Q(1, 2)
        and 4 < strength.lo <= strength.hi < Q(9, 2)
        and source.nominal_q0_bounds == reference.nominal_q0_bounds == q0
        and source.new_norm_upper_bound < Q(1, 32)
        and reference.phase_radius == Q(1, 32)
        and reference.registration_gap_limit == _DELTA / 4
    )
    mu_examples = (Q(1, 2), Q(2))
    mu_margins = tuple(8 * coupling.lo * mu - (1 - mu) ** 2 for mu in mu_examples)
    classified = domain and all(margin > 0 for margin in mu_margins)
    bracket, hidden_coercivity = _sine_noncommuting_port_coefficients(
        transport_gap_lower=Q(3), port_norm_upper=Q(2), loss_divisor=Q(9)
    )
    hidden = domain and bracket == Q(1, 36) and hidden_coercivity == Q(1, 4)
    nominal_coercivity = Q(1, 4)
    perturbation = 84 * _WEIGHT
    actual_coercivity = nominal_coercivity - perturbation
    dissipative = classified and hidden and actual_coercivity > 0
    cross = tuple((a, b) for a, b in geometry.edge_pairs if a < 9 <= b)
    neighborhoods = tuple(frozenset(b for a, b in cross if a == j) for j in range(9))
    obstruction = (
        domain
        and len(cross) == 45
        and all(len(row) == 5 for row in neighborhoods)
        and neighborhoods[0] != neighborhoods[1]
    )
    rate = reference.conservative_metric_decay_rate
    old_comparison = (
        source_ok
        and handoff.switched_metric_certified
        and reference.comparison_certified
        and reference.metric_certified
        and rate
        == handoff.metric_decay_rate
        == reference.base_metric_decay_rate / 8
        > 0
        and handoff.carried_error_upper_bound
        == reference.uniform_comparison_bound
        == 4 * (_MATURE + _SECTION_ERROR + 2 * _FORCING / rate)
        < _ERROR
        and reference.comparison_forcing_upper_bound
        == handoff.metric_forcing_upper_bound
        == _FORCING
        and reference.comparison_matrix_deviation_upper_bound == 14 * _WEIGHT
        and handoff.actual_observation_allowance == _ERROR
    )
    s = 2 * _GAMMA * _HORIZON
    norm = Q(1, 100) / (1 - s)
    entry_form_error = _ERROR / (1 + _CUT / 3) + 6 * _GAMMA * _ERROR + 3 * _FORCING
    ux = entry_form_error + _FORCING * _HORIZON + s * (_ERROR + _FORCING * _HORIZON)
    ux /= 1 - s**2
    uy = _ERROR + _FORCING * _HORIZON + s * (entry_form_error + _FORCING * _HORIZON)
    uy /= 1 - s**2
    storage_error = 9 * (ux**2 + uy**2) + 7 * _WEIGHT * norm**2
    effort_error = 14 * _WEIGHT * norm
    power_error = 36 * ux**2 + 28 * _WEIGHT * norm**2
    work_error = _HORIZON * power_error
    reference_error = 9 * norm * ux + 18 * norm * uy
    inherited = (
        domain
        and old_comparison
        and source.cut_form_error_certified
        and source.component_comparison_certified
        and source.cut_form_error_upper_bound == entry_form_error
        and source.handoff_phase_error_upper_bound == _ERROR
        and source.component_coupling == s
        and source.component_denominator == 1 - s**2
        and source.positive_form_error_upper_bound == ux
        and source.positive_phase_error_upper_bound == uy
        and source.comparison_forcing_upper_bound == _FORCING
        and source.comparison_forcing_margin == _FORCING - Q(28, 9) * _WEIGHT * norm > 0
        and source.new_norm_upper_bound == norm < Q(1, 32)
        and source.marked_gap_upper_bound
        == q0[1] + 12 * _MATURE + 4 * _GAMMA * _HORIZON * norm
        and source.registration_margin == _DELTA / 4 - source.marked_gap_upper_bound > 0
    )
    unresolved = tuple(
        label
        for good, label in (
            (source_ok, "original_source_and_reached_interface_history_unavailable"),
            (domain, "common_origin_storage_and_original_chart_unavailable"),
            (classified, "nominal_quadratic_coefficient_classification_unavailable"),
            (hidden, "noncommuting_hidden_total_dissipation_unavailable"),
            (dissipative, "actual_weight_total_dissipation_unavailable"),
            (obstruction, "fine_port_only_obstruction_geometry_unavailable"),
            (inherited, "actual_projection_and_separate_reference_budgets_unavailable"),
        )
        if not good
    )
    return _MixedCommonFormStorage(
        form_error_bounds=forms,
        phase_error_bounds=phases,
        source=source,
        source_association_certified=source_ok,
        storage_domain_certified=domain,
        nominal_classification_certified=classified,
        hidden_dissipation_certified=hidden,
        actual_dissipation_certified=dissipative,
        fine_port_only_obstruction_certified=obstruction,
        finite_inheritance_certified=inherited,
        coupling_bounds=coupling,
        cross_strength_bounds=strength,
        nominal_total_mu_margins=mu_margins,
        hidden_bracket=bracket,
        hidden_coercivity_lower_bound=hidden_coercivity,
        nominal_dissipation_lower_bound=nominal_coercivity,
        actual_weight_perturbation_upper_bound=perturbation,
        actual_dissipation_lower_bound=actual_coercivity,
        form_error_upper_bound=ux,
        phase_error_upper_bound=uy,
        norm_upper_bound=norm,
        storage_remainder_upper_bound=storage_error,
        projected_effort_error_upper_bound=effort_error,
        port_power_remainder_upper_bound=power_error,
        accumulated_port_work_remainder_upper_bound=work_error,
        projected_reference_storage_error_upper_bound=reference_error,
        total_reference_storage_error_upper_bound=storage_error + reference_error,
        certified=not unresolved,
        status="conditional_common_form_storage" if not unresolved else "unavailable",
        unresolved_conditions=unresolved,
    )
