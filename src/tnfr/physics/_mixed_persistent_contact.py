"""Persistent registration and relaxation under the unchanged unit A port.

The source is the same actual donor-recipient history already reached after
half a unit of A contact. Bounds start at that port's original activation;
they do not initialize the state again at the previously certified endpoint.
The complete actual law, an integrable nominal reference and a Bregman error
estimate give separate chart, identity and asymptotic conclusions. No source
trajectory, new observation, convergence rate or physical unit is supplied.
"""

from dataclasses import dataclass
from fractions import Fraction as Q

from ..mathematics._rational_interval import I, cos, pi_interval
from ._mixed_common_form_storage import (
    _assess_mixed_common_form_storage,
    _MixedCommonFormStorage,
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
_GAMMA_LOWER = Q(1, 3300)
_GAMMA_UPPER = Q(1, 3000)
_ERROR = 8 * _MATURE
_RADIUS = Q(1, 32)
_ACTIVATION = Q(2010000)
_REACHED = Q(1, 2)


@dataclass(frozen=True)
class _MixedPersistentContact:
    """Conditional all-time bounds, with asymptotic limits kept separate.

    The uniform comparison is a propagated error of the original nominal
    family, not a new source reading. Relaxation is the complete-law LaSalle
    conclusion on the fixed original gauge leaves, without a finite deadline.
    """

    form_error_bounds: tuple[Q, Q]
    phase_error_bounds: tuple[Q, Q]
    source: _MixedCommonFormStorage
    source_association_certified: bool
    broad_trap_certified: bool
    phase_convexity_certified: bool
    nominal_cone_certified: bool
    finite_variation_certified: bool
    bregman_comparison_certified: bool
    uniform_tube_certified: bool
    all_time_identity_certified: bool
    full_relaxation_certified: bool
    vanishing_net_transfer_certified: bool
    coupling_bounds: I
    sine_coefficient_bounds: I
    gamma_bounds: tuple[Q, Q]
    nominal_q0_bounds: tuple[Q, Q]
    nominal_cone_margins: tuple[Q, Q, Q]
    nominal_norm_upper_bound: Q
    nominal_form_integral_coefficient_margin: Q
    initial_excess_storage_upper_bound: Q
    storage_barrier: Q
    storage_margin: Q
    full_norm_squared_upper_bound: Q
    actual_dissipation_lower_bound: Q
    original_comparison_upper_bound: Q
    nominal_phase_variation_upper_bound: Q
    nominal_form_integral_upper_bound: Q
    bregman_growth_integral_upper_bound: Q
    bregman_denominator: Q
    initial_bregman_root_upper_bound: Q
    weight_matrix_deviation_upper_bound: Q
    uniform_comparison_upper_bound: Q
    uniform_comparison_budget: Q
    uniform_comparison_margin: Q
    marked_gap_upper_bound: Q
    registration_margin: Q
    certified: bool
    status: str
    unresolved_conditions: tuple[str, ...]
    activation_time: Q = _ACTIVATION
    previously_reached_duration: Q = _REACHED
    previously_reached_time: Q = _ACTIVATION + _REACHED
    activation_relative_common_form: Q = Q(0)
    fixed_port_strength: Q = Q(1)
    selected_mu: Q = Q(1)
    storage_coercivity_lower_bound: Q = Q(1, 5)
    storage_norm_upper_bound: Q = Q(9)
    bregman_coercivity_lower_bound: Q = Q(1, 5)
    bregman_norm_upper_bound: Q = Q(9)
    bregman_phase_growth_coefficient: Q = Q(45)
    bregman_weight_forcing_coefficient: Q = Q(6)
    nominal_common_form_cone_factor: Q = Q(6, 5)
    nominal_contrast_form_cone_factor: Q = Q(1)
    phase_radius: Q = _RADIUS
    registration_gap_limit: Q = _DELTA / 4
    phase_convexity_lower_bound: Q = Q(1, 10)
    whole_form_mean: Q = Q(0)
    contact_pairs: tuple[tuple[int, int], ...] = tuple((j, 18 + j) for j in range(9))
    physical_capacities: tuple[Q, ...] = (Q(1),) * 36
    full_state_dimension: int = 72
    net_transfer_limit: Q = Q(0)
    donor_total_relation: str = "Q_A=-18*M_D=-9*(M_D-M_R)"
    all_contact_durations_covered: bool = False
    original_phase_gauges_and_branches_retained: bool = True
    inherited_form_velocity_retained: bool = True
    actual_unequal_weights_retained: bool = True
    actual_transport_commutation_assumed: bool = False
    full_form_and_phase_relaxation_claimed: bool = False
    asymptotic_state: str = "X->0; y->0 on the original fixed phase-gauge leaves"
    finite_relaxation_deadline: Q | None = None
    exponential_relaxation_rate: Q | None = None
    monotone_net_transfer_claimed: bool = False
    old_form_or_phase_record_consumed: bool = False
    independently_observed_aggregates: int = 0
    numerical_response_allowance: Q = Q(0)
    state_reset_performed: bool = False
    port_switch_performed: bool = False
    new_preparation_performed: bool = False
    source_trajectory_evaluated: bool = False
    runtime_event_executed: bool = False
    outer_object_declared: bool = False
    canonical_storage_selected: bool = False
    physical_energy_identified: bool = False
    empirical_validation: bool = False
    clock: str = "tau=(1023/1024)*t; same complete law and unit A form port"


def _assess_mixed_persistent_contact(*, form_error_bounds, phase_error_bounds):
    """Rebuild only the source and premises used by the persistent theorem."""
    forms = _nonnegative_pair(form_error_bounds, "form_error_bounds")
    phases = _nonnegative_pair(phase_error_bounds, "phase_error_bounds")
    source = _assess_mixed_common_form_storage(
        form_error_bounds=forms, phase_error_bounds=phases
    )
    positive = source.source
    handoff = positive.handoff
    prefix = handoff.prefix
    reference = prefix.reference
    registered = _derive_registered_phase_probe_geometry()
    geometry = registered.successor
    pi = pi_interval()
    coupling = Q(13, 27) - _DELTA / (18 * pi)
    sine_coefficient = 1 + 2 * cos(pi / 9)
    gamma = 1 / (1023 * pi)
    q0 = (_SECTION - _SECTION_ERROR / 2, _SECTION + _SECTION_ERROR / 2)
    source_ok = (
        source.form_error_bounds == positive.form_error_bounds == forms
        and source.phase_error_bounds == positive.phase_error_bounds == phases
        and source.source_association_certified
        and positive.source_association_certified
        and handoff.source_association_certified
        and handoff.handoff_certified
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
        and positive.initial_donor_form_total == positive.whole_form_mean == 0
        and positive.exact_zero_entry_total_from_prior_laws
        and positive.physical_capacities
        == reference.physical_capacities
        == (Q(1),) * 36
        and positive.full_state_dimension == reference.full_state_dimension == 72
        and positive.contact_pairs == tuple((j, 18 + j) for j in range(9))
        and positive.prefix_full_duration == handoff.prefix_duration == Q(2000000)
        and positive.prefix_cut_duration == handoff.continuation_duration == Q(10000)
        and positive.handoff_time == handoff.total_duration == _ACTIVATION
        and positive.positive_duration == source.horizon == _REACHED
        and positive.total_duration == _ACTIVATION + _REACHED
        and positive.original_clock_horizon == (_ACTIVATION + _REACHED) * Q(1024, 1023)
        and positive.clock
        == "tau=(1023/1024)*t; complete internal rows and reciprocal A form port"
        and handoff.switch_word == (1, 0)
        and handoff.original_clock_horizon == _ACTIVATION * Q(1024, 1023)
        and handoff.clock == "tau=(1023/1024)*t; all complete internal rows retained"
        and positive.observation_error_bound == handoff.observation_error_bound == 0
        and positive.handoff_observations == handoff.handoff_observations == 0
        and positive.relative_common_form_mode_retained
        and positive.inherited_form_velocity_retained
        and positive.original_phase_gauges_and_branches_retained
        and handoff.reference_family_carried
        and handoff.reached_velocity_retained
        and handoff.source_weight_association_retained
        and handoff.original_phase_gauges_and_branches_retained
        and not positive.state_reset_performed
        and not handoff.state_reset_performed
        and not source.state_reset_performed
    )
    bracket, hidden = _sine_noncommuting_port_coefficients(
        transport_gap_lower=Q(3), port_norm_upper=Q(2), loss_divisor=Q(9)
    )
    alpha = Q(1, 4) - 84 * _WEIGHT
    energy = Q(27, 2) * _DELTA**2 + 486 * _ETA + 18 * _ETA**2
    entry_storage = energy + 9 * _MATURE**2
    barrier = _RADIUS**2 / 5
    storage_margin = barrier - entry_storage
    trap = (
        source_ok
        and source.storage_domain_certified
        and source.actual_dissipation_certified
        and positive.projected_rows_certified
        and reference.actual_trap_certified
        and geometry.nodes == tuple(range(18))
        and geometry.degrees == (9,) * 18
        and geometry.held_capacities == (Q(1),) * 18
        and geometry.support_certified
        and geometry.reference_stationary_certified
        and geometry.weighted_gap_lower_bound >= 3
        and geometry.quotient_coercivity_lower_bound >= Q(1, 5)
        and source.selected_mu == source.fixed_port_strength == 1
        and source.whole_form_mean == 0
        and source.mean_projector_rank == 1
        and source.storage_coercivity_lower_bound == Q(1, 5)
        and source.storage_norm_upper_bound == 9
        and source.transport_gap_lower_bound == 3
        and source.port_norm_upper_bound == 2
        and source.loss_divisor == 9
        and source.hidden_bracket == bracket == Q(1, 36)
        and source.hidden_coercivity_lower_bound == hidden == Q(1, 4)
        and source.nominal_dissipation_lower_bound == Q(1, 4)
        and source.actual_weight_perturbation_upper_bound == 84 * _WEIGHT
        and source.actual_dissipation_lower_bound == alpha > 0
        and reference.donor.excess_storage_upper_bound == energy
        and reference.initial_excess_storage_upper_bound == entry_storage
        and reference.full_norm_squared_upper_bound == 5 * entry_storage
        and reference.full_norm_upper_bound == Q(1, 100)
        and 5 * entry_storage < reference.full_norm_upper_bound**2 == Q(1, 10000)
        and reference.phase_radius == _RADIUS
        and storage_margin > 0
    )
    rate = reference.conservative_metric_decay_rate
    original_comparison = (
        4 * (_MATURE + _SECTION_ERROR + 4 * _WEIGHT / rate) if rate > 0 else None
    )
    compared = (
        source_ok
        and handoff.switched_metric_certified
        and reference.metric_certified
        and reference.comparison_certified
        and rate == handoff.metric_decay_rate == reference.base_metric_decay_rate / 8
        and rate > 0
        and handoff.carried_error_upper_bound
        == reference.uniform_comparison_bound
        == original_comparison
        and original_comparison < _ERROR
        and reference.comparison_forcing_upper_bound
        == handoff.metric_forcing_upper_bound
        == 2 * _WEIGHT
        and reference.comparison_matrix_deviation_upper_bound == 14 * _WEIGHT
        and handoff.actual_observation_allowance == _ERROR
    )
    cone_margins = (
        2 * coupling.lo / 3 - _GAMMA_UPPER**2,
        2 * coupling.lo
        - Q(1, 5)
        - sine_coefficient.hi / 9
        - 4 * _GAMMA_UPPER**2 * coupling.hi,
        Q(1, 5) - Q(24, 5) * _GAMMA_UPPER**2 * coupling.hi,
    )
    nominal_norm = 4 * q0[1]
    integral_margin = 10 - 9 / (4 * coupling.lo)
    cone = (
        trap
        and compared
        and handoff.continuation_wedges_certified
        and reference.nominal_wedges_certified
        and reference.donor.nominal_reference_certified
        and reference.donor.gamma_bounds == gamma
        and _GAMMA_LOWER <= gamma.lo <= gamma.hi <= _GAMMA_UPPER
        and source.coupling_bounds
        == reference.donor.nominal_coupling_bounds
        == coupling
        and Q(4, 9) < coupling.lo <= coupling.hi < Q(1, 2)
        and reference.donor.regional_sine_coefficient_bounds == sine_coefficient
        and 2 < sine_coefficient.lo <= sine_coefficient.hi < 3
        and reference.nominal_q0_bounds == positive.nominal_q0_bounds == q0
        and reference.nominal_velocity_factor_upper_bound == _GAMMA_UPPER**2
        and reference.nominal_wedge_margin == 2 * coupling.lo / 3 - _GAMMA_UPPER**2 > 0
        and reference.inherited_nominal_velocity_relation
        == "-gamma**2*q0 <= w0 <= 0 at the same isolated age"
        and reference.inherited_donor_velocity_retained
        and reference.initial_comparison_norm_upper_bound == _MATURE + _SECTION_ERROR
        and 0 < q0[0] <= q0[1] < _DELTA / 4
        and 4 * coupling.lo > 1
        and all(margin > 0 for margin in cone_margins)
        and 3 * (1 + 3 * _GAMMA_UPPER) < 4
        and nominal_norm < Q(1, 100) < _RADIUS
    )
    phase_variation = 3 * q0[1]
    form_integral = 10 * q0[1] / _GAMMA_LOWER
    variation = (
        cone and phase_variation > 0 and form_integral > 0 and integral_margin > 0
    )
    growth = 45 * phase_variation
    denominator = 1 - growth
    initial_root = 3 * _ERROR
    matrix_error = 14 * _WEIGHT
    # E_err uses the ACTUAL H in its form metric. Its forcing is deltaH*xbar,
    # whose finite L1 budget prevents an all-time accumulated mismatch floor.
    comparison = 3 * (initial_root + 6 * matrix_error * form_integral) / denominator
    error_budget = 200 * _MATURE
    error_margin = error_budget - comparison
    convexity = (
        trap
        and all(row.phase_cosine_lower_bound >= Q(1, 10) for row in reference.sources)
        and all(row.phase_curvature_bounds.lo >= Q(1, 10) for row in reference.sources)
        and source.origin_reference
        == "conserved_whole_form_mean; P_rel=P_copymean-P_wholemean"
    )
    bregman = convexity and variation and compared and denominator > 0
    tube = bregman and error_margin > 0
    marked_gap = q0[1] + 300 * _MATURE
    registration_margin = _DELTA / 4 - marked_gap
    identity = (
        tube
        and reference.registration_gap_limit == _DELTA / 4
        and Q(3, 2) * comparison < 300 * _MATURE
        and registration_margin > 0
    )
    # On the compact acute invariant set, E'=0 implies X=0. The complete
    # form row then requires S(theta)=0 for an invariant orbit. Strict phase
    # convexity on each fixed gauge leaf leaves only the original reference.
    relaxed = convexity and alpha > 0
    transfer = relaxed and positive.initial_donor_form_total == 0
    unresolved = tuple(
        label
        for good, label in (
            (source_ok, "original_source_and_unchanged_reached_A_port_unavailable"),
            (trap, "actual_storage_and_all_time_acute_chart_unavailable"),
            (convexity, "strict_phase_convexity_on_fixed_gauge_leaves_unavailable"),
            (cone, "correlated_nominal_A_port_cone_unavailable"),
            (variation, "finite_nominal_phase_variation_and_form_integral_unavailable"),
            (bregman, "actual_metric_Bregman_comparison_unavailable"),
            (tube, "all_time_actual_reference_error_budget_unavailable"),
            (identity, "all_time_original_marked_registration_unavailable"),
            (relaxed, "complete_law_phase_and_form_relaxation_unavailable"),
            (transfer, "vanishing_net_form_transfer_unavailable"),
        )
        if not good
    )
    return _MixedPersistentContact(
        form_error_bounds=forms,
        phase_error_bounds=phases,
        source=source,
        source_association_certified=source_ok,
        broad_trap_certified=trap,
        phase_convexity_certified=convexity,
        nominal_cone_certified=cone,
        finite_variation_certified=variation,
        bregman_comparison_certified=bregman,
        uniform_tube_certified=tube,
        all_time_identity_certified=identity,
        full_relaxation_certified=relaxed,
        vanishing_net_transfer_certified=transfer,
        coupling_bounds=coupling,
        sine_coefficient_bounds=sine_coefficient,
        gamma_bounds=(_GAMMA_LOWER, _GAMMA_UPPER),
        nominal_q0_bounds=q0,
        nominal_cone_margins=cone_margins,
        nominal_norm_upper_bound=nominal_norm,
        nominal_form_integral_coefficient_margin=integral_margin,
        initial_excess_storage_upper_bound=entry_storage,
        storage_barrier=barrier,
        storage_margin=storage_margin,
        full_norm_squared_upper_bound=5 * entry_storage,
        actual_dissipation_lower_bound=alpha,
        original_comparison_upper_bound=_ERROR,
        nominal_phase_variation_upper_bound=phase_variation,
        nominal_form_integral_upper_bound=form_integral,
        bregman_growth_integral_upper_bound=growth,
        bregman_denominator=denominator,
        initial_bregman_root_upper_bound=initial_root,
        weight_matrix_deviation_upper_bound=matrix_error,
        uniform_comparison_upper_bound=comparison,
        uniform_comparison_budget=error_budget,
        uniform_comparison_margin=error_margin,
        marked_gap_upper_bound=marked_gap,
        registration_margin=registration_margin,
        certified=not unresolved,
        status="conditional_persistent_contact" if not unresolved else "unavailable",
        unresolved_conditions=unresolved,
        all_contact_durations_covered=identity,
        full_form_and_phase_relaxation_claimed=relaxed,
    )
