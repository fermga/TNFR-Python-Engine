"""Formation obstructions and basin checks for exact mediated preparations.

The complete supplied support consists of two C5 rings and a live intermediary.
Only the donor starts twisted. Exact preparation, symmetry, local response and
global storage obstructions are distinct from successful receiver formation.
No graph, trajectory, support event or new evolution law is constructed.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from .._exact_time import exact_or_represented_real
from ..dynamics.relational import RelationalExchangeModel
from ..mathematics._phase_midpoint import _pi_bounds
from ..mathematics._rational_interval import INTERVAL_METHOD, I, pi_interval, sqrt
from ._cycle_algebra import ordered_vector
from ._sine_admission import _sine_model_coefficients
from .phase_cycle_geometry import PhaseCycleState
from .phase_cycle_geometry import _derive as _derive_phase_geometry
from .phase_cycle_geometry import reconstruct_phase_cycle_state
from .relational_capture import _cycle_barrier_constants, _twist_storage_bounds
from .relational_observations import _ordered
from .relational_sine_comparison import _sine_work

__all__ = (
    "SineDirectionalLossBound",
    "SinePhaseActionBound",
    "SineMaintainedTargetObstruction",
    "SineDonorWellRetention",
    "SineDonorDissipativeCapture",
    "SineMediatedFormation",
    "SineReceiverExcitation",
    "SineReceiverLocalization",
    "SineReceiverTransferAdmission",
    "assess_sine_mediated_formation",
)


@dataclass(frozen=True)
class SineDirectionalLossBound:
    """Analytic loss bound retaining the initial gradient's spectral direction.

    With K=diag(nu/d), B=sqrt(K)*L*sqrt(K), y0=sqrt(K)*q0,
    retain exact N=||y0||^2, m1=y0.T*B*y0 and m2=||B*y0||^2.
    The polynomial lower bound on ||sqrt(K)*q(t)||/sqrt(N) is consumed only
    while positive. This does not supply a trajectory or a capture result.
    """

    horizon: Q | None
    initial_dissipative_norm_squared: Q
    first_spectral_moment: Q
    second_spectral_moment: Q
    rayleigh_quotient: Q | None
    gamma_squared: Q | None
    gamma_upper_bound: Q | None
    spectral_upper_bound: Q
    oscillation_rate_squared_upper_bound: Q
    hyperbolic_argument_squared_upper_bound: Q | None
    cosh_upper_bound: Q | None
    cosh_bound_method: str | None
    linear_coefficient: Q | None
    quadratic_coefficient: Q | None
    norm_ratio_lower_bound_at_horizon: Q | None
    loss_lower_bound: Q | None
    unavailable_reasons: tuple[str, ...]
    status: str
    scope: tuple[str, ...] = (
        "initial_sine_currents_are_exactly_zero_by_shared_phase_reconstruction",
        "full_support_held_positive_capacity_and_declared_sine_model",
        "exact_gradient_spectral_moments_with_safe_lambda_at_most_2_max_capacity",
        "semigroup_Jensen_and_projected_Duhamel_estimates_not_a_linearized_trajectory",
        "declared_horizon_uses_a_short_window_or_geometric_series_cosh_upper_bound",
        "geometric_cosh_majorant_requires_squared_argument_strictly_below_two",
        "positive_polynomial_norm_bound_required_over_the_entire_declared_horizon",
        "alternative_loss_bounds_combine_by_maximum_never_addition",
    )

    @property
    def available(self):
        return self.status == "available"


@dataclass(frozen=True)
class SinePhaseActionBound:
    """Necessary joint-entry time from receiver winding and accumulated loss.

    An initially flat receiver must cross an antipodal edge before entering
    the joint acute winding-one target. Weighted Cauchy--Schwarz and the
    target's remaining loss allowance give a lower time bound, conditional
    on entry. Its failure to exclude a horizon is not evidence of formation.
    """

    receiver_edge_mobility_max: Q
    allowable_loss_bounds: I
    necessary_entry_time_lower_bound: Q | None
    horizon: Q | None
    entry_through_horizon_excluded: bool
    unavailable_reasons: tuple[str, ...]
    status: str
    scope: tuple[str, ...] = (
        "same_full_support_held_positive_capacities_and_initially_flat_receiver",
        "necessary_receiver_antipodal_crossing_before_joint_acute_target_entry",
        "weighted_phase_action_uses_actual_receiver_capacity_over_degree_mobilities",
        "entry_allows_at_most_initial_form_storage_minus_beta_times_acute_face_cost",
        "finite_time_bound_requires_strictly_positive_resolved_loss_allowance",
        "lower_bound_is_conditional_on_entry_not_a_predicted_capture_time",
        "closed_horizon_exclusion_requires_horizon_strictly_below_the_lower_bound",
        "proof_time_is_the_declared_sine_model_clock_not_operator_cycles_or_SI_time",
        "no_trajectory_global_phase_speed_bound_or_positive_formation_verdict",
    )

    @property
    def available(self):
        return self.status == "available"


@dataclass(frozen=True)
class SineMaintainedTargetObstruction:
    """Global mixed-functional obstruction to the maintained two-twist target.

    At unit storage scale, positive half contrast and 0<w/e<3/2, the law makes
    W=(4/5)*F+V-(w/(2*pi*e))*q.T*K*S nonincreasing. Exact initial sine
    cancellation gives W0=(4/5)*F0+V5; the target has W*=2*V5. A
    strictly positive target-minus-initial bound excludes convergence to
    that target and hence entry to any valid basin proving that convergence.
    It does not exclude every transient visit to the joint acute region.
    """

    form_storage_coefficient: Q | None
    mixed_term_coefficient_bounds: I | None
    mobility_spectral_bounds: tuple[Q, Q] | None
    initial_functional_bounds: I | None
    target_functional_bounds: I | None
    target_margin_bounds: I | None
    maintained_target_excluded: bool
    unavailable_reasons: tuple[str, ...]
    status: str
    proof_id: str = "eleven_node_sine_ratio_mixed_lyapunov"
    scope: tuple[str, ...] = (
        "same_full_eleven_node_support_and_exact_initial_sine_cancellation",
        "requires_positive_effective_exchange_to_loss_ratio_below_three_halves",
        "requires_beta_one_and_delta_positive_one_half",
        "ratio_uses_exact_represented_effective_model_weights_not_requested_raw_weights",
        "common_raw_weight_scaling_is_normalization_gauge_not_a_clock_change",
        "analysis_coefficient_four_fifths_is_not_a_new_constitutive_parameter",
        "W_is_a_monotone_mixed_functional_not_the_nodal_storage_energy",
        "mobility_spectrum_bounds_apply_to_the_fixed_full_support_positive_capacity_law",
        "correlated_target_gap_reuses_V5_minus_four_fifths_initial_form_storage",
        "positive_gap_excludes_maintained_target_convergence_and_its_valid_recovery_basin",
        "does_not_exclude_all_transient_joint_acute_region_visits",
        "no_form_budget_cap_is_needed_when_the_source_gap_is_resolved_positive",
        "outside_the_proved_law_or_capacity_scope_is_unavailable_not_a_negative_result",
        "no_trajectory_profile_search_event_loss_selection_or_physical_identification",
    )
    exchange_to_loss_ratio: Q | None = None
    sufficient_ratio_upper_bound: Q | None = None


@dataclass(frozen=True)
class SineMediatedFormation:
    """Static eligibility or exclusion under the declared complete sine law.

    Phases are exact rational turns multiplied by mathematical 2*pi, never
    graph floats. The odd acceleration refers to (theta_6-theta_9)'' at zero;
    its exact numerator divided by pi preserves a nonzero response even when
    an outward numerical interval also contains zero. Passing the necessary
    checks establishes neither entry into a target basin nor formation.
    ``early_loss_lower_bound`` retains the mediator-only bound; the explicit
    ``full_early_loss_lower_bound`` sums all nodal contributions. Its maximum
    with the independent directional bound drives timed exclusion through
    ``combined_early_loss_lower_bound``. Conserved phase means use continuous lifts,
    not a circular mean or a prescribed final representative. The independent
    ``phase_action_bound`` supplies a necessary earliest entry time; it can
    rule out entry through the loss horizon without a global small-phase bound.
    """

    model: RelationalExchangeModel
    amplitude: Q
    capacity_contrast: Q
    profile: str
    nodes: tuple[int, ...]
    edges: tuple[tuple[int, int], ...]
    neighbors: tuple[tuple[int, ...], ...]
    degrees: tuple[int, ...]
    cycles: tuple[tuple[int, ...], tuple[int, ...]]
    mediator: int
    ports: tuple[int, int]
    initial_epi: tuple[Q, ...]
    initial_phase_turns: tuple[Q, ...]
    capacity: tuple[Q, ...]
    initial_geometry: PhaseCycleState
    target_phase_turns: tuple[Q, ...]
    target_geometry: PhaseCycleState
    initial_form_gradient: tuple[Q, ...]
    initial_form_rate_bounds: tuple[I, ...]
    initial_phase_rate_bounds: tuple[I, ...]
    phase_odd_acceleration_pi_numerator: Q
    phase_odd_acceleration_bounds: I
    phase_odd_acceleration_sign: int
    reflection_invariant: bool
    silent_donor_subspace: bool
    initial_equilibrium: bool
    initial_form_storage: Q
    twist_phase_storage_bounds: I
    acute_face_phase_storage_bounds: I
    initial_storage_bounds: I
    target_storage_bounds: I
    entry_storage_threshold_bounds: I
    initial_continuous_loss: Q
    initial_loss_per_form_storage: Q | None
    invariant_weights: tuple[Q, ...]
    weighted_coordinate_mass: Q
    initial_weighted_form_sum: Q
    initial_weighted_phase_turn_sum: Q
    conserved_form_mean: Q
    conserved_lifted_phase_turn_mean: Q
    initial_storage_rate: Q
    initial_balance_residual_bounds: I
    target_storage_margin_bounds: I
    entry_storage_margin_bounds: I
    target_budget_status: str
    entry_budget_status: str
    exclusion_time: Q | None
    form_speed_upper_bounds: tuple[Q, ...] | None
    mediator_gradient_speed_upper_bound: Q | None
    nodal_gradient_speed_upper_bounds: tuple[Q, ...] | None
    receiver_gap_speed_upper_bound: Q | None
    receiver_gap_drift_upper_bound: Q | None
    receiver_acute_time_margin_lower_bound: Q | None
    loss_integration_time: Q | None
    early_loss_lower_bound: Q | None
    nodal_loss_integration_times: tuple[Q, ...] | None
    nodal_early_loss_lower_bounds: tuple[Q, ...] | None
    full_early_loss_lower_bound: Q | None
    directional_loss_bound: SineDirectionalLossBound
    combined_early_loss_lower_bound: Q | None
    post_time_entry_deficit_lower_bound: Q | None
    early_loss_exclusion_certified: bool
    exclusion_reasons: tuple[str, ...]
    unresolved_conditions: tuple[str, ...]
    status: str
    law: str = "normalized_sine_reciprocal_exchange"
    arithmetic_method: str = INTERVAL_METHOD
    scope: tuple[str, ...] = (
        "complete_supplied_two_C5_rings_and_live_intermediary_with_two_unit_bridges",
        "exact_donor_winding_one_flat_receiver_and_intermediary_phase_preparation",
        "localized_balanced_or_explicit_donor_forms_with_supplied_intermediary_amplitude",
        "receiver_form_is_zero_and_capacities_6_and_9_are_1_plus_or_minus_delta",
        "balanced_means_donor_form_2A_intermediary_A_receiver_zero_not_a_universal_optimum",
        "equilibrium_requires_the_full_initial_form_gradient_to_vanish_not_only_intermediary_form",
        "fixed_support_held_positive_capacity_no_input_events_or_installed_controller",
        "explicit_regular_reference_coefficients_and_positive_epi_weight_for_sine_law",
        "exact_turn_reconstruction_and_odd_sine_cancellation_not_a_float_equilibrium",
        "receiver_reflection_is_a_global_obstruction_only_when_delta_is_exactly_zero",
        "exact_antisymmetric_donor_with_zero_port_and_intermediary_keeps_receiver_silent",
        "nonzero_odd_initial_acceleration_is_not_an_eventual_winding_or_capture_claim",
        "first_joint_acute_entry_bound_does_not_assume_earlier_donor_winding_preservation",
        "initial_strict_loss_makes_target_and_entry_storage_requirements_strict",
        "target_turns_fix_only_a_representative_modulo_common_form_and_phase_origins",
        "optional_early_loss_and_receiver_phase_bounds_follow_global_storage_without_IVP",
        "timed_loss_uses_actual_full_nodal_gradients_not_an_inherited_localized_formula",
        "nodal_and_directional_loss_certificates_combine_by_maximum_without_double_counting",
        "phase_action_or_acute_gap_bound_can_exclude_entry_through_the_loss_horizon",
        "scoped_mixed_Lyapunov_obstruction_excludes_maintenance_not_every_acute_visit",
        "held_capacity_weighted_means_are_conserved_on_continuous_lifts_modulo_common_origins",
        "proof_horizon_uses_the_sine_model_clock_not_operator_cycles_or_laboratory_seconds",
        "necessary_checks_are_not_sufficient_formation_or_a_time_to_capture",
        "no_graph_materialization_native_dispatch_trajectory_solver_or_support_birth",
        "no_physical_identification_or_authentication_of_external_measurement_data",
    )
    phase_action_bound: SinePhaseActionBound | None = None
    maintained_target_obstruction: SineMaintainedTargetObstruction | None = None

    @property
    def passes_necessary_conditions(self):
        return self.status == "passes_necessary_conditions"

    @property
    def formation_unresolved(self):
        return self.status != "excluded"

    def receiver_transfer(self) -> SineReceiverTransferAdmission:
        """Assess donor unwinding/receiver recovery with separate obligations.

        This reuses the exact preparation and declared horizon. The original
        report's coexistence target, budgets and verdict remain unchanged.
        A separate donor-well certificate can exclude either receiver-only
        limit without excluding every transient target-region visit.
        """
        return _receiver_transfer(self)

    def donor_well_retention(self) -> SineDonorWellRetention:
        """Certify return to the initial relative donor pattern when admitted.

        The exact algebraic storage threshold distinguishes this basin result
        from a local recovery test or a claim about every transient winding.
        """
        return _donor_well_retention(self)

    def donor_dissipative_capture(self) -> SineDonorDissipativeCapture:
        """Test the fixed quarter-clock donor capture bound on this source.

        The full-law functional decrease and phase-path bound can place a
        source above the static retention threshold in the donor component.
        The fixed proof horizon does not evaluate or predict a trajectory.
        """
        return _donor_dissipative_capture(self)

    def receiver_excitation(self) -> SineReceiverExcitation:
        """Separate source geometry from a conditional tangent phase ceiling.

        The complete six-coordinate preparation is retained. A donor-odd
        component is silent in the tangent receiver output, but cannot be
        removed from a general nonlinear mixture. The correction bound is
        necessary if the actual receiver reaches its phase barrier; it does
        not certify or exclude that event.
        """
        return _receiver_excitation(self)

    def receiver_localization(self) -> SineReceiverLocalization:
        """Certify asymptotic receiver consensus on a regional storage budget.

        The fixed-law regional Lyapunov function excludes every nonflat
        receiver equilibrium below an exact initial form threshold. It does
        not select the donor endpoint or exclude transient barrier passages.
        """
        return _receiver_localization(self)

    def to_dict(self):
        from ..sdk.relational_reports import _project

        return {
            "schema": "tnfr.relational-sine-mediated-formation.v1",
            "report": _project(self),
        }


@dataclass(frozen=True)
class SineReceiverExcitation:
    """Exact source decomposition, tangent output and a nonlinear phase budget.

    ``source_coordinates`` are (H, D0-H, s1, s2, a1, a2), where
    s1=(D1+D4)/2-D0, s2=(D2+D3)/2-D0, a1=(D1-D4)/2 and a2=(D2-D3)/2.
    Full eleven-node even/odd vectors reconstruct the actual initial form.
    Their storage and dissipative norms use the original support and capacity.

    Under the fixed half-weight, beta-one, positive-half-contrast law, the
    tangent about the exact donor-one/receiver-flat equilibrium has receiver
    quadratic phase cost at most 3*F_even/4, strictly less when F_even>0.
    This is a ceiling for the complete damped tangent, not for the nonlinear
    trajectory. The odd tangent source has identically zero receiver output.

    If the nonlinear receiver reaches phase potential 7/2, its phase-edge
    correction relative to that tangent must have norm at least
    sqrt(7)-sqrt(3*F_even/2). ``necessary_nonlinear_correction_norm_bounds``
    encloses this analytic threshold, not the actual correction. A positive
    lower endpoint supplies a conservative necessary bound. Nonpositive or
    unresolved thresholds do not imply small nonlinear correction or passage.
    No initial/evolved basin admission, waveform or formation verdict follows.

    Separately, the same fixed law with total initial form storage F<=4 has
    actual full-network phase potential V(t)<139/20 for all future time.
    If the receiver reaches potential 7/2, the donor then has potential less
    than 69/20: its own potential barrier must have been crossed strictly
    earlier, with phase-potential release greater than V5-69/20. This is a
    conditional barrier order, not a winding-change order or observed passage.
    The required donor potential decrease is not identified receiver work or
    a reusable energy reserve; it may dissipate or remain elsewhere.
    ``full_phase_budget_status`` admits this nonlinear certificate separately
    from the uncapped tangent bound. Its optional fields remain None outside
    the original form budget or law, including legacy manual constructions.
    """

    source: SineMediatedFormation
    source_coordinates: tuple[Q, Q, Q, Q, Q, Q]
    even_initial_epi: tuple[Q, ...]
    odd_initial_epi: tuple[Q, ...]
    bridge_form_storage: Q
    even_internal_form_storage: Q
    odd_internal_form_storage: Q
    even_form_storage: Q
    odd_form_storage: Q
    even_dissipative_norm_squared: Q
    odd_dissipative_norm_squared: Q
    tangent_receiver_phase_cost_upper_bound: Q | None
    tangent_phase_cost_bound_strict: bool | None
    receiver_barrier_phase_storage: Q
    necessary_nonlinear_correction_norm_bounds: I | None
    necessary_nonlinear_correction_norm_lower_bound: Q | None
    nonlinear_correction_status: str
    tangent_odd_receiver_silence_certified: bool
    unavailable_reasons: tuple[str, ...]
    status: str
    proof_id: str = "eleven_node_sine_receiver_tangent_excitation"
    arithmetic_method: str = INTERVAL_METHOD
    scope: tuple[str, ...] = (
        "revalidated_six_form_preparation_and_complete_eleven_node_support",
        "source_coordinates_order_H_D0_minus_H_s1_s2_a1_a2",
        "even_and_odd_donor_reflection_components_reconstruct_all_initial_forms",
        "component_storage_and_gradient_norms_use_original_edges_degrees_and_capacities",
        "geometric_decomposition_does_not_depend_on_fixed_tangent_law_admission",
        "tangent_bound_requires_original_sine_half_weights_beta_one_delta_positive_half",
        "full_damped_tangent_about_exact_donor_one_receiver_zero_phase_equilibrium",
        "noncommuting_form_phase_Hessians_and_mobility_are_retained",
        "receiver_quadratic_phase_cost_ceiling_is_three_fourths_even_source_storage",
        "positive_even_storage_has_strict_ceiling_zero_even_storage_has_zero_output",
        "odd_component_is_tangent_silent_but_not_discarded_from_nonlinear_mixtures",
        "necessary_correction_uses_receiver_cycle_phase_edge_norm_on_continuous_lifts",
        "correction_bound_is_conditional_on_actual_receiver_potential_reaching_seven_halves",
        "positive_correction_lower_bound_requires_resolved_outward_interval_margin",
        "full_nonlinear_phase_ceiling_requires_total_source_form_storage_at_most_four",
        "full_phase_budget_covers_all_original_edges_and_is_strictly_below_139_over_20",
        "donor_potential_barrier_order_and_release_are_conditional_on_receiver_barrier_passage",
        "donor_phase_potential_decrease_is_not_identified_transferred_work_or_reusable_reserve",
        "no_winding_change_order_actual_receiver_work_integral_or_receiver_passage_verdict",
        "no_law_clock_input_event_or_support_change_and_no_trajectory_evaluation",
    )
    actual_full_phase_storage_upper_bound: Q | None = None
    donor_potential_barrier_first_required: bool | None = None
    necessary_donor_phase_release_lower_bound: Q | None = None
    full_phase_budget_status: str = "unavailable"

    @property
    def available(self):
        return self.status == "available"

    def to_dict(self):
        from ..sdk.relational_reports import _project

        return {
            "schema": "tnfr.relational-sine-receiver-excitation.v1",
            "report": _project(self),
        }


@dataclass(frozen=True)
class SineReceiverLocalization:
    """Full nonlinear receiver-consensus certificate from regional weights.

    The auxiliary function is U=F+V_D+(7/5)*V_R+(7/6)*V_bridges
    -(1/3)*q.T*K*S. Its decrease holds on the original complete support,
    half-weight sine law, unit storage scale and positive-half capacity
    contrast. The rational coefficients belong to this proof, not to the
    evolution law, regional pressure or receiver's physical storage ledger.

    Initially S=0 and U=F+V5. Every nonflat receiver equilibrium has
    U>=7*V5/5. Consequently F<=(5-sqrt(5))/2 excludes those limits;
    the full-law convergence theorem then gives relative receiver consensus.
    This does not select a donor equilibrium, limit transient receiver
    potential, certify barrier avoidance or predict a convergence time.

    Admission uses both 5-2*F>=0 and (5-2*F)^2-5>=0. Display intervals
    enclose analytic threshold and endpoint comparison values, not an
    evolved state. Public construction is not provenance authentication.
    """

    source: SineMediatedFormation
    initial_form_storage: Q
    form_storage_coefficient: Q | None
    mixed_term_coefficient: Q | None
    donor_phase_weight: Q | None
    receiver_phase_weight: Q | None
    bridge_phase_weight: Q | None
    critical_form_storage_bounds: I
    initial_functional_bounds: I | None
    receiver_nonflat_limit_functional_lower_bounds: I | None
    target_minus_initial_margin_bounds: I | None
    exact_localization_polynomial_margin: Q
    receiver_nonflat_equilibria_excluded: bool
    relative_receiver_consensus_certified: bool
    unavailable_reasons: tuple[str, ...]
    status: str
    proof_id: str = "eleven_node_sine_regional_receiver_localization"
    arithmetic_method: str = INTERVAL_METHOD
    scope: tuple[str, ...] = (
        "revalidated_six_form_preparation_and_complete_eleven_node_support",
        "requires_original_sine_half_weights_beta_one_delta_positive_half",
        "regional_auxiliary_weights_are_proof_coefficients_not_evolution_parameters",
        "weighted_sine_gradients_retain_both_bridges_and_full_nodal_loss",
        "exact_positive_definite_endpoint_matrices_bound_the_actual_phase_hessian_globally",
        "original_full_state_asymptotic_convergence_is_required_for_the_endpoint_conclusion",
        "nonflat_receiver_critical_values_are_at_least_single_twist_storage",
        "exact_sign_and_polynomial_control_admission_not_display_intervals",
        "initial_nonzero_form_gives_strict_auxiliary_decrease",
        "certified_result_excludes_all_nonflat_receiver_equilibria_including_saddles",
        "relative_receiver_phase_consensus_does_not_select_the_donor_endpoint",
        "no_all_time_receiver_barrier_winding_or_potential_bound",
        "no_actual_work_integral_trajectory_recovery_deadline_or_state_reconstruction",
        "no_law_clock_support_input_event_or_physical_identification_change",
        "above_sufficient_threshold_is_not_certified_not_a_transfer_verdict",
    )

    def to_dict(self):
        from ..sdk.relational_reports import _project

        return {
            "schema": "tnfr.relational-sine-receiver-localization.v1",
            "report": _project(self),
        }


@dataclass(frozen=True)
class SineReceiverTransferAdmission:
    """Target-specific checks for a flat donor and a maintained twisted receiver.

    Energy and functional margins are source-minus-target, unlike the
    target-minus-source monotonicity obstruction on the coexistence report.
    The compatible phase shift selects one lift with no extra nodal integer
    turns; it is not a unique possible limit. An actual final lift with integer
    turns m_i changes that shift by -sum((d_i/nu_i)*m_i)/sum(d_i/nu_i).

    ``passes_necessary_conditions`` means only that these checks do not
    exclude transfer. Neither a favorable target nor a possible recovery
    basin proves that this initial preparation reaches it. ``unavailable``
    denotes failed theorem premises; ``unresolved_bound`` denotes an interval
    boundary that prevents a necessary-budget verdict.
    The optional ``donor_well_retention`` instead proves convergence to the
    original relative donor pattern for an exactly admitted source subset.
    ``donor_dissipative_capture`` can extend that source-to-basin conclusion
    using fixed-window loss. Both leave receiver passage-time fields unchanged.
    ``receiver_localization`` separately certifies asymptotic receiver
    consensus through a regional auxiliary function without selecting the
    donor endpoint or excluding transient receiver barrier passage.

    Receiver acquisition necessarily crosses the receiver's phase-potential
    barrier 7/2 at a finite time. ``necessary_receiver_running_work_strict_lower_bound``
    applies to the signed input integrated up to that first crossing, not to
    instantaneous input or positive-only work. ``donor_barrier_first_required``
    is conditional on that future receiver crossing. ``simultaneous_barrier_passage_excluded``
    excludes states with both ring phase potentials at least 7/2, not
    sequential crossings or receiver acquisition. These necessary path
    restrictions do not add transfer exclusions or predict a future response.
    The shared receiver-excitation owner strengthens the energy-only order
    tests with its full nonlinear phase ceiling on the original F<=4 class.
    A false ordering/exclusion flag means its sufficient budget tests fail;
    it does not certify the opposite order or a simultaneous passage.
    ``receiver_barrier_time_loss_product_lower_bound`` bounds the product of
    that first-crossing time and accumulated full receiver dissipation. It
    evaluates neither quantity, and is not an earliest-entry time.
    """

    source: SineMediatedFormation
    target_phase_turns: tuple[Q, ...]
    target_geometry: PhaseCycleState
    compatible_form_mean: Q
    compatible_common_phase_turn_offset: Q
    target_storage_bounds: I
    target_storage_margin: Q
    target_functional_bounds: I | None
    target_functional_margin: Q | None
    entry_storage_threshold_bounds: I
    entry_loss_allowance_bounds: I
    donor_edge_mobility_max: Q
    receiver_edge_mobility_max: Q
    combined_phase_action_cost: Q | None
    necessary_entry_time_lower_bound: Q | None
    entry_through_horizon_excluded: bool
    post_time_entry_deficit_lower_bound: Q | None
    early_loss_exclusion_certified: bool
    entry_budget_status: str
    phase_action_status: str
    phase_action_unavailable_reasons: tuple[str, ...]
    hypothesis_failures: tuple[str, ...]
    exclusion_reasons: tuple[str, ...]
    unresolved_conditions: tuple[str, ...]
    status: str
    proof_id: str = "eleven_node_sine_receiver_transfer_necessary_conditions"
    scope: tuple[str, ...] = (
        "same_exact_preparation_and_complete_support_as_source_coexistence_report",
        "static_exclusions_rebuilt_from_admitted_preparation_not_cached_flags",
        "retained_early_loss_is_conditional_same_law_clock_evidence_not_replayed",
        "receiver_winding_one_donor_winding_zero_with_aligned_ports_and_uniform_form",
        "requires_effective_half_weights_beta_one_and_delta_positive_one_half",
        "target_phase_geometry_is_exact_not_a_rounded_graph_state",
        "weighted_means_allow_one_target_lift_not_a_unique_limit_or_path",
        "source_minus_target_storage_and_functional_margins_preserve_exact_cancellation",
        "joint_target_acute_entry_cost_is_B5_not_V5_plus_B5",
        "both_ring_antipodal_passages_precede_entry_without_assuming_their_order",
        "disjoint_ring_phase_action_costs_add_against_actual_total_continuous_loss",
        "strict_horizon_comparison_and_target_specific_allowance_control_timed_exclusion",
        "no_initial_or_evolved_receiver_recovery_basin_entry_certified",
        "passing_necessary_conditions_is_not_a_positive_transfer_verdict",
        "no_form_budget_cap_when_the_same_necessary_inequalities_apply",
        "no_coexistence_exclusion_transferred_to_the_receiver_only_target",
        "receiver_barrier_work_and_order_constraints_are_conditional_not_evaluated_passages",
        "receiver_excitation_owner_reuses_validated_source_to_strengthen_energy_only_barrier_order",
        "receiver_localization_owner_excludes_nonflat_receiver_limits_without_a_donor_selection",
        "signed_receiver_input_retains_full_nodal_loss_and_original_port_degree",
        "simultaneous_barrier_exclusion_does_not_exclude_sequential_receiver_acquisition",
        "no_trajectory_event_support_birth_clock_change_or_physical_identification",
        "public_dataclass_construction_is_not_provenance_authentication",
    )
    donor_well_retention: SineDonorWellRetention | None = None
    donor_dissipative_capture: SineDonorDissipativeCapture | None = None
    receiver_first_barrier_phase_storage: Q | None = None
    necessary_receiver_running_work_strict_lower_bound: Q | None = None
    donor_barrier_first_required: bool | None = None
    simultaneous_barrier_passage_excluded: bool | None = None
    receiver_barrier_time_loss_product_lower_bound: Q | None = None
    receiver_localization: SineReceiverLocalization | None = None

    def to_dict(self):
        from ..sdk.relational_reports import _project

        return {
            "schema": "tnfr.relational-sine-receiver-transfer.v1",
            "report": _project(self),
        }


@dataclass(frozen=True)
class SineDonorWellRetention:
    """Source-to-basin certificate below the exact first donor escape level.

    ``escape_margin_bounds`` is critical-minus-source, 7/2-W0. Its interval
    is only a display enclosure: the decision uses the exact polynomial
    3125-(16*F+55)^2 for the admitted nonnegative rational form storage F.
    A certified result selects the original donor-one/receiver-zero relative
    equilibrium, modulo one common phase origin and the conserved form mean.
    It does not select terminal integer phase lifts or preserve every winding
    throughout the transient. Above this sufficient threshold, ``not_certified``
    supplies no conclusion about which basin is reached.
    """

    source: SineMediatedFormation
    exchange_to_loss_ratio: Q
    sufficient_ratio_upper_bound: Q
    initial_form_storage: Q
    form_storage_coefficient: Q | None
    critical_phase_storage: Q
    critical_form_storage_bounds: I
    initial_functional_bounds: I | None
    escape_margin_bounds: I | None
    exact_retention_polynomial_margin: Q
    retained_phase_turns: tuple[Q, ...]
    retained_geometry: PhaseCycleState
    relative_donor_pattern_convergence_certified: bool
    receiver_only_targets_excluded: bool
    unavailable_reasons: tuple[str, ...]
    status: str
    proof_id: str = "eleven_node_sine_exact_donor_well_retention"
    scope: tuple[str, ...] = (
        "revalidated_exact_preparation_and_full_support_not_inherited_verdicts",
        "original_sine_law_beta_one_delta_positive_half_and_ratio_between_zero_and_three_halves",
        "strict_proper_mixed_Lyapunov_and_exact_finite_critical_catalog",
        "first_critical_level_above_single_twist_storage_is_seven_halves",
        "exact_nonnegative_form_storage_polynomial_controls_threshold_admission",
        "display_intervals_do_not_decide_the_exact_algebraic_threshold",
        "convergence_to_initial_donor_one_receiver_zero_relative_equilibrium",
        "excludes_both_receiver_only_handedness_targets_and_their_convergence_basins",
        "retained_phase_geometry_is_modulo_one_common_origin_not_terminal_integer_lifts",
        "does_not_claim_all_time_winding_preservation_or_no_transient_acute_visits",
        "no_numeric_recovery_radius_rate_deadline_or_trajectory",
        "no_law_event_support_or_clock_change_and_no_physical_identification",
        "above_sufficient_threshold_is_not_certified_not_a_transfer_verdict",
    )

    def to_dict(self):
        from ..sdk.relational_reports import _project

        return {
            "schema": "tnfr.relational-sine-donor-well-retention.v1",
            "report": _project(self),
        }


@dataclass(frozen=True)
class SineDonorDissipativeCapture:
    """Sufficient donor-component entry by one fixed analytic proof horizon.

    At T=1/4 the full-law estimates give W(T)<=V5+(4/5)*F-N/25
    and a straight phase-path storage bound V5+121*N/38400, where
    N=q0.T*K*q0. The first bound must be strictly below 7/2; N<=6F
    then implies the same for the phase path. The endpoint polynomial
    decides admission; the phase polynomial retains the derived comparison.
    Outward intervals only display values, never a reconstructed trajectory.
    Specifically, ``endpoint_functional_bounds`` and ``phase_path_storage_bounds``
    enclose the analytic upper-bound values V5+A and V5+B; their lower
    endpoints are not lower bounds on the actual evolved W or phase storage.
    Component entry implies convergence to the original relative donor
    pattern, without an all-time winding or a terminal lift prediction.
    """

    source: SineMediatedFormation
    horizon: Q
    initial_form_storage: Q
    initial_dissipative_norm_squared: Q
    functional_drop_coefficient: Q
    phase_path_storage_coefficient: Q
    functional_drop_lower_bound: Q | None
    endpoint_functional_bounds: I | None
    phase_path_storage_bounds: I | None
    endpoint_storage_allowance: Q
    phase_path_storage_allowance: Q
    endpoint_exact_polynomial_margin: Q
    phase_path_exact_polynomial_margin: Q
    retained_phase_turns: tuple[Q, ...]
    retained_geometry: PhaseCycleState
    donor_component_entry_certified: bool
    relative_donor_pattern_convergence_certified: bool
    receiver_only_targets_excluded: bool
    unavailable_reasons: tuple[str, ...]
    status: str
    proof_id: str = "eleven_node_sine_fixed_window_donor_dissipative_capture"
    scope: tuple[str, ...] = (
        "revalidated_preparation_and_recomputed_full_support_form_gradient_norm",
        "requires_original_sine_half_weights_beta_one_delta_positive_one_half",
        "fixed_quarter_unit_horizon_in_the_supplied_structural_clock",
        "full_nonlinear_gradient_majorants_not_a_frozen_tangent_evolution",
        "functional_decrease_at_least_initial_gradient_norm_squared_over_twenty_five",
        "phase_homotopy_control_and_fixed_phase_form_convexity_identify_the_donor_component",
        "single_exact_endpoint_criterion_implies_phase_path_bound_using_N_at_most_six_F",
        "display_intervals_do_not_decide_algebraic_admission",
        "donor_component_entry_by_horizon_and_original_relative_donor_limit",
        "excludes_both_receiver_only_targets_and_their_convergence_basins",
        "no_actual_endpoint_trajectory_finite_step_or_terminal_integer_lift_provided",
        "no_all_time_winding_preservation_numeric_recovery_rate_or_new_event",
        "failed_sufficient_bounds_do_not_prove_transfer_or_donor_escape",
        "no_horizon_parameter_search_clock_change_or_physical_identification",
    )

    def to_dict(self):
        from ..sdk.relational_reports import _project

        return {
            "schema": "tnfr.relational-sine-donor-dissipative-capture.v1",
            "report": _project(self),
        }


def _donor_preparation_forms(amplitude, profile, donor_epi):
    """Resolve supplied forms once without coercing exact model coordinates."""
    if not isinstance(profile, str) or profile not in (
        "localized",
        "balanced",
        "explicit",
    ):
        raise ValueError("profile must be 'localized', 'balanced' or 'explicit'")
    if profile == "explicit":
        if donor_epi is None:
            raise ValueError("explicit profile requires five donor_epi coordinates")
        forms = ordered_vector(_ordered(donor_epi, "donor_epi", limit=6), "donor_epi")
        if len(forms) != 5:
            raise ValueError("donor_epi must contain exactly five coordinates")
        return forms
    if donor_epi is not None:
        raise ValueError("donor_epi requires profile='explicit'")
    return (2 * amplitude if profile == "balanced" else Q(0),) * 5


def _reconstruct_phase_geometry(geometry, turns):
    """Admit exact circular target/preparation currents through one owner."""
    state = reconstruct_phase_cycle_state(
        geometry,
        edge_turns=tuple(
            (turns[j] - turns[i] + Q(1, 2)) % 1 - Q(1, 2) for i, j in geometry.edges
        ),
    )
    if state.sine_balance_status != "proved_by_odd_cancellation":
        raise ArithmeticError("declared preparation lost exact sine balance")
    return state


def _preparation_geometry(amplitude, contrast, donor_epi):
    """Resolved donor forms share complete-support exact phase reconstruction."""
    nodes = tuple(range(11))
    cycles = (tuple(range(5)), tuple(range(5, 10)))
    edges = tuple(
        sorted(
            [
                tuple(sorted((left, right)))
                for row in cycles
                for left, right in zip(row, row[1:] + row[:1])
            ]
            + [(0, 10), (5, 10)]
        )
    )
    geometry = _derive_phase_geometry(nodes, edges)
    initial_turns = tuple(Q(j, 5) for j in range(5)) + (Q(0),) * 6
    target_turns = tuple(Q(j, 5) for j in range(5)) * 2 + (Q(0),)

    neighbors = [[] for _ in nodes]
    for i, j in edges:
        neighbors[i].append(j)
        neighbors[j].append(i)
    epi = donor_epi + (Q(0),) * 5 + (amplitude,)
    capacity = tuple(
        1 + contrast if i == 6 else 1 - contrast if i == 9 else Q(1) for i in nodes
    )
    return dict(
        nodes=nodes,
        edges=edges,
        neighbors=tuple(tuple(row) for row in neighbors),
        degrees=tuple(map(len, neighbors)),
        cycles=cycles,
        mediator=10,
        ports=(0, 5),
        initial_epi=epi,
        initial_phase_turns=initial_turns,
        capacity=capacity,
        initial_geometry=_reconstruct_phase_geometry(geometry, initial_turns),
        target_phase_turns=target_turns,
        target_geometry=_reconstruct_phase_geometry(geometry, target_turns),
    )


def _formation_coefficients(model):
    """Admit the consumed coefficients before exact arithmetic."""
    return _sine_model_coefficients(model, positive_loss=True)


def _validated_formation_preparation(source):
    """Rebuild consumed preparation data without a field or horizon evaluation.

    Public records carry evidence but are not authenticated. Resolve primitive
    forms, profile and contrast again, and check stored derived coordinates
    before a new theorem consumes them. Existing loss evidence is not replayed.
    """
    if not isinstance(source, SineMediatedFormation):
        raise TypeError("a SineMediatedFormation source is required")
    e, w, beta = _formation_coefficients(source.model)
    amplitude = exact_or_represented_real(source.amplitude, "amplitude")
    contrast = exact_or_represented_real(source.capacity_contrast, "capacity_contrast")
    if abs(contrast) >= 1:
        raise ValueError("capacity_contrast must satisfy abs(delta)<1")
    forms = ordered_vector(source.initial_epi, "initial_epi")
    if len(forms) != 11:
        raise ValueError("initial_epi must contain all eleven preparation coordinates")
    donor = _donor_preparation_forms(
        amplitude, source.profile, forms[:5] if source.profile == "explicit" else None
    )
    preparation = _preparation_geometry(amplitude, contrast, donor)
    scalar_vectors = ("initial_epi", "initial_phase_turns", "capacity")
    for name in scalar_vectors:
        actual = ordered_vector(getattr(source, name), name)
        if actual != preparation[name]:
            raise ValueError(f"{name} does not match the declared preparation")
    # Node positions and support are exact integers, never Boolean aliases.
    for name in ("nodes", "degrees", "ports"):
        actual = getattr(source, name)
        if (
            type(actual) is not tuple
            or any(type(value) is not int for value in actual)
            or actual != preparation[name]
        ):
            raise ValueError(f"{name} does not match the preparation support")
    for name in ("edges", "neighbors", "cycles"):
        actual = getattr(source, name)
        if (
            type(actual) is not tuple
            or any(
                type(row) is not tuple or any(type(value) is not int for value in row)
                for row in actual
            )
            or actual != preparation[name]
        ):
            raise ValueError(f"{name} does not match the preparation support")
    if type(source.mediator) is not int or source.mediator != preparation["mediator"]:
        raise ValueError("mediator does not match the preparation support")
    if source.initial_geometry != preparation["initial_geometry"]:
        raise ValueError("initial_geometry does not match the declared preparation")
    form_storage = sum(
        ((forms[j] - forms[i]) ** 2 / 2 for i, j in preparation["edges"]), Q(0)
    )
    if (
        exact_or_represented_real(source.initial_form_storage, "initial_form_storage")
        != form_storage
    ):
        raise ValueError("initial_form_storage does not match the preparation")
    return e, w, beta, contrast, preparation, form_storage


def _preparation_gradient(preparation):
    """Derive the full form gradient from actual admitted primitive forms."""
    forms = preparation["initial_epi"]
    return tuple(
        sum((forms[i] - forms[j] for j in row), Q(0))
        for i, row in enumerate(preparation["neighbors"])
    )


def _silent_donor_preparation(preparation):
    forms = preparation["initial_epi"]
    return (
        forms[10] == forms[0] == 0
        and forms[1] + forms[4] == 0
        and forms[2] + forms[3] == 0
    )


def _preparation_invariants(preparation):
    weights = tuple(
        Q(degree) / nu
        for degree, nu in zip(preparation["degrees"], preparation["capacity"])
    )
    mass = sum(weights, Q(0))
    form_sum = sum((mu * x for mu, x in zip(weights, preparation["initial_epi"])), Q(0))
    phase_sum = sum(
        (mu * turn for mu, turn in zip(weights, preparation["initial_phase_turns"])),
        Q(0),
    )
    return weights, mass, form_sum, phase_sum


def _fixed_transfer_law_failures(source, e, w, beta, contrast):
    """Shared domain of the fixed-law transfer and dissipative capture proofs."""
    return tuple(
        reason
        for condition, reason in (
            (
                source.law == "normalized_sine_reciprocal_exchange",
                "requires_original_normalized_sine_law",
            ),
            (e == w == Q(1, 2), "requires_effective_half_weight_sine_law"),
            (beta == 1, "requires_unit_storage_scale"),
            (contrast == Q(1, 2), "requires_positive_half_capacity_contrast"),
        )
        if not condition
    )


def _receiver_excitation(source, admitted=None):
    """Retain exact source parity before applying the separate tangent theorem."""
    if admitted is None:
        admitted = _validated_formation_preparation(source)
    e, w, beta, contrast, preparation, form_storage = admitted
    forms = preparation["initial_epi"]
    h, d0 = forms[10], forms[0]
    s1, s2 = (forms[1] + forms[4]) / 2 - d0, (forms[2] + forms[3]) / 2 - d0
    a1, a2 = (forms[1] - forms[4]) / 2, (forms[2] - forms[3]) / 2
    even = (d0, d0 + s1, d0 + s2, d0 + s2, d0 + s1) + (Q(0),) * 5 + (h,)
    odd = (Q(0), a1, a2, -a2, -a1) + (Q(0),) * 6

    def component_evidence(values):
        component = dict(preparation, initial_epi=values)
        gradient = _preparation_gradient(component)
        storage = sum(
            ((values[i] - values[j]) ** 2 / 2 for i, j in preparation["edges"]),
            Q(0),
        )
        norm = sum(
            (
                nu * q**2 / degree
                for nu, q, degree in zip(
                    preparation["capacity"], gradient, preparation["degrees"]
                )
            ),
            Q(0),
        )
        return storage, norm

    even_storage, even_norm = component_evidence(even)
    odd_storage, odd_norm = component_evidence(odd)
    bridge_storage = ((d0 - h) ** 2 + h**2) / 2
    failures = _fixed_transfer_law_failures(source, e, w, beta, contrast)
    ceiling = strict = correction = lower = None
    correction_status = "unavailable"
    full_phase_ceiling = donor_first = donor_release = None
    full_phase_status = "unavailable_law"
    if not failures:
        ceiling = Q(3, 4) * even_storage
        strict = even_storage > 0
        if 2 * ceiling < 7:
            correction = sqrt(I(7)) - sqrt(I(2 * ceiling))
            if correction.lo > 0:
                lower = correction.lo
                correction_status = "positive_bound"
            else:
                correction_status = "unresolved_bound"
        else:
            correction_status = "nonpositive_margin"
        full_phase_status = "outside_original_form_budget"
        if form_storage <= 4:
            full_phase_ceiling = Q(139, 20)
            donor_first = True
            twist = I(*(value / 2 for value in _twist_storage_bounds(_pi_bounds())))
            donor_release = (twist - Q(69, 20)).lo
            full_phase_status = "available"
    return SineReceiverExcitation(
        source=source,
        source_coordinates=(h, d0 - h, s1, s2, a1, a2),
        even_initial_epi=even,
        odd_initial_epi=odd,
        bridge_form_storage=bridge_storage,
        even_internal_form_storage=even_storage - bridge_storage,
        odd_internal_form_storage=odd_storage,
        even_form_storage=even_storage,
        odd_form_storage=odd_storage,
        even_dissipative_norm_squared=even_norm,
        odd_dissipative_norm_squared=odd_norm,
        tangent_receiver_phase_cost_upper_bound=ceiling,
        tangent_phase_cost_bound_strict=strict,
        receiver_barrier_phase_storage=Q(7, 2),
        necessary_nonlinear_correction_norm_bounds=correction,
        necessary_nonlinear_correction_norm_lower_bound=lower,
        nonlinear_correction_status=correction_status,
        tangent_odd_receiver_silence_certified=not failures,
        unavailable_reasons=failures,
        status="unavailable" if failures else "available",
        actual_full_phase_storage_upper_bound=full_phase_ceiling,
        donor_potential_barrier_first_required=donor_first,
        necessary_donor_phase_release_lower_bound=donor_release,
        full_phase_budget_status=full_phase_status,
    )


def _receiver_localization(source, admitted=None):
    """Apply the fixed regional Lyapunov theorem to the actual preparation."""
    if admitted is None:
        admitted = _validated_formation_preparation(source)
    e, w, beta, contrast, _, form_storage = admitted
    failures = _fixed_transfer_law_failures(source, e, w, beta, contrast)
    twist = I(*(value / 2 for value in _twist_storage_bounds(_pi_bounds())))
    threshold = Q(2, 5) * twist
    unsquared = 5 - 2 * form_storage
    polynomial = unsquared**2 - 5
    certified = not failures and unsquared >= 0 and polynomial >= 0
    return SineReceiverLocalization(
        source=source,
        initial_form_storage=form_storage,
        form_storage_coefficient=None if failures else Q(1),
        mixed_term_coefficient=None if failures else Q(1, 3),
        donor_phase_weight=None if failures else Q(1),
        receiver_phase_weight=None if failures else Q(7, 5),
        bridge_phase_weight=None if failures else Q(7, 6),
        critical_form_storage_bounds=threshold,
        initial_functional_bounds=None if failures else form_storage + twist,
        receiver_nonflat_limit_functional_lower_bounds=(
            None if failures else Q(7, 5) * twist
        ),
        # Preserve the shared V5 term instead of subtracting independent
        # endpoint enclosures. This interval displays, but does not decide,
        # the exact algebraic threshold.
        target_minus_initial_margin_bounds=(
            None if failures else threshold - form_storage
        ),
        exact_localization_polynomial_margin=polynomial,
        receiver_nonflat_equilibria_excluded=certified,
        relative_receiver_consensus_certified=certified,
        unavailable_reasons=failures,
        status=(
            "unavailable" if failures else "certified" if certified else "not_certified"
        ),
    )


def _budget_status(margin):
    if margin.hi <= 0:
        return "excluded"
    return "passed" if margin.lo > 0 else "unresolved"


def _max_edge_mobility(capacity, degrees, cycle):
    mobility = tuple(nu / degree for nu, degree in zip(capacity, degrees))
    return max(mobility[i] + mobility[j] for i, j in zip(cycle, cycle[1:] + cycle[:1]))


def _phase_action_time_bound(e, w, beta, allowance, cost):
    """Bound time from independently proved squared phase excursions in pi units."""
    reasons = []
    if allowance.hi <= 0:
        reasons.append("nonpositive_loss_allowance")
    elif allowance.lo <= 0:
        reasons.append("positive_loss_allowance_unresolved")
    lower = None
    if not reasons:
        # b=w/(beta*pi): e*pi^2*cost/(b^2*allowance). Keep the
        # admitted positive rational coefficients exact, even below the
        # interval grid, rather than rounding a tiny reciprocal to zero.
        lower = e * beta**2 * cost * (pi_interval() ** 4).lo / (w**2 * allowance.hi)
    return lower, tuple(reasons)


def _phase_action_bound(preparation, e, w, beta, allowance, time):
    """Combine exact model coefficients with an outward allowed-loss bound."""
    maximum = _max_edge_mobility(
        preparation["capacity"], preparation["degrees"], preparation["cycles"][1]
    )
    lower, reasons = _phase_action_time_bound(e, w, beta, allowance, 1 / maximum)
    return SinePhaseActionBound(
        receiver_edge_mobility_max=maximum,
        allowable_loss_bounds=allowance,
        necessary_entry_time_lower_bound=lower,
        horizon=time,
        entry_through_horizon_excluded=(
            lower is not None and time is not None and time < lower
        ),
        unavailable_reasons=reasons,
        status="unavailable" if reasons else "available",
    )


def _maintained_target_obstruction(e, w, beta, contrast, form_storage, twist):
    """Apply the ratio-scoped global theorem to the admitted effective law."""
    ratio = w / e
    ratio_limit = Q(3, 2)
    reasons = []
    if not 0 < ratio < ratio_limit:
        reasons.append(
            "requires_exchange_to_loss_ratio_strictly_between_zero_and_three_halves"
        )
    if beta != 1:
        reasons.append("requires_unit_storage_scale")
    if contrast != Q(1, 2):
        reasons.append("requires_positive_half_capacity_contrast")
    coefficient = mixed = spectrum = initial = target = margin = None
    if reasons:
        status = "unavailable"
    else:
        coefficient = Q(4, 5)
        mixed = ratio / (2 * pi_interval())
        spectrum = (Q(1, 44), Q(3))
        initial = coefficient * form_storage + twist
        target = 2 * twist
        # The same V5 appears in both endpoints. Preserve that correlation
        # rather than subtracting two independently enclosed functionals.
        margin = twist - coefficient * form_storage
        status = (
            "excluded"
            if margin.lo > 0
            else "not_excluded" if margin.hi <= 0 else "unresolved"
        )
    return SineMaintainedTargetObstruction(
        form_storage_coefficient=coefficient,
        mixed_term_coefficient_bounds=mixed,
        mobility_spectral_bounds=spectrum,
        initial_functional_bounds=initial,
        target_functional_bounds=target,
        target_margin_bounds=margin,
        maintained_target_excluded=status == "excluded",
        unavailable_reasons=tuple(reasons),
        status=status,
        exchange_to_loss_ratio=ratio,
        sufficient_ratio_upper_bound=ratio_limit,
    )


def _donor_well_retention(source, admitted=None):
    """Apply the exact source-to-well threshold using the shared W owner."""
    if admitted is None:
        admitted = _validated_formation_preparation(source)
    e, w, beta, contrast, preparation, form_storage = admitted
    twist = I(*(value / 2 for value in _twist_storage_bounds(_pi_bounds())))
    functional = _maintained_target_obstruction(
        e, w, beta, contrast, form_storage, twist
    )
    reasons = list(functional.unavailable_reasons)
    if source.law != "normalized_sine_reciprocal_exchange":
        reasons.append("requires_original_normalized_sine_law")
    polynomial = 3125 - (16 * form_storage + 55) ** 2
    certified = not reasons and polynomial >= 0
    initial = None if reasons else functional.initial_functional_bounds
    return SineDonorWellRetention(
        source=source,
        exchange_to_loss_ratio=w / e,
        sufficient_ratio_upper_bound=Q(3, 2),
        initial_form_storage=form_storage,
        form_storage_coefficient=None if reasons else Q(4, 5),
        critical_phase_storage=Q(7, 2),
        critical_form_storage_bounds=(25 * sqrt(I(5)) - 55) / 16,
        initial_functional_bounds=initial,
        escape_margin_bounds=None if reasons else Q(7, 2) - initial,
        exact_retention_polynomial_margin=polynomial,
        retained_phase_turns=preparation["initial_phase_turns"],
        retained_geometry=preparation["initial_geometry"],
        relative_donor_pattern_convergence_certified=certified,
        receiver_only_targets_excluded=certified,
        unavailable_reasons=tuple(reasons),
        status=(
            "unavailable" if reasons else "certified" if certified else "not_certified"
        ),
    )


def _donor_dissipative_capture(source, admitted=None):
    """Apply fixed-window full-law capture without reusing cached gradients."""
    if admitted is None:
        admitted = _validated_formation_preparation(source)
    e, w, beta, contrast, preparation, form_storage = admitted
    gradient = _preparation_gradient(preparation)
    norm = sum(
        (
            nu * q**2 / degree
            for nu, q, degree in zip(
                preparation["capacity"], gradient, preparation["degrees"]
            )
        ),
        Q(0),
    )
    reasons = _fixed_transfer_law_failures(source, e, w, beta, contrast)
    gain, phase_coefficient = Q(1, 25), Q(121, 38400)
    endpoint_allowance = Q(4, 5) * form_storage - gain * norm
    phase_allowance = phase_coefficient * norm
    endpoint_margin = 125 - (11 + 4 * endpoint_allowance) ** 2
    phase_margin = 125 - (11 + 4 * phase_allowance) ** 2
    drop = endpoint = phase_path = None
    certified = False
    if not reasons:
        # The actual fixed support/capacities give N<=6F, so A>=14F/25.
        # Hence A<D also gives B<121D/3584<D: one admission criterion
        # suffices, while the explicit phase bound retains its causal role.
        if not 0 <= norm <= 6 * form_storage:
            raise ArithmeticError(
                "admitted preparation violates the full spectral bound"
            )
        twist = I(*(value / 2 for value in _twist_storage_bounds(_pi_bounds())))
        drop = gain * norm
        endpoint, phase_path = twist + endpoint_allowance, twist + phase_allowance
        certified = endpoint_margin > 0
    return SineDonorDissipativeCapture(
        source=source,
        horizon=Q(1, 4),
        initial_form_storage=form_storage,
        initial_dissipative_norm_squared=norm,
        functional_drop_coefficient=gain,
        phase_path_storage_coefficient=phase_coefficient,
        functional_drop_lower_bound=drop,
        endpoint_functional_bounds=endpoint,
        phase_path_storage_bounds=phase_path,
        endpoint_storage_allowance=endpoint_allowance,
        phase_path_storage_allowance=phase_allowance,
        endpoint_exact_polynomial_margin=endpoint_margin,
        phase_path_exact_polynomial_margin=phase_margin,
        retained_phase_turns=preparation["initial_phase_turns"],
        retained_geometry=preparation["initial_geometry"],
        donor_component_entry_certified=certified,
        relative_donor_pattern_convergence_certified=certified,
        receiver_only_targets_excluded=certified,
        unavailable_reasons=reasons,
        status=(
            "unavailable" if reasons else "certified" if certified else "not_certified"
        ),
    )


def _receiver_transfer(source):
    """Reuse one exact preparation with a separately declared maintained endpoint."""
    admitted = _validated_formation_preparation(source)
    e, w, beta, contrast, preparation, form_storage = admitted
    equilibrium = not any(_preparation_gradient(preparation))
    silent_donor = _silent_donor_preparation(preparation)
    weights, mass, form_sum, phase_sum = _preparation_invariants(preparation)
    retained = _donor_well_retention(source, admitted)
    captured = _donor_dissipative_capture(source, admitted)
    localized = _receiver_localization(source, admitted)
    target = (Q(0),) * 5 + tuple(Q(j, 5) for j in range(5)) + (Q(0),)
    geometry = _reconstruct_phase_geometry(
        _derive_phase_geometry(preparation["nodes"], preparation["edges"]), target
    )
    weighted_target_turns = sum((mu * turn for mu, turn in zip(weights, target)), Q(0))
    offset = (phase_sum - weighted_target_turns) / mass
    donor_max, receiver_max = (
        _max_edge_mobility(preparation["capacity"], preparation["degrees"], cycle)
        for cycle in preparation["cycles"]
    )
    failures = _fixed_transfer_law_failures(source, e, w, beta, contrast)
    pi_bounds = _pi_bounds()
    twist = I(*(value / 2 for value in _twist_storage_bounds(pi_bounds)))
    _, face_bounds = _cycle_barrier_constants(pi_bounds)
    face = I(*face_bounds)
    threshold = beta * face
    # The donor may unwind before receiver acquisition: its original V5 is
    # part of the initial energy, not a mandatory cost at the transfer face.
    allowance = I(form_storage) + beta * (twist - face)
    functional = functional_margin = cost = lower = deficit = None
    entry_status = "unavailable"
    action_reasons = ("transfer_law_hypotheses_unavailable",)
    entry_excluded = early_excluded = False
    receiver_barrier = running_work = donor_first = simultaneous_excluded = None
    receiver_time_loss = None
    exclusions, unresolved = [], []
    if not failures:
        receiver_barrier = Q(7, 2)
        running_work = beta * receiver_barrier
        donor_first = form_storage <= running_work
        # With beta=1, simultaneous ring barriers require
        # F>7-V5=(3+5*sqrt(5))/4. Keep this closed exclusion exact,
        # independently of the displayed trigonometric storage intervals.
        algebraic_gap = 4 * form_storage - 3
        simultaneous_excluded = algebraic_gap <= 0 or algebraic_gap**2 <= 125
        excitation = _receiver_excitation(source, admitted)
        if excitation.full_phase_budget_status == "available":
            donor_first = (
                donor_first or excitation.donor_potential_barrier_first_required is True
            )
            simultaneous_excluded = simultaneous_excluded or (
                excitation.actual_full_phase_storage_upper_bound < 2 * receiver_barrier
            )
        receiver_time_loss = (Q(14, 3) * pi_interval() ** 2).lo
        functional, functional_margin = twist, Q(4, 5) * form_storage
        entry_status = _budget_status(allowance)
        # Both initially distinct windings must change before entry. The
        # ring node sets are disjoint, so their Cauchy--Schwarz loss costs add.
        cost = Q(3, 5) ** 2 / donor_max + 1 / receiver_max
        lower, action_reasons = _phase_action_time_bound(e, w, beta, allowance, cost)
        time = source.exclusion_time
        if time is not None:
            time = exact_or_represented_real(time, "exclusion_time")
            if time <= 0:
                raise ValueError("exclusion_time must be strictly positive")
        entry_excluded = lower is not None and time is not None and time < lower
        if source.combined_early_loss_lower_bound is not None:
            loss = exact_or_represented_real(
                source.combined_early_loss_lower_bound,
                "combined_early_loss_lower_bound",
            )
            if time is None or loss < 0:
                raise ValueError(
                    "retained early loss requires a horizon and nonnegative bound"
                )
            deficit = loss - allowance.hi
            early_excluded = entry_excluded and deficit > 0
        if equilibrium:
            exclusions.append("initial_equilibrium")
        if silent_donor and not equilibrium:
            exclusions.append("exact_silent_donor_sign_reflection")
        if entry_status == "excluded":
            exclusions.append("strict_receiver_transfer_entry_storage_requirement")
        elif entry_status == "unresolved":
            unresolved.append("receiver_transfer_entry_storage_sign_unresolved")
        if early_excluded:
            exclusions.append("early_dissipation_before_receiver_transfer_entry")
        if retained.receiver_only_targets_excluded:
            exclusions.append("exact_donor_well_retention")
        if captured.receiver_only_targets_excluded:
            exclusions.append("donor_dissipative_capture")
        if localized.receiver_nonflat_equilibria_excluded:
            exclusions.append("weighted_receiver_localization")
    status = (
        "unavailable"
        if failures
        else (
            "excluded"
            if exclusions
            else "unresolved_bound" if unresolved else "passes_necessary_conditions"
        )
    )
    return SineReceiverTransferAdmission(
        source=source,
        target_phase_turns=target,
        target_geometry=geometry,
        compatible_form_mean=form_sum / mass,
        compatible_common_phase_turn_offset=offset,
        target_storage_bounds=beta * twist,
        target_storage_margin=form_storage,
        target_functional_bounds=functional,
        target_functional_margin=functional_margin,
        entry_storage_threshold_bounds=threshold,
        entry_loss_allowance_bounds=allowance,
        donor_edge_mobility_max=donor_max,
        receiver_edge_mobility_max=receiver_max,
        combined_phase_action_cost=cost,
        necessary_entry_time_lower_bound=lower,
        entry_through_horizon_excluded=entry_excluded,
        post_time_entry_deficit_lower_bound=deficit,
        early_loss_exclusion_certified=early_excluded,
        entry_budget_status=entry_status,
        phase_action_status="unavailable" if action_reasons else "available",
        phase_action_unavailable_reasons=action_reasons,
        hypothesis_failures=failures,
        exclusion_reasons=tuple(exclusions),
        unresolved_conditions=tuple(unresolved),
        status=status,
        donor_well_retention=retained,
        donor_dissipative_capture=captured,
        receiver_first_barrier_phase_storage=receiver_barrier,
        necessary_receiver_running_work_strict_lower_bound=running_work,
        donor_barrier_first_required=donor_first,
        simultaneous_barrier_passage_excluded=simultaneous_excluded,
        receiver_barrier_time_loss_product_lower_bound=receiver_time_loss,
        receiver_localization=localized,
    )


def _positive_part_square_integral(magnitude, speed, time):
    """Integrate max(0, magnitude-speed*t)^2 with exact bound coefficients."""
    stop = min(time, magnitude / speed)
    integral = (
        magnitude**2 * stop - magnitude * speed * stop**2 + speed**2 * stop**3 / 3
    )
    return stop, integral


def _directional_loss_bound(preparation, gradient, e, w, beta, time):
    """Retain exact full-support moments before bounding a finite-time loss."""
    capacity = preparation["capacity"]
    mobility = tuple(
        nu / degree for nu, degree in zip(capacity, preparation["degrees"])
    )
    v = tuple(k * q for k, q in zip(mobility, gradient))
    lv = tuple(
        sum((v[i] - v[j] for j in row), Q(0))
        for i, row in enumerate(preparation["neighbors"])
    )
    norm = sum((k * q**2 for k, q in zip(mobility, gradient)), Q(0))
    first = sum(((v[i] - v[j]) ** 2 for i, j in preparation["edges"]), Q(0))
    second = sum((k * value**2 for k, value in zip(mobility, lv)), Q(0))
    spectral = 2 * max(capacity)
    ab_upper = w**2 / (beta * pi_interval().lo ** 2)
    omega_squared = ab_upper * spectral**2
    argument_squared = omega_squared * time**2 if time is not None else None
    cosh_upper, cosh_method = None, None
    if argument_squared is not None:
        if argument_squared <= Q(4, 25):
            # Preserve the established short-window certificate values.
            cosh_upper, cosh_method = Q(11, 10), "fixed_short_window_11_over_10"
        elif argument_squared < 2:
            # (2j)! >= 2^j: sum v^j/(2j)! <= 1/(1-v/2) for 0 <= v < 2.
            cosh_upper = 1 / (1 - argument_squared / 2)
            cosh_method = "factorial_geometric_series"
    rayleigh = first / norm if norm else None
    gamma_squared = second / norm if norm else None
    gamma_upper = sqrt(I(gamma_squared)).hi if norm else None
    linear = e * rayleigh if norm else None
    quadratic = (
        (cosh_upper / 2) * gamma_upper * ab_upper * spectral
        if norm and cosh_upper is not None
        else None
    )
    ratio = (
        1 - linear * time - quadratic * time**2
        if norm and time is not None and quadratic is not None
        else None
    )
    reasons = []
    if time is None:
        reasons.append("exclusion_time_not_supplied")
    if not norm:
        reasons.append("zero_initial_form_gradient")
    if argument_squared is not None and argument_squared >= 2:
        reasons.append("hyperbolic_argument_bound_exceeded")
    if ratio is not None and ratio <= 0:
        reasons.append("positive_norm_ratio_not_certified")
    loss = None
    if not reasons:
        integral = (
            time
            - linear * time**2
            + (linear**2 - 2 * quadratic) * time**3 / 3
            + linear * quadratic * time**4 / 2
            + quadratic**2 * time**5 / 5
        )
        loss = e * norm * integral
    return SineDirectionalLossBound(
        horizon=time,
        initial_dissipative_norm_squared=norm,
        first_spectral_moment=first,
        second_spectral_moment=second,
        rayleigh_quotient=rayleigh,
        gamma_squared=gamma_squared,
        gamma_upper_bound=gamma_upper,
        spectral_upper_bound=spectral,
        oscillation_rate_squared_upper_bound=omega_squared,
        hyperbolic_argument_squared_upper_bound=argument_squared,
        cosh_upper_bound=cosh_upper,
        cosh_bound_method=cosh_method,
        linear_coefficient=linear,
        quadratic_coefficient=quadratic,
        norm_ratio_lower_bound_at_horizon=ratio,
        loss_lower_bound=loss,
        unavailable_reasons=tuple(reasons),
        status="unavailable" if reasons else "available",
    )


def _early_loss_bounds(
    preparation, gradient, e, w, beta, energy_upper, form_storage, time, face, action
):
    """Global energy bounds imply a finite-time obstruction without integration."""
    directional = _directional_loss_bound(preparation, gradient, e, w, beta, time)
    if time is None:
        return dict(
            exclusion_time=None,
            form_speed_upper_bounds=None,
            mediator_gradient_speed_upper_bound=None,
            nodal_gradient_speed_upper_bounds=None,
            receiver_gap_speed_upper_bound=None,
            receiver_gap_drift_upper_bound=None,
            receiver_acute_time_margin_lower_bound=None,
            loss_integration_time=None,
            early_loss_lower_bound=None,
            nodal_loss_integration_times=None,
            nodal_early_loss_lower_bounds=None,
            full_early_loss_lower_bound=None,
            directional_loss_bound=directional,
            combined_early_loss_lower_bound=None,
            post_time_entry_deficit_lower_bound=None,
            early_loss_exclusion_certified=False,
        )
    pi = pi_interval()
    a, b = w / pi, (w / beta) / pi
    degrees, capacity = preparation["degrees"], preparation["capacity"]
    speeds = tuple(
        (nu * (e * sqrt(I(2 * energy_upper / degree)) + a)).hi
        for degree, nu in zip(degrees, capacity)
    )
    gradient_speeds = tuple(
        degree * speeds[i] + sum((speeds[j] for j in row), Q(0))
        for i, (degree, row) in enumerate(zip(degrees, preparation["neighbors"]))
    )
    phase_speeds = tuple(
        (b * nu * sqrt(I(2 * energy_upper / degree))).hi
        for degree, nu in zip(degrees, capacity)
    )
    receiver = preparation["cycles"][1]
    gap_speed = max(
        phase_speeds[i] + phase_speeds[j]
        for i, j in zip(receiver, receiver[1:] + receiver[:1])
    )
    drift = time * gap_speed
    time_margin = pi.lo / 2 - drift
    integrals = tuple(
        _positive_part_square_integral(abs(q), speed, time)
        for q, speed in zip(gradient, gradient_speeds)
    )
    losses = tuple(
        e * nu * integral / degree
        for nu, degree, (_, integral) in zip(capacity, degrees, integrals)
    )
    full_loss = sum(losses, Q(0))
    combined_loss = max(full_loss, directional.loss_lower_bound or Q(0))
    mediator = preparation["mediator"]
    deficit = beta * face.lo + combined_loss - form_storage
    return dict(
        exclusion_time=time,
        form_speed_upper_bounds=speeds,
        mediator_gradient_speed_upper_bound=gradient_speeds[mediator],
        nodal_gradient_speed_upper_bounds=gradient_speeds,
        receiver_gap_speed_upper_bound=gap_speed,
        receiver_gap_drift_upper_bound=drift,
        receiver_acute_time_margin_lower_bound=time_margin,
        loss_integration_time=integrals[mediator][0],
        early_loss_lower_bound=losses[mediator],
        nodal_loss_integration_times=tuple(stop for stop, _ in integrals),
        nodal_early_loss_lower_bounds=losses,
        full_early_loss_lower_bound=full_loss,
        directional_loss_bound=directional,
        combined_early_loss_lower_bound=combined_loss,
        post_time_entry_deficit_lower_bound=deficit,
        early_loss_exclusion_certified=(
            (time_margin > 0 or action.entry_through_horizon_excluded) and deficit > 0
        ),
    )


def assess_sine_mediated_formation(
    *,
    model,
    amplitude,
    capacity_contrast,
    exclusion_time=None,
    profile="localized",
    donor_epi=None,
) -> SineMediatedFormation:
    """Assess declared donor/flat-receiver profiles without a response run.

    ``localized`` supplies intermediary form A and every other form zero.
    ``balanced`` supplies donor form 2*A, intermediary form A and receiver
    form zero. Both cost A^2 in form storage, but their actual nodal gradients
    and loss differ. ``explicit`` instead requires five ordered ``donor_epi``
    coordinates, retaining ``amplitude`` as the intermediary form H and zero
    receiver form. It does not require constant donor form or F=H^2. No
    profile is selected or optimized by this reader.

    Exact rational inputs remain exact; other real scalars use the shared
    finite represented boundary. Require |delta|<1 and positive dissipation.
    Optional positive ``exclusion_time`` is an analytic bound horizon in the
    sine model's clock, not a requested numerical trajectory or SI time.
    Failure of that optional sufficient
    exclusion leaves formation unresolved unless another obstruction applies.
    The necessary phase-action bound is also returned without a time horizon;
    when available it excludes joint entry before its lower bound, but does
    not predict that any entry subsequently occurs.

    E0=F+beta*V5 for the actual full-support form storage F, and the target
    costs 2*beta*V5. First joint entry into the
    acute winding-one product costs at least beta*(V5+B5), where
    B5=5-4*cos(3*pi/8). Any nonzero initial form gradient gives strict initial
    loss, requiring F>beta*B5. A zero intermediary form alone is not an
    equilibrium or a guarantee of no subsequent receiver response.
    This first-entry argument does not assume the donor keeps its winding
    before entry. The target has both rings twisted and aligned to the live
    intermediary, modulo common form and phase origins; its representative
    does not fix the final intermediary phase or common form to zero.
    Receiver-only formation with donor loss is a separate claim.

    At 0<w/e<3/2, beta=1 and delta=+1/2, the separate mixed-functional
    certificate excludes maintained target convergence when
    V5-(4/5)*F is proved positive. This global conclusion is not a
    claim that every transient joint acute visit is impossible. The ratio
    uses the exact represented effective model weights after normalization;
    common scaling of requested raw weights is not a change of clock.

    The separate exact condition H=u0=0, u1+u4=u2+u3=0 fixes the initial
    state under donor reflection combined with global form/phase inversion.
    Uniqueness preserves this symmetry: port zero, intermediary and receiver
    remain silent even with nonzero donor motion and capacity contrast.
    """
    e, w, beta = _formation_coefficients(model)
    amplitude = exact_or_represented_real(amplitude, "amplitude")
    donor_epi = _donor_preparation_forms(amplitude, profile, donor_epi)
    contrast = exact_or_represented_real(capacity_contrast, "capacity_contrast")
    if abs(contrast) >= 1:
        raise ValueError("capacity_contrast must satisfy abs(delta)<1")
    if exclusion_time is not None:
        exclusion_time = exact_or_represented_real(exclusion_time, "exclusion_time")
        if exclusion_time <= 0:
            raise ValueError("exclusion_time must be strictly positive")

    preparation = _preparation_geometry(amplitude, contrast, donor_epi)
    epi = preparation["initial_epi"]
    gradient = _preparation_gradient(preparation)
    equilibrium = not any(gradient)
    silent_donor = _silent_donor_preparation(preparation)
    work = _sine_work(
        model,
        preparation["degrees"],
        gradient,
        preparation["capacity"],
        (I(0),) * len(epi),
    )
    form_storage = sum(
        ((epi[j] - epi[i]) ** 2 / 2 for i, j in preparation["edges"]), Q(0)
    )
    pi_bounds = _pi_bounds()
    twist = I(*(value / 2 for value in _twist_storage_bounds(pi_bounds)))
    _, face_bounds = _cycle_barrier_constants(pi_bounds)
    face = I(*face_bounds)
    initial_storage = I(form_storage) + beta * twist
    target_margin = I(form_storage) - beta * twist
    entry_margin = I(form_storage) - beta * face
    target_status, entry_status = map(_budget_status, (target_margin, entry_margin))
    action = _phase_action_bound(preparation, e, w, beta, entry_margin, exclusion_time)
    maintained = _maintained_target_obstruction(
        e, w, beta, contrast, form_storage, twist
    )
    odd_numerator = -e * w * contrast * amplitude / (3 * beta)
    early = _early_loss_bounds(
        preparation,
        gradient,
        e,
        w,
        beta,
        initial_storage.hi,
        form_storage,
        exclusion_time,
        face,
        action,
    )
    invariant_weights, mass, form_sum, phase_sum = _preparation_invariants(preparation)

    exclusions = []
    if equilibrium:
        exclusions.append("initial_equilibrium")
    if not contrast:
        exclusions.append("exact_receiver_reflection")
    if silent_donor and not equilibrium:
        exclusions.append("exact_silent_donor_sign_reflection")
    if target_status == "excluded":
        exclusions.append("strict_target_storage_requirement")
    if entry_status == "excluded":
        exclusions.append("strict_joint_acute_entry_storage_requirement")
    if early["early_loss_exclusion_certified"]:
        exclusions.append("early_dissipation_before_joint_acute_entry")
    if maintained.maintained_target_excluded:
        exclusions.append("maintained_target_lyapunov_obstruction")
    unresolved = tuple(
        name
        for state, name in (
            (target_status, "target_storage_sign_unresolved"),
            (entry_status, "joint_acute_entry_storage_sign_unresolved"),
            (maintained.status, "maintained_target_lyapunov_margin_unresolved"),
        )
        if state == "unresolved"
    )
    status = (
        "excluded"
        if exclusions
        else "unresolved_bound" if unresolved else "passes_necessary_conditions"
    )
    return SineMediatedFormation(
        model=model,
        amplitude=amplitude,
        capacity_contrast=contrast,
        profile=profile,
        **preparation,
        initial_form_gradient=gradient,
        initial_form_rate_bounds=work["form_rates"],
        initial_phase_rate_bounds=work["phase_rates"],
        phase_odd_acceleration_pi_numerator=odd_numerator,
        phase_odd_acceleration_bounds=odd_numerator / pi_interval(),
        phase_odd_acceleration_sign=int(odd_numerator > 0) - int(odd_numerator < 0),
        reflection_invariant=not contrast,
        silent_donor_subspace=silent_donor,
        initial_equilibrium=equilibrium,
        initial_form_storage=form_storage,
        twist_phase_storage_bounds=twist,
        acute_face_phase_storage_bounds=face,
        initial_storage_bounds=initial_storage,
        target_storage_bounds=2 * beta * twist,
        entry_storage_threshold_bounds=beta * (twist + face),
        initial_continuous_loss=work["continuous_loss"],
        initial_loss_per_form_storage=(
            work["continuous_loss"] / form_storage if form_storage else None
        ),
        invariant_weights=invariant_weights,
        weighted_coordinate_mass=mass,
        initial_weighted_form_sum=form_sum,
        initial_weighted_phase_turn_sum=phase_sum,
        conserved_form_mean=form_sum / mass,
        conserved_lifted_phase_turn_mean=phase_sum / mass,
        initial_storage_rate=-work["continuous_loss"],
        initial_balance_residual_bounds=work["balance_residual"],
        target_storage_margin_bounds=target_margin,
        entry_storage_margin_bounds=entry_margin,
        target_budget_status=target_status,
        entry_budget_status=entry_status,
        phase_action_bound=action,
        maintained_target_obstruction=maintained,
        **early,
        exclusion_reasons=tuple(exclusions),
        unresolved_conditions=unresolved,
        status=status,
    )
