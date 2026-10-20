"""Four-channel actuation from eight initial child observations.

ORGANIZED_PARENT_COLLECTIVE_ACTUATION.md owns the fixed input and error
budgets. Projection constructs a reference; the actual fine state remains
unobserved and is carried under the same supplied input by analytic bounds.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from .._exact_time import exact_or_represented_real as _exact
from ..mathematics._rational_interval import I, pi_interval, sin
from ._organized_parent_acquisition import _OrganizedParentAcquisition
from ._organized_parent_constructive_actuation import (
    _CLOCK_FACTOR,
    _SLOW_DURATION,
    _WORK_BUDGET,
    _assess_organized_parent_constructive_actuation,
    _evaluate_parent_actuation_path,
    _OrganizedParentConstructiveActuation,
    _ParentActuationPath,
)
from ._organized_parent_restoring_response import (
    _EVENT,
    _OrganizedParentRestoringResponse,
)
from ._repeated_organized_composition import _vector
from ._sine_preparation import _prepared_heat_error_radii

_CHILD_DISPLACEMENT = tuple(_EVENT[9 * child] for child in range(4))
_SQUARE_EDGES = ((0, 1), (2, 3), (0, 2), (1, 3))
_ENDPOINT_BUDGET, _PREDICTION_BUDGET = Q(1, 10**7), Q(1, 10**6)


@dataclass(frozen=True)
class _CollectiveParentInput:
    """Prescribed input and predictor initialization from only eight means.

    The reference supplies the cubic state and derivative expressions. Actual
    input channels below use the exact four-child restriction: internal C9
    currents cancel, and the complete law retains denominator four.
    """

    child_form_means: tuple[Q, ...]
    child_phase_offset_means: tuple[Q, ...]
    path_fraction: Q
    reference_path: _ParentActuationPath
    child_phase_current_bounds: tuple[I, ...]
    child_form_input_tau_bounds: tuple[I, ...]
    child_form_input_slow_bounds: tuple[I, ...]
    child_form_input_original_bounds: tuple[I, ...]
    form_input_tau_bounds: tuple[I, ...]
    form_input_slow_bounds: tuple[I, ...]
    form_input_original_bounds: tuple[I, ...]
    predictor_initial_form_means: tuple[Q, ...]
    predictor_initial_phase_offset_means: tuple[Q, ...]
    child_phase_displacement: tuple[Q, ...] = _CHILD_DISPLACEMENT
    slow_duration: Q = _SLOW_DURATION
    scope: tuple[str, ...] = (
        "only_eight_initial_child_means_and_time_enter_the_recipe",
        "all_nine_primitives_admitted_before_broadcast_or_arithmetic",
        "projection_is_a_reference_not_a_reset_of_the_actual_fine_state",
        "exact_child_restriction_retains_denominator_four_and_carrier_cancellation",
        "four_identical_within_child_form_input_channels_without_fine_feedback",
        "predictor_initialized_from_initial_projected_target_not_release_observations",
        "reference_path_fields_are_analytic_enclosures_not_actual_response_samples",
        "no_source_authentication_trajectory_or_autonomous_controller",
    )


def _evaluate_collective_parent_input(
    *, child_form_means, child_phase_offset_means, path_fraction
):
    """Evaluate the fixed input from four form and four phase-offset means."""
    forms = _vector(child_form_means, "child_form_means", 4)
    phases = _vector(child_phase_offset_means, "child_phase_offset_means", 4)
    v = _exact(path_fraction, "path_fraction")
    if not 0 <= v <= 1:
        raise ValueError("path_fraction requires 0 <= path_fraction <= 1")
    reference = _evaluate_parent_actuation_path(
        form=tuple(value for value in forms for _ in range(9)),
        phase_offsets=tuple(value for value in phases for _ in range(9)),
        path_fraction=v,
    )
    phase = tuple(reference.phase_offset_bounds[9 * child] for child in range(4))
    current = [I(0) for _ in range(4)]
    for a, b in _SQUARE_EDGES:
        value = sin(phase[b] - phase[a])
        current[a] += value
        current[b] -= value
    gamma = reference.geometry.gamma_bounds
    forcing = tuple(
        reference.form_tau_rate_bounds[9 * child]
        + (reference.path_form_gradient_bounds[9 * child] - gamma * current[child]) / 4
        for child in range(4)
    )
    slow = tuple(2 * value / gamma**2 for value in forcing)
    original = tuple(_CLOCK_FACTOR * value for value in forcing)
    return _CollectiveParentInput(
        child_form_means=forms,
        child_phase_offset_means=phases,
        path_fraction=v,
        reference_path=reference,
        child_phase_current_bounds=tuple(current),
        child_form_input_tau_bounds=forcing,
        child_form_input_slow_bounds=slow,
        child_form_input_original_bounds=original,
        form_input_tau_bounds=tuple(value for value in forcing for _ in range(9)),
        form_input_slow_bounds=tuple(value for value in slow for _ in range(9)),
        form_input_original_bounds=tuple(value for value in original for _ in range(9)),
        predictor_initial_form_means=forms,
        predictor_initial_phase_offset_means=tuple(
            value + change for value, change in zip(phases, _CHILD_DISPLACEMENT)
        ),
    )


@dataclass(frozen=True)
class _OrganizedParentCollectiveActuation:
    """New same-input bounds and source-inclusive realization/release flags.

    Forced/readout/full-interval errors compare actual and projected-reference
    flows over elapsed slow times four/five/eight. Release errors instead
    compare the actual endpoint continuation with the previous exact-target
    branch; they include the original discarded residual a second time.
    """

    form_error_bounds: tuple[Q, ...]
    phase_error_bounds: tuple[Q, ...]
    construction: _OrganizedParentConstructiveActuation
    acquisition: _OrganizedParentAcquisition
    restoring_response: _OrganizedParentRestoringResponse
    forced_form_error_upper_bound: Q
    forced_phase_error_upper_bound: Q
    readout_form_error_upper_bound: Q
    readout_phase_error_upper_bound: Q
    full_interval_form_error_upper_bound: Q
    full_interval_phase_error_upper_bound: Q
    endpoint_form_error_upper_bound: Q
    endpoint_phase_error_upper_bound: Q
    child_form_mean_prediction_error_upper_bound: Q
    child_phase_mean_prediction_error_upper_bound: Q
    release_form_error_upper_bound: Q
    release_phase_error_upper_bound: Q
    reference_input_norm_upper_bound: Q
    reference_input_work_upper_bound: Q
    input_work_error_upper_bound: Q
    input_work_upper_bound: Q
    work_budget_margin: Q
    forced_child_positive_gap_margin: Q
    forced_child_acute_gap_margin: Q
    child_positive_gap_margin: Q
    child_acute_gap_margin: Q
    post_actuation_disruption_margin: Q
    parent_gap_at_recovery_upper_bound: Q
    elapsed_scaled_duration_bounds: I
    elapsed_original_duration_bounds: I
    sufficient_margins: tuple[tuple[str, Q], ...]
    source_certified: bool
    conditional_realization_certified: bool
    endpoint_certified: bool
    work_certified: bool
    child_identities_retained: bool
    disruption_certified: bool
    prediction_certified: bool
    recovery_certified: bool
    realization_certified: bool
    release_certified: bool
    admitted: bool
    status: str
    unresolved_conditions: tuple[str, ...]
    phase_displacement: tuple[Q, ...] = _EVENT
    source_slow_time: Q = Q(4)
    slow_duration: Q = _SLOW_DURATION
    release_slow_time: Q = Q(8)
    readout_slow_time: Q = Q(9)
    recovery_slow_time: Q = Q(12)
    endpoint_error_budget: Q = _ENDPOINT_BUDGET
    prediction_error_budget: Q = _PREDICTION_BUDGET
    work_budget: Q = _WORK_BUDGET
    protocol_revision: str = "a5fe8f3a"
    clock: str = "tau=(1023/1024)*t; s=gamma^2*tau/2; v=(s-4)/4"
    prediction_initialization: str = "initial eight means mapped to (X4,Phi4+k_child)"
    scope: tuple[str, ...] = (
        "original_eight_source_allowances_readmitted_before_fresh_construction",
        "fresh_projection_compatible_bounds_not_old_full_state_path_verdicts",
        "orthogonal_projection_discards_no_actual_fine_state",
        "actual_and_reference_receive_the_identical_prescribed_four_channel_input",
        "same_input_cancellation_not_comparison_of_different_state_dependent_recipes",
        "separate_full_L2_form_and_phase_endpoint_budgets_no_exact_endpoint_claim",
        "actual_work_includes_bounded_power_error_against_projected_reference",
        "all_eight_mean_forecasts_use_only_initial_observations_no_release_measurement",
        "continuous_child_retention_from_same_input_comparison_through_slow_time_twelve",
        "approximate_endpoint_mismatch_carried_into_disruption_and_parent_recovery",
        "old_prediction_contrast_work_and_overall_verdicts_do_not_gate_new_claims",
        "exact_initial_observations_and_child_uniform_input_distribution_are_premises",
        "no_noise_tracking_error_positive_only_supply_or_bandwidth_admission",
        "no_reset_trajectory_solver_search_registered_operator_or_autonomous_occurrence",
    )


def _assess_organized_parent_collective_actuation(
    *, form_error_bounds, phase_error_bounds
) -> _OrganizedParentCollectiveActuation:
    """Rebuild original sources and assess only the frozen collective recipe.

    A fresh full-state construction supplies its original acquisition and
    algebraic projected-source work bound. None of that construction's path
    or release verdicts substitute for this input's new comparison proof.
    """
    rx = _vector(form_error_bounds, "form_error_bounds", 4)
    rt = _vector(phase_error_bounds, "phase_error_bounds", 4)
    if min(*rx, *rt) < 0:
        raise ValueError("original preparation errors must be nonnegative")
    construction = _assess_organized_parent_constructive_actuation(
        form_error_bounds=rx, phase_error_bounds=rt
    )
    acquisition, restoring = construction.acquisition, construction.restoring_response
    geometry = acquisition.geometry
    gamma, h = geometry.gamma_bounds, _SLOW_DURATION
    initial_form = acquisition.full_form_error_upper_bound
    initial_phase = acquisition.full_phase_error_upper_bound

    def comparison(elapsed):
        return _prepared_heat_error_radii(
            initial_form_error=initial_form,
            initial_phase_error=initial_phase,
            gamma_upper=gamma.hi,
            phase_growth_upper=Q(3 ** (4 * elapsed)),
            form_convolution_factor=Q(24),
        )

    forced_form, forced_phase = comparison(4)
    readout_form, readout_phase = comparison(5)
    all_form, all_phase = comparison(8)
    endpoint_form, endpoint_phase = (
        forced_form + initial_form,
        forced_phase + initial_phase,
    )
    release_form, release_phase = _prepared_heat_error_radii(
        initial_form_error=endpoint_form,
        initial_phase_error=endpoint_phase,
        gamma_upper=gamma.hi,
        phase_growth_upper=Q(3**16),
        form_convolution_factor=Q(24),
    )
    m = construction.initial_velocity_coefficient_norm_upper_bound
    k = Q(9, 8)  # A strict rational upper bound on the full displacement norm.
    input_norm = (
        3 * gamma.hi**3 / h**2 * (k + m)
        + gamma.hi / (4 * h) * (3 * k + 2 * m)
        + 3 * gamma.hi
    )
    work_error = 16 * h * forced_form * input_norm / gamma.lo**2
    work_upper = construction.input_work_upper_bound + work_error
    work_margin = _WORK_BUDGET - work_upper
    forced_positive = geometry.alpha_bounds.lo - 2 * forced_phase
    forced_acute = pi_interval().lo / 2 - geometry.alpha_bounds.hi - 2 * forced_phase
    child_positive = geometry.alpha_bounds.lo - 2 * all_phase
    child_acute = pi_interval().lo / 2 - geometry.alpha_bounds.hi - 2 * all_phase
    disruption = restoring.post_event_disruption_margin - 2 * endpoint_phase
    recovery_gap = restoring.parent_gap_at_recovery_upper_bound + 2 * release_phase
    recovery_margin = acquisition.parent_identity_gap - recovery_gap
    form_prediction, phase_prediction = readout_form / 3, readout_phase / 3
    endpoint_margins = (
        ("full_form_endpoint", _ENDPOINT_BUDGET - endpoint_form),
        ("full_phase_endpoint", _ENDPOINT_BUDGET - endpoint_phase),
    )
    prediction_margins = (
        ("four_form_mean_predictions", _PREDICTION_BUDGET - form_prediction),
        ("four_phase_mean_predictions", _PREDICTION_BUDGET - phase_prediction),
    )
    source_conditions = (
        ("original_children_not_certified", acquisition.sources_certified),
        ("original_parent_entry_not_certified", acquisition.entry_certified),
        (
            "original_child_identities_not_certified",
            acquisition.child_identities_retained,
        ),
    )
    source_ok = all(passed for _, passed in source_conditions)
    endpoint_ok = all(value >= 0 for _, value in endpoint_margins)
    prediction_ok = all(value >= 0 for _, value in prediction_margins)
    work_ok, forced_children = work_margin >= 0, min(forced_positive, forced_acute) > 0
    children_ok = min(child_positive, child_acute) > 0
    disruption_ok = disruption > 0
    recovery_ok = restoring.recovery_certified and recovery_margin > 0
    conditional = endpoint_ok and work_ok and forced_children and disruption_ok
    realization = source_ok and conditional
    release = realization and children_ok and prediction_ok and recovery_ok
    margins = (
        endpoint_margins
        + prediction_margins
        + (
            ("actual_input_work_budget", work_margin),
            ("forced_child_positive_gap", forced_positive),
            ("forced_child_acute_gap", forced_acute),
            ("all_interval_child_positive_gap", child_positive),
            ("all_interval_child_acute_gap", child_acute),
            ("parent_registration_disrupted", disruption),
            ("parent_identity_at_recovery", recovery_margin),
        )
    )
    decisions = source_conditions + (
        ("endpoint_not_certified", endpoint_ok),
        ("actual_input_work_not_certified", work_ok),
        ("forced_child_identities_not_certified", forced_children),
        ("whole_interval_child_identities_not_certified", children_ok),
        ("parent_disruption_not_certified", disruption_ok),
        ("eight_mean_prediction_not_certified", prediction_ok),
        ("parent_recovery_not_certified", recovery_ok),
    )
    return _OrganizedParentCollectiveActuation(
        form_error_bounds=rx,
        phase_error_bounds=rt,
        construction=construction,
        acquisition=acquisition,
        restoring_response=restoring,
        forced_form_error_upper_bound=forced_form,
        forced_phase_error_upper_bound=forced_phase,
        readout_form_error_upper_bound=readout_form,
        readout_phase_error_upper_bound=readout_phase,
        full_interval_form_error_upper_bound=all_form,
        full_interval_phase_error_upper_bound=all_phase,
        endpoint_form_error_upper_bound=endpoint_form,
        endpoint_phase_error_upper_bound=endpoint_phase,
        child_form_mean_prediction_error_upper_bound=form_prediction,
        child_phase_mean_prediction_error_upper_bound=phase_prediction,
        release_form_error_upper_bound=release_form,
        release_phase_error_upper_bound=release_phase,
        reference_input_norm_upper_bound=input_norm,
        reference_input_work_upper_bound=construction.input_work_upper_bound,
        input_work_error_upper_bound=work_error,
        input_work_upper_bound=work_upper,
        work_budget_margin=work_margin,
        forced_child_positive_gap_margin=forced_positive,
        forced_child_acute_gap_margin=forced_acute,
        child_positive_gap_margin=child_positive,
        child_acute_gap_margin=child_acute,
        post_actuation_disruption_margin=disruption,
        parent_gap_at_recovery_upper_bound=recovery_gap,
        elapsed_scaled_duration_bounds=construction.elapsed_scaled_duration_bounds,
        elapsed_original_duration_bounds=construction.elapsed_original_duration_bounds,
        sufficient_margins=margins,
        source_certified=source_ok,
        conditional_realization_certified=conditional,
        endpoint_certified=source_ok and endpoint_ok,
        work_certified=source_ok and work_ok,
        child_identities_retained=source_ok and children_ok,
        disruption_certified=source_ok and disruption_ok,
        prediction_certified=source_ok and prediction_ok,
        recovery_certified=source_ok and recovery_ok,
        realization_certified=realization,
        release_certified=release,
        admitted=realization and release,
        status=("certified_collective_parent_actuation" if release else "unavailable"),
        unresolved_conditions=tuple(name for name, passed in decisions if not passed),
    )
