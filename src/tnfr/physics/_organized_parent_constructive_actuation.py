"""One prescribed finite-work realization of the acquired-parent phase map.

ORGANIZED_PARENT_CONSTRUCTIVE_ACTUATION.md owns the frozen cubic path and
source-inclusive proof. These detached calculations evaluate analytic bounds
and an explicitly supplied input recipe; they integrate no trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from .._exact_time import exact_or_represented_real as _exact
from ..mathematics._rational_interval import I, pi_interval
from ._organized_parent_acquisition import _OrganizedParentAcquisition
from ._organized_parent_restoring_response import (
    _EVENT,
    _assess_organized_parent_restoring_response,
    _OrganizedParentRestoringResponse,
    _parent_registration_event_work_bounds,
)
from ._repeated_organized_composition import (
    _bounded_sine_edge_fields,
    _derive_repeated_composition,
    _RepeatedCompositionGeometry,
    _vector,
)
from ._sine_preparation import _prepared_form_costs

_SLOW_DURATION, _WORK_BUDGET = Q(4), Q(3)
_CLOCK_FACTOR = Q(1023, 1024)
_INVERSE_DISPLACEMENT = tuple(
    Q(2 * (node // 9 % 2) - 1, 16)
    - Q((2 * (node // 18) - 1) * (2 * (node // 9 % 2) - 1), 32)
    for node in range(36)
)


@dataclass(frozen=True)
class _ParentActuationPath:
    """Analytic path/input enclosures for an admitted exact initial state.

    Phase offsets retain the carrier ``2*pi*(j-4)/9``. Gamma is computed
    from that prescribed initial-state recipe, never from measured response.
    Evaluating this recipe does not authenticate a prepared or acquired source.
    """

    form: tuple[Q, ...]
    phase_offsets: tuple[Q, ...]
    path_fraction: Q
    geometry: _RepeatedCompositionGeometry
    form_mean: Q
    initial_form_gradient: tuple[Q, ...]
    initial_velocity_coefficient_bounds: tuple[I, ...]
    form_bounds: tuple[I, ...]
    phase_offset_bounds: tuple[I, ...]
    form_tau_rate_bounds: tuple[I, ...]
    phase_tau_rate_bounds: tuple[I, ...]
    form_slow_rate_bounds: tuple[I, ...]
    phase_slow_rate_bounds: tuple[I, ...]
    form_original_rate_bounds: tuple[I, ...]
    phase_original_rate_bounds: tuple[I, ...]
    form_input_tau_bounds: tuple[I, ...]
    form_input_slow_bounds: tuple[I, ...]
    form_input_original_bounds: tuple[I, ...]
    path_form_gradient_bounds: tuple[I, ...]
    phase_current_bounds: tuple[I, ...]
    continuous_loss_bounds: I
    laplacian_inverse_displacement: tuple[Q, ...] = _INVERSE_DISPLACEMENT
    phase_displacement: tuple[Q, ...] = _EVENT
    slow_duration: Q = _SLOW_DURATION
    clock: str = "tau=(1023/1024)*t; s=gamma^2*tau/2; v=(s-4)/4"
    scope: tuple[str, ...] = (
        "all_initial_coordinates_and_path_fraction_admitted_before_arithmetic",
        "fixed_cubic_path_with_both_actual_endpoint_form_vectors",
        "common_form_mode_and_continuous_phase_lifts_retained",
        "explicit_initial_state_dependent_input_not_measured_response_reconstruction",
        "complete_matching_support_and_held_unit_capacities",
        "interval_evaluation_not_an_integrated_trajectory_or_source_certificate",
    )


def _evaluate_parent_actuation_path(*, form, phase_offsets, path_fraction):
    """Enclose the fixed four-slow-unit cubic recipe at ``0<=v<=1``.

    The complete initial state is supplied independently of its response. The
    immutable fixed displacement has a closed-form Laplacian inverse, so no
    solve, inversion, fitted controller or state feedback is required.
    """
    values = _vector(form, "form", 36)
    offsets = _vector(phase_offsets, "phase_offsets", 36)
    v = _exact(path_fraction, "path_fraction")
    if not 0 <= v <= 1:
        raise ValueError("path_fraction requires 0 <= path_fraction <= 1")
    geometry = _derive_repeated_composition(interface="matching")
    _, initial_gradient = _prepared_form_costs(
        form=values, weighted_edges=geometry.weighted_edges
    )
    gamma, h = geometry.gamma_bounds, _SLOW_DURATION
    mean = sum(values, Q(0)) / 36
    centered = tuple(value - mean for value in values)
    p, q = 3 * v**2 - 2 * v**3, v - 3 * v**2 + 2 * v**3
    dp, dq = 6 * v - 6 * v**2, 1 - 6 * v + 6 * v**2
    ddp, ddq = 6 - 12 * v, -6 + 12 * v
    velocity = tuple(h * value / (2 * gamma) for value in initial_gradient)
    path_form = tuple(
        mean + 2 * gamma / h * inverse * dp + value * dq
        for inverse, value in zip(_INVERSE_DISPLACEMENT, centered)
    )
    path_phase = tuple(
        origin + change * p + slope * q
        for origin, change, slope in zip(offsets, _EVENT, velocity)
    )
    form_v = tuple(
        2 * gamma / h * inverse * ddp + value * ddq
        for inverse, value in zip(_INVERSE_DISPLACEMENT, centered)
    )
    phase_v = tuple(change * dp + slope * dq for change, slope in zip(_EVENT, velocity))
    gradient = tuple(
        2 * gamma / h * change * dp + value * dq
        for change, value in zip(_EVENT, initial_gradient)
    )
    current, _, _ = _bounded_sine_edge_fields(
        phase_offsets=path_phase,
        weighted_edges=geometry.weighted_edges,
        alpha_bounds=geometry.alpha_bounds,
    )
    tau_rate = gamma**2 / (2 * h)
    dx, dt = tuple(tau_rate * value for value in form_v), tuple(
        tau_rate * value for value in phase_v
    )
    forcing = tuple(
        rate + (laplacian - gamma * sine) / 4
        for rate, laplacian, sine in zip(dx, gradient, current)
    )
    norm_squared = sum((change**2 for change in _EVENT), Q(0))
    # Equivalent to integrating the squared polynomial phase velocity;
    # this completed-square expression preserves nonnegative loss bounds.
    loss = (
        2
        / h
        * (
            norm_squared
            + sum(
                ((change - slope) ** 2 for change, slope in zip(_EVENT, velocity)), I(0)
            )
            / 5
        )
    )
    return _ParentActuationPath(
        form=values,
        phase_offsets=offsets,
        path_fraction=v,
        geometry=geometry,
        form_mean=mean,
        initial_form_gradient=initial_gradient,
        initial_velocity_coefficient_bounds=velocity,
        form_bounds=path_form,
        phase_offset_bounds=path_phase,
        form_tau_rate_bounds=dx,
        phase_tau_rate_bounds=dt,
        form_slow_rate_bounds=tuple(value / h for value in form_v),
        phase_slow_rate_bounds=tuple(value / h for value in phase_v),
        form_original_rate_bounds=tuple(_CLOCK_FACTOR * value for value in dx),
        phase_original_rate_bounds=tuple(_CLOCK_FACTOR * value for value in dt),
        form_input_tau_bounds=forcing,
        form_input_slow_bounds=tuple(2 * value / gamma**2 for value in forcing),
        form_input_original_bounds=tuple(_CLOCK_FACTOR * value for value in forcing),
        path_form_gradient_bounds=gradient,
        phase_current_bounds=current,
        continuous_loss_bounds=loss,
    )


@dataclass(frozen=True)
class _OrganizedParentConstructiveActuation:
    """Source-inclusive path/work decision and separate autonomous release."""

    form_error_bounds: tuple[Q, ...]
    phase_error_bounds: tuple[Q, ...]
    restoring_response: _OrganizedParentRestoringResponse
    acquisition: _OrganizedParentAcquisition
    initial_velocity_coefficient_norm_upper_bound: Q
    full_path_form_error_upper_bound: Q
    full_path_phase_error_upper_bound: Q
    child_positive_gap_margin: Q
    child_acute_gap_margin: Q
    event_nominal_work_bounds: I
    event_work_error_upper_bound: Q
    endpoint_storage_change_bounds: I
    continuous_loss_lower_bound: Q
    continuous_loss_upper_bound: Q
    input_work_lower_bound: Q
    input_work_upper_bound: Q
    work_budget_margin: Q
    elapsed_scaled_duration_bounds: I
    elapsed_original_duration_bounds: I
    sufficient_margins: tuple[tuple[str, Q], ...]
    source_certified: bool
    conditional_realization_certified: bool
    child_identities_retained: bool
    work_certified: bool
    realization_certified: bool
    release_certified: bool
    admitted: bool
    status: str
    unresolved_conditions: tuple[str, ...]
    phase_displacement: tuple[Q, ...] = _EVENT
    endpoint_form_change: tuple[Q, ...] = (Q(0),) * 36
    source_slow_time: Q = Q(4)
    slow_duration: Q = _SLOW_DURATION
    release_slow_time: Q = Q(8)
    readout_slow_time: Q = Q(9)
    recovery_slow_time: Q = Q(12)
    work_budget: Q = _WORK_BUDGET
    protocol_revision: str = "2f0eab07"
    clock: str = "tau=(1023/1024)*t; s=gamma^2*tau/2; v=(s-4)/4"
    input_regularity: str = (
        "analytic on the forced interval; finite switching at endpoints"
    )
    scope: tuple[str, ...] = (
        "all_eight_original_source_allowances_readmitted_before_fresh_restoring_proof",
        "actual_acquired_form_phase_and_velocity_used_without_reset",
        "all_initial_fine_directions_retained_along_the_prescribed_cubic_path",
        "exact_target_endpoint_includes_every_form_and_declared_phase_lift",
        "mean_form_retained_and_complete_phase_row_satisfied",
        "explicit_supplied_initial_state_dependent_form_input_on_all_nodes",
        "whole_path_child_windings_certified_without_parent_registration_requirement",
        "signed_net_work_includes_endpoint_storage_and_integrated_continuous_loss",
        "work_bound_does_not_bound_positive_only_supply_amplitude_or_bandwidth",
        "path_admission_independent_of_future_unforced_response_verdicts",
        "release_without_state_jump_requires_exact_endpoint_and_same_autonomous_law",
        "release_readout_and_recovery_transferred_by_elapsed_time_translation_only",
        "no_partial_observation_controller_autonomous_selector_or_physical_identification",
        "no_trajectory_duration_scan_numerical_inverse_or_claim_of_optimal_work",
    )


def _assess_organized_parent_constructive_actuation(
    *, form_error_bounds, phase_error_bounds
) -> _OrganizedParentConstructiveActuation:
    """Rebuild the source and certify the fixed cubic path, work and release.

    One fresh restoring calculation rebuilds its original acquisition; no
    supplied report is consumed. Future restoring verdicts control only the
    separate release decision, never source, forced-path or work admission.
    """
    rx = _vector(form_error_bounds, "form_error_bounds", 4)
    rt = _vector(phase_error_bounds, "phase_error_bounds", 4)
    if min(*rx, *rt) < 0:
        raise ValueError("original preparation errors must be nonnegative")
    restoring = _assess_organized_parent_restoring_response(
        form_error_bounds=rx, phase_error_bounds=rt
    )
    acquisition = restoring.acquisition
    geometry = acquisition.geometry
    gamma, h = geometry.gamma_bounds, _SLOW_DURATION
    form_error = acquisition.full_form_error_upper_bound
    phase_error = acquisition.full_phase_error_upper_bound
    velocity_norm = (
        3
        * h
        * (1 + 2 * gamma.hi**2)
        * (
            acquisition.nominal_inner_gap_at_entry_bounds.hi
            + acquisition.nominal_outer_gap_at_entry_bounds.hi
        )
        + 4 * h * form_error / gamma.lo
    )
    path_phase_error = phase_error + 2 * h * form_error / (5 * gamma.lo)
    positive = geometry.alpha_bounds.lo - 2 * path_phase_error
    acute = pi_interval().lo / 2 - geometry.alpha_bounds.hi - 2 * path_phase_error
    nominal_work, work_error, endpoint_work = _parent_registration_event_work_bounds(
        inner_gap_bounds=acquisition.nominal_inner_gap_at_entry_bounds,
        outer_gap_bounds=acquisition.nominal_outer_gap_at_entry_bounds,
        phase_error_bound=phase_error,
    )
    norm_squared = sum((change**2 for change in _EVENT), Q(0))
    loss_lower = 2 * norm_squared / h
    # ||k||=sqrt(9/8)<9/8; all remaining quantities are exact rational bounds.
    loss_upper = (
        2
        / h
        * (
            Q(6, 5) * norm_squared
            + Q(2, 5) * Q(9, 8) * velocity_norm
            + velocity_norm**2 / 5
        )
    )
    input_lower = endpoint_work.lo + loss_lower
    input_upper = endpoint_work.hi + loss_upper
    work_margin = _WORK_BUDGET - input_upper
    source_conditions = (
        ("original_children_not_certified", acquisition.sources_certified),
        ("original_parent_entry_not_certified", acquisition.entry_certified),
        (
            "original_child_identities_not_certified",
            acquisition.child_identities_retained,
        ),
    )
    source_ok = all(passed for _, passed in source_conditions)
    child_ok, work_ok = min(positive, acute) > 0, work_margin >= 0
    conditional = child_ok and work_ok
    realization = source_ok and conditional
    release_conditions = (
        ("release_child_identities_not_certified", restoring.child_identities_retained),
        ("release_recovery_not_certified", restoring.recovery_certified),
        ("release_prediction_not_certified", restoring.prediction_certified),
    )
    release = realization and all(passed for _, passed in release_conditions)
    unresolved = tuple(
        name for name, passed in source_conditions + release_conditions if not passed
    )
    if not child_ok:
        unresolved += ("forced_path_child_winding_not_certified",)
    if not work_ok:
        unresolved += ("forced_path_input_work_budget_not_certified",)
    tau = 2 * h / gamma**2
    return _OrganizedParentConstructiveActuation(
        form_error_bounds=rx,
        phase_error_bounds=rt,
        restoring_response=restoring,
        acquisition=acquisition,
        initial_velocity_coefficient_norm_upper_bound=velocity_norm,
        full_path_form_error_upper_bound=form_error,
        full_path_phase_error_upper_bound=path_phase_error,
        child_positive_gap_margin=positive,
        child_acute_gap_margin=acute,
        event_nominal_work_bounds=nominal_work,
        event_work_error_upper_bound=work_error,
        endpoint_storage_change_bounds=endpoint_work,
        continuous_loss_lower_bound=loss_lower,
        continuous_loss_upper_bound=loss_upper,
        input_work_lower_bound=input_lower,
        input_work_upper_bound=input_upper,
        work_budget_margin=work_margin,
        elapsed_scaled_duration_bounds=tau,
        elapsed_original_duration_bounds=tau / _CLOCK_FACTOR,
        sufficient_margins=(
            ("child_positive_gap", positive),
            ("child_acute_gap", acute),
            ("input_work_budget", work_margin),
        ),
        source_certified=source_ok,
        conditional_realization_certified=conditional,
        child_identities_retained=source_ok and child_ok,
        work_certified=source_ok and work_ok,
        realization_certified=realization,
        release_certified=release,
        admitted=realization and release,
        status=(
            "certified_constructive_parent_actuation"
            if realization and release
            else "unavailable"
        ),
        unresolved_conditions=unresolved,
    )
