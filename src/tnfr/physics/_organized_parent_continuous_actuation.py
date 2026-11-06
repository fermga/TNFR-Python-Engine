"""Necessary work for continuous form realization of the fixed parent phase map.

ORGANIZED_PARENT_CONTINUOUS_ACTUATION.md owns the fixed source, endpoint,
input class and clock. The full-state balance excludes every admitted input
when its necessary work exceeds the budget; no forcing path is constructed.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from ..mathematics._rational_interval import I
from ._organized_parent_acquisition import (
    _assess_organized_parent_acquisition,
    _OrganizedParentAcquisition,
)
from ._organized_parent_restoring_response import (
    _EVENT,
    _parent_registration_event_work_bounds,
)
from ._repeated_organized_composition import _vector

_SLOW_DURATION = Q(1)
_WORK_BUDGET = Q(3)


@dataclass(frozen=True)
class _OrganizedParentContinuousActuation:
    """Conditional work bound and its separate source-inclusive obstruction.

    The target is the actual hybrid-map endpoint of the acquired state, not
    its later unforced continuation. An unavailable obstruction is neither a
    realization certificate nor a failure of the previously admitted jump.
    """

    form_error_bounds: tuple[Q, ...]
    phase_error_bounds: tuple[Q, ...]
    acquisition: _OrganizedParentAcquisition
    phase_displacement: tuple[Q, ...]
    phase_displacement_norm_squared: Q
    phase_displacement_sum: Q
    event_nominal_work_bounds: I
    event_work_error_upper_bound: Q
    event_work_residual_bounds: tuple[Q, Q]
    endpoint_storage_change_bounds: I
    endpoint_storage_change_lower_bound: Q
    necessary_continuous_loss_lower_bound: Q
    input_work_lower_bound: Q
    work_budget_deficit: Q
    elapsed_scaled_duration_bounds: I
    elapsed_original_duration_bounds: I
    sufficient_margins: tuple[tuple[str, Q], ...]
    source_certified: bool
    conditional_obstruction_certified: bool
    obstruction_certified: bool
    status: str
    unresolved_conditions: tuple[str, ...]
    endpoint_form_change: tuple[Q, ...] = (Q(0),) * 36
    source_slow_time: Q = Q(4)
    slow_duration: Q = _SLOW_DURATION
    work_budget: Q = _WORK_BUDGET
    protocol_revision: str = "be66de2d"
    clock: str = "tau=(1023/1024)*t; s=gamma^2*tau/2; elapsed h=1"
    input_regularity: str = "integrable Gamma_tau; absolutely continuous form"
    input_power: str = "(L*x) dot Gamma_tau in the tau clock"
    input_clock_transforms: tuple[str, ...] = (
        "Gamma_t=(1023/1024)*Gamma_tau",
        "Gamma_s=2*Gamma_tau/gamma^2",
        "input_power_times_clock_increment_is_invariant",
    )
    scope: tuple[str, ...] = (
        "all_original_source_primitives_readmitted_before_fresh_parent_acquisition",
        "actual_acquired_forms_and_lifted_phases_are_not_reset_to_a_nominal_state",
        "target_returns_every_form_and_applies_only_the_fixed_lifted_phase_displacement",
        "fixed_matching_support_unit_capacities_and_unchanged_phase_row",
        "all_integrable_form_inputs_allowed_without_amplitude_or_bandwidth_limits",
        "no_impulses_intermediate_jumps_or_phase_forcing",
        "signed_input_work_includes_actual_endpoint_storage_change_and_continuous_loss",
        "Cauchy_Schwarz_applies_to_every_admitted_full_fine_forcing_path",
        "no_child_retention_assumption_is_needed_to_exclude_all_paths",
        "source_admission_does_not_consume_future_unforced_response_or_recovery_flags",
        "strict_budget_deficit_certifies_obstruction_not_a_failed_chosen_controller",
        "equality_or_failed_sufficient_bounds_remain_unavailable_not_realizable",
        "other_durations_budgets_channels_or_coarser_endpoint_observations_are_separate_questions",
        "no_trajectory_controller_graph_mutation_autonomous_event_or_physical_units",
    )


def _assess_organized_parent_continuous_actuation(
    *, form_error_bounds, phase_error_bounds
) -> _OrganizedParentContinuousActuation:
    """Test the fixed one-slow-unit, three-work-unit endpoint realization.

    All eight finite nonnegative source allowances precede source construction.
    No report, forcing, altered duration, work budget or endpoint is accepted.
    The exact phase displacement determines an unavoidable loss allowance;
    it is added to the separately rebuilt actual signed endpoint work.
    """
    rx = _vector(form_error_bounds, "form_error_bounds", 4)
    rt = _vector(phase_error_bounds, "phase_error_bounds", 4)
    if min(*rx, *rt) < 0:
        raise ValueError("original preparation errors must be nonnegative")
    acquisition = _assess_organized_parent_acquisition(
        form_error_bounds=rx, phase_error_bounds=rt
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
    nominal_work, work_error, endpoint_work = _parent_registration_event_work_bounds(
        inner_gap_bounds=acquisition.nominal_inner_gap_at_entry_bounds,
        outer_gap_bounds=acquisition.nominal_outer_gap_at_entry_bounds,
        phase_error_bound=acquisition.full_phase_error_upper_bound,
    )
    displacement = _EVENT
    norm_squared = sum((value**2 for value in displacement), Q(0))
    displacement_sum = sum(displacement, Q(0))
    # H_tau=-||Lx||^2/4+(Lx).Gamma_tau and theta_tau=gamma*Lx/4.
    # Cauchy--Schwarz gives loss >=4||delta||^2/(gamma^2*T_tau).
    # T_tau=2*h/gamma^2 cancels gamma exactly, without interval division.
    necessary_loss = 2 * norm_squared / _SLOW_DURATION
    input_work = endpoint_work.lo + necessary_loss
    deficit = input_work - _WORK_BUDGET
    conditional = deficit > 0
    obstruction = source_ok and conditional
    unresolved = tuple(name for name, passed in source_conditions if not passed)
    if not conditional:
        unresolved += ("strict_input_work_budget_exclusion_not_certified",)
    eta = acquisition.geometry.gamma_bounds**2
    tau = 2 * _SLOW_DURATION / eta
    return _OrganizedParentContinuousActuation(
        form_error_bounds=rx,
        phase_error_bounds=rt,
        acquisition=acquisition,
        phase_displacement=displacement,
        phase_displacement_norm_squared=norm_squared,
        phase_displacement_sum=displacement_sum,
        event_nominal_work_bounds=nominal_work,
        event_work_error_upper_bound=work_error,
        event_work_residual_bounds=(-work_error, work_error),
        endpoint_storage_change_bounds=endpoint_work,
        endpoint_storage_change_lower_bound=endpoint_work.lo,
        necessary_continuous_loss_lower_bound=necessary_loss,
        input_work_lower_bound=input_work,
        work_budget_deficit=deficit,
        elapsed_scaled_duration_bounds=tau,
        elapsed_original_duration_bounds=tau / Q(1023, 1024),
        sufficient_margins=(("strict_input_work_budget_exclusion", deficit),),
        source_certified=source_ok,
        conditional_obstruction_certified=conditional,
        obstruction_certified=obstruction,
        status=(
            "certified_continuous_form_actuation_obstruction"
            if obstruction
            else "unavailable"
        ),
        unresolved_conditions=unresolved,
    )
