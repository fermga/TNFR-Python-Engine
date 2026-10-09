"""Two-source complete four-history readouts under one fixed work policy.

The full54 producer runs in declared source order. All primitive sources are
admitted first; every consumed observation is rebuilt from retained evidence.
This installs no acquisition, class label, prediction or sensor-error premise.
"""

from dataclasses import dataclass
from fractions import Fraction as Q

from ..mathematics._rational_interval import I
from ._sine_class_readout_evidence import (
    _ClassReadoutEvidence,
    _reconstruct_class_readout,
)
from .relational_observations import _ordered
from .relational_sine_class_readout import (
    _MAX_STEPS,
    SineClassFourHistoryReadout,
    _admit_class_readout_inputs,
    bound_sine_class_four_history_readout,
)

__all__ = ("SineClassComparisonReadout", "bound_sine_class_comparison_readout")


@dataclass(frozen=True)
class SineClassComparisonReadout:
    """First-minus-second observation with honest partial child provenance.

    Each source has four matched mathematical histories. Sources are independent
    across this comparison. Their primitive Cartesian covers establish neither
    acquisition nor a cycle class, and may instead cover declared nominal
    references. Actual-source transfer then requires a separate proof.
    """

    initial_form_bounds: tuple[tuple[I, ...], ...]
    initial_phase_bounds: tuple[tuple[I, ...], ...]
    first_probe_amplitude: Q
    second_probe_amplitude: Q
    delay: Q
    total_duration: Q
    time_step: Q
    order: int
    max_steps: int
    readout_reports: tuple[SineClassFourHistoryReadout | None, ...]
    reconstructed_observations: tuple[_ClassReadoutEvidence | None, ...]
    planned_step_count: int
    attempted_step_count: int
    completed_step_count: int
    completed_source_count: int
    failed_source_index: int | None
    unattempted_source_indices: tuple[int, ...]
    class_mixed_readout_bounds: tuple[I, I] | None
    contrast_readout_bounds: I | None
    raw_endpoint_contrast_bounds: I | None
    status: str
    unavailable_reasons: tuple[str, ...]
    source_order: tuple[int, int] = (0, 1)
    contrast_coefficients: tuple[int, int] = (1, -1)
    clock: str = "tau=e*t; e=1023/1024"
    method: str = "sequential_two_source_full54_readout_reconstruction_v1"
    scope: tuple[str, ...] = (
        "both_primitive_source_covers_are_admitted_before_either_execution",
        "same_complete_law_events_clock_and_fixed_numerical_policy_for_both_sources",
        "one_shared_total_attempt_cap_and_at_most4096_attempts_per_source",
        "first_incomplete_child_stops_all_later_execution_without_retry",
        "both_complete_children_required_before_cross_source_contrast_is_available",
        "consumed_observations_are_rebuilt_from_complete_carried_Taylor_evidence",
        "retained_derivative_and_Picard_generation_remain_execution_premises",
        "no_acquisition_cycle_class_prediction_or_sensor_input_is_consumed",
        "nominal_reference_to_actual_source_transfer_is_external_to_this_producer",
        "width_includes_source_range_wrapping_arithmetic_and_truncation_not_sensor_error",
        "marginal_intervals_do_not_assert_joint_realizability_of_all_corners",
    )

    @property
    def admitted(self):
        return self.status == "admitted"

    def to_dict(self):
        from ..sdk.relational_reports import _project

        return {
            "schema": "tnfr.sine-class-comparison-readout.v1",
            "report": _project(self),
        }


def bound_sine_class_comparison_readout(
    *,
    initial_form_bounds,
    initial_phase_bounds,
    first_probe_amplitude,
    second_probe_amplitude,
    delay,
    total_duration,
    time_step,
    order,
    max_steps,
) -> SineClassComparisonReadout:
    """Admit two ordered27-node sources and run the existing producer twice.

    Each source channel is a pair of ordered27-row endpoint-pair covers.
    Shared scalar domains match the single-source producer; max_steps is an
    ordinary integer1..8192 across both children, including failed attempts.
    Each child receives min(4096,remaining). Exhaustion before a child applies
    no event there. Validation errors raise before any execution. Numerical
    failure returns retained partial children and unavailable contrast fields.
    """
    forms = _ordered(initial_form_bounds, "initial_form_bounds", limit=3)
    phases = _ordered(initial_phase_bounds, "initial_phase_bounds", limit=3)
    if len(forms) != 2 or len(phases) != 2:
        raise ValueError("each channel requires exactly two ordered source covers")
    if type(max_steps) is not int or not 1 <= max_steps <= 2 * _MAX_STEPS:
        raise ValueError("max_steps must be an ordinary integer in 1..8192")
    shared = dict(
        first_probe_amplitude=first_probe_amplitude,
        second_probe_amplitude=second_probe_amplitude,
        delay=delay,
        total_duration=total_duration,
        time_step=time_step,
        order=order,
        max_steps=min(_MAX_STEPS, max_steps),
    )
    admitted = tuple(
        _admit_class_readout_inputs(
            initial_form_bounds=form, initial_phase_bounds=phase, **shared
        )
        for form, phase in zip(forms, phases)
    )
    _, _, a, b, s, total, h, order, _ = admitted[0]
    planned = 2 * (2 * -(-s // h) + 4 * -(-(total - s) // h))
    reports, rebuilt = [None, None], [None, None]
    attempted = completed = source_count = 0
    failure = None
    reasons = []
    for index, values in enumerate(admitted):
        remaining = max_steps - attempted
        if remaining == 0:
            failure = index
            reasons.append(f"source_{index}: total_step_budget_exhausted_before_source")
            break
        cap = min(_MAX_STEPS, remaining)
        form, phase = values[:2]
        report = bound_sine_class_four_history_readout(
            initial_form_bounds=tuple((x.lo, x.hi) for x in form),
            initial_phase_bounds=tuple((x.lo, x.hi) for x in phase),
            first_probe_amplitude=a,
            second_probe_amplitude=b,
            delay=s,
            total_duration=total,
            time_step=h,
            order=order,
            max_steps=cap,
        )
        reports[index] = report
        evidence = _reconstruct_class_readout(report, (*values[:-1], cap))
        rebuilt[index] = evidence
        attempted += evidence.attempted_step_count
        completed += evidence.completed_step_count
        if not evidence.complete:
            failure = index
            reasons.append(f"source_{index}: incomplete_four_history_readout")
            break
        source_count += 1
    bands = tuple(item.mixed_bounds for item in rebuilt) if source_count == 2 else None
    contrast = bands[0] - bands[1] if bands is not None else None
    raw = (
        rebuilt[0].raw_mixed_bounds - rebuilt[1].raw_mixed_bounds
        if bands is not None
        else None
    )
    return SineClassComparisonReadout(
        initial_form_bounds=tuple(values[0] for values in admitted),
        initial_phase_bounds=tuple(values[1] for values in admitted),
        first_probe_amplitude=a,
        second_probe_amplitude=b,
        delay=s,
        total_duration=total,
        time_step=h,
        order=order,
        max_steps=max_steps,
        readout_reports=tuple(reports),
        reconstructed_observations=tuple(rebuilt),
        planned_step_count=planned,
        attempted_step_count=attempted,
        completed_step_count=completed,
        completed_source_count=source_count,
        failed_source_index=failure,
        unattempted_source_indices=tuple(
            i for i, report in enumerate(reports) if report is None
        ),
        class_mixed_readout_bounds=bands,
        contrast_readout_bounds=contrast,
        raw_endpoint_contrast_bounds=raw,
        status="admitted" if source_count == 2 else "unavailable",
        unavailable_reasons=tuple(reasons),
    )
