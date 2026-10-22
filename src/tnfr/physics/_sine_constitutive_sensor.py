"""Conditional scalar calibration of the actual-source constitutive comparison.

The owner is ORGANIZED_PATTERN_CONSTITUTIVE_FREEDOM.md, section
organized-pattern-calibrated-observation. A common static C2 sensor is a
supplied observation premise, not a changed state chart or an inferred law.
Each call rebuilds the raw comparison without raw-reading noise, then admits
errors in sensor units. No sensor, trajectory or reserved response is fitted
or evaluated. Calibration numbers do not authenticate a physical sensor.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from .._exact_time import exact_or_represented_real as _exact
from ._sine_constitutive_discrimination import (
    _assess_carried_constitutive_discrimination,
    _CarriedConstitutiveDiscrimination,
)


@dataclass(frozen=True)
class _CalibratedConstitutiveLaw:
    """One-sided mixed-reading bound under the common calibration class.

    For lambda zero the recorded mixed response is at most minus the margin;
    for lambda one it is at least the margin. No finite upper sensor gain
    is assumed, so these are not two-sided sensor-response intervals.
    Candidate arithmetic alone does not establish acquisition or calibration.
    """

    deformation: Q
    raw_signed_margin: Q
    gain_scaled_margin_lower_bound: Q | None
    recorded_signed_margin_lower_bound: Q | None
    opposite_sign_certified: bool


@dataclass(frozen=True)
class _CalibratedConstitutiveObservation:
    """Separate uniform mixed discrimination from fixed-common-map separation.

    Uniform mixed bounds hold over the entire declared calibration class.
    The gain-only full-record comparison pairs the very same fixed sensor
    across both laws; it cannot compare independently varying unknown maps.
    Both conclusions retain arbitrary allowed source and reading errors.
    Fields on failed premises retain conditional candidate arithmetic, not
    certified observation bounds.
    """

    raw_comparison: _CarriedConstitutiveDiscrimination
    calibration_radius: Q
    gain_lower_bound: Q
    curvature_upper_bound: Q
    per_reading_error_bound: Q
    branch_increment_upper_bound: Q
    baseline_absolute_upper_bound: Q
    raw_reading_radius_upper_bound: Q
    mixed_argument_radius_upper_bound: Q
    raw_domain_margin: Q
    mixed_domain_margin: Q
    curvature_error_upper_bound: Q
    mixed_reading_error_upper_bound: Q
    observations: tuple[_CalibratedConstitutiveLaw, _CalibratedConstitutiveLaw]
    raw_domain_admitted: bool
    mixed_domain_admitted: bool
    uniform_mixed_separation_lower_bound: Q | None
    uniform_record_distance_lower_bound: Q | None
    classification_threshold: Q | None
    uniform_separation_certified: bool
    opposite_signs_certified: bool
    fixed_common_map_record_distance_lower_bound: Q | None
    fixed_common_map_certified: bool
    status: str
    unresolved_conditions: tuple[str, ...]
    units: tuple[tuple[str, str], ...] = (
        ("calibration_radius", "form"),
        ("gain_lower_bound", "sensor_output/form"),
        ("curvature_upper_bound", "sensor_output/form^2"),
        ("per_reading_error_bound", "sensor_output"),
        ("classification_threshold", "sensor_output"),
    )
    scope: tuple[str, ...] = (
        "common_static_C2_sensor_on_declared_symmetric_form_interval",
        "independently_supplied_calibration_not_inferred_from_response",
        "absolute_reference_and_all_original_source_errors_retained",
        "fresh_own_law_handoff_and_raw_finite_response_with_zero_raw_noise",
        "paired_histories_share_actual_source_before_each_form_input",
        "raw_reading_domain_distinct_from_virtual_mixed_argument_domain",
        "four_arbitrary_sensor_errors_without_independence_or_cancellation",
        "unknown_common_sensor_offset_cancels_from_mixed_reading",
        "positive_gain_lower_bound_yields_only_one_sided_sensor_bounds",
        "uniform_mixed_threshold_not_necessarily_zero_or_sign_preserving",
        "gain_only_full_record_distance_requires_the_same_fixed_sensor",
        "failed_sufficient_bound_is_not_an_observational_collision",
        "no_sensor_callable_report_ingestion_or_physical_calibration_claim",
    )


def _assess_calibrated_constitutive_observation(
    *,
    form_error_bound,
    phase_error_bound,
    calibration_radius,
    gain_lower_bound,
    curvature_upper_bound,
    per_reading_error_bound,
) -> _CalibratedConstitutiveObservation:
    """Rebuild one fixed comparison from raw source and calibration bounds.

    Radius must be positive; remaining bounds must be nonnegative. A zero
    lower gain or an insufficient domain/margin gives explicit unavailability.
    Admitted exact scalars are retained even below the shared interval grid.
    The supplied gain and curvature bounds must hold on the whole declared
    domain; this calculation does not verify that premise for an actual h.
    """
    rx = _exact(form_error_bound, "form_error_bound")
    rt = _exact(phase_error_bound, "phase_error_bound")
    radius = _exact(calibration_radius, "calibration_radius")
    gain = _exact(gain_lower_bound, "gain_lower_bound")
    curvature = _exact(curvature_upper_bound, "curvature_upper_bound")
    error = _exact(per_reading_error_bound, "per_reading_error_bound")
    if min(rx, rt, gain, curvature, error) < 0:
        raise ValueError("source and calibration bounds must be nonnegative")
    if radius <= 0:
        raise ValueError("calibration_radius must be positive")

    raw = _assess_carried_constitutive_discrimination(
        form_error_bound=rx,
        phase_error_bound=rt,
        per_reading_error_bound=Q(0),
    )
    horizon, amplitude = raw.observation_duration, raw.amplitude
    eps = max(row.handoff.endpoint_radius for row in raw.observations)
    # Same-source branch differences start away from the mediator. Integrate
    # the full-field Lipschitz bound before taking its observed coordinate.
    increments = 4 * amplitude * horizon
    baseline = 2 * eps
    raw_radius = baseline + increments
    mixed_radius = baseline + 2 * increments
    raw_domain = radius >= raw_radius
    mixed_domain = radius >= mixed_radius
    transport_ok = (
        horizon > 0
        and 3 * horizon < Q(1, 2)
        and 3 / (1 - 3 * horizon) < 4
        and all(row.nominal_bound_admitted for row in raw.observations)
    )
    raw_ok = raw.status == "certified_opposite_carried_responses" and transport_ok
    curvature_error = curvature * increments**2
    mixed_error = 4 * error
    rows = []
    for row in raw.observations:
        lower, upper = row.actual_response_bounds
        signed = -upper if row.deformation == 0 else lower
        # Without an upper gain, a negative raw lower margin cannot be
        # multiplied by the gain lower bound to obtain a valid lower bound.
        scaled = gain * signed if signed >= 0 else None
        margin = scaled - curvature_error - mixed_error if scaled is not None else None
        rows.append(
            _CalibratedConstitutiveLaw(
                deformation=row.deformation,
                raw_signed_margin=signed,
                gain_scaled_margin_lower_bound=scaled,
                recorded_signed_margin_lower_bound=margin,
                opposite_sign_certified=(
                    raw_ok
                    and mixed_domain
                    and gain > 0
                    and margin is not None
                    and margin > 0
                ),
            )
        )
    b0, b1 = (row.recorded_signed_margin_lower_bound for row in rows)
    gap = b0 + b1 if b0 is not None and b1 is not None else None
    uniform = raw_ok and mixed_domain and gain > 0 and gap is not None and gap > 0
    threshold = (b1 - b0) / 2 if uniform else None
    # This different certificate compares each pair through the SAME h.
    raw_gap = raw.record_separation_lower_bound
    fixed_distance = gain * raw_gap / 4 - 2 * error if raw_gap >= 0 else None
    fixed = (
        raw_ok
        and raw_domain
        and gain > 0
        and fixed_distance is not None
        and fixed_distance > 0
    )
    reasons = []
    if not raw_ok:
        reasons.append("actual_source_and_response_not_certified")
    if gain <= 0:
        reasons.append("strictly_positive_sensor_gain_not_supplied")
    if not raw_domain:
        reasons.append("raw_reading_envelope_not_enclosed_by_calibrated_domain")
    if not mixed_domain:
        reasons.append("mixed_argument_envelope_not_enclosed_by_calibrated_domain")
    if gap is None or gap <= 0:
        reasons.append("uniform_mixed_separation_not_certified")
    if fixed_distance is None or fixed_distance <= 0:
        reasons.append("fixed_common_sensor_record_distance_not_certified")
    return _CalibratedConstitutiveObservation(
        raw_comparison=raw,
        calibration_radius=radius,
        gain_lower_bound=gain,
        curvature_upper_bound=curvature,
        per_reading_error_bound=error,
        branch_increment_upper_bound=increments,
        baseline_absolute_upper_bound=baseline,
        raw_reading_radius_upper_bound=raw_radius,
        mixed_argument_radius_upper_bound=mixed_radius,
        raw_domain_margin=radius - raw_radius,
        mixed_domain_margin=radius - mixed_radius,
        curvature_error_upper_bound=curvature_error,
        mixed_reading_error_upper_bound=mixed_error,
        observations=tuple(rows),
        raw_domain_admitted=raw_domain,
        mixed_domain_admitted=mixed_domain,
        uniform_mixed_separation_lower_bound=gap,
        uniform_record_distance_lower_bound=gap / 4 if gap is not None else None,
        classification_threshold=threshold,
        uniform_separation_certified=uniform,
        opposite_signs_certified=all(row.opposite_sign_certified for row in rows),
        fixed_common_map_record_distance_lower_bound=fixed_distance,
        fixed_common_map_certified=fixed,
        status=(
            "certified_calibrated_mixed_separation"
            if uniform
            else "fixed_common_sensor_separation_only" if fixed else "unavailable"
        ),
        unresolved_conditions=tuple(reasons),
    )
