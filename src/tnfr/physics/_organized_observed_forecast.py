"""One candidate's carried, conditional forecast from joint form readings.

Proposal generation and evidence extraction are separate owners. This
consumer validates a proposed exact initial state with the shared complete
carrier field and interval integrator, then combines its whole-record
residual with a rebuilt observation gain. It installs no network law.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q
from pathlib import Path
from typing import Any

from .._exact_time import exact_or_represented_real
from ..mathematics._exact_linear_algebra import _ordered
from ..mathematics._prediction_transfer import (
    CandidateForecastError,
    candidate_forecast_error,
)
from ..mathematics._rational_interval import I
from ._organized_computable_prediction import (
    _forecast_leg,
    _forecast_stages,
    _ForecastLeg,
)
from ._organized_observed_prediction import _observation_budget

_READING_ERROR = Q(1, 10**14)
_OUTPUT_POLICY = Q(1, 10**13)
_INITIAL_FORM_BOUND = Q(1, 100)


def _sample_times() -> tuple[Q, ...]:
    return tuple(
        value
        for stage in _forecast_stages()[1:]
        if stage.name == "plateau"
        for value in (stage.start, stage.start + stage.duration)
    )


def _admit_record(times: Any, values: Any) -> tuple[tuple[Q, ...], tuple[Q, ...]]:
    normalized_times = tuple(
        exact_or_represented_real(value, "sample time")
        for value in _ordered(times, "sample times")
    )
    readings = tuple(
        exact_or_represented_real(value, "form reading")
        for value in _ordered(values, "form readings")
    )
    if normalized_times != _sample_times() or len(readings) != 150:
        raise ValueError(
            "record must retain all 30 exact times and 150 ordered readings"
        )
    return normalized_times, readings


def _admit_candidate(values: Any) -> tuple[Q, ...]:
    candidate = tuple(
        exact_or_represented_real(value, "candidate coordinate")
        for value in _ordered(values, "candidate state")
    )
    if len(candidate) != 10:
        raise ValueError("candidate requires five forms and five phase coordinates")
    if max(map(abs, candidate[:5])) > _INITIAL_FORM_BOUND:
        raise ValueError("candidate initial forms exceed the exact prior")
    if candidate[5] != 0:
        raise ValueError("candidate must declare the initial Phi_0=0 gauge")
    return candidate


@dataclass(frozen=True)
class _ValidatedObservedForecast:
    initial_candidate: tuple[Q, ...]
    times: tuple[Q, ...]
    readings: tuple[Q, ...]
    trajectory: _ForecastLeg
    error: CandidateForecastError | None
    predicted_true_form_bounds: tuple[I, ...] | None
    predicted_reading_bounds: tuple[I, ...] | None
    admitted: bool
    reason: str | None
    target_radius: Q = _OUTPUT_POLICY
    scope: str = "conditional_on_original_fine_class_and_rebuilt_gain_evidence"


def validate_observed_candidate(
    times: Any, readings: Any, candidate: Any, *, rebuilt_gain: Any
) -> _ValidatedObservedForecast:
    """Validate one exact proposal; a caller must establish the supplied gain.

    This is an internal model consumer. The assessment obtains rebuilt_gain
    from the fixed archive's primitive coefficients and dual residuals, never
    from a cached scalar verdict. Arbitrary externally supplied gains do not
    certify this model. No other candidate, restart or precision escalation
    follows a failure. A candidate's phase gauge is initial only; both full
    state rows are subsequently integrated without resetting any coordinate.
    """
    times, readings = _admit_record(times, readings)
    candidate = _admit_candidate(candidate)
    gain = exact_or_represented_real(rebuilt_gain, "rebuilt observation gain")
    if gain < 0:
        raise ValueError("rebuilt observation gain must be nonnegative")
    trajectory = _forecast_leg(
        "record_only_candidate",
        _forecast_stages()[1:],
        candidate,
        Q(0),
        omit_word=False,
    )
    error = true_bounds = reading_bounds = None
    reason = trajectory.reason
    if trajectory.admitted:
        # The samples all belong to this freshly integrated one-state chain.
        states = {
            trajectory.start: (trajectory.initial_center, trajectory.initial_radius)
        }
        for step in trajectory.steps:
            states[step.time + step.duration] = (
                step.endpoint_center,
                step.endpoint_radius,
            )
        sample_boxes = tuple(
            I(value - radius, value + radius)
            for time in times
            for center, radius in (states[time],)
            for value in center[:5]
        )
        future_boxes = tuple(
            I(value - trajectory.endpoint_radius, value + trajectory.endpoint_radius)
            for value in trajectory.endpoint_center[:5]
        )
        error = candidate_forecast_error(
            readings,
            sample_boxes,
            future_boxes,
            gain=gain,
            source_error=_observation_budget().source_error,
            reading_error=_READING_ERROR,
        )
        if error.maximum_true_output_radius <= _OUTPUT_POLICY:
            true_bounds = tuple(
                I(center - radius, center + radius)
                for center, radius in zip(
                    error.forecast_center, error.true_output_radii
                )
            )
            reading_bounds = tuple(
                I(center - radius - _READING_ERROR, center + radius + _READING_ERROR)
                for center, radius in zip(
                    error.forecast_center, error.true_output_radii
                )
            )
        else:
            reason = "candidate_forecast_radius_exceeds_declared_policy"
    admitted = true_bounds is not None
    return _ValidatedObservedForecast(
        candidate,
        times,
        readings,
        trajectory,
        error,
        true_bounds,
        reading_bounds,
        admitted,
        reason,
    )


@dataclass(frozen=True)
class _ObservedForecastAssessment:
    record: Any
    gain_provenance: Any
    rebuilt_gain: Q
    containing_class_source_error: Q
    proposal: Any
    forecast: _ValidatedObservedForecast | None
    admitted: bool
    reason: str | None
    evaluation_kind: str = "archived_synthetic_record_only_software_demonstration"


def _assess_organized_observed_forecast(
    repository: Path,
) -> _ObservedForecastAssessment:
    """Select and evaluate the single fixed record after protocol/source freeze."""
    from ..research._organized_forecast_inputs import (
        extract_archived_form_record,
        load_archived_prediction_gain,
    )
    from ._organized_forecast_proposal import propose_candidate

    gain = load_archived_prediction_gain(repository)
    record = extract_archived_form_record(repository)
    # This is the entire proposal input. Provenance, extraction errors, phase
    # history, preparation state and future values are not inference arguments.
    readings = tuple(value for row in record.values for value in row)
    proposal = propose_candidate(record.times, readings, gain.observation_matrix)
    forecast = None
    reason = proposal.reason
    if proposal.initial_candidate is not None:
        forecast = validate_observed_candidate(
            record.times,
            readings,
            proposal.initial_candidate,
            rebuilt_gain=gain.carrier_future_gain,
        )
        reason = forecast.reason
    return _ObservedForecastAssessment(
        record,
        gain.provenance,
        gain.carrier_future_gain,
        gain.source_error,
        proposal,
        forecast,
        forecast is not None and forecast.admitted,
        reason,
    )
