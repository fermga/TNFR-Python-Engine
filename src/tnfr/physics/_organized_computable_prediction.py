"""One fixed, conditional collective forecast after finite observation.

The declaration owns preparation, branch, inputs and decision. This consumer
uses the shared full-sine field and prescribed-time metric Taylor kernel;
it neither reacquires the fine source nor evaluates a measured response.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q
from types import SimpleNamespace
from typing import Any, Callable

from ..dynamics.relational import RelationalExchangeModel
from ..mathematics._rational_interval import I, pi_interval
from ..mathematics._validated_metric import validated_metric_taylor_step
from ._organized_finite_observation import (
    _DELAY,
    _EPSILON,
    _PLATEAU_LEVELS,
    _PULSE_DURATION,
    _PULSE_MULTIPLIERS,
    _WORD_DURATION,
)
from ._sine_flow import _full_sine_field


_FORMATION = Q(200)
_HORIZON = Q(216)
_MAX_STEP = Q(1, 8)
_ORDER = 16
_MAX_STEPS = 1800
_GROWTH = Q(1, 3000)
_NUMERICAL_BUDGET = Q(1, 10**15)
_READING_ERROR = Q(1, 10**14)
_METRIC = tuple(tuple(Q(i == j) for j in range(10)) for i in range(10))


@dataclass(frozen=True)
class _ForecastStage:
    start: Q
    duration: Q
    name: str
    child: int | None = None
    base: Q = Q(0)
    increment: Q = Q(0)

    def command(self, time: Any) -> tuple[Any, ...]:
        """Polynomial branch, valid only on this declared stage's interval."""
        zero = time * 0
        if self.child is None:
            return (zero,) * 5
        r = (time - self.start) / self.duration
        level = self.base + self.increment * r * r * (3 - 2 * r)
        return tuple(level * (Q(int(c == self.child)) - Q(1, 5)) for c in range(5))


def _forecast_stages() -> tuple[_ForecastStage, ...]:
    stages = [_ForecastStage(Q(0), _FORMATION, "formation")]
    time = _FORMATION
    for child in range(5):
        for base, increment in zip(_PLATEAU_LEVELS, _PULSE_MULTIPLIERS):
            stages.append(
                _ForecastStage(time, _DELAY, "plateau", child, base * _EPSILON)
            )
            time += _DELAY
            stages.append(
                _ForecastStage(
                    time,
                    _PULSE_DURATION,
                    "ramp",
                    child,
                    base * _EPSILON,
                    increment * _EPSILON,
                )
            )
            time += _PULSE_DURATION
    if time != _FORMATION + _WORD_DURATION:
        raise ArithmeticError("stage construction differs from the declared word")
    stages.append(_ForecastStage(time, _HORIZON - time, "released"))
    return tuple(stages)


def _command_removed_field(
    stage: _ForecastStage, *, omit_word: bool
) -> tuple[Callable, Callable]:
    """Preserve both inherited degree-four rows in Y=U-z, Psi=Phi-H.

    The shared C5 field uses degree two. Halving *both* structural-time rows
    gives the degree-four carrier inherited from C9-by-C5, not a new clock.
    Command derivatives cancel analytically from the transformed equations.
    """
    model = RelationalExchangeModel(
        storage_scale=1,
        epi_weight=Q(1023, 1024),
        phase_weight=Q(1, 1024),
        phase_domain="regular",
    )
    geometry = SimpleNamespace(
        nodes=tuple(range(5)), edges=tuple((c, (c + 1) % 5) for c in range(5))
    )
    base_flow, _ = _full_sine_field(model, geometry, (2,) * 5)

    def flow(time: Any, state: tuple[Any, ...]) -> tuple[Any, ...]:
        command = (time * 0,) * 5 if omit_word else stage.command(time)
        shifted = tuple(state[:5]) + tuple(
            value + offset for value, offset in zip(state[5:], command)
        )
        return tuple(value / 2 for value in base_flow(shifted))

    def domain(time: I, _state: tuple[Any, ...]) -> tuple[Q, ...]:
        if time.lo < I(stage.start).lo or time.hi > I(stage.start + stage.duration).hi:
            return (Q(-1),)
        # Smooth polynomial extension on each closed stage; switches are split.
        return (Q(1),)

    return flow, domain


@dataclass(frozen=True)
class _ForecastStep:
    """Compact witness; the previous endpoint is this step's initial ball."""

    time: Q
    duration: Q
    tube: tuple[I, ...]
    picard_margin: Q
    polynomial: tuple[I, ...]
    remainder: tuple[I, ...]
    growth: Q
    local_error: Q
    endpoint_center: tuple[Q, ...]
    endpoint_radius: Q


@dataclass(frozen=True)
class _ForecastLeg:
    name: str
    start: Q
    end: Q
    initial_center: tuple[Q, ...]
    initial_radius: Q
    steps: tuple[_ForecastStep, ...]
    endpoint_center: tuple[Q, ...]
    endpoint_radius: Q
    admitted: bool
    reason: str | None
    failed_tube: tuple[I, ...] | None


def _forecast_leg(
    name: str,
    stages: tuple[_ForecastStage, ...],
    center: tuple[Q, ...],
    radius: Q,
    *,
    omit_word: bool,
    prefix_steps: int = 0,
) -> _ForecastLeg:
    initial_center, initial_radius = center, radius
    steps: list[_ForecastStep] = []
    elapsed = stages[0].start
    reason = None
    failed_tube = None
    for stage in stages:
        if elapsed != stage.start:
            raise ArithmeticError("forecast stages must retain continuous time")
        flow, domain = _command_removed_field(stage, omit_word=omit_word)
        endpoint = stage.start + stage.duration
        while elapsed < endpoint:
            if prefix_steps + len(steps) >= _MAX_STEPS:
                reason = "declared_step_budget_exhausted"
                break
            duration = min(_MAX_STEP, endpoint - elapsed)
            step, failed_tube, reason = validated_metric_taylor_step(
                center,
                radius,
                _METRIC,
                duration,
                flow,
                domain,
                growth_rate=_GROWTH,
                order=_ORDER,
                time=elapsed,
                time_dependent=True,
            )
            if step is None:
                break
            steps.append(
                _ForecastStep(
                    elapsed,
                    duration,
                    step.tube,
                    step.picard_interior_margin,
                    step.center_polynomial_bounds,
                    step.local_remainder_bounds,
                    step.growth_factor_upper_bound,
                    step.local_metric_error_upper_bound,
                    step.endpoint_center,
                    step.endpoint_radius,
                )
            )
            center, radius = step.endpoint_center, step.endpoint_radius
            elapsed += duration
            if radius > _NUMERICAL_BUDGET:
                reason = "declared_numerical_radius_exceeded"
                break
        if reason is not None:
            break
    return _ForecastLeg(
        name,
        stages[0].start,
        elapsed,
        initial_center,
        initial_radius,
        tuple(steps),
        center,
        radius,
        reason is None,
        reason,
        failed_tube,
    )


def _source_and_suffix_bounds() -> tuple[Q, tuple[Q, ...], Q]:
    """Rebuild the conditional error and identity extension from primitives."""
    g = _GROWTH
    z = 2 * (Q(6, 10**12) + g * Q(3, 10**9))
    source = _HORIZON * g * z**2 / (9 * (1 - (g * _HORIZON) ** 2))
    duration = _HORIZON - _FORMATION
    form = Q(1, 100) + g * duration
    phase = 2 * g * Q(1, 100) + 2 * g**2 * duration
    pi, residual = pi_interval(), Q(1, 10**6)
    margins = (
        2 * pi.lo / 9 - 2 * residual - 2 * phase,
        pi.lo / 2 - 2 * pi.hi / 9 - 2 * residual - 2 * phase,
        pi.lo / 2 - Q(7, 5) - _EPSILON - 2 * residual - 2 * phase,
        pi.lo / 2 - 2 * residual - 4 * phase,
    )
    loss = 720 * form**2 * (_HORIZON - _FORMATION - _WORD_DURATION)
    return source, margins, loss


def _endpoint_bands(
    center: tuple[Q, ...], radius: Q, allowance: Q = Q(0)
) -> tuple[I, ...]:
    return tuple(
        I(value - radius - allowance, value + radius + allowance)
        for value in center[:5]
    )


def _interval_separation(left: tuple[I, ...], right: tuple[I, ...]) -> Q:
    return max((max(Q(0), a.lo - b.hi, b.lo - a.hi) for a, b in zip(left, right)))


@dataclass(frozen=True)
class _ComputablePrediction:
    formation: _ForecastLeg
    observed: _ForecastLeg | None
    omitted: _ForecastLeg | None
    stages: tuple[_ForecastStage, ...]
    source_error: Q
    numerical_budget: Q
    reading_error: Q
    identity_margins: tuple[Q, ...]
    unforced_tail_loss_upper: Q
    predicted_form_bounds: tuple[I, ...] | None
    omitted_form_bounds: tuple[I, ...] | None
    predicted_record_bounds: tuple[I, ...] | None
    omitted_record_bounds: tuple[I, ...] | None
    numerical_interval_separation: Q | None
    required_separation: Q
    numerical_certified: bool
    admitted: bool
    reason: str | None
    input_error: Q = Q(0)
    clock_error: Q = Q(0)
    unforced_tail_supplied_work: Q = Q(0)
    branch: str = "outer_one"
    clock: str = "tau=(1023/1024)*t"
    coordinate_order: str = "Y[0:5],Psi[0:5]; exact command removed"
    phase_domain: str = "full_circle_with_declared_continuous_lifts"
    numerical_method: str = "prescribed_time_identity_metric_Taylor16_dyadic128"
    growth_upper: Q = _GROWTH
    maximum_step: Q = _MAX_STEP
    maximum_steps_per_branch: int = _MAX_STEPS
    source_scope: str = "conditional_on_original_independent_preparation_and_zero_sums"


def _assess_organized_computable_prediction() -> _ComputablePrediction:
    """Evaluate the frozen forecast once; no inputs, retries or source replay."""
    stages = _forecast_stages()
    center = tuple(Q(4092 * (c - 2)) for c in range(5)) + (Q(0),) * 5
    source, margins, loss = _source_and_suffix_bounds()
    formation = _forecast_leg(
        "common_formation", stages[:1], center, Q(0), omit_word=True
    )
    observed = omitted = None
    if formation.admitted:
        observed = _forecast_leg(
            "finite_observation",
            stages[1:],
            formation.endpoint_center,
            formation.endpoint_radius,
            omit_word=False,
            prefix_steps=len(formation.steps),
        )
        omitted = _forecast_leg(
            "omitted_observation",
            stages[1:],
            formation.endpoint_center,
            formation.endpoint_radius,
            omit_word=True,
            prefix_steps=len(formation.steps),
        )
    numeric = bool(
        formation.admitted
        and observed is not None
        and omitted is not None
        and observed.admitted
        and omitted.admitted
    )
    left = right = records_left = records_right = separation = None
    reason = formation.reason
    required = 2 * (_READING_ERROR + source)
    if numeric and observed is not None and omitted is not None:
        left = _endpoint_bands(observed.endpoint_center, observed.endpoint_radius)
        right = _endpoint_bands(omitted.endpoint_center, omitted.endpoint_radius)
        separation = _interval_separation(left, right)
        records_left = _endpoint_bands(
            observed.endpoint_center, observed.endpoint_radius, source + _READING_ERROR
        )
        records_right = _endpoint_bands(
            omitted.endpoint_center, omitted.endpoint_radius, source + _READING_ERROR
        )
        if separation <= required:
            reason = "endpoint_records_not_separated_by_declared_rule"
    elif observed is not None and omitted is not None:
        reason = observed.reason or omitted.reason
    admitted = bool(
        numeric
        and min(margins) > 0
        and separation is not None
        and separation > required
    )
    return _ComputablePrediction(
        formation,
        observed,
        omitted,
        stages,
        source,
        _NUMERICAL_BUDGET,
        _READING_ERROR,
        margins,
        loss,
        left,
        right,
        records_left,
        records_right,
        separation,
        required,
        bool(numeric),
        admitted,
        reason,
    )
