"""Sampling admission and explicit research responses for one K3 intervention.

This research calculation bounds both named candidate laws over one supplied
box and parameter prior. The separate response API samples the admitted ideal
laws only when explicitly invoked. Neither installs a countermodel in the
engine or authenticates preparation, observation error or physical evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from ..mathematics._interval_taylor import MAX_ORDER, Jet, cos, sin, sinc
from ..mathematics._rational_interval import INTERVAL_METHOD, I, pi_interval
from ..mathematics._rational_interval import sin as interval_sin
from ..physics.relational_observations import (
    RelationalRateContrastBounds,
    RelationalRateSampleBounds,
    bound_relational_rate_contrast,
    bound_relational_rate_from_samples,
)

__all__ = (
    "RelationalCapacitySamplingCase",
    "RelationalCapacitySamplingCertificate",
    "certify_relational_capacity_sampling",
    "RelationalCapacityResponseProtocol",
    "RelationalCapacityResponseSample",
    "RelationalCapacityResponseCase",
    "RelationalCapacityResponseAnalysis",
    "RelationalCapacityResponse",
    "prepare_relational_capacity_response",
    "evaluate_relational_capacity_response",
)

_NODES = (0, 1, 2)
_FORM = (Q(1, 3), Q(0), Q(-1, 3))
_PHASE_PI = (Q(0), Q(1, 6), Q(-1, 6))
_BETA = (Q(3, 4), Q(5, 4))
_RADIUS = Q(1, 64)
_SAMPLE_STEP = Q(1, 64)
_SAMPLE_ERROR = Q(1, 2**24)
_THIRD_BOUND = Q(1, 2)
_RATE_ERROR_LIMIT = Q(1, 2**11)
_BASELINE_LOWER = Q(1, 8)


def _projection(report, schema):
    from ..sdk.relational_reports import _project

    return {"schema": schema, "report": _project(report)}


@dataclass(frozen=True)
class RelationalCapacitySamplingCase:
    """Whole-box derivative evidence for one law and held-capacity arm."""

    law: str
    arm: str
    capacity: tuple[Q, ...]
    coordinate_speed_upper_bounds: tuple[Q, ...]
    first_exit_margin: Q
    third_phase_contrast_upper_bound: Q
    initial_phase_contrast_rate_bounds: tuple[Q, Q]


@dataclass(frozen=True)
class RelationalCapacitySamplingCertificate:
    """Conditional prior-domain/error admission, with no acquired response."""

    nodes: tuple[int, ...]
    edges: tuple[tuple[int, int], ...]
    initial_form: tuple[Q, ...]
    initial_phase_pi_multipliers: tuple[Q, ...]
    beta_bounds: tuple[Q, Q]
    effective_weights: tuple[Q, Q]
    initial_bounds: tuple[tuple[Q, Q], ...]
    box_bounds: tuple[tuple[Q, Q], ...]
    coordinate_radius: Q
    acute_margin: Q
    sinc_argument_upper_bound: Q
    phase_metric_lower_bounds: tuple[Q, ...]
    sample_step: Q
    horizon: Q
    sample_error_bound: Q
    third_derivative_bound: Q
    rate_error_bound: Q
    rate_error_limit: Q
    baseline_rate_lower_bound: Q
    relative_rate_error_upper_bound: Q
    cases: tuple[RelationalCapacitySamplingCase, ...]
    arithmetic_method: str
    scope: tuple[str, ...] = (
        "fixed_unit_K3_exact_form_and_mathematical_pi_phase_preparation",
        "capacity_separable_and_named_capacity_mediated_candidate_laws_only",
        "held_capacities_support_beta_and_equal_half_channel_weights",
        "normalized_form_chart_and_relative_structural_clock_shared_between_arms",
        "strict_first_exit_box_and_whole_window_phase_contrast_C3_bound",
        "phase_contrast_is_theta_0_minus_theta_2_on_consistent_real_lifts",
        "sample_error_is_a_supplied_absolute_bound_for_each_phase_contrast_sample",
        "three_samples_at_zero_h_two_h_with_no_unknown_clock_or_gain_drift",
        "no_source_beta_response_samples_trajectory_fit_or_constitutive_selection",
        "no_preparation_error_derivation_observation_authentication_or_physical_bridge",
    )

    def to_dict(self):
        """Reuse exact dataclass projection without a new SDK report domain."""
        return _projection(self, "tnfr.relational-capacity-sampling.v1")


def _field_jets(state, capacity, beta, *, mediated):
    """Fixed K3 algebra in its acute chart; no native execution dispatch.

    Every entry is a shared interval Jet. K3 has two neighbors, so its local
    source angle is their mean minus the local angle, and its positive metric
    is 2*pi*cos(their half difference)*sinc(the source angle).
    """
    form, phase = state[:3], state[3:]
    pi = pi_interval()
    gradient = tuple(
        2 * form[i] - sum(form[j] for j in _NODES if j != i) for i in _NODES
    )
    displacement = tuple(
        sum(phase[j] for j in _NODES if j != i) / 2 - phase[i] for i in _NODES
    )
    metrics = []
    for i in _NODES:
        j, k = tuple(j for j in _NODES if j != i)
        metrics.append(
            cos((phase[j] - phase[k]) / 2) * sinc(displacement[i]) * (2 * pi)
        )
    form_rate = tuple(
        (-gradient[i] / 4 + displacement[i] / (2 * pi)) * capacity[i] for i in _NODES
    )
    phase_rate = [gradient[i] * capacity[i] / metrics[i] / (2 * beta) for i in _NODES]
    if mediated:
        phase_gradient = tuple(
            sum(sin(phase[i] - phase[j]) for j in _NODES if j != i) for i in _NODES
        )
        for i in _NODES:
            phase_rate[i] += sum(
                (gradient[i] + gradient[j])
                * sin(phase[j] - phase[i])
                * phase_gradient[j]
                * (capacity[i] * capacity[j] / (capacity[i] + capacity[j]))
                / (16 * beta)
                for j in _NODES
                if j != i
            )
    return form_rate + tuple(phase_rate)


def _flow_coefficients(box, capacity, beta, *, mediated, order=3):
    """Local flow jets over every point of a supplied box, default order three.

    The coefficient at degree n bounds z^(n)/n!. Recentring the initial jet
    at every box point supplies a whole-window derivative bound once the
    separate first-exit argument proves that the solution remains there.
    No endpoint or trajectory is evaluated by this recurrence.
    """
    if type(order) is not int or not 1 <= order <= MAX_ORDER:
        raise ValueError(f"flow jet order must be an integer from 1 to {MAX_ORDER}")
    coefficients = [[value] for value in box]
    for degree in range(1, order + 1):
        rate = _field_jets(
            tuple(Jet(tuple(row)) for row in coefficients),
            capacity,
            beta,
            mediated=mediated,
        )
        for row, value in zip(coefficients, rate, strict=True):
            row.append(value.coeffs[degree - 1] / degree)
    return tuple(tuple(row) for row in coefficients)


def _chart_bounds(box):
    """Require an acute chart and the shared sinc-jet convergence domain."""
    phase = box[3:]
    pi = pi_interval()
    maximum_gap = max(
        (phase[j] - phase[i]).abs_max for i in _NODES for j in _NODES if i < j
    )
    acute_margin = pi.lo / 2 - maximum_gap
    if acute_margin <= 0:
        raise ValueError("the complete sampling box must have strictly acute gaps")
    displacements = tuple(
        sum((phase[j] for j in _NODES if j != i), I(0)) / 2 - phase[i] for i in _NODES
    )
    maximum_displacement = max(value.abs_max for value in displacements)
    if maximum_displacement > 1:
        raise ValueError("the sampling box exceeds the shared sinc-jet domain")
    metrics = []
    for i in _NODES:
        j, k = tuple(j for j in _NODES if j != i)
        metric = (
            cos(Jet.constant((phase[j] - phase[k]) / 2, 0))
            * sinc(Jet.constant(displacements[i], 0))
            * (2 * pi)
        ).coeffs[0]
        if metric.lo <= 0:
            raise ValueError("the complete sampling box needs positive phase metrics")
        metrics.append(metric.lo)
    return acute_margin, maximum_displacement, tuple(metrics)


def certify_relational_capacity_sampling() -> RelationalCapacitySamplingCertificate:
    """Certify one fixed four-case temporal protocol before any response exists.

    The interval beta prior, box, capacities, clock, sample noise bound and
    horizon are declared. Derivative and first-exit bounds are computed from
    the two complete candidate laws, uniformly over that prior. They neither
    identify the true law nor establish that a device satisfies the noise,
    clock, coordinate, capacity or exact-preparation premises.
    """
    pi, beta = pi_interval(), I(*_BETA)
    initial = tuple(map(I, _FORM)) + tuple(pi * value for value in _PHASE_PI)
    box = tuple(value + I(-_RADIUS, _RADIUS) for value in initial)
    acute_margin, sinc_maximum, metrics = _chart_bounds(box)
    horizon = 2 * _SAMPLE_STEP
    cases = []
    for arm, capacity in (
        ("before", (Q(1), Q(1), Q(1))),
        ("after", (Q(1), Q(1, 2), Q(1))),
    ):
        for mediated in (False, True):
            coefficients = _flow_coefficients(box, capacity, beta, mediated=mediated)
            image = tuple(
                value + row[1] * I(0, horizon)
                for value, row in zip(initial, coefficients, strict=True)
            )
            margin = min(
                min(value.lo - domain.lo, domain.hi - value.hi)
                for value, domain in zip(image, box, strict=True)
            )
            third = 6 * (coefficients[3][3] - coefficients[5][3]).abs_max
            initial_rate = _field_jets(
                tuple(Jet.constant(value, 0) for value in initial),
                capacity,
                beta,
                mediated=mediated,
            )
            relative = (initial_rate[3] - initial_rate[5]).coeffs[0]
            speeds = tuple(row[1].abs_max for row in coefficients)
            if (
                margin <= 0
                or max(speeds) >= Q(2, 5)
                or third > _THIRD_BOUND
                or relative.lo <= _BASELINE_LOWER
            ):
                raise ArithmeticError("fixed sampling protocol admission is unresolved")
            cases.append(
                RelationalCapacitySamplingCase(
                    law="capacity_mediated" if mediated else "capacity_separable",
                    arm=arm,
                    capacity=capacity,
                    coordinate_speed_upper_bounds=speeds,
                    first_exit_margin=margin,
                    third_phase_contrast_upper_bound=third,
                    initial_phase_contrast_rate_bounds=(relative.lo, relative.hi),
                )
            )
    rate_error = 4 * _SAMPLE_ERROR / _SAMPLE_STEP + _THIRD_BOUND * _SAMPLE_STEP**2 / 3
    if rate_error >= _RATE_ERROR_LIMIT:
        raise ArithmeticError("fixed rate observation precision is unresolved")
    if 200 * rate_error >= _BASELINE_LOWER - 2 * rate_error:
        raise ArithmeticError("fixed normalized observation precision is unresolved")
    return RelationalCapacitySamplingCertificate(
        nodes=_NODES,
        edges=((0, 1), (0, 2), (1, 2)),
        initial_form=_FORM,
        initial_phase_pi_multipliers=_PHASE_PI,
        beta_bounds=_BETA,
        effective_weights=(Q(1, 2), Q(1, 2)),
        initial_bounds=tuple((value.lo, value.hi) for value in initial),
        box_bounds=tuple((value.lo, value.hi) for value in box),
        coordinate_radius=_RADIUS,
        acute_margin=acute_margin,
        sinc_argument_upper_bound=sinc_maximum,
        phase_metric_lower_bounds=metrics,
        sample_step=_SAMPLE_STEP,
        horizon=horizon,
        sample_error_bound=_SAMPLE_ERROR,
        third_derivative_bound=_THIRD_BOUND,
        rate_error_bound=rate_error,
        rate_error_limit=_RATE_ERROR_LIMIT,
        baseline_rate_lower_bound=_BASELINE_LOWER,
        relative_rate_error_upper_bound=rate_error / _BASELINE_LOWER,
        cases=tuple(cases),
        arithmetic_method=INTERVAL_METHOD,
    )


@dataclass(frozen=True)
class RelationalCapacityResponseProtocol:
    """Fixed prospective specification; constructing it evaluates no response."""

    protocol_id: str
    source_beta: Q
    taylor_order: int
    sampling: RelationalCapacitySamplingCertificate
    predictions: tuple[tuple[str, tuple[Q, Q]], ...]
    arithmetic_allowance: Q
    envelope_radius: Q
    whole_window_remainder_bounds: tuple[tuple[str, str, Q], ...]
    scope: tuple[str, ...] = (
        "known_model_ideal_source_not_blind_calibration_or_physical_observation",
        "one_initial_order_six_Taylor_polynomial_and_prior_box_order_seven_remainder",
        "fixed_protocol_no_refinement_fallback_parameter_fit_or_law_selection",
        "sample_midpoints_have_certified_radii_within_the_fixed_error_budget",
        "neutral_shared_sample_rate_and_normalized_contrast_observers",
        "producer_must_freeze_protocol_source_and_runtime_before_explicit_evaluation",
    )

    def to_dict(self):
        return _projection(self, "tnfr.relational-capacity-response-protocol.v1")


@dataclass(frozen=True)
class RelationalCapacityResponseSample:
    time: Q
    polynomial_bounds: tuple[tuple[Q, Q], ...]
    remainder_bounds: tuple[tuple[Q, Q], ...]
    state_bounds: tuple[tuple[Q, Q], ...]
    phase_contrast_bounds: tuple[Q, Q]
    midpoint: Q
    radius: Q
    admitted: bool


@dataclass(frozen=True)
class RelationalCapacityResponseCase:
    law: str
    arm: str
    capacity: tuple[Q, ...]
    status: str
    initial_coefficients: tuple[tuple[tuple[Q, Q], ...], ...]
    remainder_coefficients: tuple[tuple[Q, Q], ...]
    samples: tuple[RelationalCapacityResponseSample, ...]
    rate_observation: RelationalRateSampleBounds | None
    unavailable_reasons: tuple[str, ...]


@dataclass(frozen=True)
class RelationalCapacityResponseAnalysis:
    law: str
    contrast: RelationalRateContrastBounds | None
    baseline_lower_bound: Q | None
    exact_corner_hull: tuple[Q, Q] | None
    arithmetic_inflation: tuple[Q, Q] | None
    checks: tuple[tuple[str, bool], ...]
    passed: bool
    unavailable_reasons: tuple[str, ...]


@dataclass(frozen=True)
class RelationalCapacityResponse:
    protocol: RelationalCapacityResponseProtocol
    cases: tuple[RelationalCapacityResponseCase, ...]
    analyses: tuple[RelationalCapacityResponseAnalysis, ...]
    completed: bool
    contrasts_disjoint: bool
    passed: bool
    unavailable_reasons: tuple[str, ...]
    scope: tuple[str, ...] = (
        "known_model_ideal_K3_responses_with_conditional_Taylor_enclosures",
        "same_truth_independent_prior_domain_and_sample_error_budget_for_both_laws",
        "all_supplied_arms_observed_without_pressure_reconstruction_or_refitting",
        "partial_samples_failures_and_original_verdict_retained_without_fallback",
        "no_physical_acquisition_unique_constitutive_selection_or_default_engine_change",
        "typed_protocol_and_result_do_not_authenticate_external_freeze_or_execution",
    )

    def to_dict(self):
        return _projection(self, "tnfr.relational-capacity-response.v1")


def prepare_relational_capacity_response() -> RelationalCapacityResponseProtocol:
    """Declare the one supported response, computing only prior proof bounds.

    Whole-box seventh coefficients bound the Taylor remainder prospectively.
    Initial Taylor polynomials and sampled endpoints are not evaluated here.
    """
    admission = certify_relational_capacity_sampling()
    pi = pi_interval()
    root = 2 * interval_sin(pi / 3)
    alpha = 1 / (pi * root) + 1 / (2 * (1 + root))
    correction = (2 + root) / 32
    mediated_prediction = -correction / (3 * (alpha + correction))
    box, beta = tuple(I(*pair) for pair in admission.box_bounds), I(*_BETA)
    remainders = []
    for case in admission.cases:
        coefficients = _flow_coefficients(
            box, case.capacity, beta, mediated=case.law == "capacity_mediated", order=7
        )
        bound = (coefficients[3][7] - coefficients[5][7]).abs_max * admission.horizon**7
        remainders.append((case.law, case.arm, bound))
    return RelationalCapacityResponseProtocol(
        protocol_id="relational-k3-capacity-response-v1",
        source_beta=Q(1),
        taylor_order=6,
        sampling=admission,
        predictions=(
            ("capacity_separable", (Q(0), Q(0))),
            ("capacity_mediated", (mediated_prediction.lo, mediated_prediction.hi)),
        ),
        arithmetic_allowance=Q(1, 1584),
        envelope_radius=Q(1, 48),
        whole_window_remainder_bounds=tuple(remainders),
    )


def _polynomial(coefficients, time):
    """Horner evaluation of one normalized interval Taylor coefficient row."""
    value = coefficients[-1]
    for coefficient in reversed(coefficients[:-1]):
        value = value * time + coefficient
    return value


def _response_case(protocol, case):
    admission = protocol.sampling
    coefficients, remainder_coefficients, samples, reasons = (), (), [], []
    rate_observation = None
    try:
        initial = tuple(I(*pair) for pair in admission.initial_bounds)
        box = tuple(I(*pair) for pair in admission.box_bounds)
        coefficients = _flow_coefficients(
            initial,
            case.capacity,
            I(protocol.source_beta),
            mediated=case.law == "capacity_mediated",
            order=protocol.taylor_order,
        )
        whole_box = _flow_coefficients(
            box,
            case.capacity,
            I(*admission.beta_bounds),
            mediated=case.law == "capacity_mediated",
            order=protocol.taylor_order + 1,
        )
        remainder_coefficients = tuple(row[-1] for row in whole_box)
        for time in (Q(0), admission.sample_step, admission.horizon):
            polynomial = tuple(_polynomial(row, time) for row in coefficients)
            remainder = tuple(
                value * time ** (protocol.taylor_order + 1)
                for value in remainder_coefficients
            )
            state = tuple(a + b for a, b in zip(polynomial, remainder, strict=True))
            contrast = state[3] - state[5]
            admitted = contrast.radius <= admission.sample_error_bound
            samples.append(
                RelationalCapacityResponseSample(
                    time=time,
                    polynomial_bounds=tuple(
                        (value.lo, value.hi) for value in polynomial
                    ),
                    remainder_bounds=tuple((value.lo, value.hi) for value in remainder),
                    state_bounds=tuple((value.lo, value.hi) for value in state),
                    phase_contrast_bounds=(contrast.lo, contrast.hi),
                    midpoint=contrast.midpoint,
                    radius=contrast.radius,
                    admitted=admitted,
                )
            )
            if not admitted:
                reasons.append("sample_enclosure_exceeds_frozen_error_budget")
                break
        if not reasons and len(samples) == 3:
            rate_observation = bound_relational_rate_from_samples(
                tuple(sample.midpoint for sample in samples),
                sample_step=admission.sample_step,
                sample_error_bound=admission.sample_error_bound,
                third_derivative_bound=admission.third_derivative_bound,
            )
    except Exception as exc:
        reasons.append(f"response_stopped: {type(exc).__name__}: {exc}")
    return RelationalCapacityResponseCase(
        law=case.law,
        arm=case.arm,
        capacity=case.capacity,
        status="stopped" if reasons else "completed",
        initial_coefficients=tuple(
            tuple((value.lo, value.hi) for value in row) for row in coefficients
        ),
        remainder_coefficients=tuple(
            (value.lo, value.hi) for value in remainder_coefficients
        ),
        samples=tuple(samples),
        rate_observation=rate_observation,
        unavailable_reasons=tuple(reasons),
    )


def _response_analysis(protocol, law, cases):
    contrast, lower, hull, inflation = None, None, None, None
    checks, reasons = [], []
    try:
        selected = {case.arm: case for case in cases if case.law == law}
        if set(selected) != {"before", "after"} or any(
            case.status != "completed" or case.rate_observation is None
            for case in selected.values()
        ):
            raise ValueError("incomplete source cases do not admit a contrast")
        before = selected["before"].rate_observation
        after = selected["after"].rate_observation
        contrast = bound_relational_rate_contrast(
            before_bounds=before.rate_bounds, after_bounds=after.rate_bounds
        )
        if contrast.normalized_change_bounds is None:
            raise ValueError("normalized rate contrast is unavailable")
        # This fixed preparation has a positive oriented baseline. The generic
        # contrast observer admits either sign, but this protocol does not.
        lower = before.rate_bounds[0]
        corners = tuple(
            a / b - 1 for a in after.rate_bounds for b in before.rate_bounds
        )
        hull = min(corners), max(corners)
        observed = contrast.normalized_change_bounds
        inflation = hull[0] - observed[0], observed[1] - hull[1]
        predictions = dict(protocol.predictions)
        own = predictions[law]
        other = next(value for name, value in protocol.predictions if name != law)
        checks = [
            ("resolved_baseline", lower > 0),
            (
                "operational_rate_precision",
                200 * max(before.rate_error_bound, after.rate_error_bound) < lower,
            ),
            (
                "arithmetic_inflation_budget",
                all(0 <= value <= protocol.arithmetic_allowance for value in inflation),
            ),
            ("contains_own_prediction", observed[0] <= own[0] <= own[1] <= observed[1]),
            (
                "excludes_other_prediction",
                observed[1] < other[0] or other[1] < observed[0],
            ),
            (
                "within_own_envelope",
                own[0] - protocol.envelope_radius <= observed[0]
                and observed[1] <= own[1] + protocol.envelope_radius,
            ),
        ]
        reasons.extend(name for name, passed in checks if not passed)
    except Exception as exc:
        reasons.append(f"analysis_unavailable: {type(exc).__name__}: {exc}")
    return RelationalCapacityResponseAnalysis(
        law=law,
        contrast=contrast,
        baseline_lower_bound=lower,
        exact_corner_hull=hull,
        arithmetic_inflation=inflation,
        checks=tuple(checks),
        passed=not reasons,
        unavailable_reasons=tuple(reasons),
    )


def evaluate_relational_capacity_response(
    protocol: RelationalCapacityResponseProtocol,
) -> RelationalCapacityResponse:
    """Explicitly evaluate the fixed prospective research response once admitted.

    This API authenticates neither a file freeze nor a source archive. Its
    producer must establish those independently before calling it. A changed
    typed protocol rejects before response evaluation. Failed cases retain
    their available coefficients/samples, with no refinement or fallback.
    """
    from ..utils.io import json_dumps

    if not isinstance(protocol, RelationalCapacityResponseProtocol):
        raise TypeError("protocol must be a RelationalCapacityResponseProtocol")
    expected = prepare_relational_capacity_response()
    if json_dumps(protocol.to_dict(), sort_keys=True) != json_dumps(
        expected.to_dict(), sort_keys=True
    ):
        raise ValueError("protocol differs from the supported prospective declaration")
    cases = []
    for case in protocol.sampling.cases:
        result = _response_case(protocol, case)
        cases.append(result)
        if result.status != "completed":
            break
    analyses = tuple(
        _response_analysis(protocol, law, tuple(cases))
        for law, _ in protocol.predictions
    )
    intervals = tuple(
        analysis.contrast.normalized_change_bounds
        for analysis in analyses
        if analysis.contrast is not None
        and analysis.contrast.normalized_change_bounds is not None
    )
    disjoint = len(intervals) == 2 and (
        intervals[0][1] < intervals[1][0] or intervals[1][1] < intervals[0][0]
    )
    completed = len(cases) == len(protocol.sampling.cases) and all(
        case.status == "completed" for case in cases
    )
    reasons = []
    if not completed:
        reasons.append("source_incomplete")
    reasons.extend(
        f"{analysis.law}: {reason}"
        for analysis in analyses
        for reason in analysis.unavailable_reasons
    )
    if not disjoint:
        reasons.append("normalized_responses_not_separated")
    return RelationalCapacityResponse(
        protocol=protocol,
        cases=tuple(cases),
        analyses=analyses,
        completed=completed,
        contrasts_disjoint=disjoint,
        passed=not reasons,
        unavailable_reasons=tuple(reasons),
    )
