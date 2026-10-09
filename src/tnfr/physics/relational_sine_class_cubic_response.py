"""Complete cubic-amplitude response with certified finite time-series tails.

This triangular variational calculation retains every nodal coordinate and
the full linear feedback, quadratic odd correction and cubic recoupling. It
does not integrate a nonlinear trajectory or acquire an initial pattern.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q
from functools import lru_cache
from math import factorial

from ..mathematics._rational_interval import INTERVAL_METHOD, I, cos, pi_interval, sin
from ._sine_class_contrast import _contrast_decision, _ContrastDecision
from .relational_sine_class_mediation import _CONTACTS, _EDGES, _NODES
from .relational_sine_class_superposition import (
    _admit_probe_primitives,
    _ClassProbeHistoryBound,
    _probe_event_ledger,
)
from .relational_sine_port_composition import _parameters

__all__ = ("SineClassCubicResponse", "bound_sine_class_cubic_response")

_ORDER = 64
_LINEAR_NORM = Q(201, 100)
_CAUCHY_GAMMA = Q(1, 3000)
_ZERO = I(0)
_ZERO_STATE = (_ZERO,) * 54
_ZERO_LEVELS = (_ZERO_STATE,) * 3


@dataclass(frozen=True)
class _CubicParameters:
    classes: tuple[int, int, int]
    gamma: I
    eta: I
    degrees: tuple[int, ...]
    edge_sines: tuple[I, ...]
    edge_cosines: tuple[I, ...]


def _cubic_parameters(mediator_class):
    """Enclose coefficients from the actual lifted targets and oriented edges."""
    if type(mediator_class) is not int or mediator_class not in (1, 2):
        raise ValueError("mediator class must be ordinary integer one or two")
    classes = (1, mediator_class, 1)
    gamma, _ = _parameters(classes)
    degrees = tuple(sum(node in edge for edge in _EDGES) for node in _NODES)
    turns = tuple(Q(classes[c] * (j - 4), 9) for c in range(3) for j in range(9))
    sine, cosine = [], []
    pi = pi_interval()
    for left, right in _EDGES:
        gap = turns[right] - turns[left]
        wrapped = gap - (gap + Q(1, 2)).__floor__()
        if wrapped == 0:
            sine.append(_ZERO)
            cosine.append(I(1))
        else:
            angle = 2 * wrapped * pi
            sine.append(sin(angle))
            cosine.append(cos(angle))
    eta = gamma**2
    if gamma.hi >= _CAUCHY_GAMMA or 2 + 2 * eta.hi >= _LINEAR_NORM:
        raise ArithmeticError("fixed variational coefficient majorants failed")
    if any(value.abs_max > 1 for value in (*sine, *cosine)):
        raise ArithmeticError("fixed trigonometric coefficient bounds failed")
    return _CubicParameters(classes, gamma, eta, degrees, tuple(sine), tuple(cosine))


def _linear_variation(state, parameters):
    """M(u,w)=(-A*u-eta*C*w,A*u) in all original node coordinates."""
    if len(state) != 54:
        raise ValueError("each amplitude level must retain all 54 coordinates")
    form, phase = [_ZERO] * 27, [_ZERO] * 27
    for (left, right), cosine in zip(_EDGES, parameters.edge_cosines):
        du = state[right] - state[left]
        dw = state[27 + right] - state[27 + left]
        current = du + dw * (parameters.eta * cosine)
        form[left] += current / parameters.degrees[left]
        form[right] -= current / parameters.degrees[right]
        phase[left] -= du / parameters.degrees[left]
        phase[right] += du / parameters.degrees[right]
    return tuple(form + phase)


def _convolution(left, right, degree):
    """Bounded polynomial multiplication for the triangular time recurrence."""
    result = [_ZERO] * (degree + 1)
    for i, a in enumerate(left[: degree + 1]):
        if a.lo == a.hi == 0:
            continue
        for j, b in enumerate(right[: degree + 1 - i]):
            if b.lo != 0 or b.hi != 0:
                result[i + j] += a * b
    return tuple(result)


def _edge_phase_series(series, left, right):
    return tuple(row[27 + right] - row[27 + left] for row in series)


def _time_coefficients(initial_levels, parameters, *, order=_ORDER):
    """Compute normalized time coefficients, sequentially in amplitude degree.

    Lower orders support independent polynomial controls; the public bound
    fixes 64. This is not a change to the shared ODE integrator's work limits.
    """
    if type(order) is not int or not 1 <= order <= _ORDER:
        raise ValueError("variational polynomial order must lie in 1..64")
    if len(initial_levels) != 3 or any(len(level) != 54 for level in initial_levels):
        raise ValueError("three full 54-coordinate amplitude levels are required")
    initial = tuple(tuple(I.coerce(v) for v in level) for level in initial_levels)

    def coefficients(start, forcing=None):
        rows = [start]
        for n in range(order):
            rates = _linear_variation(rows[-1], parameters)
            if forcing is not None:
                rates = tuple(value + force for value, force in zip(rates, forcing[n]))
            rows.append(tuple(value / (n + 1) for value in rates))
        return tuple(rows)

    first = coefficients(initial[0])
    first_edges, squares = [], []
    quadratic = [[_ZERO] * 54 for _ in range(order)]
    for edge, sine in zip(_EDGES, parameters.edge_sines):
        left, right = edge
        p = _edge_phase_series(first, left, right)
        square = _convolution(p, p, order - 1)
        first_edges.append(p)
        squares.append(square)
        for n, value in enumerate(square):
            current = -value * sine / 2
            quadratic[n][left] += current / parameters.degrees[left]
            quadratic[n][right] -= current / parameters.degrees[right]
    second = coefficients(initial[1], quadratic)
    cubic = [[_ZERO] * 54 for _ in range(order)]
    for index, (left, right) in enumerate(_EDGES):
        p = first_edges[index]
        q = _edge_phase_series(second, left, right)
        recoupling = _convolution(p, q, order - 1)
        direct = _convolution(squares[index], p, order - 1)
        for n in range(order):
            current = (
                -recoupling[n] * (parameters.eta * parameters.edge_sines[index])
                - direct[n] * parameters.edge_cosines[index] / 6
            )
            cubic[n][left] += current / parameters.degrees[left]
            cubic[n][right] -= current / parameters.degrees[right]
    third = coefficients(initial[2], cubic)
    return first, second, third


def _exponential_tail(argument, first_omitted):
    """Positive exponential-series tail without subtracting rounded exponentials."""
    if argument < 0 or argument >= first_omitted + 1:
        raise ValueError("exponential tail is outside its geometric ratio domain")
    return (
        argument**first_omitted
        / factorial(first_omitted)
        / (1 - argument / (first_omitted + 1))
    )


def _time_tails(initial_levels, duration, eta_upper):
    norms = tuple(max(value.abs_max for value in level) for level in initial_levels)
    v1, v2, v3 = norms
    h, rate = duration, _LINEAR_NORM
    tail1 = v1 * _exponential_tail(rate * h, _ORDER + 1)
    tail2 = v2 * _exponential_tail(
        2 * rate * h, _ORDER + 1
    ) + 2 * v1**2 * h * _exponential_tail(2 * rate * h, _ORDER)
    tail3 = (
        v3 * _exponential_tail(3 * rate * h, _ORDER + 1)
        + (4 * eta_upper * v1 * v2 + Q(4, 3) * v1**3)
        * h
        * _exponential_tail(3 * rate * h, _ORDER)
        + 4 * eta_upper * v1**3 * h**2 * _exponential_tail(3 * rate * h, _ORDER - 1)
    )
    return norms, (tail1, tail2, tail3)


def _event_levels(levels, amplitude):
    """A form event changes only the first level's donor form coefficient."""
    first = tuple(
        value + amplitude if i == 4 and amplitude else value
        for i, value in enumerate(levels[0])
    )
    return first, levels[1], levels[2]


@dataclass(frozen=True)
class _CubicSegment:
    label: str
    parent_segment_index: int | None
    start_time: Q
    end_time: Q
    form_jump: Q
    initial_levels: tuple[tuple[I, ...], ...]
    time_coefficients: tuple[tuple[tuple[I, ...], ...], ...]
    initial_level_norm_upper_bounds: tuple[Q, ...]
    time_tail_upper_bounds: tuple[Q, ...]
    endpoint_levels: tuple[tuple[I, ...], ...]


def _coefficient_segment(label, parent, start, end, jump, before, parameters):
    initial = _event_levels(before, jump)
    series = _time_coefficients(initial, parameters)
    h = end - start
    norms, tails = _time_tails(initial, h, parameters.eta.hi)
    endpoints = []
    for rows, tail in zip(series, tails):
        state = rows[-1]
        for row in reversed(rows[:-1]):
            state = tuple(
                value * h + coefficient for value, coefficient in zip(state, row)
            )
        endpoints.append(tuple(value + I(-tail, tail) for value in state))
    return _CubicSegment(
        label, parent, start, end, jump, initial, series, norms, tails, tuple(endpoints)
    )


@lru_cache(maxsize=2)
def _class_cubic_coefficients(mediator_class, a, b, delay, total):
    """Reuse immutable coefficients only after the public primitive admission.

    Two entries hold the fixed class pair for one design. Source/error/work
    policies and their decisions are rebuilt by every public call.
    """
    parameters = _cubic_parameters(mediator_class)
    prefix = _coefficient_segment(
        "first_prefix", None, Q(0), delay, a, _ZERO_LEVELS, parameters
    )
    first = _coefficient_segment(
        "first_suffix", 0, delay, total, Q(0), prefix.endpoint_levels, parameters
    )
    both = _coefficient_segment(
        "both_suffix", 0, delay, total, b, prefix.endpoint_levels, parameters
    )
    second = _coefficient_segment(
        "second_only_suffix", None, delay, total, b, _ZERO_LEVELS, parameters
    )
    mixed = (
        both.endpoint_levels[2][22]
        - first.endpoint_levels[2][22]
        - second.endpoint_levels[2][22]
    )
    return parameters, (prefix, first, both, second), mixed


def _higher_amplitude_remainder(amplitude, total):
    if amplitude == 0:
        return Q(0)
    g = _CAUCHY_GAMMA
    return (
        256
        * g**6
        * amplitude**5
        * total
        / ((1 - 2 * g**2 * total**2) * (1 - 4 * g**2 * amplitude**2))
    )


@dataclass(frozen=True)
class SineClassCubicResponse:
    """A complete nominal cubic coefficient plus a conditional full-law bound.

    Sources may differ across classes; only each class's four histories share
    one full original state. Nominal reflection parity bounds the degree-five
    amplitude remainder, not the actual source errors. Scalar cancellation
    does not mean overlap of the eight original endpoint record vectors.
    """

    first_probe_amplitude: Q
    second_probe_amplitude: Q
    delay: Q
    total_duration: Q
    endpoint_radius: Q
    readout_error_bound: Q
    radius: Q
    contact_work_allowance: Q
    first_probe_work_allowance: Q
    second_probe_work_allowance: Q
    gamma_bounds: I
    cauchy_bootstrap_margin: Q
    cauchy_radius_margin: Q
    cauchy_admitted: bool
    coefficient_evaluated: bool
    class_parameters: tuple[_CubicParameters, ...]
    class_segments: tuple[tuple[_CubicSegment, ...], ...]
    scaled_class_cubic_bounds: tuple[tuple[Q, Q], ...] | None
    complete_cubic_contrast_bounds: tuple[Q, Q] | None
    per_history_higher_amplitude_remainder_upper_bounds: tuple[Q, ...] | None
    higher_amplitude_contrast_error_upper_bound: Q | None
    source_contrast_error_upper_bound: Q
    decision: _ContrastDecision | None
    response_bound_available: bool
    exact_contrast_zero: bool
    uniform_history_bounds: tuple[_ClassProbeHistoryBound, ...]
    joined_initial_identity_certified: bool
    contact_work_within_allowance: bool
    all_identities_certified: bool
    all_work_within_allowances: bool
    status: str
    unavailable_reasons: tuple[str, ...]
    compared_classes: tuple[tuple[int, ...], ...] = ((1, 1, 1), (1, 2, 1))
    nodes: tuple[int, ...] = _NODES
    edges: tuple[tuple[int, int], ...] = _EDGES
    contacts: tuple[tuple[int, int], ...] = _CONTACTS
    time_polynomial_order: int = _ORDER
    linear_majorant_rate: Q = _LINEAR_NORM
    cauchy_gamma_upper_bound: Q = _CAUCHY_GAMMA
    amplitude_level_scaling: tuple[str, ...] = (
        "x1=u1; y1=gamma*w1",
        "x2=gamma^3*u2; y2=gamma^4*w2",
        "x3=gamma^4*u3; y3=gamma^5*w3",
    )
    clock: str = "tau=e*t; e=1023/1024"
    method: str = "complete_amplitude_cubic_time_polynomial64_dyadic128_majorant_v1"
    interval_method: str = INTERVAL_METHOD
    scope: tuple[str, ...] = (
        "full54_original_coordinates_at_each_of_three_scaled_amplitude_levels",
        "linear_feedback_and_quadratic_odd_to_cubic_recoupling_are_retained",
        "both_event_times_carry_all_variation_levels_only_first_level_donor_form_jumps",
        "delayed_only_history_has_its_own_exact_nominal_zero_prefix",
        "analytic_triangular_coefficient_recurrence_is_not_a_nonlinear_trajectory_solver",
        "time_polynomial_tail_and_higher_amplitude_remainder_are_distinct",
        "nominal_parity_does_not_remove_independent_actual_source_errors",
        "per_history_cauchy_radii_use_each_present_or_absent_event_amplitude",
        "unsupported_cauchy_domain_withholds_full_response_except_exact_pairing_zero",
        "scalar_noise_cancellation_does_not_prove_raw_record_vector_overlap",
        "source_formation_work_identity_and_physical_identification_are_not_observations",
    )

    def to_dict(self):
        from ..sdk.relational_reports import _project

        return {"schema": "tnfr.sine-class-cubic-response.v1", "report": _project(self)}


def bound_sine_class_cubic_response(
    *,
    first_probe_amplitude,
    second_probe_amplitude,
    delay,
    total_duration,
    endpoint_radius,
    readout_error_bound,
    radius,
    contact_work_allowance,
    first_probe_work_allowance,
    second_probe_work_allowance,
) -> SineClassCubicResponse:
    """Bound the fixed class contrast; no source or saved response is consumed.

    Admit the ten shared event/source/error primitives, with signed amplitudes,
    0<=delay<=total_duration<=2 and 0<radius<=1/12. The analytic work policy is
    fixed at order 64 and eight segments, without adaptation. Unsupported
    complex-amplitude domains remain unavailable, not negative error bounds.
    """
    raw = dict(
        first_probe_amplitude=first_probe_amplitude,
        second_probe_amplitude=second_probe_amplitude,
        delay=delay,
        total_duration=total_duration,
        endpoint_radius=endpoint_radius,
        readout_error_bound=readout_error_bound,
        radius=radius,
        contact_work_allowance=contact_work_allowance,
        first_probe_work_allowance=first_probe_work_allowance,
        second_probe_work_allowance=second_probe_work_allowance,
    )
    values = _admit_probe_primitives(1, raw, Q(2))
    a, b, s, t, eps, delta = (
        values[key]
        for key in (
            "first_probe_amplitude",
            "second_probe_amplitude",
            "delay",
            "total_duration",
            "endpoint_radius",
            "readout_error_bound",
        )
    )
    gamma, _ = _parameters((1, 2, 1))
    amplitude = abs(a) + abs(b)
    bootstrap = 1 - 16 * _CAUCHY_GAMMA**2 * t**2
    radius_margin = 1 - 2 * _CAUCHY_GAMMA * amplitude
    admitted = bootstrap > 0 and radius_margin > 0
    exact_zero = a == 0 or b == 0 or s == t
    parameters, segments = [], []
    class_bounds = contrast = remainders = higher_error = decision = None
    source_error = 8 * eps / (1 - 2 * gamma.hi * t)
    if exact_zero:
        class_bounds = ((Q(0), Q(0)),) * 2
        contrast = (Q(0), Q(0))
        decision = _contrast_decision(contrast, Q(0), delta, exact_zero=True)
    elif admitted:
        coefficients = []
        for mediator_class in (1, 2):
            params, history, coefficient = _class_cubic_coefficients(
                mediator_class, a, b, s, t
            )
            parameters.append(params)
            segments.append(history)
            coefficients.append(coefficient)

        def scale(interval):
            products = tuple(
                g**4 * v
                for g in (gamma.lo, gamma.hi)
                for v in (interval.lo, interval.hi)
            )
            return min(products), max(products)

        class_bounds = tuple(scale(value) for value in coefficients)
        contrast = scale(coefficients[0] - coefficients[1])
        remainders = tuple(
            _higher_amplitude_remainder(value, t)
            for value in (Q(0), abs(a), abs(b), amplitude)
        )
        higher_error = 2 * sum(remainders, Q(0))
        decision = _contrast_decision(contrast, higher_error + source_error, delta)
    ledger = _probe_event_ledger(values, gamma.hi, (None,) * 4)
    return SineClassCubicResponse(
        **values,
        gamma_bounds=gamma,
        cauchy_bootstrap_margin=bootstrap,
        cauchy_radius_margin=radius_margin,
        cauchy_admitted=admitted,
        coefficient_evaluated=bool(segments),
        class_parameters=tuple(parameters),
        class_segments=tuple(segments),
        scaled_class_cubic_bounds=class_bounds,
        complete_cubic_contrast_bounds=contrast,
        per_history_higher_amplitude_remainder_upper_bounds=remainders,
        higher_amplitude_contrast_error_upper_bound=higher_error,
        source_contrast_error_upper_bound=source_error,
        decision=decision,
        response_bound_available=decision is not None,
        exact_contrast_zero=exact_zero,
        uniform_history_bounds=ledger.histories,
        joined_initial_identity_certified=ledger.initial_identity,
        contact_work_within_allowance=ledger.contact_allowed,
        all_identities_certified=all(h.identity_certified for h in ledger.histories),
        all_work_within_allowances=all(
            h.work_within_allowances for h in ledger.histories
        ),
        status=decision.status if decision is not None else "unavailable",
        unavailable_reasons=(
            () if decision is not None else ("complex_amplitude_domain_not_admitted",)
        ),
    )
