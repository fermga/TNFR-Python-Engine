"""Shared finite-dimensional Picard/Taylor/comparison enclosure steps.

The caller supplies a smooth interval/jet field and sufficient whole-box
domain margins. This kernel selects neither a model nor a trajectory target.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from ._comparison_flow import (
    MAX_COMPARISON_DIMENSION,
    _exact,
    _ordered,
    comparison_flow_upper,
)
from ._interval_taylor import MAX_ORDER, Jet
from ._rational_interval import I

MAX_BOX_TAYLOR_DIMENSION = 64


@dataclass(frozen=True)
class ValidatedBoxTaylorStep:
    """One direct source-box Taylor certificate, without radius propagation.

    Every coefficient encloses derivatives for every initial point. The
    increment omits the common order-zero coordinate symbolically, retaining
    initial/endpoint correlation that interval subtraction would discard.
    """

    time: Q
    duration: Q
    order: int
    initial_box: tuple[I, ...]
    tube: tuple[I, ...]
    series: tuple[tuple[I, ...], ...]
    local_remainder_bounds: tuple[I, ...]
    increment: tuple[I, ...]
    endpoint: tuple[I, ...]
    picard_interior_margin: Q
    domain_lower_bounds: tuple[Q, ...]
    method: str = "direct_source_box_Picard_Taylor_dyadic128_v1"


@dataclass(frozen=True)
class ValidatedTaylorStep:
    time: Q
    duration: Q
    tube: tuple[I, ...]
    endpoint: tuple[I, ...]
    picard_interior_margin: Q
    domain_lower_bounds: tuple[Q, ...]
    propagated_initial_radii: tuple[Q, ...]
    local_remainder_bounds: tuple[I, ...]


def _jet_rows(values, size, order):
    rows = _ordered(values, "field jets")
    if len(rows) != size:
        raise ValueError("field dimension differs from state")
    if any(not isinstance(row, Jet) for row in rows):
        raise TypeError("jet field must return derivative jets")
    if any(row.order != order for row in rows):
        raise ValueError("field jets must retain the requested derivative order")
    return rows


def flow_jets(box, order, flow, *, prescribed_time=None):
    """Enclose normalized solution derivatives through the declared order.

    An optional prescribed time is a point or interval of possible expansion
    origins. Its solution jet has first derivative one and higher derivatives
    zero. The field then takes ``(time_jet, state_jets)``; time is not an
    uncertain state coordinate or an independently perturbed initial value.
    """
    if type(order) is not int or not 0 <= order <= MAX_ORDER + 1:
        raise ValueError("solution Taylor order outside the shared jet domain")
    coefficients = [[I.coerce(value)] for value in _ordered(box, "state")]
    clock = None if prescribed_time is None else I.coerce(prescribed_time)
    for index in range(1, order + 1):
        variables = tuple(Jet(tuple(row)) for row in coefficients)
        if clock is None:
            values = flow(variables)
        else:
            time_coefficients = (clock,) + tuple(
                I(int(degree == 1)) for degree in range(1, index)
            )
            values = flow(Jet(time_coefficients), variables)
        rows = _jet_rows(
            values,
            len(coefficients),
            index - 1,
        )
        for row, rate in zip(coefficients, rows):
            row.append(rate.coeffs[index - 1] / index)
    return tuple(tuple(row) for row in coefficients)


def interval_jacobian(tube, flow, *, prescribed_time=None):
    """Enclose signed state derivatives, holding any prescribed time fixed.

    A time interval covers every common time in the step. Its seed is zero
    in every state derivative, so no fictitious clock uncertainty enters the
    same-time variational equation.
    """
    tube = tuple(I.coerce(value) for value in _ordered(tube, "Jacobian tube"))
    size = len(tube)
    if not 1 <= size <= MAX_COMPARISON_DIMENSION:
        raise ValueError("Jacobian dimension outside the shared comparison domain")
    columns = []
    clock = (
        None if prescribed_time is None else Jet.constant(I.coerce(prescribed_time), 1)
    )
    for column in range(size):
        variables = tuple(
            Jet((value, I(int(index == column)))) for index, value in enumerate(tube)
        )
        values = flow(variables) if clock is None else flow(clock, variables)
        rows = _jet_rows(values, size, 1)
        columns.append(tuple(row.coeffs[1] for row in rows))
    return tuple(tuple(columns[j][i] for j in range(size)) for i in range(size))


def comparison_matrix(tube, flow, *, prescribed_time=None):
    """One-sided diagonal and absolute off-diagonal Jacobian upper bounds."""
    jacobian = interval_jacobian(tube, flow, prescribed_time=prescribed_time)
    return tuple(
        tuple(value.hi if i == j else value.abs_max for j, value in enumerate(row))
        for i, row in enumerate(jacobian)
    )


def picard_tube(box, duration, flow, domain, *, domain_failure, prescribed_time=None):
    """Strict first-exit inclusion on a state box and optional time interval.

    The caller supplies the entire prescribed time interval for this step.
    Both the field and the domain reader consume that same interval.
    Inflation alone never certifies a tube.
    """
    tube = box
    clock = None if prescribed_time is None else I.coerce(prescribed_time)
    epsilon = Q(1, 1 << 90)
    for _ in range(16):
        try:
            margins = domain(tube) if clock is None else domain(clock, tube)
            lower = tuple(
                _exact(value) for value in _ordered(margins, "domain margins")
            )
            if not lower or min(lower) <= 0:
                return None, tube, domain_failure
            values = flow(tube) if clock is None else flow(clock, tube)
            rate = tuple(I.coerce(value) for value in _ordered(values, "field rates"))
            if len(rate) != len(box):
                raise ValueError("field dimension differs from state")
            image = tuple(x + f * I(0, duration) for x, f in zip(box, rate))
            margin = min(min(y.lo - b.lo, b.hi - y.hi) for y, b in zip(image, tube))
            if margin > 0:
                return (tube, margin, lower), None, None
            enlarged = tuple(x.hull(y) for x, y in zip(box, image))
            tube = tuple(
                I(
                    value.midpoint - value.radius * Q(5, 4) - epsilon,
                    value.midpoint + value.radius * Q(5, 4) + epsilon,
                )
                for value in enlarged
            )
        except (ValueError, ZeroDivisionError, ArithmeticError) as exc:
            return None, tube, f"tube_arithmetic_unavailable: {exc}"
    return None, tube, "strict_Picard_inclusion_not_resolved"


def validated_taylor_step(
    box,
    duration,
    flow,
    domain,
    *,
    order,
    time=Q(0),
    domain_failure="whole_tube_domain_not_admitted",
):
    """Return one certificate, or the failed enclosure and explicit reason.

    Strict Picard inclusion admits every initial point in the input box.
    The center Taylor endpoint uses order+1 derivatives on that same tube.
    A Metzler flow propagates initial uncertainty. No retries, step splitting
    or empirical convergence are substituted for these premises.
    State dimension follows the shared comparison-flow work policy; it is
    independent of the Taylor-order limit and the mathematical theorem.
    """
    for value, label in ((duration, "duration"), (time, "time")):
        if type(value) is not int and not isinstance(value, Q):
            raise TypeError(f"{label} must be an exact rational")
    duration, time = _exact(duration), _exact(time)
    if duration <= 0 or time < 0:
        raise ValueError("require positive duration and nonnegative time")
    if type(order) is not int or not 1 <= order <= MAX_ORDER:
        raise ValueError("Taylor order outside the shared jet domain")
    box = tuple(I.coerce(value) for value in _ordered(box, "state"))
    if not 1 <= len(box) <= MAX_COMPARISON_DIMENSION:
        raise ValueError(
            f"validated state dimension must lie between 1 and {MAX_COMPARISON_DIMENSION}"
        )
    admission, failed, reason = picard_tube(
        box, duration, flow, domain, domain_failure=domain_failure
    )
    if admission is None:
        return None, failed, reason
    tube, margin, bounds = admission
    try:
        center = tuple(I(value.midpoint) for value in box)
        series = flow_jets(center, order, flow)
        remainder = tuple(
            row[-1] * duration ** (order + 1)
            for row in flow_jets(tube, order + 1, flow)
        )
        polynomial = []
        for row in series:
            value = row[-1]
            for coefficient in reversed(row[:-1]):
                value = value * duration + coefficient
            polynomial.append(value)
        propagated = comparison_flow_upper(
            comparison_matrix(tube, flow),
            tuple(value.radius for value in box),
            duration,
        )
        endpoint = tuple(
            value + error + I(-radius, radius)
            for value, error, radius in zip(polynomial, remainder, propagated)
        )
        if any(max(x.lo, b.lo) > min(x.hi, b.hi) for x, b in zip(endpoint, tube)):
            raise ArithmeticError("disjoint endpoint and whole-time enclosures")
        endpoint = tuple(
            I(max(x.lo, b.lo), min(x.hi, b.hi)) for x, b in zip(endpoint, tube)
        )
    except (ValueError, ZeroDivisionError, ArithmeticError) as exc:
        return None, tube, f"Taylor_comparison_unavailable: {exc}"
    return (
        ValidatedTaylorStep(
            time, duration, tube, endpoint, margin, bounds, propagated, remainder
        ),
        None,
        None,
    )


def validated_box_taylor_step(
    box,
    duration,
    flow,
    domain,
    *,
    order,
    time=Q(0),
    domain_failure="whole_tube_domain_not_admitted",
):
    """Enclose one complete step by direct interval source-box coefficients.

    This bounded alternative shares Picard inclusion and formal solution jets
    with ``validated_taylor_step``. It evaluates every coefficient on the
    initial box, rather than a center followed by a comparison flow. No
    initial uncertainty is dropped and no interval Jacobian is required.
    All order+1 derivatives are evaluated on the strict whole-time tube.

    Dimension 1..64 and the shared order limit are explicit work policies;
    they do not change comparison-flow admission. Numerical failures return
    the last tube and a reason, without retries or changing the declared step.
    The increment excludes the order-zero term before interval arithmetic.
    Only coordinate endpoints may be intersected with the Picard tube.
    """
    for value, label in ((duration, "duration"), (time, "time")):
        if type(value) is not int and not isinstance(value, Q):
            raise TypeError(f"{label} must be an exact rational")
    duration, time = _exact(duration), _exact(time)
    if duration <= 0 or time < 0:
        raise ValueError("require positive duration and nonnegative time")
    if type(order) is not int or not 1 <= order <= MAX_ORDER:
        raise ValueError("Taylor order outside the shared jet domain")
    box = tuple(I.coerce(value) for value in _ordered(box, "state"))
    if not 1 <= len(box) <= MAX_BOX_TAYLOR_DIMENSION:
        raise ValueError(
            f"source-box Taylor dimension must lie between 1 and {MAX_BOX_TAYLOR_DIMENSION}"
        )
    admission, failed, reason = picard_tube(
        box, duration, flow, domain, domain_failure=domain_failure
    )
    if admission is None:
        return None, failed, reason
    tube, margin, bounds = admission
    try:
        series = flow_jets(box, order, flow)
        remainder_scale = duration ** (order + 1)
        remainder = tuple(
            row[-1] * remainder_scale for row in flow_jets(tube, order + 1, flow)
        )
        increment, endpoint = reconstruct_box_taylor_arithmetic(
            box, tube, series, remainder, duration, order=order
        )
    except (ValueError, ZeroDivisionError, ArithmeticError) as exc:
        return None, tube, f"Taylor_source_box_unavailable: {exc}"
    return (
        ValidatedBoxTaylorStep(
            time=time,
            duration=duration,
            order=order,
            initial_box=box,
            tube=tube,
            series=series,
            local_remainder_bounds=remainder,
            increment=increment,
            endpoint=endpoint,
            picard_interior_margin=margin,
            domain_lower_bounds=bounds,
        ),
        None,
        None,
    )


def reconstruct_box_taylor_arithmetic(
    initial_box,
    tube,
    series,
    local_remainder_bounds,
    duration,
    *,
    order,
):
    """Rebuild source-box Taylor increments and endpoints without a field call.

    This is an arithmetic check, not a trajectory certificate. The caller must
    separately establish the derivative enclosures, whole-time remainder and
    strict Picard/domain admission. Retained-report readers must also match
    their model, source, clock and events and compare consumed cached outputs.

    Exact positive duration, order1..16 and complete dimensions1..64 are
    admitted here. Each source coefficient must equal its initial interval,
    and the tube must contain the initial box. Horner evaluation excludes the
    common order-zero term before interval arithmetic. Only the endpoint is
    intersected with the tube; its increment retains its original enclosure.
    """
    duration = _exact(duration)
    if duration <= 0:
        raise ValueError("source-box Taylor duration must be positive")
    if type(order) is not int or not 1 <= order <= MAX_ORDER:
        raise ValueError("Taylor order outside the shared jet domain")
    initial = tuple(I.coerce(value) for value in _ordered(initial_box, "initial box"))
    size = len(initial)
    if not 1 <= size <= MAX_BOX_TAYLOR_DIMENSION:
        raise ValueError("source-box Taylor dimension outside the shared domain")
    whole = tuple(I.coerce(value) for value in _ordered(tube, "whole-time tube"))
    remainder = tuple(
        I.coerce(value)
        for value in _ordered(local_remainder_bounds, "local remainder bounds")
    )
    coefficients = tuple(
        tuple(I.coerce(value) for value in _ordered(row, "Taylor coefficient row"))
        for row in _ordered(series, "Taylor series")
    )
    if len(whole) != size or len(remainder) != size or len(coefficients) != size:
        raise ValueError("source-box Taylor evidence dimensions differ")
    if any(len(row) != order + 1 for row in coefficients):
        raise ValueError("Taylor coefficient count differs from the declared order")
    if any(row[0] != value for row, value in zip(coefficients, initial)):
        raise ValueError("Taylor source coefficient differs from the initial box")
    if any(not value.subset_of(bound) for value, bound in zip(initial, whole)):
        raise ValueError("initial box is not contained in the whole-time tube")
    increment = []
    for row, error in zip(coefficients, remainder):
        value = row[-1]
        for coefficient in reversed(row[1:-1]):
            value = value * duration + coefficient
        increment.append(value * duration + error)
    increment = tuple(increment)
    endpoint = tuple(value + change for value, change in zip(initial, increment))
    if any(max(x.lo, b.lo) > min(x.hi, b.hi) for x, b in zip(endpoint, whole)):
        raise ArithmeticError("disjoint endpoint and whole-time enclosures")
    endpoint = tuple(
        I(max(x.lo, b.lo), min(x.hi, b.hi)) for x, b in zip(endpoint, whole)
    )
    return increment, endpoint
