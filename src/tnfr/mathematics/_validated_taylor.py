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


def flow_jets(box, order, flow):
    """Enclose normalized solution derivatives through the declared order."""
    if type(order) is not int or not 0 <= order <= MAX_ORDER + 1:
        raise ValueError("solution Taylor order outside the shared jet domain")
    coefficients = [[I.coerce(value)] for value in _ordered(box, "state")]
    for index in range(1, order + 1):
        rows = _jet_rows(
            flow(tuple(Jet(tuple(row)) for row in coefficients)),
            len(coefficients),
            index - 1,
        )
        for row, rate in zip(coefficients, rows):
            row.append(rate.coeffs[index - 1] / index)
    return tuple(tuple(row) for row in coefficients)


def interval_jacobian(tube, flow):
    """Enclose every signed derivative of the admitted whole-tube field."""
    tube = tuple(I.coerce(value) for value in _ordered(tube, "Jacobian tube"))
    size = len(tube)
    if not 1 <= size <= MAX_COMPARISON_DIMENSION:
        raise ValueError("Jacobian dimension outside the shared comparison domain")
    columns = []
    for column in range(size):
        variables = tuple(
            Jet((value, I(int(index == column)))) for index, value in enumerate(tube)
        )
        rows = _jet_rows(flow(variables), size, 1)
        columns.append(tuple(row.coeffs[1] for row in rows))
    return tuple(tuple(columns[j][i] for j in range(size)) for i in range(size))


def comparison_matrix(tube, flow):
    """One-sided diagonal and absolute off-diagonal Jacobian upper bounds."""
    jacobian = interval_jacobian(tube, flow)
    return tuple(
        tuple(value.hi if i == j else value.abs_max for j, value in enumerate(row))
        for i, row in enumerate(jacobian)
    )


def picard_tube(box, duration, flow, domain, *, domain_failure):
    """Strict first-exit inclusion; inflation alone never certifies a tube."""
    tube = box
    epsilon = Q(1, 1 << 90)
    for _ in range(16):
        try:
            lower = tuple(
                _exact(value) for value in _ordered(domain(tube), "domain margins")
            )
            if not lower or min(lower) <= 0:
                return None, tube, domain_failure
            rate = tuple(
                I.coerce(value) for value in _ordered(flow(tube), "field rates")
            )
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
    duration, time = Q(duration), Q(time)
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
