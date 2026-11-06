"""Conditional Taylor enclosures for contractive linear prescribed-time laws.

The supplied law is q'=A*q+f(t), with a constant Metzler matrix whose row
sums are nonpositive. Its homogeneous flow contracts the infinity norm.
Forcing jets must enclose the derivatives of one smooth prescribed function
on the whole step; callers split its branches and retain its clock. This
kernel proves no nonlinear reduction, state admission or forcing provenance.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q
from typing import Any, Callable, Iterable

from .._exact_time import exact_or_represented_real
from ._exact_linear_algebra import _ordered, _rational_matrix, _rational_vector
from ._interval_taylor import MAX_ORDER, Jet
from ._rational_interval import I

__all__ = ("ValidatedLinearForcingStep", "validated_linear_forcing_step")

_MAX_DIMENSION = 64


@dataclass(frozen=True, slots=True)
class ValidatedLinearForcingStep:
    """One step with a uniform infinity-ball error and explicit local evidence.

    The remainder tube bounds the exact solution starting at initial_center,
    not the preceding uncertain ball. Homogeneous contraction propagates that
    ball's radius separately. Endpoint centers are exact dyadic midpoints;
    local_error includes both polynomial rounding and Taylor truncation.
    """

    initial_center: tuple[Q, ...]
    initial_radius: Q
    generator: tuple[tuple[Q, ...], ...]
    time: Q
    duration: Q
    order: int
    forcing_point: tuple[Jet, ...]
    forcing_tube: tuple[Jet, ...]
    forcing_bound: Q
    state_bound: Q
    polynomial: tuple[I, ...]
    remainder: tuple[I, ...]
    local_error: Q
    endpoint_center: tuple[Q, ...]
    endpoint_radius: Q


def _forcing_jets(
    forcing: Callable[[Jet], Any], clock: Jet, dimension: int, order: int
) -> tuple[Jet, ...]:
    values = _ordered(forcing(clock), "forcing result")
    if len(values) != dimension:
        raise ValueError("forcing result must match the generator dimension")
    if any(not isinstance(value, Jet) for value in values):
        raise TypeError("forcing result must contain Taylor jets")
    if any(value.order != order for value in values):
        raise ValueError("forcing jets must match the requested Taylor order")
    return values


def _coefficients(
    initial: Iterable[Any],
    rows: tuple[tuple[tuple[int, Q], ...], ...],
    forcing: tuple[Jet, ...],
    count: int,
) -> list[tuple[I, ...]]:
    """Enclose q^(k)/k! by the exact affine derivative recurrence."""
    result = [tuple(I.coerce(value) for value in initial)]
    for degree in range(count):
        previous = result[-1]
        result.append(
            tuple(
                (
                    sum((value * previous[j] for j, value in row), I(0))
                    + forcing[i].coeffs[degree]
                )
                / (degree + 1)
                for i, row in enumerate(rows)
            )
        )
    return result


def validated_linear_forcing_step(
    center: Any,
    radius: Any,
    generator: Any,
    duration: Any,
    forcing: Callable[[Jet], Any],
    *,
    order: int,
    time: Any = Q(0),
) -> ValidatedLinearForcingStep:
    """Enclose one fixed-order step of q'=A*q+f(t), without adaptive retries.

    A is a nonempty square constant Metzler matrix, with nonpositive row
    sums and dimension at most 64. Scalar inputs share exact-or-represented
    admission: rational values remain exact, other admitted real values
    retain their finite binary64 representation. The certificate concerns
    that supplied matrix, not an ideal matrix approximated by its entries.
    Radius and absolute time are nonnegative; duration is positive, and
    order is an ordinary integer from 1 through the shared MAX_ORDER.

    ``forcing(time_jet)`` returns a matching tuple of normalized derivative
    jets. It is called twice, at the point time and at the entire closed
    step interval, each with time derivative one. Its outputs must enclose
    one C^order forcing function and all consumed derivatives throughout
    those domains. These smoothness and enclosure premises remain the
    caller's obligation: a callback's values cannot authenticate them.
    In particular, split nonsmooth input branches before calling this step.

    With F=max_i sup|f_i| from the whole-step zeroth coefficients, variation
    of constants gives ||q(t)||_infinity<=||center||_infinity+h*F for the
    center-started solution. That proved tube and the forcing derivative
    jets enclose the order+1 derivative via
    q[k+1]=(A*q[k]+f[k])/(k+1). Taylor's theorem bounds the local remainder.
    The prior radius is added unchanged because ||exp(h*A)||_infinity<=1;
    it is not fed back as an independent box at each derivative order.

    No input fit, numerical trajectory search, stochastic error assumption,
    actual initial-state certificate or nonlinear observation follows.
    """
    if type(order) is not int:
        raise TypeError("Taylor order must be an ordinary integer")
    if not 1 <= order <= MAX_ORDER:
        raise ValueError(f"Taylor order must lie between 1 and {MAX_ORDER}")
    matrix = _rational_matrix(generator, "generator")
    dimension = len(matrix)
    if not 1 <= dimension <= _MAX_DIMENSION:
        raise ValueError(f"generator dimension must lie between 1 and {_MAX_DIMENSION}")
    if any(len(row) != dimension for row in matrix):
        raise ValueError("generator must be square")
    if any(
        value < 0
        for i, row in enumerate(matrix)
        for j, value in enumerate(row)
        if i != j
    ):
        raise ValueError("generator must be Metzler")
    if any(sum(row, Q(0)) > 0 for row in matrix):
        raise ValueError("generator row sums must be nonpositive")
    center = _rational_vector(center, "center")
    if len(center) != dimension:
        raise ValueError("center must match the generator dimension")
    radius, time, duration = (
        exact_or_represented_real(value, label)
        for value, label in ((radius, "radius"), (time, "time"), (duration, "duration"))
    )
    if radius < 0 or time < 0 or duration <= 0:
        raise ValueError("radius/time must be nonnegative and duration positive")
    tail = (I(1),) + (I(0),) * (order - 1)
    point = _forcing_jets(forcing, Jet((I(time),) + tail), dimension, order)
    whole = _forcing_jets(
        forcing, Jet((I(time, time + duration),) + tail), dimension, order
    )
    forcing_bound = max(value.coeffs[0].abs_max for value in whole)
    state_bound = max(map(abs, center)) + duration * forcing_bound
    rows = tuple(
        tuple((j, value) for j, value in enumerate(row) if value) for row in matrix
    )
    coefficients = _coefficients(center, rows, point, order)
    tube_coefficients = _coefficients(
        (I(-state_bound, state_bound),) * dimension, rows, whole, order + 1
    )
    polynomial = []
    for i in range(dimension):
        value = coefficients[-1][i]
        for coefficient in reversed(coefficients[:-1]):
            value = value * duration + coefficient[i]
        polynomial.append(value)
    polynomial = tuple(polynomial)
    remainder = tuple(
        value * duration ** (order + 1) for value in tube_coefficients[-1]
    )
    local_error = I(
        max(
            value.radius + residual.abs_max
            for value, residual in zip(polynomial, remainder)
        )
    ).hi
    return ValidatedLinearForcingStep(
        initial_center=center,
        initial_radius=radius,
        generator=matrix,
        time=time,
        duration=duration,
        order=order,
        forcing_point=point,
        forcing_tube=whole,
        forcing_bound=forcing_bound,
        state_bound=state_bound,
        polynomial=polynomial,
        remainder=remainder,
        local_error=local_error,
        endpoint_center=tuple(value.midpoint for value in polynomial),
        endpoint_radius=I(radius + local_error).hi,
    )
