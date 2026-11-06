"""Conditional arithmetic for supplied complete-law preparation errors.

These helpers admit no graph, historical preparation, nominal trajectory or
response certificate. Their caller establishes common initial-state association,
both complete laws and the Dini comparison premises. In particular, a family of
ideal real states is not rematerialized to its displayed binary64 center.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from .._binary64 import uses_ieee_binary64_rounding
from .._exact_time import represented_fraction
from ..mathematics._comparison_flow import (
    COMPARISON_FLOW_METHOD,
    _exact,
    _ordered,
    comparison_flow_upper,
)
from ..physics.binary64_nodal_flow import _rounding_cell

__all__ = (
    "AdjacentCoordinateHull",
    "TwoBlockErrorBound",
    "adjacent_coordinate_hull",
    "adjacent_coordinate_hulls",
    "bound_two_block_errors",
    "connected_marked_error_squared",
    "cut_marked_error_squared",
)


@dataclass(frozen=True)
class AdjacentCoordinateHull:
    """Closed predecessor-to-successor hull, with exact asymmetric widths.

    This is wider than the nearest-even midpoint cell. It is an explicitly
    supplied real-coordinate family, not inferred acquisition uncertainty or
    the inverse image of an API that rejects nonzero underflows. No circular
    wrapping is applied to a real phase-lift coordinate.
    """

    center: Q
    lower: Q
    upper: Q

    @property
    def lower_error(self) -> Q:
        return self.center - self.lower

    @property
    def upper_error(self) -> Q:
        return self.upper - self.center

    @property
    def radius(self) -> Q:
        return max(self.lower_error, self.upper_error)


def adjacent_coordinate_hull(value: float) -> AdjacentCoordinateHull:
    """Retain exact finite neighbors of a genuine canonical-zero float.

    Reuse the shared nearest-even owner's adjacent values; do not reproduce
    nextafter or significand logic here. Its finite-neighbor restriction also
    applies. Exact Fraction endpoints preserve subnormal widths before any
    optional outward interval arithmetic by a later consumer.
    """
    center = represented_fraction(value, "coordinate center")
    if not uses_ieee_binary64_rounding():
        raise ValueError("coordinate hull requires IEEE binary64 nearest-even rounding")
    cell = _rounding_cell(value, center)
    return AdjacentCoordinateHull(
        center, Q.from_float(cell.predecessor), Q.from_float(cell.successor)
    )


def adjacent_coordinate_hulls(values) -> tuple[AdjacentCoordinateHull, ...]:
    """Capture a nonempty ordered coordinate row without a model-specific cap.

    The caller owns complete state dimension, coordinate order and work budget.
    This helper neither selects coordinates nor derives a source association.
    """
    values = _ordered(values, "coordinates")
    if not values:
        raise ValueError("coordinate row must be nonempty")
    return tuple(adjacent_coordinate_hull(value) for value in values)


def _nonnegative(value, label):
    value = _exact(value)
    if value < 0:
        raise ValueError(f"{label} must be nonnegative")
    return value


@dataclass(frozen=True)
class TwoBlockErrorBound:
    """Fixed-step upper comparison, including its exact initial error bounds.

    For independently justified ``D+U <= A*V`` and ``D+V <= B*U``, the
    nonnegative comparison matrix has a nondecreasing exact solution. Each
    returned endpoint bounds its entire preceding time interval; the last
    endpoint bounds the whole window. Numerical outward excess is retained
    between steps. This is no trajectory of either underlying complete law.
    """

    A: Q
    B: Q
    horizon: Q
    steps: int
    endpoints: tuple[tuple[Q, Q], ...]
    method: str = COMPARISON_FLOW_METHOD

    @property
    def step_size(self) -> Q:
        return self.horizon / self.steps

    @property
    def final(self) -> tuple[Q, Q]:
        return self.endpoints[-1]


def bound_two_block_errors(A, B, U0, V0, T, *, steps: int) -> TwoBlockErrorBound:
    """Propagate the supplied exact nonnegative two-block comparison only.

    Exactly ``steps`` equal substeps are requested from the shared comparison
    kernel. Admit ``h*max(A,B)<=1`` before propagation, including zero initial
    errors. Unsupported work domains raise; no subdivision, retry, new
    exponential implementation or fitted coefficient is introduced.
    Coefficient/error units and the meaning of each block belong to the caller.
    """
    A, B, U0, V0, T = (
        _nonnegative(value, label)
        for value, label in ((A, "A"), (B, "B"), (U0, "U0"), (V0, "V0"), (T, "T"))
    )
    if type(steps) is not int:
        raise TypeError("steps must be a positive ordinary integer")
    if steps <= 0:
        raise ValueError("steps must be positive")
    h = T / steps
    if h * max(A, B) > 1:
        raise ValueError("fixed comparison steps require h*max(A,B)<=1")
    matrix = ((Q(0), A), (B, Q(0)))
    endpoints = [(U0, V0)]
    for _ in range(steps):
        endpoints.append(comparison_flow_upper(matrix, endpoints[-1], h))
    return TwoBlockErrorBound(A, B, T, steps, tuple(endpoints))


def _positive_weight(value):
    value = _exact(value)
    if value <= 0:
        raise ValueError("observation weight must be positive")
    return value


def connected_marked_error_squared(U, V, w) -> Q:
    """Bound squared error in (delta, sqrt(w)*r) from full nodal sup errors.

    Both marked endpoints may move, hence each difference error is at most
    twice its supplied block bound. No interval-grid rounding or square root
    is needed for this exact rational squared bound.
    """
    U, V = _nonnegative(U, "U"), _nonnegative(V, "V")
    w = _positive_weight(w)
    return 4 * (V * V + w * U * U)


def cut_marked_error_squared(Uhost, Vhost, form_child, phase_child, w) -> Q:
    """Retain host errors and both uncertain frozen-child coordinates.

    The isolated child is constant along each supplied member's cut flow;
    it need not equal the nominal child's initialization. These separate
    nonnegative bounds therefore remain in the marked difference errors.
    """
    Uhost, Vhost, form_child, phase_child = (
        _nonnegative(value, label)
        for value, label in (
            (Uhost, "Uhost"),
            (Vhost, "Vhost"),
            (form_child, "form_child"),
            (phase_child, "phase_child"),
        )
    )
    w = _positive_weight(w)
    return (Vhost + phase_child) ** 2 + w * (Uhost + form_child) ** 2
