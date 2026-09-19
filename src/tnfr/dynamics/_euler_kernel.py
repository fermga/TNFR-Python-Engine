"""Shared arithmetic order for an unclipped explicit nodal Euler update.

The scalar and array callers supply already resolved rates. Multiplication
and addition remain separate operations; capacity, pressure, Gamma, clipping
and derivative/history writes belong to the surrounding integrator. This
owner supplies neither a pressure law nor a backend accuracy guarantee.

An explicit remainder-carrying alternative retains the exact dyadic nodal
area between calls. It is separate from the existing Euler function and
does not change its production callers or introduce another physical field.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from .._binary64 import uses_ieee_binary64_rounding

__all__ = [
    "euler_update",
    "NODAL_REMAINDER_DENOMINATOR_BITS",
    "NODAL_REMAINDER_ABSOLUTE_BOUND",
    "NodalRemainderState",
    "NodalRemainderStep",
    "initialize_nodal_remainder",
    "advance_nodal_remainder",
]

# Each of the three represented factors has dyadic denominator at most
# 2**1074. These representation limits are not structural parameters.
NODAL_REMAINDER_DENOMINATOR_BITS = 3 * 1074
NODAL_REMAINDER_ABSOLUTE_BOUND = Fraction(1, 2**53)


def euler_update(epi: Any, dt_step: float, rate: Any) -> Any:
    """Apply ``epi + dt_step * rate`` in the production operation order."""
    increment = dt_step * rate
    return epi + increment


def _finite_binary64(value, label):
    if type(value) is not float:
        raise TypeError(f"{label} must be an actual binary64 float")
    if not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    return value


def _binary64_tuple(values, label):
    if type(values) is not tuple:
        raise TypeError(f"{label} must be an ordered tuple of binary64 floats")
    if not values:
        raise ValueError(f"{label} must be nonempty")
    return tuple(
        _finite_binary64(value, f"{label}[{i}]") for i, value in enumerate(values)
    )


def _remainder_band(lower, upper):
    lower = _finite_binary64(lower, "epi_lower")
    upper = _finite_binary64(upper, "epi_upper")
    if not 0 < lower <= upper <= 1:
        raise ValueError("the EPI band must satisfy 0 < lower <= upper <= 1")
    return lower, upper


def _require_remainder_rounding():
    if not uses_ieee_binary64_rounding():
        raise RuntimeError(
            "nodal remainder updates require the declared IEEE binary64 rounding behavior"
        )


@dataclass(frozen=True)
class NodalRemainderState:
    """Numerical encoding X=Fraction(epi)+remainder, with epi=RN(X).

    Both X and its displayed float lie in the declared positive band. The
    exact dyadic remainder is solver metadata derived by the update, not
    an independently chosen TNFR field. Initialization sets it to zero;
    continuation validates the complete public encoding before using it.
    This detached value does not authenticate a graph or event history.
    """

    epi: tuple[float, ...]
    remainder: tuple[Fraction, ...]
    epi_lower: float
    epi_upper: float

    @property
    def exact_epi(self) -> tuple[Fraction, ...]:
        """Reconstruct the exact coordinates after validating their encoding."""
        return _validate_nodal_remainder_state(self)


def _validate_nodal_remainder_state(state):
    _require_remainder_rounding()
    if not isinstance(state, NodalRemainderState):
        raise TypeError("state must be a NodalRemainderState")
    epi = _binary64_tuple(state.epi, "state.epi")
    lower, upper = _remainder_band(state.epi_lower, state.epi_upper)
    if type(state.remainder) is not tuple or len(state.remainder) != len(epi):
        raise ValueError("the remainder tuple must match the EPI dimensions")
    for remainder in state.remainder:
        if type(remainder) is not Fraction:
            raise TypeError("every remainder must be an exact Fraction")
        denominator = remainder.denominator
        if (
            denominator & (denominator - 1)
            or denominator.bit_length() - 1 > NODAL_REMAINDER_DENOMINATOR_BITS
        ):
            raise ValueError(
                "the remainder must use the bounded dyadic nodal representation"
            )
    if any(not lower <= value <= upper for value in epi):
        raise ValueError("visible EPI leaves the declared positive band")
    exact = tuple(
        Fraction.from_float(value) + remainder
        for value, remainder in zip(epi, state.remainder, strict=True)
    )
    exact_lower, exact_upper = Fraction.from_float(lower), Fraction.from_float(upper)
    if any(not exact_lower <= value <= exact_upper for value in exact):
        raise ValueError("reconstructed exact EPI leaves the declared positive band")
    if any(float(value) != visible for value, visible in zip(exact, epi, strict=True)):
        raise ValueError("visible EPI must be the nearest-even encoding of exact EPI")
    if any(abs(value) > NODAL_REMAINDER_ABSOLUTE_BOUND for value in state.remainder):
        raise ValueError("the remainder exceeds the positive unit-band rounding bound")
    return exact


@dataclass(frozen=True, slots=True)
class NodalRemainderStep:
    """One exact represented-input nodal area and its visible/carry split."""

    before: NodalRemainderState
    after: NodalRemainderState
    timestep: float
    capacity: tuple[float, ...]
    pressure: tuple[float, ...]
    exact_increment: tuple[Fraction, ...]
    visible_increment: tuple[Fraction, ...]
    carry_transfer: tuple[Fraction, ...]
    nodal_balance_residual: tuple[Fraction, ...]


def initialize_nodal_remainder(
    epi: tuple[float, ...],
    *,
    epi_lower: float = 0.05,
    epi_upper: float = 1.0,
) -> NodalRemainderState:
    """Start the explicit numerical representation with zero carry.

    Inputs must already be finite binary64 floats. There is no graph write,
    pressure inference, projection or clipping. The visible and exact
    initial coordinates coincide in the declared positive unit subinterval.
    """
    values = _binary64_tuple(epi, "epi")
    lower, upper = _remainder_band(epi_lower, epi_upper)
    state = NodalRemainderState(values, (Fraction(0),) * len(values), lower, upper)
    _validate_nodal_remainder_state(state)
    return state


def advance_nodal_remainder(
    state: NodalRemainderState,
    *,
    timestep: float,
    capacity: tuple[float, ...],
    pressure: tuple[float, ...],
) -> NodalRemainderStep:
    """Carry exact dyadic nodal area into the next visible EPI update.

    For finite represented h>=0, nu_i>=0 and p_i, form exactly
    Z_i=Fraction(x_i)+r_i+Fraction(h)*Fraction(nu_i)*Fraction(p_i).
    Return y_i=RN(Z_i) and r_i_next=Z_i-Fraction(y_i), retaining the latter
    in the next state. Both the reconstructed Z and visible y must remain
    in the declared band. Failure has no partial writes and never clips.
    Zero duration or capacity preserves the corresponding exact encoding.

    The visible nodal balance is y-x=a+r-r_next. Over any finite carried
    sequence it telescopes to x_N-x_0=sum(a)+r_0-r_N. Thus, with zero initial
    carry and zero accumulated represented nodal mean source, the visible
    arithmetic mean has error at most 2^-53 independent of the step count.
    More precise endpoint bounds follow from the actual rounding cells.
    Pressure-realization error remains in sum(a); it is not projected out.
    Under heterogeneous capacities the relevant source is mean(nu*p), not
    merely mean(p).

    Equal exact accumulated areas give identical final encodings, provided
    all intermediate states satisfy the band. This arithmetic partition
    identity does not cover pressure refreshes that change the supplied
    area, solver convergence or physical event/REMESH jumps. Resetting or
    transporting the carry needs an explicit additional balance rule.

    Three binary64 factors have denominator dividing 2**3222; addition and
    subtraction preserve that dyadic bound. The initial zero carry and all
    continuations therefore use bounded-denominator numerical metadata.
    This opt-in scalar kernel neither alters euler_update nor certifies a
    live integrator, pressure producer, graph, backend or future band.
    """
    exact_before = _validate_nodal_remainder_state(state)
    h = _finite_binary64(timestep, "timestep")
    capacities = _binary64_tuple(capacity, "capacity")
    pressures = _binary64_tuple(pressure, "pressure")
    if len(capacities) != len(exact_before) or len(pressures) != len(exact_before):
        raise ValueError("capacity and pressure tuples must match the EPI dimensions")
    if h < 0 or any(value < 0 for value in capacities):
        raise ValueError("timestep and capacity must be nonnegative")
    exact_h = Fraction.from_float(h)
    increment = tuple(
        exact_h * Fraction.from_float(nu) * Fraction.from_float(force)
        for nu, force in zip(capacities, pressures, strict=True)
    )
    exact_after = tuple(
        value + change for value, change in zip(exact_before, increment, strict=True)
    )
    lower, upper = Fraction.from_float(state.epi_lower), Fraction.from_float(
        state.epi_upper
    )
    if any(not lower <= value <= upper for value in exact_after):
        raise ValueError("exact nodal update leaves the declared positive EPI band")
    visible = tuple(float(value) for value in exact_after)
    remainder = tuple(
        value - Fraction.from_float(encoded)
        for value, encoded in zip(exact_after, visible, strict=True)
    )
    after = NodalRemainderState(visible, remainder, state.epi_lower, state.epi_upper)
    _validate_nodal_remainder_state(after)
    visible_increment = tuple(
        Fraction.from_float(final) - Fraction.from_float(initial)
        for initial, final in zip(state.epi, after.epi, strict=True)
    )
    transfer = tuple(
        before - final
        for before, final in zip(state.remainder, after.remainder, strict=True)
    )
    residual = tuple(
        actual - area - carried
        for actual, area, carried in zip(
            visible_increment, increment, transfer, strict=True
        )
    )
    if any(residual):
        raise RuntimeError(
            "nodal remainder update lost its exact visible/carry balance"
        )
    return NodalRemainderStep(
        state,
        after,
        h,
        capacities,
        pressures,
        increment,
        visible_increment,
        transfer,
        residual,
    )
