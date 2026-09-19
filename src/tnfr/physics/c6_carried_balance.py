"""Exact static pressure compensation inside a carried C6 spatial envelope.

The coefficients below are algebraic witnesses, not probabilities, a mixing
law or additional TNFR parameters. A convex pressure balance does not supply
a temporal itinerary, a reachable graph state or an invariant trajectory.
"""

from dataclasses import dataclass
from fractions import Fraction as F

from ..dynamics._euler_kernel import (
    NodalRemainderState,
    _validate_nodal_remainder_state,
)
from ._cycle_algebra import Vector, dot
from ._exact_linear_algebra import exact_matrix_inverse
from .c6_carried_closure import C6CarriedClosure, derive_c6_carried_closure
from .c6_pressure_lattice import (
    C6PressureLatticeObservation,
    _observe_rebuilt_c6_pressure_lattice,
)
from .forced_support import observe_forced_support_pattern

__all__ = [
    "C6CarriedPressurePoint",
    "observe_c6_carried_pressure_point",
    "C6CarriedPressureBalance",
    "derive_c6_carried_pressure_balance",
]


def _closure(closure):
    if type(closure) is not C6CarriedClosure:
        raise TypeError("closure must be a C6CarriedClosure")
    tube = closure.base_tube
    return derive_c6_carried_closure(
        tube.contraction.profile,
        state=tube.state,
        timestep=tube.contraction.timestep,
    )


@dataclass(frozen=True, slots=True)
class C6CarriedPressurePoint:
    """One static encoded point admitted by the spatial envelope.

    This point need not occur on the trajectory from the envelope's initial
    state. Its pressure is freshly bound to the existing CPU source, and its
    reconstructed error is checked against the rebuilt energy bound.
    """

    state: NodalRemainderState
    observation: C6PressureLatticeObservation
    relative_error: Vector
    energy: F

    @property
    def reachable_from_initial_state_certified(self) -> bool:
        return False


def _point(closure, state):
    tube = closure.base_tube
    exact = _validate_nodal_remainder_state(state)
    if (state.epi_lower, state.epi_upper) != (
        tube.state.epi_lower,
        tube.state.epi_upper,
    ):
        raise ValueError(
            "the static state's declared band must equal the rebuilt closure band"
        )
    state = NodalRemainderState(
        state.epi, state.remainder, state.epi_lower, state.epi_upper
    )
    profile = tube.contraction.profile
    observation = _observe_rebuilt_c6_pressure_lattice(profile.lattice, state.epi)
    pattern = observe_forced_support_pattern(
        profile.forced_balance,
        nodes=tuple(range(6)),
        epi=exact,
    )
    energy = dot(pattern.relative_error, pattern.relative_error)
    if energy > closure.energy_bound:
        raise ValueError(
            "the static point lies outside the rebuilt centered energy envelope"
        )
    return C6CarriedPressurePoint(state, observation, pattern.relative_error, energy)


def observe_c6_carried_pressure_point(
    closure, *, state: NodalRemainderState
) -> C6CarriedPressurePoint:
    """Validate one detached supplied point without modifying the saved state."""
    return _point(_closure(closure), state)


@dataclass(frozen=True, slots=True)
class C6CarriedPressureBalance:
    """Seven admitted pressure vectors with an exact positive convex balance.

    For any fixed linear functional c, a strict common sign of c dot p on
    the entire admitted class would contradict sum(weight_j*p_j)=0 with
    every weight positive. This refutes that class-wide proof strategy.
    It neither rules out a separator on a smaller reachable subset nor
    supplies the transition ordering and carry compatibility needed for
    actual temporal compensation. No state transition is executed here.
    """

    closure: C6CarriedClosure
    points: tuple[C6CarriedPressurePoint, ...]
    weights: Vector
    weight_sum: F
    pressure_balance: Vector
    reconstructed_means: Vector
    common_reconstructed_mean: F | None
    on_origin_mean_slice: bool

    @property
    def class_wide_strict_linear_drift_excluded(self) -> bool:
        return True

    @property
    def temporal_compensation_certified(self) -> bool:
        return False

    @property
    def bounded_trajectory_certified(self) -> bool:
        return False

    @property
    def future_runtime_certified(self) -> bool:
        return False


def derive_c6_carried_pressure_balance(
    closure,
    *,
    states: tuple[NodalRemainderState, ...],
) -> C6CarriedPressureBalance:
    """Derive and verify seven exact algebraic compensation coefficients.

    Rebuild the enclosure and every pressure from primitive inputs. Invert
    the rational matrix whose columns are (p_j,1), using the shared exact
    linear-algebra owner, and solve M*weight=(0,0,0,0,0,0,1). Singular
    point families and nonpositive weights are rejected. A common exact
    reconstructed mean is recorded separately, including whether it equals
    the closure's initial mean. No carry is reset or inferred. Float optimizers,
    tolerances and cached pressure or energy values do not enter admission.
    """
    bound = _closure(closure)
    if type(states) is not tuple or len(states) != 7:
        raise ValueError(
            "the affine pressure witness requires exactly seven ordered states"
        )
    points = tuple(_point(bound, state) for state in states)
    pressures = tuple(
        tuple(F(value) for value in point.observation.pressure) for point in points
    )
    matrix = tuple(tuple(row[i] for row in pressures) for i in range(6)) + (
        (F(1),) * 7,
    )
    inverse = exact_matrix_inverse(matrix)
    weights = tuple(row[-1] for row in inverse)
    if any(value <= 0 for value in weights):
        raise ValueError(
            "the seven canonical pressures must admit strictly positive exact weights"
        )
    total = sum(weights, F(0))
    balance = tuple(
        sum(
            (weight * row[i] for weight, row in zip(weights, pressures, strict=True)),
            F(0),
        )
        for i in range(6)
    )
    if total != 1 or any(balance):
        raise RuntimeError(
            "the exact affine inverse lost its normalized pressure balance"
        )
    means = tuple(
        sum(_validate_nodal_remainder_state(point.state), F(0)) / 6 for point in points
    )
    common = means[0] if all(value == means[0] for value in means) else None
    matches = common is not None and common == bound.base_tube.initial_mean
    return C6CarriedPressureBalance(
        bound, points, weights, total, balance, means, common, matches
    )
