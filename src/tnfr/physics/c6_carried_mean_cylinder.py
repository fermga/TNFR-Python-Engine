"""Obstruction to a carried C6 energy tube with an independent mean interval.

Uniform translations inside exact rounding cells preserve displayed EPI,
canonical pressure and centered energy. Opposite mean-pressure signs then
supply outward witnesses at either mean boundary. These are hypothetical
states in the candidate class, not continuations of a saved trajectory.
"""

from dataclasses import dataclass
from fractions import Fraction as F

from ..dynamics._euler_kernel import (
    NODAL_REMAINDER_DENOMINATOR_BITS,
    NodalRemainderState,
    NodalRemainderStep,
    _validate_nodal_remainder_state,
    advance_nodal_remainder,
)
from ._cycle_algebra import Vector
from .c6_carried_balance import C6CarriedPressurePoint, _closure, _point
from .c6_carried_closure import C6CarriedClosure
from .nodal_remainder import derive_nodal_remainder_itinerary

__all__ = [
    "C6CarriedMeanCylinderObstruction",
    "derive_c6_carried_mean_cylinder_obstruction",
    "C6CarriedMeanCylinderBoundary",
    "C6CarriedMeanCylinderEscape",
    "observe_c6_carried_mean_cylinder_escape",
]


def _mean(values):
    return sum(values, F(0)) / len(values)


def _translation_window(point, grid):
    """Intersect the six exact legal-cell intervals for a common grid shift."""
    state = point.state
    exact = _validate_nodal_remainder_state(state)
    indices = tuple(value / grid for value in exact)
    if any(value.denominator != 1 for value in indices):
        raise RuntimeError(
            "the canonical reconstructed state left the shared dyadic grid"
        )
    itinerary = derive_nodal_remainder_itinerary(
        epi_states=(state.epi, state.epi),
        timesteps=(0.0,),
        capacities=((1.0,) * 6,),
        pressures=(point.observation.pressure,),
        epi_lower=state.epi_lower,
        epi_upper=state.epi_upper,
    )
    if not itinerary.feasible:
        raise RuntimeError(
            "a valid static point must belong to its zero-area rounding itinerary"
        )
    lower = max(
        cell.first_grid_index - value.numerator
        for cell, value in zip(itinerary.coordinates, indices, strict=True)
    )
    upper = min(
        cell.last_grid_index - value.numerator
        for cell, value in zip(itinerary.coordinates, indices, strict=True)
    )
    if not lower <= 0 <= upper:
        raise RuntimeError(
            "the original valid encoding is absent from its translation window"
        )
    return lower, upper


@dataclass(frozen=True, slots=True)
class C6CarriedMeanCylinderObstruction:
    """Opposite static mean increments exclude independent mean confinement.

    For every real interval [a,b] inside [mean_lower,mean_upper] containing
    the origin mean, the class E<=energy_bound with mean(X) in [a,b]
    has hypothetical outward witnesses. The interval endpoints need not
    lie on the encoding grid. Other correlations between mean, shape and
    carry can exclude these witnesses and are outside this obstruction.
    """

    closure: C6CarriedClosure
    positive_point: C6CarriedPressurePoint
    negative_point: C6CarriedPressurePoint
    grid_quantum: F
    translation_grid_lower: int
    translation_grid_upper: int
    mean_lower: F
    mean_upper: F
    positive_mean_increment: F
    negative_mean_increment: F

    @property
    def invariance_excluded(self) -> bool:
        return True

    @property
    def saved_trajectory_escape_certified(self) -> bool:
        return False

    @property
    def future_runtime_certified(self) -> bool:
        return False

    @property
    def general_correlated_region_excluded(self) -> bool:
        return False


def derive_c6_carried_mean_cylinder_obstruction(
    closure: C6CarriedClosure,
    *,
    positive_state: NodalRemainderState,
    negative_state: NodalRemainderState,
) -> C6CarriedMeanCylinderObstruction:
    """Certify a translation window and opposing canonical mean increments.

    Rebuild the spatial envelope and both supplied static points. Their
    reconstructed arithmetic means must equal the rebuilt origin mean
    mu0. At the shared encoding quantum g=2^-3222, require h*mean(p)>g
    at the positive point and h*mean(p)<-g at the negative point.

    For either point X and common integer shift k, zero-area inverse
    itinerary cells give the exact admissible inequalities
      first_i-X_i/g <= k <= last_i-X_i/g.
    Intersect all coordinates and both points. Every retained shift leaves
    displayed EPI and pressure fixed and sends mean(X) to mu0+k*g while
    preserving centered energy. The resulting interval always contains
    mu0. No sampled path, temporal mixture or newly fitted coefficient
    enters the argument.
    """
    bound = _closure(closure)
    positive, negative = _point(bound, positive_state), _point(bound, negative_state)
    origin_mean = bound.base_tube.initial_mean
    means = tuple(
        _mean(_validate_nodal_remainder_state(point.state))
        for point in (positive, negative)
    )
    if means != (origin_mean, origin_mean):
        raise ValueError(
            "both static points must have the rebuilt closure's exact origin mean"
        )
    grid = F(1, 2**NODAL_REMAINDER_DENOMINATOR_BITS)
    h = F(bound.base_tube.contraction.timestep)
    increments = tuple(
        h * _mean(tuple(map(F, point.observation.pressure)))
        for point in (positive, negative)
    )
    if not increments[0] > grid or not increments[1] < -grid:
        raise ValueError(
            "the supplied canonical mean increments must be greater than g and less than -g"
        )
    positive_window, negative_window = (
        _translation_window(point, grid) for point in (positive, negative)
    )
    lower = max(positive_window[0], negative_window[0])
    upper = min(positive_window[1], negative_window[1])
    if not lower <= 0 <= upper:
        raise RuntimeError(
            "the common translation window lost its two original witnesses"
        )
    return C6CarriedMeanCylinderObstruction(
        bound,
        positive,
        negative,
        grid,
        lower,
        upper,
        origin_mean + lower * grid,
        origin_mean + upper * grid,
        increments[0],
        increments[1],
    )


@dataclass(frozen=True, slots=True)
class C6CarriedMeanCylinderBoundary:
    """One legal translated input whose exact next mean crosses its boundary."""

    direction: str
    state: NodalRemainderState
    translation_grid_index: int
    translation: F
    mean_pressure: F
    mean_before: F
    mean_after: F
    exact_increment: Vector
    exact_candidate: Vector
    energy: F
    band_failure: bool
    step: NodalRemainderStep | None


@dataclass(frozen=True, slots=True)
class C6CarriedMeanCylinderEscape:
    """Two hypothetical outward points for one supplied independent mean interval.

    A band failure records the formal exact candidate and does not run
    an inadmissible shared-kernel step. Otherwise the complete supplied
    nodal step is replayed, with retained carry and canonical pressure.
    """

    obstruction: C6CarriedMeanCylinderObstruction
    mean_lower: F
    mean_upper: F
    upper_boundary: C6CarriedMeanCylinderBoundary
    lower_boundary: C6CarriedMeanCylinderBoundary

    @property
    def invariance_excluded(self) -> bool:
        return True

    @property
    def saved_trajectory_escape_certified(self) -> bool:
        return False

    @property
    def future_runtime_certified(self) -> bool:
        return False

    @property
    def general_correlated_region_excluded(self) -> bool:
        return False


def _boundary(bound, point, *, direction, lower, upper):
    origin_mean, grid = bound.closure.base_tube.initial_mean, bound.grid_quantum
    target = upper if direction == "upper" else lower
    scaled = (target - origin_mean) / grid
    index = scaled.__floor__() if direction == "upper" else scaled.__ceil__()
    if not bound.translation_grid_lower <= index <= bound.translation_grid_upper:
        raise RuntimeError(
            "the selected mean-boundary shift left the certified translation window"
        )
    shift = index * grid
    source = point.state
    state = NodalRemainderState(
        source.epi,
        tuple(value + shift for value in source.remainder),
        source.epi_lower,
        source.epi_upper,
    )
    translated = _point(bound.closure, state)
    exact = _validate_nodal_remainder_state(translated.state)
    before = _mean(exact)
    if (
        translated.energy != point.energy
        or translated.observation.pressure != point.observation.pressure
        or before != origin_mean + shift
        or not lower <= before <= upper
    ):
        raise RuntimeError(
            "uniform translation lost its pressure, centered energy or interval membership"
        )
    gap = upper - before if direction == "upper" else before - lower
    if not 0 <= gap < grid:
        raise RuntimeError(
            "the chosen lattice mean is not within one grid quantum of its boundary"
        )
    pressure = tuple(map(F, translated.observation.pressure))
    h = F(bound.closure.base_tube.contraction.timestep)
    added = tuple(h * value for value in pressure)
    candidate = tuple(
        value + increment for value, increment in zip(exact, added, strict=True)
    )
    after = _mean(candidate)
    if (direction == "upper" and not after > upper) or (
        direction == "lower" and not after < lower
    ):
        raise RuntimeError(
            "the exact nodal mean increment failed its strict outward crossing"
        )
    failure = any(
        not F(state.epi_lower) <= value <= F(state.epi_upper) for value in candidate
    )
    step = None
    if not failure:
        step = advance_nodal_remainder(
            translated.state,
            timestep=bound.closure.base_tube.contraction.timestep,
            capacity=(1.0,) * 6,
            pressure=translated.observation.pressure,
        )
        if (
            _validate_nodal_remainder_state(step.after) != candidate
            or step.exact_increment != added
            or any(step.nodal_balance_residual)
        ):
            raise RuntimeError(
                "the outward candidate differs from its canonical carried replay"
            )
    return C6CarriedMeanCylinderBoundary(
        direction,
        translated.state,
        index,
        shift,
        _mean(pressure),
        before,
        after,
        added,
        candidate,
        translated.energy,
        failure,
        step,
    )


def observe_c6_carried_mean_cylinder_escape(
    obstruction: C6CarriedMeanCylinderObstruction,
    *,
    mean_lower: F,
    mean_upper: F,
) -> C6CarriedMeanCylinderEscape:
    """Exhibit outward points for an exact interval containing the origin mean.

    Rebuild all public caches. At the upper boundary translate the positive
    template by floor((b-mu0)/g)*g; at the lower boundary translate the
    negative template by ceil((a-mu0)/g)*g. Both translated means belong
    to [a,b] and lie less than g from their respective boundary. The signed
    increments exceed that possible gap, proving strict escape. This also
    covers [mu0,mu0] and non-grid rational interval endpoints. No witness
    is claimed to lie on a saved or future live trajectory.
    """
    if type(obstruction) is not C6CarriedMeanCylinderObstruction:
        raise TypeError("obstruction must be a C6CarriedMeanCylinderObstruction")
    if type(mean_lower) is not F or type(mean_upper) is not F:
        raise TypeError("mean interval endpoints must be exact Fraction values")
    bound = derive_c6_carried_mean_cylinder_obstruction(
        obstruction.closure,
        positive_state=obstruction.positive_point.state,
        negative_state=obstruction.negative_point.state,
    )
    origin_mean = bound.closure.base_tube.initial_mean
    if (
        not bound.mean_lower
        <= mean_lower
        <= origin_mean
        <= mean_upper
        <= bound.mean_upper
    ):
        raise ValueError(
            "the mean interval must contain the origin mean and lie inside the rebuilt window"
        )
    positive = _boundary(
        bound,
        bound.positive_point,
        direction="upper",
        lower=mean_lower,
        upper=mean_upper,
    )
    negative = _boundary(
        bound,
        bound.negative_point,
        direction="lower",
        lower=mean_lower,
        upper=mean_upper,
    )
    return C6CarriedMeanCylinderEscape(
        bound, mean_lower, mean_upper, positive, negative
    )
