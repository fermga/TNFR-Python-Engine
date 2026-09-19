"""Mean-boundary witnesses in the fixed carried C6 affine increment class.

The coordinate spacings are exact gcds of the canonical nodal increments
over the rebuilt necessary gradient intervals. They restrict the earlier
energy/mean cylinder without assuming that every member is reachable.
"""

from dataclasses import dataclass
from fractions import Fraction as F
from math import gcd, lcm

from ..dynamics._euler_kernel import (
    NodalRemainderState,
    NodalRemainderStep,
    _validate_nodal_remainder_state,
    advance_nodal_remainder,
)
from ._cycle_algebra import Vector
from .c6_carried_balance import _point
from .c6_carried_closure import C6CarriedClosure
from .c6_carried_mean_cylinder import (
    C6CarriedMeanCylinderObstruction,
    derive_c6_carried_mean_cylinder_obstruction,
)
from .c6_pressure_lattice import _pressure_at_gradient_index

__all__ = [
    "C6CarriedAffineMeanObstruction",
    "derive_c6_carried_affine_mean_obstruction",
    "C6CarriedAffineMeanBoundary",
    "C6CarriedAffineMeanEscape",
    "observe_c6_carried_affine_mean_escape",
]


def _mean(values):
    return sum(values, F(0)) / len(values)


def _rational_gcd(values):
    denominator = lcm(*(value.denominator for value in values))
    numerator = gcd(
        *(abs(value.numerator) * (denominator // value.denominator) for value in values)
    )
    return F(numerator, denominator)


@dataclass(frozen=True, slots=True)
class C6CarriedAffineMeanObstruction:
    """Independent mean confinement still fails inside an affine lattice.

    Under the retained fixed phase, unit capacity, timestep, band and
    gradient premises, X_i-X_initial_i remains a multiple of the stored
    coordinate spacing. This necessary condition does not identify the
    reachable subset or its temporal ordering. The obstruction concerns
    the energy tube intersected with these cosets and an independent
    mean interval containing the initial mean inside the stored window.
    """

    base_obstruction: C6CarriedMeanCylinderObstruction
    max_gradient_values: int
    gradient_value_count: int
    coordinate_spacings: Vector
    coordinate_residues: Vector
    global_spacing: F
    pivot: int
    mean_quantum: F
    carry_lower_slack: F
    carry_upper_slack: F
    lift_error_squared_bound: F
    positive_energy_bound: F
    negative_energy_bound: F
    mean_lower: F
    mean_upper: F

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


def derive_c6_carried_affine_mean_obstruction(
    closure: C6CarriedClosure,
    *,
    positive_state: NodalRemainderState,
    negative_state: NodalRemainderState,
    max_gradient_values: int = 4096,
) -> C6CarriedAffineMeanObstruction:
    """Derive an affine-coset lift from the canonical represented pressure law.

    Enumerate every necessary integer gradient in each rebuilt row and
    derive gamma_i=gcd(h*p_i(m)). The enumeration budget is a computational
    guard, not a physical coefficient, and is checked before enumeration.
    Require positive spacings and a pivot attaining g=gcd(gamma_i).
    Then every admissible carried transition preserves X_i modulo gamma_i,
    and the arithmetic mean changes by an integer multiple of q=g/6.

    Both static templates must have uniform carry and the origin mean.
    At an auxiliary target mean mu0+k*q, floor the other five carries
    into their coordinate cosets and choose the pivot to restore the
    exact mean. With G=sum(nonpivot gamma_i), the centered lift error has
    squared norm at most G^2+sum(nonpivot gamma_i^2). Young's inequality
    bounds the lifted energy by twice the template energy plus twice
    this error. Shrinking the earlier common cell window by max(gamma_i)
    below and G above keeps every lifted coordinate inside its RN cell.
    The intermediate uniform target may be rational and non-dyadic;
    only the resulting lifted state is asserted to be a legal encoding.
    """
    if type(max_gradient_values) is not int or max_gradient_values <= 0:
        raise ValueError("max_gradient_values must be a positive exact integer")
    base = derive_c6_carried_mean_cylinder_obstruction(
        closure,
        positive_state=positive_state,
        negative_state=negative_state,
    )
    bound = base.closure
    count = sum(
        upper - lower + 1
        for lower, upper in zip(
            bound.gradient_index_lower,
            bound.gradient_index_upper,
            strict=True,
        )
    )
    if count > max_gradient_values:
        raise ValueError(
            "the necessary gradient enumeration exceeds max_gradient_values"
        )
    h = F(bound.base_tube.contraction.timestep)
    reference = bound.base_tube.contraction.profile.lattice
    spacings = tuple(
        _rational_gcd(
            tuple(
                h * F(_pressure_at_gradient_index(reference, node, index))
                for index in range(lower, upper + 1)
            )
        )
        for node, (lower, upper) in enumerate(
            zip(
                bound.gradient_index_lower,
                bound.gradient_index_upper,
                strict=True,
            )
        )
    )
    if any(value <= 0 for value in spacings):
        raise ValueError(
            "the affine lift requires six positive coordinate increment spacings"
        )
    common = _rational_gcd(spacings)
    if common not in spacings:
        raise ValueError(
            "the affine lift requires a coordinate attaining the global increment gcd"
        )
    if any((value / common).denominator != 1 for value in spacings):
        raise RuntimeError("the derived coordinate spacings lost their common divisor")
    if (common / base.grid_quantum).denominator != 1:
        raise RuntimeError(
            "the derived nodal increment gcd left the shared encoding grid"
        )
    pivot = spacings.index(common)
    exact_origin = _validate_nodal_remainder_state(bound.base_tube.state)
    residues = tuple(
        value % spacing for value, spacing in zip(exact_origin, spacings, strict=True)
    )
    points = (base.positive_point, base.negative_point)
    if any(
        any(value != point.state.remainder[0] for value in point.state.remainder)
        for point in points
    ):
        raise ValueError(
            "the affine lift requires uniform carry in each static template"
        )
    quantum = common / 6
    if (
        not base.positive_mean_increment > quantum
        or not base.negative_mean_increment < -quantum
    ):
        raise ValueError(
            "the canonical mean increments must be greater than q and less than -q"
        )
    other_spacings = tuple(
        value for node, value in enumerate(spacings) if node != pivot
    )
    lower_slack, upper_slack = max(other_spacings), sum(other_spacings, F(0))
    error = upper_slack**2 + sum((value**2 for value in other_spacings), F(0))
    energies = tuple(2 * point.energy + 2 * error for point in points)
    if any(value > bound.energy_bound for value in energies):
        raise ValueError(
            "the lifted template energy bounds exceed the rebuilt spatial envelope"
        )
    lower, upper = base.mean_lower + lower_slack, base.mean_upper - upper_slack
    if not lower <= bound.base_tube.initial_mean <= upper:
        raise ValueError(
            "the shrunken rounding-cell window must contain the origin mean"
        )
    return C6CarriedAffineMeanObstruction(
        base,
        max_gradient_values,
        count,
        spacings,
        residues,
        common,
        pivot,
        quantum,
        lower_slack,
        upper_slack,
        error,
        energies[0],
        energies[1],
        lower,
        upper,
    )


@dataclass(frozen=True, slots=True)
class C6CarriedAffineMeanBoundary:
    """One legal affine-coset input with an outward exact nodal mean step.

    carry_adjustment is the six-coordinate change from the supplied
    template carry; it need not equal the uniform mean_translation.
    """

    direction: str
    mean_grid_index: int
    mean_translation: F
    carry_adjustment: Vector
    state: NodalRemainderState
    energy: F
    mean_pressure: F
    mean_before: F
    mean_after: F
    exact_increment: Vector
    exact_candidate: Vector
    band_failure: bool
    step: NodalRemainderStep | None


@dataclass(frozen=True, slots=True)
class C6CarriedAffineMeanEscape:
    """Hypothetical outward points for one mean interval inside the affine class."""

    obstruction: C6CarriedAffineMeanObstruction
    mean_lower: F
    mean_upper: F
    upper_boundary: C6CarriedAffineMeanBoundary
    lower_boundary: C6CarriedAffineMeanBoundary

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
    closure = bound.base_obstruction.closure
    origin_mean, quantum = closure.base_tube.initial_mean, bound.mean_quantum
    target = upper if direction == "upper" else lower
    scaled = (target - origin_mean) / quantum
    index = scaled.__floor__() if direction == "upper" else scaled.__ceil__()
    shift = index * quantum
    source = point.state
    common_carry = source.remainder[0] + shift
    carry = [F(0)] * 6
    for node, spacing in enumerate(bound.coordinate_spacings):
        if node != bound.pivot:
            # Residues describe reconstructed coordinates, so subtract the
            # represented visible coordinate before flooring the carry.
            offset = bound.coordinate_residues[node] - F(source.epi[node])
            carry[node] = (
                offset + spacing * ((common_carry - offset) / spacing).__floor__()
            )
    carry[bound.pivot] = 6 * common_carry - sum(carry, F(0))
    state = NodalRemainderState(
        source.epi, tuple(carry), source.epi_lower, source.epi_upper
    )
    lifted = _point(closure, state)
    exact = _validate_nodal_remainder_state(lifted.state)
    origin = _validate_nodal_remainder_state(closure.base_tube.state)
    if any(
        ((value - initial) / spacing).denominator != 1
        for value, initial, spacing in zip(
            exact,
            origin,
            bound.coordinate_spacings,
            strict=True,
        )
    ):
        raise RuntimeError("the lifted state left an origin coordinate coset")
    before = _mean(exact)
    if (
        lifted.observation.pressure != point.observation.pressure
        or before != origin_mean + shift
        or not lower <= before <= upper
    ):
        raise RuntimeError(
            "the affine lift lost its pressure, exact target mean or interval membership"
        )
    errors = tuple(value - common_carry for value in carry)
    energy_bound = (
        bound.positive_energy_bound
        if direction == "upper"
        else bound.negative_energy_bound
    )
    if (
        sum(errors, F(0)) != 0
        or sum((value**2 for value in errors), F(0)) > bound.lift_error_squared_bound
        or lifted.energy > energy_bound
    ):
        raise RuntimeError("the affine lift failed its centered-error or energy bound")
    gap = upper - before if direction == "upper" else before - lower
    if not 0 <= gap < quantum:
        raise RuntimeError(
            "the chosen affine mean is not within one quantum of its boundary"
        )
    if any(
        not first <= value <= last
        for value, first, last in zip(
            lifted.observation.gradient_indices,
            closure.gradient_index_lower,
            closure.gradient_index_upper,
            strict=True,
        )
    ):
        raise RuntimeError(
            "the lifted pressure left the enumerated necessary gradient intervals"
        )
    pressure = tuple(map(F, lifted.observation.pressure))
    h = F(closure.base_tube.contraction.timestep)
    added = tuple(h * value for value in pressure)
    if any(
        (value / spacing).denominator != 1
        for value, spacing in zip(
            added,
            bound.coordinate_spacings,
            strict=True,
        )
    ):
        raise RuntimeError(
            "the canonical nodal increment left its derived coordinate spacing"
        )
    candidate = tuple(
        value + increment for value, increment in zip(exact, added, strict=True)
    )
    after = _mean(candidate)
    if (direction == "upper" and not after > upper) or (
        direction == "lower" and not after < lower
    ):
        raise RuntimeError(
            "the canonical mean increment failed its strict outward crossing"
        )
    failure = any(
        not F(state.epi_lower) <= value <= F(state.epi_upper) for value in candidate
    )
    step = None
    if not failure:
        step = advance_nodal_remainder(
            lifted.state,
            timestep=closure.base_tube.contraction.timestep,
            capacity=(1.0,) * 6,
            pressure=lifted.observation.pressure,
        )
        if (
            _validate_nodal_remainder_state(step.after) != candidate
            or step.exact_increment != added
            or any(step.nodal_balance_residual)
        ):
            raise RuntimeError(
                "the outward candidate differs from its canonical carried replay"
            )
    adjustment = tuple(
        value - initial for value, initial in zip(carry, source.remainder, strict=True)
    )
    return C6CarriedAffineMeanBoundary(
        direction,
        index,
        shift,
        adjustment,
        lifted.state,
        lifted.energy,
        _mean(pressure),
        before,
        after,
        added,
        candidate,
        failure,
        step,
    )


def observe_c6_carried_affine_mean_escape(
    obstruction: C6CarriedAffineMeanObstruction,
    *,
    mean_lower: F,
    mean_upper: F,
) -> C6CarriedAffineMeanEscape:
    """Rebuild the certificate and lift two outward points at exact mean boundaries.

    The interval may have arbitrary rational endpoints or be the singleton
    origin mean. Choose the upper mean by floor((b-mu0)/q) and the lower
    by ceil((a-mu0)/q). The respective signed increments exceed the gaps,
    while the coset lift preserves pressure and the proven energy bound.
    A band failure retains the formal exact candidate instead of invoking
    an inadmissible nodal step. No actual saved-trajectory escape follows.
    """
    if type(obstruction) is not C6CarriedAffineMeanObstruction:
        raise TypeError("obstruction must be a C6CarriedAffineMeanObstruction")
    if type(mean_lower) is not F or type(mean_upper) is not F:
        raise TypeError("mean interval endpoints must be exact Fraction values")
    base = obstruction.base_obstruction
    if type(base) is not C6CarriedMeanCylinderObstruction:
        raise TypeError("base_obstruction must be a C6CarriedMeanCylinderObstruction")
    bound = derive_c6_carried_affine_mean_obstruction(
        base.closure,
        positive_state=base.positive_point.state,
        negative_state=base.negative_point.state,
        max_gradient_values=obstruction.max_gradient_values,
    )
    origin_mean = bound.base_obstruction.closure.base_tube.initial_mean
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
        bound.base_obstruction.positive_point,
        direction="upper",
        lower=mean_lower,
        upper=mean_upper,
    )
    negative = _boundary(
        bound,
        bound.base_obstruction.negative_point,
        direction="lower",
        lower=mean_lower,
        upper=mean_upper,
    )
    return C6CarriedAffineMeanEscape(bound, mean_lower, mean_upper, positive, negative)
