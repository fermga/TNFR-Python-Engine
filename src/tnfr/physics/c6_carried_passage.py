"""Finite nonpositive-pressure passage from a carried C6 spatial tube.

The exact nodal increment and a separately controlled mean turn bounded
centered shape into a first-passage theorem. This is a conditional result
for the fixed numerical map; no future live operator admission is inferred.
"""

import math
import sys
from dataclasses import dataclass
from fractions import Fraction as F

from ._cycle_algebra import Vector, laplacian_action
from .c6_carried_tube import (
    C6CarriedBandHorizon,
    C6CarriedTube,
    _tube,
    derive_c6_carried_band_horizon,
)
from .c6_pressure_lattice import (
    C6PressureSignSector,
    derive_c6_pressure_sign_sector,
    observe_c6_pressure_lattice,
)

__all__ = [
    "C6CarriedPositivePressurePassage",
    "derive_c6_carried_positive_pressure_passage",
]


def _ceil_sqrt(value: F) -> int:
    """Return the least nonnegative integer whose square encloses value."""
    root = math.isqrt(value.numerator // value.denominator)
    return root + (root * root < value)


@dataclass(frozen=True, slots=True)
class C6CarriedPositivePressurePassage:
    """A finite sign-hit guarantee, or an explicit abstention.

    If ``sign_hit_certified`` is true, some pressure readout with index at
    most ``latest_pressure_index`` is nonpositive. Index zero is the
    supplied initial state. The contradiction horizon counts transitions;
    it must fit the independent sufficient band horizon before promotion.
    Neither the first actual hit nor complete-runtime provenance is given.
    """

    tube: C6CarriedTube
    band_horizon: C6CarriedBandHorizon
    positive_sector: C6PressureSignSector
    node: int
    initial_pressure: F
    gradient_bounds: Vector
    product_error_bounds: Vector
    assembly_error_bounds: Vector
    rounding_bounds: Vector
    mean_pressure_upper: F
    minimum_positive_index: int | None
    minimum_positive_pressure: F | None
    centered_increment_lower: F | None
    centered_coordinate_squared_bound: F
    contradiction_steps: int | None
    latest_pressure_index: int | None
    band_covers_contradiction: bool
    sign_hit_certified: bool
    abstention_reason: str | None

    @property
    def graph_provenance_certified(self) -> bool:
        return False

    @property
    def future_runtime_certified(self) -> bool:
        return False

    @property
    def infinite_trapping_certified(self) -> bool:
        return False


def derive_c6_carried_positive_pressure_passage(
    tube, *, node: int
) -> C6CarriedPositivePressurePassage:
    """Force a nonpositive pressure readout using an exact finite budget.

    With X=x+r and y=P*X-z, the carried equation gives
    y_i(next)-y_i=h*(p_i-mean(p)). For the rebuilt tube,
    |L*x| <= |L*z|+sqrt(3*E_bound/2)+2*R. An outward rational square-root
    enclosure yields rowwise IEEE error bounds for the existing pressure
    product and assembly. Their means bound mean(p) from above; the carry
    feedback has exactly zero mean and is not charged to this budget.

    The shared sign-sector owner clips the positive integer-gradient
    sector to the declared slab and supplies its exact monotone pressure
    lower bound. Its pressure p_min is a lower bound even when that integer
    is not jointly realizable by a complete EPI tuple. It is not a newly
    prescribed pressure law or a fabricated graph observation.

    If v=h*(p_min-mean_pressure_upper)>0, sustained positive pressure
    would imply y_i(n)>=y_i(0)+n*v. The least integer N whose right side
    is positive and its square exceeds 5*E_bound/6 contradicts the tube.
    When the independent band bootstrap covers N transitions, some
    pressure at index j in [0,N-1] must be nonpositive. Otherwise this
    observer abstains rather than declaring a band exit or a sign hit.
    """
    if type(node) is not int or not 0 <= node < 6:
        raise ValueError("node must be an integer C6 index")
    bound = _tube(tube)
    band = derive_c6_carried_band_horizon(bound)
    ref = bound.contraction.profile
    lattice = ref.lattice
    initial = observe_c6_pressure_lattice(lattice, epi=bound.state.epi)
    initial_pressure = F(initial.pressure[node])
    quantum = lattice.epi_quantum
    laplacian_radius = _ceil_sqrt(F(3, 2) * bound.energy_bound / quantum**2) * quantum
    gradients = tuple(
        abs(value) + laplacian_radius + 2 * bound.carry_bound
        for value in laplacian_action(ref.forced_balance.relative_profile)
    )
    weight = ref.forced_balance.epi_weight
    unit, half_subnormal = F(1, 2**53), F(1, 2**1075)
    products = tuple(weight * value for value in gradients)
    product_errors = tuple(unit * value + half_subnormal for value in products)
    arguments = tuple(
        abs(source) + product + error
        for source, product, error in zip(
            ref.forced_balance.forcing, products, product_errors, strict=True
        )
    )
    if max(products + arguments) > F(sys.float_info.max):
        raise ValueError(
            "the centered pressure envelope cannot certify finite binary64 operations"
        )
    assembly_errors = tuple(unit * value + half_subnormal for value in arguments)
    errors = tuple(a + b for a, b in zip(product_errors, assembly_errors, strict=True))
    mean_upper = (sum(ref.forced_balance.forcing, F(0)) + sum(errors, F(0))) / 6
    squared = F(5, 6) * bound.energy_bound
    sector = derive_c6_pressure_sign_sector(lattice, node=node, sign=1)
    positive_index = sector.bounding_gradient_index
    minimum = sector.signed_pressure_margin
    increment = (
        None
        if minimum is None
        else F(bound.contraction.timestep) * (minimum - mean_upper)
    )
    steps = latest = None
    covers = hit = False
    reason = None
    if initial_pressure <= 0:
        steps = latest = 0
        covers = hit = True
    elif increment is None:
        raise RuntimeError(
            "an actual positive pressure contradicts the relaxed lattice range"
        )
    elif increment <= 0:
        reason = "the centered increment has no strictly positive lower bound"
    else:
        y0 = bound.initial_error[node]

        def contradicts(n):
            value = y0 + n * increment
            return value > 0 and value * value > squared

        coordinate_radius = _ceil_sqrt(squared / quantum**2) * quantum
        low, high = 0, (coordinate_radius - y0) // increment + 1
        if high < 1 or contradicts(0) or not contradicts(high):
            raise RuntimeError(
                "the initial tube or finite passage bracket is inconsistent"
            )
        while low + 1 < high:
            middle = (low + high) // 2
            if contradicts(middle):
                high = middle
            else:
                low = middle
        steps = high
        covers = band.tube_initially_admitted and (
            band.unbounded_conditional_prefix
            or band.maximum_steps is not None
            and steps <= band.maximum_steps
        )
        if covers:
            latest, hit = steps - 1, True
        else:
            reason = "the independent sufficient band horizon does not cover the contradiction"
    return C6CarriedPositivePressurePassage(
        bound,
        band,
        sector,
        node,
        initial_pressure,
        gradients,
        product_errors,
        assembly_errors,
        errors,
        mean_upper,
        positive_index,
        minimum,
        increment,
        squared,
        steps,
        latest,
        covers,
        hit,
        reason,
    )
