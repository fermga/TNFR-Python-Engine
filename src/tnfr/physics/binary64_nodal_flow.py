"""Exact held-flow pressure cells and a restricted paired diffusion class.

The observers reuse the scalar Euler arithmetic owner for unit capacity,
zero Gamma and four held steps of length 1/16. The separate paired C6 class
also binds the shared pure-EPI pressure reducers and proves a conditional
numeric invariant. Neither observer supplies graph provenance, a future
complete-runtime bound or a guarantee for arbitrary native numerical backends.
An inverse-cell observer identifies the maximal represented pressure box
preserving one complete four-state EPI trace of this held numeric kernel.
"""

import math
from dataclasses import dataclass
from fractions import Fraction

from .._binary64 import uses_ieee_binary64_rounding
from .._exact_time import fraction_upper_signed_float
from ..dynamics._euler_kernel import euler_update
from ..mathematics._neighbor_differences import (
    edge_mean_differences,
    mean_neighbor_difference,
)
from ..mathematics.unified_numerical import np
from ._cycle_algebra import Vector, c6_pair_sums, laplacian_action

__all__ = [
    "Binary64AdditionCell",
    "Binary64QuarterSubstep",
    "Binary64UnitQuarterFlow",
    "observe_binary64_unit_quarter_flow",
    "Binary64PairedC6Diffusion",
    "observe_binary64_paired_c6_diffusion",
    "Binary64PressureTraceCell",
    "Binary64QuarterPressureBox",
    "derive_binary64_quarter_pressure_box",
]


def _float(value, label):
    if type(value) is not float:
        raise TypeError(f"{label} must be an actual binary64 float")
    if not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    return value


def _float_tuple(values, label):
    if type(values) is not tuple:
        raise TypeError(f"{label} must be an ordered tuple of binary64 floats")
    if not values:
        raise ValueError(f"{label} must be nonempty")
    return tuple(_float(value, f"{label}[{i}]") for i, value in enumerate(values))


@dataclass(frozen=True)
class Binary64AdditionCell:
    """The exact nearest-even cell of one finite represented number.

    Both midpoint endpoints belong to the cell exactly when its represented
    significand is even. Membership refers to the supplied exact input,
    not to an approximation of that input by another float. Signed zeros
    share one numeric cell; their sign bits are not distinguished here.
    The adjacent represented values must both remain finite.
    """

    rounded_value: float
    predecessor: float
    successor: float
    lower: Fraction
    upper: Fraction
    even_significand: bool
    exact_input: Fraction
    contains_exact_input: bool
    lower_tie: bool
    upper_tie: bool


def _rounding_cell(value, exact_input):
    value = _float(value, "rounding-cell center")
    predecessor = math.nextafter(value, -math.inf)
    successor = math.nextafter(value, math.inf)
    if not math.isfinite(predecessor) or not math.isfinite(successor):
        raise ValueError("a rounding cell requires two finite adjacent values")
    center = Fraction.from_float(value)
    previous, following = Fraction.from_float(predecessor), Fraction.from_float(
        successor
    )
    lower, upper = (previous + center) / 2, (center + following) / 2
    index = center / Fraction.from_float(math.ulp(value))
    if index.denominator != 1:
        raise RuntimeError("binary64 value lost its significand lattice")
    even = index.numerator % 2 == 0
    lower_tie, upper_tie = exact_input == lower, exact_input == upper
    contains = lower < exact_input < upper or even and (lower_tie or upper_tie)
    return Binary64AdditionCell(
        value,
        predecessor,
        successor,
        lower,
        upper,
        even,
        exact_input,
        contains,
        lower_tie,
        upper_tie,
    )


@dataclass(frozen=True)
class Binary64QuarterSubstep:
    """One scalar-owner replay with independently checked addition cells."""

    ordinal: int
    before: tuple[float, ...]
    after: tuple[float, ...]
    unrounded_sum: Vector
    addition_error: Vector
    source_cells: tuple[Binary64AdditionCell, ...]
    result_cells: tuple[Binary64AdditionCell, ...]
    stasis: tuple[bool, ...]


@dataclass(frozen=True)
class Binary64UnitQuarterFlow:
    """Exact local error budget of supplied held numeric inputs.

    Half-EPI pressure-band membership is reported independently of the
    supplied EPI coordinate. It proves stasis only for source EPI exactly
    0.5. Signed mean defects need not vanish for zero-mean stored pressure.
    No finite local bound establishes a uniform signed mean-prefix bound.
    """

    epi: tuple[float, ...]
    pressure: tuple[float, ...]
    epi_lower: float
    epi_upper: float
    rate: tuple[float, ...]
    rounded_increment: tuple[float, ...]
    exact_epi: Vector
    exact_pressure: Vector
    exact_increment: Vector
    scaling_error: Vector
    substeps: tuple[Binary64QuarterSubstep, ...]
    endpoint: tuple[float, ...]
    exact_endpoint: Vector
    ideal_endpoint: Vector
    endpoint_defect: Vector
    scaling_defect: Vector
    addition_defect: Vector
    error_identity_residual: Vector
    local_endpoint_error_bound: Fraction
    addition_error_bound: Fraction
    scaling_error_bound: Fraction
    mean_before: Fraction
    mean_after: Fraction
    mean_pressure: Fraction
    mean_scaling_defect: Fraction
    mean_addition_defect: Fraction
    mean_endpoint_defect: Fraction
    mean_identity_residual: Fraction
    stasis: tuple[bool, ...]
    half_epi_pressure_band: tuple[Fraction, Fraction]
    half_epi_pressure_band_membership: tuple[bool, ...]


def observe_binary64_unit_quarter_flow(
    *,
    epi: tuple[float, ...],
    pressure: tuple[float, ...],
    epi_lower: float = 0.05,
    epi_upper: float = 1.0,
) -> Binary64UnitQuarterFlow:
    """Replay four unit-capacity, zero-Gamma held Euler steps of length 1/16.

    Every input scalar must already be a finite binary64 float. EPI inputs
    and all unprojected rounded substep outputs must lie in the declared
    [epi_lower,epi_upper], with 0<epi_lower<=epi_upper<=1. Thus the observer
    never uses clipping to make a failed update admissible. Pressure is a
    supplied held input, not inferred as the output of a canonical channel.

    The production rate is (1.0*p)+0.0 and the rounded increment is
    q=RN(p/16). Power-of-two scaling is exact unless subnormal rounding,
    whose absolute error is at most 2^-1075. Each call of the shared scalar
    euler_update must independently match the exact nearest-even addition
    cell of its result. Positive results up to one have addition error at
    most half ulp(1)=2^-53, including the wider right cell at one. Hence
    abs(x4-x0-p/4)<=2^-51+2^-1073 for every coordinate. The exact budget
    retains all four addition errors and four copies of the scaling error.

    At source x=0.5, the addition cell has increment bounds
    [-2^-55,2^-54], inclusive because its significand is even. Exactly the
    represented pressures in [-2^-51,2^-50] therefore freeze all four
    updates. At other sources use their own exact cells; binade boundaries
    invalidate a symmetric half-ulp shortcut for negative increments.

    The IEEE predicate is a scoped platform precondition, not an exhaustive
    hardware/backend conformance proof or a trigonometric accuracy bound.
    This detached scalar replay does not bind a graph, pressure-production
    path, vector backend, operator word or future repeated execution.
    """
    if not uses_ieee_binary64_rounding():
        raise RuntimeError(
            "this observation requires the declared IEEE binary64 rounding behavior"
        )
    values = _float_tuple(epi, "epi")
    held = _float_tuple(pressure, "pressure")
    lower, upper = _float(epi_lower, "epi_lower"), _float(epi_upper, "epi_upper")
    if len(values) != len(held):
        raise ValueError("EPI and pressure tuples must have matching dimensions")
    if not 0 < lower <= upper <= 1:
        raise ValueError("the EPI band must satisfy 0 < lower <= upper <= 1")
    if any(not lower <= value <= upper for value in values):
        raise ValueError("every initial EPI must lie in the declared positive band")
    initial = tuple(Fraction.from_float(value) for value in values)
    exact_pressure = tuple(Fraction.from_float(value) for value in held)
    rate = tuple(1.0 * value + 0.0 for value in held)
    exact_increment = tuple(value / 16 for value in exact_pressure)
    increment = tuple(0.0625 * value for value in rate)
    represented_increment = tuple(Fraction.from_float(value) for value in increment)
    scaling = tuple(
        actual - expected
        for actual, expected in zip(represented_increment, exact_increment, strict=True)
    )
    scaling_bound, addition_bound = Fraction(1, 2**1075), Fraction(1, 2**53)
    if any(abs(value) > scaling_bound for value in scaling):
        raise RuntimeError(
            "binary64 power-of-two scaling exceeds its exact rounding bound"
        )
    steps = []
    current = values
    for ordinal in range(1, 5):
        following = tuple(
            _float(euler_update(value, 0.0625, slope), "Euler endpoint")
            for value, slope in zip(current, rate, strict=True)
        )
        if any(not lower <= value <= upper for value in following):
            raise ValueError(
                "unclipped Euler output leaves the declared positive EPI band"
            )
        exact_sum = tuple(
            Fraction.from_float(value) + change
            for value, change in zip(current, represented_increment, strict=True)
        )
        source_cells = tuple(
            _rounding_cell(value, total)
            for value, total in zip(current, exact_sum, strict=True)
        )
        result_cells = tuple(
            _rounding_cell(value, total)
            for value, total in zip(following, exact_sum, strict=True)
        )
        errors = tuple(
            Fraction.from_float(value) - total
            for value, total in zip(following, exact_sum, strict=True)
        )
        stasis = tuple(
            before == after for before, after in zip(current, following, strict=True)
        )
        if (
            not all(cell.contains_exact_input for cell in result_cells)
            or stasis != tuple(cell.contains_exact_input for cell in source_cells)
            or any(abs(error) > addition_bound for error in errors)
        ):
            raise RuntimeError(
                "production Euler disagrees with an exact nearest-even addition cell"
            )
        steps.append(
            Binary64QuarterSubstep(
                ordinal,
                current,
                following,
                exact_sum,
                errors,
                source_cells,
                result_cells,
                stasis,
            )
        )
        current = following
    endpoint = tuple(Fraction.from_float(value) for value in current)
    ideal = tuple(
        value + force / 4 for value, force in zip(initial, exact_pressure, strict=True)
    )
    defect = tuple(
        actual - expected for actual, expected in zip(endpoint, ideal, strict=True)
    )
    scaling_defect = tuple(4 * value for value in scaling)
    addition_defect = tuple(
        sum((step.addition_error[i] for step in steps), Fraction(0))
        for i in range(len(values))
    )
    residual = tuple(
        total - scaled - added
        for total, scaled, added in zip(
            defect, scaling_defect, addition_defect, strict=True
        )
    )
    bound = 4 * (scaling_bound + addition_bound)
    if any(residual) or any(abs(value) > bound for value in defect):
        raise RuntimeError("binary64 held interval lost its exact local error budget")

    def mean(vector):
        return sum(vector, Fraction(0)) / len(vector)

    mean_before, mean_after, mean_pressure = (
        mean(initial),
        mean(endpoint),
        mean(exact_pressure),
    )
    mean_defect = mean(defect)
    mean_residual = mean_after - mean_before - mean_pressure / 4 - mean_defect
    half_band = (Fraction(-1, 2**51), Fraction(1, 2**50))
    half_membership = tuple(
        half_band[0] <= value <= half_band[1] for value in exact_pressure
    )
    stasis = tuple(
        before == after for before, after in zip(values, current, strict=True)
    )
    if mean_residual or any(
        value == 0.5 and frozen != member
        for value, frozen, member in zip(values, stasis, half_membership, strict=True)
    ):
        raise RuntimeError(
            "binary64 held interval lost its mean or half-EPI stasis identity"
        )
    return Binary64UnitQuarterFlow(
        values,
        held,
        lower,
        upper,
        rate,
        increment,
        initial,
        exact_pressure,
        exact_increment,
        scaling,
        tuple(steps),
        current,
        endpoint,
        ideal,
        defect,
        scaling_defect,
        addition_defect,
        residual,
        bound,
        addition_bound,
        scaling_bound,
        mean_before,
        mean_after,
        mean_pressure,
        mean(scaling_defect),
        mean(addition_defect),
        mean_defect,
        mean_residual,
        stasis,
        half_band,
        half_membership,
    )


@dataclass(frozen=True)
class Binary64PairedC6Diffusion:
    """One verified member of the fixed-coefficient pure-channel invariant.

    All pair sums use exact rational values of the represented coordinates.
    The repeated-class flag concerns only refreshed unit-C6 EPI pressure and
    the four shared scalar Euler updates under the declared IEEE premises.
    It neither certifies the full multichannel pressure nor guarantees
    convergence: a nonuniform represented pattern can be a numeric plateau.
    """

    epi: tuple[float, ...]
    epi_weight: float
    binade_lower: Fraction
    binade_upper: Fraction
    spacing: Fraction
    center: Fraction
    source_interval: tuple[Fraction, Fraction]
    endpoint_interval: tuple[Fraction, Fraction]
    source_pair_sums: Vector
    pressure: tuple[float, ...]
    vector_pressure: tuple[float, ...]
    ideal_pressure: Vector
    pressure_rounding_defect: Vector
    pressure_binding_residual: Vector
    pressure_pair_sums: Vector
    substep_pair_sums: tuple[Vector, ...]
    endpoint_pair_sums: Vector
    mean_drift: Fraction
    flow: Binary64UnitQuarterFlow
    interval_preserved: bool
    mean_preserved: bool
    repeated_pure_channel_invariant: bool
    nonuniform_plateau: bool
    full_runtime_certified: bool


def observe_binary64_paired_c6_diffusion(
    *,
    epi: tuple[float, ...],
    epi_weight: float,
) -> Binary64PairedC6Diffusion:
    """Bind one update to a conditional invariant paired binary64 C6 class.

    The fixed support is the ordered unit cycle 0,...,5, with opposite
    pairs (0,3), (1,4), (2,5). Inputs are actual finite floats in [0.05,1],
    strictly inside one common normal binade (B,2B). Their exact pair sum
    must be 2c, where c/spacing is an integer. The supplied represented
    coefficient satisfies 0<=epi_weight<=1; it is held fixed for the
    repeated pure-channel theorem, without substituting another default.

    In this binade, subtraction and the two-neighbor difference average
    g_i are exact. Both shared scalar and vector reducers must therefore
    return RN(epi_weight*g_i), including their mixed-sign fallback. Their
    opposite rows are sign reflections. Unit capacity and zero Gamma then
    give opposite increments q_i=RN(p_i/16).

    Nearest-even addition commutes with reflection x->2c-x on the common
    lattice because 2c/spacing is even, so midpoint parity is preserved.
    All four updates stay within the source interval [m,M]: for positive
    pressure let M-x_i=n*spacing. Monotonic rounding gives 0<=p_i<=M-x_i
    and q_i<=n*spacing/16. Each lattice addition advances at most
    floor(n/16+1/2) grid units, and 4*floor(n/16+1/2)<=n for integers n>=0.
    Negative pressure has the reflected argument. Division of these
    lattice distances by 16 is exact throughout the declared EPI band.
    Thus the initial interval, pair sums and mean are preserved under any
    finite repetition of this fixed-coefficient pressure/Euler map.

    This observer binds both pressure reducers and the B18 scalar Euler
    replay for the supplied tuple. Its class proof does not bind a live
    graph, phase channel, operator admission, changing support/capacity,
    arbitrary numerical backend or complete runtime. A binade-boundary
    center such as 0.5 and half-grid centers are outside this class.
    """
    if not uses_ieee_binary64_rounding():
        raise RuntimeError(
            "this observation requires the declared IEEE binary64 rounding behavior"
        )
    if np is None:
        raise RuntimeError("paired C6 pressure binding requires the NumPy backend")
    values = _float_tuple(epi, "epi")
    weight = _float(epi_weight, "epi_weight")
    if len(values) != 6:
        raise ValueError(
            "paired C6 diffusion requires exactly six ordered EPI coordinates"
        )
    if not 0 <= weight <= 1:
        raise ValueError("the represented EPI coefficient must lie in [0,1]")
    if any(not 0.05 <= value <= 1.0 for value in values):
        raise ValueError("paired C6 EPI must lie in the declared [0.05,1] band")
    exact = tuple(Fraction.from_float(value) for value in values)
    _, exponent = math.frexp(values[0])
    lower = Fraction.from_float(math.ldexp(1.0, exponent - 1))
    upper, spacing = 2 * lower, lower / 2**52
    if any(not lower < value < upper for value in exact):
        raise ValueError(
            "paired C6 EPI must lie strictly inside one common normal binade"
        )
    source_pairs = c6_pair_sums(exact)
    if len(set(source_pairs)) != 1:
        raise ValueError("opposite C6 nodes must have one common exact pair sum")
    center = source_pairs[0] / 2
    if (center / spacing).denominator != 1:
        raise ValueError(
            "the exact pair center must be on the common lattice, not a half-grid point"
        )

    neighbors = tuple(((i - 1) % 6, (i + 1) % 6) for i in range(6))
    pressure = tuple(
        _float(
            mean_neighbor_difference(
                values[i],
                tuple(values[j] for j in row),
                coefficient=weight,
            ),
            "scalar EPI pressure",
        )
        for i, row in enumerate(neighbors)
    )
    source = tuple(i for i in range(6) for _ in neighbors[i])
    target = tuple(j for row in neighbors for j in row)
    vector_pressure = tuple(
        float(value)
        for value in edge_mean_differences(
            values,
            source,
            target,
            coefficient=weight,
        )
    )
    rational_weight = Fraction.from_float(weight)
    ideal_pressure = tuple(
        -rational_weight * value for value in laplacian_action(exact)
    )
    expected_pressure = tuple(float(value) for value in ideal_pressure)
    if pressure != expected_pressure or vector_pressure != expected_pressure:
        raise RuntimeError(
            "shared pure-EPI reducers disagree with the exact rounded C6 pressure"
        )
    represented_pressure = tuple(Fraction.from_float(value) for value in pressure)
    binding = tuple(
        value - Fraction.from_float(other)
        for value, other in zip(represented_pressure, vector_pressure, strict=True)
    )
    pressure_defect = tuple(
        value - ideal
        for value, ideal in zip(represented_pressure, ideal_pressure, strict=True)
    )
    pressure_pairs = c6_pair_sums(represented_pressure)
    if any(pressure_pairs):
        raise RuntimeError(
            "shared C6 EPI pressure lost exact opposite-node antisymmetry"
        )

    source_interval = min(exact), max(exact)
    flow = observe_binary64_unit_quarter_flow(
        epi=values,
        pressure=pressure,
        epi_lower=float(source_interval[0]),
        epi_upper=float(source_interval[1]),
    )
    substep_pairs = tuple(
        c6_pair_sums(tuple(Fraction.from_float(value) for value in step.after))
        for step in flow.substeps
    )
    endpoint_pairs = c6_pair_sums(flow.exact_endpoint)
    endpoint_interval = min(flow.exact_endpoint), max(flow.exact_endpoint)
    interval_preserved = all(
        source_interval[0] <= Fraction.from_float(value) <= source_interval[1]
        for step in flow.substeps
        for value in step.after
    )
    mean_drift = flow.mean_after - center
    if (
        not interval_preserved
        or mean_drift
        or any(pair != source_pairs for pair in substep_pairs)
    ):
        raise RuntimeError(
            "pure-EPI C6 update lost its exact interval or reflected mean invariant"
        )
    return Binary64PairedC6Diffusion(
        values,
        weight,
        lower,
        upper,
        spacing,
        center,
        source_interval,
        endpoint_interval,
        source_pairs,
        pressure,
        vector_pressure,
        ideal_pressure,
        pressure_defect,
        binding,
        pressure_pairs,
        substep_pairs,
        endpoint_pairs,
        mean_drift,
        flow,
        interval_preserved,
        True,
        True,
        source_interval[0] < source_interval[1] and all(flow.stasis),
        False,
    )


def _represented_interval_extrema(lower, upper, lower_closed, upper_closed):
    """Read the first and last finite floats of a bounded exact interval.

    The shared signed upper-rounding owner gives the ceiling. Negating a
    ceiling supplies the floor; an excluded represented endpoint advances
    once to its adjacent float. All intervals here have magnitude below
    32, from EPI values in (0,1], so overflow saturation is never needed.
    """
    first = fraction_upper_signed_float(lower)
    last = -fraction_upper_signed_float(-upper)
    if not math.isfinite(first) or not math.isfinite(last):
        raise RuntimeError("a held trace produced an unbounded represented interval")
    if not lower_closed and Fraction.from_float(first) == lower:
        first = math.nextafter(first, math.inf)
    if not upper_closed and Fraction.from_float(last) == upper:
        last = math.nextafter(last, -math.inf)
    if first > last or not math.isfinite(first) or not math.isfinite(last):
        raise RuntimeError(
            "a verified held trace produced an empty represented interval"
        )
    return (0.0 if first == 0.0 else first), (0.0 if last == 0.0 else last)


@dataclass(frozen=True)
class Binary64PressureTraceCell:
    """Maximal held-pressure interval for one coordinate's four-step trace.

    The rational pressure boundaries are real preimages of nearest-even
    scaling; their flags retain ties. The first/last pressure fields are
    inclusive represented extrema. Membership concerns numeric values,
    including both signed zeros, and no future pressure-production rule.
    """

    increment_lower: Fraction
    increment_upper: Fraction
    increment_lower_closed: bool
    increment_upper_closed: bool
    first_increment: float
    last_increment: float
    pressure_lower: Fraction
    pressure_upper: Fraction
    pressure_lower_closed: bool
    pressure_upper_closed: bool
    first_pressure: float
    last_pressure: float
    recorded_pressure_margin: Fraction

    def contains(self, value: float) -> bool:
        """Test a finite represented pressure against the inclusive extrema."""
        value = _float(value, "pressure")
        return self.first_pressure <= value <= self.last_pressure


@dataclass(frozen=True)
class Binary64QuarterPressureBox:
    """Cartesian pressure box preserving one full held EPI trace.

    The source EPI, four held substeps and band are those of ``flow``.
    This is maximal for that complete coordinate trace, not for its final
    endpoint or mean alone. Pressure, rate and other runtime metadata may
    change within the box. No graph, future refresh, operator word or
    pressure-production class is certified by numeric membership.
    """

    flow: Binary64UnitQuarterFlow
    coordinates: tuple[Binary64PressureTraceCell, ...]

    def contains(self, pressures: tuple[float, ...]) -> bool:
        """Test a dimension-matched tuple of finite represented pressures."""
        values = _float_tuple(pressures, "pressures")
        if len(values) != len(self.coordinates):
            raise ValueError("pressure tuple must match the trace-box dimensions")
        return all(
            cell.contains(value)
            for cell, value in zip(self.coordinates, values, strict=True)
        )


def derive_binary64_quarter_pressure_box(
    *,
    epi: tuple[float, ...],
    pressure: tuple[float, ...],
    epi_lower: float = 0.05,
    epi_upper: float = 1.0,
) -> Binary64QuarterPressureBox:
    """Invert exact cells to preserve a supplied four-step held EPI trace.

    First reuse the B18 scalar observer, including its IEEE precondition,
    positive unclipped band, unit capacity and zero Gamma. For each node,
    let x_j be the stored source and y_j the result of substep j. A fixed
    represented increment q preserves the full trace exactly when
    q belongs to the intersection of C(y_j)-x_j over all four substeps.
    Binding interval endpoints retain their result-significand tie flags.

    Select the first/last represented increments q_min,q_max in that
    intersection. Adjacent nearest-even cells partition the real line, so
    the union of cells between them is bounded by the lower edge of
    C(q_min) and upper edge of C(q_max), with those two cells' own parity
    flags. Multiplying these real bounds by 16 inverts q=RN(p/16). Finally
    select the first/last represented pressures in the inverse interval.
    Every candidate q here has magnitude below two; p=16*q is finite and
    exactly represented, so no selected q lacks a reachable held pressure.

    Signed zero increments share one numeric cell. The production rate
    (1.0*p)+0.0 only canonicalizes zero signs; adding either zero sign to
    positive EPI gives the same trace. Subnormal scaling ties remain in
    the inverse bounds, so the pressure extrema need not equal 16*q_min
    and 16*q_max. No floating tolerance or enumeration is used.

    The resulting Cartesian box is maximal for all four EPI states with
    the fixed source tuple and declared arithmetic. Equal endpoints or
    means can occur for other traces outside it. This observer does not
    produce pressures, bind a graph or prove future cell membership.
    """
    flow = observe_binary64_unit_quarter_flow(
        epi=epi,
        pressure=pressure,
        epi_lower=epi_lower,
        epi_upper=epi_upper,
    )
    coordinates = []
    for index, recorded in enumerate(flow.exact_pressure):
        intervals = tuple(
            (
                step.result_cells[index].lower
                - Fraction.from_float(step.before[index]),
                step.result_cells[index].upper
                - Fraction.from_float(step.before[index]),
                step.result_cells[index].even_significand,
            )
            for step in flow.substeps
        )
        lower = max(item[0] for item in intervals)
        upper = min(item[1] for item in intervals)
        lower_closed = all(item[2] for item in intervals if item[0] == lower)
        upper_closed = all(item[2] for item in intervals if item[1] == upper)
        first, last = _represented_interval_extrema(
            lower, upper, lower_closed, upper_closed
        )
        if not -2 < first <= last < 2:
            raise RuntimeError(
                "a positive unit-band EPI trace lost its bounded increment range"
            )
        first_cell = _rounding_cell(first, Fraction.from_float(first))
        last_cell = _rounding_cell(last, Fraction.from_float(last))
        pressure_lower, pressure_upper = 16 * first_cell.lower, 16 * last_cell.upper
        pressure_lower_closed, pressure_upper_closed = (
            first_cell.even_significand,
            last_cell.even_significand,
        )
        first_pressure, last_pressure = _represented_interval_extrema(
            pressure_lower,
            pressure_upper,
            pressure_lower_closed,
            pressure_upper_closed,
        )
        margin = min(recorded - pressure_lower, pressure_upper - recorded)
        cell = Binary64PressureTraceCell(
            lower,
            upper,
            lower_closed,
            upper_closed,
            first,
            last,
            pressure_lower,
            pressure_upper,
            pressure_lower_closed,
            pressure_upper_closed,
            first_pressure,
            last_pressure,
            margin,
        )
        if margin < 0 or not cell.contains(flow.pressure[index]):
            raise RuntimeError("inverse pressure cells lost the recorded held trace")
        coordinates.append(cell)
    return Binary64QuarterPressureBox(flow, tuple(coordinates))
