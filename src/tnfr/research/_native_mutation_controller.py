"""Universal controller restrictions for the declared Mutation renewal domain.

The two-call law and memory are those of the private source-feasibility
compiler. This instrument evaluates nine carried controller branch pairs
and five algebraic endpoint multiplicities, never a parameter preparation,
source, trajectory or solver. A surviving branch remains conditional.

For incoming phase diameter D, the five-cycle coordinator has oscillation
coefficient 1-G-L/2 when A=1-G-L >= L/2. A target-only negative shift of
magnitude s changes the coordinator output by a vector of diameter A*s.
Complete renewal therefore requires D<T=A*s/(G+L/2).

If the shifted target were still inside the old phase hull, a positive
coordinator with row sum 1-G would shrink its diameter. Thus, after setting
the old minimum to zero, the postevent target is -a with 0<a<=s and the
other four angles lie in [0,D]. On the common arc D+s<pi, increasing a to
s cannot increase R squared. Holding four angles fixed, the remaining
phasor has no interior minimum against their nonzero resultant: its opposite
lies outside that short arc. Replacing each free angle by an endpoint
therefore bounds R squared from below by five endpoint counts. Every count
is nonincreasing in D, so replacing D by T is a valid upward domain
relaxation. These extrema are not asserted to lie in the image of the first
coordinator or to satisfy any pressure, form, capacity or history condition.

The fixed work policy is eight cosine-series terms (degree-14 lower bound)
and outward 128-bit dyadic rounding. No result-dependent refinement occurs.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import product

from ..config.constants import DISRUPTORS
from ..mathematics._phase_resultant_chamber import (
    COSINE_ENCLOSURE_METHOD,
    certified_cosine_bounds,
)
from ._native_mutation_feasibility import (
    BRANCHES,
    ExecutionPremises,
    NativeMutationLaw,
    _gains,
    _owned_law,
)

Q = Fraction
COSINE_TERMS = 8
COSINE_BITS = 128


@dataclass(frozen=True, slots=True)
class EndpointMultiplicity:
    """Enclosure of one relaxed endpoint expression, not an actual state."""

    high_count: int
    low_count: int
    r_squared_lower: Fraction
    r_squared_upper: Fraction


@dataclass(frozen=True, slots=True)
class ControllerDomainCell:
    first_branch: str
    second_branch: str
    first_gains: tuple[Fraction, Fraction]
    second_gains: tuple[Fraction, Fraction]
    shift: Fraction
    diameter_ceiling: Fraction
    cosine_angles: tuple[Fraction, Fraction, Fraction]
    cosine_enclosures: tuple[tuple[Fraction, Fraction], ...]
    endpoints: tuple[EndpointMultiplicity, ...]
    universal_r_squared_lower: Fraction
    # This is an upper bound on the relaxed MINIMUM, not on the actual R.
    relaxed_minimum_upper: Fraction
    status: str
    exclusion_reason: str | None


@dataclass(frozen=True, slots=True)
class ControllerDomainRestriction:
    law: NativeMutationLaw
    execution: ExecutionPremises
    loads: tuple[Fraction, Fraction]
    r_squared_thresholds: tuple[Fraction, Fraction]
    cells: tuple[ControllerDomainCell, ...]
    terms: int
    bits: int
    cosine_method: str
    obligations: tuple[str, ...]


def derive_native_mutation_controller_domain() -> ControllerDomainRestriction:
    """Rebuild the owned law and evaluate the single frozen coefficient pass.

    There are no source parameters or input reports to trust. Exclusion is
    one-way and concerns only the assumed second branch under complete
    target-minimum renewal. A conditional result does not admit that branch,
    establish joint source existence or select any evolution law.
    """
    law, execution = _owned_law()
    policy = law.phase_policy
    disruptive = {getattr(mark, "value", mark) for mark in DISRUPTORS}
    loads = tuple(
        Q(
            sum(mark in disruptive for trace in traces for mark in trace),
            sum(map(len, traces)),
        )
        for traces in execution.traces[1:]
    )
    if loads != (Q(1, 20), Q(2, 25)) or max(loads) > policy.disr_lo:
        raise ValueError("the frozen two-call low-load controller domain changed")
    lower_cut, upper_cut = policy.R_lo**2, policy.R_hi**2
    cells = []
    for first, second in product(BRANCHES, repeat=2):
        gains1 = _gains(law, law.phase_gains, first)
        gains2 = _gains(law, gains1, second)
        global_gain, local_gain = gains2
        diagonal = 1 - global_gain - local_gain
        if not (0 < global_gain and 0 < local_gain and diagonal >= local_gain / 2):
            raise ValueError("the declared five-cycle oscillation row changed")
        diameter = diagonal * law.mutation_shift / (global_gain + local_gain / 2)
        angles = law.mutation_shift, diameter, diameter + law.mutation_shift
        if not 0 < min(angles) <= max(angles) < 3:
            raise ValueError("the fixed degree-14 short-arc domain is unavailable")
        enclosures = tuple(
            certified_cosine_bounds(angle, terms=COSINE_TERMS, bits=COSINE_BITS)
            for angle in angles
        )
        endpoints = []
        for high in range(5):
            low = 4 - high
            constant = 1 + low * low + high * high
            weights = 2 * low, 2 * high * low, 2 * high
            bounds = tuple(
                (
                    constant
                    + sum(
                        weight * interval[side]
                        for weight, interval in zip(weights, enclosures)
                    )
                )
                / 25
                for side in (0, 1)
            )
            endpoints.append(EndpointMultiplicity(high, low, *bounds))
        lower = min(endpoint.r_squared_lower for endpoint in endpoints)
        relaxed_upper = min(endpoint.r_squared_upper for endpoint in endpoints)
        excluded, reason = False, None
        if second == "dissonant" and lower > lower_cut:
            excluded = True
            reason = "renewal implies R>R_lo while the actual disruptor load is low"
        elif second == "transition" and lower >= upper_cut:
            excluded = True
            reason = "renewal implies the native stable-first branch"
        cells.append(
            ControllerDomainCell(
                first,
                second,
                gains1,
                gains2,
                law.mutation_shift,
                diameter,
                angles,
                enclosures,
                tuple(endpoints),
                lower,
                relaxed_upper,
                "excluded" if excluded else "conditional",
                reason,
            )
        )
    return ControllerDomainRestriction(
        law,
        execution,
        loads,
        (lower_cut, upper_cut),
        tuple(cells),
        COSINE_TERMS,
        COSINE_BITS,
        COSINE_ENCLOSURE_METHOD,
        (
            "unchanged two-call acquired negative-outward target-minimum renewal domain",
            "actual first controller branch remains to be admitted from its source phases",
            "positive capacities, both actual IL masks and fresh pressure peaks remain joint",
            "all fifteen form boundaries share the same offset and unclipped held-flow histories",
            "endpoint multiplicities are outer phase extrema, not admitted parameter sources",
            "conditional means not excluded by this bound, never feasible or branch-admitted",
            "ideal common-lift scope; binary64 wrap and negative-zero realization remain separate",
        ),
    )
