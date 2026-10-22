"""Detached phase-reset certificates rebuilt from primitive phase coordinates.

This reader never executes an operator. It bounds the relative resultant at
the input phases directly with the shared 128-bit trigonometric owner, rather
than promoting an already rounded lower-precision resultant. Its actual lift
is the principal wrapped difference, accepted as the nearest lift to the ideal
reset only when their complete defect interval lies strictly inside (-pi, pi).
Unsupported branch rectangles reject; unresolved lift or tolerance bounds do
not certify a violation. The caller supplies the law coefficient and budget.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from ..mathematics._phase_resultant_chamber import _ordered, _phase_support
from ..mathematics._rational_interval import I, arg, cos, pi_interval, sin
from .artifact_io import exact_record

__all__ = ("PhaseResetCertificate", "bound_phase_reset")


@dataclass(frozen=True, slots=True)
class PhaseResetCertificate:
    """One target's conditional ideal-reset accuracy and lift association."""

    target: int
    relative_real: I
    relative_imaginary: I
    ideal_angle: I
    ideal_displacement: I
    represented_displacement: I
    defect: I
    nearest_lift_certified: bool
    budget: Fraction
    classification: str


def bound_phase_reset(before, after, neighbors, targets, *, coefficient, budget):
    """Enclose a supplied simultaneous ``coefficient * Arg(resultant)`` reset.

    Coordinates, coefficient and budget require exact integer/Fraction values;
    represented observations must be admitted before calling this function.
    The coefficient lies in [0, 1], the budget is nonnegative, and ordered
    targets are distinct indices. Bare support uses the shared unique-neighbor
    admission. Conductances never weight this phase resultant. No assumption
    about operator admission, whole-word validity or stability is made here.

    A defect box inside the fixed budget certifies accuracy. A sign-separated
    box outside it certifies excess only after nearest-lift admission. All
    remaining finite boxes are unresolved. Principal-Arg cut intersections
    raise ValueError rather than silently selecting a branch.
    """
    before = tuple(exact_record(value) for value in _ordered(before, "before"))
    before, neighbors = _phase_support(before, neighbors)
    after = tuple(exact_record(value) for value in _ordered(after, "after"))
    if len(after) != len(before):
        raise ValueError("before and after phase coordinates must have equal length")
    targets = _ordered(targets, "targets")
    if not targets or any(
        type(target) is not int or not 0 <= target < len(before) for target in targets
    ):
        raise ValueError("targets must be nonempty ordinary integer node indices")
    if len(set(targets)) != len(targets):
        raise ValueError("targets must not repeat an index")
    coefficient, budget = exact_record(coefficient), exact_record(budget)
    if not 0 <= coefficient <= 1 or budget < 0:
        raise ValueError("coefficient must lie in [0, 1] and budget be nonnegative")
    results = []
    for target in targets:
        gaps = tuple(I(before[j] - before[target]) for j in neighbors[target])
        real = sum((cos(gap) for gap in gaps), I(0))
        imaginary = sum((sin(gap) for gap in gaps), I(0))
        ideal_angle = arg(real, imaginary)
        ideal = coefficient * ideal_angle
        change = I(after[target] - before[target])
        represented = arg(cos(change), sin(change))
        defect = represented - ideal
        lift_certified = defect.abs_max < pi_interval().lo
        classification = "unresolved"
        if lift_certified:
            if defect.abs_max <= budget:
                classification = "certified_within_budget"
            elif defect.lo > budget or defect.hi < -budget:
                classification = "certified_exceeds_budget"
        results.append(
            PhaseResetCertificate(
                target,
                real,
                imaginary,
                ideal_angle,
                ideal,
                represented,
                defect,
                lift_certified,
                budget,
                classification,
            )
        )
    return tuple(results)
