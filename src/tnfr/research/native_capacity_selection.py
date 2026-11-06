"""Bounded proof adapter for the matched six-coordinate capacity comparison.

This is the copied/reflected two-C5, two-bridge model, with e=w=1/2 and
beta=kappa=1. It installs no runtime dynamics. The paired caller owns the
exact preparation, source freeze, stopping policy and resource budget.
"""

from __future__ import annotations

from fractions import Fraction as Q
from functools import lru_cache

from ..mathematics._comparison_flow import _ordered
from ..mathematics._interval_taylor import Jet
from ..mathematics._rational_interval import I, pi_interval
from ..mathematics._validated_taylor import validated_taylor_step
from ..physics import relational_transit as native

__all__ = ("flow", "domain", "capture", "step")


def _dynamic(value):
    if type(value) is not bool:
        raise TypeError("dynamic must be an explicit Boolean")
    return value


def _state(values, *, allow_jets=False):
    rows = _ordered(values, "six-coordinate state")
    if len(rows) != 6:
        raise ValueError("state must contain exactly (q,r,a,b,s,t)")
    if any(isinstance(value, Jet) for value in rows):
        if not allow_jets or not all(isinstance(value, Jet) for value in rows):
            raise TypeError("field jets must occupy all six coordinates")
        if len({value.order for value in rows}) != 1:
            raise ValueError("all six field jets must have the same order")
        return rows
    return tuple(I.coerce(value) for value in rows)


@lru_cache(maxsize=128)
def _domain(box):
    """Sufficient whole-box margins on the retained positive-real branch."""
    _, _, a, b, s, t = box
    u = 5 - 2 * s - 2 * t
    pi = pi_interval()
    return native._regular_bounds(box[:4]) + (
        s.lo,
        t.lo,
        u.lo,
        pi.lo - a.abs_max,
        pi.lo / 2 - b.abs_max,
        Q(1) - (a / 2 - b).abs_max,
    )


def domain(box):
    """Return exact lower margins, including the central resultant and u.

    A nonpositive margin means this sufficient proof domain is unavailable;
    it is not evidence that the actual trajectory is singular. The final
    margin retains the shared sinc-jet domain strictly inside |a/2-b|<1.
    The mean is one identically because u=5-2s-2t is reconstructed, never
    evolved independently or repaired by projection.
    """
    return _domain(_state(box))


def flow(state, *, dynamic):
    """Enclose the complete field with interval or same-order jet inputs.

    Coefficients are fixed by the declared comparison. Multiplying the
    native A/B rows by s/t, then returning to q=3A-B,r=2B-A, gives the
    displayed conjugation of the unchanged four-coordinate native field.
    Held and dynamic evaluations differ only in the final two rows.
    """
    dynamic = _dynamic(dynamic)
    rows = _state(state, allow_jets=True)
    constants = tuple(
        value.coeffs[0] if isinstance(value, Jet) else value for value in rows
    )
    if min(_domain(constants)) <= 0:
        raise ValueError("whole reflected field domain is not admitted")
    _, _, _, _, s, t = rows
    fq, fr, fa, fb = native._flow(rows[:4], Q(1, 2), Q(1, 2), Q(1))
    rates = (
        ((6 * s - t) * fq + 3 * (s - t) * fr) / 5,
        (2 * (t - s) * fq + (6 * t - s) * fr) / 5,
        s * fa,
        t * fb,
    )
    if dynamic:
        return rates + ((t - s) / 2, (5 - s - 4 * t) / 2)
    return rates + (0 * s, 0 * t)


def capture(box):
    """Classify the entire box by the retained strict capture inequalities.

    The selected sector is None unless domain, rectangle and storage all
    pass. Sector zero is a successful consensus certificate, not a falsey
    missing result. Returned intervals retain the unsuccessful candidates.
    This uses the extended heterogeneous-capacity capture proof in CS2;
    it does not invoke the historical unit-capacity admission certificate.
    """
    rows = _state(box)
    lower = _domain(rows)
    storage = native._storage(rows[:4], Q(1))
    candidates = native._rectangle_margins(rows[:4])
    admitted = tuple(
        (sector, margins)
        for sector, margins in candidates
        if min(lower) > 0 and storage.hi < 7 and all(value.lo > 0 for value in margins)
    )
    if len(admitted) > 1:
        raise ArithmeticError("disjoint capture rectangles cannot both pass")
    sector, margins = admitted[0] if admitted else (None, ())
    return {
        "sector": sector,
        "margins": margins,
        "candidates": candidates,
        "storage": storage,
        "domain": lower,
    }


def step(box, *, dynamic, duration=Q(1, 8), order=14, time=Q(0)):
    """Return the shared (certificate, failed tube, reason) without retry.

    The caller declares the total horizon. Every accepted step retains its
    whole-time tube, error terms and exact time; an unavailable enclosure
    is returned unchanged. No endpoint midpoint or capacity reset is used.
    """
    dynamic = _dynamic(dynamic)
    rows = _state(box)
    return validated_taylor_step(
        rows,
        duration,
        lambda state: flow(state, dynamic=dynamic),
        domain,
        order=order,
        time=time,
        domain_failure="whole_reflected_capacity_domain_not_admitted",
    )
