"""Exact reflected source/receiver restriction of the native relational law.

This private proof owner fixes two unit C5 rings, matching bridges at positions
zero and one, unit held capacities, no forcing and no event. It evaluates an
eight-coordinate interval field or its formal Taylor jets; it does not advance
a trajectory, validate a time tube, or project a live graph onto a symmetry.

Coordinates are ``(p,r,P,R,a,b,A,B)``. Ring forms reconstruct as
``(p,-p,-r,0,r)`` and phases as ``(a,-a,-b,0,b)``, with uppercase coordinates
for the receiver. Joint sign/reflection symmetry supplies all omitted rows.
The effective coefficients are explicit and are never normalized here.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from ..mathematics._interval_taylor import Jet
from ..mathematics._interval_taylor import arg as jet_arg
from ..mathematics._interval_taylor import atan_ratio as jet_atan_ratio
from ..mathematics._interval_taylor import cos as jet_cos
from ..mathematics._interval_taylor import sin as jet_sin
from ..mathematics._phase_resultant_chamber import regular_resultant_margin
from ..mathematics._rational_interval import I, arg, atan_ratio, cos, pi_interval, sin

__all__ = ("ReflectedRegularFlow", "evaluate_reflected_regular_flow")


@dataclass(frozen=True)
class ReflectedRegularFlow:
    """Detached exact-model field and domain evidence on its expansion box.

    Resultants, phase sources and inverse metrics follow ``row_order``. The
    other four rows are their conjugate/sign partners, not independently
    discarded observations. Margins concern the entire constant interval of
    each Taylor jet; nonconstant coefficients do not enclose any future path.
    Storage and loss are interval/jet expressions for the same supplied law.
    """

    state: tuple[I | Jet, ...]
    epi_weight: Q
    phase_weight: Q
    storage_scale: Q
    rates: tuple[I | Jet, ...]
    resultants: tuple[tuple[I | Jet, I | Jet], ...]
    resultant_margin_lower_bounds: tuple[Q, ...]
    phase_sources: tuple[I | Jet, ...]
    inverse_phase_metrics: tuple[I | Jet, ...]
    storage: I | Jet
    continuous_loss: I | Jet
    row_order: tuple[int, ...] = (0, 4, 3, 5, 9, 8)


def _constant(value):
    return value.coeffs[0] if isinstance(value, Jet) else value


def _coefficient(value, label, *, allow_zero=False):
    if not isinstance(value, Q) and type(value) is not int:
        raise TypeError(f"{label} requires an exact Fraction or integer")
    value = Q(value)
    if value < 0 or (not allow_zero and not value):
        relation = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"{label} must be {relation}")
    return value


def _state(values):
    values = tuple(values)
    if len(values) != 8:
        raise ValueError("reflected flow requires exactly eight coordinates")
    jets = tuple(isinstance(value, Jet) for value in values)
    if any(jets):
        if not all(jets):
            raise TypeError(
                "reflected coordinates must all be intervals or all be jets"
            )
        if len({value.order for value in values}) != 1:
            raise ValueError("reflected coordinate jets require the same order")
        return values
    return tuple(I.coerce(value) for value in values)


def _ring_resultants(a, b, other, cs, sn, zero):
    return (
        (
            cs(2 * a) + cs(b - a) + cs(other - a),
            -sn(2 * a) + sn(b - a) + sn(other - a),
        ),
        (cs(a - b) + cs(b), sn(a - b) - sn(b)),
        (2 * cs(b), zero),
    )


def _phase_coefficients(real, imaginary, *, is_jet, pi):
    """The two regular charts share the native Arg pressure and H*g=S.

    Positive real part uses the removable-zero chart atan(S/C)/(S/C), even
    when S is sign-separated: dividing Arg by a tiny interval S would lose
    the cancellation and inflate otherwise bounded derivatives. Without a
    positive real bound, a sign-separated imaginary part permits Arg/S.
    A mathematically
    regular but unresolved wide atan-ratio jet box raises rather than losing
    analytic derivative evidence or replacing the constitutive law.
    """
    argument = jet_arg if is_jet else arg
    ratio = jet_atan_ratio if is_jet else atan_ratio
    angle = argument(real, imaginary)
    if _constant(real).lo > 0:
        inverse = ratio(imaginary / real) / real / pi
    else:
        inverse = angle / imaginary / pi
    if _constant(inverse).lo <= 0:
        raise ValueError("positive inverse phase metric is unresolved on this box")
    return angle / pi, inverse


def _reflected_rate_rows(
    form_contrasts,
    phase_sources,
    inverse_phase_metrics,
    *,
    epi_weight,
    phase_weight,
    storage_scale,
):
    """Shared algebra for the four consumed form/phase row pairs.

    This helper performs no state admission. Live evaluation must first admit
    all six full resultants, including the two symmetry-fixed central rows.
    A static boundary certificate can reuse the algebra of limiting consumed
    coefficients without defining or executing the missing full-state field.
    """
    form_rates = tuple(
        -epi_weight * contrast / degree + phase_weight * source
        for contrast, degree, source in zip(form_contrasts, (3, 2, 3, 2), phase_sources)
    )
    phase_rates = tuple(
        contrast * inverse * (phase_weight / storage_scale)
        for contrast, inverse in zip(form_contrasts, inverse_phase_metrics)
    )
    return form_rates + phase_rates


def evaluate_reflected_regular_flow(
    state, *, epi_weight, phase_weight, storage_scale
) -> ReflectedRegularFlow:
    """Evaluate the eight-coordinate law with six full resultant admissions.

    Scalar coordinates accept exact integers, Fractions or rational intervals;
    all-jet coordinates must share one Taylor order. Floating conversion and
    graph reconstruction belong to explicit callers. The supplied effective
    coefficients satisfy ``e>=0,w>0,beta>0`` and are not inferred or normalized.

    General complex resultants retain the complete regular phase domain.
    No artificial ``abs(a/2-b)<=1`` sinc-series condition or chosen lift is
    imposed on the mathematical law. Numerical interval/jet overestimation
    can still leave regularity or a removable-axis derivative unresolved.
    """
    state = _state(state)
    e = _coefficient(epi_weight, "epi_weight", allow_zero=True)
    w = _coefficient(phase_weight, "phase_weight")
    beta = _coefficient(storage_scale, "storage_scale")
    p, r, capital_p, capital_r, a, b, capital_a, capital_b = state
    is_jet = isinstance(a, Jet)
    cs, sn = (jet_cos, jet_sin) if is_jet else (cos, sin)
    zero = Jet.constant(0, a.order) if is_jet else I(0)
    resultants = _ring_resultants(a, b, capital_a, cs, sn, zero) + _ring_resultants(
        capital_a, capital_b, a, cs, sn, zero
    )
    margins = tuple(
        regular_resultant_margin(
            (_constant(real).lo, _constant(real).hi),
            (_constant(imaginary).lo, _constant(imaginary).hi),
        )
        for real, imaginary in resultants
    )
    if any(value <= 0 for value in margins):
        raise ValueError(
            "all six full resultant rows must exclude the nonpositive-real ray"
        )
    pi = pi_interval()
    coefficients = tuple(
        _phase_coefficients(real, imaginary, is_jet=is_jet, pi=pi)
        for real, imaginary in resultants
    )
    sources = tuple(row[0] for row in coefficients)
    inverse_metrics = tuple(row[1] for row in coefficients)
    q_left = 4 * p - r - capital_p
    v_left = 2 * r - p
    q_right = 4 * capital_p - capital_r - p
    v_right = 2 * capital_r - capital_p
    rates = _reflected_rate_rows(
        (q_left, v_left, q_right, v_right),
        tuple(sources[index] for index in (0, 1, 3, 4)),
        tuple(inverse_metrics[index] for index in (0, 1, 3, 4)),
        epi_weight=e,
        phase_weight=w,
        storage_scale=beta,
    )
    form_storage = (
        2 * p**2
        + (p - r) ** 2
        + r**2
        + 2 * capital_p**2
        + (capital_p - capital_r) ** 2
        + capital_r**2
        + (p - capital_p) ** 2
    )
    phase_storage = (
        12
        - cs(2 * a)
        - 2 * cs(a - b)
        - 2 * cs(b)
        - cs(2 * capital_a)
        - 2 * cs(capital_a - capital_b)
        - 2 * cs(capital_b)
        - 2 * cs(a - capital_a)
    )
    loss = e * (Q(2, 3) * (q_left**2 + q_right**2) + v_left**2 + v_right**2)
    return ReflectedRegularFlow(
        state,
        e,
        w,
        beta,
        rates,
        resultants,
        margins,
        sources,
        inverse_metrics,
        form_storage + beta * phase_storage,
        loss,
    )
