"""Exact reductions of finite, already materialized phasor components.

No trigonometric function is evaluated here. Permutation invariance concerns
the same binary64 component multiset, not a change of phase chart, trigonometric
backend, gauge, node state or complete runtime evolution.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Set
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from .._binary64 import uses_ieee_binary64_rounding
from .._exact_time import finite_represented_real
from ._exact_weighted import exact_weighted_sum_ratio

__all__ = ["RepresentedPhasorResultant", "reduce_phasor_components"]


@dataclass(frozen=True)
class RepresentedPhasorResultant:
    """Detached exact sum and a range-safe numerical direction readout.

    ``real_sum`` and ``imag_sum`` are sums, not means. At nonzero resultant,
    ``scale`` is their maximum absolute value, so one exact scaled component
    is +1 or -1. Binary64 conversion therefore cannot overflow or round both
    components to zero. The smaller component may underflow (preserving the
    conversion's signed zero); its exact value
    and signed ``binary64 - exact`` defect remain recorded. ``angle`` is only
    the numerical ``math.atan2`` of the represented scaled pair, without a
    certified angular error bound. Exact joint zero has no chosen direction.
    """

    components: tuple[tuple[float, float], ...]
    count: int
    real_sum: Fraction
    imag_sum: Fraction
    joint_zero: bool
    scale: Fraction | None
    scaled_exact: tuple[Fraction, Fraction] | None
    scaled_binary64: tuple[float, float] | None
    scaled_rounding_defect: tuple[Fraction, Fraction] | None
    angle: float | None


def _materialized_pair(value: Any, index: int) -> tuple[float, float]:
    if isinstance(value, (str, bytes, bytearray, Mapping, Set)):
        raise TypeError(f"components[{index}] must be an ordered pair of real scalars")
    try:
        iterator = iter(value)
    except TypeError as exc:
        raise TypeError(
            f"components[{index}] must be an ordered pair of real scalars"
        ) from exc
    # At most three reads suffice to reject malformed or unbounded inner pairs.
    missing = object()
    first, second, extra = (next(iterator, missing) for _ in range(3))
    if first is missing or second is missing or extra is not missing:
        raise ValueError(f"components[{index}] must contain exactly two scalars")
    real = finite_represented_real(first, f"components[{index}].real")[0]
    imag = finite_represented_real(second, f"components[{index}].imag")[0]
    return real, imag


def reduce_phasor_components(components: Iterable[Any]) -> RepresentedPhasorResultant:
    """Sum a nonempty finite iterable of represented real component pairs.

    The caller must provide a finite outer iterable; generators are consumed
    once. Each pair is materialized through ``finite_represented_real`` before
    reduction, rejecting booleans, nonfinite or unrepresentable real inputs.
    Signed input zeros follow that owner's canonical positive-zero contract.
    No assumption that pairs have unit norm is made. Unit weights feed the
    shared exact dyadic reduction, without an intermediate rounded sum.
    The shared IEEE binary64 rounding precondition must hold; otherwise this
    function rejects before consuming caller input.

    A fixed component multiset has identical exact sums and normalized output
    under permutation. Scaling by their positive common maximum preserves the
    exact ray; it does not improve the conditioning of an uncertain input or
    establish symmetry/phase invariance for transcendental materialization.
    """
    if not uses_ieee_binary64_rounding():
        raise ValueError(
            "represented phasor reduction requires IEEE binary64 nearest-even rounding"
        )
    if isinstance(components, (str, bytes, bytearray, Mapping)):
        raise TypeError("components must be a finite iterable of real component pairs")
    try:
        iterator = iter(components)
    except TypeError as exc:
        raise TypeError(
            "components must be a finite iterable of real component pairs"
        ) from exc
    pairs = tuple(
        _materialized_pair(value, index) for index, value in enumerate(iterator)
    )
    if not pairs:
        raise ValueError("at least one phasor component pair is required")
    weights = (1.0,) * len(pairs)
    real_sum = Fraction(*exact_weighted_sum_ratio(weights, (real for real, _ in pairs)))
    imag_sum = Fraction(*exact_weighted_sum_ratio(weights, (imag for _, imag in pairs)))
    if real_sum == 0 and imag_sum == 0:
        return RepresentedPhasorResultant(
            pairs, len(pairs), real_sum, imag_sum, True, None, None, None, None, None
        )
    scale = max(abs(real_sum), abs(imag_sum))
    exact = (real_sum / scale, imag_sum / scale)
    represented = (float(exact[0]), float(exact[1]))
    defect = tuple(
        Fraction.from_float(value) - target for value, target in zip(represented, exact)
    )
    angle = math.atan2(represented[1], represented[0])
    return RepresentedPhasorResultant(
        pairs,
        len(pairs),
        real_sum,
        imag_sum,
        False,
        scale,
        exact,
        represented,
        defect,
        angle,
    )
