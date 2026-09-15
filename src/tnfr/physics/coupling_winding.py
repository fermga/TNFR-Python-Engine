"""Exact gap transport for the target-only canonical Coupling realization.

On a simple cycle, with ``UM_BIDIRECTIONAL=False``, no functional links,
and both neighbors strictly inside the U3 gate, their circular mean has
the midpoint lift. Single-target UM mixes its two incident oriented gaps;
an all-target immutable stage applies cycle diffusion to those gaps.

These detached algebraic observations do not infer a winding integer from
the supplied rational gap sum, admit an operator word, bind a live graph,
or certify a binary64 trajectory. See ``COUPLING_WINDING_PERSISTENCE.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from numbers import Integral

from .._exact_time import exact_or_represented_real as _rational
from ..constants.canonical import DELTA_PHI_MAX, UM_THETA_PUSH
from ._cycle_algebra import (
    Matrix, Vector, laplacian_matrix, ordered_vector,
)

__all__ = ["CouplingGapStep", "observe_coupling_gap_step"]


def _gap_vector(gaps) -> Vector:
    values = ordered_vector(gaps, "gap")
    if len(values) < 3:
        raise ValueError("a simple cycle requires at least three gaps")
    return values


def _gap_matrix(count: int, eta: Fraction, target: int | None) -> Matrix:
    if target is None:
        return tuple(
            tuple(Fraction(i == j) - eta * value for j, value in enumerate(row))
            for i, row in enumerate(laplacian_matrix(count))
        )
    rows = [
        [Fraction(i == j) for j in range(count)] for i in range(count)
    ]
    half = eta / 2
    left, right = (target - 1) % count, target
    rows[left][left], rows[left][right] = 1 - half, half
    rows[right][left], rows[right][right] = half, 1 - half
    return tuple(tuple(row) for row in rows)


@dataclass(frozen=True)
class CouplingGapStep:
    """One exact interval-preserving gap step in a declared UM model.

    The spread is ``sum((gap-mean_gap)**2)/2``. No phase-loop closure or
    runtime provenance is inferred from these supplied gap coordinates.
    """

    input_gaps: Vector
    output_gaps: Vector
    eta: Fraction
    phase_gate: Fraction
    mode: str
    target: int | None
    matrix: Matrix
    gap_sum: Fraction
    mean_gap: Fraction
    spread_before: Fraction
    spread_after: Fraction
    spread_drop: Fraction
    expected_spread_drop: Fraction
    sum_residual: Fraction
    minimum_before: Fraction
    maximum_before: Fraction
    minimum_after: Fraction
    maximum_after: Fraction
    invariant_interval_preserved: bool


def observe_coupling_gap_step(
    gaps,
    *,
    target=None,
    eta=UM_THETA_PUSH,
    phase_gate=DELTA_PHI_MAX,
) -> CouplingGapStep:
    """Observe one exact target-only UM gap map, without graph mutation.

    ``target=None`` means one simultaneous all-target Jacobi stage. An
    integer target changes its two incident gaps at indices ``target-1``
    and ``target``. Both realizations require a fixed simple undirected
    cycle, target-only UM, no functional-link creation and two compatible
    neighbors per target when interpreted as canonical phase operations.

    Exact rational inputs are preserved. Other real inputs retain their
    shared materialized binary64 rational value; in particular the default
    eta is the represented canonical UM factor. Require ``0 < eta <= 1``
    and every gap strictly inside the declared gate, itself at most the
    represented canonical pi/2 limit. A rational sum does not certify an
    exact multiple of mathematical 2*pi. An external phase observation must
    establish cycle closure and winding separately.
    """
    values = _gap_vector(gaps)
    count = len(values)
    coefficient = _rational(eta, "eta")
    gate = _rational(phase_gate, "phase_gate")
    canonical_gate = Fraction.from_float(float(DELTA_PHI_MAX))
    if not 0 < coefficient <= 1:
        raise ValueError("eta must lie in (0, 1]")
    if not 0 < gate <= canonical_gate:
        raise ValueError("phase_gate must lie in (0, canonical pi/2]")
    if any(abs(value) >= gate for value in values):
        raise ValueError("every gap must lie strictly inside the phase gate")
    if target is not None:
        if isinstance(target, bool) or not isinstance(target, Integral):
            raise TypeError("target must be an integer cycle index or None")
        target = int(target)
        if not 0 <= target < count:
            raise ValueError("target must belong to the declared cycle")

    matrix = _gap_matrix(count, coefficient, target)
    output = tuple(
        sum((weight * value for weight, value in zip(row, values)), Fraction(0))
        for row in matrix
    )
    total = sum(values, Fraction(0))
    mean = total / count
    before = sum(((value - mean) ** 2 for value in values), Fraction(0)) / 2
    after = sum(((value - mean) ** 2 for value in output), Fraction(0)) / 2
    if target is None:
        edge_square = sum(
            ((values[(i + 1) % count] - values[i]) ** 2 for i in range(count)),
            Fraction(0),
        )
        two_hop_square = sum(
            (
                (values[(i + 1) % count] - values[(i - 1) % count]) ** 2
                for i in range(count)
            ),
            Fraction(0),
        )
        expected = (
            coefficient * (1 - coefficient) * edge_square / 2
            + coefficient**2 * two_hop_square / 8
        )
    else:
        difference = values[target] - values[(target - 1) % count]
        expected = coefficient * (2 - coefficient) * difference**2 / 4
    drop = before - after
    residual = sum(output, Fraction(0)) - total
    lower, upper = min(values), max(values)
    next_lower, next_upper = min(output), max(output)
    interval_preserved = lower <= next_lower <= next_upper <= upper
    if residual or drop != expected or drop < 0 or not interval_preserved:
        raise RuntimeError("exact Coupling gap transport lost its identities")
    return CouplingGapStep(
        input_gaps=values,
        output_gaps=output,
        eta=coefficient,
        phase_gate=gate,
        mode="all_targets" if target is None else "single_target",
        target=target,
        matrix=matrix,
        gap_sum=total,
        mean_gap=mean,
        spread_before=before,
        spread_after=after,
        spread_drop=drop,
        expected_spread_drop=expected,
        sum_residual=residual,
        minimum_before=lower,
        maximum_before=upper,
        minimum_after=next_lower,
        maximum_after=next_upper,
        invariant_interval_preserved=interval_preserved,
    )
