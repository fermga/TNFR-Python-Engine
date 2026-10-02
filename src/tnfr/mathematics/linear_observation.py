"""Exact linear observations and coordinate memory of a supplied generator.

The positive-sign convention is z'=J*z. The invariant row space is a model
property, not a graph, constitutive-law, trajectory or provenance certificate.
Known affine sources can be projected once this linear space is admitted.
"""

from __future__ import annotations

from collections.abc import Mapping, Set
from dataclasses import dataclass
from fractions import Fraction
from numbers import Integral

from .._exact_time import exact_or_represented_real
from ._exact_linear_algebra import exact_matrix_inverse, exact_matrix_product
from .krylov import exact_rank

__all__ = [
    "LinearObservationLevel",
    "LinearObservation",
    "derive_linear_observation",
    "LinearCoordinateMemory",
    "derive_coordinate_memory",
]

Vector = tuple[Fraction, ...]
Matrix = tuple[Vector, ...]


@dataclass(frozen=True)
class LinearObservationLevel:
    """One completely examined output-Krylov level O*J**power."""

    power: int
    rank_before: int
    rank_after: int
    candidate_rows: int
    selected_row_indices: tuple[int, ...]


@dataclass(frozen=True)
class LinearObservation:
    """Minimal all-state linear realization of y=O*z under z'=J*z.

    With s=C*z, the exact identities C*T=I, C*J=G*C and O=D*C imply
    s'=G*s and y=D*s. The selected coordinates need not be local, physical
    nodes or a nonlinear observation. Source/initial-state admissibility is
    separate. Rationalized floating coefficients certify only that supplied
    rational model, not the exact transcendental matrix they approximate.
    """

    generator: Matrix
    output_rows: Matrix
    output_count: int
    output_rank: int
    dimension: int
    full_state_dimension: int
    extra_coordinates: int
    rank_progression: tuple[int, ...]
    level_records: tuple[LinearObservationLevel, ...]
    selected_row_labels: tuple[tuple[int, int], ...]
    pivot_columns: tuple[int, ...]
    observation: Matrix
    right_inverse: Matrix
    reduced_generator: Matrix
    output_map: Matrix
    exact_identity_checks: tuple[str, ...]
    rank_calls: int
    max_rank_calls: int
    matrix_product_calls: int
    completed_levels: int
    stabilization_power: int
    max_coefficient_bits: int
    scope: str


@dataclass(frozen=True)
class LinearCoordinateMemory:
    """Exact blocks of y'=A*y+B*h, h'=C*y+D*h under z'=J*z.

    Visible coordinates retain the requested order; hidden coordinates keep
    their original order. Eliminating h gives the initial-state contribution
    B*exp(D*t)*h0 and convolution kernel K(t)=B*exp(D*t)*C. Only K(0)=B*C is
    evaluated here. A vanishing K(0) does not imply a vanishing kernel or
    initial-state contribution. Blocks may be signed and D need not be stable.
    """

    generator: Matrix
    visible_indices: tuple[int, ...]
    hidden_indices: tuple[int, ...]
    visible_generator: Matrix
    hidden_to_visible: Matrix
    visible_to_hidden: Matrix
    hidden_generator: Matrix
    kernel_at_zero: Matrix
    scope: str


def _ordered(values, label):
    if isinstance(values, (str, bytes, bytearray, Mapping, Set)):
        raise TypeError(f"{label} must be an ordered collection")
    try:
        return tuple(values)
    except TypeError as exc:
        raise TypeError(f"{label} must be an ordered collection") from exc


def _matrix(values, label):
    rows = _ordered(values, label)
    if not rows:
        raise ValueError(f"{label} must be nonempty")
    return tuple(
        tuple(
            exact_or_represented_real(value, f"{label}[{i}]")
            for value in _ordered(row, f"{label}[{i}]")
        )
        for i, row in enumerate(rows)
    )


def derive_coordinate_memory(generator, visible_indices) -> LinearCoordinateMemory:
    """Partition a fixed linear law without discarding its hidden dynamics.

    J is a finite, ordered, nonempty square matrix with the shared exact-or-
    represented scalar admission. Rational coefficients remain exact; other
    admitted real coefficients retain their binary64 values. Visible indices
    must be an ordered, nonempty proper subset of distinct nonboolean integers.

    For z'=J*z and visible/hidden blocks (y,h), the exact elimination identity is

        y'(t) = A*y(t) + B*exp(D*t)*h(0)
                + integral_0^t B*exp(D*(t-s))*C*y(s) ds.

    The returned matrices specify this identity without evaluating an
    exponential, supplying an initial state or evolving a graph. Neither
    memory decay, kernel positivity, autonomous instantaneous closure nor a
    physical connection follows from this decomposition. In particular, a
    zero B*C need not make B*exp(D*t)*C identically zero. No approximation,
    Schur-complement substitution or selected constitutive law is introduced.
    """
    admitted = _matrix(generator, "generator")
    dimension = len(admitted)
    if any(len(row) != dimension for row in admitted):
        raise ValueError("generator must be a nonempty square matrix")
    supplied_indices = _ordered(visible_indices, "visible_indices")
    if any(
        isinstance(index, bool) or not isinstance(index, Integral)
        for index in supplied_indices
    ):
        raise TypeError("visible_indices must contain non-boolean integers")
    visible = tuple(int(index) for index in supplied_indices)
    if not 0 < len(visible) < dimension:
        raise ValueError("visible_indices must be a nonempty proper subset")
    selected = set(visible)
    if len(selected) != len(visible):
        raise ValueError("visible_indices must be distinct")
    if any(index < 0 or index >= dimension for index in visible):
        raise ValueError("visible_indices must lie within the generator dimension")
    hidden = tuple(index for index in range(dimension) if index not in selected)

    def block(rows, columns):
        return tuple(tuple(admitted[i][j] for j in columns) for i in rows)

    a, b = block(visible, visible), block(visible, hidden)
    c, d = block(hidden, visible), block(hidden, hidden)
    return LinearCoordinateMemory(
        generator=admitted,
        visible_indices=visible,
        hidden_indices=hidden,
        visible_generator=a,
        hidden_to_visible=b,
        visible_to_hidden=c,
        hidden_generator=d,
        kernel_at_zero=exact_matrix_product(b, c),
        scope=(
            "Exact coordinate decomposition of one supplied fixed rational generator "
            "with positive sign z'=J*z. General initial-hidden-state influence is retained. "
            "No exponential evaluation, decay/positivity/instantaneous-closure theorem, "
            "graph authentication, autonomous connection law or physical identity "
            "is certified. Represented coefficients do not certify an ideal "
            "transcendental generator."
        ),
    )


def derive_linear_observation(
    generator,
    output_rows,
    *,
    max_rank_calls=4096,
) -> LinearObservation:
    r"""Close the row space of O under a fixed positive-sign generator J.

    Inputs are finite ordered nonempty matrices. J must be square and each
    output row must have the generator's state dimension. Rational scalars remain
    exact; other admitted real scalars use the shared binary64 admission.
    Boolean, nonfinite and nonzero-underflow scalars are rejected. Redundant
    and all-zero output rows are permitted.

    Examine complete levels O, OJ, ... through the first stabilized level,
    selecting independent rows deterministically. All-state minimality follows
    because any invariant row space retaining O contains every OJ**k. A first
    row of rank one can require n+1 complete levels, through power n. A zero
    output requires the power-one check and yields an empty reduced state.

    Rank invocations are bounded by a positive nonboolean max_rank_calls.
    Exhaustion raises without a partial result or approximate fallback. The
    count does not bound input materialization, coefficient bit growth, memory
    or wall time; product counts exclude internal elimination operations.
    Neither stability, reachable-state minimality nor a nonlinear reduction
    follows. No graph, trajectory, affine source or initial state is supplied.
    """
    if type(max_rank_calls) is not int:
        raise TypeError("max_rank_calls must be a positive non-boolean integer")
    if max_rank_calls < 1:
        raise ValueError("max_rank_calls must be positive")
    a = _matrix(generator, "generator")
    if any(len(row) != len(a) for row in a):
        raise ValueError("generator must be a nonempty square matrix")
    r = _matrix(output_rows, "output_rows")
    if any(len(row) != len(a) for row in r):
        raise ValueError("each output row must match the generator dimension")
    n, m = len(a), len(r)
    rank_calls = product_calls = max_bits = 0

    def retain(matrix):
        nonlocal max_bits
        for row in matrix:
            for value in row:
                max_bits = max(
                    max_bits,
                    abs(value.numerator).bit_length(),
                    value.denominator.bit_length(),
                )
        return matrix

    def rank(matrix):
        nonlocal rank_calls
        if rank_calls >= max_rank_calls:
            raise ValueError("exact rank-call budget exhausted; realization incomplete")
        rank_calls += 1
        return exact_rank(matrix)

    def product(left, right):
        nonlocal product_calls
        product_calls += 1
        return retain(exact_matrix_product(left, right))

    retain(a)
    retain(r)
    basis, labels, levels = [], [], []
    level = r
    for power in range(n + 1):
        previous_rank = len(basis)
        selected = []
        for row_index, row in enumerate(level):
            observed_rank = rank((*basis, row))
            if observed_rank not in (len(basis), len(basis) + 1):
                raise RuntimeError(
                    "exact independent-row selection lost rank consistency"
                )
            if observed_rank > len(basis):
                basis.append(row)
                labels.append((power, row_index))
                selected.append(row_index)
        levels.append(
            LinearObservationLevel(
                power,
                previous_rank,
                len(basis),
                m,
                tuple(selected),
            )
        )
        if power > 0 and len(basis) == previous_rank:
            break
        if power < n:
            level = product(level, a)
    else:
        raise RuntimeError(
            "exact row-space closure did not stabilize within its dimension bound"
        )
    c = tuple(basis)
    dimension = len(c)
    output_rank = levels[0].rank_after
    columns, chosen = [], []
    zero, one = Fraction(0), Fraction(1)
    if dimension:
        for j in range(n):
            column = tuple(row[j] for row in c)
            observed_rank = rank((*columns, column))
            if observed_rank not in (len(columns), len(columns) + 1):
                raise RuntimeError("exact pivot-column selection lost rank consistency")
            if observed_rank > len(columns):
                columns.append(column)
                chosen.append(j)
            if len(columns) == dimension:
                break
        if len(columns) != dimension:
            raise RuntimeError(
                "independent observation rows have no invertible column minor"
            )
        minor = tuple(tuple(row[j] for j in chosen) for row in c)
        inverse = retain(exact_matrix_inverse(minor))
        selected_positions = {j: i for i, j in enumerate(chosen)}
        t = retain(
            tuple(
                (
                    inverse[selected_positions[i]]
                    if i in selected_positions
                    else (zero,) * dimension
                )
                for i in range(n)
            )
        )
        ca = product(c, a)
        g, d = product(ca, t), product(r, t)
    else:
        # Empty tuples alone lose their column count. Handle the known
        # 0-by-n, n-by-0, 0-by-0 and m-by-0 shapes explicitly rather than
        # changing the nonempty shared product/inverse contracts.
        t = ((),) * n
        ca, g, d = (), (), ((),) * m
    identity = tuple(
        tuple(one if i == j else zero for j in range(dimension))
        for i in range(dimension)
    )
    checks = {
        "rank(C)=dimension": rank(c) == dimension,
        "C T=I": (product(c, t) if dimension else ()) == identity,
        "C J=G C": ca == (product(g, c) if dimension else ()),
        "O=D C": r == (product(d, c) if dimension else ((zero,) * n,) * m),
        "complete-level stabilization": levels[-1].rank_before == levels[-1].rank_after,
        "initial output rank retained": output_rank <= dimension,
    }
    failed = tuple(name for name, passed in checks.items() if not passed)
    if failed:
        raise RuntimeError(f"exact sufficient-observation identities failed: {failed}")
    return LinearObservation(
        generator=a,
        output_rows=r,
        output_count=m,
        output_rank=output_rank,
        dimension=dimension,
        full_state_dimension=n,
        extra_coordinates=dimension - output_rank,
        rank_progression=tuple(level.rank_after for level in levels),
        level_records=tuple(levels),
        selected_row_labels=tuple(labels),
        pivot_columns=tuple(chosen),
        observation=c,
        right_inverse=t,
        reduced_generator=g,
        output_map=d,
        exact_identity_checks=tuple(checks),
        rank_calls=rank_calls,
        max_rank_calls=max_rank_calls,
        matrix_product_calls=product_calls,
        completed_levels=len(levels),
        stabilization_power=levels[-1].power,
        max_coefficient_bits=max_bits,
        scope=(
            "Minimal all-state linear observation of one supplied fixed rational "
            "generator with positive sign z'=J*z. No nonlinear/reachable-state "
            "minimality, graph authentication, constitutive selection, trajectory "
            "or physical identity is certified."
        ),
    )
