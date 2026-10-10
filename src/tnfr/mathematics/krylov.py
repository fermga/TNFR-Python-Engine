r"""Exact Krylov and Hankel rank utilities (R2 arithmetic-pulse support).

For rational ``L`` and seed ``v``, the reachable Krylov dimension is
``dim span{v, Lv, L²v, …}``. The scalar moment sequence ``μ_m = vᵀ Lᵐ v``
also depends on the output ``vᵀ``: its full Hankel rank counts the reachable
part visible to that output and can be smaller. For example,
``L = [[0, 0], [-1, 1]]`` and ``v = [1, 0]`` give Krylov dimension two but
moments ``(1, 0, 0, ...)`` and Hankel rank one.

Equality holds for a real symmetric operator with the same nonzero seed as
input and output, or more generally when the output is observable on the
reachable subspace. The pointed circulant arithmetic fixtures establish
their own equality through nonzero Fourier participation in both input and
output. A reversible operator needs its appropriate metric or a separately
proved observability condition; these functions do not infer one.

The functions compute ranks separately over ℚ using :class:`fractions.Fraction`.
A user-shortened moment sequence can undercount the full Hankel rank.
Integer and rational inputs remain exact; other admitted real coefficients
retain their binary64 values before rational arithmetic. Such a represented
matrix does not certify an ideal irrational model. No physical-state or
particle interpretation follows from these finite linear-algebra identities.
"""

from __future__ import annotations

from fractions import Fraction
from numbers import Real
from typing import Sequence

from ._exact_linear_algebra import _rational_matrix, _rational_vector
from ._integer_admission import _integer_argument

__all__ = [
    "matvec",
    "moment_sequence",
    "hankel_matrix",
    "exact_rank",
    "krylov_dimension",
    "hankel_rank",
]

Matrix = Sequence[Sequence[Real]]
Vector = Sequence[Real]


def _system(L, v):
    matrix = _rational_matrix(L, "matrix", allow_empty=True)
    vector = _rational_vector(v, "vector")
    if len(matrix) != len(vector) or any(len(row) != len(vector) for row in matrix):
        raise ValueError("matrix must be square and match the vector dimension")
    return matrix, vector


def _count(value):
    count = _integer_argument(value, "count")
    if count < 0:
        raise ValueError("count must be nonnegative")
    return count


def _matvec(L, v):
    return [sum((a * b for a, b in zip(row, v)), Fraction(0)) for row in L]


def matvec(L: Matrix, v: Vector) -> list[Fraction]:
    """Square matrix-vector product after shared real admission (exact)."""
    matrix, vector = _system(L, v)
    return _matvec(matrix, vector)


def _dot(u: Vector, v: Vector) -> Fraction:
    return sum((a * b for a, b in zip(u, v)), Fraction(0))


def moment_sequence(L: Matrix, v: Vector, count: int) -> list[Fraction]:
    """The moments ``μ_m = vᵀ Lᵐ v`` for ``m = 0 … count−1`` (exact)."""
    count = _count(count)
    matrix, vector = _system(L, v)
    return _moments(matrix, vector, count)


def _moments(L, v, count):
    """Advance already admitted coordinates only as far as the last moment."""
    moments: list[Fraction] = []
    w = list(v)
    for index in range(count):
        moments.append(_dot(v, w))
        if index + 1 < count:
            w = _matvec(L, w)
    return moments


def hankel_matrix(moments: Sequence[Fraction]) -> list[list[Fraction]]:
    """The ``r×r`` Hankel matrix ``H_{ij} = μ_{i+j}`` (``r = ⌈len/2⌉``)."""
    moments = _rational_vector(moments, "moments")
    r = (len(moments) + 1) // 2
    return [[moments[i + j] for j in range(r)] for i in range(r)]


def exact_rank(matrix: Matrix) -> int:
    """Exact rank of the admitted rational matrix, including empty shapes.

    Integer division must not introduce floating arithmetic. Reject ragged
    rows and invalid scalars before pivot selection, even for zero columns.
    """
    admitted = _rational_matrix(matrix, "matrix", allow_empty=True)
    M = [list(row) for row in admitted]
    rows = len(M)
    cols = len(M[0]) if rows else 0
    if any(len(row) != cols for row in M):
        raise ValueError("matrix must be rectangular")
    used = [False] * rows
    rank = 0
    for col in range(cols):
        piv = next((r for r in range(rows) if not used[r] and M[r][col] != 0), None)
        if piv is None:
            continue
        used[piv] = True
        rank += 1
        pivot = M[piv][col]
        M[piv] = [x / pivot for x in M[piv]]
        for r in range(rows):
            if r != piv and M[r][col] != 0:
                factor = M[r][col]
                M[r] = [a - factor * b for a, b in zip(M[r], M[piv])]
    return rank


def krylov_dimension(L: Matrix, v: Vector) -> int:
    """``dim span{v, Lv, L²v, …}`` (exact incremental row reduction)."""
    L, v = _system(L, v)
    n = len(v)
    pivots: list[tuple[int, list[Fraction]]] = []

    def reduce(row: list[Fraction]) -> list[Fraction]:
        row = list(row)
        for col, prow in pivots:
            if row[col] != 0:
                factor = row[col]
                row = [a - factor * b for a, b in zip(row, prow)]
        return row

    w = list(v)
    dim = 0
    for _ in range(n):
        r = reduce(list(w))
        nz = next((c for c in range(n) if r[c] != 0), None)
        if nz is None:
            break
        r = [x / r[nz] for x in r]
        pivots.append((nz, r))
        dim += 1
        if dim < n:
            w = _matvec(L, w)
    return dim


def hankel_rank(L: Matrix, v: Vector, *, count: int | None = None) -> int:
    """Rank of the Hankel moment matrix of ``(L, v)`` (exact)."""
    L, v = _system(L, v)
    n = len(v)
    if count is None:
        count = max(0, 2 * n - 1)
    count = _count(count)
    return exact_rank(hankel_matrix(_moments(L, v, count)))
