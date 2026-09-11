r"""Exact Krylov and Hankel rank utilities (R2 arithmetic-pulse support).

For an operator ``L`` and a seed vector ``v`` the moment sequence
``μ_m = vᵀ Lᵐ v`` obeys a linear recurrence whose order is the **Krylov
dimension** ``dim span{v, Lv, L²v, …}`` — equivalently (Kronecker's theorem) the
rank of the Hankel matrix ``H_{ij} = μ_{i+j}``.  These functions compute both
**exactly** over ℚ using :class:`fractions.Fraction`, so a small theorem case has
no floating-point ambiguity; the caller supplies rational ``L`` and ``v``.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Sequence

__all__ = [
    "matvec",
    "moment_sequence",
    "hankel_matrix",
    "exact_rank",
    "krylov_dimension",
    "hankel_rank",
]

Matrix = Sequence[Sequence[Fraction]]
Vector = Sequence[Fraction]


def matvec(L: Matrix, v: Vector) -> list[Fraction]:
    """Matrix-vector product ``L·v`` (exact)."""
    n = len(v)
    return [sum(L[i][j] * v[j] for j in range(n)) for i in range(n)]


def _dot(u: Vector, v: Vector) -> Fraction:
    return sum((a * b for a, b in zip(u, v)), Fraction(0))


def moment_sequence(L: Matrix, v: Vector, count: int) -> list[Fraction]:
    """The moments ``μ_m = vᵀ Lᵐ v`` for ``m = 0 … count−1`` (exact)."""
    moments: list[Fraction] = []
    w = list(v)
    for _ in range(count):
        moments.append(_dot(v, w))
        w = matvec(L, w)
    return moments


def hankel_matrix(moments: Sequence[Fraction]) -> list[list[Fraction]]:
    """The ``r×r`` Hankel matrix ``H_{ij} = μ_{i+j}`` (``r = ⌈len/2⌉``)."""
    r = (len(moments) + 1) // 2
    return [[moments[i + j] for j in range(r)] for i in range(r)]


def exact_rank(matrix: Matrix) -> int:
    """Exact rank over ℚ via fraction Gaussian elimination."""
    M = [list(row) for row in matrix]
    rows = len(M)
    cols = len(M[0]) if rows else 0
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
        w = matvec(L, w)
    return dim


def hankel_rank(L: Matrix, v: Vector, *, count: int | None = None) -> int:
    """Rank of the Hankel moment matrix of ``(L, v)`` (exact)."""
    n = len(v)
    if count is None:
        count = 2 * n - 1
    return exact_rank(hankel_matrix(moment_sequence(L, v, count)))
