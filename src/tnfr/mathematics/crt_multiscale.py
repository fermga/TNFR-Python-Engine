r"""CRT multiscale composition of residue networks (R3).

This module realises the Chinese Remainder Theorem (CRT) as the **U5 multiscale**
axiom of TNFR: for coprime moduli ``a, b`` the additive group factors
``ℤ/abℤ ≅ ℤ/aℤ × ℤ/bℤ``, and — restricted to the **unit** power-residue
connection set — the Cayley operator factors as an exact Kronecker product.

Writing ``L_m`` for the random-walk Laplacian ``I − (1/d) W`` of the Cayley
digraph ``Cay(ℤ/mℤ, S_m)`` with ``S_m = unit_power_residue_set(m, k)``, the CRT
permutation ``σ`` (node ``r ↦ (r mod a, r mod b)``) gives, **exactly over ℚ**,

    A_ab = A_a ⊗ A_b,      P_ab = P_a ⊗ P_b,
    L_ab = I − (I − L_a) ⊗ (I − L_b)          (up to σ),

because the unit restriction makes ``S_ab`` CRT-factor as ``S_a × S_b`` (the CRT
bijection of unit groups ``(ℤ/abℤ)^* ≅ (ℤ/aℤ)^* × (ℤ/bℤ)^*``).  From the Kronecker
structure the parent spectrum is the **child eigenvalue composition**

    λ_parent = λ + μ − λμ = 1 − (1 − λ)(1 − μ),

exactly.  Because every child eigenvalue embeds in the parent through the trivial
mode (``μ = 0``), the parent spectral gap obeys the exact U5 bound
``λ₂(ab) ≤ min(λ₂(a), λ₂(b))`` — composing scales never speeds up the slowest
sub-EPI, it only adds slower cross-scale modes (with equality for real,
symmetric-connection spectra).

**Scope (honest).**  This is a *structural* branch: it **uses the known factors**
``a, b`` to exhibit how sub-networks assemble into the whole — it is **not** a
factoring algorithm and makes **no** complexity, cryptographic, or Millennium
claim.  The exact algebraic product theorem (Kronecker identity + eigenvalue law)
is stated and tested **separately** from any inverse/factoring use.  The
non-factorizing control is :func:`full_power_residue_laplacian`: the *unrestricted*
power-residue set (non-unit entries included) does **not** CRT-factor in general,
so the Kronecker identity fails for it — the unit restriction is necessary.
"""

from __future__ import annotations

import math
from fractions import Fraction
from typing import Sequence

import numpy as np

from ..physics.spectral_projectors import derived_tolerance
from .cayley import cayley_laplacian
from .number_theory import power_residue_set, unit_power_residue_set

__all__ = [
    "unit_power_residue_laplacian",
    "full_power_residue_laplacian",
    "crt_ordering",
    "kron",
    "crt_kronecker_residual",
    "residue_set_factors",
    "child_parent_eigenvalue",
    "composed_spectrum",
    "verify_spectrum_composition",
    "spectral_gap",
    "u5_spectral_gap_composition",
]

Matrix = list[list[Fraction]]


def unit_power_residue_laplacian(m: int, k: int) -> Matrix:
    r"""Exact ``L_rw`` for the **unit** k-th power-residue Cayley digraph on
    ``ℤ/mℤ`` (connection set :func:`unit_power_residue_set`)."""
    return cayley_laplacian(m, set(unit_power_residue_set(m, k)))


def full_power_residue_laplacian(m: int, k: int) -> Matrix:
    r"""Exact ``L_rw`` for the **unrestricted** k-th power-residue Cayley digraph
    on ``ℤ/mℤ`` (connection set :func:`power_residue_set`).

    This is the *control* operator: because :func:`power_residue_set` includes
    non-unit residues, its connection set does not CRT-factor in general, so the
    Kronecker identity :func:`crt_kronecker_residual` does **not** hold for it.
    """
    return cayley_laplacian(m, set(power_residue_set(m, k)))


def crt_ordering(a: int, b: int) -> list[int]:
    r"""CRT permutation ``σ`` for coprime ``a, b``.

    Returns a list ``perm`` of length ``a*b`` where ``perm[i*b + j]`` is the unique
    node ``r ∈ ℤ/abℤ`` with ``r ≡ i (mod a)`` and ``r ≡ j (mod b)`` — i.e. the map
    from Kronecker (row-major ``(i, j)``) order to natural ``ℤ/abℤ`` order.
    """
    if math.gcd(a, b) != 1:
        raise ValueError(f"CRT requires coprime moduli; gcd({a}, {b}) != 1")
    ab = a * b
    inv_b = pow(b, -1, a)  # b^{-1} mod a
    inv_a = pow(a, -1, b)  # a^{-1} mod b
    perm = [0] * ab
    for i in range(a):
        term_i = i * b * inv_b
        for j in range(b):
            perm[i * b + j] = (term_i + j * a * inv_a) % ab
    return perm


def kron(A: Matrix, B: Matrix) -> Matrix:
    r"""Exact Kronecker product ``A ⊗ B`` over ℚ."""
    na, ma = len(A), len(A[0])
    nb, mb = len(B), len(B[0])
    out: Matrix = [
        [Fraction(0) for _ in range(ma * mb)] for _ in range(na * nb)
    ]
    for ia in range(na):
        for ja in range(ma):
            aij = A[ia][ja]
            if aij == 0:
                continue
            for ib in range(nb):
                row = ia * nb + ib
                for jb in range(mb):
                    out[row][ja * mb + jb] = aij * B[ib][jb]
    return out


def _identity(n: int) -> Matrix:
    return [[Fraction(1) if i == j else Fraction(0) for j in range(n)]
            for i in range(n)]


def _sub(A: Matrix, B: Matrix) -> Matrix:
    return [[A[i][j] - B[i][j] for j in range(len(A[0]))] for i in range(len(A))]


def crt_kronecker_residual(a: int, b: int, k: int) -> Fraction:
    r"""Exact max-abs residual of the CRT Kronecker identity.

    Builds ``L_ab`` on ``ℤ/abℤ`` and the composed operator
    ``I − (I − L_a) ⊗ (I − L_b)``, reorders ``L_ab`` by the CRT permutation
    :func:`crt_ordering`, and returns ``max |L_ab[σ(u)][σ(v)] − composed[u][v]|``
    over ℚ.  Zero **iff** the identity holds exactly (coprime ``a, b``, unit set).
    """
    La = unit_power_residue_laplacian(a, k)
    Lb = unit_power_residue_laplacian(b, k)
    Lab = unit_power_residue_laplacian(a * b, k)
    Ia, Ib = _identity(a), _identity(b)
    composed = _sub(_identity(a * b), kron(_sub(Ia, La), _sub(Ib, Lb)))
    perm = crt_ordering(a, b)
    worst = Fraction(0)
    ab = a * b
    for u in range(ab):
        pu = perm[u]
        for v in range(ab):
            diff = Lab[pu][perm[v]] - composed[u][v]
            if diff < 0:
                diff = -diff
            if diff > worst:
                worst = diff
    return worst


def residue_set_factors(a: int, b: int, k: int, *, unit: bool = True) -> bool:
    r"""Whether the ``k``-th power-residue set of ``ab`` CRT-factors as a product.

    Returns ``True`` iff the CRT image ``{(s mod a, s mod b) : s ∈ S_ab}`` equals
    the full product ``S_a × S_b``.  With ``unit=True`` (the theorem) this holds
    for every coprime ``a, b``; with ``unit=False`` (the control) it generally
    fails, because non-unit residues do not respect the CRT unit bijection.
    """
    chooser = unit_power_residue_set if unit else power_residue_set
    sa = set(chooser(a, k))
    sb = set(chooser(b, k))
    sab = set(chooser(a * b, k))
    image = {(s % a, s % b) for s in sab}
    return image == {(x, y) for x in sa for y in sb}


def child_parent_eigenvalue(lam: complex, mu: complex) -> complex:
    r"""Child→parent eigenvalue composition ``λ + μ − λμ = 1 − (1−λ)(1−μ)``."""
    return lam + mu - lam * mu


def composed_spectrum(
    spec_a: Sequence[complex], spec_b: Sequence[complex]
) -> list[complex]:
    r"""All parent eigenvalues ``{λ + μ − λμ}`` from child spectra."""
    return [child_parent_eigenvalue(la, mu) for la in spec_a for mu in spec_b]


def _to_float(L: Matrix) -> np.ndarray:
    return np.array([[float(x) for x in row] for row in L], dtype=float)


def _sorted_key(z: complex) -> tuple[float, float]:
    return (round(z.real, 12), round(z.imag, 12))


def verify_spectrum_composition(a: int, b: int, k: int) -> float:
    r"""Max eigenvalue mismatch between ``spec(L_ab)`` and the composed child
    spectra ``{λ + μ − λμ}`` (numerical corroboration of the exact identity).

    The tolerance is derived (:func:`derived_tolerance`), never a magic constant;
    the returned residual should sit well below it.
    """
    La = unit_power_residue_laplacian(a, k)
    Lb = unit_power_residue_laplacian(b, k)
    Lab = unit_power_residue_laplacian(a * b, k)
    eig_a = np.linalg.eigvals(_to_float(La))
    eig_b = np.linalg.eigvals(_to_float(Lb))
    eig_ab = sorted(np.linalg.eigvals(_to_float(Lab)), key=_sorted_key)
    composed = sorted(composed_spectrum(eig_a, eig_b), key=_sorted_key)
    return max(abs(x - y) for x, y in zip(eig_ab, composed))


def spectral_gap(L: Matrix, *, tol: float | None = None) -> float:
    r"""Slowest non-zero relaxation rate ``λ₂`` = smallest ``|λ| > 0`` of ``L``."""
    arr = _to_float(L)
    if tol is None:
        tol = derived_tolerance(arr)
    mags = sorted(abs(z) for z in np.linalg.eigvals(arr))
    for m in mags:
        if m > tol:
            return float(m)
    return 0.0


def u5_spectral_gap_composition(
    a: int, b: int, k: int
) -> tuple[float, float, float, bool]:
    r"""U5 read-out: parent spectral gap vs child gaps.

    Every child eigenvalue embeds in the parent spectrum through the trivial mode
    (``μ = 0`` gives ``λ + 0 − 0 = λ``), so the parent's non-zero spectrum contains
    the union of the child non-zero spectra **plus** genuinely multiscale cross
    modes ``λ + μ − λμ`` (``λ, μ ≠ 0``).  Hence the exact, always-true bound

        λ₂(ab) = min(λ₂(a), λ₂(b), min_{λ,μ≠0}|λ + μ − λμ|) ≤ min(λ₂(a), λ₂(b)),

    with equality when the spectra are real in ``[0, 1]`` (symmetric connection).
    Returns ``(gap_ab, gap_a, gap_b, bounded)`` where ``bounded`` checks the bound:
    composing scales never speeds up the slowest sub-EPI, it only adds slower
    cross-scale modes.
    """
    La = unit_power_residue_laplacian(a, k)
    Lb = unit_power_residue_laplacian(b, k)
    Lab = unit_power_residue_laplacian(a * b, k)
    gap_a = spectral_gap(La)
    gap_b = spectral_gap(Lb)
    gap_ab = spectral_gap(Lab)
    tol = derived_tolerance(_to_float(Lab))
    bounded = bool(gap_ab <= min(gap_a, gap_b) + tol)
    return gap_ab, gap_a, gap_b, bounded
