r"""Finite CRT product identities for declared residue transport (R3).

For coprime supplied moduli a,b and unit k-th-power connection sets, CRT
identifies the parent connection set with S_a x S_b. Consequently, in CRT
order, P_ab=P_a tensor P_b and L_ab=I-(I-L_a) tensor (I-L_b), exactly over Q.
Parent eigenvalues are lambda+mu-lambda*mu. Every child eigenvalue embeds by
pairing with the other factor's zero mode. Other prescribed connection sets
can also factor; unit membership is a sufficient family, not a universal
necessary condition. The unrestricted power-residue family supplies failing
controls in selected cases.

The helper named spectral_gap returns the minimum computed eigenvalue modulus
above its tolerance. The corresponding exact nonzero-modulus bound follows
from child-mode embedding, excluding any zero cross modes. Directed heat-flow
decay depends on real parts, so this modulus is not generally a decay rate.
Real symmetric random-walk Laplacians have spectra in [0,2]; equality of the
child/parent modulus minimum follows under the stronger [0,1] hypothesis.

This is a finite construction using known factors. It does not execute U5,
REMESH, a joint phase/capacity/support evolution or a discovery algorithm.
Compatibility names containing U5 denote this limited product comparison."""

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
    out: Matrix = [[Fraction(0) for _ in range(ma * mb)] for _ in range(na * nb)]
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
    return [
        [Fraction(1) if i == j else Fraction(0) for j in range(n)] for i in range(n)
    ]


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
    r"""Return the smallest computed eigenvalue modulus above the selected tolerance.

    Return zero if none survives. For directed matrices this is not generally
    the slowest heat-decay rate, which depends on eigenvalue real parts. Numerical
    thresholding also differs from an exact minimum over nonzero eigenvalues."""
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
    r"""Compare thresholded parent and child nonzero-eigenvalue moduli.

    Return (gap_ab, gap_a, gap_b, bounded). In exact arithmetic child-mode
    embedding gives g_ab<=min(g_a,g_b), where g selects nonzero moduli. Zero cross
    modes must be excluded from that minimum and can add stationary directions.
    Equality follows if both child spectra lie in [0,1], not from symmetry alone.
    The returned boolean applies the numerical tolerance to this finite comparison;
    it does not certify U5, connectedness, a continuum limit or a directed decay
    rate."""
    La = unit_power_residue_laplacian(a, k)
    Lb = unit_power_residue_laplacian(b, k)
    Lab = unit_power_residue_laplacian(a * b, k)
    gap_a = spectral_gap(La)
    gap_b = spectral_gap(Lb)
    gap_ab = spectral_gap(Lab)
    tol = derived_tolerance(_to_float(Lab))
    bounded = bool(gap_ab <= min(gap_a, gap_b) + tol)
    return gap_ab, gap_a, gap_b, bounded
