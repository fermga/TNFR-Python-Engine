r"""Gaussian-integer residue networks and decomposition signatures (R5).

For a rational prime ``p`` the Gaussian-integer quotient ``Z[i]/(p)`` has three
classical shapes, fixed by ``p mod 4``:

* ``p = 2`` — **ramified**: ``(2) = −i(1+i)^2``; the quotient is a local ring with
  the nilpotent ``1+i``;
* ``p ≡ 1 (mod 4)`` — **split**: ``(p) = 𝔭 𝔭̄`` and ``Z[i]/(p) ≅ F_p × F_p``;
* ``p ≡ 3 (mod 4)`` — **inert**: ``(p)`` stays prime and ``Z[i]/(p) ≅ F_{p^2}``.

This module builds the additive Cayley network on ``(Z[i]/(p), +)`` (``p^2``
nodes) with connection set the non-zero (unit) ``k``-th powers, and asks whether
its spectrum distinguishes the three types.  The classical classification is used
only as a **ground-truth label** for scoring — never as an input to the observable
(a *descriptive* study in the sense of the C5 circularity audit).

**Measured result (honest).**  At ``k = 2`` the distinct-eigenvalue count of the
unit-power network separates the three types cleanly — ramified ``2``, inert
``3``, split ``6`` — on the tested primes.  The separation is ``k``-sensitive
(higher ``k`` does not give a universal detector) and tested only on small ``p``,
so the ``pulse-detects-decomposition`` claim (``NT-P05``) stays CONJECTURAL.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = [
    "decomposition_type",
    "gaussian_mul",
    "gaussian_elements",
    "gaussian_is_unit",
    "gaussian_kth_power_set",
    "gaussian_cayley_spectrum_count",
    "decomposition_signature",
    "signature_separates_types",
]

Gaussian = tuple[int, int]


def decomposition_type(p: int) -> str:
    r"""Classical splitting type of ``p`` in ``Z[i]`` (ground-truth label only)."""
    if p < 2:
        raise ValueError("p must be a prime >= 2")
    if p == 2:
        return "ramified"
    return "split" if p % 4 == 1 else "inert"


def gaussian_mul(u: Gaussian, v: Gaussian, p: int) -> Gaussian:
    r"""Product in ``Z[i]/(p)``: ``(a+bi)(c+di) = (ac−bd) + (ad+bc)i``."""
    a, b = u
    c, d = v
    return ((a * c - b * d) % p, (a * d + b * c) % p)


def gaussian_elements(p: int) -> list[Gaussian]:
    return [(a, b) for a in range(p) for b in range(p)]


def gaussian_is_unit(u: Gaussian, p: int) -> bool:
    r"""A residue is a unit iff its norm ``a^2 + b^2`` is invertible mod ``p``."""
    a, b = u
    return (a * a + b * b) % p != 0


def _gaussian_power(u: Gaussian, k: int, p: int) -> Gaussian:
    r = (1, 0)
    base = u
    while k > 0:
        if k & 1:
            r = gaussian_mul(r, base, p)
        base = gaussian_mul(base, base, p)
        k >>= 1
    return r


def gaussian_kth_power_set(
    p: int, k: int, *, units_only: bool = True
) -> set[Gaussian]:
    r"""Non-zero ``k``-th powers in ``Z[i]/(p)`` (units only by default)."""
    if k < 1:
        raise ValueError("power k must be >= 1")
    out: set[Gaussian] = set()
    for e in gaussian_elements(p):
        if e == (0, 0):
            continue
        if units_only and not gaussian_is_unit(e, p):
            continue
        r = _gaussian_power(e, k, p)
        if r != (0, 0):
            out.add(r)
    return out


def _count_distinct(values: np.ndarray, tol: float) -> int:
    uniq: list[complex] = []
    for v in values:
        if not any(abs(v - u) <= tol for u in uniq):
            uniq.append(complex(v))
    return len(uniq)


def gaussian_cayley_spectrum_count(
    p: int, k: int, *, units_only: bool = True
) -> int:
    r"""Distinct eigenvalue count of the additive ``k``-th power Cayley graph.

    Nodes are the ``p^2`` residues of ``Z[i]/(p)``; ``u → u + s`` for every
    connection ``s``.  The count is the decomposition observable.
    """
    els = gaussian_elements(p)
    idx = {e: i for i, e in enumerate(els)}
    q = len(els)
    S = gaussian_kth_power_set(p, k, units_only=units_only)
    A = np.zeros((q, q), dtype=float)
    for u in els:
        iu = idx[u]
        for s in S:
            v = ((u[0] + s[0]) % p, (u[1] + s[1]) % p)
            A[iu][idx[v]] = 1.0
    ev = np.linalg.eigvals(A)
    scale = max(1.0, float(np.max(np.abs(ev))) if q else 1.0)
    tol = math.sqrt(np.finfo(float).eps) * scale
    return _count_distinct(ev, tol)


def decomposition_signature(
    p: int, k: int = 2, *, units_only: bool = True
) -> tuple[str, int]:
    r"""``(ground_truth_type, spectrum_count)`` for prime ``p`` at power ``k``.

    The type is the classical label (scoring only); the count is the observable.
    """
    return (
        decomposition_type(p),
        gaussian_cayley_spectrum_count(p, k, units_only=units_only),
    )


def signature_separates_types(
    primes: list[int], k: int = 2, *, units_only: bool = True
) -> bool:
    r"""Whether the ``k`` spectrum count is constant within each decomposition
    type and distinct across types on the given primes (the NT-P05 test)."""
    by_type: dict[str, set[int]] = {}
    for p in primes:
        t, c = decomposition_signature(p, k, units_only=units_only)
        by_type.setdefault(t, set()).add(c)
    # each type must map to a single count, and the counts must be disjoint
    if any(len(counts) != 1 for counts in by_type.values()):
        return False
    singles = [next(iter(counts)) for counts in by_type.values()]
    return len(singles) == len(set(singles))
