r"""Finite-field residue networks and trace additive characters (R5).

R2 proved that on the k-th power residue Cayley network over the **prime** field
``F_p`` the pulse rank is the cyclotomy count ``s_k(p) = gcd(k, p−1) + 1``.  This
module extends the construction to a general finite field ``F_q`` (``q = p^f``)
via the additive characters built from the field **trace**

    ψ_a(x) = exp(2πi/p · Tr_{F_q/F_p}(a x)),      Tr(y) = y + y^p + ⋯ + y^{p^{f−1}},

and asks when the number of distinct Gauss periods still equals
``gcd(k, q−1) + 1``.  The eigenvalues of the additive Cayley graph
``Cay(F_q, S)`` with ``S`` the non-zero k-th powers are exactly the normalised
periods ``η_a = (1/|S|) Σ_{s∈S} ψ_a(s)``, constant on cosets of the k-th power
subgroup.

**Measured result (honest).**  For ``f = 1`` the count matches ``gcd(k, p−1)+1``
(R2 regression, exact).  For extensions ``f ≥ 2`` the trace is many-to-one, so
distinct cosets can share a period: the count is ``≤ gcd(k, q−1) + 1`` and is
**strictly smaller** for some ``(p, f, k)`` (e.g. ``F_9, k=4``: 5 → 2).  The
prime-field independence argument does **not** transfer unchanged.

No external field package is required (pure Python + numpy); extension degrees
``f ≤ 3`` are supported via a no-root irreducibility search.
"""

from __future__ import annotations

import cmath
import itertools
import math
from math import gcd

import numpy as np

__all__ = [
    "FiniteField",
    "presentation_isomorphism",
    "additive_character",
    "gauss_period",
    "normalized_periods",
    "distinct_period_count",
    "cyclotomic_period_count",
    "prime_field_matches_cyclotomy",
    "explicit_cayley_spectrum_count",
]


def _poly_eval(coeffs: list[int], x: int, p: int) -> int:
    v = 0
    for c in reversed(coeffs):
        v = (v * x + c) % p
    return v


def _find_irreducible(p: int, f: int) -> list[int]:
    r"""A monic degree-``f`` irreducible polynomial over ``F_p`` (``f ≤ 3``).

    For ``f ∈ {2, 3}`` a monic polynomial is irreducible iff it has no root, so a
    root-free search suffices; degree ``f ≥ 4`` is not supported here.
    """
    if f == 1:
        return [0, 1]
    if f > 3:
        raise NotImplementedError(
            "extension degree f > 3 unsupported; supply an irreducible poly"
        )
    for tail in itertools.product(range(p), repeat=f):
        coeffs = list(tail) + [1]
        if all(_poly_eval(coeffs, x, p) != 0 for x in range(p)):
            return coeffs
    raise RuntimeError(f"no irreducible polynomial found for F_{p}^{f}")


class FiniteField:
    r"""The finite field ``F_q`` with ``q = p^f`` (``f ≤ 3``).

    Elements are integers ``0 … q−1`` encoding degree-``<f`` polynomials over
    ``F_p`` in base ``p``.  Arithmetic is exact; the trace lands in ``F_p``.
    """

    def __init__(
        self,
        p: int,
        f: int = 1,
        *,
        modulus: list[int] | tuple[int, ...] | None = None,
    ) -> None:
        if p < 2:
            raise ValueError("p must be a prime >= 2")
        if f < 1:
            raise ValueError("extension degree f must be >= 1")
        self.p = int(p)
        self.f = int(f)
        self.q = p ** f
        if modulus is None:
            self.modulus = _find_irreducible(p, f)
        else:
            coefficients = [int(value) % p for value in modulus]
            if len(coefficients) != f + 1 or coefficients[-1] != 1:
                raise ValueError("modulus must be monic of declared degree")
            if f == 1:
                if coefficients != [0, 1]:
                    raise ValueError("prime fields use the canonical x modulus")
            elif f > 3:
                raise NotImplementedError(
                    "extension degree f > 3 unsupported"
                )
            elif any(
                _poly_eval(coefficients, value, p) == 0
                for value in range(p)
            ):
                raise ValueError("modulus must be irreducible over F_p")
            self.modulus = coefficients

    def _to_list(self, a: int) -> list[int]:
        d = []
        for _ in range(self.f):
            d.append(a % self.p)
            a //= self.p
        return d

    def _to_int(self, d: list[int]) -> int:
        v = 0
        for c in reversed(d):
            v = v * self.p + (c % self.p)
        return v

    @property
    def one(self) -> int:
        return 1

    def add(self, a: int, b: int) -> int:
        da, db = self._to_list(a), self._to_list(b)
        return self._to_int([(x + y) % self.p for x, y in zip(da, db)])

    def mul(self, a: int, b: int) -> int:
        if self.f == 1:
            return (a * b) % self.p
        da, db = self._to_list(a), self._to_list(b)
        p, f = self.p, self.f
        prod = [0] * (2 * f)
        for i in range(f):
            if da[i]:
                for j in range(f):
                    prod[i + j] = (prod[i + j] + da[i] * db[j]) % p
        for deg in range(2 * f - 1, f - 1, -1):
            c = prod[deg]
            if c:
                for k in range(f + 1):
                    prod[deg - f + k] = (
                        prod[deg - f + k] - c * self.modulus[k]
                    ) % p
        return self._to_int(prod[:f])

    def power(self, a: int, n: int) -> int:
        r = 1 if self.f == 1 else self._to_int([1] + [0] * (self.f - 1))
        base = a
        while n > 0:
            if n & 1:
                r = self.mul(r, base)
            base = self.mul(base, base)
            n >>= 1
        return r

    def trace(self, a: int) -> int:
        r"""``Tr_{F_q/F_p}(a) = a + a^p + ⋯ + a^{p^{f−1}} ∈ F_p`` (returned as int)."""
        s = 0
        cur = a
        for _ in range(self.f):
            s = self.add(s, cur)
            cur = self.power(cur, self.p)
        coeffs = self._to_list(s)
        return coeffs[0]  # trace is a constant polynomial (in F_p)

    def elements(self) -> range:
        return range(self.q)

    def kth_power_set(self, k: int) -> set[int]:
        r"""The non-zero ``k``-th powers ``{a^k : a ∈ F_q^*}``."""
        if k < 1:
            raise ValueError("power k must be >= 1")
        return {self.power(a, k) for a in range(1, self.q)} - {0}


def _evaluate_in_field(
    coefficients: list[int], value: int, field: FiniteField
) -> int:
    result = 0
    for coefficient in reversed(coefficients):
        result = field.add(field.mul(result, value), coefficient % field.p)
    return result


def presentation_isomorphism(
    source: FiniteField, target: FiniteField
) -> tuple[int, ...]:
    """Return the explicit base-field-preserving map between presentations.

    The source polynomial-basis generator is sent to the first target-field
    root of the source modulus.  The result maps integer encodings; it is a
    deterministic presentation control, not a preferred canonical basis.
    """
    if source.p != target.p or source.f != target.f:
        raise ValueError("presentations must have the same p and degree")
    roots = [
        value for value in target.elements()
        if _evaluate_in_field(source.modulus, value, target) == 0
    ]
    if not roots:
        raise ValueError("no presentation isomorphism root found")
    root = roots[0]
    mapping = []
    for element in source.elements():
        coefficients = source._to_list(element)
        mapping.append(_evaluate_in_field(coefficients, root, target))
    if len(set(mapping)) != source.q:
        raise ValueError("presentation map is not bijective")
    return tuple(mapping)


def additive_character(field: FiniteField, a: int, x: int) -> complex:
    r"""Trace additive character ``ψ_a(x) = exp(2πi/p · Tr(a x))``."""
    t = field.trace(field.mul(a, x))
    return cmath.exp(2j * math.pi * t / field.p)


def gauss_period(field: FiniteField, k: int, a: int) -> complex:
    r"""Unnormalised Gauss period ``Σ_{s∈S} ψ_a(s)`` (``S`` = non-zero k-th powers)."""
    S = field.kth_power_set(k)
    return sum(additive_character(field, a, s) for s in S)


def normalized_periods(field: FiniteField, k: int) -> list[complex]:
    r"""The Cayley eigenvalues ``η_a = (1/|S|) Σ_{s∈S} ψ_a(s)`` for all ``a``."""
    S = field.kth_power_set(k)
    d = len(S)
    inv = 1.0 / d
    out: list[complex] = []
    for a in field.elements():
        out.append(inv * sum(additive_character(field, a, s) for s in S))
    return out


def _count_distinct(values: list[complex], tol: float) -> int:
    uniq: list[complex] = []
    for v in values:
        if not any(abs(v - u) <= tol for u in uniq):
            uniq.append(v)
    return len(uniq)


def _derived_tol() -> float:
    # sqrt(machine epsilon): distinct Gauss periods of small fields separate far
    # above this, while accumulated character-sum error stays far below it.
    return math.sqrt(np.finfo(float).eps)


def distinct_period_count(field: FiniteField, k: int) -> int:
    r"""Number of distinct normalised periods ``η_a`` (the Cayley spectrum size)."""
    return _count_distinct(normalized_periods(field, k), _derived_tol())


def cyclotomic_period_count(q: int, k: int) -> int:
    r"""The prime-field prediction ``gcd(k, q−1) + 1`` (upper bound in general)."""
    return gcd(k, q - 1) + 1


def prime_field_matches_cyclotomy(p: int, k: int) -> bool:
    r"""Whether ``F_p`` reproduces R2: ``distinct = gcd(k, p−1) + 1`` (always True)."""
    field = FiniteField(p, 1)
    return distinct_period_count(field, k) == cyclotomic_period_count(p, k)


def explicit_cayley_spectrum_count(field: FiniteField, k: int) -> int:
    r"""Distinct eigenvalues of the explicit additive Cayley graph (cross-check).

    Builds ``A[u][v] = 1`` iff ``v − u ∈ S`` on the additive group and counts
    distinct eigenvalues; must agree with :func:`distinct_period_count`.
    """
    S = field.kth_power_set(k)
    q = field.q
    A = np.zeros((q, q), dtype=float)
    for u in range(q):
        for s in S:
            A[u][field.add(u, s)] = 1.0
    ev = np.linalg.eigvals(A)
    scale = max(1.0, float(np.max(np.abs(ev))) if q else 1.0)
    return _count_distinct(list(ev), _derived_tol() * scale)
