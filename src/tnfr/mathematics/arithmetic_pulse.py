r"""Arithmetic pulse recurrence on pointed residue networks (R2).

The **static** cyclotomy law (``TNFR_NUMBER_THEORY.md``) says the k-th power
residue Cayley operator ``L_rw`` on ``ℤ/pℤ`` (``p`` prime) has exactly
``s_k(p) = gcd(k, p−1) + 1`` distinct eigenvalues.  This module gives that rank a
**dynamic** reading through the *pointed* network ``(G_{p,k}, 0)`` seeded at the
additive identity ``e₀ = (1,0,…,0)ᵀ`` (the point is the neutral element, so it is
chosen without using any factor):

* the pulse moments ``μ_m = e₀ᵀ L_rw^m e₀`` obey a linear recurrence;
* its order is the **Krylov dimension** ``dim span{e₀, L e₀, L² e₀, …}`` =
  (Kronecker) the **Hankel rank** ``rank[μ_{i+j}]``;
* because ``L_rw`` is a circulant, ``e₀`` excites every Fourier mode, so that
  dimension equals the number of distinct eigenvalues.

Hence, on the stated domain,

    Hankel rank = Krylov dimension = #distinct eigenvalues = gcd(k, p−1) + 1,

converting the static spectral rank into the order of the temporal pulse
``h(t) = e₀ᵀ e^{−ν_f L t} e₀ = Σ_j a_j e^{−ν_f λ_j t}``.  Everything here is exact
rational linear algebra (:mod:`tnfr.mathematics.krylov`); composites are controls
outside the theorem.  Honest scope: this is a **recurrence-order** identity, not a
fast primality test — building the ``p``-node network is exponential in the input
size ``log₂ p``.
"""

from __future__ import annotations

from fractions import Fraction
from math import gcd

from .cayley import cayley_laplacian
from .krylov import hankel_rank, krylov_dimension
from .number_theory import power_residue_set

__all__ = [
    "power_residue_laplacian",
    "cyclotomic_rank",
    "pointed_pulse_hankel_rank",
    "pointed_pulse_krylov_dimension",
    "pulse_recurrence_matches_cyclotomy",
]


def power_residue_laplacian(p: int, k: int) -> list[list[Fraction]]:
    r"""Exact ``L_rw = I − (1/d) W`` for the k-th power residue Cayley digraph on
    ``ℤ/pℤ`` (out-degree ``d = |R_k|``; edge ``i→j`` iff ``(j−i) mod p ∈ R_k``)."""
    return cayley_laplacian(p, set(power_residue_set(p, k)))


def cyclotomic_rank(p: int, k: int) -> int:
    r"""The claimed pulse rank ``s_k(p) = gcd(k, p−1) + 1``."""
    return gcd(k, p - 1) + 1


def _e0(p: int) -> list[Fraction]:
    e = [Fraction(0)] * p
    e[0] = Fraction(1)
    return e


def pointed_pulse_hankel_rank(p: int, k: int) -> int:
    r"""Hankel rank of the pointed pulse ``μ_m = e₀ᵀ L^m e₀`` (exact)."""
    return hankel_rank(power_residue_laplacian(p, k), _e0(p))


def pointed_pulse_krylov_dimension(p: int, k: int) -> int:
    r"""Krylov dimension ``dim span{e₀, L e₀, …}`` of the pointed pulse (exact)."""
    return krylov_dimension(power_residue_laplacian(p, k), _e0(p))


def pulse_recurrence_matches_cyclotomy(p: int, k: int) -> bool:
    r"""Whether Hankel rank = Krylov dimension = ``gcd(k, p−1) + 1`` for ``(p, k)``."""
    target = cyclotomic_rank(p, k)
    return (
        pointed_pulse_hankel_rank(p, k) == target
        and pointed_pulse_krylov_dimension(p, k) == target
    )
