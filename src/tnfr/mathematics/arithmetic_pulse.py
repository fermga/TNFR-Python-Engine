r"""Arithmetic pulse recurrence on declared pointed residue networks (R2).

The caller supplies the arithmetic carrier, power-residue connection set and
localized seed e_0. For the resulting circulant L_rw, the scalar moments
mu_m=e_0^T L_rw^m e_0 have nonzero spectral weights m_lambda/n on every distinct
eigenvalue. Hence their Hankel rank equals the reachable Krylov dimension and
the distinct-eigenvalue count. At a prime p, classical cyclotomy gives the
count gcd(k,p-1)+1; composite constructions are controls, not a general converse.

Hankel rank need not equal Krylov dimension for an arbitrary square matrix:
the selected output can miss reachable modes. The circulant input/output
participation is an essential hypothesis here. Rank computations use exact
rational arithmetic; a numerical eigencount is a separate comparison.

The optional interpretation h(t)=e_0^T exp(-nu_f*L*t)e_0 is a declared fixed
linear response. It derives neither autonomous AL occurrence nor a phase law,
sustained oscillation or an arithmetic substrate from the nodal identity.
Building an n-node network is exponential in the bit length of n, so this
construction supplies no factoring or primality speedup."""

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
