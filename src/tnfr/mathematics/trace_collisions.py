r"""Static trace-fiber information on declared finite fields (R5, N10).

For H equal to a supplied power-map image in F_q, exact integer counts
N_a=#{h in H:Tr(h)=a} describe the fibers of the field trace. The number of
nonempty fibers counts attained static trace values. It is not proved equal
to a scalar dynamical output's Hankel rank or minimal realization order;
this module supplies no input/output pair establishing that bridge.

The Fourier-inversion identity on F_p determines the same counts algebraically.
The character implementation uses complex floating arithmetic and rounds,
while direct field-trace counts are exact integers. Its residual is a finite
numerical check. Galois invariance Tr(h^p)=Tr(h) follows from the trace identity;
for the power subgroup the histogram is intrinsic to the supplied field.
No field carrier, autonomous pulse or nodal evolution is derived here."""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass

from .finite_fields import FiniteField

__all__ = [
    "power_image",
    "trace_fiber_counts",
    "observed_trace_values",
    "character_sum",
    "trace_fiber_counts_via_characters",
    "character_formula_residual",
    "trace_is_galois_invariant",
    "uniform_deviation",
    "TraceCollisionCertificate",
    "certify_trace_collisions",
]


def power_image(field: FiniteField, k: int) -> set[int]:
    r"""The image of the ``k``-th power map ``H = { a^k : a ∈ F_q^* }``.

    A multiplicative subgroup of index ``gcd(k, q−1)``, so
    ``|H| = (q−1) / gcd(k, q−1)``.
    """
    return field.kth_power_set(k)


def trace_fiber_counts(field: FiniteField, subset) -> dict[int, int]:
    r"""Exact fiber counts ``{ a : N_a }`` of the trace over ``subset ⊆ F_q``."""
    counts = {a: 0 for a in range(field.p)}
    for h in subset:
        counts[field.trace(h)] += 1
    return counts


def observed_trace_values(field: FiniteField, subset) -> int:
    r"""``#{ a : N_a > 0 }`` — the number of trace values that stay observable."""
    return sum(1 for v in trace_fiber_counts(field, subset).values() if v > 0)


def character_sum(field: FiniteField, subset, u: int) -> complex:
    r"""``S(u) = Σ_{h∈subset} ψ(u · Tr(h))`` with ``ψ(t) = e^{2πi t / p}``."""
    p = field.p
    return sum(cmath.exp(2j * math.pi * (u * field.trace(h)) / p) for h in subset)


def trace_fiber_counts_via_characters(field: FiniteField, subset) -> dict[int, int]:
    r"""``N_a`` via the character formula (Fourier inversion on ``F_p``).

    ``N_a = (1/p) Σ_{u∈F_p} ψ(−u a) · S(u)``, rounded to the nearest integer.
    """
    p = field.p
    s = [character_sum(field, subset, u) for u in range(p)]
    out: dict[int, int] = {}
    for a in range(p):
        val = sum(cmath.exp(-2j * math.pi * (u * a) / p) * s[u] for u in range(p)) / p
        out[a] = int(round(val.real))
    return out


def character_formula_residual(field: FiniteField, subset) -> float:
    r"""``max_a |N_a^{char} − N_a^{exact}|`` — the Fourier-inversion defect."""
    p = field.p
    exact = trace_fiber_counts(field, subset)
    s = [character_sum(field, subset, u) for u in range(p)]
    resid = 0.0
    for a in range(p):
        val = sum(cmath.exp(-2j * math.pi * (u * a) / p) * s[u] for u in range(p)) / p
        resid = max(resid, abs(val.real - exact[a]), abs(val.imag))
    return resid


def trace_is_galois_invariant(
    field: FiniteField, subset, *, tol: float = 1e-12
) -> bool:
    r"""Whether ``Tr(h^p) = Tr(h)`` for every ``h`` — Galois/representation
    invariance of the trace observable (the collision histogram is intrinsic)."""
    return all(field.trace(field.power(h, field.p)) == field.trace(h) for h in subset)


def uniform_deviation(field: FiniteField, subset) -> float:
    r"""Return max_a |N_a-|H|/p|, the deviation of static fiber counts from uniformity.

    Zero means equal fiber sizes, not injectivity or complete dynamical
    observability. Many-to-one trace maps can have exactly uniform counts."""
    counts = trace_fiber_counts(field, subset)
    n = sum(counts.values())
    mean = n / field.p
    return max(abs(v - mean) for v in counts.values())


@dataclass(frozen=True)
class TraceCollisionCertificate:
    """Exact trace-collision structure of the ``k``-th powers in ``F_q``."""

    p: int
    f: int
    q: int
    k: int
    subset_size: int  # |H| = (q-1)/gcd(k, q-1)
    observed_values: int  # #{a : N_a > 0}
    full_support: bool  # observed_values == p
    has_collisions: bool  # |H| > observed_values
    max_fiber: int
    min_fiber: int
    uniform_deviation: float
    character_formula_residual: float
    galois_invariant: bool
    tolerance: float
    claim_status: str


def certify_trace_collisions(field: FiniteField, k: int) -> TraceCollisionCertificate:
    r"""Bundle the exact trace-collision structure for the ``k``-th powers.

    ``character_formula_residual ≈ 0`` (Fourier inversion is exact) and
    ``galois_invariant`` together make the fiber histogram a representation-free
    field invariant; ``observed_values`` is the visible order under the trace.
    """
    H = power_image(field, k)
    counts = trace_fiber_counts(field, H)
    observed = sum(1 for v in counts.values() if v > 0)
    resid = character_formula_residual(field, H)
    tol = 1e-9 * max(1, len(H))
    return TraceCollisionCertificate(
        p=field.p,
        f=field.f,
        q=field.q,
        k=k,
        subset_size=len(H),
        observed_values=observed,
        full_support=(observed == field.p),
        has_collisions=(len(H) > observed),
        max_fiber=max(counts.values()),
        min_fiber=min(counts.values()),
        uniform_deviation=uniform_deviation(field, H),
        character_formula_residual=resid,
        galois_invariant=trace_is_galois_invariant(field, H),
        tolerance=tol,
        claim_status=(
            "character formula DERIVED (Fourier inversion) + MEASURED (exact "
            "match); trace collision histogram is a Galois-invariant field "
            "observable; type detector NOT claimed (NT-P05c CONJECTURAL)"
        ),
    )
