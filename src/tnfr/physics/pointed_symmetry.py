r"""Pointed symmetry: the declared break Aut(G) → Γ_v (R1, selector stage).

The per-operator and word audits show the canonical operators are *equivariant as
maps*.  Symmetry breaking enters through the **selector**: choosing an origin
``v`` to act at.  A pointed action ``O@v`` on a symmetric seed cannot be
Aut(G)-equivariant unless ``v`` is a global fixed point — it can only respect the
**stabilizer**

    Γ_v = { g ∈ Aut(G) : g(v) = v } ≤ Aut(G).

This is the residual symmetry of the pointed graph ``(G, v)``.  Three facts make
the break *declared, not spontaneous*:

* **Orbit–stabilizer** (DERIVED, MEASURED): ``|Aut(G)| = |Γ_v| · |orbit(v)|``.
* **Residual sectors** (DERIVED): ``Fix(Aut(G)) ⊆ Fix(Γ_v)`` and
  ``dim Fix(Γ_v) = #orbits(Γ_v) ≥ #orbits(Aut(G))`` — the origin refines the
  Reynolds decomposition, it never coarsens it.
* **Break localization** (MEASURED): a pointed operator maps ``Fix(Aut(G))`` into
  ``Fix(Γ_v)`` but out of ``Fix(Aut(G))`` — the break is exactly the coset space
  ``Aut(G)/Γ_v = orbit(v)``.

Different origins in one orbit are **conjugate**: ``Γ_{g(v)} = g Γ_v g⁻¹`` and
``O@{g(v)} = g (O@v) g⁻¹``, so no origin is privileged — the basis of the R2
pointed residue networks ``(G_{p,k}, 0)``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

from ..alias import get_attr
from .operator_equivariance import _CHANNELS, _isolate_graph_caches
from .symmetry_sectors import (
    automorphism_orbits,
    automorphism_permutations,
    decompose_state,
)

__all__ = [
    "stabilizer_permutations",
    "stabilizer_orbits",
    "orbit_of",
    "orbit_stabilizer_holds",
    "conjugate_stabilizer_holds",
    "PointedSymmetryContext",
    "pointed_symmetry_context",
    "PointedBreakResult",
    "pointed_break_residuals",
    "audit_pointed_symmetry",
]


def stabilizer_permutations(G, v, *, weight=None, cap: int = 2000):
    r"""Automorphisms fixing the origin ``v`` — the stabilizer ``Γ_v ≤ Aut(G)``.

    ``Γ_v`` is a subgroup: it contains the identity and is closed under
    composition and inverse (the fixing condition ``g(v) = v`` is preserved).
    """
    return [
        g for g in automorphism_permutations(G, weight=weight, cap=cap)
        if g[v] == v
    ]


def orbit_of(G, v, *, weight=None, cap: int = 2000) -> tuple:
    r"""The ``Aut(G)`` orbit containing ``v`` (the reachable relabelings)."""
    for orbit in automorphism_orbits(G, weight=weight, cap=cap):
        if v in orbit:
            return orbit
    return (v,)


def stabilizer_orbits(G, v, *, weight=None, cap: int = 2000) -> list[tuple]:
    r"""Vertex orbits of the stabilizer ``Γ_v`` — the residual (pointed) sectors.

    Finer than the ``Aut(G)`` orbits: ``v`` is its own singleton orbit and the
    rest split by the residual symmetry.
    """
    perms = stabilizer_permutations(G, v, weight=weight, cap=cap)
    return automorphism_orbits(G, permutations=perms)


def orbit_stabilizer_holds(G, v, *, weight=None, cap: int = 2000) -> bool:
    r"""Orbit–stabilizer theorem ``|Aut(G)| = |Γ_v| · |orbit(v)|`` (MEASURED)."""
    full = len(automorphism_permutations(G, weight=weight, cap=cap))
    stab = len(stabilizer_permutations(G, v, weight=weight, cap=cap))
    orbit = len(orbit_of(G, v, weight=weight, cap=cap))
    return full == stab * orbit


def _compose(g: dict, h: dict) -> dict:
    r"""Permutation composition ``(g∘h)(x) = g[h[x]]``."""
    return {x: g[h[x]] for x in h}


def _inverse(g: dict) -> dict:
    return {dst: src for src, dst in g.items()}


def conjugate_stabilizer_holds(G, v, g: dict, *, weight=None,
                               cap: int = 2000) -> bool:
    r"""Whether ``Γ_{g(v)} = g Γ_v g⁻¹`` for an automorphism ``g`` (MEASURED).

    The pointed structures at ``v`` and ``g(v)`` are conjugate, so origins in one
    orbit are equivalent — no privileged origin.
    """
    w = g[v]
    stab_v = stabilizer_permutations(G, v, weight=weight, cap=cap)
    stab_w = stabilizer_permutations(G, w, weight=weight, cap=cap)
    ginv = _inverse(g)
    conj = {
        tuple(sorted(_compose(g, _compose(h, ginv)).items()))
        for h in stab_v
    }
    target = {tuple(sorted(h.items())) for h in stab_w}
    return conj == target


@dataclass(frozen=True)
class PointedSymmetryContext:
    r"""The residual symmetry of a pointed graph ``(G, v)``."""

    origin: object
    full_order: int            # |Aut(G)|
    stabilizer_order: int      # |Γ_v|
    origin_orbit_size: int     # |orbit(v)| = |Aut(G)| / |Γ_v|
    aut_orbit_count: int       # dim Fix(Aut(G))
    residual_orbit_count: int  # dim Fix(Γ_v) >= aut_orbit_count
    orbit_stabilizer_holds: bool


def pointed_symmetry_context(G, v, *, weight=None,
                             cap: int = 2000) -> PointedSymmetryContext:
    r"""Build the :class:`PointedSymmetryContext` for origin ``v``."""
    perms = automorphism_permutations(G, weight=weight, cap=cap)
    stab = [g for g in perms if g[v] == v]
    orbit = len(orbit_of(G, v, weight=weight, cap=cap))
    aut_orbits = automorphism_orbits(G, permutations=perms)
    res_orbits = automorphism_orbits(G, permutations=stab)
    return PointedSymmetryContext(
        origin=v,
        full_order=len(perms),
        stabilizer_order=len(stab),
        origin_orbit_size=orbit,
        aut_orbit_count=len(aut_orbits),
        residual_orbit_count=len(res_orbits),
        orbit_stabilizer_holds=(len(perms) == len(stab) * orbit),
    )


@dataclass(frozen=True)
class PointedBreakResult:
    r"""How a pointed operator breaks ``Aut(G)`` down to ``Γ_v``."""

    origin: object
    break_magnitude: float       # ‖Fix(Aut)^⊥ component‖ after O@v
    stabilizer_residual: float   # ‖Fix(Γ_v)^⊥ component‖ after O@v
    broke_full_symmetry: bool    # left Fix(Aut(G))
    preserves_stabilizer: bool   # stayed in Fix(Γ_v)


def pointed_break_residuals(op, G, v, *, channels=None,
                            tol: float = 1e-6) -> PointedBreakResult:
    r"""Apply ``op@v`` to the seed and split the result by ``Aut(G)`` and ``Γ_v``.

    A genuine pointed action leaves ``Fix(Aut(G))`` (``break_magnitude > tol``)
    yet stays in ``Fix(Γ_v)`` (``stabilizer_residual ≤ tol``): the break is
    exactly ``Aut(G) → Γ_v``.
    """
    from ..dynamics import default_compute_delta_nfr

    channels = _CHANNELS if channels is None else channels
    perms = automorphism_permutations(G)
    stab = [g for g in perms if g[v] == v]
    H = G.copy()
    _isolate_graph_caches(H)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        op(H, v)
    default_compute_delta_nfr(H)

    break_mag = 0.0
    stab_res = 0.0
    for ch in channels:
        field = {nd: float(get_attr(H.nodes[nd], ch, 0.0)) for nd in H.nodes()}
        _, perp_full = decompose_state(G, field, permutations=perms)
        _, perp_stab = decompose_state(G, field, permutations=stab)
        break_mag = max(break_mag, float((perp_full ** 2).sum() ** 0.5))
        stab_res = max(stab_res, float((perp_stab ** 2).sum() ** 0.5))
    return PointedBreakResult(
        origin=v,
        break_magnitude=break_mag,
        stabilizer_residual=stab_res,
        broke_full_symmetry=break_mag > tol,
        preserves_stabilizer=stab_res <= tol,
    )


def _pointed_test_cases():
    r"""Seeded pointed graphs at a **non-fixed** origin (nontrivial orbit)."""
    import networkx as nx

    from .operator_equivariance import _seed

    # Star K1,4, origin = leaf 1 (orbit {1,2,3,4}, stabilizer S_3 on {2,3,4}).
    star = nx.star_graph(4)
    _seed(
        star,
        lambda n: 0.1 if n == 0 else 0.3,
        lambda n: 0.5 if n == 0 else 0.3,
        lambda n: 1.0,
    )
    # Cycle C6, origin = vertex 0 (orbit = all, stabilizer = reflection Z_2).
    cyc = nx.cycle_graph(6)
    _seed(cyc, lambda n: 0.2, lambda n: 0.4, lambda n: 1.0)
    return [("star_leaf", star, 1), ("cycle_vertex", cyc, 0)]


def audit_pointed_symmetry(*, tol: float = 1e-6):
    r"""Audit the pointed break of a localized Emission at a non-fixed origin.

    Returns ``(label, PointedSymmetryContext, PointedBreakResult)`` per case,
    confirming the orbit–stabilizer count and the ``Aut(G) → Γ_v`` localization.
    """
    from ..operators.definitions import Emission

    out = []
    for label, G, v in _pointed_test_cases():
        ctx = pointed_symmetry_context(G, v)
        res = pointed_break_residuals(Emission(), G, v, tol=tol)
        out.append((label, ctx, res))
    return out
