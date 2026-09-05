r"""Tests for pointed symmetry — the declared break Aut(G) → Γ_v (R1, N07).

Acting at a chosen origin ``v`` reduces the symmetry from ``Aut(G)`` to the
stabilizer ``Γ_v``. Orbit–stabilizer counts, residual-sector refinement, the
``Aut(G) → Γ_v`` break localization, and origin conjugation are checked.
"""

from __future__ import annotations

import networkx as nx

from tnfr.operators.definitions import Emission
from tnfr.physics import pointed_symmetry as ps
from tnfr.physics.operator_equivariance import _seed
from tnfr.physics.pointed_symmetry import (
    PointedBreakResult,
    PointedSymmetryContext,
    audit_pointed_symmetry,
    conjugate_stabilizer_holds,
    orbit_of,
    orbit_stabilizer_holds,
    pointed_break_residuals,
    pointed_symmetry_context,
    stabilizer_orbits,
    stabilizer_permutations,
)
from tnfr.physics.symmetry_sectors import automorphism_permutations


def _star():
    star = nx.star_graph(4)
    _seed(star, lambda n: 0.1 if n == 0 else 0.3,
          lambda n: 0.5 if n == 0 else 0.3, lambda n: 1.0)
    return star


def _cycle():
    cyc = nx.cycle_graph(6)
    _seed(cyc, lambda n: 0.2, lambda n: 0.4, lambda n: 1.0)
    return cyc


# --------------------------------------------------------------------------- #
# Stabilizer subgroup
# --------------------------------------------------------------------------- #
def test_stabilizer_fixes_origin_and_contains_identity():
    star = _star()
    stab = stabilizer_permutations(star, 1)
    assert all(g[1] == 1 for g in stab)
    assert {i: i for i in star.nodes()} in stab   # identity


def test_stabilizer_is_closed_under_composition():
    star = _star()
    stab = stabilizer_permutations(star, 1)
    stab_set = {tuple(sorted(g.items())) for g in stab}
    for g in stab:
        for h in stab:
            comp = tuple(sorted({x: g[h[x]] for x in h}.items()))
            assert comp in stab_set        # subgroup closure


def test_orbit_stabilizer_theorem():
    assert orbit_stabilizer_holds(_star(), 1)   # 24 = 6 * 4
    assert orbit_stabilizer_holds(_cycle(), 0)  # 12 = 2 * 6


# --------------------------------------------------------------------------- #
# Residual sectors refine Aut(G) sectors
# --------------------------------------------------------------------------- #
def test_residual_orbits_refine_aut_orbits():
    ctx = pointed_symmetry_context(_star(), 1)
    assert isinstance(ctx, PointedSymmetryContext)
    assert ctx.residual_orbit_count >= ctx.aut_orbit_count
    assert ctx.residual_orbit_count == 3   # {0}, {1}, {2,3,4}
    assert ctx.aut_orbit_count == 2        # {0}, {1,2,3,4}


def test_stabilizer_isolates_origin_as_singleton_orbit():
    orbits = stabilizer_orbits(_star(), 1)
    assert (1,) in orbits                  # origin is its own orbit


# --------------------------------------------------------------------------- #
# Break localization: Aut(G) → Γ_v
# --------------------------------------------------------------------------- #
def test_pointed_action_breaks_aut_but_preserves_stabilizer():
    for G, v in [(_star(), 1), (_cycle(), 0)]:
        res = pointed_break_residuals(Emission(), G, v)
        assert isinstance(res, PointedBreakResult)
        assert res.broke_full_symmetry              # left Fix(Aut(G))
        assert res.preserves_stabilizer             # stayed in Fix(Γ_v)
        assert res.stabilizer_residual < 1e-6
        assert res.break_magnitude > 1e-6


def test_singleton_orbit_origin_does_not_break_aut():
    # the star centre is a fixed point of Aut (orbit {0}); acting there keeps
    # the state in Fix(Aut(G)) — no symmetry break
    star = _star()
    assert orbit_of(star, 0) == (0,)
    res = pointed_break_residuals(Emission(), star, 0)
    assert not res.broke_full_symmetry
    assert res.break_magnitude < 1e-6


# --------------------------------------------------------------------------- #
# Origin conjugation: no privileged origin
# --------------------------------------------------------------------------- #
def test_stabilizers_of_orbit_mates_are_conjugate():
    star = _star()
    g = next(p for p in automorphism_permutations(star) if p[1] == 3)
    assert conjugate_stabilizer_holds(star, 1, g)   # Γ_3 = g Γ_1 g^{-1}


def test_conjugate_origins_have_equal_stabilizer_order():
    star = _star()
    ctx1 = pointed_symmetry_context(star, 1)
    ctx3 = pointed_symmetry_context(star, 3)
    assert ctx1.stabilizer_order == ctx3.stabilizer_order
    assert ctx1.origin_orbit_size == ctx3.origin_orbit_size


# --------------------------------------------------------------------------- #
# Audit + exports
# --------------------------------------------------------------------------- #
def test_audit_pointed_symmetry_reports_both_cases():
    results = audit_pointed_symmetry()
    assert len(results) == 2
    for _label, ctx, res in results:
        assert ctx.orbit_stabilizer_holds
        assert res.broke_full_symmetry
        assert res.preserves_stabilizer


def test_module_exports_complete():
    expected = {
        "stabilizer_permutations", "stabilizer_orbits", "orbit_of",
        "orbit_stabilizer_holds", "conjugate_stabilizer_holds",
        "PointedSymmetryContext", "pointed_symmetry_context",
        "PointedBreakResult", "pointed_break_residuals",
        "audit_pointed_symmetry",
    }
    assert expected <= set(ps.__all__)
