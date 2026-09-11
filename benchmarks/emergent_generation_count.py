"""Why is the generation tower FINITE? Localization truncates it, and the count
is the coherent core's simplex grade = its symmetry cardinal (Layer 3, sharpened).

THE CONVICTION (theory creator): the internal-mode tower of
benchmarks/emergent_internal_quantum_numbers.py is INFINITE on a fixed manifold,
yet nature has a FINITE count (3 lepton/quark generations). The answer must come
from the FULL TNFR unification -- graph theory, number theory, fractality,
resonance -- combined, not from the particle layer alone. It does:

  a real species is not a whole manifold but a LOCALIZED coherent core -- a
  maximally-coupled cluster of mutually-resonant NFRs, which is exactly the
  SIMPLEX K_{g+1} of grade g (theory/EMERGENT_ONTOLOGY.md Sec.3.2: "dimension =
  simplex grade = cardinal"). A localized core binds only FINITELY many internal
  states (the tight-binding bound states, Sec.7.4), so the tower is TRUNCATED,
  and the count is a spectral-counting quantity fixed by the core's grade and
  symmetry -- one object read across every domain (Sec.2.4).

THE UNIFICATION (each domain supplies one face of the same count):
  - GRAPH THEORY   -> the internal spectrum is the L_sym spectrum of the core;
    "how many generations" = how many eigenvalues below the continuum band
    (bound-state / Weyl counting).
  - PHYSICS        -> a localized well binds only FINITELY many states (Sec.7.4),
    so the infinite manifold tower is TRUNCATED to a finite core tower.
  - NUMBER THEORY  -> the simplex K_{g+1} has a g-fold internal level whose
    multiplicity g is the standard-irrep dimension of S_{g+1} = the CARDINAL =
    the same integer that is the emergent dimension (Sec.3.2). The count is a
    cardinal, the very object number theory reads off the fixed point.
  - FRACTALITY     -> the core is the canonical THOL coherent cluster; nesting it
    (U5) bands the spectrum self-similarly, so the finite count recurs per level.
  - RESONANCE      -> the core is the maximally-coupled (mutually resonant)
    cluster; the bound states are its localized resonant modes (high coherence),
    the continuum the delocalized (unbound) rest.

WHAT EMERGES (measured):
  - M1: a localized well binds a FINITE number of internal states that SATURATES
    at the core size as the depth grows -- the infinite tower is truncated.
  - M2: the count is the coherent core's grade: the simplex K_{g+1} carries a
    g-fold internal level (a ground mode + g excited internal states); embedded
    in a bath it binds exactly (g+1) localized states. The generation count is
    the simplex grade = the cardinal (graph theory and number theory, one count).
  - M3: the g-fold internal level is the standard irrep of S_{g+1} (dimension g)
    -- the degeneracy is representation-theoretic (the symmetry cardinals
    2, 6, 12, 20 of emergent_substrate_symmetry.py), not free.
  - M4: the tetrahedron K_4 (grade 3 -- the minimal 3D coherent core, which
    Sec.3.2 nests to the locked U(2) fibre) carries a THREE-fold internal level:
    this is where a count of 3 would come from.
  - M5: a generic asymmetric environment (distinct on-site structural pressures)
    LIFTS the S_4-protected degeneracy, splitting the 3-fold level into THREE
    DISTINCT masses -- three generations with different masses, not a degenerate
    triple.

HONEST SCOPE: the FINITENESS (localization truncates the tower), the count
= simplex grade = cardinal = standard-irrep dimension, and the degeneracy
SPLITTING into a distinct-mass triple under a generic environment are DERIVED
(tight-binding + spectral graph theory + the Sec.3.2 cardinal identity + basic
perturbation theory). This is a real advance over "infinite, unexplained": the
generation count is a FINITE cardinal fixed by the coherent core, and the three
masses are the split components of the grade-3 core's internal level. It does
NOT derive the real VALUE: (i) WHY the physical core is grade 3 (a 3D coherent
cluster) is not selected by any principle here, and (ii) the split RATIOS depend
on the specific environment (~1 : 1.1 : 1.2 in the demo, NOT the real
1 : 207 : 3477). So "the count is the coherent core's grade and the triple
splits" is DERIVED; "the grade is 3, split into those exact masses" stays OPEN
(theory/EMERGENT_ONTOLOGY.md Sec.9.1). Standard tools (K_m Laplacian, tight-
binding, S_n irreps, perturbation theory); the TNFR content is the unification
of the count across domains. Closes no open problem.

Run:
    python benchmarks/emergent_generation_count.py

Theoretical anchor: theory/EMERGENT_ONTOLOGY.md Sec.2.4 (one object, every
domain), Sec.3.2 (dimension = simplex grade = cardinal), Sec.7.4 (composite
matter / bound states), Sec.9.1 (OPEN properties); benchmarks/
emergent_internal_quantum_numbers.py (the infinite tower this truncates);
emergent_substrate_symmetry.py (the symmetry cardinals).
Status: RESEARCH (Layer-3 truncation via the cross-domain unification).
"""

from __future__ import annotations

import pathlib
import sys

import networkx as nx
import numpy as np

_SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from tnfr.physics.structural_diffusion import (  # noqa: E402
    symmetric_normalized_laplacian,
)


def lsym(G) -> np.ndarray:
    """Symmetric normalized Laplacian as a dense array."""
    _, lap = symmetric_normalized_laplacian(G)
    return np.asarray(lap, dtype=float)


def core_in_bath(m: int, *, bath: int = 40, depth: float = 3.0):
    """A coherent simplex core K_m weakly contacting a ring bath; well depth on
    the core. Returns (eigenvalues, n_core)."""
    G = nx.Graph()
    G.add_edges_from(nx.complete_graph(m).edges())
    ring = list(range(m, m + bath))
    for k, b in enumerate(ring):
        G.add_edge(b, ring[(k + 1) % bath])
    G.add_edge(0, ring[0])  # single weak contact core -> bath
    nodes = list(G.nodes())
    idx = {n: i for i, n in enumerate(nodes)}
    _, lap = symmetric_normalized_laplacian(G, nodes=nodes)
    L = np.asarray(lap, dtype=float)
    P = np.zeros(len(nodes))
    for cnode in range(m):
        P[idx[cnode]] = 1.0
    ev = np.linalg.eigvalsh(L - depth * np.diag(P))
    return ev, m


def count_bound(ev: np.ndarray) -> int:
    """Number of internal states below the continuum band (eigenvalues < 0)."""
    return int(np.sum(ev < -1e-9))


def distinct_levels(ev: np.ndarray, *, tol: float = 1e-4):
    """(value, multiplicity) of the distinct eigenvalue levels, ascending."""
    out: list[list[float]] = []
    for v in np.sort(ev):
        if not out or abs(v - out[-1][0]) > tol:
            out.append([float(v), 1])
        else:
            out[-1][1] += 1
    return [(v, int(m)) for v, m in out]


def main() -> None:
    print("=" * 74)
    print("GENERATION COUNT -- localization truncates the tower to a cardinal")
    print("=" * 74)

    # -- M1: localization truncates the tower to FINITE -----------------------
    print("\n[M1] TRUNCATION: a localized well binds FINITELY many internal states.")
    print(f"     {'well depth':>11} {'#bound':>8}  (core = simplex K_5, size 5)")
    counts = []
    for depth in (0.5, 1.0, 2.0, 3.0, 5.0, 8.0):
        ev, m = core_in_bath(5, depth=depth)
        nb = count_bound(ev)
        counts.append(nb)
        print(f"     {depth:>11.1f} {nb:>8}")
    assert counts[-1] == 5 and counts == sorted(counts), "count not finite/monotone"
    print("     -> the count grows then SATURATES at the core size (=5): the")
    print("        infinite manifold tower is truncated to a finite core tower.")

    # -- M2: the count is the coherent core's grade = the cardinal -------------
    print("\n[M2] THE COUNT = SIMPLEX GRADE = CARDINAL (graph theory + numbers).")
    print(f"     {'core':>6} {'grade':>6} {'internal level':>15} {'#bound(embed)':>14}")
    for m in (2, 3, 4, 5):
        ev_iso = np.linalg.eigvalsh(lsym(nx.complete_graph(m)))
        levels = distinct_levels(ev_iso)
        top_val, top_mult = levels[-1]
        ev_emb, _ = core_in_bath(m, depth=3.0)
        nb = count_bound(ev_emb)
        print(f"     K_{m:<4} {m - 1:>6} {str(top_mult) + '-fold':>15} {nb:>14}")
        assert top_mult == m - 1, f"K_{m} internal level not (m-1)-fold"
        assert nb == m, f"K_{m} did not bind m states"
    print("     -> K_{g+1} has a g-fold internal level; embedded it binds g+1")
    print("        states. The count IS the simplex grade = the cardinal (the")
    print("        same integer that is the emergent dimension, Sec.3.2).")

    # -- M3: the degeneracy is the standard irrep of S_{g+1} -------------------
    print("\n[M3] DEGENERACY = STANDARD IRREP dim of S_{g+1} (representation).")
    print(f"     {'core':>6} {'S_n':>6} {'internal mult':>14} {'std-irrep dim':>14}")
    for m in (3, 4, 5):
        ev = np.linalg.eigvalsh(lsym(nx.complete_graph(m)))
        mult = distinct_levels(ev)[-1][1]
        print(f"     K_{m:<4} {'S_' + str(m):>6} {mult:>14} {m - 1:>14}")
        assert mult == m - 1
    print("     -> the internal multiplicity is the S_{g+1} standard-irrep dim g")
    print("        = the symmetry cardinal (2,6,12,20 chain); not free.")

    # -- M4: grade 3 gives a THREE-fold internal level ------------------------
    print("\n[M4] GRADE 3 -> a THREE-fold internal level (the tetrahedron K_4).")
    ev = np.linalg.eigvalsh(lsym(nx.complete_graph(4)))
    levels = distinct_levels(ev)
    print(
        f"     K_4 spectrum levels (value x mult): "
        f"{[(round(v, 3), mlt) for v, mlt in levels]}"
    )
    assert levels[-1][1] == 3, "tetrahedron internal level not 3-fold"
    print("     -> the minimal 3D coherent core (grade 3, which Sec.3.2 nests to")
    print("        the U(2) fibre) carries EXACTLY a 3-fold internal level: this")
    print("        is where a count of 3 would come from.")

    # -- M5: an asymmetric environment splits the 3-fold into 3 masses ---------
    print("\n[M5] SPLITTING: a generic environment lifts the 3-fold into 3 masses.")
    k4 = lsym(nx.complete_graph(4))
    print(f"     {'asymmetry':>10} {'excited internal levels':>28} {'distinct':>9}")
    for eps in (0.0, 0.05, 0.1, 0.3):
        # distinct on-site structural pressures = the generic environment
        shift = np.array([0.0, 1.0, 2.0, 3.0]) * eps
        excited = np.sort(np.linalg.eigvalsh(k4 - np.diag(shift)))[1:]
        ndist = len({round(float(v), 3) for v in excited})
        vals = ", ".join(f"{v:.3f}" for v in excited)
        print(f"     {eps:>10.2f}   {vals:>26}   {ndist:>7}")
    shift = np.array([0.0, 1.0, 2.0, 3.0]) * 0.1
    excited = np.sort(np.linalg.eigvalsh(k4 - np.diag(shift)))[1:]
    assert len({round(float(v), 3) for v in excited}) == 3, "3-fold did not split"
    print("     -> the S_4-protected degeneracy is lifted by ANY asymmetry: the")
    print("        3-fold level becomes THREE DISTINCT masses = three generations.")

    print("\n" + "=" * 74)
    print("SHARPENED RESULT (Layer 3):")
    print("  DERIVED: localization truncates the tower to FINITE; the count is the")
    print("    coherent core's simplex GRADE = the CARDINAL = the S_{g+1} standard-")
    print("    irrep dimension. 'Infinite, unexplained' -> 'finite, a cardinal'.")
    print("  DERIVED: grade 3 (the tetrahedron) gives a 3-fold level, and a generic")
    print("    asymmetric environment SPLITS it into 3 DISTINCT masses = 3 gens.")
    print("  UNIFIED: one count read by graph theory (spectral counting), physics")
    print("    (bound states), number theory (the cardinal), fractality (the THOL")
    print("    simplex), resonance (the coupled core).")
    print("  STILL OPEN: WHY grade 3 is not selected by a principle, and the split")
    print("    RATIOS depend on the perturbation (~1:1.1:1.2 here, not 1:207:3477).")
    print("    The count STRUCTURE and the 3-way splitting emerge; the choice of")
    print("    grade 3 and the exact ratios stay open.")
    print("=" * 74)


if __name__ == "__main__":
    main()
