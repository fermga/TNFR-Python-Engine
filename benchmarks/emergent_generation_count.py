"""Finite spectra of selected simplex cores, imposed wells and perturbations.

The program constructs K_m and an attached finite ring, supplies the spectral
well L_sym-depth*P_core, and counts negative eigenvalues using a tolerance.
The isolated core has nonconstant multiplicity m-1; its selected embedded well
can have m negative modes including the ground state. These are distinct
counts, and a finite graph has no literal continuum band. A supplied diagonal
spectral perturbation tests splitting; it is not canonical nodal pressure.

No THOL/UM trajectory, autonomous core selection, physical generation, mass
map or equality of simplex/fractal dimension with U(2) sectors is established.
The calculations are retained as conditional finite spectral comparisons.
Scope: theory/EMERGENT_ONTOLOGY.md sections 3.2, 7.4 and 9.1.
Run: python benchmarks/emergent_generation_count.py
Status: RESEARCH (finite auxiliary spectral controls).
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
    """Selected K_m/ring graph with one unit edge and an imposed diagonal well.
    Returns eigenvalues and core size; no physical binding law is derived."""
    G = nx.Graph()
    G.add_edges_from(nx.complete_graph(m).edges())
    ring = list(range(m, m + bath))
    for k, b in enumerate(ring):
        G.add_edge(b, ring[(k + 1) % bath])
    G.add_edge(0, ring[0])  # single unit-conductance core-to-bath edge
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
    """Count modes below the chosen negative numerical cutoff."""
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
    print("SIMPLEX SPECTRA -- distinct multiplicity, well count and splitting")
    print("=" * 74)

    # -- M1: localization truncates the tower to FINITE -----------------------
    print("\n[M1] FINITE CONTROL: negative modes of a configured spectral well.")
    print(f"     {'well depth':>11} {'#bound':>8}  (core = simplex K_5, size 5)")
    counts = []
    for depth in (0.5, 1.0, 2.0, 3.0, 5.0, 8.0):
        ev, m = core_in_bath(5, depth=depth)
        nb = count_bound(ev)
        counts.append(nb)
        print(f"     {depth:>11.1f} {nb:>8}")
    assert counts[-1] == 5 and counts == sorted(counts), "count not finite/monotone"
    print("     -> the sampled depths reach five negative eigenvalues;")
    print("        this does not test an infinite manifold or physical generations.")

    # -- M2: compare isolated multiplicity with embedded mode count -------------
    print("\n[M2] ISOLATED MULTIPLICITY versus EMBEDDED NEGATIVE-MODE COUNT.")
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
    print("        states including its ground state: these are different counts.")
    print("        Neither count selects a physical generation or dimension.")

    # -- M3: the degeneracy is the standard irrep of S_{g+1} -------------------
    print("\n[M3] DEGENERACY = STANDARD IRREP dim of S_{g+1} (representation).")
    print(f"     {'core':>6} {'S_n':>6} {'internal mult':>14} {'std-irrep dim':>14}")
    for m in (3, 4, 5):
        ev = np.linalg.eigvalsh(lsym(nx.complete_graph(m)))
        mult = distinct_levels(ev)[-1][1]
        print(f"     K_{m:<4} {'S_' + str(m):>6} {mult:>14} {m - 1:>14}")
        assert mult == m - 1
    print("     -> the internal multiplicity is the S_{g+1} standard-irrep dim g")
    print("        = m-1 for this complete graph; no particle mapping is supplied.")

    # -- M4: grade 3 gives a THREE-fold internal level ------------------------
    print("\n[M4] GRADE 3 -> a THREE-fold internal level (the tetrahedron K_4).")
    ev = np.linalg.eigvalsh(lsym(nx.complete_graph(4)))
    levels = distinct_levels(ev)
    print(
        f"     K_4 spectrum levels (value x mult): "
        f"{[(round(v, 3), mlt) for v, mlt in levels]}"
    )
    assert levels[-1][1] == 3, "tetrahedron internal level not 3-fold"
    print("     -> the selected K_4 has a three-dimensional nonconstant level.")
    print("        This does not select K_4 or identify spatial dimension with U(2).")
    print("        A physical generation count is not derived.")

    # -- M5: the selected diagonal perturbation splits the triplet ---------
    print("\n[M5] SPLITTING: a supplied diagonal spectral perturbation of K_4.")
    k4 = lsym(nx.complete_graph(4))
    print(f"     {'asymmetry':>10} {'excited internal levels':>28} {'distinct':>9}")
    for eps in (0.0, 0.05, 0.1, 0.3):
        # supplied spectral offsets; not canonical nodal pressure
        shift = np.array([0.0, 1.0, 2.0, 3.0]) * eps
        excited = np.sort(np.linalg.eigvalsh(k4 - np.diag(shift)))[1:]
        ndist = len({round(float(v), 3) for v in excited})
        vals = ", ".join(f"{v:.3f}" for v in excited)
        print(f"     {eps:>10.2f}   {vals:>26}   {ndist:>7}")
    shift = np.array([0.0, 1.0, 2.0, 3.0]) * 0.1
    excited = np.sort(np.linalg.eigvalsh(k4 - np.diag(shift)))[1:]
    assert len({round(float(v), 3) for v in excited}) == 3, "3-fold did not split"
    print("     -> the tested perturbation splits this S_4-protected level;")
    print("        other perturbations need separate tests; no masses are mapped.")

    print("\n" + "=" * 74)
    print("SCOPED RESULT:")
    print("  MEASURED: finite negative-mode counts in the configured well family.")
    print("    EXACT GRAPH FACT: K_(g+1) has nonconstant multiplicity g.")
    print("    Its embedded well can instead have g+1 negative modes.")
    print("  MEASURED: the selected perturbation splits the K_4 triplet.")
    print("    Neither masses nor physical generations follow from these spectra.")
    print("  REUSED: graph spectra and representation multiplicities.")
    print("    The graph, spectral well and perturbation are configured;")
    print("    no canonical birth or autonomous topology selection is executed.")
    print("  OPEN: physical state map, formation law and empirical validation.")
    print("    Eigenvalue ratios depend on the selected perturbation.")
    print("    These finite checks do not identify graph grade with physical")
    print("    dimension, substrate sector count or a particle catalog.")
    print("=" * 74)


if __name__ == "__main__":
    main()
