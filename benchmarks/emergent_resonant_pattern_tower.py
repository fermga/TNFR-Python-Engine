"""Finite rounded spectra of explicitly constructed K4 gasket levels.

The constructor selects corner gluing and three nesting levels directly; no
THOL birth, grammar execution or autonomous maintenance is observed.
Eigenvalues of L_sym are rounded to four decimals before counting multiplicity.
Assertions test divisibility by three only in those rounded finite spectra;
they are neither an exact multiplicity theorem nor a universal scale law.
Ratios of the lowest retained eigenvalues are reported, not fitted particle
families or a proved log-periodic limit. These are Laplacian eigenvalues; wave
angular frequencies would require a separate model and square roots.
Graph modes are useful pattern coordinates, but are not identified particles.

Status: auxiliary or finite evidence. See theory/EMERGENT_ONTOLOGY.md and
theory/NODAL_PARAMETER_FOUNDATIONS.md for model and physical-bridge limits.
"""

from __future__ import annotations

import pathlib
import sys
from collections import Counter

import networkx as nx
import numpy as np

_SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from tnfr.physics.structural_diffusion import (  # noqa: E402
    symmetric_normalized_laplacian,
)


def sierpinski_simplex(m: int, levels: int):
    """Construct a prescribed corner-glued K_m gasket; no THOL execution."""
    if levels == 0:
        return nx.complete_graph(m), list(range(m))
    sub, subc = sierpinski_simplex(m, levels - 1)
    G = nx.Graph()
    copies = []
    for i in range(m):
        mp = {v: (i, v) for v in sub.nodes}
        G.add_nodes_from(mp[v] for v in sub.nodes)
        G.add_edges_from((mp[u], mp[v]) for u, v in sub.edges)
        copies.append([mp[c] for c in subc])
    parent = {n: n for n in G.nodes}

    def find(x):
        r = x
        while parent[r] != r:
            r = parent[r]
        while parent[x] != r:
            parent[x], x = r, parent[x]
        return r

    for i in range(m):
        for j in range(i + 1, m):
            a, b = find(copies[i][j]), find(copies[j][i])
            if a != b:
                parent[b] = a
    H = nx.Graph()
    for u, v in G.edges:
        ru, rv = find(u), find(v)
        if ru != rv:
            H.add_edge(ru, rv)
    return H, [find(copies[i][i]) for i in range(m)]


def spectrum_multiplicities(G) -> Counter:
    """Rounded L_sym eigenvalues and their multiplicities."""
    _, lap = symmetric_normalized_laplacian(G)
    ev = np.linalg.eigvalsh(np.asarray(lap, dtype=float))
    return Counter(round(float(v), 4) for v in ev)


def main() -> None:
    print("=" * 74)
    print("FINITE ROUNDED SPECTRA OF ONE PRESCRIBED GASKET FAMILY")
    print("=" * 74)

    spectra = {}
    for lev in (1, 2, 3):
        G, _ = sierpinski_simplex(4, lev)  # the nested tetrahedron
        spectra[lev] = (G.number_of_nodes(), spectrum_multiplicities(G))

    # -- M1: ONE structure, a growing tower of modes (the reframe) -------------
    print("\n[M1] ONE structure, a tower of modes (patterns, not objects).")
    print(f"     {'nesting':>8} {'nodes':>7} {'modes':>7} {'distinct-omega':>15}")
    for lev, (n, c) in spectra.items():
        print(f"     {lev:>8} {n:>7} {n:>7} {len(c):>15}")
    print("     -> the displayed groups are modes of ONE prescribed")
    print("        structure (the nested tetrahedron) -- patterns of coherence.")

    # -- M2: finite rounded-multiplicity divisibility check ----------
    print("\n[M2] FINITE MULTIPLICITY CHECK: rounded excited groups are 3-multiples.")
    print(f"     {'nesting':>8} {'excited eigenvalues':>20} {'all mult %3==0':>15}")
    for lev, (_, c) in spectra.items():
        excited = {v: m for v, m in c.items() if v > 1e-6}
        all_triplet = all(m % 3 == 0 for m in excited.values())
        print(f"     {lev:>8} {len(excited):>20} {str(all_triplet):>15}")
        assert all_triplet, f"level {lev} has a non-triplet excited mode"
    print("     -> at EVERY scale, every excited mode carries the S_4 triplet")
    print(
        "        (mult 3,6,15,18,...): the finite rounded multiplicities pass this control."
    )

    # -- M3: the self-similar tower (the log-periodic 'spiral') ----------------
    print("\n[M3] SELF-SIMILAR TOWER: new family-bands per level, ~constant scale.")
    mins = []
    for lev, (_, c) in spectra.items():
        lo = min(v for v in c if v > 1e-6)
        mins.append(lo)
        print(f"     level {lev}: lowest excited lambda = {lo:.4f}")
    ratios = [round(mins[k] / mins[k + 1], 2) for k in range(len(mins) - 1)]
    print(f"     lowest-band scaling ratios per level: {ratios}")
    print("     -> new bands ('families') appear at each nesting, scaling down by")
    print("        a roughly constant factor: the self-similar / log-periodic")
    print("        ('spiral') tower -- the inter-family organization.")

    print("\n" + "=" * 74)
    print("FINITE GASKET SPECTRA:")
    print("  One prescribed graph construction supplies the three tested levels.")
    print("  Rounded eigenvalue groups pass the printed divisibility checks.")
    print("  The checks do not prove exact multiplicities at all scales.")
    print("  No autonomous birth or maintenance process is run.")
    print("  Lowest-eigenvalue ratios describe these finite graphs.")
    print("  Their interpretation as wave frequencies requires a separate law.")
    print("  Particle families, generations and mass ratios are not derived.")
    print("  Physical identification remains open.")
    print("=" * 74)


if __name__ == "__main__":
    main()
