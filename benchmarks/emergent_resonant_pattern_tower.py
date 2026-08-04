"""Particles as resonant patterns of ONE fractal coherent structure -- not
distinct objects (Layer 3, the coherence-pattern reframe).

THE REFRAME (theory creator): the emergent species are NOT distinct particles
but coherent PATTERNS of one coherence -- different resonant modes of the same
structure. So the inter-family organization should be a single fractal resonant
geometry whose self-similar tower of modes IS the family/generation grid, with a
log-periodic ("spiral") scaling between scales. This is canonical TNFR (AGENTS.md
"model coherence, not objects"; the pulse; operational fractality U5).

THE STRUCTURE (measured): take ONE coherent core, the tetrahedron K_4 (grade 3,
the generation-carrying simplex; benchmarks/emergent_generation_count.py), and
nest it self-similarly (THOL / U5 -- the Sierpinski simplex, the canonical
fractal lift of benchmarks/emergent_fractal_simplex_dimension.py). The resonant
modes omega_n = sqrt(lambda_n) of this ONE structure form a tower:

  - the modes come in GENERATION-TRIPLETS at EVERY scale: every excited
    eigenvalue has multiplicity a multiple of 3 (the S_4 standard irrep, the
    generation motif) -- the "3 generations" recur fractally, universally;
  - NEW mode-bands ("families") appear at each nesting level, at a roughly
    constant scaling ratio -- the self-similar / log-periodic tower, the
    "spiral" pitch between families;
  - so the many "particles" are one structure's modes: patterns of coherence at
    a tower of scales, each scale a triplet -- not distinct objects.

WHAT EMERGES (measured):
  - M1: ONE nested tetrahedron carries a growing tower of resonant modes
    (10 -> 34 -> 130 modes at nesting levels 1 -> 2 -> 3). The "particles" are
    modes of one coherent structure.
  - M2: the generation-triplet is UNIVERSAL -- every excited mode has
    multiplicity divisible by 3 (the S_4 triplet) at every scale. The 3-fold
    generation motif recurs fractally (the fractal pulse, EMERGENT_ONTOLOGY
    Sec.5.5).
  - M3: the tower is SELF-SIMILAR -- new mode-bands appear at each nesting level
    and the lowest band scales down by a roughly constant ratio per level (the
    log-periodic "spiral" pitch). Successive families are self-similar copies.
  - M4 (HONEST): the reframe (coherent patterns of one structure) and the fractal
    triplet-tower (the form) are genuine; the specific eigenvalues are the
    fractal spectrum, NOT the real particle masses, and the spiral pitch is a
    property of the chosen geometry (K_4 Sierpinski), not tuned to real ratios.

HONEST SCOPE: the self-similar spectrum of a Sierpinski simplex (high, structured
degeneracies; new bands per level; log-periodic density of states) is STANDARD
fractal spectral theory. The TNFR content is the reading: the species are modes
of ONE coherent structure (not distinct objects), the generation-triplet is the
S_4 motif recurring fractally, and the inter-family tower is the self-similar
("spiral") scaling. It does NOT derive the real family/generation masses or their
ratios (Layer 3, OPEN, theory/EMERGENT_ONTOLOGY.md Sec.9.1) -- the form-vs-values
split of the whole arc holds: the FORMS emerge, the VALUES do not. Closes no
open problem.

Run:
    python benchmarks/emergent_resonant_pattern_tower.py

Theoretical anchor: AGENTS.md (coherence not objects; the pulse; U5 fractality);
theory/EMERGENT_ONTOLOGY.md Sec.5.5 (the fractal pulse), Sec.3.2 (simplex nesting),
Sec.9.1 (OPEN properties); benchmarks/emergent_generation_count.py (grade 3 =
the generation triplet), emergent_fractal_simplex_dimension.py (the THOL nest).
Status: RESEARCH (Layer-3 coherence-pattern reframe; honest form-vs-values).
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
    """THOL self-similar nesting of K_m (corner-glued copies = the fractal lift)."""
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
    print("PARTICLES AS RESONANT PATTERNS OF ONE FRACTAL COHERENT STRUCTURE")
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
    print("     -> the many 'particles' are resonant modes of ONE coherent")
    print("        structure (the nested tetrahedron) -- patterns of coherence.")

    # -- M2: the generation-triplet is UNIVERSAL (fractal recurrence) ----------
    print("\n[M2] UNIVERSAL GENERATION-TRIPLET: every excited mode is a 3-multiple.")
    print(f"     {'nesting':>8} {'excited eigenvalues':>20} {'all mult %3==0':>15}")
    for lev, (_, c) in spectra.items():
        excited = {v: m for v, m in c.items() if v > 1e-6}
        all_triplet = all(m % 3 == 0 for m in excited.values())
        print(f"     {lev:>8} {len(excited):>20} {str(all_triplet):>15}")
        assert all_triplet, f"level {lev} has a non-triplet excited mode"
    print("     -> at EVERY scale, every excited mode carries the S_4 triplet")
    print("        (mult 3,6,15,18,...): the '3 generations' recur fractally.")

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
    print("VERDICT (Layer 3, the coherence-pattern reframe):")
    print("  DERIVED (the FORM): the species are modes of ONE fractal coherent")
    print("    structure -- a self-similar tower where the generation-triplet")
    print("    recurs at every scale and new family-bands appear log-periodically.")
    print("    Patterns of coherence, not distinct objects (exactly the reframe).")
    print("  NOT DERIVED (the VALUES): the eigenvalues are the fractal spectrum,")
    print("    NOT the real masses; the spiral pitch is set by the geometry")
    print("    (K_4 Sierpinski), not tuned to the real family ratios. The whole-")
    print("    arc discipline holds: the FORMS emerge, the VALUES do not.")
    print("=" * 74)


if __name__ == "__main__":
    main()
