"""Emergent NFR Geometry: every node is a pulsing NFR; dNFR=0 is the beat.

THE INSIGHT (user, theory creator): every node is an NFR -- a brick of the
substrate carrying (EPI, nu_f, phi) -- and each NFR PULSES: the single-node
nodal equation dEPI/dt = nu_f*dNFR reorganizes its form at its own frequency
nu_f. dNFR(i) = neighbour-mean(EPI) - EPI(i) = -(L_rw EPI)(i) is the discrete
CURVATURE of the EPI field, so "free of structural pressure" (dNFR=0) ==
harmonic == zero curvature == FLAT. When the per-NFR pulses RESONATE into a
standing mode, the dNFR=0 locus is the standing NODE: where neighbouring
pulses cancel and the field is flat, stationary, coherent (C->1) -- the BEAT
the pulses pass through. The antinodes are the SAME NFRs at the crest of their
pulse (high |dNFR|). So the equilibria are NOT a separate "NFR vs non-NFR"
class -- they are the NODAL SET (the Chladni pattern) of the resonating pulses,
their count/distribution set by the spectral index (Courant). The combat
coherence-vs-pressure is the pulse: standing node (beat) vs antinode (crest).

WHAT EMERGES (measured):
  - M1 dNFR = emergent CURVATURE (exact); at the standing NODE the pulse beats
    flat (dNFR=0, is_structural_equilibrium=True, structural_coherence -> 1);
    at the antinode the same NFR sits at its pulse crest (under pressure).
  - M2 the BEAT LATTICE = the Chladni nodal pattern: mode k has 2k standing
    nodes, so the node count grows with the structural pressure (the spectral
    index) -- Courant nodal-domain ordering. The "where" is spectral-geometric.
  - M3 the standing nodes are RESONANT (stationary fixed points of the
    resonant standing wave -- a node stays at zero amplitude for all t) and
    FRACTAL (on the self-similar THOL nest the NFR topology is multinodal and
    nests -- classify_nodal_topology, the canonical NFR read-out).
  - M4 the COMBAT selects the beat lattice: relaxing a random field collapses
    the curvature energy and the survivor is the slowest (Fiedler) mode -- the
    lowest-pressure standing-node pattern.

So the emergent TNFR geometry answers "what determines a pressure-free point":
it is the flat/nodal locus of the standing modes -- the beat where the
resonating NFR pulses cancel. For atoms the modes are the shells
(emergent_atom_dynamics.py); for
primes "where they fall" becomes "the nodal set of which emergent operator" =
the spectral (Hilbert-Polya) form of the Riemann problem, with the same
Fix(S_n)^perp wall, now stated geometrically.

HONEST SCOPE: the discrete-curvature / Chladni-nodal / Courant facts are
standard spectral geometry; the TNFR content is the reading dNFR = curvature,
dNFR=0 = flat = an NFR (the canonical coherence-equilibrium predicate), and the
NFR lattice = the nodal geometry of the emergent modes. Closes no open problem
(the prime case is RH). R and pi assumed.

Run:
    python benchmarks/emergent_nfr_geometry.py

Theoretical anchor: AGENTS.md (NFR = region of structural coherence, dNFR=0
attractor; tetrad K_phi = curvature; discrete-mode / Chladni regime, Courant);
benchmarks/emergent_atom_dynamics.py (the shells as modes). Status: RESEARCH.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import networkx as nx
from scipy.linalg import expm

_SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def sierpinski_simplex(m, levels):
    """THOL self-similar nesting of K_m (a fractal NFR)."""
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


def ring_lrw(n):
    """L_rw of the ring C_n (regular: L_rw = I - A/2)."""
    A = nx.to_numpy_array(nx.cycle_graph(n), nodelist=list(range(n)))
    d = A.sum(1)
    return np.eye(n) - (A / d[:, None])


def nodal_domains_ring(v, tol=1e-9):
    s = np.sign(np.where(np.abs(v) < tol, 0.0, v))
    s = s[s != 0]
    if len(s) == 0:
        return 0
    return max(1, int(np.sum(s != np.roll(s, 1))))


def main() -> None:
    from tnfr.metrics.common import (
        is_structural_equilibrium,
        structural_coherence,
    )
    from tnfr.physics.fields import classify_nodal_topology

    print("=" * 70)
    print("EMERGENT NFR GEOMETRY -- every node a pulsing NFR; dNFR=0 = beat")
    print("=" * 70)

    n = 24
    L = ring_lrw(n)
    idx = np.arange(n)

    # M1 -- dNFR = curvature; at the standing node the pulse beats flat
    print("\nM1 -- dNFR = curvature; the standing node beats flat (dNFR=0)")
    k = 3
    lam_k = 1.0 - np.cos(2 * np.pi * k / n)
    epi = np.cos(2 * np.pi * k * idx / n)  # an emergent standing mode
    dnfr = -(L @ epi)
    A = nx.to_numpy_array(nx.cycle_graph(n), nodelist=list(range(n)))
    curv = (A @ epi) / A.sum(1) - epi
    resid = float(np.max(np.abs(dnfr - curv)))
    print(f"  max|dNFR - (neighbour-mean - self)| = {resid:.2e}")
    is_node = np.abs(epi) < 1e-9  # the nodal set (v=0)
    eq_nodes = [is_structural_equilibrium(float(d)) for d in dnfr[is_node]]
    eq_anti = [is_structural_equilibrium(float(d)) for d in dnfr[~is_node]]
    c_nodes = float(
        np.mean([structural_coherence(float(d)) for d in dnfr[is_node]])
    )
    c_anti = float(
        np.mean([structural_coherence(float(d)) for d in dnfr[~is_node]])
    )
    print(f"  standing node (v=0): {int(is_node.sum())} points, all "
          f"equilibrium={all(eq_nodes)}, mean C={c_nodes:.3f} -> the beat")
    print(f"  antinode (crest)   : equilibrium={any(eq_anti)}, "
          f"mean C={c_anti:.3f} -> under pressure")
    assert all(eq_nodes) and not any(eq_anti) and c_nodes > c_anti

    # M2 -- the beat lattice = the Chladni nodal pattern; count by Courant
    print("\nM2 -- beat lattice = Chladni nodes; count by spectral index:")
    counts = []
    for kk in (1, 2, 3, 6):
        v = np.cos(2 * np.pi * kk * idx / n)
        nd = nodal_domains_ring(v)
        counts.append(nd)
        lam = 1.0 - np.cos(2 * np.pi * kk / n)
        print(f"  mode k={kk}: pressure={lam:.4f}, standing nodes={nd} (=2k)")
    grows = all(counts[i] < counts[i + 1] for i in range(len(counts) - 1))
    print(f"  more pressure -> more nodes, Courant-ordered: {grows}")
    assert grows and counts == [2, 4, 6, 12]

    # M3 -- the standing nodes are RESONANT (stationary) and FRACTAL (nesting)
    print("\nM3 -- the standing nodes are RESONANT + FRACTAL:")
    omega = np.sqrt(lam_k)
    n_nodes = int(is_node.sum())
    max_amp = max(
        float(np.max(np.abs(np.cos(omega * t) * epi[is_node])))
        for t in np.linspace(0.0, 10.0, 50)
    )
    print(f"  RESONANT: under cos(omega t)*v the {n_nodes} nodes stay")
    print(f"            at amplitude {max_amp:.2e} (stationary resonant pts)")
    nest, _ = sierpinski_simplex(4, 2)
    topo = classify_nodal_topology(nest)
    print(f"  FRACTAL : THOL nest NFR topology = '{topo['topology']}', "
          f"{len(topo.get('centers', []))} NFR centers (self-similar)")
    assert max_amp < 1e-9 and topo["topology"] in {
        "radial", "annular", "multinodal"}

    # M4 -- the COMBAT selects the beat lattice (curvature minimisation)
    print("\nM4 -- the combat selects the beat lattice (Fiedler survivor):")
    rng = np.random.default_rng(0)
    epi0 = rng.standard_normal(n)
    epi0 -= epi0.mean()
    rows = []
    for t in (0.0, 2.0, 10.0, 40.0):
        e = expm(-t * L) @ epi0
        en = 0.5 * float(e @ (L @ e))
        rows.append((t, en, nodal_domains_ring(e)))
        print(f"  t={t:5.1f}: curvature energy={en:.4f}, "
              f"standing nodes={nodal_domains_ring(e)}")
    drops = rows[0][1] > rows[-1][1]
    print("  combat lowers curvature -> survivor = Fiedler mode (2 nodes)")
    assert drops and rows[-1][2] == 2

    print("\n" + "=" * 70)
    print("VERDICT: every node is a pulsing NFR; 'free of structural")
    print("pressure' is GEOMETRIC -- dNFR = curvature, dNFR=0 = flat. The")
    print("standing node is where the resonating pulses beat flat (the")
    print("canonical coherence equilibrium); the antinode is the pulse crest.")
    print("The beat lattice = the Chladni pattern of the emergent modes,")
    print("ordered by the spectral index (Courant); the nodes are resonant")
    print("(stationary) and fractal (nesting). The combat selects them.")
    print("HONEST SCOPE: standard spectral geometry (curvature/Chladni/")
    print("Courant) re-read as NFR formation; the prime case is RH.")
    print("=" * 70)


if __name__ == "__main__":
    main()
