"""Prepared eigenmode zeros and relative-form relaxation on fixed graphs.

For the pure EPI channel p=-L_rw x, a nonzero-eigenvalue mode has p=0 at its
sampled zeros. This is a local zero of a discrete Laplacian, not geometric
flatness of the graph, absence of an NFR or a proof of autonomous formation.
Capacity scales a declared form response and does not imply a periodic pulse.
The standing-wave interpretation separately prescribes oscillatory dynamics.
The ring fixture counts sampled zeros/sign domains of chosen modes; Courant
bounds do not imply equally spaced zeros for arbitrary symmetric graphs.
Relative diffusion may approach the slowest excited eigenspace while its
amplitude decays. It need not select a unique Fiedler shape or maintain identity.
The recursive gasket is constructed directly, without a THOL invocation.

Status: auxiliary or finite evidence. See theory/EMERGENT_ONTOLOGY.md and
theory/NODAL_PARAMETER_FOUNDATIONS.md for model and physical-bridge limits.
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
    """Construct a prescribed K_m gasket directly."""
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
    print("EMERGENT NFR GEOMETRY -- prepared mode zeros and diffusive relaxation")
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
    c_nodes = float(np.mean([structural_coherence(float(d)) for d in dnfr[is_node]]))
    c_anti = float(np.mean([structural_coherence(float(d)) for d in dnfr[~is_node]]))
    print(
        f"  standing node (v=0): {int(is_node.sum())} points, all "
        f"equilibrium={all(eq_nodes)}, mean C={c_nodes:.3f} -> the beat"
    )
    print(
        f"  antinode (crest)   : equilibrium={any(eq_anti)}, "
        f"mean C={c_anti:.3f} -> under pressure"
    )
    assert all(eq_nodes) and not any(eq_anti) and c_nodes > c_anti

    # M2 -- count selected ring-mode sign domains
    print("\nM2 -- selected ring-mode sign domains by mode index:")
    counts = []
    for kk in (1, 2, 3, 6):
        v = np.cos(2 * np.pi * kk * idx / n)
        nd = nodal_domains_ring(v)
        counts.append(nd)
        lam = 1.0 - np.cos(2 * np.pi * kk / n)
        print(f"  mode k={kk}: pressure={lam:.4f}, standing nodes={nd} (=2k)")
    grows = all(counts[i] < counts[i + 1] for i in range(len(counts) - 1))
    print(f"  selected ring mode indices -> larger sign-domain counts: {grows}")
    assert grows and counts == [2, 4, 6, 12]

    # M3 -- supplied cosine time dependence and independent gasket read-out
    print("\nM3 -- supplied standing-wave zeros and gasket classification:")
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
    print(
        f"  GASKET : supplied-graph topology = '{topo['topology']}', "
        f"{len(topo.get('centers', []))} NFR centers (self-similar)"
    )
    assert max_amp < 1e-9 and topo["topology"] in {"radial", "annular", "multinodal"}

    # M4 -- pure-EPI diffusion reduces Dirichlet energy in this finite run
    print("\nM4 -- finite pure-EPI relative relaxation:")
    rng = np.random.default_rng(0)
    epi0 = rng.standard_normal(n)
    epi0 -= epi0.mean()
    rows = []
    for t in (0.0, 2.0, 10.0, 40.0):
        e = expm(-t * L) @ epi0
        en = 0.5 * float(e @ (L @ e))
        rows.append((t, en, nodal_domains_ring(e)))
        print(
            f"  t={t:5.1f}: curvature energy={en:.4f}, "
            f"standing nodes={nodal_domains_ring(e)}"
        )
    drops = rows[0][1] > rows[-1][1]
    print("  this finite endpoint has lower energy and two sign domains")
    assert drops and rows[-1][2] == 2

    print("\n" + "=" * 70)
    print("RESULT SCOPE: p=-L_rw x belongs to the isolated EPI channel.")
    print("A sampled eigenmode zero has zero local pressure in this prepared field.")
    print("It is not a proof that the graph is flat or that an NFR has formed.")
    print("Capacity nu_f is not automatically an oscillation frequency.")
    print("The ring supplies specially structured nodal sets.")
    print("General symmetric graphs need not have equally spaced mode zeros.")
    print("Relative diffusion suppresses faster modes while amplitude also decays.")
    print("The gasket geometry and any standing-wave time dependence are supplied.")
    print("Autonomous maintained identity and physical particles remain unproved.")
    print("=" * 70)


if __name__ == "__main__":
    main()
