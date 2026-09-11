#!/usr/bin/env python3
"""
Example 136 — The Heat-Kernel Coefficients: Hearing the Network's Geometry
(the Discrete Minakshisundaram-Pleijel Expansion)
==============================================================================

Example 134 used the LONG-time / return-probability scaling of the heat kernel
to read the spectral dimension. This example reads the SHORT-time expansion. The
heat trace of a canonical structural operator L,

    Z(t) = Tr(e^{-t L}) = sum_k (-t)^k / k! * Tr(L^k),

is a Taylor series whose coefficients are the SPECTRAL MOMENTS Tr(L^k). These
are EXACT graph invariants -- equal to weighted closed-walk counts -- and the
low-order ones HEAR the combinatorial geometry: the node count (volume), the edge
count (boundary), and the triangle count (curvature). This is the discrete
analogue of the Minakshisundaram-Pleijel / Seeley-deWitt heat-kernel expansion
(Weyl 1911; Minakshisundaram-Pleijel 1949), the same content behind Kac's 1966
question "Can one hear the shape of a drum?".

The operator used is the COMBINATORIAL Laplacian L = D - W, which is the canonical
Kirchhoff current operator of the structural-flow layer (current_divergence:
div(J) = L*EPI, example 99). The coupling matrix W = A (the adjacency) is the
canonical coupling of the nodal equation; its closed-walk moments Tr(A^k) give
the cleanest triangle reading.

Doctrine compliance
-------------------
Everything emerges from canonical structural objects: L = D - W is the Kirchhoff
operator (verified == current_divergence), W is the canonical coupling matrix.
The heat-trace coefficients are read off the spectrum of the canonical operator;
nothing is imposed. The quantities are standard spectral-graph invariants (closed
walks, the heat-kernel expansion); the example measures them, it does not invent
them.

Three measured results
----------------------
M1 THE HEAT-TRACE COEFFICIENTS ARE THE SPECTRAL MOMENTS. The short-time Taylor
   coefficients of Z(t) = Tr(e^{-t L}) are exactly Tr(L^k) = sum_i lambda_i^k,
   verified two independent ways (sum of powered eigenvalues vs trace of the
   matrix power) to machine precision. The truncated series reproduces Z(t) at
   small t. This is the discrete Minakshisundaram-Pleijel expansion.

M2 THE MOMENTS HEAR THE GEOMETRY (closed-walk counts). Tr(M^k) counts weighted
   closed walks of length k. For the Kirchhoff Laplacian: Tr(L^0) = n (nodes /
   volume), Tr(L^1) = 2m (edges / boundary), Tr(L^2) = 2m + sum d^2 (degree
   spread). For the canonical coupling matrix W = A: Tr(A^2) = 2m (edges) and
   Tr(A^3) = 6 * #triangles (triangles / curvature) -- verified against
   networkx. The heat trace hears volume, boundary, and curvature.

M3 "CAN ONE HEAR THE SHAPE OF A DRUM?" -- NO (Kac 1966). A cospectral pair of
   non-isomorphic graphs has IDENTICAL heat traces (all coefficients Tr(L^k)
   equal) yet DIFFERENT triangle counts (0 vs 1): the degree sequence and the
   triangle count conspire to give the same spectral moments. The heat-kernel
   coefficients are invariants but NOT a complete invariant -- you cannot always
   hear the shape.

Honest scope
------------
The heat-kernel coefficients = spectral moments = closed-walk counts are standard
spectral graph theory (the discrete heat-kernel / Minakshisundaram-Pleijel
expansion), exact and provable. The "hearing the geometry" reading is the
empirically-celebrated Weyl law / Kac drum problem. This re-expresses the
short-time heat-kernel structure of the canonical Kirchhoff operator; it is
complementary to example 134 (long-time spectral dimension). It is not new
mathematics and closes no open problem.

References
----------
- src/tnfr/physics/structural_diffusion.py (current_divergence,
  structural_current, _adjacency_degree)
- AGENTS.md "Transport Content of the Nodal Equation (Structural Diffusion)"
- examples/08_emergent_geometry/134_spectral_dimension_heat_kernel.py (long-time)
- examples/08_emergent_geometry/99_structural_diffusion.py (the Kirchhoff layer)
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import itertools
import math

import networkx as nx
import numpy as np

from tnfr.alias import set_attr
from tnfr.constants.aliases import ALIAS_EPI
from tnfr.physics.structural_diffusion import _adjacency_degree, current_divergence


def kirchhoff_laplacian(G):
    """Binary combinatorial Laplacian L = D - A.

    Matches the canonical Kirchhoff current operator (current_divergence masks
    by adjacency != 0, i.e. the unweighted structural current). Returns the
    canonical node order, L, the binary adjacency A, and the degree vector.
    """
    nodes, W, _ = _adjacency_degree(G)
    A = (W != 0.0).astype(float)
    deg = A.sum(axis=1)
    return nodes, np.diag(deg) - A, A, deg


def spectral_moments(L, kmax=5):
    """Tr(L^k) for k=0..kmax, computed by trace of the matrix power."""
    moments = []
    Lk = np.eye(L.shape[0])
    for k in range(kmax):
        if k > 0:
            Lk = Lk @ L
        moments.append(float(np.trace(Lk)))
    return moments


def experiment_1_moments_are_coefficients():
    """M1: heat-trace coefficients == spectral moments Tr(L^k)."""
    print("=" * 74)
    print("M1: THE HEAT-TRACE COEFFICIENTS ARE THE SPECTRAL MOMENTS Tr(L^k)")
    print("=" * 74)
    G = nx.watts_strogatz_graph(40, 6, 0.15, seed=1)
    nodes, L, A, deg = kirchhoff_laplacian(G)
    # anchor: L = D - A is the canonical Kirchhoff operator
    epi = np.random.default_rng(0).standard_normal(len(nodes))
    for i, nd in enumerate(nodes):
        set_attr(G.nodes[nd], ALIAS_EPI, float(epi[i]))
    _, divJ = current_divergence(G)
    print(
        f"  anchor: ||L*EPI - current_divergence|| = "
        f"{np.linalg.norm(L @ epi - divJ):.1e} "
        f"(L = D - A IS the Kirchhoff operator)"
    )
    print()
    eigs = np.linalg.eigvalsh(L)
    moments = spectral_moments(L)
    print(
        f"  {'k':>3} {'Tr(L^k)=sum lambda^k':>21} {'Tr(L^k)=tr(L^k)':>17} "
        f"{'|diff|':>8}"
    )
    for k in range(5):
        m_spec = float(np.sum(eigs**k))
        print(
            f"  {k:>3} {m_spec:>21.4f} {moments[k]:>17.4f} "
            f"{abs(m_spec - moments[k]):>8.1e}"
        )
    print()
    print("  truncated short-time series  Z(t) ~ sum (-t)^k/k! Tr(L^k):")
    for t in [0.001, 0.005, 0.02]:
        Z_exact = float(np.sum(np.exp(-t * eigs)))
        Z_series = sum(((-t) ** k) / math.factorial(k) * moments[k] for k in range(5))
        print(
            f"    t={t:>6.3f}:  Z_exact={Z_exact:.6f}  series={Z_series:.6f}"
            f"  |diff|={abs(Z_exact - Z_series):.1e}"
        )
    print()
    print("  -> the heat trace is generated by the spectral moments Tr(L^k)")
    print("     (the discrete Minakshisundaram-Pleijel expansion).")


def experiment_2_hearing_the_geometry():
    """M2: the moments hear nodes / edges / triangles (closed walks)."""
    print()
    print("=" * 74)
    print("M2: THE MOMENTS HEAR THE GEOMETRY (closed-walk counts)")
    print("=" * 74)
    G = nx.watts_strogatz_graph(40, 6, 0.15, seed=1)
    nodes, L, A, deg = kirchhoff_laplacian(G)
    moments = spectral_moments(L)
    n, m = G.number_of_nodes(), G.number_of_edges()
    tri = sum(nx.triangles(G).values()) // 3
    sum_d2 = float(np.sum(deg**2))
    print("  Kirchhoff Laplacian L = D - A:")
    print(f"    Tr(L^0) = {moments[0]:>7.0f}   <->  nodes  n  = {n} (volume)")
    print(
        f"    Tr(L^1) = {moments[1]:>7.0f}   <->  2m         = {2 * m} "
        f"(edges m = {m}, boundary)"
    )
    print(
        f"    Tr(L^2) = {moments[2]:>7.0f}   <->  2m + sum d^2 = "
        f"{2 * m + sum_d2:.0f}"
    )
    print()
    print("  canonical coupling matrix W = A (closed walks Tr(A^k)):")
    trA2 = float(np.trace(A @ A))
    trA3 = float(np.trace(A @ A @ A))
    print(f"    Tr(A^2) = {trA2:>7.0f}   <->  2m            = {2 * m} (edges)")
    print(
        f"    Tr(A^3) = {trA3:>7.0f}   <->  6 * #triangles = {6 * tri} "
        f"(triangles = curvature; networkx: {tri})"
    )
    print()
    print("  -> Tr(M^k) = weighted closed walks of length k; the heat trace")
    print("     hears volume (nodes), boundary (edges), curvature (triangles).")


def experiment_3_cannot_hear_the_shape():
    """M3: cospectral graphs -- identical heat traces, different geometry."""
    print()
    print("=" * 74)
    print("M3: 'CAN ONE HEAR THE SHAPE OF A DRUM?' -- NO (Kac 1966)")
    print("=" * 74)
    print("A cospectral non-isomorphic pair has identical heat traces (all")
    print("Tr(L^k) equal) yet different local geometry.")
    print()
    graphs = [
        g for g in nx.graph_atlas_g() if g.number_of_nodes() == 6 and nx.is_connected(g)
    ]
    for Ga, Gb in itertools.combinations(graphs, 2):
        La = nx.laplacian_matrix(Ga).toarray().astype(float)
        Lb = nx.laplacian_matrix(Gb).toarray().astype(float)
        la = np.sort(np.linalg.eigvalsh(La))
        lb = np.sort(np.linalg.eigvalsh(Lb))
        if np.allclose(la, lb, atol=1e-9) and not nx.is_isomorphic(Ga, Gb):
            ta = sum(nx.triangles(Ga).values()) // 3
            tb = sum(nx.triangles(Gb).values()) // 3
            da = sorted(d for _, d in Ga.degree())
            db = sorted(d for _, d in Gb.degree())
            print(
                f"  cospectral pair on 6 nodes (both {Ga.number_of_edges()} " f"edges):"
            )
            print(f"    shared L-spectrum = {np.round(la, 4)}")
            print(
                f"    {'Tr(L^k):':>12} "
                + "  ".join(
                    f"k={k}:{np.trace(np.linalg.matrix_power(La, k)):.0f}"
                    for k in range(4)
                )
            )
            print(
                f"    {'(graph B):':>12} "
                + "  ".join(
                    f"k={k}:{np.trace(np.linalg.matrix_power(Lb, k)):.0f}"
                    for k in range(4)
                )
            )
            print(
                f"    BUT triangles differ: {ta} vs {tb}; "
                f"degree sequences {da} vs {db}"
            )
            break
    print()
    print("  -> identical heat traces (all coefficients), yet non-isomorphic with")
    print("     DIFFERENT triangle counts: the degree sequence and the triangle")
    print("     count conspire to the same spectral moments. The heat-kernel")
    print("     coefficients are invariants but NOT complete -- you cannot")
    print("     always hear the shape.")


def main():
    print()
    print("  ===============================================================")
    print("  The Heat-Kernel Coefficients: Hearing the Network's Geometry")
    print("  The Discrete Minakshisundaram-Pleijel Expansion")
    print("  ===============================================================")
    print()
    experiment_1_moments_are_coefficients()
    experiment_2_hearing_the_geometry()
    experiment_3_cannot_hear_the_shape()
    print()
    print("=" * 74)
    print("WHAT THIS ESTABLISHES")
    print("=" * 74)
    print("The short-time heat trace Z(t) = Tr(e^{-t L}) of the canonical Kirchhoff")
    print("operator is a Taylor series whose coefficients are the spectral moments")
    print("Tr(L^k) (M1, machine precision) -- the discrete Minakshisundaram-Pleijel")
    print("expansion. These moments are weighted closed-walk counts that HEAR the")
    print("geometry: nodes (volume), edges (boundary), triangles (curvature, via")
    print("Tr(A^3) = 6*#triangles) (M2). But they are NOT a complete invariant: a")
    print("cospectral pair has identical heat traces yet different triangle counts")
    print("(M3) -- you cannot always hear the shape (Kac 1966). HONEST SCOPE: this")
    print("is standard spectral graph theory (heat-kernel coefficients = closed")
    print("walks), exact and provable, the celebrated Weyl law / drum problem; it")
    print("complements example 134 (long-time spectral dimension). Not new")
    print("mathematics, closes no open problem.")


if __name__ == "__main__":
    main()
