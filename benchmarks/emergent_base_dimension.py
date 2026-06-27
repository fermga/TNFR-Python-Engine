"""
Emergent Base Dimension: does a fixed spatial (spectral) dimension -- in
particular d=3 -- emerge in the network BASE from TNFR dynamics?
============================================================================

CONTEXT (benchmarks/emergent_substrate_symmetry.py): the substrate FIBER is
structurally locked to U(2) -- two conjugate sectors, intrinsically 2D. Any 3D
shell physics must therefore live in the network BASE: the effective / SPECTRAL
DIMENSION of the emergent structure. The spatial ball put d=3 in the base BY
HAND. This benchmark asks whether d=3 (or any fixed dimension) EMERGES in the
base from the TNFR-native structure-builders -- THOL nesting (U5 fractal
hierarchy) and U3 resonant coupling -- with no spatial embedding.

SPECTRAL DIMENSION d_s: from the integrated density of states of the structural
Laplacian, N(lambda) ~ lambda^(d_s/2) for small lambda (heat-kernel return
P(t) ~ t^(-d_s/2)). For a d-dimensional lattice d_s = d; for a tree d_s is
tree-like; for a dense random graph d_s is large. (Finite-size fits slightly
UNDER-estimate higher d_s, so we assert the ORDERING, not exact values.)

WHAT EMERGES (measured below):
  - THOL nesting (the U5 fractal hierarchy = a tree) has d_s ~ 1.6 -- low,
    tree-like, NOT 3.
  - U3 resonant coupling (link if |phi_i - phi_j| <= dphi_max) has a d_s that
    VARIES CONTINUOUSLY with the gate dphi_max (a free parameter): a tight gate
    gives ~2.5, a wide gate ~7. It is TUNABLE, not a fixed emergent value.
  => No TNFR-native structure-builder pins the base to d=3. The base spectral
     dimension is a FREE TOPOLOGY INPUT, not a prediction of the dynamics.

THE CLOSURE (fiber + base): the FIBER is intrinsically 2D (U(2),
structurally locked); the BASE dimension is a free input (THOL ~1.6,
resonant tunable). So 3D space is NOT singled out by TNFR structure or
dynamics. TNFR's emergent geometry is intrinsically 2D at the fiber;
spatial dimensionality of the base is an INPUT, not emergent. (Sharper
still: the shell CARDINALS need a symmetry group Aut(G); random resonant
coupling has trivial Aut -> no shells at all. The continuous 3D tower
SO(3) needs a continuum-sphere symmetry, which is exactly the imported
geometry the spatial ball supplied.)

HONEST SCOPE: the spectral-dimension estimator and the lattice / tree /
random-graph spectral facts are STANDARD (the comparison framework). The
TNFR result is that the native structure-builders (THOL nesting, U3
coupling) do NOT pin the base to d=3 -- the dimension is free -- so, with
the locked 2D fiber, 3D is not emergent.

Run:
    python benchmarks/emergent_base_dimension.py

Theoretical anchor: AGENTS.md (U5 fractality / THOL nesting; U3 resonant
coupling; discrete-mode regime); benchmarks/emergent_substrate_symmetry.py
(the locked 2D fiber this complements). Status: RESEARCH (emergence falsifier).
"""

from __future__ import annotations

import pathlib
import sys

import networkx as nx
import numpy as np

_SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def spectral_dimension(
    G: nx.Graph, frac: float = 0.15, kmin: int = 8
) -> float:
    """d_s from N(lambda)~lambda^(d_s/2): slope of log-count vs log-eig."""
    L = nx.laplacian_matrix(G).toarray().astype(float)
    ev = np.sort(np.linalg.eigvalsh(L))
    ev = ev[ev > 1e-9]
    k = min(max(kmin, int(frac * len(ev))), len(ev))
    x = np.log(ev[:k])
    y = np.log(np.arange(1, k + 1))
    return float(2.0 * np.polyfit(x, y, 1)[0])


def resonant_graph(
    n: int = 400, dphi_max: float = np.pi / 2, seed: int = 0
) -> nx.Graph:
    """U3 resonant coupling: random phases, link if |phi_i-phi_j| <= dphi."""
    rng = np.random.default_rng(seed)
    phi = rng.uniform(0, 2 * np.pi, n)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        d = np.abs(np.angle(np.exp(1j * (phi - phi[i]))))
        d[i] = np.inf
        for j in np.where(d <= dphi_max)[0]:
            G.add_edge(i, int(j))
    return G


def main() -> None:
    print("=" * 70)
    print("EMERGENT BASE DIMENSION (does d=3 emerge? no spatial embedding)")
    print("=" * 70)

    # -- M1: calibrate the estimator (ordering, not exact values) -----------
    ds_ring = spectral_dimension(nx.cycle_graph(600))
    ds_2d = spectral_dimension(nx.grid_2d_graph(26, 26))
    ds_3d = spectral_dimension(nx.grid_graph([9, 9, 9]))
    print("\n[M1] Calibration (finite-size under-estimates; ordering holds):")
    print(f"     ring (1D) d_s   = {ds_ring:.2f}")
    print(f"     grid 2D   d_s   = {ds_2d:.2f}")
    print(f"     grid 3D   d_s   = {ds_3d:.2f}")
    assert ds_ring < ds_2d < ds_3d, "estimator ordering failed"
    print("     -> PASS: d_s increases with lattice dimension (1 < 2 < 3).")

    # -- M2: THOL nesting (U5 fractal hierarchy) -> tree, low d_s ------------
    ds_tree2 = spectral_dimension(nx.balanced_tree(2, 9))
    ds_tree3 = spectral_dimension(nx.balanced_tree(3, 6))
    print("\n[M2] THOL nesting (U5 hierarchy = a tree):")
    print(f"     balanced tree b=2 d_s = {ds_tree2:.2f}")
    print(f"     balanced tree b=3 d_s = {ds_tree3:.2f}")
    assert ds_tree2 < 2.0 and ds_tree3 < 2.0, "THOL tree not low-dim"
    print("     -> PASS: THOL nesting is tree-like (d_s ~ 1.6), NOT 3.")

    # -- M3: U3 resonant coupling -> d_s tunable by the gate, not fixed ------
    ds_wide = spectral_dimension(resonant_graph(400, np.pi / 2))
    ds_narrow = spectral_dimension(resonant_graph(400, np.pi / 6))
    print("\n[M3] U3 resonant coupling (link if |phi_i-phi_j| <= dphi_max):")
    print(f"     dphi=pi/2 (wide)  d_s = {ds_wide:.2f}")
    print(f"     dphi=pi/6 (tight) d_s = {ds_narrow:.2f}")
    assert ds_wide - ds_narrow > 2.0, "resonant d_s is not gate-tunable"
    print("     -> PASS: resonant d_s VARIES with the gate (free param);")
    print("        it is tunable, NOT a fixed emergent dimension.")

    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    print(
        "No TNFR-native structure-builder pins the base to d=3. THOL\n"
        "  nesting (the U5 fractal hierarchy) is tree-like (d_s ~ 1.6); U3\n"
        "  resonant coupling gives a d_s that varies continuously with the\n"
        "  gate dphi_max (~2.5 to ~7). The base spectral dimension is a FREE\n"
        "  TOPOLOGY INPUT, not a prediction of the dynamics.\n"
        "THE CLOSURE (fiber + base): the substrate FIBER is structurally\n"
        "  locked to U(2) -- intrinsically 2D; the BASE dimension is free.\n"
        "  So 3D space is NOT singled out by TNFR structure or dynamics.\n"
        "  TNFR's emergent geometry is intrinsically 2D at the fiber; the\n"
        "  spatial dimension of the base is an INPUT, not emergent.\n"
        "SHARPER (symmetry, not just dimension): the shell CARDINALS need a\n"
        "  symmetry group Aut(G) to make degeneracies; random resonant\n"
        "  coupling has trivial Aut -> no shells at all. The continuous 3D\n"
        "  tower (SO(3), 2l+1) needs a continuum-sphere symmetry -- exactly\n"
        "  the geometry the spatial ball imported. TNFR does not build it\n"
        "  from the dynamics.\n"
        "ULTIMATE CONSEQUENCE: TNFR is intrinsically a 2D resonant theory\n"
        "  (the U(2) fiber); spatial dimensionality and 3D rotational\n"
        "  symmetry are inputs, not emergent. An honest structural limit."
    )


if __name__ == "__main__":
    main()
