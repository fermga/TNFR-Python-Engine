"""Finite low-spectrum slope estimates on supplied graph families.

The estimator uses the combinatorial Laplacian and a chosen low-eigenvalue
window. It compares explicit cycles, grids, balanced trees and phase-threshold
graphs. A balanced_tree constructor is not THOL execution, and selecting all
phase-compatible edges is not a derived autonomous Coupling schedule.
The tests compare a few estimates and two gate values; they prove neither a
continuous parameter law nor a universal spatial-dimension restriction.
The auxiliary substrate's two complex sectors count coordinates in that model;
they do not imply that TNFR or physical space is intrinsically two-dimensional.
Spectral slope, simplex grade, similarity dimension and sector count differ.

Status: auxiliary or finite evidence. See theory/EMERGENT_ONTOLOGY.md and
theory/NODAL_PARAMETER_FOUNDATIONS.md for model and physical-bridge limits.
"""

from __future__ import annotations

import pathlib
import sys

import networkx as nx
import numpy as np

_SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def spectral_dimension(G: nx.Graph, frac: float = 0.15, kmin: int = 8) -> float:
    """d_s from N(lambda)~lambda^(d_s/2): slope of log-count vs log-eig."""
    # Spectral dimension reads the low-eigenvalue SCALING EXPONENT, invariant
    # under the uniform rescaling L_rw = (D - A)/deg on regular graphs; the
    # canonical emergent operator is L_rw = I - D^-1 W (D - A used here for the
    # bare combinatorial spectrum).
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
    """Construct edges from declared phases and a circular-distance threshold."""
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
    print("\n[M2] Explicit balanced-tree controls:")
    print(f"     balanced tree b=2 d_s = {ds_tree2:.2f}")
    print(f"     balanced tree b=3 d_s = {ds_tree3:.2f}")
    assert ds_tree2 < 2.0 and ds_tree3 < 2.0, "THOL tree not low-dim"
    print(
        "     -> PASS: these two tree estimates satisfy the selected low-slope check."
    )

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
        "Finite scope: the displayed estimates belong to explicitly supplied graphs.\nBalanced trees and phase-threshold graphs are constructions, not native THOL trajectories.\nThe fit window and topology affect the estimated slope. Two gate values do not prove a continuous law.\nNo spatial dimension is selected here, and no impossibility for other nodal laws follows.\nThe two auxiliary complex sectors do not determine physical spatial dimension."
    )


if __name__ == "__main__":
    main()
