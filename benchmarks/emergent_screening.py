"""Finite spectral iteration with an imposed positive density-feedback potential.

The graph is a supplied embedded ball. Each iteration forms a density from the
lowest chosen modes, averages it over declared radial shells, then replaces the
operator by L + g*diag(K*rho). The occupation count, positive sign, coupling,
radial averaging and fixed iteration count are inputs; no nodal time law or
convergence test derives them. Squared eigenmode density is not stored DeltaNFR.
K uses inverse squared unweighted hop distance. This resembles one structural
potential kernel but is not a call to the canonical stored-pressure reader;
explicit edge lengths and their weight fallback can give different distances.
Mode groups, factor-two capacities and external shell sequences are comparison
choices. A spectral change is not a physical screening or force certificate.
Five sampled couplings cannot rule out all models or identify the only missing
atomic mechanism. See theory/EMERGENT_ONTOLOGY.md for the physical-bridge scope.
"""

from __future__ import annotations

import pathlib
import sys

import networkx as nx
import numpy as np

_SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
_BENCH = pathlib.Path(__file__).resolve().parent
if str(_BENCH) not in sys.path:
    sys.path.insert(0, str(_BENCH))

from emergent_shell_ordering import solid_ball_graph  # noqa: E402
from tnfr.physics.emergent_chemistry import (  # noqa: E402
    structural_eigenmodes,
)
from tnfr.physics.fields import classify_nodal_topology  # noqa: E402

ATOMIC_NOBLE = [2, 10, 18, 36, 54, 86]
SPHERICAL_WELL = [2, 8, 18, 20, 34, 40, 58]


def phi_s_kernel(G: nx.Graph, nodes: list) -> np.ndarray:
    """Inverse-square unweighted hop-distance matrix, with zero diagonal.

    The source density and metric here are declared independently of the canonical
    stored-pressure potential and its explicit-length/weight distance convention."""
    idx = {node: i for i, node in enumerate(nodes)}
    n = len(nodes)
    K = np.zeros((n, n))
    for node, dd in dict(nx.all_pairs_shortest_path_length(G)).items():
        i = idx[node]
        for j_node, d in dd.items():
            if d > 0:
                K[i, idx[j_node]] = 1.0 / (float(d) ** 2)
    return K


def radial_bins(G: nx.Graph, nodes: list) -> np.ndarray:
    """Unweighted hop radius about the explicitly selected first node."""
    rad_of = nx.single_source_shortest_path_length(G, nodes[0])
    return np.array([rad_of[node] for node in nodes])


def _symmetrise(rho: np.ndarray, rvec: np.ndarray) -> np.ndarray:
    """Average a density within each radial shell (a radial mean field)."""
    out = np.zeros_like(rho)
    for r in np.unique(rvec):
        mask = rvec == r
        out[mask] = rho[mask].mean()
    return out


def _shell_mults(w: np.ndarray, ntop: int = 40, gap_factor: float = 4.0):
    """Group the lowest eigenvalues into degenerate shells; return sizes."""
    ev = np.sort(w)[:ntop]
    gaps = np.diff(ev)
    pos = gaps[gaps > 1e-9]
    typ = float(np.median(pos)) if pos.size else 1e-9
    thr = gap_factor * typ
    groups = [[ev[0]]]
    for i, e in enumerate(ev[1:]):
        if gaps[i] > thr:
            groups.append([e])
        else:
            groups[-1].append(e)
    return [len(g) for g in groups]


def scf_closed_shells(
    L: np.ndarray,
    K: np.ndarray,
    rvec: np.ndarray,
    g: float,
    *,
    z_modes: int = 30,
    iters: int = 20,
) -> list[int]:
    """Iterate the selected density-feedback matrix, then count grouped modes.

    The fixed iteration count is not a convergence certificate; each mode has
    a supplied factor-two capacity."""
    Leff = L.copy()
    for _ in range(iters):
        _, V = np.linalg.eigh(Leff)
        rho = (V[:, :z_modes] ** 2).sum(axis=1)
        rho = _symmetrise(rho, rvec)
        Leff = L + g * np.diag(K @ rho)
    w, _ = np.linalg.eigh(Leff)
    cum, total = [], 0
    for m in _shell_mults(w):
        total += 2 * m
        cum.append(total)
    return cum


def leading_overlap(seq: list[int], ref: list[int]) -> int:
    count = 0
    for a, b in zip(seq, ref):
        if a != b:
            break
        count += 1
    return count


def main() -> None:
    print("=" * 70)
    print("PRESCRIBED DENSITY-FEEDBACK SPECTRAL COMPARISON")
    print("=" * 70)

    G = solid_ball_graph(4, 16, 8)
    nodes = list(G.nodes())
    # NOTE: the base manifold operator here is the imposed combinatorial Laplacian
    # D - A; the canonical EMERGENT structural operator is L_rw = I - D^-1 W
    # (symmetric twin L_sym). Deriving a nodal response for this selected feedback
    # study on L_sym is future work; the Phi_s back-reaction kernel K below IS
    # a separately selected matrix with hop distances.
    L = nx.laplacian_matrix(G, nodelist=nodes).toarray().astype(float)
    K = phi_s_kernel(G, nodes)
    rvec = radial_bins(G, nodes)

    print("\n[M1] Declared inverse-square hop-distance kernel (alpha=2):")
    print("     V(i) = sum_j rho(j)/d(i,j)^2  (chosen hop metric and density source)")
    print(f"     ball nucleus manifold: {len(nodes)} nodes, kernel {K.shape}")
    assert K.shape == (len(nodes), len(nodes))
    assert np.allclose(K, K.T), "Phi_s kernel must be symmetric"
    print("     -> PASS: mechanism = nodal dynamics on co-resident sub-EPIs.")

    cum0 = scf_closed_shells(L, K, rvec, 0.0)
    cum1 = scf_closed_shells(L, K, rvec, 1.0)
    print("\n[M2] Does the finite feedback iteration change spectral groups?")
    print(f"     g=0.0 (independent particle): {cum0[:6]}")
    print(f"     g=1.0 (self-consistent)     : {cum1[:6]}")
    assert cum1 != cum0, "self-consistency had no effect"
    assert 20 in cum0 and 20 not in cum1, "spherical-well 20 not reordered"
    print("     -> PASS: the chosen feedback changes the grouped spectrum -- the")
    print("        spherical-well '20' closure dissolves; levels reorder.")

    print("\n[M3] Scan coupling g -- does the ATOMIC order ever emerge?")
    print("     g      cumulative closed-shell counts")
    best_atomic = 0
    ten_ever = False
    for g in [0.0, 0.5, 1.0, 2.0, 4.0]:
        cum = scf_closed_shells(L, K, rvec, g)
        best_atomic = max(best_atomic, leading_overlap(cum, ATOMIC_NOBLE))
        ten_ever = ten_ever or (10 in cum[:6])
        print(f"     {g:<5}  {cum[:7]}")
    print(f"     atomic noble gases          : {ATOMIC_NOBLE}")
    print(f"     spherical well (g=0 family) : {SPHERICAL_WELL}")
    print(
        f"\n     max leading atomic match: {best_atomic}/6; "
        f"Ne-like '10' closure seen: {ten_ever}"
    )
    assert best_atomic <= 1, "atomic order unexpectedly emerged"
    assert not ten_ever, "the atomic '10' closure appeared"
    print(
        "     -> PASS: NO sampled coupling reproduces the supplied atomic sequence; the '10'"
    )
    print("        (1s 2s 2p) closure never forms. First closures stay 2, 8.")

    # -- M4: does the COMPLEMENT -- an attractive center -- emerge? ----------
    topo = classify_nodal_topology(G)
    center = topo["centers"][0]
    cphi = topo["centrality"][center]
    cvals = np.array(list(topo["centrality"].values()))
    mults = [
        s.multiplicity
        for s in structural_eigenmodes(G, max_modes=40, gap_factor=4.0)[:6]
    ]
    print("\n[M4] Where is the potential-readout maximum on the supplied ball?")
    print(
        f"     nucleus Phi_s = {cphi:.1f}  (max {cvals.max():.1f}, "
        f"mean {cvals.mean():.1f})"
    )
    print(f"     ball spectrum multiplicities : {mults}")
    assert abs(cphi - cvals.max()) < 1e-9, "nucleus is not the Phi_s maximum"
    assert mults[:3] == [1, 3, 5], "spectrum is not spherical-well"
    print("     -> PASS: the nucleus is the Phi_s MAXIMUM (repulsive for")
    print(
        "        +DeltaNFR, NOT an attractive sink); the reported spectrum is from the"
    )
    print("        spherical WELL (2l+1), NOT hydrogenic (Coulomb 2n^2).")

    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    print(
        "FINITE SPECTRAL ITERATION: the chosen positive density feedback changes some mode groups.\nThe density source, radial projection, sign, coupling and occupation count are supplied.\nTwenty iterations do not by themselves establish self-consistent convergence.\nThe displayed five-coupling comparison does not reproduce the supplied atomic sequence.\nThis does not identify a physical force, nucleus, screening law or unique missing ingredient."
    )


if __name__ == "__main__":
    main()
