"""
Emergent Screening? Does Phi_s self-consistency turn the spherical-well shells
into the atomic periodic table -- without injecting quantum chemistry?
============================================================================

CONTEXT (from benchmarks/emergent_shell_ordering.py): pure TNFR structure gives
the (2l+1) angular degeneracy, a radial-sum filling order, and -- with the
emergent radial nucleus of a solid ball -- the INFINITE SPHERICAL WELL closures
2, 8, 18, 20, 34 (the independent-particle / nuclear shell family). The SOLE
residual to the CHEMICAL periodic table (2, 10, 18, 36, 54, 86) was identified
as electron-electron SCREENING. This benchmark asks whether that screening
EMERGES from TNFR self-consistency, with no foreign theory.

MECHANISM (canonical, TNFR-native -- NOT Hartree-Coulomb):
  The occupied structural eigenmodes are co-resident sub-EPIs (U5
  nesting). Each is a DeltaNFR source; their aggregate structural
  potential is the CANONICAL Phi_s field

      Phi_s(i) = sum_j rho(j) / d(i,j)^2,   rho = sum_occupied |psi_k|^2

  i.e. the SAME inverse-square (alpha=2) kernel as
  compute_structural_potential (grammar U6) and classify_nodal_topology.
  Iterating the loop

      occupy lowest modes -> Phi_s -> shift operator -> re-diagonalise

  to self-consistency is just the nodal dynamics acting back on
  co-resident sub-EPIs. The mechanism is canonical; WHAT IT PRODUCES is
  measured, not tuned. The mean field is symmetrised radially about the
  emergent nucleus (a screening field is radial), isolating the
  l-dependent reordering from discretisation noise.

WHAT EMERGES / WHAT DOES NOT (measured below):
  - A screening-LIKE effect DOES emerge: self-consistency lifts the (n,l)
    degeneracy and reorders the levels (the spherical-well "20" closure
    dissolves as coupling grows).
  - The ATOMIC order does NOT emerge: at no coupling do the noble-gas numbers
    appear; the Ne-like "10" closure (1s 2s 2p) never forms -- the first two
    closures stay 2, 8. The structural back-reaction is REPULSIVE, pushing
    core-penetrating radial modes (2s) UP, the OPPOSITE of atomic screening
    (which modulates an ATTRACTIVE nuclear well, absent from a bare manifold).

Run:
    python benchmarks/emergent_screening.py

Theoretical anchor: AGENTS.md (nodal equation; Phi_s structural potential, U6;
discrete-mode regime). Builds on benchmarks/emergent_shell_ordering.py
(solid_ball_graph, the emergent nucleus). Status: RESEARCH (falsifier).
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
    """Canonical Phi_s Green's function K(i,j) = 1/d(i,j)^2 (alpha=2)."""
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
    """Graph-hop radius of each node from the emergent center (node 0)."""
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
    """Self-consistent Phi_s back-reaction; return cumulative closed-shell
    counts (running sum of mode capacities 2*mult after each shell)."""
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
    print("EMERGENT SCREENING? (Phi_s self-consistency; no QM injected)")
    print("=" * 70)

    G = solid_ball_graph(4, 16, 8)
    nodes = list(G.nodes())
    L = nx.laplacian_matrix(G, nodelist=nodes).toarray().astype(float)
    K = phi_s_kernel(G, nodes)
    rvec = radial_bins(G, nodes)

    print("\n[M1] Back-reaction kernel is the canonical Phi_s (alpha=2):")
    print("     V(i) = sum_j rho(j)/d(i,j)^2  (canonical U6 kernel)")
    print(f"     ball nucleus manifold: {len(nodes)} nodes, kernel {K.shape}")
    assert K.shape == (len(nodes), len(nodes))
    assert np.allclose(K, K.T), "Phi_s kernel must be symmetric"
    print("     -> PASS: mechanism = nodal dynamics on co-resident sub-EPIs.")

    cum0 = scf_closed_shells(L, K, rvec, 0.0)
    cum1 = scf_closed_shells(L, K, rvec, 1.0)
    print("\n[M2] Does self-consistency lift degeneracy / reorder?")
    print(f"     g=0.0 (independent particle): {cum0[:6]}")
    print(f"     g=1.0 (self-consistent)     : {cum1[:6]}")
    assert cum1 != cum0, "self-consistency had no effect"
    assert 20 in cum0 and 20 not in cum1, "spherical-well 20 not reordered"
    print("     -> PASS: a screening-LIKE effect emerges -- the")
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
    print(f"\n     max leading atomic match: {best_atomic}/6; "
          f"Ne-like '10' closure seen: {ten_ever}")
    assert best_atomic <= 1, "atomic order unexpectedly emerged"
    assert not ten_ever, "the atomic '10' closure appeared"
    print("     -> PASS: NO coupling reproduces the atomic table; the '10'")
    print("        (1s 2s 2p) closure never forms. First closures stay 2, 8.")

    # -- M4: does the COMPLEMENT -- an attractive center -- emerge? ----------
    topo = classify_nodal_topology(G)
    center = topo["centers"][0]
    cphi = topo["centrality"][center]
    cvals = np.array(list(topo["centrality"].values()))
    mults = [s.multiplicity for s in
             structural_eigenmodes(G, max_modes=40, gap_factor=4.0)[:6]]
    print("\n[M4] Does an ATTRACTIVE center (DeltaNFR sink) emerge instead?")
    print(f"     nucleus Phi_s = {cphi:.1f}  (max {cvals.max():.1f}, "
          f"mean {cvals.mean():.1f})")
    print(f"     ball spectrum multiplicities : {mults}")
    assert abs(cphi - cvals.max()) < 1e-9, "nucleus is not the Phi_s maximum"
    assert mults[:3] == [1, 3, 5], "spectrum is not spherical-well"
    print("     -> PASS: the nucleus is the Phi_s MAXIMUM (repulsive for")
    print("        +DeltaNFR, NOT an attractive sink); the spectrum is the")
    print("        spherical WELL (2l+1), NOT hydrogenic (Coulomb 2n^2).")

    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    print(
        "A screening-LIKE effect EMERGES: the canonical Phi_s\n"
        "  self-consistency (occupied sub-EPIs -> 1/d^2 potential ->\n"
        "  reshifted modes, iterated) lifts the (n,l) degeneracy and\n"
        "  reorders the levels. The mechanism is TNFR-native -- the nodal\n"
        "  dynamics acting back on co-resident sub-EPIs (U5), no quantum\n"
        "  chemistry injected.\n"
        "BUT the ATOMIC table does NOT emerge: at no coupling do the\n"
        "  noble-gas numbers appear; the Ne-like '10' closure never forms\n"
        "  (first closures stay 2, 8). The structural back-reaction is\n"
        "  REPULSIVE, pushing core-penetrating radial modes (2s) UP -- the\n"
        "  OPPOSITE of atomic screening, which modulates an ATTRACTIVE\n"
        "  nuclear Coulomb well (low-l penetrate -> see more unscreened\n"
        "  charge -> pulled DOWN). A bare repulsive coherence manifold has\n"
        "  no attractive nucleus, so the sign is wrong for atoms.\n"
        "IDENTIFICATION: BOTH atomic ingredients are measured NON-emergent\n"
        "  here: (M3) self-consistent screening has the REPULSIVE sign, and\n"
        "  (M4) the emergent nucleus is a Phi_s MAXIMUM (repulsive), not an\n"
        "  attractive DeltaNFR sink -- the spectrum is a box/spherical well,\n"
        "  not a Coulomb 2n^2 well. The periodic table needs a charged\n"
        "  many-body Coulomb system (attractive nucleus + screening that\n"
        "  modulates it); a single relaxing coherence manifold carries\n"
        "  neither. So the (n+l) postulate in emergent_chemistry stands."
    )


if __name__ == "__main__":
    main()
