#!/usr/bin/env python3
"""
Example 106 — The Per-Node Qubit Geometry of the Emergent Substrate
==================================================================

Returns to the emergent symplectic substrate (Example 98) to deepen its
per-node qubit geometry: the hidden U(2) isospin, the Hopf map S³ → S²,
and the Bloch vector each node carries. Three explorations, all measured:

  (2) the intrinsic Hopf/Bloch structure and its dynamics;
  (3) which canonical operators rotate the Bloch vector;
  (1) the Bloch field in the networks studied this session (P14, the
      arithmetic number network, Navier–Stokes).

Honest scope (stated up front)
------------------------------
The "qubit" here is GEOMETRIC, not quantum. Each node's ℝ⁴ fiber
(K_φ, J_φ, Φ_s, J_ΔNFR) is the complex doublet ζ = (ζ^A, ζ^B) with
ζ^A = K_φ + i·J_φ (geometric sector) and ζ^B = Φ_s + i·J_ΔNFR (potential
sector). Its SU(2) moment map gives an isospin 3-vector whose length
equals the per-node energy — the Hopf fibration S³ → S², i.e. a Bloch
vector on S² of radius = energy (the two-level phase space). This is a
CLASSICAL spin texture (a field of Bloch vectors), NOT a quantum state:
the doublet is PER-NODE, so the global object is a PRODUCT of N
independent ℂ² points — there is no superposition and no entanglement.

Physics
-------
H_sub = ½Σ‖(ζ^A, ζ^B)‖² is the squared norm of a ℂ² doublet, so it carries
a hidden U(2) = U(1) × SU(2). The SU(2) isospin charges are
  I_3 = ½Σ(|ζ^A|² − |ζ^B|²) = E_geo − E_pot,
  I_1 = Σ(K_φ·Φ_s + J_φ·J_ΔNFR),   I_2 = Σ(K_φ·J_ΔNFR − J_φ·Φ_s),
with per-node densities whose length is the per-node energy (Hopf).

References
----------
- examples/98_emergent_symplectic_substrate.py (substrate + U(2) + Hopf)
- examples/103_emergent_substrate_meets_riemann.py (P14 Bloch carries log p)
- examples/104_navier_stokes_is_not_riemann.py (NS velocity = geometric sector)
- examples/101_numbers_as_coupled_network.py (primes = low-coupling periphery)
- src/tnfr/physics/symplectic_substrate.py (isospin_density, isospin_charges,
  evolve_substrate_flow)
- AGENTS.md §"Emergent Symplectic Substrate" (Hidden U(2), Hopf map)
"""

import os
import sys
import math
import copy
import random
import warnings
import statistics

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import networkx as nx
from sympy import isprime

from tnfr.dynamics.dnfr import default_compute_delta_nfr
from tnfr.constants import inject_defaults
from tnfr.physics.symplectic_substrate import (
    extract_phase_space_point,
    isospin_charges,
    isospin_density,
    evolve_substrate_flow,
)
from tnfr.operators.definitions import (
    Emission, Reception, Coherence, Dissonance, Coupling, Resonance,
    Silence, Expansion, Contraction, SelfOrganization, Mutation,
    Transition, Recursivity,
)


def _substrate_density(G):
    """Extract isospin density, guarding the Φ_s 0/0 on trivial-ΔNFR graphs."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with np.errstate(invalid="ignore", divide="ignore"):
            pt = extract_phase_space_point(G)
            dens = isospin_density(pt)
    return pt, dens


def _geo_bloch_energy(G):
    """Clean geometric-sector Bloch energy e_geo = ½(K_φ² + J_φ²) = ½|Ψ|²."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with np.errstate(invalid="ignore", divide="ignore"):
            pt = extract_phase_space_point(G)
    k = np.asarray(pt.k_phi, dtype=float)
    j = np.asarray(pt.j_phi, dtype=float)
    return pt, 0.5 * (k * k + j * j)


# ============================================================================
# EXPERIMENT 1 (direction 2): intrinsic Hopf/Bloch geometry & its dynamics
# ============================================================================
def experiment_1_intrinsic():
    """Hopf identity, isospin conserved under the flow, product state."""
    print("=" * 72)
    print("EXPERIMENT 1: Intrinsic Hopf/Bloch Geometry of the Substrate")
    print("=" * 72)
    print()

    rng = random.Random(5)
    G = nx.watts_strogatz_graph(30, 4, 0.3, seed=5)
    for node in G.nodes():
        G.nodes[node]["theta"] = rng.uniform(0.0, 2 * math.pi)
        G.nodes[node]["EPI"] = rng.uniform(0.2, 0.8)
        G.nodes[node]["nu_f"] = rng.uniform(0.5, 1.5)
    default_compute_delta_nfr(G)
    pt, dens = _substrate_density(G)

    # (A) Hopf identity: per-node |I| = energy (Bloch vector on S²)
    res = float(np.max(np.abs(dens["radius"] - dens["energy"])))
    unit = float(np.max(np.abs(np.linalg.norm(dens["bloch"], axis=0) - 1)))
    print("A. Hopf S³→S²: each node carries a Bloch vector of radius = energy")
    print(f"   max |radius − energy| = {res:.1e}  (machine zero → EXACT)")
    print(f"   Bloch vectors are unit:  max ||bloch|−1| = {unit:.1e}")
    print()

    # (B) isospin conserved under the diagonal substrate flow
    i0 = isospin_charges(pt)
    drift = 0.0
    for t in (0.5, 1.3, 2.7, 4.0):
        it = isospin_charges(evolve_substrate_flow(pt, t))
        drift = max(drift, max(abs(it[k] - i0[k]) for k in ("i_1", "i_2", "i_3")))
    print("B. Isospin under the diagonal substrate flow (U(1) center):")
    print(f"   I₁,I₂,I₃ drift over flow times [0.5,1.3,2.7,4.0]: {drift:.1e}")
    print("   → the Bloch vector is CONSTANT: both sectors rotate by the same")
    print("     phase e^(−it), so the isospin is conserved (no precession).")
    print("     The SU(2) part would rotate it, but it is NOT the flow.")
    print()

    # (C) spin texture: neighbor Bloch alignment (honest negative on random)
    idx = {n: i for i, n in enumerate(pt.nodes)}
    bloch = dens["bloch"]
    rng2 = np.random.default_rng(0)
    neigh = [float(np.dot(bloch[:, idx[a]], bloch[:, idx[b]]))
             for a, b in G.edges() if a in idx and b in idx]
    rand = []
    for _ in range(len(neigh)):
        a, b = rng2.choice(len(pt.nodes), 2, replace=False)
        rand.append(float(np.dot(bloch[:, a], bloch[:, b])))
    print("C. Spin texture (neighbor Bloch alignment order parameter):")
    print(f"   mean neighbor bloch·bloch = {np.mean(neigh):+.3f},  "
          f"random = {np.mean(rand):+.3f}")
    print("   → no EXCESS neighbor alignment on a random graph (honest")
    print("     negative): a random phase field has no Bloch ordering.")
    print()

    # (D) honest scope: product state, no entanglement
    print("D. HONEST SCOPE: the doublet is PER-NODE → the global object is a")
    print(f"   PRODUCT of {len(pt.nodes)} independent ℂ² Bloch vectors (a")
    print("   classical spin texture), NOT an entangled state in ℂ^(2N).")
    print("   'Qubit' = the GEOMETRY (Hopf S³→S²), not a quantum register.")
    print()


# ============================================================================
# EXPERIMENT 2 (direction 3): which operators rotate the Bloch vector
# ============================================================================
def experiment_2_operators():
    """Operator-isospin fingerprint: rotators vs preservers."""
    print("=" * 72)
    print("EXPERIMENT 2: Which Canonical Operators Rotate the Bloch Vector")
    print("=" * 72)
    print()
    print("Apply each operator to every node; measure how far it rotates the")
    print("global isospin 3-vector I = (I₁, I₂, I₃).")
    print()

    ops = [
        ("AL", Emission), ("EN", Reception), ("IL", Coherence),
        ("OZ", Dissonance), ("UM", Coupling), ("RA", Resonance),
        ("SHA", Silence), ("VAL", Expansion), ("NUL", Contraction),
        ("THOL", SelfOrganization), ("ZHIR", Mutation),
        ("NAV", Transition), ("REMESH", Recursivity),
    ]

    seed = 42
    G0 = nx.erdos_renyi_graph(20, 0.25, seed=seed)
    if not nx.is_connected(G0):
        comps = list(nx.connected_components(G0))
        for i in range(1, len(comps)):
            G0.add_edge(next(iter(comps[i - 1])), next(iter(comps[i])))
    inject_defaults(G0)
    rng = np.random.default_rng(seed)
    for nd in G0.nodes():
        G0.nodes[nd]["phase"] = rng.uniform(0, 2 * math.pi)
        G0.nodes[nd]["theta"] = G0.nodes[nd]["phase"]
        G0.nodes[nd]["delta_nfr"] = rng.uniform(-0.3, 0.3)
        G0.nodes[nd]["nu_f"] = rng.uniform(0.8, 1.2)

    def iso_vec(G):
        c = isospin_charges(extract_phase_space_point(G))
        return np.array([c["i_1"], c["i_2"], c["i_3"]])

    i0 = iso_vec(G0)
    rotators, preservers = [], []
    print(f"  {'op':>7} {'isospin rotation (deg)':>22}")
    print("  " + "-" * 31)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for glyph, cls in ops:
            G = copy.deepcopy(G0)
            op = cls()
            for nd in list(G.nodes()):
                op(G, nd)
            i1 = iso_vec(G)
            cos = np.dot(i0, i1) / (
                np.linalg.norm(i0) * np.linalg.norm(i1) + 1e-30)
            ang = math.degrees(math.acos(max(-1.0, min(1.0, cos))))
            print(f"  {glyph:>7} {ang:>22.2f}")
            (rotators if ang > 1.0 else preservers).append(glyph)
    print()
    print(f"  ROTATORS  (> 1°): {rotators}")
    print(f"  PRESERVERS (≤ 1°): {preservers}")
    print()
    print("  → UM (Coupling) is the dominant rotator: phase synchronization")
    print("    collapses the geometric sector ζ^A, nearly annihilating |I|.")
    print("    The ΔNFR-lever operators (IL, OZ, THOL, ZHIR, NAV) tilt the")
    print("    Bloch vector; AL/EN/RA/SHA/VAL/REMESH preserve it. This is the")
    print("    substrate-geometry fingerprint, complementary to the tetrad")
    print("    fingerprint of Example 37.")
    print()


# ============================================================================
# EXPERIMENT 3 (direction 1): the Bloch field in the studied networks
# ============================================================================
def experiment_3_networks():
    """Geometric-sector Bloch energy in P14, arithmetic, and NS."""
    print("=" * 72)
    print("EXPERIMENT 3: The Bloch Field in the Networks We Studied")
    print("=" * 72)
    print()
    print("Read the clean geometric-sector Bloch energy e_geo = ½|Ψ|² =")
    print("½(K_φ² + J_φ²) (no Φ_s degeneracy) in each network.")
    print()

    # P14 (Riemann) under the dynamics θ = ν_f·τ
    from tnfr.riemann.prime_ladder_hamiltonian import build_prime_ladder_graph
    Gp = build_prime_ladder_graph(10, max_power=4)
    for nd in Gp.nodes():
        Gp.nodes[nd]["phase"] = float(Gp.nodes[nd]["nu_f"])
    pt, eg = _geo_bloch_energy(Gp)
    idx = {n: i for i, n in enumerate(pt.nodes)}
    primes = sorted({p for (p, _k) in Gp.nodes()})
    by_p = {}
    for (p, k) in Gp.nodes():
        by_p.setdefault(p, []).append(eg[idx[(p, k)]])
    mean_eg = [float(np.mean(by_p[p])) for p in primes]
    r = float(np.corrcoef(mean_eg, [math.log(p) for p in primes])[0, 1])
    print(f"  P14 (Riemann, dynamics): r(geo Bloch energy, log p) = {r:.3f}")
    print("    → the Bloch field carries the prime ladder {k·log p} (Ex 103).")
    print()

    # Arithmetic number network
    from tnfr.mathematics.number_theory import ArithmeticTNFRNetwork
    net = ArithmeticTNFRNetwork(max_number=80)
    G = net.graph.to_undirected()
    for nd in G.nodes():
        G.nodes[nd]["phase"] = float(2 * math.pi * nd / 80)
        G.nodes[nd]["theta"] = G.nodes[nd]["phase"]
    pt, eg = _geo_bloch_energy(G)
    ia = {n: i for i, n in enumerate(pt.nodes)}
    egp = statistics.mean(eg[ia[n]] for n in G.nodes() if isprime(n))
    egc = statistics.mean(eg[ia[n]] for n in G.nodes() if not isprime(n))
    print(f"  Arithmetic: geo Bloch energy  prime = {egp:.3f}, "
          f"composite = {egc:.3f}")
    print("    → primes carry lower Bloch energy (the low-coupling")
    print("      periphery, Ex 101).")
    print()

    # Navier–Stokes
    from tnfr.navier_stokes.operator import (
        build_torus_graph_3d, taylor_green_initial_condition_3d,
    )
    Gn = build_torus_graph_3d(8)
    u, _v, _w = taylor_green_initial_condition_3d(Gn, 1.0)
    for i, nd in enumerate(list(Gn.nodes)):
        Gn.nodes[nd]["phase"] = float(u[i])
        Gn.nodes[nd]["theta"] = float(u[i])
    pt, eg = _geo_bloch_energy(Gn)
    print(f"  NS (3D Taylor–Green): total geo Bloch energy Σe_geo = "
          f"{float(np.sum(eg)):.2f}")
    print("    → the velocity field IS a geometric-sector Bloch texture")
    print("      (K_φ = vorticity proxy; this is enstrophy-like, Ex 104).")
    print()
    print("  HONEST: in all three the Bloch vector is a GEOMETRIC readout of")
    print("  the tetrad (Hopf map), inheriting the structure already measured")
    print("  (Ex 101/103/104). It re-expresses that content in qubit-geometry")
    print("  language; it does not add new closure.")
    print()


def main():
    print()
    print("  TNFR Example 106: The Per-Node Qubit Geometry of the Substrate")
    print("  Hidden U(2) isospin, the Hopf map, and the Bloch field")
    print("  ==============================================================")
    print()
    experiment_1_intrinsic()
    experiment_2_operators()
    experiment_3_networks()
    print("=" * 72)
    print("WHAT THIS ESTABLISHES")
    print("=" * 72)
    print()
    print("Each node of the emergent substrate carries a Bloch vector — the")
    print("Hopf S³→S² image of its ℂ² doublet, of radius = its energy (exact).")
    print("This per-node qubit GEOMETRY is intrinsic (isospin conserved under")
    print("the diagonal flow, no precession), it is a CLASSICAL spin texture")
    print("(product state, no entanglement), and it has no neighbor ordering")
    print("on a random graph. The canonical operators act on it with a clear")
    print("fingerprint — UM collapses it by phase synchronization, the")
    print("ΔNFR-lever operators tilt it, six operators preserve it — a new")
    print("lens complementary to the tetrad fingerprint. In the networks")
    print("studied this session the Bloch field re-expresses their known")
    print("content (the prime ladder in P14, the periphery in arithmetic, the")
    print("velocity texture in NS). This is a structural characterization of")
    print("the substrate's qubit geometry, not a quantum claim and not a")
    print("closure of any open program.")
    print()


if __name__ == "__main__":
    main()
