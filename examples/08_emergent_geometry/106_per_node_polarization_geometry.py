#!/usr/bin/env python3
"""
Example 106 — Polarization Geometry of the Auxiliary Substrate
===============================================================

The auxiliary substrate represents each extracted graph-field tuple
``(K_phi, J_phi, Phi_s, J_dNFR)`` as a classical complex doublet.  Its
quadratic isotropic Hamiltonian admits U(2) rotations, and the associated
Stokes vector obeys the usual Poincare-sphere identities.  These are exact
identities of the declared ambient model.

The per-node tuples are computed from a coupled graph snapshot.  Calling them
N doublets does not make their coordinates dynamically or statistically
independent, and it does not show that graph trajectories remain on the
ambient harmonic flow.  The global data are a classical collection of
doublets, not a quantum tensor-product state.

Experiment 1 verifies algebraic identities and conserved Stokes quantities
under ``evolve_substrate_flow`` itself.  Experiment 2 instead applies each
engine operator once to one seeded fixture and compares the extracted
before/after vectors.  That table is a finite response audit, not a general
classification of induced symplectic maps or grammar-valid words.  Experiment
3 uses explicit encodings for three domain fixtures and reports read-outs; it
does not derive those encodings from the nodal equation.

References
----------
- examples/08_emergent_geometry/98_emergent_symplectic_substrate.py (substrate + polarization)
- examples/08_emergent_geometry/103_emergent_substrate_meets_riemann.py
  (constructed P14 phase encoding)
- examples/08_emergent_geometry/104_navier_stokes_is_not_riemann.py (NS adapter)
- examples/07_number_theory/101_numbers_as_coupled_network.py (primes = low-coupling periphery)
- src/tnfr/physics/symplectic_substrate.py (polarization_density,
  polarization_vector, evolve_substrate_flow)
- AGENTS.md §"Emergent Symplectic Substrate" (polarization symmetry U(2))
"""

import copy
import math
import os
import random
import statistics
import sys
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import networkx as nx
import numpy as np
from sympy import isprime

from tnfr.constants import inject_defaults
from tnfr.dynamics.dnfr import default_compute_delta_nfr
from tnfr.operators.definitions import (
    Coherence,
    Contraction,
    Coupling,
    Dissonance,
    Emission,
    Expansion,
    Mutation,
    Reception,
    Recursivity,
    Resonance,
    SelfOrganization,
    Silence,
    Transition,
)
from tnfr.operators.preconditions import OperatorPreconditionError
from tnfr.physics.symplectic_substrate import (
    evolve_substrate_flow,
    extract_phase_space_point,
    polarization_density,
    polarization_vector,
)


def _substrate_density(G):
    """Polarization density, guarding the Φ_s 0/0 on trivial-ΔNFR graphs."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with np.errstate(invalid="ignore", divide="ignore"):
            pt = extract_phase_space_point(G)
            dens = polarization_density(pt)
    return pt, dens


def _geo_polarization_energy(G):
    """Clean geometric-sector energy e_geo = ½(K_φ² + J_φ²) = ½|Ψ|²."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with np.errstate(invalid="ignore", divide="ignore"):
            pt = extract_phase_space_point(G)
    k = np.asarray(pt.k_phi, dtype=float)
    j = np.asarray(pt.j_phi, dtype=float)
    return pt, 0.5 * (k * k + j * j)


# ============================================================================
# EXPERIMENT 1 (direction 2): intrinsic polarization geometry & its dynamics
# ============================================================================
def experiment_1_intrinsic():
    """Full-polarization identity, Stokes conserved under flow, product state."""
    print("=" * 72)
    print("EXPERIMENT 1: Intrinsic Polarization Geometry of the Substrate")
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

    # (A) Hopf identity: per-node |polarization| = energy (Poincaré sphere)
    res = float(np.max(np.abs(dens["radius"] - dens["energy"])))
    unit = float(np.max(np.abs(np.linalg.norm(dens["poincare"], axis=0) - 1)))
    print("A. Poincaré sphere: each node is fully polarized, radius = energy")
    print(f"   max |radius − energy| = {res:.1e}  (machine zero → EXACT)")
    print(f"   Poincaré vectors are unit:  max ||p|−1| = {unit:.1e}")
    print()

    # (B) Stokes vector conserved under the diagonal substrate flow
    p0 = polarization_vector(pt)
    drift = 0.0
    for t in (0.5, 1.3, 2.7, 4.0):
        pt_t = polarization_vector(evolve_substrate_flow(pt, t))
        drift = max(drift, max(abs(pt_t[k] - p0[k]) for k in ("p_1", "p_2", "p_3")))
    print("B. Stokes vector under the diagonal substrate flow (U(1) center):")
    print(f"   P₁,P₂,P₃ drift over flow times [0.5,1.3,2.7,4.0]: {drift:.1e}")
    print("   → the polarization vector is CONSTANT: both sectors rotate by")
    print("     the same phase e^(−it), so the Stokes vector is conserved (no")
    print("     precession). The SU(2) part would rotate it, but is NOT the flow.")
    print()

    # (C) polarization texture: neighbor alignment (honest negative on random)
    idx = {n: i for i, n in enumerate(pt.nodes)}
    poincare = dens["poincare"]
    rng2 = np.random.default_rng(0)
    neigh = [
        float(np.dot(poincare[:, idx[a]], poincare[:, idx[b]]))
        for a, b in G.edges()
        if a in idx and b in idx
    ]
    rand = []
    for _ in range(len(neigh)):
        a, b = rng2.choice(len(pt.nodes), 2, replace=False)
        rand.append(float(np.dot(poincare[:, a], poincare[:, b])))
    print("C. Polarization texture (neighbor Poincaré-vector alignment):")
    print(
        f"   mean neighbor p·p = {np.mean(neigh):+.3f},  "
        f"random = {np.mean(rand):+.3f}"
    )
    print("   → this one neighbor sample shows no excess over its random-pair")
    print("     control; it is not a population-level ordering result.")
    print()

    # (D) honest scope: product state, no entanglement
    print("D. SCOPE: the read-out contains one classical doublet per node.")
    print(f"   The {len(pt.nodes)} tuples depend on coupled graph fields and are")
    print("   not independent dynamical coordinates. No quantum state or")
    print("   entanglement is defined by this classical Poincare map.")
    print()


# ============================================================================
# EXPERIMENT 2 (direction 3): which operators rotate the Stokes vector
# ============================================================================
def experiment_2_operators():
    """Finite before/after response table on one seeded fixture."""
    print("=" * 72)
    print("EXPERIMENT 2: Finite Operator-to-Read-out Response Table")
    print("=" * 72)
    print()
    print("Apply each operator once to every node of one seeded fixture and")
    print("compare the extracted global Stokes vectors. These isolated calls")
    print("are not a grammar word or a universal operator classification.")
    print()

    ops = [
        ("AL", Emission),
        ("EN", Reception),
        ("IL", Coherence),
        ("OZ", Dissonance),
        ("UM", Coupling),
        ("RA", Resonance),
        ("SHA", Silence),
        ("VAL", Expansion),
        ("NUL", Contraction),
        ("THOL", SelfOrganization),
        ("ZHIR", Mutation),
        ("NAV", Transition),
        ("REMESH", Recursivity),
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

    def stokes_vec(G):
        c = polarization_vector(extract_phase_space_point(G))
        return np.array([c["p_1"], c["p_2"], c["p_3"]])

    p0 = stokes_vec(G0)
    rotators, preservers, blocked = [], [], []
    print(f"  {'op':>7} {'Stokes rotation (deg)':>22}")
    print("  " + "-" * 31)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for glyph, cls in ops:
            G = copy.deepcopy(G0)
            op = cls()
            try:
                for nd in list(G.nodes()):
                    op(G, nd)
            except OperatorPreconditionError as exc:
                blocked.append((glyph, str(exc).split(":", 1)[-1].strip()))
                print(f"  {glyph:>7} {'BLOCKED':>22}")
                continue
            p1 = stokes_vec(G)
            cos = np.dot(p0, p1) / (np.linalg.norm(p0) * np.linalg.norm(p1) + 1e-30)
            ang = math.degrees(math.acos(max(-1.0, min(1.0, cos))))
            print(f"  {glyph:>7} {ang:>22.2f}")
            (rotators if ang > 1.0 else preservers).append(glyph)
    print()
    print(f"  ROTATORS  (> 1°): {rotators}")
    print(f"  PRESERVERS (≤ 1°): {preservers}")
    print(f"  BLOCKED by hard precondition: {[glyph for glyph, _ in blocked]}")
    print()
    print("  → these labels describe only the >1-degree rule on this fixture.")
    print("    A general preserve/rotate claim would require trajectories,")
    print("    grammar context, and a proof for each induced field map.")
    print()


# ============================================================================
# EXPERIMENT 3 (direction 1): the polarization field in the studied networks
# ============================================================================
def experiment_3_networks():
    """Geometric-sector polarization energy in P14, arithmetic, and NS."""
    print("=" * 72)
    print("EXPERIMENT 3: The Polarization Field in the Networks We Studied")
    print("=" * 72)
    print()
    print("Read the clean geometric-sector energy e_geo = ½|Ψ|² =")
    print("½(K_φ² + J_φ²) (no Φ_s degeneracy) in each network.")
    print()

    # P14 with a deliberately imposed phase encoding theta = nu_f.
    from tnfr.riemann.prime_ladder_hamiltonian import build_prime_ladder_graph

    Gp = build_prime_ladder_graph(10, max_power=4)
    for nd in Gp.nodes():
        Gp.nodes[nd]["phase"] = float(Gp.nodes[nd]["nu_f"] % (2 * math.pi))
    pt, eg = _geo_polarization_energy(Gp)
    idx = {n: i for i, n in enumerate(pt.nodes)}
    primes = sorted({p for (p, _k) in Gp.nodes()})
    by_p = {}
    for p, k in Gp.nodes():
        by_p.setdefault(p, []).append(eg[idx[(p, k)]])
    mean_eg = [float(np.mean(by_p[p])) for p in primes]
    r = float(np.corrcoef(mean_eg, [math.log(p) for p in primes])[0, 1])
    print(f"  P14 (phase=nu_f encoding): r(geo energy, log p) = {r:.3f}")
    print(
        "    → the read-out retains the prime ladder inserted into phase" " (Ex 103)."
    )
    print()

    # Arithmetic number network
    from tnfr.mathematics.number_theory import ArithmeticTNFRNetwork

    net = ArithmeticTNFRNetwork(max_number=80)
    G = net.graph.to_undirected()
    for nd in G.nodes():
        G.nodes[nd]["phase"] = float(2 * math.pi * nd / 80)
        G.nodes[nd]["theta"] = G.nodes[nd]["phase"]
    pt, eg = _geo_polarization_energy(G)
    ia = {n: i for i, n in enumerate(pt.nodes)}
    egp = statistics.mean(eg[ia[n]] for n in G.nodes() if isprime(n))
    egc = statistics.mean(eg[ia[n]] for n in G.nodes() if not isprime(n))
    print(
        f"  Arithmetic: geo polariz. energy  prime = {egp:.3f}, "
        f"composite = {egc:.3f}"
    )
    print("    → in this n<=80 fixture, the sampled prime mean is lower.")
    print("      No population-wide or causal claim follows from this encoding.")
    print()

    # Navier–Stokes
    from tnfr.navier_stokes.operator import (
        build_torus_graph_3d,
        taylor_green_initial_condition_3d,
    )

    Gn = build_torus_graph_3d(8)
    u, _v, _w = taylor_green_initial_condition_3d(Gn, 1.0)
    for i, nd in enumerate(list(Gn.nodes)):
        encoded_phase = float(u[i] % (2 * math.pi))
        Gn.nodes[nd]["phase"] = encoded_phase
        Gn.nodes[nd]["theta"] = encoded_phase
    pt, eg = _geo_polarization_energy(Gn)
    print(
        f"  NS (3D Taylor–Green): total geo polariz. energy Σe_geo = "
        f"{float(np.sum(eg)):.2f}"
    )
    print("    → after velocity is encoded as phase, the auxiliary geometric")
    print("      sector supplies the displayed derived texture (Ex 104).")
    print()
    print("  SCOPE: each row maps an explicitly prepared graph snapshot to the")
    print("  auxiliary field doublet. It re-expresses those inputs in classical")
    print("  polarization language and adds no new domain closure.")
    print()


def main():
    print()
    print("  TNFR Example 106: The Per-Node Polarization Geometry")
    print("  U(2) polarization symmetry, Stokes vector, Poincaré sphere")
    print("  ==========================================================")
    print()
    experiment_1_intrinsic()
    experiment_2_operators()
    experiment_3_networks()
    print("=" * 72)
    print("WHAT THIS ESTABLISHES")
    print("=" * 72)
    print()
    print("Nonzero auxiliary doublets obey the classical Stokes/Poincare")
    print("identities, and their Stokes vector is conserved by the declared")
    print("diagonal harmonic flow. Per-node tuples remain graph-dependent")
    print("read-outs. The operator table is one finite snapshot-response audit;")
    print("it does not classify the induced maps generally. The domain examples")
    print("re-express explicitly encoded graph data and close no open program.")
    print()


if __name__ == "__main__":
    main()
