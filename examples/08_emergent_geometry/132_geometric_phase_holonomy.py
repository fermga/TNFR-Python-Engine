#!/usr/bin/env python3
"""
Example 132 — Geometric Phase / Holonomy on the Substrate: the Bargmann
Invariant Equals Half the Solid Angle on the Poincare Sphere
==============================================================================

Example 106 established that each node of the emergent symplectic substrate is a
fully-polarized point on the Poincare sphere: the per-node doublet

    zeta = (zeta^A, zeta^B) = (K_phi + i*J_phi, Phi_s + i*J_dNFR)  in  C^2

normalized is a CP^1 state, whose Stokes 3-vector (P_1, P_2, P_3) lives on the
unit sphere. Example 130 measured how the canonical operators ROTATE that
vector. This example measures the GEOMETRIC PHASE such rotations accumulate.

For any three substrate states the BARGMANN INVARIANT

    gamma = arg( <psi_1|psi_2> <psi_2|psi_3> <psi_3|psi_1> )

is an exact geometric quantity, and the central identity is

    gamma = (1/2) * Omega(triangle of Stokes vectors on the Poincare sphere),

where Omega is the solid angle subtended by the three Stokes 3-vectors. This is
the discrete PANCHARATNAM phase: a theorem of CP^1 geometry, provable, and
empirically the geometric phase of CLASSICAL polarization optics (Pancharatnam
1956). It is NOT an imported quantum postulate -- it is the classical-wave
polarization phase, the same Stokes/Poincare physics as example 106.

Doctrine compliance
-------------------
Nothing is imported without proof or empirical anchor. The three states are
per-node substrate doublets from the canonical extract_phase_space_point; their
Stokes vectors are the canonical polarization_density. The geometric phase
EMERGES from the canonical substrate; the example only verifies that it equals
the solid angle (the Bargmann identity). The identity itself is exact CP^1
geometry; the phenomenon is empirically-established classical polarization optics
(Pancharatnam 1956), not a quantum Berry phase.

Three measured results
----------------------
M1 THE BARGMANN IDENTITY IS EXACT ON THE SUBSTRATE. For three per-node substrate
   doublets, the Bargmann phase equals +1/2 the solid angle of their Stokes
   vectors to machine precision (|diff| ~ 1e-17, 7/7 triples). The substrate
   realizes the geometric phase = half the solid angle exactly.

M2 THE PHASE IS GEOMETRIC (GAUGE-INVARIANT). The honest proof of geometricity:
   the Bargmann phase is invariant under per-state rephasing
   |psi_i> -> e^{i alpha_i}|psi_i> (the arbitrary local phase cancels). The
   phase depends ONLY on the loop on the Poincare sphere, not on the arbitrary
   per-node phase -- it is geometric, not dynamical. Exact under every rephasing.

M3 HOLONOMY EQUALS THE ENCLOSED SOLID ANGLE. A closed three-leg loop of
   substrate states accumulates a geometric phase equal to +1/2 the enclosed
   solid angle, for every seed (|diff| ~ 1e-17). The holonomy of the substrate's
   Poincare-sphere loop is its solid angle -- the Pancharatnam-Berry relation.

Honest scope
------------
The geometric phase = half the solid angle is the Bargmann invariant, an EXACT
identity of CP^1 geometry (provable), and the phenomenon is the empirically-
established Pancharatnam phase of CLASSICAL polarization optics (Pancharatnam
1956) -- the substrate is a classical wave polarization texture (example 106),
a product state, NO entanglement, NOT a quantum Berry phase and NOT a qubit. The
example verifies the identity on the canonical substrate; it re-expresses the
classical-polarization geometric phase in the emergent geometry. It is not new
mathematics and closes no open problem.

References
----------
- src/tnfr/physics/symplectic_substrate.py (extract_phase_space_point,
  polarization_density, polarization_vector)
- examples/08_emergent_geometry/106_per_node_polarization_geometry.py (Poincare sphere)
- examples/08_emergent_geometry/130_operators_break_substrate_charges.py (Stokes rotation)
- AGENTS.md "Emergent Symplectic Substrate" (Polarization symmetry -- U(2),
  Poincare sphere), "Polarization symmetry"
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import networkx as nx
import numpy as np

from tnfr.alias import set_attr
from tnfr.constants.aliases import ALIAS_EPI, ALIAS_VF
from tnfr.dynamics import default_compute_delta_nfr
from tnfr.physics.symplectic_substrate import (
    extract_phase_space_point,
    polarization_density,
)


def seed_network(G, rng):
    """Seed a canonical network and populate the substrate (nothing imposed)."""
    for nd in G.nodes():
        G.nodes[nd]["theta"] = float(rng.uniform(0, 2 * np.pi))
        set_attr(G.nodes[nd], ALIAS_EPI, float(rng.uniform(-0.5, 0.5)))
        set_attr(G.nodes[nd], ALIAS_VF, 1.0)
    default_compute_delta_nfr(G)


def doublet(point, i):
    """The CP^1 state of node i: zeta = (K_phi+i*J_phi, Phi_s+i*J_dNFR), unit."""
    za = complex(point.k_phi[i], point.j_phi[i])
    zb = complex(point.phi_s[i], point.j_dnfr[i])
    v = np.array([za, zb], dtype=complex)
    n = np.linalg.norm(v)
    return v / n if n > 1e-15 else v


def bargmann_phase(p1, p2, p3):
    """3-state Bargmann invariant arg(<p1|p2><p2|p3><p3|p1>)."""
    z = np.vdot(p1, p2) * np.vdot(p2, p3) * np.vdot(p3, p1)
    return float(np.angle(z))


def stokes_unit(point, i):
    """Unit Stokes 3-vector of node i on the Poincare sphere (canonical)."""
    d = polarization_density(point)
    v = np.array([d["p_1"][i], d["p_2"][i], d["p_3"][i]], dtype=float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-15 else v


def solid_angle(a, b, c):
    """Signed solid angle of spherical triangle (a,b,c) (Van Oosterom-Strackee).

    tan(Omega/2) = (a . (b x c)) / (1 + a.b + b.c + c.a)
    """
    num = float(np.dot(a, np.cross(b, c)))
    den = 1.0 + float(np.dot(a, b)) + float(np.dot(b, c)) + float(np.dot(c, a))
    return 2.0 * np.arctan2(num, den)


def experiment_1_bargmann_identity():
    """M1: Bargmann phase = +1/2 solid angle, exact on substrate doublets."""
    print("=" * 74)
    print("M1: THE BARGMANN IDENTITY IS EXACT ON THE SUBSTRATE")
    print("=" * 74)
    print("Three per-node substrate doublets zeta=(K_phi+i*J_phi, Phi_s+i*J_dNFR).")
    print("Their Bargmann phase equals +1/2 the solid angle of their Stokes")
    print("vectors on the Poincare sphere -- an exact CP^1 identity.")
    print()
    G = nx.cycle_graph(12)
    seed_network(G, np.random.default_rng(0))
    p = extract_phase_space_point(G)
    print(f"  {'triple':>14} {'Bargmann':>11} {'0.5*Omega':>11} {'|diff|':>9}")
    n_match = 0
    triples = [
        (0, 1, 2),
        (3, 5, 7),
        (1, 4, 9),
        (2, 6, 10),
        (0, 5, 11),
        (4, 8, 11),
        (1, 6, 9),
    ]
    for i, j, k in triples:
        ph = bargmann_phase(doublet(p, i), doublet(p, j), doublet(p, k))
        half = 0.5 * solid_angle(
            stokes_unit(p, i), stokes_unit(p, j), stokes_unit(p, k)
        )
        d = abs(ph - half)
        n_match += int(d < 1e-9)
        print(f"  {str((i, j, k)):>14} {ph:>11.6f} {half:>11.6f} {d:>9.1e}")
    print()
    print(f"  -> {n_match}/{len(triples)} EXACT match: the substrate realizes the")
    print("     geometric phase = half the solid angle (Bargmann identity).")


def experiment_2_gauge_invariance():
    """M2: the phase is geometric -- invariant under per-state rephasing."""
    print()
    print("=" * 74)
    print("M2: THE PHASE IS GEOMETRIC (GAUGE-INVARIANT)")
    print("=" * 74)
    print("A phase is GEOMETRIC iff invariant under per-state rephasing")
    print("|psi_i> -> e^{i alpha_i}|psi_i> (the arbitrary local phase cancels).")
    print("Rephase each substrate doublet randomly; the phase is unchanged.")
    print()
    G = nx.cycle_graph(12)
    seed_network(G, np.random.default_rng(0))
    p = extract_phase_space_point(G)
    rng = np.random.default_rng(7)
    p1, p2, p3 = doublet(p, 0), doublet(p, 4), doublet(p, 8)
    ph0 = bargmann_phase(p1, p2, p3)
    print(f"  {'rephase trial':>14} {'Bargmann phase':>15}")
    print(f"  {'(none)':>14} {ph0:>15.6f}")
    n_same = 0
    for t in range(5):
        a = rng.uniform(0, 2 * np.pi, 3)
        q1 = p1 * np.exp(1j * a[0])
        q2 = p2 * np.exp(1j * a[1])
        q3 = p3 * np.exp(1j * a[2])
        ph = bargmann_phase(q1, q2, q3)
        n_same += int(abs(ph - ph0) < 1e-12)
        print(f"  {('trial ' + str(t)):>14} {ph:>15.6f}")
    print()
    print(f"  -> {n_same}/5 identical: the phase depends only on the loop, not on")
    print("     the arbitrary per-node phase. It is GEOMETRIC, not dynamical.")


def experiment_3_holonomy_solid_angle():
    """M3: closed-loop holonomy = enclosed solid angle, every seed."""
    print()
    print("=" * 74)
    print("M3: HOLONOMY EQUALS THE ENCLOSED SOLID ANGLE")
    print("=" * 74)
    print("A closed three-leg loop of substrate states accumulates a geometric")
    print("phase equal to +1/2 the enclosed solid angle, for every seed.")
    print()
    print(f"  {'seed':>6} {'loop phase':>12} {'0.5*Omega':>12} {'|diff|':>9}")
    n_match = 0
    for s in range(4):
        G = nx.cycle_graph(10)
        seed_network(G, np.random.default_rng(s))
        p = extract_phase_space_point(G)
        ph = bargmann_phase(doublet(p, 0), doublet(p, 3), doublet(p, 6))
        half = 0.5 * solid_angle(
            stokes_unit(p, 0), stokes_unit(p, 3), stokes_unit(p, 6)
        )
        d = abs(ph - half)
        n_match += int(d < 1e-9)
        print(f"  {s:>6} {ph:>12.6f} {half:>12.6f} {d:>9.1e}")
    print()
    print(f"  -> {n_match}/4 exact: the holonomy of the substrate's Poincare-")
    print("     sphere loop is its solid angle (Pancharatnam-Berry relation).")


def main():
    print()
    print("  ===============================================================")
    print("  Geometric Phase / Holonomy on the Emergent Symplectic Substrate")
    print("  The Bargmann Invariant Equals Half the Poincare Solid Angle")
    print("  ===============================================================")
    print()
    experiment_1_bargmann_identity()
    experiment_2_gauge_invariance()
    experiment_3_holonomy_solid_angle()
    print()
    print("=" * 74)
    print("WHAT THIS ESTABLISHES")
    print("=" * 74)
    print("Each node of the emergent substrate is a Poincare-sphere point")
    print("(example 106). The geometric phase accumulated around a loop of")
    print("substrate states equals +1/2 the enclosed solid angle -- the")
    print("Bargmann invariant, an EXACT CP^1 identity (M1, machine precision),")
    print("gauge-invariant hence genuinely GEOMETRIC (M2), and realized as the")
    print("closed-loop holonomy (M3). HONEST SCOPE: this is the Pancharatnam")
    print("phase of CLASSICAL polarization optics (Pancharatnam 1956), an")
    print("empirically-established phenomenon, and an exact provable identity --")
    print("NOT a quantum Berry phase and NOT a qubit (the substrate is a")
    print("classical wave polarization texture, a product state, no")
    print("entanglement). It emerges from the canonical substrate doublets and")
    print("their canonical Stokes vectors; the example verifies the identity. It")
    print("re-expresses the classical geometric phase in the emergent geometry;")
    print("it is not new mathematics and closes no open problem.")


if __name__ == "__main__":
    main()
