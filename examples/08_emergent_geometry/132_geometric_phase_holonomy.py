#!/usr/bin/env python3
"""
Example 132 — Bargmann Identity on Auxiliary Substrate Doublets
===============================================================

The auxiliary substrate packages each nonzero extracted tuple as a normalized
complex doublet

    zeta = (K_phi + i*J_phi, Phi_s + i*J_dNFR) in C^2.

For three nonorthogonal normalized doublets, the argument of their Bargmann
product agrees, modulo branch conventions, with half the oriented solid angle
of the corresponding Stokes vectors.  This is the standard CP^1 identity behind
the classical Pancharatnam phase.

The script checks selected triples in one seeded snapshot, verifies algebraic
rephasing invariance, and repeats a selected triple for four seeds.  The nodes
are used as three points in CP^1; no engine evolution, parallel transport, or
operator-generated closed path is performed.  Calling the result a holonomy is
therefore a geometric interpretation of the three-state invariant, not a
measured TNFR trajectory.

The doublets are auxiliary read-outs initialized from coupled graph fields.
The identity does not show that the nodal equation generates the ambient
substrate or the Pancharatnam phase, and it makes no quantum claim.

References
----------
- src/tnfr/physics/symplectic_substrate.py (extract_phase_space_point,
  polarization_density, polarization_vector)
- examples/08_emergent_geometry/106_per_node_polarization_geometry.py (Poincare sphere)
- examples/08_emergent_geometry/130_operators_break_substrate_charges.py (snapshot responses)
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
    """Impose one random graph snapshot for auxiliary-field extraction."""
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
    """Unit Stokes 3-vector of one extracted auxiliary doublet."""
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


def phase_distance(a, b):
    """Circular distance between two phase representatives."""
    return abs(float(np.angle(np.exp(1j * (a - b)))))


def experiment_1_bargmann_identity():
    """M1: Bargmann phase = +1/2 solid angle, exact on substrate doublets."""
    print("=" * 74)
    print("M1: BARGMANN IDENTITY ON SELECTED AUXILIARY DOUBLETS")
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
        d = phase_distance(ph, half)
        n_match += int(d < 1e-9)
        print(f"  {str((i, j, k)):>14} {ph:>11.6f} {half:>11.6f} {d:>9.1e}")
    print()
    print(f"  -> {n_match}/{len(triples)} selected triples match on this branch.")
    print("     This verifies the standard CP^1 identity on the extracted data.")


def experiment_2_gauge_invariance():
    """M2: the phase is geometric -- invariant under per-state rephasing."""
    print()
    print("=" * 74)
    print("M2: BARGMANN-PRODUCT REPHASING INVARIANCE")
    print("=" * 74)
    print("The Bargmann product is invariant under per-state rephasing")
    print("|psi_i> -> e^{i alpha_i}|psi_i> because the phases cancel.")
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
    print("     arbitrary representatives. This is an algebraic invariance check.")


def experiment_3_holonomy_solid_angle():
    """M3: repeat one three-state identity for four seeded snapshots."""
    print()
    print("=" * 74)
    print("M3: THREE-STATE IDENTITY ACROSS FOUR SEEDS")
    print("=" * 74)
    print("Treat three node read-outs as a closed CP^1 triangle and compare its")
    print("Bargmann phase with half the oriented solid angle.")
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
        d = phase_distance(ph, half)
        n_match += int(d < 1e-9)
        print(f"  {s:>6} {ph:>12.6f} {half:>12.6f} {d:>9.1e}")
    print()
    print(f"  -> {n_match}/4 seeded triples match on the selected phase branch.")
    print("     No time evolution or operator-generated loop was simulated.")


def main():
    print()
    print("  ===============================================================")
    print("  Bargmann Identity on Auxiliary Substrate Doublets")
    print("  The Bargmann Invariant Equals Half the Poincare Solid Angle")
    print("  ===============================================================")
    print()
    experiment_1_bargmann_identity()
    experiment_2_gauge_invariance()
    experiment_3_holonomy_solid_angle()
    print()
    print("=" * 74)
    print("SCOPED FINDINGS")
    print("=" * 74)
    print("Selected nonzero auxiliary doublets satisfy the standard Bargmann/")
    print("solid-angle CP^1 identity and its rephasing invariance. The four-seed")
    print("table repeats the same algebraic construction. It contains no engine")
    print("trajectory or parallel-transport experiment and does not derive the")
    print("auxiliary substrate from TNFR dynamics. This is classical geometric")
    print("polarization algebra and closes no open problem.")


if __name__ == "__main__":
    main()
