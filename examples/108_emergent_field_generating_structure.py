#!/usr/bin/env python3
"""
Example 108 — The Generating Structure of the Emergent Fields
=============================================================

A hidden algebraic dependence in the TNFR ontology, measured exactly.

The unified-field layer exposes six "emergent fields" as if they were
independent diagnostics:

    Psi  = K_phi + i*J_phi        (complex geometric field)
    chi  = |grad phi|*K_phi - J_phi*J_dnfr        (chirality)
    Q    = |grad phi|*J_phi - K_phi*J_dnfr        (topological charge)
    S    = (|grad phi|^2 - K_phi^2) + (J_phi^2 - J_dnfr^2)  (symmetry breaking)
    E    = Phi_s^2 + |grad phi|^2 + K_phi^2 + J_phi^2 + J_dnfr^2  (energy density)
    C    = Phi_s * |Psi|          (coherence coupling)

They are NOT independent. Introducing the second natural complex field

    Omega = |grad phi| + i*J_dnfr   (the "gradient-flux" sector)

— which already exists in the codebase, buried inside
``gauge.compute_topological_norm`` as the gauge singlet that makes
|T|^2 = |Psi|^2 * |Omega|^2 manifestly invariant — every emergent field
collapses to the scalar Phi_s and the two complex fields Psi, Omega:

    E   = Phi_s^2 + |Psi|^2 + |Omega|^2           (a norm)
    C   = Phi_s * |Psi|
    S   = Re(Omega^2 - Psi^2)                      (difference of squares)
    chi = Re(Psi * Omega)                          (product, real part)
    Q   = Im(Psi * conj(Omega))                    (product, imaginary part)

So the six emergent fields carry NO information beyond the five base
fields, repackaged as Phi_s (scalar) + Psi (geometric) + Omega
(gradient-flux). The two complex sectors are the whole story; the rest
are their bilinear contractions.

Geometric reading
-----------------
With the geometric 2-vector psi = (K_phi, J_phi) and the gradient-flux
2-vector omega = (|grad phi|, J_dnfr):

    Q   = Im(Psi * conj(Omega)) = |grad phi|*J_phi - K_phi*J_dnfr
        = omega x psi    (the 2D oriented area / cross product)
    Q~  = Re(Psi * conj(Omega)) = K_phi*|grad phi| + J_phi*J_dnfr
        = psi . omega    (the dot product)

so Q^2 + Q~^2 = |psi|^2 |omega|^2 (Lagrange's identity), which is exactly
the gauge-invariant |T|^2 of gauge.compute_topological_norm. The
topological charge Q is the ORIENTED AREA spanned by the geometric and
gradient-flux sectors — this is *why* it is a topological invariant
(area is conserved under continuous, area-preserving deformation) and
why a gauge rotation Psi -> e^{i a} Psi merely rotates the pair
(Q, Q~) without changing |T|.

A clean factorisation falls out (the two "sector imbalances"):

    P = |grad phi|^2 - J_dnfr^2 = Re(Omega^2)
    R = K_phi^2     - J_phi^2   = Re(Psi^2)
    chi^2 - Q^2 = P * R         S = P - R

Honest scope
------------
- Every identity below holds to MACHINE PRECISION (chi, Q exact to the
  bit; E, S to ~1e-15). The arithmetic is elementary (complex products,
  difference of squares); the result is a CHARACTERISATION, not new
  physics.
- Omega is NOT a new field: it is the gradient-flux sector already named
  in gauge.py. What was unnoticed is that (Phi_s, Psi, Omega) GENERATE
  all six emergent fields — the pieces were scattered (Psi in unified.py
  and gauge.py, Omega only inside compute_topological_norm) and never
  connected into one generating structure.
- Omega pairs |grad phi| (1st-order) with J_dnfr (flux). It is NOT a
  symplectic conjugate pair (the substrate pairs are (K_phi, J_phi) and
  (Phi_s, J_dnfr)); it is the grouping under which the emergent-field
  definitions factorise. This is a statement about how the emergent
  fields are built, not a new dynamical conjugate pair.
- Net: the emergent-field "basis" is redundant. 6 emergent fields = 5
  reals = Phi_s + Psi + Omega.

References
----------
- src/tnfr/physics/unified.py (the six emergent fields)
- src/tnfr/physics/gauge.py::compute_topological_norm (Omega, |T|^2)
- AGENTS.md section "Mathematical Unification Discoveries"
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import random
import networkx as nx

from tnfr.physics.canonical import (
    compute_phase_gradient,
    compute_phase_curvature,
)
from tnfr.physics.extended import compute_phase_current
from tnfr.physics.unified import (
    compute_chirality_field,
    compute_topological_charge,
    compute_symmetry_breaking_field,
    compute_dnfr_flux,
    compute_energy_density,
    compute_coherence_coupling_field,
)
from tnfr.physics.fields import compute_structural_potential
from tnfr.physics.gauge import compute_topological_norm


def build_graph(seed, n):
    """A fresh randomized graph (fresh topology per call avoids the
    phase-curvature cache collision documented in the field layer)."""
    G = nx.watts_strogatz_graph(n, 4, 0.3, seed=seed)
    rng = random.Random(seed)
    for nd in G.nodes():
        G.nodes[nd]["EPI"] = rng.uniform(-0.5, 0.5)
        G.nodes[nd]["theta"] = rng.uniform(0, 6.283185)
        G.nodes[nd]["nu_f"] = rng.uniform(0.5, 1.5)
        G.nodes[nd]["vf"] = G.nodes[nd]["nu_f"]
    return G


# ============================================================================
# EXPERIMENT 1: the six emergent fields collapse to (Phi_s, Psi, Omega)
# ============================================================================
def experiment_1_generating_structure():
    print("=" * 72)
    print("EXPERIMENT 1: Six Emergent Fields = (Phi_s, Psi, Omega)")
    print("=" * 72)
    print()
    print("Psi   = K_phi + i*J_phi       (geometric sector)")
    print("Omega = |grad phi| + i*J_dnfr (gradient-flux sector, gauge.py)")
    print()
    print("Claimed (machine precision):")
    print("  E = Phi_s^2 + |Psi|^2 + |Omega|^2     C = Phi_s*|Psi|")
    print("  S = Re(Omega^2 - Psi^2)")
    print("  chi = Re(Psi*Omega)   Q = Im(Psi*conj(Omega))")
    print()
    print(f"  {'graph':>14} {'E':>9} {'C':>9} {'S':>9} {'chi':>9} {'Q':>9}")

    worst = {k: 0.0 for k in ("E", "C", "S", "chi", "Q")}
    for seed in range(6):
        n = 22 + seed
        G = build_graph(seed, n)
        grad = compute_phase_gradient(G)
        kphi = compute_phase_curvature(G)
        jphi = compute_phase_current(G)
        jdnfr = compute_dnfr_flux(G)
        phis = compute_structural_potential(G)
        chi = compute_chirality_field(G)
        Q = compute_topological_charge(G)
        S = compute_symmetry_breaking_field(G)
        E = compute_energy_density(G)
        Cc = compute_coherence_coupling_field(G)

        r = {k: 0.0 for k in worst}
        for nd in G.nodes():
            Psi = complex(kphi[nd], jphi[nd])
            Om = complex(grad[nd], jdnfr[nd])
            ps = phis[nd]
            r["E"] = max(r["E"], abs(E[nd] - (ps * ps + abs(Psi) ** 2 + abs(Om) ** 2)))
            r["C"] = max(r["C"], abs(Cc[nd] - ps * abs(Psi)))
            r["S"] = max(r["S"], abs(S[nd] - (Om * Om - Psi * Psi).real))
            r["chi"] = max(r["chi"], abs(chi[nd] - (Psi * Om).real))
            r["Q"] = max(r["Q"], abs(Q[nd] - (Psi * Om.conjugate()).imag))
        for k in worst:
            worst[k] = max(worst[k], r[k])
        print(f"  ws(n={n},s={seed})  {r['E']:9.1e} {r['C']:9.1e} "
              f"{r['S']:9.1e} {r['chi']:9.1e} {r['Q']:9.1e}")

    print()
    print(f"  worst residual over all graphs: "
          f"E={worst['E']:.1e} C={worst['C']:.1e} S={worst['S']:.1e} "
          f"chi={worst['chi']:.1e} Q={worst['Q']:.1e}")
    print()
    print("VERDICT: the six emergent fields carry NO information beyond the")
    print("scalar Phi_s and the two complex fields Psi and Omega. The")
    print("emergent-field basis is redundant: 6 fields = 5 reals.")
    print()


# ============================================================================
# EXPERIMENT 2: Q is the oriented area between the two sectors
# ============================================================================
def experiment_2_oriented_area():
    print("=" * 72)
    print("EXPERIMENT 2: Topological Charge = Oriented Area (cross product)")
    print("=" * 72)
    print()
    print("psi = (K_phi, J_phi)   omega = (|grad phi|, J_dnfr)")
    print("  Q  = Im(Psi*conj(Omega)) = omega x psi   (oriented area)")
    print("  Q~ = Re(Psi*conj(Omega)) = psi . omega    (dot product)")
    print("  => Q^2 + Q~^2 = |psi|^2 |omega|^2 = |T|^2 (gauge invariant)")
    print()
    print(f"  {'graph':>14} {'max|Q-cross|':>13} {'max|Q^2+Q~^2 - |T|^2|':>22}")

    for seed in range(5):
        n = 20 + seed
        G = build_graph(seed, n)
        grad = compute_phase_gradient(G)
        kphi = compute_phase_curvature(G)
        jphi = compute_phase_current(G)
        jdnfr = compute_dnfr_flux(G)
        Q = compute_topological_charge(G)
        Tnorm = compute_topological_norm(G)

        e_cross = e_lagrange = 0.0
        for nd in G.nodes():
            psi = (kphi[nd], jphi[nd])
            om = (grad[nd], jdnfr[nd])
            cross = om[0] * psi[1] - om[1] * psi[0]       # omega x psi
            dot = psi[0] * om[0] + psi[1] * om[1]         # psi . omega
            e_cross = max(e_cross, abs(Q[nd] - cross))
            lagr = cross * cross + dot * dot
            tnorm = (psi[0] ** 2 + psi[1] ** 2) * (om[0] ** 2 + om[1] ** 2)
            e_lagrange = max(e_lagrange, abs(lagr - tnorm))
            # cross-check against gauge.py's |T|^2
            e_lagrange = max(e_lagrange, abs(Tnorm[nd] - tnorm))
        print(f"  ws(n={n},s={seed})  {e_cross:13.1e} {e_lagrange:22.1e}")

    print()
    print("VERDICT: Q is the ORIENTED AREA spanned by the geometric and")
    print("gradient-flux sectors. This is *why* it is a topological")
    print("invariant (area is preserved under continuous area-preserving")
    print("deformation) and why a gauge rotation Psi -> e^{i a} Psi only")
    print("rotates (Q, Q~) without changing |T| = |Psi||Omega|.")
    print()


def main():
    print()
    print("  TNFR Example 108: The Generating Structure of the Emergent Fields")
    print("  Six emergent fields collapse to (Phi_s, Psi, Omega)")
    print("  ================================================================")
    print()
    experiment_1_generating_structure()
    experiment_2_oriented_area()
    print("=" * 72)
    print("WHAT THIS ESTABLISHES")
    print("=" * 72)
    print()
    print("A hidden algebraic dependence in the emergent-field ontology,")
    print("measured to machine precision. The six emergent fields (Psi, chi,")
    print("S, C, E, Q) are not independent: with the gradient-flux complex")
    print("Omega = |grad phi| + i*J_dnfr (already present in gauge.py), all")
    print("of them are generated by the scalar Phi_s and the two complex")
    print("fields Psi and Omega -- E is their norm, chi and Q are the real")
    print("and imaginary parts of the sector product, S is their difference")
    print("of squares, and Q is the oriented area between the sectors. The")
    print("emergent-field basis is redundant; the two complex sectors plus")
    print("Phi_s are the whole content. Characterization, not new physics.")
    print()


if __name__ == "__main__":
    main()
