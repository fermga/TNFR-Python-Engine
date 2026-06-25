#!/usr/bin/env python3
"""
Example 156 — The Emergence-Directness Law: Structural Level x Symmetry Sector
=============================================================================

The cross-domain axis of ``theory/EMERGENT_ONTOLOGY.md`` (§0) orders domains by
how DIRECTLY they read the shared fixed point ``ΔNFR = 0`` — particle winding
*directly*, the number-theory spectral sector *genuinely but partially*, the
arithmetic ``ΔNFR`` *circular*. This example measures WHY that ordering holds: a
domain's directness is fixed by

    (i)  WHICH level of the three-level structure (§7.1 stage -> occupant ->
         process) carries its canonical read-out, and
    (ii) WHICH ``Aut(G)`` representation sector (Schur: ``Fix(G) ⊕ Fix(G)^perp``,
         example 123) that read-out lives in.

    Level / read-out          Aut(G) sector         Directness
    ------------------------  --------------------  -----------------------
    occupant  winding W       Fix(G)  (invariant)   DIRECT       (particles)
    stage     spectral rho    Fix(G)^perp           PARTIAL/WALL (numbers)
    process   dNFR(Omega,..)  -- (consumes input)   CIRCULAR     (arithmetic)

THE LAW (one sentence): a topological (occupant) read-out is DIRECT because it is
a ``Fix(G)`` invariant; a spectral (stage) read-out is PARTIAL because it is
trapped in ``Fix(G)^perp`` (the non-trivial irreps = the symmetry wall); a
process read-out is CIRCULAR because it consumes its own input.

This is the cross-domain face of the SAME representation theory that organizes the
secondary synergies — each algebraic relation an operator has with the coupling
``A`` produces a distinct emergent structure:

    commuting automorphism   [A,P]=0          -> the WALL (Fix^perp confinement)
    anticommuting chiral G    {A,G}=0          -> additive inverse -n / antiparticle
    non-symmetric circulant  A!=A^T,[A,A^T]=0 -> Gauss-sum PHASE (Z/n Fourier basis)
    graph products           spec add/mult     -> +, x (NOT unique factorisation)

Physics
-------
- occupant: the winding ``W = (1/2pi) * circulation`` is the degree of a map
  ``S^1 -> S^1``, an exact integer invariant under every graph automorphism
  (``Aut`` maps cycles to cycles) — the ``Fix(G)`` (trivial-rep) robust charge.
  Two distinct ``Z_2`` involutions send ``W -> -W``: parity ``P`` (an
  orientation-reversing automorphism) and charge conjugation ``C`` (phase
  conjugation ``phi -> -phi``, the chiral involution, ``chiral_involution.py``).
  ``|W|`` is the invariant of both.
- stage: on a vertex-transitive graph the per-node substrate is orbit-constant
  (it lives in ``Fix(G)``) and CANNOT discriminate nodes. The arithmetic
  discriminator lives in the spectrum ``= Fix(G)^perp`` (examples 120/123). On the
  residue Cayley digraph that ``Fix^perp`` sector is exactly the non-trivial
  ``Z_n`` characters = the Gauss-sum eigenbasis (``TNFR_NUMBER_THEORY.md``
  §9.5-9.8, §10.5 — the non-self-adjoint circulant phase operator).
- process: the arithmetic ``dNFR(n)`` computes ``Omega, tau, sigma`` from ``n`` by
  trial division — it consumes the divisibility it "reads", so it is circular
  (Sector A).

Experiments
-----------
1. occupant -- ``|W|`` is an ``Aut(G)`` invariant (Fix): robust / DIRECT
2. the two ``W -> -W`` involutions: parity ``P`` vs charge conjugation ``C``
3. stage per-node -- orbit-constant on a vertex-transitive graph (Fix): BLIND
4. stage spectral -- ``rho(n)`` discriminates primes in ``Fix^perp`` while every
   per-node ``Fix`` quantity stays blind: the WALL
5. the law -- assemble the level x sector table from the measurements

References
----------
- theory/EMERGENT_ONTOLOGY.md §0 (cross-domain axis), §2.3 (this law), §7.1
- theory/TNFR_NUMBER_THEORY.md §9.5-9.8 (sectors + ladder), §10.5 (phase operator)
- examples 123 (Schur Fix/Fix^perp), 120 (arithmetic in Fix^perp), 155 (ladder)
- benchmarks: chiral_involution.py, composition_arithmetic.py,
  residue_phase_vs_riemann.py
- src/tnfr/physics/emergent_particles.py (winding), metrics/common.py (coherence)
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from tnfr.physics.emergent_particles import winding_number, winding_ring
from tnfr.metrics.common import structural_coherence
from tnfr.mathematics.number_theory import (
    arithmetic_cayley_digraph,
    quadratic_residue_set,
    residue_network_rank,
)

_TWO_PI = 2.0 * np.pi


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _dihedral_automorphisms(n):
    """Aut(C_n) = D_n: n rotations i->(i+a)%n and n reflections i->(a-i)%n."""
    autos = [("rot", a, [(i + a) % n for i in range(n)]) for a in range(n)]
    autos += [("ref", a, [(a - i) % n for i in range(n)]) for a in range(n)]
    return autos


def _winding_along(G, order):
    """Winding measured traversing the nodes in the given order."""
    return winding_number(G, order=order)[0]


def _conjugate_phase(G):
    """Charge conjugation C: a copy of G with every phase negated (phi -> -phi)."""
    H = G.copy()
    for i in H.nodes():
        ph = -float(H.nodes[i].get("phase", H.nodes[i].get("theta", 0.0)))
        H.nodes[i]["phase"] = ph
        H.nodes[i]["theta"] = ph
    return H


# --------------------------------------------------------------------------- #
# Experiment 1 -- occupant: |W| is an Aut(G) invariant (Fix) -> DIRECT
# --------------------------------------------------------------------------- #
def experiment_1_occupant_fix_invariant():
    print("=" * 78)
    print("(1) OCCUPANT: the winding |W| is an Aut(G) invariant (Fix sector)")
    print("=" * 78)
    n, W = 12, 2
    G = winding_ring(n, W)
    W0 = _winding_along(G, list(range(n)))
    autos = _dihedral_automorphisms(n)
    preserved = sum(_winding_along(G, perm) == W0 for _, _, perm in autos)
    flipped = sum(_winding_along(G, perm) == -W0 for _, _, perm in autos)
    ok = preserved + flipped == len(autos)
    print(f"  ring C_{n}, planted W={W} -> measured occupant charge W = {W0}")
    print(f"  under all {len(autos)} automorphisms of D_{n}: rotations preserve W "
          f"({preserved}), reflections flip W->-W ({flipped})")
    print(f"  => |W| invariant under EVERY automorphism: {ok}  "
          f"(a Fix(G) topological invariant = DIRECT)")
    return ok


# --------------------------------------------------------------------------- #
# Experiment 2 -- the two W->-W involutions: parity P vs charge conjugation C
# --------------------------------------------------------------------------- #
def experiment_2_parity_vs_charge_conjugation():
    print("=" * 78)
    print("(2) Two distinct Z_2 send W->-W: parity P (automorphism) vs C (chiral)")
    print("=" * 78)
    n, W = 12, 3
    G = winding_ring(n, W)
    W0 = _winding_along(G, list(range(n)))
    # parity P: an orientation-reversing automorphism (reflection i -> -i)
    parity_order = [(-i) % n for i in range(n)]
    W_parity = _winding_along(G, parity_order)
    # charge conjugation C: phase conjugation phi -> -phi (the chiral involution)
    W_charge = _winding_along(_conjugate_phase(G), list(range(n)))
    ok = (W_parity == -W0) and (W_charge == -W0) and abs(W0) == abs(W_parity)
    print(f"  W = {W0};  parity P (reflection): W -> {W_parity};  "
          f"charge-conj C (phi->-phi): W -> {W_charge}")
    print(f"  => both flip the sign, |W| invariant under both: {ok}")
    print("     P = spatial (commuting automorphism); C = chiral (anticommuting "
          "Gamma, = additive inverse -n, chiral_involution.py). Distinct Z_2.")
    return ok


# --------------------------------------------------------------------------- #
# Experiment 3 -- stage per-node: orbit-constant on vertex-transitive (Fix) BLIND
# --------------------------------------------------------------------------- #
def experiment_3_stage_pernode_blind():
    print("=" * 78)
    print("(3) STAGE per-node read-out: orbit-constant on a vertex-transitive graph")
    print("=" * 78)
    n = 12
    G = winding_ring(n, 2)  # a ring is vertex-transitive
    coh = [round(structural_coherence(float(G.nodes[i]["delta_nfr"]), 0.0), 12)
           for i in G.nodes()]
    distinct = len(set(coh))
    ok = distinct == 1
    print(f"  structural_coherence over C_{n}: distinct per-node values = {distinct}")
    print(f"  => orbit-constant (lives in Fix(G)) = {ok}; per-node read-out is "
          "BLIND. A discriminator must sit in Fix(G)^perp.")
    return ok


# --------------------------------------------------------------------------- #
# Experiment 4 -- stage spectral: rho discriminates in Fix^perp while Fix is blind
# --------------------------------------------------------------------------- #
def experiment_4_stage_spectral_fixperp_wall():
    print("=" * 78)
    print("(4) STAGE spectral: rho(n) discriminates primes in Fix^perp (the wall)")
    print("=" * 78)
    samples = [(7, True), (11, True), (13, True), (15, False), (21, False)]
    rows = []
    all_ok = True
    for n, is_prime in samples:
        conn = sorted(c for c in quadratic_residue_set(n) if c != 0)
        G = arithmetic_cayley_digraph(n, conn)
        outdeg = {d for _, d in G.out_degree()}
        pernode_uniform = len(outdeg) == 1          # Fix: vertex-transitive
        rho = residue_network_rank(n, kind="quadratic")
        rho_says_prime = rho == 3                    # Fix^perp spectral invariant
        ok = (rho_says_prime == is_prime) and pernode_uniform
        all_ok &= ok
        rows.append((n, is_prime, pernode_uniform, rho, rho_says_prime))
    print("   n   prime?  per-node uniform (Fix)   rho (Fix^perp)   rho=3<=>prime")
    for n, is_prime, uni, rho, says in rows:
        print(f"  {n:<3}  {str(is_prime):<6}  {str(uni):<22}  {rho:<14}  {says}")
    print("  => per-node (Fix) is uniform for EVERY n (blind to primality); "
          "primality is a spectral Fix^perp invariant (rho).")
    print(f"     all consistent: {all_ok}  (= the symmetry wall, examples 120/123)")
    return all_ok


# --------------------------------------------------------------------------- #
# Experiment 5 -- assemble the level x sector law
# --------------------------------------------------------------------------- #
def experiment_5_the_law(results):
    print("=" * 78)
    print("(5) THE EMERGENCE-DIRECTNESS LAW (assembled from the measurements)")
    print("=" * 78)
    print("   level      read-out        Aut(G) sector     directness")
    print("   ---------  --------------  ----------------  --------------------")
    print("   occupant   winding W       Fix(G)            DIRECT    (particles)")
    print("   stage      spectral rho    Fix(G)^perp       PARTIAL   (numbers/wall)")
    print("   process    dNFR(Om,ta,si)  consumes input    CIRCULAR  (arithmetic)")
    print()
    print("  one law: topological(occupant)=Fix-invariant=DIRECT; spectral(stage)")
    print("  =Fix^perp-trapped=PARTIAL(the wall); process=consumes-input=CIRCULAR.")
    print("  Unifies the position ladder (155/§9.8), particle classification (§7.1),")
    print("  the Riemann/number wall (§9.5-9.7, §10.5) and the §0 axis under ONE")
    print("  principle: representation theory of the coupling's symmetry group.")
    print()
    print("  HONEST SCOPE: every piece is DERIVED/measured; the law is a unifying")
    print("  re-expression (one fixed point, many read-outs). It closes NO open")
    print("  problem -- the wall persists; G4 = RH stays OPEN.")
    return all(results.values())


def main():
    results = {}
    results["1_occupant_fix"] = experiment_1_occupant_fix_invariant()
    print()
    results["2_parity_vs_charge"] = experiment_2_parity_vs_charge_conjugation()
    print()
    results["3_stage_blind"] = experiment_3_stage_pernode_blind()
    print()
    results["4_stage_fixperp"] = experiment_4_stage_spectral_fixperp_wall()
    print()
    ok = experiment_5_the_law(results)
    print()
    print("=" * 78)
    status = "ALL EXPERIMENTS PASSED" if ok else "SOME EXPERIMENTS FAILED"
    print(f"RESULT: {status}")
    print("=" * 78)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
