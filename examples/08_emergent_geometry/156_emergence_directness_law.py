#!/usr/bin/env python3
"""Example 156 — finite comparisons between independently chosen read-outs.

This script retains the numerical experiments historically presented as an
"emergence-directness law", but the measurements do not establish such a law.
They concern three separately constructed objects:

* a phase field planted on a cycle, whose winding is measured along selected
  orientations;
* a cycle whose node pressures are assigned uniformly by its constructor;
* quadratic-residue Cayley digraphs, whose rank statistic is compared with
  externally supplied prime/composite labels on five inputs.

The cycle automorphisms, pointwise phase negation and residue graphs are chosen
before measurement. Agreement between their finite outputs does not put the
read-outs in a common representation sector, derive one from another, or rank
their degree of emergence. In particular, winding sectors are not physical
particle species, and phase negation is not identified with any physical
conjugation operation.

Experiments
-----------
1. Measure winding on all dihedral traversals of one prepared ``C_12`` field.
2. Compare traversal reflection with pointwise phase negation on that field.
3. Confirm that the constructor's uniform pressure gives uniform coherence.
4. Compare ``residue_network_rank`` with five external arithmetic labels.
5. State the common limits of those finite comparisons.

References
----------
* ``src/tnfr/physics/emergent_particles.py`` for winding measurement.
* ``src/tnfr/mathematics/number_theory.py`` for the residue-graph statistics.
* ``theory/EMERGENT_ONTOLOGY.md`` for the separation between constructed input,
  measured read-out and dynamical emergence.

Status: RESEARCH example; finite comparison and negative identification result.
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


def _negate_phase(G):
    """Return a copy of ``G`` with every phase multiplied by ``-1``."""
    H = G.copy()
    for i in H.nodes():
        ph = -float(H.nodes[i].get("phase", H.nodes[i].get("theta", 0.0)))
        H.nodes[i]["phase"] = ph
        H.nodes[i]["theta"] = ph
    return H


# --------------------------------------------------------------------------- #
# Experiment 1 -- finite dihedral-traversal comparison
# --------------------------------------------------------------------------- #
def experiment_1_dihedral_winding_comparison():
    print("=" * 78)
    print("(1) PREPARED LOOP: winding under the 24 dihedral traversals of C_12")
    print("=" * 78)
    n, W = 12, 2
    G = winding_ring(n, W)
    W0 = _winding_along(G, list(range(n)))
    autos = _dihedral_automorphisms(n)
    preserved = sum(_winding_along(G, perm) == W0 for _, _, perm in autos)
    flipped = sum(_winding_along(G, perm) == -W0 for _, _, perm in autos)
    ok = preserved + flipped == len(autos)
    print(f"  ring C_{n}, planted target W={W} -> measured W = {W0}")
    print(f"  under all {len(autos)} dihedral automorphisms of C_{n}: rotations preserve W "
          f"({preserved}), reflections flip W->-W ({flipped})")
    print(f"  => |W| is unchanged in this complete finite traversal check: {ok}")
    print("     The phase field and the group action were supplied as inputs; this")
    print("     does not show that TNFR dynamics formed the winding sector.")
    return ok


# --------------------------------------------------------------------------- #
# Experiment 2 -- two externally selected sign-reversing operations
# --------------------------------------------------------------------------- #
def experiment_2_reflection_vs_phase_negation():
    print("=" * 78)
    print("(2) Reflection of traversal vs pointwise phase negation")
    print("=" * 78)
    n, W = 12, 3
    G = winding_ring(n, W)
    W0 = _winding_along(G, list(range(n)))
    # Orientation-reversing traversal (reflection i -> -i).
    reflection_order = [(-i) % n for i in range(n)]
    W_reflection = _winding_along(G, reflection_order)
    # A separate, pointwise transformation of the supplied phase field.
    W_negated = _winding_along(_negate_phase(G), list(range(n)))
    ok = (
        (W_reflection == -W0)
        and (W_negated == -W0)
        and abs(W0) == abs(W_reflection)
    )
    print(f"  W = {W0};  reflected traversal: W -> {W_reflection};  "
          f"negated phase field: W -> {W_negated}")
    print(f"  => both flip the sign, |W| invariant under both: {ok}")
    print("     These are two externally selected involutions on different inputs.")
    print("     Their matching output does not identify the transformations or give")
    print("     the winding a physical interpretation.")
    return ok


# --------------------------------------------------------------------------- #
# Experiment 3 -- uniform constructor input gives a uniform node read-out
# --------------------------------------------------------------------------- #
def experiment_3_uniform_node_readout():
    print("=" * 78)
    print("(3) UNIFORM INPUT: per-node coherence on a prepared cycle")
    print("=" * 78)
    n = 12
    G = winding_ring(n, 2)  # a ring is vertex-transitive
    coh = [round(structural_coherence(float(G.nodes[i]["delta_nfr"]), 0.0), 12)
           for i in G.nodes()]
    distinct = len(set(coh))
    ok = distinct == 1
    print(f"  structural_coherence over C_{n}: distinct per-node values = {distinct}")
    print(f"  => uniform on this constructed field: {ok}")
    print("     The constructor assigns the same delta_nfr to every node, so this")
    print("     result does not locate a universal representation sector or constrain")
    print("     where an unrelated arithmetic diagnostic must live.")
    return ok


# --------------------------------------------------------------------------- #
# Experiment 4 -- finite residue-rank comparison with external truth labels
# --------------------------------------------------------------------------- #
def experiment_4_residue_rank_comparison():
    print("=" * 78)
    print("(4) RESIDUE GRAPHS: finite rank comparison with arithmetic labels")
    print("=" * 78)
    samples = [(7, True), (11, True), (13, True), (15, False), (21, False)]
    rows = []
    all_ok = True
    for n, is_prime in samples:
        conn = sorted(c for c in quadratic_residue_set(n) if c != 0)
        G = arithmetic_cayley_digraph(n, conn)
        outdeg = {d for _, d in G.out_degree()}
        pernode_uniform = len(outdeg) == 1
        rho = residue_network_rank(n, kind="quadratic")
        rho_says_prime = rho == 3
        ok = (rho_says_prime == is_prime) and pernode_uniform
        all_ok &= ok
        rows.append((n, is_prime, pernode_uniform, rho, rho_says_prime))
    print("   n   prime?  uniform out-degree   residue rank rho   rho==3")
    for n, is_prime, uni, rho, says in rows:
        print(f"  {n:<3}  {str(is_prime):<6}  {str(uni):<22}  {rho:<14}  {says}")
    print("  => every selected Cayley graph has uniform out-degree, while rho=3")
    print(f"     matches the five supplied labels: {all_ok}")
    print("     This finite table is not a general primality theorem and does not")
    print("     prove that one read-out is the hidden complement of another.")
    return all_ok


# --------------------------------------------------------------------------- #
# Experiment 5 -- summarize what the finite checks do and do not identify
# --------------------------------------------------------------------------- #
def experiment_5_scope_summary(results):
    print("=" * 78)
    print("(5) SCOPE: finite comparisons, with no cross-domain identification")
    print("=" * 78)
    print("   constructed input       measured output       justified conclusion")
    print("   ----------------------  --------------------  --------------------------")
    print("   phased cycle C_12       winding by traversal  finite orientation check")
    print("   uniform cycle pressure  node coherence        uniform-input response")
    print("   five residue digraphs   rank rho              finite label agreement")
    print()
    print("  The experiments use different state spaces and externally selected")
    print("  constructions. They do not derive a directness ordering, a shared")
    print("  symmetry decomposition, particle physics, or a new arithmetic proof.")
    return all(results.values())


# Historical callable names remain aliases so existing imports keep working.
# Their former labels are not interpretations of the measurements.
experiment_1_occupant_fix_invariant = experiment_1_dihedral_winding_comparison
experiment_2_parity_vs_charge_conjugation = experiment_2_reflection_vs_phase_negation
experiment_3_stage_pernode_blind = experiment_3_uniform_node_readout
experiment_4_stage_spectral_fixperp_wall = experiment_4_residue_rank_comparison
experiment_5_the_law = experiment_5_scope_summary


def main():
    results = {}
    results["1_dihedral_winding"] = experiment_1_dihedral_winding_comparison()
    print()
    results["2_reflection_vs_negation"] = experiment_2_reflection_vs_phase_negation()
    print()
    results["3_uniform_node_readout"] = experiment_3_uniform_node_readout()
    print()
    results["4_residue_rank"] = experiment_4_residue_rank_comparison()
    print()
    ok = experiment_5_scope_summary(results)
    print()
    print("=" * 78)
    status = "ALL EXPERIMENTS PASSED" if ok else "SOME EXPERIMENTS FAILED"
    print(f"RESULT: {status}")
    print("=" * 78)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
