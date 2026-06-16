#!/usr/bin/env python3
"""
Example 121 — Can a Canonical Symmetry-Break Make the Substrate See Arithmetic?
(The B2-P2 Lever, Measured at the Number-Theory Level — a Clean Negative)
==============================================================================

Example 120 located the wall: vertex-transitivity confines the residue
digraph's arithmetic to the GLOBAL spectrum (Fix(G_aut)^perp), leaving the
per-node symplectic substrate in the symmetric sector Fix(G_aut), blind. The
natural next question — the one the TNFR-Riemann program calls candidate
**B2-P2 (NodeIndexedCouplingWeights)** — is whether a CANONICAL per-node lever
could break that symmetry and let the substrate see the arithmetic.

The analytical answer is already on record: AGENTS.md "B0★-β-P2 FAILS"
(§13sexagesima-sexta) closes P2 at the slot level — the nodal equation
dEPI/dt = nu_f * dNFR has NO per-node-weight slot; weights enter only as
graph-level CHANNEL scalars (DNFR_WEIGHTS = {phase, epi, vf, topo}), so any
per-node law needs an external rule-selection axiom not derivable from the
catalog. This example **measures** that closure at the number-theory level.

The code fact (the missing slot)
--------------------------------
`tnfr.dynamics.dnfr._configure_dnfr_weights` produces ONE graph-level dict of
channel weights, normalized once and reused for every node. There is no
per-node weight in the canonical machinery. The only per-node levers are (a)
the initial seed and (b) the per-node nu_f. We test both.

Three measured results (the three levers)
-----------------------------------------
D1 SYMMETRIC SEED -> ZERO per-node structure. With an identical initial state
   on every node, the canonical dynamics produces dNFR = 0 exactly (uniform
   field = the diffusive steady state), so the substrate stays uniform
   (sigma ~ 1e-32) for prime AND composite n. The dynamics ALONE creates no
   per-node structure: all the variation in example 120 came from the
   (arithmetic-neutral) random seed.

D2 STRUCTURE-DERIVED nu_f IS CONSTANT. On the vertex-transitive residue
   digraph every per-node structural invariant (in-degree, out-degree, local
   triangle count) is EXACTLY constant (sigma = 0). So any canonical nu_f
   derived from graph structure is uniform across nodes -> no symmetry break.

D3 ARITHMETIC-INJECTED nu_f IS CIRCULAR ECHO. The only way to make nu_f
   per-node non-uniform is to inject an external arithmetic label (e.g.
   nu_f = 1 + [node is a QR mod n]). The substrate then varies — but a
   SHUFFLED control with the same nu_f multiset (QR labels destroyed) gives a
   STATISTICALLY IDENTICAL dispersion (ratio ~ 1). The substrate echoes the
   injected nu_f multiset, not the arithmetic structure. This is the example
   116 echo mechanism: injection, not emergence — and circular.

Conclusion
----------
There is NO canonical (non-circular) per-node lever that breaks vertex-
transitivity: the nodal equation has no per-node weight slot (code fact);
structure-derived levers are uniform (D2); the symmetric dynamics makes no
structure (D1); and the only lever that does break uniformity is an external
arithmetic injection that the shuffled control reveals as echo (D3). This is
the empirical, number-theory-level confirmation of the analytical B2-P2
closure (AGENTS.md §13sexagesima-sexta). The wall of example 120 is structural,
not an artefact of which canonical knob we turned.

Honest scope
------------
A clean MEASURED NEGATIVE: it confirms the analytical closure, it does NOT
break the wall, and it closes no open problem. It is a re-expression of
vertex-transitivity (no canonical per-node observable on a homogeneous graph)
plus the absence of a per-node weight slot in the nodal equation — both known
structural facts, measured here in TNFR's own substrate.

Doctrine compliance
-------------------
Everything uses the canonical emergent substrate: per-node fields from
`extract_phase_space_point`, the dynamics from the canonical nodal equation
dEPI/dt = nu_f * dNFR (`default_compute_delta_nfr`). Only arithmetic input:
x^2 mod n.

References
----------
- src/tnfr/dynamics/dnfr.py (_configure_dnfr_weights = graph-level channels)
- src/tnfr/physics/symplectic_substrate.py (extract_phase_space_point)
- examples/08_emergent_geometry/120_symmetry_wall_substrate_vs_spectrum.py (the wall)
- examples/07_number_theory/116_nuf_emergent_prime_visibility.py (the echo mechanism)
- theory/TNFR_NUMBER_THEORY.md §9.8 (this example; the B2-P2 measured negative)
- AGENTS.md "B0★-β-P2 FAILS" (§13sexagesima-sexta, the analytical closure)
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import numpy as np
import networkx as nx
from sympy import isprime

from tnfr.alias import get_attr, set_attr
from tnfr.constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_VF
from tnfr.dynamics import default_compute_delta_nfr
from tnfr.physics.symplectic_substrate import extract_phase_space_point


def _qr(n):
    return {(x * x) % n for x in range(1, n)} - {0}


def residue_digraph(n):
    """Directed residue Cayley graph: edge i->j iff (j-i) mod n is a QR."""
    R = _qr(n)
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(n):
            if i != j and ((j - i) % n) in R:
                G.add_edge(i, j)
    return G


def _seed_symmetric(G):
    """Identical state on every node -- the symmetric Fix(G_aut) seed."""
    for nd in G.nodes():
        G.nodes[nd]["theta"] = 0.3
        set_attr(G.nodes[nd], ALIAS_EPI, 0.2)
        set_attr(G.nodes[nd], ALIAS_VF, 1.0)
    default_compute_delta_nfr(G)


def _seed_random(G, seed):
    """Arithmetic-neutral random initial state (the only neutral lever)."""
    rng = np.random.default_rng(seed)
    for nd in G.nodes():
        G.nodes[nd]["theta"] = float(rng.uniform(0, 2 * np.pi))
        set_attr(G.nodes[nd], ALIAS_EPI, float(rng.uniform(-0.35, 0.35)))
        set_attr(G.nodes[nd], ALIAS_VF, 1.0)
    default_compute_delta_nfr(G)


def _set_vf(G, vf_array):
    for nd, v in zip(G.nodes(), vf_array):
        set_attr(G.nodes[nd], ALIAS_VF, float(v))


def _evolve(G, steps=16, dt=0.05):
    """Canonical nodal equation EPI <- EPI + dt * nu_f * dNFR (mid-transient)."""
    for _ in range(steps):
        default_compute_delta_nfr(G)
        for nd in G.nodes():
            epi = float(get_attr(G.nodes[nd], ALIAS_EPI, 0.0))
            vf = float(get_attr(G.nodes[nd], ALIAS_VF, 0.0))
            dnfr = float(get_attr(G.nodes[nd], ALIAS_DNFR, 0.0))
            set_attr(G.nodes[nd], ALIAS_EPI, epi + dt * vf * dnfr)


def _substrate_std(G):
    p = extract_phase_space_point(G)
    return float(np.std(p.phi_s)), float(np.std(p.k_phi))


def experiment_1_symmetric_seed():
    """D1: symmetric seed -> substrate uniform (no per-node structure)."""
    print("=" * 74)
    print("EXPERIMENT 1: Symmetric Seed -> Zero Per-Node Structure")
    print("=" * 74)
    print("Identical state on every node. The canonical dynamics gives dNFR=0")
    print("(uniform field = diffusive steady state), so the substrate stays")
    print("uniform for prime AND composite n. The dynamics alone makes no")
    print("per-node structure -- ex 120's variation came from the random seed.")
    print()
    print(f"  {'n':>4} {'prime':>6} {'phi_s_std':>12} {'k_phi_std':>12}")
    for n in [11, 13, 15, 19, 21, 23, 25]:
        G = residue_digraph(n)
        _seed_symmetric(G)
        _evolve(G)
        sp, sk = _substrate_std(G)
        print(f"  {n:>4} {str(isprime(n)):>6} {sp:>12.2e} {sk:>12.2e}")
    print()
    print("  -> sigma ~ 1e-32 (machine zero), prime and composite alike.")


def experiment_2_structure_constant():
    """D2: per-node structural invariants are constant (vertex-transitive)."""
    print()
    print("=" * 74)
    print("EXPERIMENT 2: Structure-Derived nu_f Is Constant (Vertex-Transitive)")
    print("=" * 74)
    print("On the vertex-transitive residue digraph every per-node structural")
    print("invariant is EXACTLY constant, so any canonical nu_f derived from")
    print("graph structure is uniform across nodes -> no symmetry break.")
    print()
    print(f"  {'n':>4} {'in-deg std':>12} {'out-deg std':>12} {'tri std':>10}")
    for n in [11, 19, 23, 31, 43]:
        G = residue_digraph(n)
        indeg = [d for _, d in G.in_degree()]
        outdeg = [d for _, d in G.out_degree()]
        tri = list(nx.triangles(G.to_undirected()).values())
        print(f"  {n:>4} {np.std(indeg):>12.2e} {np.std(outdeg):>12.2e} "
              f"{np.std(tri):>10.2e}")
    print()
    print("  -> all sigma = 0: every node is structurally equivalent.")


def experiment_3_injection_echo():
    """D3: arithmetic-injected nu_f is circular echo (shuffled control)."""
    print()
    print("=" * 74)
    print("EXPERIMENT 3: Arithmetic-Injected nu_f Is Circular Echo")
    print("=" * 74)
    print("The only per-node lever that breaks uniformity is an EXTERNAL")
    print("arithmetic label: nu_f = 1.0 + 0.8*[node index is a QR mod n].")
    print("A shuffled control keeps the same nu_f multiset but destroys the QR")
    print("labels. If the substrate dispersion is unchanged -> it echoes the")
    print("injected multiset, not the arithmetic (ex 116 mechanism), circular.")
    print()
    print(f"  {'n':>4} {'arith phiStd':>13} {'shuffled phiStd':>16} "
          f"{'ratio':>7}")
    for n in [11, 19, 23, 31, 43]:
        R = _qr(n)
        G = residue_digraph(n)
        _seed_random(G, 0)
        vf_arith = np.array([1.8 if i in R else 1.0 for i in range(n)])
        _set_vf(G, vf_arith)
        _evolve(G)
        sp_arith, _ = _substrate_std(G)

        Gs = residue_digraph(n)
        _seed_random(Gs, 0)
        rng = np.random.default_rng(7)
        vf_shuf = vf_arith.copy()
        rng.shuffle(vf_shuf)
        _set_vf(Gs, vf_shuf)
        _evolve(Gs)
        sp_shuf, _ = _substrate_std(Gs)

        print(f"  {n:>4} {sp_arith:>13.4f} {sp_shuf:>16.4f} "
              f"{sp_arith / max(sp_shuf, 1e-12):>7.3f}")
    print()
    print("  -> ratio ~ 1: the substrate echoes the injected nu_f multiset,")
    print("     not the QR structure. Injection, not emergence -- and circular.")


def main():
    print()
    print("  TNFR Example 121: Can a Canonical Symmetry-Break Make the")
    print("  Substrate See Arithmetic? (B2-P2 lever, a clean measured negative)")
    print("  =================================================================")
    print()
    experiment_1_symmetric_seed()
    experiment_2_structure_constant()
    experiment_3_injection_echo()
    print()
    print("=" * 74)
    print("WHAT THIS ESTABLISHES")
    print("=" * 74)
    print("There is NO canonical (non-circular) per-node lever that breaks")
    print("vertex-transitivity. The nodal equation has no per-node weight slot")
    print("(weights are graph-level CHANNEL scalars); structure-derived levers")
    print("are uniform on the homogeneous graph (D2); the symmetric dynamics")
    print("makes zero per-node structure (D1); and the only lever that does")
    print("break uniformity is an external arithmetic injection that the")
    print("shuffled control reveals as echo (D3). This is the empirical,")
    print("number-theory-level confirmation of the analytical B2-P2 closure")
    print("(AGENTS.md B0*-beta-P2 FAILS, section 13sexagesima-sexta). The wall")
    print("of example 120 is STRUCTURAL, not an artefact of the chosen knob.")
    print("A clean measured NEGATIVE: it confirms the closure, does not break")
    print("the wall, and closes no open problem.")


if __name__ == "__main__":
    main()
