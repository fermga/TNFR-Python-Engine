#!/usr/bin/env python3
"""
Example 101 — Numbers as a Coupled Network: Ω-Graded Centrality & the Prime Periphery
====================================================================================

A measured study of the deep idea that the natural numbers form a TNFR
complex system: the SAME factor structure that fixes a number's arithmetic
pressure ΔNFR also fixes its coupling position in the divisibility/GCD
network. The two TNFR pictures — per-node pressure (Ω, τ, σ) and
network transport (degree, resistance, diffusion) — are two faces of the
factorization, linked through the prime-factor count Ω(n).

Physics
-------
The arithmetic network (theory/TNFR_NUMBER_THEORY.md §2) couples numbers
n, m by:
  - divisibility edges (n | m), and
  - GCD coupling edges (gcd(n, m) > 1, i.e. they share a prime factor).

This is a genuine coupled TNFR network. On it we measure two things:

1. **Arithmetic pressure** ΔNFR(n) = ζ(Ω−1) + η(τ−2) + θ(σ/n−(1+1/n)),
   the per-node structural pressure (= 0 ⟺ prime, the §4 theorem).

2. **Transport position**: where n sits in the network under the
   diffusion machinery (Example 99) — degree, stationary mass
   π = deg/Σdeg, effective resistance, isolation.

The measured result (all verified below)
----------------------------------------
The prime-factor count Ω(n) is the common structural coordinate:

  • r(Ω, ΔNFR)  ≈ 0.94   — Ω drives the arithmetic pressure;
  • r(Ω, degree) ≈ 0.75  — Ω drives the network centrality;
  • r(ΔNFR, degree) ≈ 0.81 — so the two pictures are LINKED.

This produces Ω-graded shells: as Ω grows, BOTH ΔNFR and degree grow
monotonically. Primes (Ω = 1, ΔNFR = 0) sit at the **transport
periphery**:

  • mean degree ≈ 7 (vs 60–100 for composites);
  • ≈ 2.4× the effective resistance (harder to reach by random walk);
  • large primes (p > N/2) are literally ISOLATED — degree 0, zero
    structural coupling.

So ΔNFR = 0 (arithmetic inertness) and network peripherality are the SAME
structural fact: a prime couples to the network ONLY through its multiples,
so when its multiples leave the range it decouples entirely. The
zero-pressure fixed point IS the zero-coupling isolate.

Honest scope
------------
- This is a MEASURED structural correspondence, not a dynamical identity.
  The per-node arithmetic ΔNFR (a function of Ω, τ, σ) is NOT the
  graph-diffusion Laplacian −L_rw·EPI of Example 99; the two are LINKED
  through the common driver Ω (strong correlation r ≈ 0.8–0.9), not equal.
- The network is NOT scale-free: the degree has a structured ceiling at
  multiples of 2·3·5, coefficient of variation < 1 — no power-law tail.
  Do not claim "scale-free".
- "Primes are peripheral" is the transport-language RESTATEMENT of the
  classical fact "gcd(p, m) > 1 ⟺ p | m" (a prime shares a factor only
  with its multiples). It is faithful, made precise (Ω-graded centrality,
  isolation of large primes), not a new theorem.
- The theory's §7.1 reading "primes act as Φ_s sinks" is directionally
  consistent (primes carry far lower |Φ_s|), but the *dynamical*
  "attract composites" claim is NOT tested here — only the static
  potential and transport position are measured.
- Detection of primality remains the exact §4 theorem (ΔNFR = 0).

References
----------
- theory/TNFR_NUMBER_THEORY.md §2 (arithmetic network), §4 (ΔNFR=0), §7.1 (Φ_s)
- examples/99_structural_diffusion.py (diffusion / resistance machinery)
- examples/100_prime_families_orbits.py (the zero-pressure fixed-point set)
- src/tnfr/mathematics/number_theory.py (ArithmeticTNFRNetwork)
- src/tnfr/physics/structural_diffusion.py (transport quantities)
- AGENTS.md §"Transport Content of the Nodal Equation"
"""

import os
import sys
import statistics

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import networkx as nx
from sympy import isprime, factorint

from tnfr.mathematics.number_theory import ArithmeticTNFRNetwork
from tnfr.physics.structural_diffusion import (
    stationary_distribution,
    effective_resistance,
)

N = 160


def _build():
    """Build the arithmetic network and its undirected transport view."""
    net = ArithmeticTNFRNetwork(max_number=N)
    G = net.graph.to_undirected()
    nodes = sorted(G.nodes())
    props = {n: net.get_tnfr_properties(n) for n in nodes}
    omega = {n: sum(factorint(n).values()) for n in nodes}
    dnfr = {n: abs(props[n]['DELTA_NFR']) for n in nodes}
    return net, G, nodes, props, omega, dnfr


# ============================================================================
# EXPERIMENT 1: The arithmetic network is a real coupled system (not scale-free)
# ============================================================================
def experiment_1_network(G, nodes):
    """Structure of the divisibility/GCD network; honest non-scale-free."""
    print("=" * 72)
    print("EXPERIMENT 1: The Arithmetic Network (divisibility + GCD coupling)")
    print("=" * 72)
    print()
    print("Nodes = 2..N; edges couple n,m by divisibility (n|m) or shared")
    print("prime factor (gcd>1). A genuine coupled TNFR network.")
    print()

    deg = dict(G.degree())
    degs = np.array([deg[n] for n in nodes], float)
    cv = degs.std() / degs.mean()
    top = sorted(nodes, key=lambda n: -deg[n])[:5]
    print(f"  nodes = {len(nodes)}, edges = {G.number_of_edges()}, "
          f"density = {nx.density(G):.3f}")
    print(f"  degree: mean {degs.mean():.1f}, max {int(degs.max())}, "
          f"CV = {cv:.2f}  ->  CV < 1: NOT scale-free (degree ceiling)")
    print(f"  top hubs: {[(n, deg[n]) for n in top]}")
    print("    (all multiples of 2·3·5 = 30 -> the small primes are the")
    print("     coupling backbone, not a power-law tail)")
    print()


# ============================================================================
# EXPERIMENT 2: Ω(n) is the common structural coordinate (correlation triangle)
# ============================================================================
def experiment_2_correlation_triangle(G, nodes, omega, dnfr):
    """ΔNFR (arithmetic) and degree (transport) are linked through Ω."""
    print("=" * 72)
    print("EXPERIMENT 2: Ω(n) Links the Pressure and Transport Pictures")
    print("=" * 72)
    print()
    print("Ω(n) = prime-factor count (with multiplicity). Test whether it")
    print("drives BOTH the arithmetic pressure ΔNFR and the network degree.")
    print()

    deg = dict(G.degree())
    om = np.array([omega[n] for n in nodes], float)
    dg = np.array([deg[n] for n in nodes], float)
    dn = np.array([dnfr[n] for n in nodes], float)
    print(f"  r(Ω,    ΔNFR)   = {np.corrcoef(om, dn)[0, 1]:.3f}   "
          f"(Ω drives arithmetic pressure)")
    print(f"  r(Ω,    degree) = {np.corrcoef(om, dg)[0, 1]:.3f}   "
          f"(Ω drives network centrality)")
    print(f"  r(ΔNFR, degree) = {np.corrcoef(dn, dg)[0, 1]:.3f}   "
          f"(so the two pictures are LINKED)")
    print()
    print("VERDICT: the factorization (via Ω) is the common coordinate. A")
    print("number's arithmetic pressure and its coupling position are two")
    print("faces of the same factor structure — LINKED, not identical.")
    print()


# ============================================================================
# EXPERIMENT 3: Ω-graded shells — pressure and centrality rise together
# ============================================================================
def experiment_3_shells(G, nodes, omega, dnfr):
    """Bin by Ω: mean degree and mean |ΔNFR| both rise monotonically."""
    print("=" * 72)
    print("EXPERIMENT 3: Ω-Graded Shells (pressure and centrality together)")
    print("=" * 72)
    print()

    deg = dict(G.degree())
    print(f"  {'Ω':>3} {'count':>6} {'mean degree':>12} {'mean |ΔNFR|':>13}")
    print("  " + "-" * 38)
    for k in range(1, max(omega.values()) + 1):
        grp = [n for n in nodes if omega[n] == k]
        if not grp:
            continue
        md = statistics.mean(deg[n] for n in grp)
        mp = statistics.mean(dnfr[n] for n in grp)
        tag = "  <- primes (ΔNFR=0)" if k == 1 else ""
        print(f"  {k:>3} {len(grp):>6} {md:>12.1f} {mp:>13.3f}{tag}")
    print()
    print("Both columns rise with Ω (the single Ω=7 dip is 128 = 2^7, a")
    print("prime POWER: one distinct prime, so it couples only to powers of")
    print("2 — distinct-prime count ω matters for degree, not just Ω).")
    print("VERDICT: the SAME factor ladder orders pressure and transport.")
    print()


# ============================================================================
# EXPERIMENT 4: Primes are the transport periphery
# ============================================================================
def experiment_4_prime_periphery(G, nodes):
    """Primes: low stationary mass, high resistance, large ones isolated."""
    print("=" * 72)
    print("EXPERIMENT 4: Primes Are the Transport Periphery")
    print("=" * 72)
    print()
    print("ΔNFR = 0 means zero arithmetic pressure. Measure the NETWORK")
    print("counterpart: where do primes sit under diffusion transport?")
    print()

    deg = dict(G.degree())
    primes = [n for n in nodes if isprime(n)]
    comps = [n for n in nodes if not isprime(n)]

    # stationary mass π = deg / Σ deg
    sd_nodes, sd_pi = stationary_distribution(G)
    pi = {n: sd_pi[i] for i, n in enumerate(sd_nodes)}
    pp = statistics.mean(pi[p] for p in primes)
    pc = statistics.mean(pi[c] for c in comps)
    print("  diffusion stationary mass π = deg/Σdeg:")
    print(f"    mean π(prime)     = {pp:.5f}")
    print(f"    mean π(composite) = {pc:.5f}   -> primes {pp / pc:.2f}× "
          f"(periphery)")

    # effective resistance on the giant component (isolated primes excluded)
    giant = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    gnodes, R = effective_resistance(giant)
    idx = {n: i for i, n in enumerate(gnodes)}
    periph = {n: R[idx[n]].mean() for n in gnodes}
    gp = [n for n in gnodes if isprime(n)]
    gc = [n for n in gnodes if not isprime(n)]
    rp = statistics.mean(periph[n] for n in gp)
    rc = statistics.mean(periph[n] for n in gc)
    print("  mean effective resistance to the rest of the network:")
    print(f"    prime     = {rp:.4f}")
    print(f"    composite = {rc:.4f}   -> primes {rp / rc:.2f}× "
          f"(harder to reach)")

    # isolation of large primes
    iso = [n for n in nodes if deg[n] == 0]
    print(f"  fully ISOLATED nodes (degree 0): {len(iso)}")
    print(f"    = primes p > N/2 (no multiples left in range): {iso[:6]}...")
    print()
    print("VERDICT: ΔNFR = 0 (arithmetic inertness) and network")
    print("peripherality/isolation are the SAME fact — a prime couples only")
    print("through its multiples; when they leave the range it decouples.")
    print()


# ============================================================================
# EXPERIMENT 5: Synthesis
# ============================================================================
def experiment_5_synthesis():
    """The factorization is the hidden structural coordinate."""
    print("=" * 72)
    print("EXPERIMENT 5: Synthesis — Factorization as the Hidden Coordinate")
    print("=" * 72)
    print()
    print("  Picture A (pressure):   ΔNFR(n) from (Ω, τ, σ), = 0 ⟺ prime")
    print("  Picture B (transport):  position of n in the divisibility/GCD")
    print("                          network (degree, resistance, diffusion)")
    print()
    print("  Both are governed by the FACTORIZATION of n, graded by Ω:")
    print("    • composites (large Ω) = high pressure AND central hubs;")
    print("    • primes (Ω=1, ΔNFR=0) = zero pressure AND peripheral/isolated.")
    print()
    print("So numbers DO reflect TNFR structure deeply — but the honest form")
    print("is a CORRESPONDENCE through Ω, not a dynamical identity, and it")
    print("INVERTS the naive picture: primes are the inert, decoupled")
    print("periphery, not central attractors.")
    print()


def main():
    print()
    print("  TNFR Example 101: Numbers as a Coupled Network")
    print("  Ω-graded centrality and the prime periphery")
    print("  ============================================")
    print()
    _net, G, nodes, _props, omega, dnfr = _build()
    experiment_1_network(G, nodes)
    experiment_2_correlation_triangle(G, nodes, omega, dnfr)
    experiment_3_shells(G, nodes, omega, dnfr)
    experiment_4_prime_periphery(G, nodes)
    experiment_5_synthesis()
    print("=" * 72)
    print("WHAT THIS ESTABLISHES")
    print("=" * 72)
    print()
    print("On the arithmetic divisibility/GCD network, the prime-factor")
    print("count Ω(n) is the common structural coordinate that grades BOTH")
    print("the per-node arithmetic pressure ΔNFR and the network-transport")
    print("centrality (degree, resistance, diffusion). Primes (Ω=1, ΔNFR=0)")
    print("are the zero-pressure, zero-coupling periphery — large primes are")
    print("literally isolated. The arithmetic and transport pictures are")
    print("two faces of the factorization, LINKED through Ω (r ≈ 0.8–0.9),")
    print("not identical. The network is not scale-free; the result is an")
    print("honest structural correspondence, made precise, that inverts the")
    print("naive 'primes as central attractors' into 'primes as inert")
    print("isolates' — faithful to ΔNFR = 0.")
    print()


if __name__ == "__main__":
    main()
