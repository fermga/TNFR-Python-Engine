#!/usr/bin/env python3
"""
Example 103 — Auxiliary Substrate Read-outs on the Prime-Ladder Graph
=====================================================================

This finite characterization compares two snapshots of the prime-ladder graph
with the auxiliary symplectic-substrate read-out.  The substrate is an ambient
harmonic model initialized from extracted graph fields; it is not derived from
the nodal dynamics, and this example does not evolve its harmonic flow.

The default P14 fixture stores ``nu_f = k*log(p)`` while phase and pressure are
zero.  The extracted phase/pressure coordinates consequently vanish at that
snapshot.  The second snapshot deliberately encodes the same frequency data as
``phase = (tau*nu_f) mod 2*pi``.  This is a constructed embedding, not a consequence of
the nodal equation, whose canonical evolution law concerns EPI rather than
phase.  A correlation after this embedding therefore checks that the extractor
retains deliberately supplied data; it is not evidence that the engine creates
the embedding.

The final spacing calculation is an illustrative small-sample diagnostic of
the bare frequencies and per-node auxiliary actions.  Because the latter are a
deterministic function of the constructed snapshot, they add no information.
The calculation neither supplies the conjectured T-HP map nor advances the
Riemann hypothesis.  Branch and REMESH-infinity claims remain in their dedicated
research notes and are not established here.

References
----------
- examples/08_emergent_geometry/98_emergent_symplectic_substrate.py
- src/tnfr/riemann/prime_ladder_hamiltonian.py
- src/tnfr/physics/symplectic_substrate.py
- theory/TNFR_RIEMANN_RESEARCH_NOTES.md
"""

import math
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import numpy as np

from tnfr.physics.symplectic_substrate import extract_phase_space_point
from tnfr.riemann.prime_ladder_hamiltonian import build_prime_ladder_graph

N_PRIMES = 10
K = 4


def _ks_vs_gue(spectrum):
    """Indicative KS distance of unit-mean spacings to the GUE surmise."""
    s = np.sort(np.asarray(spectrum, dtype=float))
    s = s[np.isfinite(s)]
    sp = np.diff(s)
    sp = sp[sp > 1e-12]
    if len(sp) < 3:
        return float("nan")
    sp = sp / sp.mean()
    xs = np.sort(sp)
    emp = np.arange(1, len(xs) + 1) / len(xs)
    # Integral of (32/pi^2) s^2 exp(-4 s^2/pi), the GUE Wigner surmise.
    gue = np.array(
        [
            math.erf(2.0 * x / math.sqrt(math.pi))
            - (4.0 * x / math.pi) * math.exp(-4.0 * x * x / math.pi)
            for x in xs
        ]
    )
    return float(np.max(np.abs(emp - gue)))


# ============================================================================
# EXPERIMENT 1: default-snapshot auxiliary-coordinate check
# ============================================================================
def experiment_1_static_blindness(G):
    """Read the auxiliary coordinates of the default P14 snapshot."""
    print("=" * 72)
    print("EXPERIMENT 1: Default-Snapshot Auxiliary Coordinates")
    print("=" * 72)
    print()
    print("P14 stores the displayed frequencies ν_f = k·log p, with phase = 0 and")
    print("ΔNFR = 0. The substrate reads the tetrad (K_φ, J_φ, Φ_s, J_ΔNFR)")
    print("from θ and ΔNFR — never from ν_f. So on the static state:")
    print()

    pt = extract_phase_space_point(G)
    psi = np.abs(pt.k_phi + 1j * pt.j_phi)
    print(f"  |Ψ| = |K_φ + i·J_φ|:  max = {psi.max():.2e}, mean = {psi.mean():.2e}")
    print(f"  |Φ_s|:  max = {np.abs(pt.phi_s).max():.2e}")
    print(f"  |∇φ|:   max = {np.abs(pt.grad_phi).max():.2e}")
    blind = psi.max() < 1e-9 and np.abs(pt.phi_s).max() < 1e-9
    print()
    print(f"  -> displayed auxiliary coordinates vanish: {blind}")
    print("VERDICT: at this particular snapshot the extractor has no direct")
    print("ν_f coordinate. This finite check does not establish why any")
    print("separate catalog or equivariance result succeeds or fails.")
    print()


# ============================================================================
# EXPERIMENT 2: deliberately encode the frequencies in phase
# ============================================================================
def experiment_2_dynamics_carries_primes(G):
    """Measure a deliberately constructed phase encoding ``theta=tau*nu_f``."""
    print("=" * 72)
    print("EXPERIMENT 2: Constructed Phase Encoding of the Frequencies")
    print("=" * 72)
    print()
    print("Set θ = (ν_f·τ) mod 2π by hand, then extract the auxiliary")
    print("coordinates. The nodal equation does not imply this phase law.")
    print()

    nodes = list(G.nodes())
    tau = 1.0
    for n in nodes:
        G.nodes[n]["phase"] = float((G.nodes[n]["nu_f"] * tau) % (2 * math.pi))
    pt = extract_phase_space_point(G)
    idx = {n: i for i, n in enumerate(pt.nodes)}

    primes = sorted({p for (p, _k) in nodes})
    by_prime = defaultdict(list)
    for p, k in nodes:
        by_prime[p].append(abs(pt.grad_phi[idx[(p, k)]]))
    mean_gp = [float(np.mean(by_prime[p])) for p in primes]
    logp = [math.log(p) for p in primes]
    r = float(np.corrcoef(mean_gp, logp)[0, 1])

    print(f"  primes:            {primes}")
    print(f"  mean |∇φ| / prime: {[round(x, 3) for x in mean_gp]}")
    print(f"  r(mean |∇φ|, log p) = {r:.3f}")
    print()
    print("VERDICT: the read-out retains information deliberately encoded in")
    print("phase. This is a consistency check of the extractor, not emergence")
    print("of prime information from a TNFR trajectory.")
    print()
    return pt


# ============================================================================
# EXPERIMENT 3: It re-expresses {k·log p}; it does not add Riemann structure
# ============================================================================
def experiment_3_reexpresses_not_adds(G, pt):
    """Compare spacings of bare frequencies and derived auxiliary actions."""
    print("=" * 72)
    print("EXPERIMENT 3: It Re-Expresses {k·log p}, It Does Not Add Riemann")
    print("=" * 72)
    print()
    print("The auxiliary fields are a deterministic function of the full graph")
    print("snapshot after wrapped θ = (k·log p)·τ is imposed. They add no information")
    print("beyond that input snapshot. The following spacing statistic is only")
    print("illustrative; it does not classify either sequence universally:")
    print()

    nodes = list(G.nodes())
    bare = [G.nodes[n]["nu_f"] for n in nodes]  # {k·log p}
    action = 0.5 * (
        pt.k_phi**2 + pt.j_phi**2 + pt.phi_s**2 + pt.j_dnfr**2
    )  # substrate action
    d_bare = _ks_vs_gue(bare)
    d_sub = _ks_vs_gue(action)

    print(f"  KS-vs-GUE of bare prime-ladder {{k·log p}}:   D ≈ {d_bare:.3f}")
    print(f"  KS-vs-GUE of substrate action ½|ζ|²:        D ≈ {d_sub:.3f}")
    print("  [KS values are INDICATIVE — crude unfolding and a small sample]")
    print()
    print("VERDICT: the auxiliary action RE-EXPRESSES the constructed")
    print("prime-ladder snapshot. This finite statistic does not produce or")
    print("identify the Riemann-zero spectrum, and it supplies no T-HP map.")
    print()


# ============================================================================
# EXPERIMENT 4: Synthesis — what the new physics does and does not give
# ============================================================================
def experiment_4_synthesis():
    """Honest placement relative to the frozen program."""
    print("=" * 72)
    print("EXPERIMENT 4: Synthesis — the New Physics, Honestly Placed")
    print("=" * 72)
    print()
    print("  Default P14 snapshot: extracted phase/pressure tuple is zero.")
    print("    -> no causal explanation for separate B1 results is inferred.")
    print("  Constructed phase encoding:     carries log(p) by design (Exp 2).")
    print("    -> an extractor consistency check, not an engine trajectory.")
    print("  Auxiliary action read-out (Exp 3):")
    print("    -> deterministic in the constructed full graph snapshot.")
    print()
    print("  The example neither proves the auxiliary substrate necessary for")
    print("  T-HP nor supplies the admissible rescaling F. Any remaining")
    print("  oscillatory residual is part of the separate, open program.")
    print()
    print("  STATUS: the program remains PAUSED at T-HP. G4 = RH is OPEN.")
    print("  This is a finite read-out characterization, not a closure.")
    print()


def main():
    print()
    print("  TNFR Example 103: Auxiliary Read-outs on the Prime Ladder")
    print("  Characterization, not closure — G4 = RH remains open")
    print("  =====================================================")
    print()
    G = build_prime_ladder_graph(N_PRIMES, max_power=K)
    experiment_1_static_blindness(G)
    pt = experiment_2_dynamics_carries_primes(G)
    experiment_3_reexpresses_not_adds(G, pt)
    experiment_4_synthesis()
    print("=" * 72)
    print("WHAT THIS ESTABLISHES")
    print("=" * 72)
    print()
    print("The default snapshot has zero extracted phase/pressure coordinates.")
    print("After phase = nu_f is imposed, the read-out correlates with log(p)")
    print("because that information was explicitly inserted. The auxiliary")
    print("action is deterministic in the constructed snapshot and supplies no")
    print("new arithmetic information or T-HP rescaling. The Riemann program")
    print("remains open; this example is only a finite read-out characterization.")
    print()


if __name__ == "__main__":
    main()
