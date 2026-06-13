#!/usr/bin/env python3
"""
Example 103 — The Emergent Substrate Meets the Riemann Program (Characterization)
================================================================================

Revisits the (paused) TNFR-Riemann program through the new physics built
this session: the emergent symplectic substrate (Example 98) and the
structural-transport view (Example 99), rather than the static graph
geometry of G_P14. It does NOT reopen or advance G4 = RH. It precisely
CHARACTERIZES how the emergent geometry relates to the frozen program —
crediting the structural intuition that one should work with the geometry
that emerges from the nodal dynamics, not the imposed graph, while keeping
the honest scope intact.

Background (why the program froze)
----------------------------------
The TNFR-Riemann program is paused at the boundary of Conjecture T-HP:
∃ an admissible operator F built only from the tetrad (Φ_s, |∇φ|, K_φ,
ξ_C) + canonical constants + grammar U1–U6 such that F·H_P14·F* has
spectrum {γ_n} (the Riemann zeros). P28/P30 closed the SMOOTH half of F;
the residual is the OSCILLATORY half S(T) = (1/π)·arg ζ(½+iT), which is
RH-equivalent. The branch B1 (closeable inside the 13-operator catalog)
was structurally CLOSED on G_P14 by the Canonical Catalog Equivariance
Theorem: every catalog operator on G_P14 commutes with the S_n
prime-relabelling, so it cannot encode Riemann level statistics. ALL of
that is about operators on the STATIC graph geometry.

The structural fact this example measures
-----------------------------------------
The prime-ladder Hamiltonian P14 places its entire prime content in the
structural frequency ν_f = k·log p (each node (p,k); phase = 0, ΔNFR = 0
by construction). The emergent symplectic substrate, however, is built
from the tetrad coordinates (K_φ, J_φ, Φ_s, J_ΔNFR), which are computed
from the PHASE θ and the pressure ΔNFR — never from ν_f. Three measured
consequences (all verified below, n_primes=10, K=4 → 40 nodes):

1. STATIC blindness: on the default P14 state (θ = 0, ΔNFR = 0) the whole
   substrate is EXACTLY zero (|Ψ| = |Φ_s| = |∇φ| = 0 to machine
   precision). The substrate is BLIND to the primes — this is the
   structural reason the static-graph analysis closed B1: the tetrad does
   not read ν_f.

2. DYNAMICS carries the primes: the nodal equation advances phase at the
   structural frequency (θ̇ ∝ ν_f), so the dynamics-emergent state
   θ = ν_f·τ = (k·log p)·τ makes the tetrad prime-specific:
   r(mean|∇φ| per prime, log p) ≈ 0.99. The geometry that emerges from
   the DYNAMICS — unlike the static graph — does see ν_f. This is the
   structural intuition, made precise: the right object is the emergent
   geometry.

3. But it RE-EXPRESSES, it does not ADD: the substrate fields are a
   DETERMINISTIC function of the state θ = (k·log p)·τ, so the substrate
   spectrum is a function of {k·log p}. It cannot contain more information
   than the prime-ladder spectrum already has. Its level statistics stay
   in the integrable / Poisson-like class (far from the Riemann/GUE
   class), exactly like the bare {k·log p}. The substrate does NOT, by
   itself, supply the rescaling to {γ_n}.

Honest scope
------------
- This does NOT close, reopen, or advance G4 = RH. The program remains
  PAUSED at T-HP. The oscillatory half S(T) (= ker of the REMESH-∞
  projection, N15; RH-equivalent) remains the genuine open residual.
- The POSITIVE content is a consistency/characterization result: the
  emergent substrate is non-trivially populated by the prime-ladder
  content UNDER THE DYNAMICS (a prerequisite for any tetrad-built F of
  T-HP), and the static blindness pins down precisely why graph-geometry
  arguments (CCET) closed B1. This strengthens, and is consistent with,
  the existing P28/P30 smooth-half closure and the N15 smooth/oscillatory
  split — it does not supply a new F.
- "The substrate carries log p" is, at bottom, the statement that
  ν_f = k·log p is prime-specific (true by construction) and that the
  dynamics propagates it into θ. It is a faithful structural restatement,
  not a new theorem, and emphatically not a route to RH.

References
----------
- AGENTS.md §"TNFR-Riemann Program" (T-HP, branches B1/B2/B3, frozen)
- AGENTS.md §"Emergent Symplectic Substrate" (the new geometry)
- examples/08_emergent_geometry/98_emergent_symplectic_substrate.py (substrate construction)
- src/tnfr/riemann/prime_ladder_hamiltonian.py (P14: ν_f = k·log p)
- src/tnfr/physics/symplectic_substrate.py (extract_phase_space_point)
- theory/TNFR_RIEMANN_RESEARCH_NOTES.md §13septies (T-HP), §13vicies-novies (CCET)
"""

import os
import sys
import math
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import numpy as np

from tnfr.riemann.prime_ladder_hamiltonian import build_prime_ladder_graph
from tnfr.physics.symplectic_substrate import extract_phase_space_point

N_PRIMES = 10
K = 4


def _ks_vs_gue(spectrum):
    """Indicative KS distance of unit-mean nn-spacings to the GUE surmise."""
    s = np.sort(np.asarray(spectrum, dtype=float))
    s = s[np.isfinite(s)]
    sp = np.diff(s)
    sp = sp[sp > 1e-12]
    if len(sp) < 3:
        return float('nan')
    sp = sp / sp.mean()
    xs = np.sort(sp)
    emp = np.arange(1, len(xs) + 1) / len(xs)
    gue = 1.0 - np.exp(-(4.0 / np.pi) * xs ** 2)
    return float(np.max(np.abs(emp - gue)))


# ============================================================================
# EXPERIMENT 1: Static blindness — the substrate does not read ν_f
# ============================================================================
def experiment_1_static_blindness(G):
    """Default P14 (θ=0, ΔNFR=0): the whole substrate is exactly zero."""
    print("=" * 72)
    print("EXPERIMENT 1: Static Blindness — the Substrate Does Not Read ν_f")
    print("=" * 72)
    print()
    print("P14 puts all prime content in ν_f = k·log p, with phase = 0 and")
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
    print(f"  -> substrate is EXACTLY blind to the primes: {blind}")
    print("VERDICT: this is the structural reason the static-graph analysis")
    print("(CCET, Euler-Orthogonality) closed B1 — the tetrad/substrate does")
    print("not see ν_f, where P14's prime content lives.")
    print()


# ============================================================================
# EXPERIMENT 2: The dynamics-emergent substrate carries the primes
# ============================================================================
def experiment_2_dynamics_carries_primes(G):
    """θ = ν_f·τ makes the tetrad prime-specific: r(|∇φ|, log p) ≈ 0.99."""
    print("=" * 72)
    print("EXPERIMENT 2: The Dynamics-Emergent Substrate Carries the Primes")
    print("=" * 72)
    print()
    print("The nodal equation advances phase at the structural frequency")
    print("(θ̇ ∝ ν_f). The dynamics-emergent state θ = ν_f·τ = (k·log p)·τ")
    print("makes the tetrad prime-specific:")
    print()

    nodes = list(G.nodes())
    tau = 1.0
    for n in nodes:
        G.nodes[n]['phase'] = float(G.nodes[n]['nu_f'] * tau)
    pt = extract_phase_space_point(G)
    idx = {n: i for i, n in enumerate(pt.nodes)}

    primes = sorted({p for (p, _k) in nodes})
    by_prime = defaultdict(list)
    for (p, k) in nodes:
        by_prime[p].append(abs(pt.grad_phi[idx[(p, k)]]))
    mean_gp = [float(np.mean(by_prime[p])) for p in primes]
    logp = [math.log(p) for p in primes]
    r = float(np.corrcoef(mean_gp, logp)[0, 1])

    print(f"  primes:            {primes}")
    print(f"  mean |∇φ| / prime: {[round(x, 3) for x in mean_gp]}")
    print(f"  r(mean |∇φ|, log p) = {r:.3f}")
    print()
    print("VERDICT: the geometry that emerges from the DYNAMICS — unlike the")
    print("static graph — DOES see ν_f. The emergent substrate is the right")
    print("object, exactly as the structural intuition says.")
    print()
    return pt


# ============================================================================
# EXPERIMENT 3: It re-expresses {k·log p}; it does not add Riemann structure
# ============================================================================
def experiment_3_reexpresses_not_adds(G, pt):
    """Substrate spectrum is a function of {k·log p}: integrable, not Riemann."""
    print("=" * 72)
    print("EXPERIMENT 3: It Re-Expresses {k·log p}, It Does Not Add Riemann")
    print("=" * 72)
    print()
    print("The substrate fields are a DETERMINISTIC function of the state")
    print("θ = (k·log p)·τ, so the substrate spectrum is a function of the")
    print("prime-ladder spectrum {k·log p} — it cannot carry more")
    print("information. Its level statistics stay in the integrable class:")
    print()

    nodes = list(G.nodes())
    bare = [G.nodes[n]['nu_f'] for n in nodes]               # {k·log p}
    action = 0.5 * (pt.k_phi ** 2 + pt.j_phi ** 2
                    + pt.phi_s ** 2 + pt.j_dnfr ** 2)         # substrate action
    d_bare = _ks_vs_gue(bare)
    d_sub = _ks_vs_gue(action)

    print(f"  KS-vs-GUE of bare prime-ladder {{k·log p}}:   D ≈ {d_bare:.3f}")
    print(f"  KS-vs-GUE of substrate action ½|ζ|²:        D ≈ {d_sub:.3f}")
    print("  (reference: Riemann zeros ≈ 0.08, GUE ≈ 0, Poisson ≈ 0.30)")
    print("  [KS values are INDICATIVE — crude unfolding, small N]")
    print()
    print("VERDICT: both stay far from the Riemann/GUE class — the substrate")
    print("RE-EXPRESSES the integrable prime-ladder content; it does NOT")
    print("produce the Riemann statistics. The rescaling {k·log p} → {γ_n}")
    print("(the operator F of T-HP) is NOT supplied by the substrate alone.")
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
    print("  Static graph geometry (G_P14):  BLIND to the primes (Exp 1).")
    print("    -> structural origin of the B1 closure (CCET on G_P14).")
    print("  Dynamics-emergent geometry:     CARRIES the primes (Exp 2).")
    print("    -> the right object; the structural intuition, made precise.")
    print("  But the substrate RE-EXPRESSES {k·log p} (Exp 3):")
    print("    -> it is a deterministic function of the prime-ladder")
    print("       spectrum; it adds no Riemann structure by itself.")
    print()
    print("  So the emergent substrate is a NECESSARY arena for T-HP (it is")
    print("  non-trivially populated by the prime data under the dynamics),")
    print("  but it does NOT supply the admissible rescaling F. The residual")
    print("  is precisely the OSCILLATORY half S(T) = (1/π)·arg ζ(½+iT) —")
    print("  the RH-equivalent kernel already isolated by P28/P30 and N15.")
    print()
    print("  STATUS: the program remains PAUSED at T-HP. G4 = RH is OPEN.")
    print("  This is a characterization that STRENGTHENS the honest picture,")
    print("  not a closure or a reopening.")
    print()


def main():
    print()
    print("  TNFR Example 103: The Emergent Substrate Meets Riemann")
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
    print("Working with the geometry that emerges from the nodal dynamics")
    print("(the symplectic substrate) rather than the static graph G_P14 is")
    print("the correct stance: the static graph is exactly blind to the")
    print("primes (which is why graph-operator arguments closed B1), while")
    print("the dynamics-emergent substrate carries the prime-ladder content")
    print("(r ≈ 0.99 with log p). But the substrate is a deterministic")
    print("function of {k·log p}; it re-expresses, it does not add Riemann")
    print("structure. The admissible rescaling F of T-HP — specifically its")
    print("oscillatory half S(T), RH-equivalent — is NOT supplied by the")
    print("substrate alone. The program stays paused at T-HP; G4 = RH")
    print("remains open. This is an honest characterization of where the new")
    print("physics helps (the arena) and where it does not (the rescaling).")
    print()


if __name__ == "__main__":
    main()
