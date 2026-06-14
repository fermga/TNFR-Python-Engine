"""TNFR Example 113: The Overdamped Projection Bridge — from the conservative
symplectic substrate to dissipative structural diffusion.

Two emergent-geometry pieces of TNFR have, so far, been studied separately:

- the **conservative symplectic substrate** (example 98): the geometry the
  nodal dynamics generates from itself, carrying the graph *wave* equation
  q̈ = −L q (second order, mode k oscillating at √λ_k);
- the **dissipative structural diffusion** (example 99): the literal
  transport content of the nodal equation, q̇ = −νf·L·q (first order, mode
  k relaxing at νf·λ_k).

AGENTS.md states that the nodal equation is the *overdamped projection* of
the Hamiltonian flow on the substrate. This example MEASURES that bridge.

THE BRIDGE (precise statement on a graph)
=========================================
Damp the conservative substrate wave with a coefficient γ:

    q̈ + γ q̇ + L q = 0          (damped graph oscillator)

Per mode k the characteristic equation s² + γs + λ_k = 0 has a **slow** root
s₋ → −λ_k/γ and a **fast** root s₊ → −γ. In the strong-damping limit the
fast root is an instantaneous transient and mode k relaxes at λ_k/γ. That is
exactly the structural-diffusion rate νf·λ_k under the identification

    νf = 1/γ        (structural frequency = inverse damping = MOBILITY).

This is the same νf-as-mobility refinement of the classical-regime
correspondence: the nodal equation is first order, so νf is a mobility, not
an inverse mass. The conservative second-order substrate, damped, projects
onto the first-order diffusion — both empirically grounded (damped
oscillator; Smoluchowski overdamped limit) and both already canonical TNFR
objects.

WHAT EMERGES (measured, not asserted)
=====================================
- The damped slow rate converges to the diffusion rate; the bridge error
  scales as O(λ_max/γ²) (exact leading order).
- The slowest overdamped mode equals the diffusion spectral gap νf·λ₂.
- The damped-wave trajectory collapses onto the diffusion trajectory
  exp(−L t/γ)·q₀ as γ grows.

HONEST SCOPE
============
This connects two canonical modules (symplectic_substrate ↔
structural_diffusion) and confirms νf = 1/γ. It is foundational geometry,
not a resolution of any open program (Riemann G4, Navier–Stokes).

References:
- src/tnfr/physics/structural_diffusion.py (verify_overdamped_projection)
- src/tnfr/physics/symplectic_substrate.py (the conservative substrate)
- examples/08_emergent_geometry/98_emergent_symplectic_substrate.py
- examples/08_emergent_geometry/99_structural_diffusion.py
- AGENTS.md §"Regime Correspondences from Nodal Dynamics"
"""

import os
import sys
import math
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import networkx as nx

from tnfr.dynamics import default_compute_delta_nfr
from tnfr.physics.structural_diffusion import (
    damped_wave_rates,
    verify_overdamped_projection,
)


def _build(n=40, seed=11):
    rng = random.Random(seed)
    G = nx.watts_strogatz_graph(n, 6, 0.2, seed=seed)
    for nd in G.nodes():
        G.nodes[nd]["theta"] = rng.uniform(0.0, 2.0 * math.pi)
        G.nodes[nd]["EPI"] = rng.uniform(-0.4, 0.4)
        G.nodes[nd]["nu_f"] = rng.uniform(0.5, 1.5)
    default_compute_delta_nfr(G)
    return G


def experiment_1_rate_convergence():
    """The damped slow rate converges to the diffusion rate λ_k/γ."""
    print("=" * 72)
    print("EXPERIMENT 1: Damped substrate wave -> diffusion rates")
    print("=" * 72)
    print()
    print("Per mode k:  s^2 + gamma*s + lambda_k = 0")
    print("  slow root s_- -> -lambda_k/gamma  (the diffusion rate nu_f*lambda_k)")
    print("  fast root s_+ -> -gamma           (an instantaneous transient)")
    print("Bridge error should scale as O(lambda_max / gamma^2).")
    print()

    G = _build(40)
    print(f"{'gamma':>7} {'nu_f=1/g':>9} {'rate_err':>11} {'err*g^2':>9} "
          f"{'~lambda_max':>11} {'valid':>6}")
    for gamma in (5.0, 10.0, 20.0, 50.0, 100.0, 200.0):
        cert = verify_overdamped_projection(G, gamma=gamma)
        print(f"{gamma:>7.0f} {cert.nu_f_effective:>9.4f} "
              f"{cert.max_rate_rel_error:>11.3e} "
              f"{cert.rate_error_times_gamma_sq:>9.3f} "
              f"{cert.lambda_max:>11.3f} "
              f"{str(cert.is_valid_projection):>6}")
    print()
    print("-> err*gamma^2 stabilises near lambda_max: the bridge error is")
    print("   O(lambda_max/gamma^2). The projection is exact in the limit.")
    print()


def experiment_2_spectral_gap_match():
    """The slowest overdamped mode equals the diffusion spectral gap νf·λ₂."""
    print("=" * 72)
    print("EXPERIMENT 2: Slowest mode = diffusion spectral gap nu_f*lambda_2")
    print("=" * 72)
    print()

    G = _build(40)
    cert = verify_overdamped_projection(G, gamma=100.0)
    ratio = (
        cert.slowest_slow_rate / cert.spectral_gap
        if cert.spectral_gap > 0
        else 0.0
    )
    print(f"  damped slowest (Fiedler) slow rate : {cert.slowest_slow_rate:.6f}")
    print(f"  diffusion spectral gap nu_f*lambda_2: "
          f"{cert.slowest_diffusion_rate:.6f}")
    print(f"  slow_rate / lambda_2 (recovers nu_f): {ratio:.6f} "
          f"(= 1/gamma = {cert.nu_f_effective:.6f})")
    print()
    print("-> the slowest surviving overdamped mode is exactly the diffusion")
    print("   spectral gap, and slow_rate/lambda_2 recovers nu_f = 1/gamma.")
    print()


def experiment_3_nu_f_is_mobility():
    """νf = 1/γ: structural frequency is the inverse damping (mobility)."""
    print("=" * 72)
    print("EXPERIMENT 3: nu_f = 1/gamma  (mobility = inverse damping)")
    print("=" * 72)
    print()
    print("The nodal equation is FIRST order, so nu_f is a mobility, not an")
    print("inverse mass. The overdamped projection makes this exact: the")
    print("damping gamma of the conservative wave is precisely 1/nu_f.")
    print()

    G = _build(40)
    for gamma in (10.0, 25.0, 80.0):
        cert = verify_overdamped_projection(G, gamma=gamma)
        lambdas, s_slow, s_fast = damped_wave_rates(G, gamma)
        # the fast roots cluster near -gamma (the transient), the slow near 0
        fast_mean = float(s_fast.mean())
        print(f"  gamma={gamma:>6.1f} -> nu_f=1/gamma={cert.nu_f_effective:.4f}; "
              f"fast roots ~ {fast_mean:.2f} (= -gamma transient); "
              f"trajectory err {cert.trajectory_max_rel_error:.2e}")
    print()
    print("-> nu_f = 1/gamma confirmed: the structural frequency IS the")
    print("   mobility (inverse damping) of the conservative substrate.")
    print()


def main():
    print()
    print("#" * 72)
    print("# TNFR Example 113: The Overdamped Projection Bridge")
    print("# conservative symplectic substrate  -->  structural diffusion")
    print("#" * 72)
    print()
    experiment_1_rate_convergence()
    experiment_2_spectral_gap_match()
    experiment_3_nu_f_is_mobility()
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)
    G = _build(40)
    cert = verify_overdamped_projection(G, gamma=100.0)
    print(cert.summary())
    print()
    print("The dissipative structural diffusion (the nodal equation's literal")
    print("content) is the overdamped projection of the conservative")
    print("symplectic-substrate wave, with nu_f = 1/gamma. Two emergent-")
    print("geometry pieces, one bridge. Foundational geometry, no open")
    print("program resolved.")


if __name__ == "__main__":
    main()
