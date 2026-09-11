"""TNFR Example 113: A damped graph-wave limit matching structural diffusion.

Two auxiliary dynamics are compared under explicit fixed-graph assumptions:

- the **graph wave** q̈ = −Lq (second order, mode k oscillating at √λ_k);
- the **dissipative structural diffusion** (example 99): the literal
  transport content of the nodal equation, q̇ = −νf·L·q (first order, mode
  k relaxing at νf·λ_k).

The calculation establishes an overdamped limit for this graph wave. The
isotropic auxiliary substrate in example 98 has identity stiffness; therefore
this is not a derivation of the full nodal equation from that substrate.

THE BRIDGE (precise statement on a graph)
=========================================
Damp the graph wave with a coefficient γ:

    q̈ + γ q̇ + L q = 0          (damped graph oscillator)

Per mode k the characteristic equation s² + γs + λ_k = 0 has a **slow** root
s₋ → −λ_k/γ and a **fast** root s₊ → −γ. In the strong-damping limit the
fast root is an instantaneous transient and mode k relaxes at λ_k/γ. That is
exactly the structural-diffusion rate νf·λ_k under the identification

    νf = 1/γ        (structural frequency = inverse damping = MOBILITY).

This realizes the νf-as-mobility reading in the declared graph-wave model:
the nodal equation's EPI diffusion channel is first order, so νf multiplies
pressure as a mobility. The equality νf=1/γ is the parameter matching used in
this model, not a universal identity for every TNFR evolution.

WHAT EMERGES (measured, not asserted)
=====================================
- The damped slow rate converges to the diffusion rate; the bridge error
  scales as O(λ_max/γ²) (exact leading order).
- The slowest overdamped mode equals the diffusion spectral gap νf·λ₂.
- The damped-wave trajectory collapses onto the diffusion trajectory
  exp(−L t/γ)·q₀ as γ grows.

HONEST SCOPE
============
This connects a separately specified damped graph wave with restricted
pure-EPI structural diffusion and verifies the stated asymptotic rates. It
does not connect the full four-channel dynamics to the isotropic substrate or
resolve an open program.

References:
- src/tnfr/physics/structural_diffusion.py (verify_overdamped_projection)
- src/tnfr/physics/symplectic_substrate.py (distinct isotropic model)
- examples/08_emergent_geometry/98_emergent_symplectic_substrate.py
- examples/08_emergent_geometry/99_structural_diffusion.py
- AGENTS.md §"Regime Correspondences from Nodal Dynamics"
"""

import math
import os
import random
import sys

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
    print(
        f"{'gamma':>7} {'nu_f=1/g':>9} {'rate_err':>11} {'err*g^2':>9} "
        f"{'~lambda_max':>11} {'valid':>6}"
    )
    for gamma in (5.0, 10.0, 20.0, 50.0, 100.0, 200.0):
        cert = verify_overdamped_projection(G, gamma=gamma)
        print(
            f"{gamma:>7.0f} {cert.nu_f_effective:>9.4f} "
            f"{cert.max_rate_rel_error:>11.3e} "
            f"{cert.rate_error_times_gamma_sq:>9.3f} "
            f"{cert.lambda_max:>11.3f} "
            f"{str(cert.is_valid_projection):>6}"
        )
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
    ratio = cert.slowest_slow_rate / cert.spectral_gap if cert.spectral_gap > 0 else 0.0
    print(f"  damped slowest (Fiedler) slow rate : {cert.slowest_slow_rate:.6f}")
    print(
        f"  diffusion spectral gap nu_f*lambda_2: " f"{cert.slowest_diffusion_rate:.6f}"
    )
    print(
        f"  slow_rate / lambda_2 (recovers nu_f): {ratio:.6f} "
        f"(= 1/gamma = {cert.nu_f_effective:.6f})"
    )
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
    print("The nodal equation is FIRST order, so nu_f multiplies pressure as")
    print("a mobility. For this graph-wave comparison we set gamma=1/nu_f.")
    print()

    G = _build(40)
    for gamma in (10.0, 25.0, 80.0):
        cert = verify_overdamped_projection(G, gamma=gamma)
        lambdas, s_slow, s_fast = damped_wave_rates(G, gamma)
        # the fast roots cluster near -gamma (the transient), the slow near 0
        fast_mean = float(s_fast.mean())
        print(
            f"  gamma={gamma:>6.1f} -> nu_f=1/gamma={cert.nu_f_effective:.4f}; "
            f"fast roots ~ {fast_mean:.2f} (= -gamma transient); "
            f"trajectory err {cert.trajectory_max_rel_error:.2e}"
        )
    print()
    print("-> within this model, nu_f=1/gamma matches the diffusion rate.")
    print("   The equality is a declared parameter identification.")
    print()


def main():
    print()
    print("#" * 72)
    print("# TNFR Example 113: The Overdamped Projection Bridge")
    print("# damped graph wave  -->  restricted structural diffusion")
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
    print("Restricted pure-EPI diffusion is the overdamped limit of the")
    print("declared graph wave with nu_f=1/gamma. The result does not derive")
    print("the full nodal equation from the isotropic substrate and resolves")
    print("no open program.")


if __name__ == "__main__":
    main()
