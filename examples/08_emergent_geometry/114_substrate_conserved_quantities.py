"""TNFR Example 114: What the Substrate Conserves — symplectomorphism,
adiabatic invariance, and the γ-dial.

Example 113 built the bridge between the conservative symplectic substrate
and the dissipative structural diffusion. This example explores three
further consequences of that emergent geometry, none of which reopens any
paused program:

(a) OPERATOR SYMPLECTOMORPHISM. The substrate Hamiltonian flow U(t) =
    exp(t·X_H) is an EXACT symplectomorphism: it preserves the symplectic
    form, the loop action ∮p·dq, the Hamiltonian H_sub, and every action
    variable I = ½|ζ|². The 13 canonical operators are canonical
    transformations ON this substrate: most leave the global substrate
    energy invariant, while UM (Coupling) collapses the geometric sector by
    phase-synchronisation — the same collapse seen in the polarization
    geometry (example 106).

(b) ADIABATIC INVARIANCE. The action I = E/ω is an ADIABATIC INVARIANT of
    a slowly-varying structural frequency: when ν_f ramps slowly the action
    is conserved, when it ramps suddenly the action drifts. This is the
    empirically-established adiabatic theorem (Ehrenfest 1916). ν_f is the
    clock; the 13 operators are the canonical transforms that redistribute
    the (adiabatically-conserved) actions.

(c) THE γ-DIAL. The damped substrate wave q̈ + γq̇ + Lq = 0 has a single
    dial γ. Example 113 took γ→∞ (overdamped projection onto diffusion).
    Here we take γ→0: the roots become the pure-imaginary pair s = ±i√λ_k,
    so every mode oscillates undamped at the standing-wave frequency
    ω_k = √λ_k — exactly the discrete modes of a bounded elastic medium.
    γ→∞ diffusion, γ→0 standing waves: one dial, two empirically-grounded
    regimes.

HONEST SCOPE
============
Foundational geometry of the emergent substrate. The flow symplectomorphism
and action-angle structure are exact; the adiabatic invariance is the
Ehrenfest theorem with ν_f as the slow frequency; the γ-dial connects two
canonical TNFR modules. Resolves no open program (Riemann G4,
Navier–Stokes).

References:
- src/tnfr/physics/symplectic_substrate.py (verify_adiabatic_invariance)
- src/tnfr/physics/structural_diffusion.py (verify_undamped_limit)
- examples/08_emergent_geometry/98_emergent_symplectic_substrate.py
- examples/08_emergent_geometry/99_structural_diffusion.py
- examples/08_emergent_geometry/113_overdamped_projection_bridge.py
- AGENTS.md §"Emergent Symplectic Substrate (CANONICAL)"
"""

import math
import os
import random
import sys
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import networkx as nx
import numpy as np

from tnfr.dynamics import default_compute_delta_nfr
from tnfr.operators.definitions import (
    Coherence,
    Contraction,
    Coupling,
    Dissonance,
    Emission,
    Expansion,
    Mutation,
    Reception,
    Recursivity,
    Resonance,
    SelfOrganization,
    Silence,
    Transition,
)
from tnfr.physics.structural_diffusion import (
    verify_overdamped_projection,
    verify_undamped_limit,
)
from tnfr.physics.symplectic_substrate import (
    evolve_substrate_flow,
    extract_phase_space_point,
    geometric_sector_energy,
    potential_sector_energy,
    substrate_flow_matrix,
    substrate_hamiltonian,
    symplectic_form_matrix,
    to_action_angle,
    verify_adiabatic_invariance,
)

ALL_OPS = [
    ("AL", Emission),
    ("EN", Reception),
    ("IL", Coherence),
    ("OZ", Dissonance),
    ("UM", Coupling),
    ("RA", Resonance),
    ("SHA", Silence),
    ("VAL", Expansion),
    ("NUL", Contraction),
    ("THOL", SelfOrganization),
    ("ZHIR", Mutation),
    ("NAV", Transition),
    ("REMESH", Recursivity),
]


def _build(n=24, seed=11):
    rng = random.Random(seed)
    G = nx.watts_strogatz_graph(n, 4, 0.2, seed=seed)
    for nd in G.nodes():
        G.nodes[nd]["theta"] = rng.uniform(0.0, 2.0 * math.pi)
        G.nodes[nd]["EPI"] = rng.uniform(-0.4, 0.4)
        G.nodes[nd]["nu_f"] = rng.uniform(0.5, 1.5)
    default_compute_delta_nfr(G)
    return G


def experiment_a_symplectomorphism():
    """The substrate flow is an exact symplectomorphism; operators act on it."""
    print("=" * 72)
    print("(a) OPERATOR SYMPLECTOMORPHISM: the flow preserves everything")
    print("=" * 72)
    print()

    G = _build(24)
    p0 = extract_phase_space_point(G)
    H0 = substrate_hamiltonian(p0)

    # the flow matrix M(t) is exactly symplectic: M^T Omega M = Omega
    omega = symplectic_form_matrix(p0.n_nodes)
    print("Flow matrix M(t) is symplectic (M^T Omega M = Omega):")
    for t in (0.5, 1.7, 3.1):
        M = substrate_flow_matrix(p0.n_nodes, t)
        residual = float(np.max(np.abs(M.T @ omega @ M - omega)))
        detM = float(np.linalg.det(M))
        print(f"  t={t:>4.1f}  |M^T Omega M - Omega|={residual:.2e}  det M={detM:.6f}")
    print()

    # the flow preserves H_sub and the total action exactly
    print("Flow preserves H_sub and the total action exactly:")
    for t in (0.0, 0.7, 1.7, 3.1):
        pt = evolve_substrate_flow(p0, t)
        aa = to_action_angle(pt)
        itot = float(aa["action_geometric"].sum() + aa["action_potential"].sum())
        print(f"  t={t:>4.1f}  H_sub={substrate_hamiltonian(pt):.6f}  sum I={itot:.6f}")
    print()

    # the 13 operators act ON the substrate: most preserve global energy,
    # UM collapses the geometric sector (phase-sync, cf. polarization ex.106)
    print("The 13 operators acting on the substrate (energy budget):")
    print(f"  baseline H_sub={H0:.4f}")
    print(f"  {'op':>7} {'H/H0':>8} {'E_geo/0':>8} {'E_pot/0':>8}  effect")
    Eg0 = geometric_sector_energy(p0)
    Ep0 = potential_sector_energy(p0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, cls in ALL_OPS:
            Gc = _build(24)
            op = cls()
            for nd in list(Gc.nodes()):
                op(Gc, nd)
            default_compute_delta_nfr(Gc)
            p1 = extract_phase_space_point(Gc)
            hr = substrate_hamiltonian(p1) / H0
            egr = geometric_sector_energy(p1) / Eg0
            epr = potential_sector_energy(p1) / Ep0
            effect = (
                "preserves H_sub"
                if abs(hr - 1) < 0.05
                else ("collapses E_geo" if egr < 0.5 else "redistributes")
            )
            print(f"  {name:>7} {hr:>8.4f} {egr:>8.4f} {epr:>8.4f}  {effect}")
    print()
    print("-> the FLOW is an exact symplectomorphism (preserves omega, H_sub,")
    print("   actions, loop integral); the OPERATORS are canonical transforms")
    print("   on it. UM collapses the geometric sector by phase-sync, exactly")
    print("   the polarization-vector collapse of example 106.")
    print()


def experiment_b_adiabatic():
    """The action is an adiabatic invariant of a slow nu_f ramp."""
    print("=" * 72)
    print("(b) ADIABATIC INVARIANCE: slow nu_f conserves the action I=E/omega")
    print("=" * 72)
    print()
    print("Oscillator q'' + omega(t)^2 q = 0, omega ramped over time T_ramp.")
    print("Action I=E/omega is conserved adiabatically (Ehrenfest 1916).")
    print()

    cert = verify_adiabatic_invariance(
        omega_start=1.0,
        omega_end=3.0,
        ramp_times=(1.0, 5.0, 20.0, 80.0, 320.0),
    )
    print(f"  {'T_ramp':>8} {'eps~1/T':>9} {'rel_action_drift':>17}")
    for t, d in zip(cert.ramp_times, cert.action_drifts):
        print(f"  {t:>8.0f} {1.0 / t:>9.4f} {d:>17.3e}")
    print()
    print(f"  fast-ramp drift (T=1)   : {cert.fast_drift:.3e}")
    print(f"  slow-ramp drift (T=320) : {cert.slow_drift:.3e}")
    print(f"  adiabatic invariant     : {cert.is_adiabatic_invariant}")
    print()
    print("-> as the ramp slows (eps=omega_dot/omega^2 -> 0) the action drift")
    print("   collapses: I is the ADIABATIC INVARIANT. nu_f is the clock; the")
    print("   13 operators redistribute the adiabatically-conserved actions.")
    print()


def experiment_c_gamma_dial():
    """The gamma-dial: gamma->inf diffusion (113), gamma->0 standing waves."""
    print("=" * 72)
    print("(c) THE gamma-DIAL: gamma->0 recovers the standing waves")
    print("=" * 72)
    print()

    G = _build(24)
    print("gamma->0 end (this example): damped wave -> standing waves")
    print(
        f"  {'gamma':>8} {'max|Re s|':>11} {'freq_err':>11} {'/gamma^2':>9} "
        f"{'matches':>8}"
    )
    for gamma in (0.5, 0.1, 0.01, 0.001):
        cert = verify_undamped_limit(G, gamma=gamma)
        print(
            f"  {gamma:>8.3f} {cert.max_decay_rate:>11.3e} "
            f"{cert.max_freq_rel_error:>11.3e} "
            f"{cert.freq_error_times_inv_gamma_sq:>9.3f} "
            f"{str(cert.matches_discrete_modes):>8}"
        )
    cert0 = verify_undamped_limit(G, gamma=1e-3)
    print(
        f"  standing-wave frequencies omega_k=sqrt(lambda_k): "
        f"{[round(f, 4) for f in cert0.standing_wave_frequencies]}"
    )
    print()
    print("gamma->inf end (example 113): damped wave -> structural diffusion")
    proj = verify_overdamped_projection(G, gamma=100.0)
    print(
        f"  nu_f=1/gamma={proj.nu_f_effective:.4f}, rate error "
        f"{proj.max_rate_rel_error:.2e}, trajectory error "
        f"{proj.trajectory_max_rel_error:.2e}"
    )
    print()
    print("-> ONE dial gamma: gamma->0 conservative standing waves (Re s->0,")
    print("   Im s->sqrt(lambda_k)); gamma->inf dissipative diffusion. The")
    print("   substrate wave and the structural diffusion are the two ends.")
    print()


def main():
    print()
    print("#" * 72)
    print("# TNFR Example 114: What the Substrate Conserves")
    print("# symplectomorphism (a) + adiabatic invariance (b) + gamma-dial (c)")
    print("#" * 72)
    print()
    experiment_a_symplectomorphism()
    experiment_b_adiabatic()
    experiment_c_gamma_dial()
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print("(a) the substrate flow is an EXACT symplectomorphism; operators")
    print("    are canonical transforms on it (UM collapses E_geo).")
    print("(b) the action I=E/omega is an ADIABATIC INVARIANT of slow nu_f.")
    print("(c) one dial gamma: gamma->0 standing waves, gamma->inf diffusion.")
    print()
    print("Foundational emergent geometry. No open program resolved.")


if __name__ == "__main__":
    main()
