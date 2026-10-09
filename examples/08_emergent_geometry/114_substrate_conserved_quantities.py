"""TNFR Example 114: What the specified substrate conserves — symplectic flow,
adiabatic invariance, and the γ-dial.

Example 113 placed the auxiliary conservative harmonic model beside the
separate dissipative structural-diffusion model. This example explores three
further consequences of that emergent geometry, none of which reopens any
paused program:

(a) FLOW SYMPLECTICITY. The substrate Hamiltonian flow U(t) =
    exp(t·X_H) is an EXACT symplectomorphism: it preserves the symplectic
    form, the loop action ∮p·dq, the Hamiltonian H_sub, and every action
    variable I = ½|ζ|². Applying the 13 engine operators to a graph and then
    re-extracting substrate coordinates gives finite-snapshot energy changes;
    it does not test the Jacobian pullback required for symplecticity.

(b) ADIABATIC INVARIANCE. The action I = E/ω is an ADIABATIC INVARIANT of
    the specified slowly-varying oscillator: when its frequency ramps slowly the action
    is conserved, when it ramps suddenly the action drifts. This is the
    numerical realization of the standard adiabatic result. Reading ω as ν_f
    is a declared model convention, not an operator or nodal-equation theorem.

(c) THE γ-DIAL. The separately posited damped graph-wave model
    q̈ + γq̇ + Lq = 0 has a single
    dial γ. Example 113 took γ→∞ (overdamped projection onto diffusion).
    Here we take γ→0: the roots become the pure-imaginary pair s = ±i√λ_k,
    so every mode oscillates undamped at the standing-wave frequency
    ω_k = √λ_k — exactly the discrete modes of a bounded elastic medium.
    γ→∞ diffusion, γ→0 standing waves: one dial, two empirically-grounded
    regimes.

HONEST SCOPE
============
The flow symplectomorphism and action-angle structure are exact for the
specified harmonic model; the adiabatic calculation uses a prescribed scalar
oscillator; the γ-dial belongs to the separate damped graph-wave model. None
of these checks proves engine-operator symplecticity or resolves an open
program.

References:
- src/tnfr/physics/symplectic_substrate.py (verify_adiabatic_invariance)
- src/tnfr/physics/structural_diffusion.py (verify_undamped_limit)
- examples/08_emergent_geometry/98_emergent_symplectic_substrate.py
- examples/08_emergent_geometry/99_structural_diffusion.py
- examples/08_emergent_geometry/113_overdamped_projection_bridge.py
- AGENTS.md §"Emergent symplectic substrate"
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
    """Verify the substrate flow and report separate operator snapshot changes."""
    print("=" * 72)
    print("(a) FLOW SYMPLECTICITY AND SEPARATE OPERATOR SNAPSHOTS")
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

    # Re-extraction after operator application is an energy-response diagnostic.
    # It does not evaluate an operator Jacobian or its symplectic pullback.
    print("Re-extracted coordinates after each operator (snapshot response):")
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
                "small H_sub change"
                if abs(hr - 1) < 0.05
                else ("collapses E_geo" if egr < 0.5 else "redistributes")
            )
            print(f"  {name:>7} {hr:>8.4f} {egr:>8.4f} {epr:>8.4f}  {effect}")
    print()
    print("-> the FLOW is an exact symplectomorphism (preserves omega, H_sub,")
    print("   actions and the loop integral). Operator rows only compare two")
    print("   extracted snapshots; they do not certify symplectic maps.")
    print()


def experiment_b_adiabatic():
    """The action is an adiabatic invariant of a slow nu_f ramp."""
    print("=" * 72)
    print("(b) ADIABATIC INVARIANCE OF A PRESCRIBED OSCILLATOR")
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
    print("   collapses. Identifying omega with nu_f is an explicit convention")
    print("   for this auxiliary oscillator; no operator result follows.")
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
    print("# TNFR Example 114: What the Specified Substrate Conserves")
    print("# symplectomorphism (a) + adiabatic invariance (b) + gamma-dial (c)")
    print("#" * 72)
    print()
    experiment_a_symplectomorphism()
    experiment_b_adiabatic()
    experiment_c_gamma_dial()
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print("(a) the substrate flow is an EXACT symplectomorphism; the operator")
    print("    table is a snapshot response and does not test symplecticity.")
    print("(b) I=E/omega is adiabatically invariant for the prescribed ramp.")
    print("(c) one dial gamma: gamma->0 standing waves, gamma->inf diffusion.")
    print()
    print("Declared auxiliary models; no open program resolved.")


if __name__ == "__main__":
    main()
