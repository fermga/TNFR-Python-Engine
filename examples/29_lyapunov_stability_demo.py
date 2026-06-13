"""TNFR Lyapunov Stability & Structural Lifecycle.

Demonstrates the formal Lyapunov stability proof for TNFR dynamics:
grammar rule U2 guarantees that operator sequences are net-contractive
on the structural energy functional E[G].

Key results shown:
1. Per-operator energy bounds: all 13 operators classified (stabiliser /
   destabiliser / neutral / mixed) with contraction/expansion rates
2. Sequence Lyapunov proof: grammar-compliant sequences have product
   of energy multipliers <= 1 (net-contractive)
3. Spectral gap analysis: algebraic connectivity, relaxation time, mixing
4. Operator convergence: combined Lyapunov + spectral analysis
5. Grammar U2 in action: comparing compliant vs non-compliant sequences
6. Life emergence detection: autopoietic coefficient and vitality index

See: theory/STRUCTURAL_STABILITY_AND_DYNAMICS.md for the full treatment.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import networkx as nx

from tnfr.constants import inject_defaults
from tnfr.physics.lyapunov import (
    OPERATOR_LYAPUNOV_BOUNDS,
    EnergyClass,
    get_bound,
    compute_operator_energy_bound,
    prove_sequence_lyapunov,
    analyze_spectral_gap,
    analyze_operator_convergence,
    verify_operator_lyapunov,
)
from tnfr.physics.life import (
    compute_self_generation,
    compute_autopoietic_coefficient,
    compute_stability_margin,
    detect_life_emergence,
)

SEED = 42


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _build_graph(n: int = 20, seed: int = SEED) -> nx.Graph:
    """Build a Watts-Strogatz network with canonical TNFR attributes."""
    rng = np.random.default_rng(seed)
    G = nx.watts_strogatz_graph(n, 4, 0.3, seed=seed)
    inject_defaults(G)
    for node in G.nodes():
        G.nodes[node]['EPI'] = float(rng.uniform(0.5, 2.0))
        G.nodes[node]['nu_f'] = float(rng.uniform(0.5, 2.0))
        G.nodes[node]['phase'] = float(rng.uniform(0, 2 * np.pi))
        G.nodes[node]['delta_nfr'] = float(rng.uniform(-0.3, 0.3))
    return G


# ------------------------------------------------------------------
# 1. Per-operator Lyapunov bounds — the full registry
# ------------------------------------------------------------------

def demo_operator_bounds() -> None:
    """Display formal energy bounds for all 13 canonical operators."""
    print("=" * 72)
    print("1. PER-OPERATOR LYAPUNOV BOUNDS — all 13 canonical operators")
    print("=" * 72)

    # Group by energy class
    by_class: dict[EnergyClass, list] = {c: [] for c in EnergyClass}
    for name, bound in OPERATOR_LYAPUNOV_BOUNDS.items():
        by_class[bound.energy_class].append(bound)

    for cls in [EnergyClass.STABILISER, EnergyClass.DESTABILISER,
                EnergyClass.NEUTRAL, EnergyClass.MIXED]:
        ops = by_class[cls]
        if not ops:
            continue
        print(f"\n  {cls.value.upper()} operators:")
        print(f"  {'Name':20s}  {'Glyph':6s}  {'Rate':>10s}  Factor")
        print(f"  {'─' * 20}  {'─' * 6}  {'─' * 10}  {'─' * 20}")
        for b in sorted(ops, key=lambda x: x.contraction_rate, reverse=True):
            rate_label = {
                EnergyClass.STABILISER: f"ρ={b.contraction_rate:.4f}",
                EnergyClass.DESTABILISER: f"κ={b.contraction_rate:.4f}",
                EnergyClass.NEUTRAL: f"ε={b.contraction_rate:.4f}",
                EnergyClass.MIXED: f"κ={b.contraction_rate:.4f}",
            }[cls]
            print(f"  {b.operator_name:20s}  {b.glyph:6s}  {rate_label:>10s}  "
                  f"{b.glyph_factor_name}={b.glyph_factor_value:.4f}")
    print()


# ------------------------------------------------------------------
# 2. Sequence Lyapunov proof — grammar-compliant vs non-compliant
# ------------------------------------------------------------------

def demo_sequence_proof() -> None:
    """Prove that grammar-compliant sequences are net-contractive."""
    print("=" * 72)
    print("2. SEQUENCE LYAPUNOV PROOF — U2 guarantees net-contractivity")
    print("=" * 72)

    sequences = {
        "Bootstrap (AL, UM, IL)": ["Emission", "Coupling", "Coherence"],
        "Explore (OZ, ZHIR, IL)": ["Dissonance", "Mutation", "Coherence"],
        "Stabilize (IL, SHA)": ["Coherence", "Silence"],
        "Propagate (RA, UM)": ["Resonance", "Coupling"],
        "Full cycle": [
            "Emission", "Coupling", "Coherence",
            "Dissonance", "Mutation", "Coherence",
            "Resonance", "Coupling", "Coherence",
            "Silence",
        ],
        "VIOLATION: OZ without IL": ["Dissonance", "Silence"],
        "VIOLATION: OZ, VAL, no stabilizer": [
            "Dissonance", "Expansion", "Silence",
        ],
    }

    for label, seq in sequences.items():
        proof = prove_sequence_lyapunov(seq)
        status = "STABLE" if proof.is_net_contractive else "UNSTABLE"
        print(f"\n  {label}")
        print(f"    Operators:    {' → '.join(proof.operators)}")
        print(f"    Multipliers:  {' × '.join(f'{m:.4f}' for m in proof.energy_multipliers)}")
        print(f"    Product:      {proof.cumulative_product:.6f}")
        print(f"    Contractive?  {status}  "
              f"(net contraction = {proof.net_contraction:+.4f})")
    print()


# ------------------------------------------------------------------
# 3. Spectral gap analysis
# ------------------------------------------------------------------

def demo_spectral_gap() -> None:
    """Compute spectral gap and derived time-scales for different topologies."""
    print("=" * 72)
    print("3. SPECTRAL GAP ANALYSIS — topology controls relaxation")
    print("=" * 72)

    topologies = {
        "Ring (20)": nx.cycle_graph(20),
        "Watts-Strogatz (20, k=4, p=0.3)": nx.watts_strogatz_graph(20, 4, 0.3, seed=SEED),
        "Barabasi-Albert (20, m=2)": nx.barabasi_albert_graph(20, 2, seed=SEED),
        "Complete (10)": nx.complete_graph(10),
        "Star (20)": nx.star_graph(19),
    }

    print(f"\n  {'Topology':40s} {'λ₁':>8s} {'τ_relax':>10s} {'t_mix':>10s} {'Ratio':>8s}")
    print(f"  {'─' * 40} {'─' * 8} {'─' * 10} {'─' * 10} {'─' * 8}")

    for label, G in topologies.items():
        # Inject defaults for TNFR attributes
        inject_defaults(G)
        rng = np.random.default_rng(SEED)
        for node in G.nodes():
            G.nodes[node]['EPI'] = float(rng.uniform(0.5, 2.0))
            G.nodes[node]['nu_f'] = float(rng.uniform(0.5, 2.0))
            G.nodes[node]['phase'] = float(rng.uniform(0, 2 * np.pi))
            G.nodes[node]['delta_nfr'] = float(rng.uniform(-0.3, 0.3))

        spec = analyze_spectral_gap(G)
        tau_str = f"{spec.relaxation_time:.4f}" if spec.relaxation_time < 1e6 else "∞"
        mix_str = f"{spec.mixing_time_bound:.4f}" if spec.mixing_time_bound < 1e6 else "∞"
        ratio_str = f"{spec.spectral_ratio:.2f}" if spec.spectral_ratio < 1e6 else "∞"

        print(f"  {label:40s} {spec.spectral_gap:8.4f} {tau_str:>10s} {mix_str:>10s} {ratio_str:>8s}")
    print()


# ------------------------------------------------------------------
# 4. Operator convergence — Lyapunov + spectral combined
# ------------------------------------------------------------------

def demo_operator_convergence() -> None:
    """Show effective convergence rate combining operator and spectral gap."""
    print("=" * 72)
    print("4. OPERATOR CONVERGENCE — Lyapunov rate vs spectral gap")
    print("=" * 72)

    G = _build_graph()

    stabilisers = ["Coherence", "Reception", "Coupling", "SelfOrganization", "Transition"]

    print(f"\n  {'Operator':20s} {'ρ (Lyapunov)':>14s} {'λ₁ (spectral)':>14s} "
          f"{'Effective':>10s} {'Steps to ½E':>12s}")
    print(f"  {'─' * 20} {'─' * 14} {'─' * 14} {'─' * 10} {'─' * 12}")

    for name in stabilisers:
        summary = analyze_operator_convergence(G, name)
        steps_str = (f"{summary.steps_to_half_energy:.2f}"
                     if summary.steps_to_half_energy < 1e6 else "∞")
        print(f"  {name:20s} {summary.operator_bound.contraction_rate:14.4f} "
              f"{summary.spectral.spectral_gap:14.4f} "
              f"{summary.effective_convergence_rate:10.4f} {steps_str:>12s}")
    print()


# ------------------------------------------------------------------
# 5. Empirical Lyapunov verification on a real graph
# ------------------------------------------------------------------

def demo_empirical_verification() -> None:
    """Verify operator energy bounds against actual E[G] measurements."""
    print("=" * 72)
    print("5. EMPIRICAL LYAPUNOV VERIFICATION — bounds vs reality")
    print("=" * 72)

    from tnfr.operators import apply_glyph
    from tnfr.physics.canonical import (
        compute_structural_potential,
        compute_phase_gradient,
        compute_phase_curvature,
    )

    def _energy(G: nx.Graph) -> float:
        """Compute structural energy E = ½ Σ (Φ_s² + |∇φ|² + K_φ²)."""
        phi_s = compute_structural_potential(G)
        grad = compute_phase_gradient(G)
        k_phi = compute_phase_curvature(G)
        return sum(
            0.5 * (phi_s[n] ** 2 + grad[n] ** 2 + k_phi[n] ** 2)
            for n in G.nodes()
        )

    glyphs = [
        ("Coherence (IL)",  "IL"),
        ("Dissonance (OZ)", "OZ"),
        ("Emission (AL)",   "AL"),
    ]

    for label, glyph in glyphs:
        G = _build_graph()
        n = G.number_of_nodes()

        E_before = _energy(G)

        # Apply operator to a node
        test_node = list(G.nodes())[0]
        try:
            apply_glyph(G, test_node, glyph)
        except Exception:
            print(f"  {label:25s}  (skipped — apply error)")
            continue

        E_after = _energy(G)

        vf = verify_operator_lyapunov(glyph, E_before, E_after, n_nodes=n)
        print(f"\n  {label}:")
        print(f"    E_before = {vf.energy_before:.6f}")
        print(f"    E_after  = {vf.energy_after:.6f}")
        print(f"    delta_E  = {vf.delta_e:+.6f}")
        print(f"    Bound    = {vf.theoretical_bound:+.6f}")
        print(f"    Within?  = {vf.within_bound}  (margin = {vf.margin:.6f})")
    print()


# ------------------------------------------------------------------
# 6. Life emergence detection
# ------------------------------------------------------------------

def demo_life_emergence() -> None:
    """Detect autopoietic life-like behavior in a simulated TNFR trajectory."""
    print("=" * 72)
    print("6. LIFE EMERGENCE — autopoietic coefficient and vitality")
    print("=" * 72)

    rng = np.random.default_rng(SEED)
    T = 100
    dt = 0.1
    times = [i * dt for i in range(T)]

    # Simulate ‖EPI‖ trajectory with logistic-like growth
    gamma = 0.5      # autopoietic strength
    epi_max = 3.0    # carrying capacity
    epsilon = 0.3    # self-feedback strength

    epi = np.zeros(T)
    epi[0] = 0.1

    # External ΔNFR: decaying noise (environment becomes quieter)
    dnfr_ext = 0.5 * np.exp(-np.linspace(0, 3, T)) * (1 + 0.2 * rng.standard_normal(T))
    d_dnfr_ext_dt = np.gradient(dnfr_ext, dt)

    for i in range(T - 1):
        G_epi = gamma * epi[i] * (1 - epi[i] / epi_max)
        epi[i + 1] = epi[i] + dt * (G_epi + dnfr_ext[i])
        epi[i + 1] = max(0, epi[i + 1])

    dEPI_dt = np.gradient(epi, dt)

    # Detect life emergence
    telem = detect_life_emergence(
        times=times,
        epi_series=epi,
        dEPI_dt=dEPI_dt,
        dnfr_external=dnfr_ext,
        d_dnfr_external_dt=d_dnfr_ext_dt,
        epsilon=epsilon,
        gamma=gamma,
        epi_max=epi_max,
    )

    print(f"\n  Simulation: T={T} steps, dt={dt}")
    print(f"  Parameters: gamma={gamma}, epi_max={epi_max}, epsilon={epsilon}")

    if telem.life_threshold_time is not None:
        print(f"  Life emergence at t = {telem.life_threshold_time:.3f}  (A > 1)")
    else:
        print(f"  No life emergence detected (A never exceeded 1)")

    # Show milestones
    milestones = [0, 10, 25, 50, 75, T - 1]
    print(f"\n  {'Step':>6s}  {'t':>6s}  {'||EPI||':>8s}  {'A(t)':>8s}  {'Vi(t)':>8s}  {'M(t)':>8s}")
    print(f"  {'─' * 6}  {'─' * 6}  {'─' * 8}  {'─' * 8}  {'─' * 8}  {'─' * 8}")
    for m in milestones:
        print(f"  {m:6d}  {times[m]:6.2f}  {epi[m]:8.4f}  "
              f"{float(telem.autopoietic_coefficient[m]):8.4f}  "
              f"{float(telem.vitality_index[m]):8.4f}  "
              f"{float(telem.stability_margin[m]):8.4f}")

    # Stability margin interpretation
    M_final = float(telem.stability_margin[-1])
    if M_final > 0:
        interpretation = "above carrying capacity midpoint (potential saturation)"
    elif M_final > -0.25:
        interpretation = "growth regime (below midpoint)"
    else:
        interpretation = "early development (far from saturation)"
    print(f"\n  Final stability margin M = {M_final:.4f}: {interpretation}")
    print()


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> None:
    print()
    print("TNFR LYAPUNOV STABILITY & STRUCTURAL LIFECYCLE")
    print("Grammar U2 guarantees net-contractive energy evolution.")
    print("E[G] = 1/2 * Sum_i [Phi_s^2 + |grad_phi|^2 + K_phi^2 + ...]")
    print()

    demo_operator_bounds()
    demo_sequence_proof()
    demo_spectral_gap()
    demo_operator_convergence()
    demo_empirical_verification()
    demo_life_emergence()

    print("=" * 72)
    print("CONCLUSION: Grammar U2 ensures Lyapunov stability.")
    print("Every destabiliser is compensated by a stabiliser,")
    print("guaranteeing the product of energy multipliers <= 1.")
    print("See: theory/STRUCTURAL_STABILITY_AND_DYNAMICS.md")
    print("=" * 72)


if __name__ == "__main__":
    main()
