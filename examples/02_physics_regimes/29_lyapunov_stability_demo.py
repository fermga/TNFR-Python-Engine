"""TNFR Lyapunov policy diagnostics and structural lifecycle.

Demonstrates the registered U2-role multipliers, finite trajectory checks and
the exact spectral result available for restricted diffusion. The nominal
multiplier product is a policy diagnostic; U2 alone does not prove that the
structural energy decreases for every operator realization.

Key results shown:
1. Per-operator U2 policy multipliers for all 13 operators
2. Sequence multiplier diagnostic under the declared policy model
3. Separate combinatorial and normalized diffusion gaps
4. Side-by-side policy-step and pure-EPI continuous-time scales
5. Measured five-field energy compared with the nominal policy score
6. Life emergence detection: autopoietic coefficient and vitality index

See: theory/STRUCTURAL_STABILITY_AND_DYNAMICS.md for the full treatment.
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import networkx as nx
import numpy as np

from tnfr.constants import inject_defaults
from tnfr.physics.life import (
    compute_autopoietic_coefficient,
    compute_self_generation,
    compute_stability_margin,
    detect_life_emergence,
)
from tnfr.physics.lyapunov import (
    OPERATOR_POLICY_MULTIPLIERS,
    U2PolicyRole,
    analyze_operator_policy_context,
    analyze_spectral_gap,
    compare_operator_energy_to_policy,
    evaluate_sequence_policy,
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
        G.nodes[node]["EPI"] = float(rng.uniform(0.5, 2.0))
        G.nodes[node]["nu_f"] = float(rng.uniform(0.5, 2.0))
        G.nodes[node]["phase"] = float(rng.uniform(0, 2 * np.pi))
        G.nodes[node]["delta_nfr"] = float(rng.uniform(-0.3, 0.3))
    return G


# ------------------------------------------------------------------
# 1. Per-operator U2 policy multipliers — the full registry
# ------------------------------------------------------------------


def demo_operator_bounds() -> None:
    """Display the policy multipliers for all 13 canonical operators."""
    print("=" * 72)
    print("1. PER-OPERATOR U2 POLICY MULTIPLIERS — all 13 operators")
    print("=" * 72)

    # Group by energy class
    by_class: dict[U2PolicyRole, list] = {c: [] for c in U2PolicyRole}
    for name, bound in OPERATOR_POLICY_MULTIPLIERS.items():
        by_class[bound.energy_class].append(bound)

    for cls in [
        U2PolicyRole.STABILISER,
        U2PolicyRole.DESTABILISER,
        U2PolicyRole.NEUTRAL,
        U2PolicyRole.MIXED,
    ]:
        ops = by_class[cls]
        if not ops:
            continue
        print(f"\n  {cls.value.upper()} U2 role:")
        print(f"  {'Name':20s}  {'Glyph':6s}  {'Multiplier':>10s}  Factor")
        print(f"  {'─' * 20}  {'─' * 6}  {'─' * 10}  {'─' * 20}")
        for b in sorted(ops, key=lambda x: x.contraction_rate, reverse=True):
            print(
                f"  {b.operator_name:20s}  {b.glyph:6s}  "
                f"{b.policy_multiplier:10.4f}  "
                f"{b.glyph_factor_name}={b.glyph_factor_value:.4f}"
            )
    print("\n  Multipliers are bookkeeping values, not measured-energy bounds.")
    print()


# ------------------------------------------------------------------
# 2. Sequence multiplier diagnostic — grammar-compliant vs non-compliant
# ------------------------------------------------------------------


def demo_sequence_policy() -> None:
    """Evaluate nominal multipliers without promoting them to a proof."""
    print("=" * 72)
    print("2. SEQUENCE MULTIPLIER DIAGNOSTIC — policy model only")
    print("=" * 72)

    sequences = {
        "Bootstrap fragment": ["Emission", "Coupling", "Coherence"],
        "Explore fragment": ["Dissonance", "Mutation", "Coherence"],
        "Stabilize fragment": ["Coherence", "Silence"],
        "Propagate fragment": ["Resonance", "Coupling"],
        "Closed sample word": [
            "Emission",
            "Coupling",
            "Coherence",
            "Dissonance",
            "Mutation",
            "Coherence",
            "Resonance",
            "Coupling",
            "Coherence",
            "Silence",
        ],
        "Unbalanced sample: OZ without IL": ["Dissonance", "Silence"],
        "Unbalanced sample: OZ and VAL": [
            "Dissonance",
            "Expansion",
            "Silence",
        ],
    }

    for label, seq in sequences.items():
        result = evaluate_sequence_policy(seq)
        status = "<= 1" if result.policy_product_at_most_one else "> 1"
        print(f"\n  {label}")
        print(f"    Operators:    {' → '.join(result.operators)}")
        print(
            f"    Multipliers:  {' × '.join(f'{m:.4f}' for m in result.policy_multipliers)}"
        )
        print(f"    Policy product: {result.cumulative_product:.6f} ({status})")
    print("\n  This calculation neither validates grammar nor predicts energy change.")
    print()


# ------------------------------------------------------------------
# 3. Spectral gap analysis
# ------------------------------------------------------------------


def demo_spectral_gap() -> None:
    """Compare combinatorial and normalized gaps across topologies."""
    print("=" * 72)
    print("3. SPECTRAL READ-OUTS — combinatorial vs pure-EPI diffusion gap")
    print("=" * 72)

    topologies = {
        "Ring (20)": nx.cycle_graph(20),
        "Watts-Strogatz (20, k=4, p=0.3)": nx.watts_strogatz_graph(
            20, 4, 0.3, seed=SEED
        ),
        "Barabasi-Albert (20, m=2)": nx.barabasi_albert_graph(20, 2, seed=SEED),
        "Complete (10)": nx.complete_graph(10),
        "Star (20)": nx.star_graph(19),
    }

    print(
        f"\n  {'Topology':36s} {'λ_comb':>9s} {'λ_rw':>9s} "
        f"{'τ_rw':>9s} {'log(N)/λ':>10s}"
    )
    print(f"  {'─' * 36} {'─' * 9} {'─' * 9} {'─' * 9} {'─' * 10}")

    for label, G in topologies.items():
        # Inject defaults for TNFR attributes
        inject_defaults(G)
        rng = np.random.default_rng(SEED)
        for node in G.nodes():
            G.nodes[node]["EPI"] = float(rng.uniform(0.5, 2.0))
            G.nodes[node]["nu_f"] = float(rng.uniform(0.5, 2.0))
            G.nodes[node]["phase"] = float(rng.uniform(0, 2 * np.pi))
            G.nodes[node]["delta_nfr"] = float(rng.uniform(-0.3, 0.3))

        spec = analyze_spectral_gap(G)
        tau_str = f"{spec.relaxation_time:.4f}" if spec.relaxation_time < 1e6 else "∞"
        mix_str = (
            f"{spec.mixing_time_bound:.4f}" if spec.mixing_time_bound < 1e6 else "∞"
        )
        print(
            f"  {label:36s} {spec.spectral_gap:9.4f} "
            f"{spec.diffusion_gap:9.4f} {tau_str:>9s} {mix_str:>10s}"
        )
    print("\n  τ_rw is per unit homogeneous capacity on fixed pure-EPI diffusion.")
    print("  log(N)/λ is a topology scale, not a universal mixing-time bound.")
    print()


# ------------------------------------------------------------------
# 4. Policy and spectrum — deliberately not combined
# ------------------------------------------------------------------


def demo_policy_and_spectrum() -> None:
    """Show policy-position and continuous-time quantities side by side."""
    print("=" * 72)
    print("4. POLICY AND SPECTRUM — two scopes, no effective-rate formula")
    print("=" * 72)

    G = _build_graph()

    operators = [
        "Coherence",
        "Reception",
        "Coupling",
        "SelfOrganization",
        "Transition",
    ]

    print(
        f"\n  {'Operator':20s} {'U2 role':>13s} {'policy m':>10s} "
        f"{'score half':>11s} {'λ_rw':>9s} {'τ_rw':>9s}"
    )
    print(
        f"  {'─' * 20} {'─' * 13} {'─' * 10} {'─' * 11} "
        f"{'─' * 9} {'─' * 9}"
    )

    for name in operators:
        summary = analyze_operator_policy_context(G, name)
        half_str = (
            f"{summary.policy_half_steps:.2f}"
            if math.isfinite(summary.policy_half_steps)
            else "∞"
        )
        print(
            f"  {name:20s} {summary.operator_bound.policy_role.value:>13s} "
            f"{summary.policy_multiplier:10.4f} {half_str:>11s} "
            f"{summary.spectral.diffusion_gap:9.4f} "
            f"{summary.diffusion_relaxation_time:9.4f}"
        )
    print("\n  Policy m counts operator positions; λ_rw and τ_rw describe a")
    print("  separate homogeneous pure-EPI continuous-time model.")
    print()


# ------------------------------------------------------------------
# 5. Measured energy versus policy-score comparison
# ------------------------------------------------------------------


def demo_empirical_verification() -> None:
    """Compare actual five-field energy changes with the independent policy."""
    print("=" * 72)
    print("5. MEASURED ENERGY VS POLICY SCORE — mismatch is allowed")
    print("=" * 72)

    from tnfr.operators import apply_glyph
    from tnfr.physics.conservation import compute_energy_functional

    glyphs = [
        ("Coherence (IL)", "IL"),
        ("Dissonance (OZ)", "OZ"),
        ("Emission (AL)", "AL"),
    ]

    for label, glyph in glyphs:
        G = _build_graph()
        n = G.number_of_nodes()

        E_before = compute_energy_functional(G)

        # Apply operator to a node
        test_node = list(G.nodes())[0]
        apply_glyph(G, test_node, glyph)

        E_after = compute_energy_functional(G)

        comparison = compare_operator_energy_to_policy(
            glyph, E_before, E_after, n_nodes=n
        )
        print(f"\n  {label}:")
        print(f"    E_before          = {comparison.energy_before:.6f}")
        print(f"    E_after           = {comparison.energy_after:.6f}")
        print(f"    measured delta_E  = {comparison.delta_e:+.6f}")
        print(f"    policy-score delta= {comparison.policy_delta:+.6f}")
        print(f"    observed E ratio  = {comparison.observed_energy_ratio:.6f}")
        print(f"    policy multiplier = {comparison.policy_multiplier:.6f}")
        print(f"    multiplier residual = {comparison.multiplier_residual:+.6f}")
        print(f"    one-sided screen  = {comparison.policy_screen_passed}")
    print("\n  A failed screen is a model mismatch, not an operator-contract failure;")
    print("  a passed screen is not a Lyapunov certificate.")
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
    gamma = 0.5  # autopoietic strength
    epi_max = 3.0  # carrying capacity
    epsilon = 0.3  # self-feedback strength

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
    print(
        f"\n  {'Step':>6s}  {'t':>6s}  {'||EPI||':>8s}  {'A(t)':>8s}  {'Vi(t)':>8s}  {'M(t)':>8s}"
    )
    print(f"  {'─' * 6}  {'─' * 6}  {'─' * 8}  {'─' * 8}  {'─' * 8}  {'─' * 8}")
    for m in milestones:
        print(
            f"  {m:6d}  {times[m]:6.2f}  {epi[m]:8.4f}  "
            f"{float(telem.autopoietic_coefficient[m]):8.4f}  "
            f"{float(telem.vitality_index[m]):8.4f}  "
            f"{float(telem.stability_margin[m]):8.4f}"
        )

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
    print("TNFR LYAPUNOV POLICY DIAGNOSTICS & STRUCTURAL LIFECYCLE")
    print("U2 role balance and measured energy change are distinct checks.")
    print("E[G] = 1/2 * Sum_i [Phi_s^2 + |grad_phi|^2 + K_phi^2 + ...]")
    print()

    demo_operator_bounds()
    demo_sequence_policy()
    demo_spectral_gap()
    demo_policy_and_spectrum()
    demo_empirical_verification()
    demo_life_emergence()

    print("=" * 72)
    print("CONCLUSION: U2 constrains composition; it is not by itself a")
    print("global Lyapunov theorem. Policy multipliers organize U2 bookkeeping,")
    print("while exact stability needs a specified model and state functional.")
    print("See: theory/STRUCTURAL_STABILITY_AND_DYNAMICS.md")
    print("=" * 72)


if __name__ == "__main__":
    main()
