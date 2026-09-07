"""TNFR structural-balance diagnostics — finite demonstration.

Evaluates the implemented Noether-like balance candidate on a declared
trajectory. It does not turn grammar compliance into a general conservation or
Lyapunov theorem.

Balance candidate evaluated here:

    dρ/dt + div(J) = S_residual

The trajectory below follows an explicitly declared auxiliary smoothing rule.
It does not execute a U1-U6 word, so its residual cannot establish that grammar
compliance implies conservation.

This example shows:
1. Structural-balance tracking across a multi-step evolution
2. Two-sector decomposition (potential vs geometric)
3. Ward-style diagnostics for auxiliary evolution steps
4. Finite-step energy-derivative measurement
5. Static spectral activity analysis (normalized-Laplacian decomposition)
6. Balance-alert screening on baseline and perturbed trajectories

See: theory/STRUCTURAL_CONSERVATION_THEOREM.md for the construction and scope.
"""

from __future__ import annotations

import math

import networkx as nx
import numpy as np

from tnfr.constants import inject_defaults
from tnfr.physics.conservation import (
    ConservationTracker,
    analyze_sector_coupling,
    capture_conservation_snapshot,
    compute_energy_functional,
    compute_grammar_conservation_bounds,
    compute_lyapunov_derivative,
    compute_noether_charge,
    compute_spectral_conservation,
    compute_ward_identity,
    detect_grammar_violations_from_conservation,
    verify_conservation_balance,
    verify_sequence_ward_identity,
)


def _build_graph(n: int = 30, seed: int = 42) -> nx.Graph:
    """Build a Watts-Strogatz TNFR network with canonical attributes."""
    rng = np.random.default_rng(seed)
    G = nx.watts_strogatz_graph(n, 4, 0.3, seed=seed)
    inject_defaults(G)
    for node in G.nodes():
        G.nodes[node]["phase"] = rng.uniform(0, 2 * math.pi)
        G.nodes[node]["theta"] = G.nodes[node]["phase"]
        G.nodes[node]["frequency"] = rng.uniform(0.1, 1.0)
        G.nodes[node]["nu_f"] = G.nodes[node]["frequency"]
        G.nodes[node]["delta_nfr"] = rng.uniform(-0.5, 0.5)
        G.nodes[node]["EPI"] = rng.uniform(0.5, 2.0)
    return G


def _evolve_step(G: nx.Graph, dt: float = 0.01) -> None:
    """Apply the example's auxiliary phase/ΔNFR smoothing rule.

    This is a declared numerical model, not a canonical operator application
    and not evidence that the resulting trajectory is grammar-compliant.
    """
    for n in G.nodes():
        nu_f = G.nodes[n].get("nu_f", 1.0)
        dnfr = G.nodes[n].get("delta_nfr", 0.0)
        G.nodes[n]["phase"] += dt * nu_f * dnfr * 0.1
        G.nodes[n]["theta"] = G.nodes[n]["phase"]
        nbrs = list(G.neighbors(n))
        if nbrs:
            mean_dnfr = float(np.mean([G.nodes[j].get("delta_nfr", 0.0) for j in nbrs]))
            G.nodes[n]["delta_nfr"] += dt * 0.1 * (mean_dnfr - dnfr)


def demo_balance_tracking() -> None:
    """1. Track finite structural-balance diagnostics."""
    print("=" * 65)
    print("  1. STRUCTURAL-BALANCE TRACKING — Multi-step evolution")
    print("=" * 65)

    G = _build_graph()
    Q0 = compute_noether_charge(G)
    E0 = compute_energy_functional(G)

    print(f"Network: Watts-Strogatz(30, k=4, p=0.3)")
    print(f"Initial charge candidate Q = {Q0:.6f}")
    print(f"Initial energy candidate E = {E0:.6f}")
    print()

    tracker = ConservationTracker(G)
    tracker.record(t=0.0)

    dt = 0.01
    n_steps = 20
    for step in range(n_steps):
        _evolve_step(G, dt)
        tracker.record(t=(step + 1) * dt)

    report = tracker.report()
    mean_sampled_quality = report.sampled_mean_quality
    Q_final = compute_noether_charge(G)
    E_final = compute_energy_functional(G)

    print(f"After {n_steps} steps (dt={dt}):")
    print(f"  Final Q = {Q_final:.6f}  (drift = {abs(Q_final - Q0):.6f})")
    print(f"  Final E = {E_final:.6f}  (drift = {abs(E_final - E0):.6f})")
    print(f"  Mean sampled balance quality = {mean_sampled_quality:.4f}")
    print(
        f"  Charge relative drift     = {abs(Q_final - Q0) / max(abs(Q0), 1e-15):.2e}"
    )
    print()


def demo_sector_decomposition() -> None:
    """2. Two-sector decomposition (potential vs geometric)."""
    print("=" * 65)
    print("  2. TWO-SECTOR DECOMPOSITION — Potential vs Geometric")
    print("=" * 65)

    G = _build_graph()
    before = capture_conservation_snapshot(G)

    # Evolve a few steps
    for _ in range(5):
        _evolve_step(G, dt=0.01)
    after = capture_conservation_snapshot(G)

    coupling = analyze_sector_coupling(before, after, dt=0.05)

    print(f"Potential sector RMS residual: {coupling['potential_sector_residual']:.6f}")
    print(f"Geometric sector RMS residual: {coupling['geometric_sector_residual']:.6f}")
    print(f"Cross-coupling correlation:    {coupling['cross_coupling_strength']:.4f}")
    print(f"Dominant sector:               {coupling['dominant_sector']}")
    print(f"Sector asymmetry:              {coupling['sector_asymmetry']:.3f}")
    print()
    print("Interpretation: this correlation is a finite diagnostic of the")
    print("declared trajectory. It does not prove conjugacy or a causal coupling;")
    print("Psi = K_phi + i*J_phi is the algebraic field definition.")
    print()


def demo_ward_identities() -> None:
    """3. Ward-style diagnostics for individual auxiliary steps."""
    print("=" * 65)
    print("  3. WARD-STYLE DIAGNOSTICS — Per-step balance signatures")
    print("=" * 65)

    G = _build_graph()
    identities = []

    step_labels = [
        "aux_step_1",
        "aux_step_2",
        "aux_step_3",
        "aux_step_4",
        "aux_step_5",
    ]

    for label in step_labels:
        before = capture_conservation_snapshot(G)
        _evolve_step(G, dt=0.01)
        after = capture_conservation_snapshot(G)
        ward = compute_ward_identity(before, after, operator_name=label)
        identities.append(ward)

    print(
        f"{'Step':<15} {'DQ':>10} {'DE':>10} "
        f"{'Legacy Q class':>16} {'Legacy E class':>16}"
    )
    print("-" * 71)
    for w in identities:
        print(
            f"{w.operator_name:<15} {w.delta_charge:>+10.6f} "
            f"{w.delta_energy:>+10.6f} {w.charge_character:>16} "
            f"{w.energy_character:>16}"
        )

    print("  Legacy class names are finite threshold labels, not exact laws.")

    seq = verify_sequence_ward_identity(identities)
    print()
    print("Sequence diagnostic:")
    print(f"  Total source         = {seq['total_source']:+.6f}")
    print(f"  Total charge change  = {seq['total_charge_change']:+.6f}")
    print(f"  Total energy change  = {seq['total_energy_change']:+.6f}")
    print(f"  Legacy aggregate alert passed = {seq['sequence_conserved']}")
    print("  This threshold result neither validates grammar nor proves conservation.")
    print()


def demo_candidate_energy_trend() -> None:
    """4. Measure the candidate-energy derivative on one trajectory."""
    print("=" * 65)
    print("  4. LYAPUNOV-CANDIDATE DIAGNOSTIC — sampled energy change")
    print("=" * 65)

    G = _build_graph()
    energies = [compute_energy_functional(G)]
    stable_count = 0

    for step in range(10):
        before = capture_conservation_snapshot(G)
        _evolve_step(G, dt=0.01)
        after = capture_conservation_snapshot(G)

        lyap = compute_lyapunov_derivative(before, after, dt=0.01)
        energies.append(lyap.energy_after)

        status = "DESCENT" if lyap.is_stable else "INCREASE"
        if lyap.is_stable:
            stable_count += 1
        print(
            f"  Step {step + 1:2d}: E = {lyap.energy_after:.4f}, "
            f"dE/dt = {lyap.energy_derivative:+.6f}, "
            f"D = {lyap.dissipation:.6f}  [{status}]"
        )

    print()
    print(f"Non-increasing sampled steps: {stable_count}/10")
    print(f"Energy trend: {energies[0]:.4f} -> {energies[-1]:.4f}")
    print(f"Total energy change: {energies[-1] - energies[0]:+.6f}")
    print()


def demo_spectral_activity() -> None:
    """5. Static spectral activity analysis (Laplacian eigenbasis)."""
    print("=" * 65)
    print("  5. STATIC SPECTRAL ACTIVITY — Laplacian decomposition")
    print("=" * 65)

    G = _build_graph()
    spec = compute_spectral_conservation(G)

    print(f"Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"Spectral gap (lambda_1): {spec.spectral_gap:.4f}")
    print(
        f"Below-median divergence-activity modes: "
        f"{spec.low_divergence_activity_modes}/{len(spec.eigenvalues)}"
    )
    print()

    # Show a few modes
    print(f"{'Mode':<6} {'lambda_k':>10} {'rho_hat':>10} {'Activity':>10}")
    print("-" * 36)
    n_show = min(8, len(spec.eigenvalues))
    for k in range(n_show):
        print(
            f"{k:<6} {spec.eigenvalues[k]:>10.4f} "
            f"{spec.rho_spectrum[k]:>10.4f} "
            f"{spec.modal_divergence_magnitude[k]:>10.6f}"
        )

    print()
    print("Interpretation: the table reports a static activity proxy in this")
    print("graph's normalized-Laplacian basis. Without d(rho_hat)/dt it is not")
    print("a modal balance residual and cannot verify U5 or conservation.")
    print()


def demo_balance_alerts() -> None:
    """6. Compare finite-balance alerts on two labelled fixtures."""
    print("=" * 65)
    print("  6. BALANCE ALERTS — Baseline vs perturbation")
    print("=" * 65)

    G = _build_graph()
    bounds = compute_grammar_conservation_bounds(G)

    # Baseline auxiliary evolution; no canonical operator history is executed.
    before = capture_conservation_snapshot(G)
    for _ in range(3):
        _evolve_step(G, dt=0.01)
    after = capture_conservation_snapshot(G)

    balance = verify_conservation_balance(before, after, dt=0.03)
    alerts = detect_grammar_violations_from_conservation(balance, bounds)

    print("Baseline auxiliary smoothing (grammar status not asserted):")
    print(f"  Balance quality        = {balance.conservation_quality:.4f}")
    print(f"  Balance alerts         = {alerts['alerts_detected']}")
    print(f"  Alert types            = {alerts['alert_types']}")
    print(f"  Grammar validated      = {alerts['grammar_validated']}")
    print(f"  Alert severity         = {alerts['severity']:.4f}")
    print()

    # Deliberately perturbed comparison fixture.
    G2 = _build_graph(seed=99)
    before2 = capture_conservation_snapshot(G2)
    # Large sudden perturbation, with no canonical operator label.
    rng = np.random.default_rng(123)
    for n in G2.nodes():
        G2.nodes[n]["delta_nfr"] += rng.uniform(-5.0, 5.0)
    after2 = capture_conservation_snapshot(G2)

    balance2 = verify_conservation_balance(before2, after2, dt=1.0)
    alerts2 = detect_grammar_violations_from_conservation(balance2, bounds)

    print("Unlabelled large perturbation:")
    print(f"  Balance quality        = {balance2.conservation_quality:.4f}")
    print(f"  Balance alerts         = {alerts2['alerts_detected']}")
    print(f"  Alert types            = {alerts2['alert_types']}")
    print(f"  Grammar validated      = {alerts2['grammar_validated']}")
    print(f"  Alert severity         = {alerts2['severity']:.4f}")
    print(f"  Nodes alerted          = {len(alerts2['nodes_alerted'])}")
    print()
    print("Interpretation: the legacy-named routine reports balance alerts only.")
    print("Without validated operator history and live state, it cannot decide")
    print("whether a U1-U6 rule was violated or satisfied.")
    print()


def main() -> None:
    print()
    print("*" * 65)
    print("  TNFR STRUCTURAL-BALANCE DIAGNOSTICS — FINITE DEMONSTRATION")
    print("  Noether-like balance diagnostics on an auxiliary trajectory")
    print("*" * 65)
    print()

    demo_balance_tracking()
    demo_sector_decomposition()
    demo_ward_identities()
    demo_candidate_energy_trend()
    demo_spectral_activity()
    demo_balance_alerts()

    print("=" * 65)
    print("  SUMMARY")
    print("=" * 65)
    print()
    print("This run evaluates the implemented structural-balance residual and")
    print("candidate energy on one reproducible trajectory:")
    print()
    print("  Charge residual    -> measured balance quality")
    print("  Energy derivative  -> measured descent/increase per sampled step")
    print("  Sector correlation -> measured association between diagnostics")
    print()
    print("None of these observations proves a result for arbitrary U1-U6 words.")
    print("See: theory/STRUCTURAL_CONSERVATION_THEOREM.md")
    print()


if __name__ == "__main__":  # pragma: no cover - manual example
    main()
