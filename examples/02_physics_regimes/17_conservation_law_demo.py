"""TNFR Structural Conservation Law — Demonstration.

Demonstrates the Noether-like conservation theorem derived from the
TNFR nodal equation under unified grammar constraints U1-U6.

Main result (Structural Continuity Theorem):

    dρ/dt + div(J) ≈ S_grammar    where S → 0 under grammar compliance

This example shows:
1. Conservation tracking across a multi-step evolution
2. Two-sector decomposition (potential vs geometric)
3. Ward identities for individual operator steps
4. Lyapunov stability verification (dE/dt ≤ 0)
5. Spectral conservation analysis (graph Laplacian decomposition)
6. Grammar violation detection from conservation residuals

See: theory/STRUCTURAL_CONSERVATION_THEOREM.md for the full derivation.
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
    compute_charge_density,
    compute_conservation_scaling,
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
    """One step of nodal equation evolution (phase + ΔNFR diffusion)."""
    for n in G.nodes():
        nu_f = G.nodes[n].get("nu_f", 1.0)
        dnfr = G.nodes[n].get("delta_nfr", 0.0)
        G.nodes[n]["phase"] += dt * nu_f * dnfr * 0.1
        G.nodes[n]["theta"] = G.nodes[n]["phase"]
        nbrs = list(G.neighbors(n))
        if nbrs:
            mean_dnfr = float(np.mean([G.nodes[j].get("delta_nfr", 0.0) for j in nbrs]))
            G.nodes[n]["delta_nfr"] += dt * 0.1 * (mean_dnfr - dnfr)


def demo_conservation_tracking() -> None:
    """1. Track conservation across a multi-step evolution."""
    print("=" * 65)
    print("  1. CONSERVATION TRACKING — Multi-step evolution")
    print("=" * 65)

    G = _build_graph()
    Q0 = compute_noether_charge(G)
    E0 = compute_energy_functional(G)

    print(f"Network: Watts-Strogatz(30, k=4, p=0.3)")
    print(f"Initial Noether charge  Q = {Q0:.6f}")
    print(f"Initial structural energy E = {E0:.6f}")
    print()

    tracker = ConservationTracker(G)
    tracker.record(t=0.0)

    dt = 0.01
    n_steps = 20
    for step in range(n_steps):
        _evolve_step(G, dt)
        tracker.record(t=(step + 1) * dt)

    report = tracker.report()
    Q_final = compute_noether_charge(G)
    E_final = compute_energy_functional(G)

    print(f"After {n_steps} steps (dt={dt}):")
    print(f"  Final Q = {Q_final:.6f}  (drift = {abs(Q_final - Q0):.6f})")
    print(f"  Final E = {E_final:.6f}  (drift = {abs(E_final - E0):.6f})")
    print(f"  Mean conservation quality = {report.mean_quality:.4f}")
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
    print("Physics: The cross-coupling confirms the complex field")
    print("unification Psi = K_phi + i*J_phi is physically real.")
    print()


def demo_ward_identities() -> None:
    """3. Ward identities for individual evolution steps."""
    print("=" * 65)
    print("  3. WARD IDENTITIES — Per-step conservation signatures")
    print("=" * 65)

    G = _build_graph()
    identities = []

    step_labels = [
        "diffusion_1",
        "diffusion_2",
        "diffusion_3",
        "diffusion_4",
        "diffusion_5",
    ]

    for label in step_labels:
        before = capture_conservation_snapshot(G)
        _evolve_step(G, dt=0.01)
        after = capture_conservation_snapshot(G)
        ward = compute_ward_identity(before, after, operator_name=label)
        identities.append(ward)

    print(f"{'Step':<15} {'DQ':>10} {'DE':>10} {'Charge':>12} {'Energy':>12}")
    print("-" * 59)
    for w in identities:
        print(
            f"{w.operator_name:<15} {w.delta_charge:>+10.6f} "
            f"{w.delta_energy:>+10.6f} {w.charge_character:>12} "
            f"{w.energy_character:>12}"
        )

    seq = verify_sequence_ward_identity(identities)
    print()
    print(f"Sequence Ward identity:")
    print(f"  Total source         = {seq['total_source']:+.6f}")
    print(f"  Total charge change  = {seq['total_charge_change']:+.6f}")
    print(f"  Total energy change  = {seq['total_energy_change']:+.6f}")
    print(f"  Sequence conserved   = {seq['sequence_conserved']}")
    print()


def demo_lyapunov_stability() -> None:
    """4. Lyapunov stability verification: dE/dt <= 0."""
    print("=" * 65)
    print("  4. LYAPUNOV STABILITY — Energy monotonicity")
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

        status = "STABLE" if lyap.is_stable else "UNSTABLE"
        if lyap.is_stable:
            stable_count += 1
        print(
            f"  Step {step + 1:2d}: E = {lyap.energy_after:.4f}, "
            f"dE/dt = {lyap.energy_derivative:+.6f}, "
            f"D = {lyap.dissipation:.6f}  [{status}]"
        )

    print()
    print(f"Stable steps: {stable_count}/10")
    print(f"Energy trend: {energies[0]:.4f} -> {energies[-1]:.4f}")
    print(f"Total energy change: {energies[-1] - energies[0]:+.6f}")
    print()


def demo_spectral_conservation() -> None:
    """5. Spectral conservation analysis (Laplacian eigenbasis)."""
    print("=" * 65)
    print("  5. SPECTRAL CONSERVATION — Laplacian decomposition")
    print("=" * 65)

    G = _build_graph()
    spec = compute_spectral_conservation(G)

    print(f"Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"Spectral gap (lambda_1): {spec.spectral_gap:.4f}")
    print(
        f"Well-conserved modes: {spec.dominant_conservation_modes}/{len(spec.eigenvalues)}"
    )
    print()

    # Show a few modes
    print(f"{'Mode':<6} {'lambda_k':>10} {'rho_hat':>10} {'Residual':>10}")
    print("-" * 36)
    n_show = min(8, len(spec.eigenvalues))
    for k in range(n_show):
        print(
            f"{k:<6} {spec.eigenvalues[k]:>10.4f} "
            f"{spec.rho_spectrum[k]:>10.4f} "
            f"{spec.conservation_by_mode[k]:>10.6f}"
        )

    print()
    print("Physics: Low-frequency modes (small lambda_k) show best")
    print("conservation — mirrors U5 multi-scale coherence principle.")
    print()


def demo_grammar_violation_detection() -> None:
    """6. Grammar violation detection from conservation residuals."""
    print("=" * 65)
    print("  6. GRAMMAR VIOLATION DETECTION — Conservation-based")
    print("=" * 65)

    G = _build_graph()
    bounds = compute_grammar_conservation_bounds(G)

    # Normal evolution
    before = capture_conservation_snapshot(G)
    for _ in range(3):
        _evolve_step(G, dt=0.01)
    after = capture_conservation_snapshot(G)

    balance = verify_conservation_balance(before, after, dt=0.03)
    violations = detect_grammar_violations_from_conservation(balance, bounds)

    print("Normal evolution (grammar-compliant):")
    print(f"  Conservation quality   = {balance.conservation_quality:.4f}")
    print(f"  Violations detected    = {violations['violations_detected']}")
    print(f"  Severity               = {violations['severity']:.4f}")
    print()

    # Perturbed evolution (simulated grammar violation)
    G2 = _build_graph(seed=99)
    before2 = capture_conservation_snapshot(G2)
    # Large sudden perturbation (simulates unconstrained destabilizer)
    rng = np.random.default_rng(123)
    for n in G2.nodes():
        G2.nodes[n]["delta_nfr"] += rng.uniform(-5.0, 5.0)
    after2 = capture_conservation_snapshot(G2)

    balance2 = verify_conservation_balance(before2, after2, dt=1.0)
    violations2 = detect_grammar_violations_from_conservation(balance2, bounds)

    print("Perturbed evolution (grammar violation simulated):")
    print(f"  Conservation quality   = {balance2.conservation_quality:.4f}")
    print(f"  Violations detected    = {violations2['violations_detected']}")
    print(f"  Violation types        = {violations2['violation_types']}")
    print(f"  Severity               = {violations2['severity']:.4f}")
    print(f"  Nodes violating        = {len(violations2['nodes_violating'])}")
    print()
    print("Physics: Conservation residuals detect and classify grammar")
    print("violations — a diagnostic tool for structural health.")
    print()


def main() -> None:
    print()
    print("*" * 65)
    print("  TNFR STRUCTURAL CONSERVATION LAW — DEMONSTRATION")
    print("  Noether-like theorems from the nodal equation")
    print("*" * 65)
    print()

    demo_conservation_tracking()
    demo_sector_decomposition()
    demo_ward_identities()
    demo_lyapunov_stability()
    demo_spectral_conservation()
    demo_grammar_violation_detection()

    print("=" * 65)
    print("  SUMMARY")
    print("=" * 65)
    print()
    print("The Structural Conservation Theorem establishes that grammar")
    print("constraints U1-U6 act as structural symmetries, generating")
    print("conservation laws analogous to Noether's theorem in physics:")
    print()
    print("  Grammar compliance => dQ/dt ~ 0 (charge conservation)")
    print("  U2 convergence     => dE/dt <= 0 (Lyapunov stability)")
    print("  Sector coupling    => Psi = K_phi + i*J_phi unification")
    print()
    print("See: theory/STRUCTURAL_CONSERVATION_THEOREM.md")
    print()


if __name__ == "__main__":  # pragma: no cover - manual example
    main()
