"""Example 36: heuristic alerts from structural-balance residuals.

Compares a baseline with three synthetic perturbations and reports finite
balance telemetry. The public detector retains historical grammar-shaped keys for
compatibility, but those labels are heuristic hypotheses: residuals also depend
on the pressure law, topology, timestep and discretization. They neither imply
nor are implied by grammar validity.

U6 in particular observes mean absolute Phi_s drift between declared fields.
The Noether-like charge drift printed here is a different quantity and cannot
certify U6. Validate operator words with the grammar validator and phase/U6
contracts with their dedicated checks.
"""

from __future__ import annotations

import math
import os
import sys

import networkx as nx
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from tnfr.constants import inject_defaults
from tnfr.physics.conservation import (
    ConservationTracker,
    capture_conservation_snapshot,
    compute_noether_charge,
    detect_grammar_violations_from_conservation,
    verify_conservation_balance,
)
from tnfr.physics.fields import (
    compute_phase_curvature,
    compute_phase_gradient,
    compute_structural_potential,
)


def _build_graph(n: int = 40, seed: int = 42) -> nx.Graph:
    """Build a standard WS graph with TNFR defaults."""
    rng = np.random.default_rng(seed)
    G = nx.watts_strogatz_graph(n, 4, 0.3, seed=seed)
    inject_defaults(G)
    for node in G.nodes():
        G.nodes[node]["phase"] = rng.uniform(0, 2 * math.pi)
        G.nodes[node]["theta"] = G.nodes[node]["phase"]
        G.nodes[node]["delta_nfr"] = rng.uniform(-0.3, 0.3)
        G.nodes[node]["nu_f"] = rng.uniform(0.8, 1.2)
    return G


def _evolve_compliant(G: nx.Graph, dt: float = 0.05) -> None:
    """Grammar-compliant IL-like diffusion step."""
    for n in G.nodes():
        neighbors = list(G.neighbors(n))
        if neighbors:
            mean_phase = np.mean([G.nodes[nb]["phase"] for nb in neighbors])
            G.nodes[n]["phase"] += dt * (mean_phase - G.nodes[n]["phase"])
            G.nodes[n]["theta"] = G.nodes[n]["phase"]
            mean_dnfr = np.mean([G.nodes[nb]["delta_nfr"] for nb in neighbors])
            G.nodes[n]["delta_nfr"] += dt * (mean_dnfr - G.nodes[n]["delta_nfr"])


def _inject_pressure_amplification(G: nx.Graph) -> None:
    """Apply a synthetic global pressure amplification."""
    for n in G.nodes():
        G.nodes[n]["delta_nfr"] *= 5.0  # Amplify without stabilizer


def _inject_antiphase_state(G: nx.Graph) -> None:
    """Set an antiphase state without claiming a coupling event occurred."""
    nodes = sorted(G.nodes())
    for i, n in enumerate(nodes):
        if i % 2 == 0:
            G.nodes[n]["phase"] = 0.0
        else:
            G.nodes[n]["phase"] = math.pi  # Antiphase


def _inject_local_pressure_spike(G: nx.Graph) -> None:
    """Apply a synthetic localized pressure spike at a hub."""
    # Inject extreme DELTA_NFR at highest-degree node
    hub = max(G.nodes(), key=lambda n: G.degree(n))
    G.nodes[hub]["delta_nfr"] = 20.0
    for nb in G.neighbors(hub):
        G.nodes[nb]["delta_nfr"] = 10.0


# ---------------------------------------------------------------------------
# 1. Baseline: grammar-compliant evolution
# ---------------------------------------------------------------------------


def demo_baseline() -> dict:
    """Measure conservation residuals under compliant evolution."""
    print("=" * 65)
    print("  1. BASELINE — Grammar-Compliant Evolution")
    print("=" * 65)

    G = _build_graph()
    snap_before = capture_conservation_snapshot(G)

    for _ in range(10):
        _evolve_compliant(G)

    snap_after = capture_conservation_snapshot(G)
    balance = verify_conservation_balance(snap_before, snap_after, dt=10 * 0.05)
    violations = detect_grammar_violations_from_conservation(balance)

    print(f"\n  WS (N=40), 10 IL-like steps")
    print(f"    Conservation quality: {balance.conservation_quality:.4f}")
    print(f"    RMS residual:        {balance.rms_residual:.6f}")
    print(f"    Max residual:        {balance.max_residual:.6f}")
    print(f"    Charge drift:        {balance.charge_drift:.6f}")
    print(f"    GVI:                 {balance.grammar_violation_index:.6f}")
    print(f"    Balance alert:       {violations['alerts_detected']}")
    print(f"    Alert types:         {violations['alert_types']}")
    print(f"    Severity:            {violations['severity']:.4f}")
    print("\n  Baseline reference only; no grammar verdict follows from residual size.")

    return {
        "quality": balance.conservation_quality,
        "rms": balance.rms_residual,
        "max": balance.max_residual,
        "drift": balance.charge_drift,
    }


# ---------------------------------------------------------------------------
# 2. Global pressure amplification
# ---------------------------------------------------------------------------


def demo_pressure_amplification(baseline: dict) -> None:
    """Compare global pressure amplification with the baseline."""
    print("\n" + "=" * 65)
    print("  2. SYNTHETIC GLOBAL PRESSURE AMPLIFICATION")
    print("=" * 65)

    G = _build_graph()
    snap_before = capture_conservation_snapshot(G)

    # Comply for a few steps, then violate U2
    for _ in range(5):
        _evolve_compliant(G)
    _inject_pressure_amplification(G)
    for _ in range(5):
        _evolve_compliant(G)

    snap_after = capture_conservation_snapshot(G)
    balance = verify_conservation_balance(snap_before, snap_after, dt=10 * 0.05)
    violations = detect_grammar_violations_from_conservation(balance)

    print("\n  Protocol: 5 baseline steps -> pressure amplification -> 5 steps")
    print(
        f"    Conservation quality: {balance.conservation_quality:.4f}  "
        f"(baseline: {baseline['quality']:.4f})"
    )
    print(
        f"    RMS residual:        {balance.rms_residual:.6f}  "
        f"(baseline: {baseline['rms']:.6f})"
    )
    print(
        f"    Max residual:        {balance.max_residual:.6f}  "
        f"(baseline: {baseline['max']:.6f})"
    )
    print(
        f"    Charge drift:        {balance.charge_drift:.6f}  "
        f"(baseline: {baseline['drift']:.6f})"
    )
    print(f"    GVI:                 {balance.grammar_violation_index:.6f}")
    print(f"    Balance alert:       {violations['alerts_detected']}")
    print(f"    Alert types:         {violations['alert_types']}")
    print(f"    Severity:            {violations['severity']:.4f}")
    print(f"    Nodes alerted:       {len(violations['nodes_alerted'])} nodes")

    rms_ratio = balance.rms_residual / max(baseline["rms"], 1e-10)
    print(f"\n  Diagnostic: RMS ratio (perturbed/baseline) = {rms_ratio:.1f}x")
    print("  Observation: global amplification changes the RMS balance residual.")


# ---------------------------------------------------------------------------
# 3. Antiphase-state perturbation
# ---------------------------------------------------------------------------


def demo_antiphase_state(baseline: dict) -> None:
    """Compare an antiphase-state perturbation with the baseline."""
    print("\n" + "=" * 65)
    print("  3. SYNTHETIC ANTIPHASE STATE")
    print("=" * 65)

    G = _build_graph()
    snap_before = capture_conservation_snapshot(G)

    for _ in range(5):
        _evolve_compliant(G)
    _inject_antiphase_state(G)
    for _ in range(5):
        _evolve_compliant(G)

    snap_after = capture_conservation_snapshot(G)
    balance = verify_conservation_balance(snap_before, snap_after, dt=10 * 0.05)
    violations = detect_grammar_violations_from_conservation(balance)

    print("\n  Protocol: 5 baseline steps -> antiphase state -> 5 steps")
    print(
        f"    Conservation quality: {balance.conservation_quality:.4f}  "
        f"(baseline: {baseline['quality']:.4f})"
    )
    print(
        f"    RMS residual:        {balance.rms_residual:.6f}  "
        f"(baseline: {baseline['rms']:.6f})"
    )
    print(
        f"    Max residual:        {balance.max_residual:.6f}  "
        f"(baseline: {baseline['max']:.6f})"
    )
    print(
        f"    Charge drift:        {balance.charge_drift:.6f}  "
        f"(baseline: {baseline['drift']:.6f})"
    )
    print(f"    GVI:                 {balance.grammar_violation_index:.6f}")
    print(f"    Balance alert:       {violations['alerts_detected']}")
    print(f"    Alert types:         {violations['alert_types']}")
    print(f"    Severity:            {violations['severity']:.4f}")
    print(f"    Nodes alerted:       {len(violations['nodes_alerted'])} nodes")

    max_ratio = balance.max_residual / max(baseline["max"], 1e-10)
    print(f"\n  Diagnostic: Max residual ratio = {max_ratio:.1f}x")
    print("  Observation: this state perturbation changes the maximum residual.")


# ---------------------------------------------------------------------------
# 4. Localized pressure spike
# ---------------------------------------------------------------------------


def demo_local_pressure_spike(baseline: dict) -> None:
    """Compare a localized pressure spike with the baseline."""
    print("\n" + "=" * 65)
    print("  4. SYNTHETIC LOCAL PRESSURE SPIKE")
    print("=" * 65)

    G = _build_graph()
    snap_before = capture_conservation_snapshot(G)

    for _ in range(5):
        _evolve_compliant(G)
    _inject_local_pressure_spike(G)
    for _ in range(5):
        _evolve_compliant(G)

    snap_after = capture_conservation_snapshot(G)
    balance = verify_conservation_balance(snap_before, snap_after, dt=10 * 0.05)
    violations = detect_grammar_violations_from_conservation(balance)

    print("\n  Protocol: 5 baseline steps -> local pressure spike -> 5 steps")
    print(
        f"    Conservation quality: {balance.conservation_quality:.4f}  "
        f"(baseline: {baseline['quality']:.4f})"
    )
    print(
        f"    RMS residual:        {balance.rms_residual:.6f}  "
        f"(baseline: {baseline['rms']:.6f})"
    )
    print(
        f"    Max residual:        {balance.max_residual:.6f}  "
        f"(baseline: {baseline['max']:.6f})"
    )
    print(
        f"    Charge drift:        {balance.charge_drift:.6f}  "
        f"(baseline: {baseline['drift']:.6f})"
    )
    print(f"    GVI:                 {balance.grammar_violation_index:.6f}")
    print(f"    Balance alert:       {violations['alerts_detected']}")
    print(f"    Alert types:         {violations['alert_types']}")
    print(f"    Severity:            {violations['severity']:.4f}")
    print(f"    Nodes alerted:       {len(violations['nodes_alerted'])} nodes")

    drift_ratio = balance.charge_drift / max(baseline["drift"], 1e-10)
    print(f"\n  Diagnostic: Charge drift ratio = {drift_ratio:.1f}x")
    print("  Observation: charge drift changes, but this is not the U6 Phi_s drift.")


# ---------------------------------------------------------------------------
# 5. Comparative severity analysis
# ---------------------------------------------------------------------------


def demo_severity_comparison() -> None:
    """Compare all synthetic perturbations on the same graph configuration."""
    print("\n" + "=" * 65)
    print("  5. COMPARATIVE SEVERITY ANALYSIS")
    print("=" * 65)

    scenarios = [
        ("Compliant", None),
        ("Pressure amp", _inject_pressure_amplification),
        ("Antiphase", _inject_antiphase_state),
        ("Local spike", _inject_local_pressure_spike),
    ]

    print(
        f"\n  {'Scenario':<14}  {'Quality':>10}  {'RMS':>10}  "
        f"{'Max_res':>10}  {'Drift':>10}  {'GVI':>10}  {'Severity':>10}"
    )
    print("  " + "-" * 76)

    for name, inject_fn in scenarios:
        G = _build_graph()
        snap_before = capture_conservation_snapshot(G)

        for _ in range(5):
            _evolve_compliant(G)
        if inject_fn is not None:
            inject_fn(G)
        for _ in range(5):
            _evolve_compliant(G)

        snap_after = capture_conservation_snapshot(G)
        balance = verify_conservation_balance(snap_before, snap_after, dt=10 * 0.05)
        violations = detect_grammar_violations_from_conservation(balance)

        print(
            f"  {name:<14}  {balance.conservation_quality:10.4f}  "
            f"{balance.rms_residual:10.6f}  {balance.max_residual:10.6f}  "
            f"{balance.charge_drift:10.6f}  {balance.grammar_violation_index:10.6f}  "
            f"{violations['severity']:10.4f}"
        )

    print(
        f"""
  These are overlapping response patterns, not rule identifiers:
    global pressure amplification can raise RMS residuals;
    an antiphase state can change maximum residuals;
    a local pressure spike can change charge drift.

  The same residual can arise from dynamics, topology or discretization.
  Grammar and U6 decisions require their dedicated validators.
"""
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print()
    print("*" * 65)
    print("  TNFR Example 36: Structural-Balance Heuristic Alerts")
    print("  Finite residual comparison; no grammar classifier")
    print("*" * 65)

    baseline = demo_baseline()
    demo_pressure_amplification(baseline)
    demo_antiphase_state(baseline)
    demo_local_pressure_spike(baseline)
    demo_severity_comparison()

    print("=" * 65)
    print("  SUMMARY")
    print("=" * 65)
    print(
        f"""
  Structural-balance residuals provide useful anomaly telemetry.
  The finite scenarios show that different perturbations alter RMS, maximum
  residual and charge drift in different amounts, but they do not establish
  unique signatures or grammar equivalences.

  Use validate_grammar for U1-U5 symbol/history rules, explicit phase checks
  for U3, and before/after structural-potential fields for the U6 policy.
"""
    )


if __name__ == "__main__":
    main()
