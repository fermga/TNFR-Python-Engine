"""Example 36: Grammar Violation Detector via Conservation Residuals.

Demonstrates real-time detection and classification of grammar violations
through conservation residuals.  When operator sequences violate U1-U6,
the structural conservation law produces non-zero source terms:

  d(rho)/dt + div(J) = S_grammar   where S != 0 iff grammar violated

This detector uses conservation residuals to classify violations:
  - U2 breach (convergence failure): elevated RMS residuals
  - U3 breach (phase incompatibility): localized max residual spikes
  - U6 breach (confinement): charge drift exceeds phi ~ 1.618

Protocol (theory/STRUCTURAL_CONSERVATION_THEOREM.md ss 12):
  1. Run grammar-compliant sequence -> measure baseline residuals
  2. Introduce intentional violations -> measure elevated residuals
  3. Classify violation type from residual signature
  4. Quantify severity via grammar violation index (GVI)

Physics basis:
  Grammar symmetry (U1-U6) => S_grammar = 0 (conservation)
  Breaking any grammar rule => S_grammar != 0 (detectable residual)
  Each violation type produces a distinctive residual signature.

  See: theory/STRUCTURAL_CONSERVATION_THEOREM.md ss 12
  See: src/tnfr/physics/conservation.py::detect_grammar_violations_from_conservation
"""

from __future__ import annotations

import math
import sys
import os

import numpy as np
import networkx as nx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from tnfr.constants import inject_defaults
from tnfr.constants.canonical import PHI, GAMMA, PI
from tnfr.physics.conservation import (
    capture_conservation_snapshot,
    verify_conservation_balance,
    detect_grammar_violations_from_conservation,
    compute_noether_charge,
    ConservationTracker,
)
from tnfr.physics.fields import (
    compute_structural_potential,
    compute_phase_gradient,
    compute_phase_curvature,
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


def _inject_u2_violation(G: nx.Graph) -> None:
    """Simulate U2 (convergence) breach: unbounded DELTA_NFR growth."""
    for n in G.nodes():
        G.nodes[n]["delta_nfr"] *= 5.0  # Amplify without stabilizer


def _inject_u3_violation(G: nx.Graph) -> None:
    """Simulate U3 (phase incompatibility) breach: force antiphase coupling."""
    nodes = sorted(G.nodes())
    for i, n in enumerate(nodes):
        if i % 2 == 0:
            G.nodes[n]["phase"] = 0.0
        else:
            G.nodes[n]["phase"] = math.pi  # Antiphase


def _inject_u6_violation(G: nx.Graph) -> None:
    """Simulate U6 (confinement) breach: explosive DELTA_NFR at hub."""
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
    print(f"    Violations detected: {violations['violations_detected']}")
    print(f"    Violation types:     {violations['violation_types']}")
    print(f"    Severity:            {violations['severity']:.4f}")
    print(f"\n  Expected: No violations, low residuals (S_grammar ~ 0)")

    return {
        "quality": balance.conservation_quality,
        "rms": balance.rms_residual,
        "max": balance.max_residual,
        "drift": balance.charge_drift,
    }


# ---------------------------------------------------------------------------
# 2. U2 violation: convergence failure
# ---------------------------------------------------------------------------

def demo_u2_violation(baseline: dict) -> None:
    """Detect U2 (convergence) violation from residuals."""
    print("\n" + "=" * 65)
    print("  2. U2 VIOLATION — Convergence Failure (Destabilizer w/o Stabilizer)")
    print("=" * 65)

    G = _build_graph()
    snap_before = capture_conservation_snapshot(G)

    # Comply for a few steps, then violate U2
    for _ in range(5):
        _evolve_compliant(G)
    _inject_u2_violation(G)  # Amplify DELTA_NFR without stabilizer
    for _ in range(5):
        _evolve_compliant(G)

    snap_after = capture_conservation_snapshot(G)
    balance = verify_conservation_balance(snap_before, snap_after, dt=10 * 0.05)
    violations = detect_grammar_violations_from_conservation(balance)

    print(f"\n  Protocol: 5 compliant steps -> U2 breach -> 5 more steps")
    print(f"    Conservation quality: {balance.conservation_quality:.4f}  "
          f"(baseline: {baseline['quality']:.4f})")
    print(f"    RMS residual:        {balance.rms_residual:.6f}  "
          f"(baseline: {baseline['rms']:.6f})")
    print(f"    Max residual:        {balance.max_residual:.6f}  "
          f"(baseline: {baseline['max']:.6f})")
    print(f"    Charge drift:        {balance.charge_drift:.6f}  "
          f"(baseline: {baseline['drift']:.6f})")
    print(f"    GVI:                 {balance.grammar_violation_index:.6f}")
    print(f"    Violations detected: {violations['violations_detected']}")
    print(f"    Violation types:     {violations['violation_types']}")
    print(f"    Severity:            {violations['severity']:.4f}")
    print(f"    Nodes violating:     {len(violations['nodes_violating'])} nodes")

    rms_ratio = balance.rms_residual / max(baseline["rms"], 1e-10)
    print(f"\n  Diagnostic: RMS ratio (violation/baseline) = {rms_ratio:.1f}x")
    print(f"  Signature: ELEVATED RMS residual = convergence failure (U2)")


# ---------------------------------------------------------------------------
# 3. U3 violation: phase incompatibility
# ---------------------------------------------------------------------------

def demo_u3_violation(baseline: dict) -> None:
    """Detect U3 (phase incompatibility) violation from residuals."""
    print("\n" + "=" * 65)
    print("  3. U3 VIOLATION — Phase Incompatibility (Antiphase Coupling)")
    print("=" * 65)

    G = _build_graph()
    snap_before = capture_conservation_snapshot(G)

    for _ in range(5):
        _evolve_compliant(G)
    _inject_u3_violation(G)  # Force antiphase
    for _ in range(5):
        _evolve_compliant(G)

    snap_after = capture_conservation_snapshot(G)
    balance = verify_conservation_balance(snap_before, snap_after, dt=10 * 0.05)
    violations = detect_grammar_violations_from_conservation(balance)

    print(f"\n  Protocol: 5 compliant steps -> U3 breach -> 5 more steps")
    print(f"    Conservation quality: {balance.conservation_quality:.4f}  "
          f"(baseline: {baseline['quality']:.4f})")
    print(f"    RMS residual:        {balance.rms_residual:.6f}  "
          f"(baseline: {baseline['rms']:.6f})")
    print(f"    Max residual:        {balance.max_residual:.6f}  "
          f"(baseline: {baseline['max']:.6f})")
    print(f"    Charge drift:        {balance.charge_drift:.6f}  "
          f"(baseline: {baseline['drift']:.6f})")
    print(f"    GVI:                 {balance.grammar_violation_index:.6f}")
    print(f"    Violations detected: {violations['violations_detected']}")
    print(f"    Violation types:     {violations['violation_types']}")
    print(f"    Severity:            {violations['severity']:.4f}")
    print(f"    Nodes violating:     {len(violations['nodes_violating'])} nodes")

    max_ratio = balance.max_residual / max(baseline["max"], 1e-10)
    print(f"\n  Diagnostic: Max residual ratio = {max_ratio:.1f}x")
    print(f"  Signature: LOCALIZED MAX residual spikes = phase incompatibility (U3)")


# ---------------------------------------------------------------------------
# 4. U6 violation: confinement breach
# ---------------------------------------------------------------------------

def demo_u6_violation(baseline: dict) -> None:
    """Detect U6 (confinement) violation from charge drift."""
    print("\n" + "=" * 65)
    print("  4. U6 VIOLATION — Confinement Breach (Phi_s Escape)")
    print("=" * 65)

    G = _build_graph()
    snap_before = capture_conservation_snapshot(G)

    for _ in range(5):
        _evolve_compliant(G)
    _inject_u6_violation(G)  # Explosive DELTA_NFR at hub
    for _ in range(5):
        _evolve_compliant(G)

    snap_after = capture_conservation_snapshot(G)
    balance = verify_conservation_balance(snap_before, snap_after, dt=10 * 0.05)
    violations = detect_grammar_violations_from_conservation(balance)

    print(f"\n  Protocol: 5 compliant steps -> U6 breach -> 5 more steps")
    print(f"    Conservation quality: {balance.conservation_quality:.4f}  "
          f"(baseline: {baseline['quality']:.4f})")
    print(f"    RMS residual:        {balance.rms_residual:.6f}  "
          f"(baseline: {baseline['rms']:.6f})")
    print(f"    Max residual:        {balance.max_residual:.6f}  "
          f"(baseline: {baseline['max']:.6f})")
    print(f"    Charge drift:        {balance.charge_drift:.6f}  "
          f"(baseline: {baseline['drift']:.6f})")
    print(f"    GVI:                 {balance.grammar_violation_index:.6f}")
    print(f"    Violations detected: {violations['violations_detected']}")
    print(f"    Violation types:     {violations['violation_types']}")
    print(f"    Severity:            {violations['severity']:.4f}")
    print(f"    Nodes violating:     {len(violations['nodes_violating'])} nodes")

    drift_ratio = balance.charge_drift / max(baseline["drift"], 1e-10)
    print(f"\n  Diagnostic: Charge drift ratio = {drift_ratio:.1f}x")
    print(f"  Signature: CHARGE DRIFT > phi (1.618) = confinement breach (U6)")


# ---------------------------------------------------------------------------
# 5. Comparative severity analysis
# ---------------------------------------------------------------------------

def demo_severity_comparison() -> None:
    """Compare all violation types on the same graph configuration."""
    print("\n" + "=" * 65)
    print("  5. COMPARATIVE SEVERITY ANALYSIS")
    print("=" * 65)

    scenarios = [
        ("Compliant", None),
        ("U2 breach", _inject_u2_violation),
        ("U3 breach", _inject_u3_violation),
        ("U6 breach", _inject_u6_violation),
    ]

    print(f"\n  {'Scenario':<14}  {'Quality':>10}  {'RMS':>10}  "
          f"{'Max_res':>10}  {'Drift':>10}  {'GVI':>10}  {'Severity':>10}")
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

        print(f"  {name:<14}  {balance.conservation_quality:10.4f}  "
              f"{balance.rms_residual:10.6f}  {balance.max_residual:10.6f}  "
              f"{balance.charge_drift:10.6f}  {balance.grammar_violation_index:10.6f}  "
              f"{violations['severity']:10.4f}")

    print(f"""
  Violation signatures (from ss 12):
    U2 (convergence):  High RMS residual (global instability)
    U3 (phase):        High MAX residual (local incompatibility)
    U6 (confinement):  High charge DRIFT (global escape)

  The conservation law acts as a universal grammar violation detector:
    S_grammar = 0  <=>  Grammar U1-U6 satisfied
    S_grammar > 0  ==>  Violation type classifiable from residual pattern
""")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print()
    print("*" * 65)
    print("  TNFR Example 36: Grammar Violation Detector")
    print("  Theory: STRUCTURAL_CONSERVATION_THEOREM.md ss 12")
    print("*" * 65)

    baseline = demo_baseline()
    demo_u2_violation(baseline)
    demo_u3_violation(baseline)
    demo_u6_violation(baseline)
    demo_severity_comparison()

    print("=" * 65)
    print("  SUMMARY")
    print("=" * 65)
    print(f"""
  Grammar Violation Detection via Conservation Residuals (ss 12):

  Main result: S_grammar != 0  detects and classifies grammar violations.

  Detected violation types:
    U2 — Convergence failure:    RMS residual elevated globally
    U3 — Phase incompatibility:  Max residual spikes at boundary nodes
    U6 — Confinement breach:     Charge drift exceeds phi ~ 1.618

  The conservation law provides a SINGLE diagnostic framework
  for ALL grammar violations — no separate checker needed per rule.
  This is the Noether-like consequence of grammar symmetry.
""")


if __name__ == "__main__":
    main()
