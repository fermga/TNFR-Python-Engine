"""Example 34: finite reproduction of a conservation protocol suite.

Reproduces the numerical checks proposed in
theory/STRUCTURAL_CONSERVATION_THEOREM.md ss 10:

  1. Charge drift test: |Q(t_f) - Q(t_0)| / |Q(t_0)| < threshold
  2. Balance quality tracking: q = 1 / (1 + RMS_residual)
  3. Sector decomposition: potential (Phi_s, J_DELTA_NFR) vs
     geometric (K_phi, J_phi) contributions
  4. Finite cross-topology comparison: WS, BA, Grid, Ring, Complete
  5. Scaling analysis: balance quality vs network size
  6. Candidate-energy monotonicity on the sampled trajectory

The implemented Noether-like balance candidate is evaluated along a seeded
auxiliary smoothing trajectory:

  d(rho)/dt + div(J) = S_residual

where rho = Phi_s + K_phi and J = (J_phi, J_DELTA_NFR). The residual is
measured; it is not forced to zero by an asserted grammar label. This example
does not execute canonical operators and therefore makes no U1-U6 compliance
claim.

Legacy numerical targets under test:
  - Charge drift < 0.03% across topologies
  - Mean sampled balance quality q ~ 0.6-0.65
  - Sector asymmetry ratio ~ 1.0-1.2
  - Negative 1/sqrt(N) coefficient in the proposed finite-size form
  - Non-increasing candidate energy at every sampled step

Outcome policy:
  Every target is reported as PASS or FAIL from the values produced here. A
  failed target is retained as a negative result rather than relabelled WARN.
  See: theory/STRUCTURAL_CONSERVATION_THEOREM.md ss 10
  See: src/tnfr/physics/conservation.py
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
    compute_energy_functional,
    compute_noether_charge,
    verify_conservation_balance,
)
from tnfr.physics.extended import compute_dnfr_flux, compute_phase_current
from tnfr.physics.fields import (
    compute_phase_curvature,
    compute_phase_gradient,
    compute_structural_potential,
)


def _build_graph(n: int, topology: str, seed: int = 42) -> nx.Graph:
    """Build TNFR-initialized graph."""
    rng = np.random.default_rng(seed)
    if topology == "WS":
        G = nx.watts_strogatz_graph(n, 4, 0.3, seed=seed)
    elif topology == "BA":
        G = nx.barabasi_albert_graph(n, 2, seed=seed)
    elif topology == "Grid":
        side = int(math.sqrt(n))
        G = nx.grid_2d_graph(side, side)
        mapping = {node: i for i, node in enumerate(G.nodes())}
        G = nx.relabel_nodes(G, mapping)
    elif topology == "Ring":
        G = nx.cycle_graph(n)
    elif topology == "Complete":
        G = nx.complete_graph(n)
    else:
        G = nx.watts_strogatz_graph(n, 4, 0.3, seed=seed)

    inject_defaults(G)
    for node in G.nodes():
        G.nodes[node]["phase"] = rng.uniform(0, 2 * math.pi)
        G.nodes[node]["theta"] = G.nodes[node]["phase"]
        G.nodes[node]["delta_nfr"] = rng.uniform(-0.3, 0.3)
        G.nodes[node]["nu_f"] = rng.uniform(0.8, 1.2)
    return G


def _evolve_step(G: nx.Graph, dt: float = 0.05) -> None:
    """Apply the example's auxiliary phase/DELTA_NFR smoothing rule.

    This direct numerical rule is neither an IL application nor a validated
    grammar word.
    """
    for n in G.nodes():
        neighbors = list(G.neighbors(n))
        if neighbors:
            mean_phase = np.mean([G.nodes[nb]["phase"] for nb in neighbors])
            G.nodes[n]["phase"] += dt * (mean_phase - G.nodes[n]["phase"])
            G.nodes[n]["theta"] = G.nodes[n]["phase"]
            mean_dnfr = np.mean([G.nodes[nb]["delta_nfr"] for nb in neighbors])
            G.nodes[n]["delta_nfr"] += dt * (mean_dnfr - G.nodes[n]["delta_nfr"])


# ---------------------------------------------------------------------------
# 1. Charge drift test
# ---------------------------------------------------------------------------


def demo_charge_drift() -> dict[str, float | int | bool]:
    """Test the legacy 0.03% charge-drift target across five fixtures."""
    print("=" * 65)
    print("  1. CHARGE DRIFT TEST (ss 10.1)")
    print("=" * 65)

    topologies = [
        ("WS (N=50)", "WS", 50),
        ("BA (N=50)", "BA", 50),
        ("Grid (7x7)", "Grid", 49),
        ("Ring (N=50)", "Ring", 50),
        ("Complete (N=15)", "Complete", 15),
    ]
    n_steps = 20
    dt = 0.05

    print(f"\n  Protocol: Evolve {n_steps} auxiliary smoothing steps, measure charge drift")
    print("  Legacy target: |Q(t_f) - Q(t_0)| / |Q(t_0)| < 0.03%")
    print()
    print(
        f"  {'Topology':<18}  {'Q(t_0)':>10}  {'Q(t_f)':>10}  "
        f"{'|drift|':>10}  {'drift %':>8}  {'Status':>8}"
    )
    print("  " + "-" * 68)

    drift_percentages: list[float] = []
    for name, topo, n in topologies:
        G = _build_graph(n, topo)
        snap_before = capture_conservation_snapshot(G)
        q_0 = sum(snap_before.charge_density.values())

        for _ in range(n_steps):
            _evolve_step(G, dt)

        snap_after = capture_conservation_snapshot(G)
        q_f = sum(snap_after.charge_density.values())

        drift_abs = abs(q_f - q_0)
        drift_pct = drift_abs / abs(q_0) * 100 if abs(q_0) > 1e-10 else 0.0
        drift_percentages.append(drift_pct)
        status = "PASS" if drift_pct < 0.03 else "FAIL"
        print(
            f"  {name:<18}  {q_0:10.4f}  {q_f:10.4f}  "
            f"{drift_abs:10.6f}  {drift_pct:8.4f}  {status:>8}"
        )

    passed = sum(value < 0.03 for value in drift_percentages)
    target_met = passed == len(drift_percentages)
    print(
        f"\n  Target result: {'PASS' if target_met else 'FAIL'} "
        f"({passed}/{len(drift_percentages)} fixtures below 0.03%)"
    )
    return {
        "passed": passed,
        "total": len(drift_percentages),
        "target_met": target_met,
        "max_drift_pct": max(drift_percentages),
    }


# ---------------------------------------------------------------------------
# 2. Balance quality tracking
# ---------------------------------------------------------------------------


def demo_balance_quality() -> dict[str, float | bool]:
    """Track balance quality q = 1/(1 + RMS_residual) over time."""
    print("\n" + "=" * 65)
    print("  2. BALANCE QUALITY TRACKING (ss 10.2)")
    print("=" * 65)

    G = _build_graph(50, "WS")
    n_steps = 30
    dt = 0.05

    tracker = ConservationTracker(G)
    tracker.record(t=0.0)

    print(f"\n  WS (N=50), {n_steps} steps, dt = {dt}")
    print("  Legacy target band: mean quality q in [0.60, 0.65]")
    print()
    print(
        f"  {'Step':>6}  {'Quality':>10}  {'RMS_res':>10}  "
        f"{'Drift':>10}  {'Alert idx':>10}"
    )
    print("  " + "-" * 50)

    for step in range(1, n_steps + 1):
        _evolve_step(G, dt)
        tracker.record(t=step * dt)

    # Print report from tracker
    report = tracker.report()
    step_idx = 0
    for t, q, rms, drift, alert_index in zip(
        report.times[1:],
        report.conservation_quality[1:],
        report.rms_residuals[1:],
        report.charge_drift[1:],
        report.grammar_violation_index[1:],
    ):
        step_idx += 1
        if step_idx % 5 == 0 or step_idx <= 3:
            print(
                f"  {step_idx:6d}  {q:10.4f}  {rms:10.6f}  "
                f"{drift:10.6f}  {alert_index:10.6f}"
            )

    mean_sampled_quality = report.sampled_mean_quality
    print(f"\n  Summary:")
    print(f"    Mean sampled quality: {mean_sampled_quality:.4f}")
    target_met = 0.60 <= mean_sampled_quality <= 0.65
    print(f"    Legacy target met: {'PASS' if target_met else 'FAIL'}")
    print(
        "    Legacy aggregate alert (sampled mean q >= 0.9): "
        f"{report.aggregate_balance_within_alert}"
    )
    print(
        f"    Final charge drift: {report.charge_drift[-1]:.6f}"
        if report.charge_drift
        else ""
    )
    return {"mean_quality": mean_sampled_quality, "target_met": target_met}


# ---------------------------------------------------------------------------
# 3. Sector decomposition
# ---------------------------------------------------------------------------


def demo_sector_decomposition() -> dict[str, float | bool]:
    """Decompose conservation into potential vs geometric sectors."""
    print("\n" + "=" * 65)
    print("  3. SECTOR DECOMPOSITION (ss 10.3)")
    print("=" * 65)

    G = _build_graph(50, "WS")
    for _ in range(10):
        _evolve_step(G)

    phi_s = compute_structural_potential(G)
    k_phi = compute_phase_curvature(G)
    j_phi = compute_phase_current(G)
    j_dnfr = compute_dnfr_flux(G)
    grad_phi = compute_phase_gradient(G)

    nodes = sorted(G.nodes())
    ps = np.array([phi_s[n] for n in nodes])
    kp = np.array([k_phi[n] for n in nodes])
    jp = np.array([j_phi[n] for n in nodes])
    jd = np.array([j_dnfr[n] for n in nodes])
    gp = np.array([grad_phi[n] for n in nodes])

    # Potential sector: (Phi_s, J_DELTA_NFR)
    V_pot = 0.5 * np.sum(ps**2 + jd**2)
    # Geometric sector: (K_phi, J_phi)
    V_geo = 0.5 * np.sum(kp**2 + jp**2)
    # Gradient contribution
    V_grad = 0.5 * np.sum(gp**2)

    E_total = V_pot + V_geo + V_grad

    print(f"\n  Two-sector energy structure (WS N=50):")
    print(
        f"    Potential sector  (Phi_s, J_DELTA_NFR): {V_pot:10.4f}  "
        f"({V_pot/E_total*100:.1f}%)"
    )
    print(
        f"    Geometric sector  (K_phi, J_phi):       {V_geo:10.4f}  "
        f"({V_geo/E_total*100:.1f}%)"
    )
    print(
        f"    Gradient sector   (|grad_phi|):         {V_grad:10.4f}  "
        f"({V_grad/E_total*100:.1f}%)"
    )
    print(f"    Total energy E:                         {E_total:10.4f}")

    # Sector asymmetry ratio
    ratio = V_pot / V_geo if V_geo > 1e-10 else float("inf")
    print(f"\n  Sector asymmetry ratio V_pot / V_geo = {ratio:.4f}")
    target_met = 1.0 <= ratio <= 1.2
    print("  Legacy target range: 1.0 - 1.2")
    print(f"  Target result: {'PASS' if target_met else 'FAIL'}")

    # Conjugate pair correlations
    r_potential = (
        np.corrcoef(ps, jd)[0, 1] if np.std(ps) > 1e-10 and np.std(jd) > 1e-10 else 0.0
    )
    r_geometric = (
        np.corrcoef(kp, jp)[0, 1] if np.std(kp) > 1e-10 and np.std(jp) > 1e-10 else 0.0
    )

    print(f"\n  Conjugate pair correlations:")
    print(f"    r(Phi_s, J_DELTA_NFR)   = {r_potential:.4f}  (potential sector)")
    print(f"    r(K_phi, J_phi)          = {r_geometric:.4f}  (geometric sector)")
    print("    These are finite correlations; they do not prove conjugacy.")
    return {"ratio": ratio, "target_met": target_met}


# ---------------------------------------------------------------------------
# 4. Finite cross-topology comparison
# ---------------------------------------------------------------------------


def demo_cross_topology() -> None:
    """Compare the balance diagnostics on nine finite fixtures."""
    print("\n" + "=" * 65)
    print("  4. FINITE CROSS-TOPOLOGY COMPARISON (ss 10.4)")
    print("=" * 65)

    configs = [
        ("WS (N=30)", "WS", 30),
        ("WS (N=50)", "WS", 50),
        ("WS (N=100)", "WS", 100),
        ("BA (N=30)", "BA", 30),
        ("BA (N=50)", "BA", 50),
        ("Grid (5x5)", "Grid", 25),
        ("Grid (7x7)", "Grid", 49),
        ("Ring (N=40)", "Ring", 40),
        ("Complete (N=12)", "Complete", 12),
    ]
    n_steps = 15
    dt = 0.05

    print(f"\n  Protocol: {n_steps} steps, dt = {dt}")
    print(
        f"  {'Config':<18}  {'Charge Q*':>10}  {'Candidate E':>11}  "
        f"{'Quality':>10}  {'Drift %':>8}"
    )
    print("  " + "-" * 62)

    for name, topo, n in configs:
        G = _build_graph(n, topo)
        snap_0 = capture_conservation_snapshot(G)
        q_0 = sum(snap_0.charge_density.values())

        for _ in range(n_steps):
            _evolve_step(G, dt)

        snap_f = capture_conservation_snapshot(G)
        balance = verify_conservation_balance(snap_0, snap_f, dt=n_steps * dt)

        noether_q = compute_noether_charge(G)
        energy = compute_energy_functional(G)
        drift_pct = balance.charge_drift / abs(q_0) * 100 if abs(q_0) > 1e-10 else 0.0

        print(
            f"  {name:<18}  {noether_q:10.4f}  {energy:10.4f}  "
            f"{balance.conservation_quality:10.4f}  {drift_pct:8.4f}"
        )


# ---------------------------------------------------------------------------
# 5. Scaling analysis
# ---------------------------------------------------------------------------


def demo_scaling_analysis() -> dict[str, float | bool]:
    """Compare finite-size data with the proposed 1-C/sqrt(N) form."""
    print("\n" + "=" * 65)
    print("  5. SCALING ANALYSIS — Quality vs Network Size (ss 10.5)")
    print("=" * 65)

    sizes = [10, 20, 30, 50, 75, 100, 150, 200]
    n_steps = 15
    dt = 0.05

    print(f"\n  WS topology, k=4, p=0.3, {n_steps} steps")
    print("  Legacy target: q(N) ~ 1 - C/sqrt(N), with C > 0")
    print()
    print(
        f"  {'N':>6}  {'Quality':>10}  {'RMS_res':>10}  "
        f"{'Charge Q*':>10}  {'Candidate E':>11}"
    )
    print("  " + "-" * 50)

    qualities: list[float] = []
    for n in sizes:
        G = _build_graph(n, "WS")
        snap_0 = capture_conservation_snapshot(G)

        for _ in range(n_steps):
            _evolve_step(G, dt)

        snap_f = capture_conservation_snapshot(G)
        balance = verify_conservation_balance(snap_0, snap_f, dt=n_steps * dt)

        noether_q = compute_noether_charge(G)
        energy = compute_energy_functional(G)
        qualities.append(balance.conservation_quality)

        print(
            f"  {n:6d}  {balance.conservation_quality:10.4f}  "
            f"{balance.rms_residual:10.6f}  {noether_q:10.4f}  {energy:10.4f}"
        )

    # An unconstrained fit exposes both claims in q = 1 - C/sqrt(N):
    # the intercept should be one and the 1/sqrt(N) coefficient negative.
    x = 1.0 / np.sqrt(np.array(sizes, dtype=float))
    y = np.array(qualities)
    slope, intercept = np.polyfit(x, y, 1)
    direction_met = slope < 0.0
    print(f"\n  Unconstrained fit: q = {intercept:.4f} + {slope:.4f} / sqrt(N)")
    print("  Target requires: intercept -> 1 and coefficient < 0")
    print(
        f"  Direction result: {'PASS' if direction_met else 'FAIL'} "
        f"(coefficient = {slope:+.4f})"
    )
    print(
        "  This finite fit does not support the proposed scaling law. Its "
        "extrapolated intercept is descriptive, not an asymptotic estimate."
    )
    return {
        "intercept": float(intercept),
        "slope": float(slope),
        "target_met": direction_met,
    }


# ---------------------------------------------------------------------------
# 6. Candidate-energy monotonicity check
# ---------------------------------------------------------------------------


def demo_candidate_energy_monotonicity() -> dict[str, int | bool]:
    """Measure candidate-energy monotonicity on the auxiliary trajectory."""
    print("\n" + "=" * 65)
    print("  6. CANDIDATE-ENERGY MONOTONICITY — finite trajectory (ss 10.6)")
    print("=" * 65)

    G = _build_graph(50, "WS")
    n_steps = 25
    dt = 0.05

    print(f"\n  WS (N=50), {n_steps} steps, dt = {dt}")
    print("  Sampled target: E >= 0 and dE/dt <= 0 at every recorded step")
    print()
    print(
        f"  {'Step':>6}  {'Energy E':>12}  {'dE/dt':>12}  {'E >= 0':>8}  {'dE/dt <= 0':>12}"
    )
    print("  " + "-" * 54)

    E_prev = compute_energy_functional(G)
    positive_derivative_steps = 0
    for step in range(1, n_steps + 1):
        _evolve_step(G, dt)
        E_curr = compute_energy_functional(G)
        dE_dt = (E_curr - E_prev) / dt

        positive = E_curr >= -1e-10
        decreasing = dE_dt <= 0.0
        if not decreasing:
            positive_derivative_steps += 1

        if step % 5 == 0 or step <= 3:
            print(
                f"  {step:6d}  {E_curr:12.4f}  {dE_dt:12.6f}  "
                f"{'YES' if positive else 'NO':>8}  {'YES' if decreasing else 'NO':>12}"
            )
        E_prev = E_curr

    target_met = positive_derivative_steps == 0
    print(
        f"\n  Positive-derivative steps: {positive_derivative_steps}/{n_steps}"
    )
    print(f"  Finite target result: {'PASS' if target_met else 'FAIL'}")
    print(
        "  Passing this sample does not establish a Lyapunov theorem "
        "or U1-U6 result."
    )
    return {
        "positive_derivative_steps": positive_derivative_steps,
        # Backward-compatible historical result key.
        "violations": positive_derivative_steps,
        "steps": n_steps,
        "target_met": target_met,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print()
    print("*" * 65)
    print("  TNFR Example 34: Finite Conservation-Protocol Reproduction")
    print("  Theory: STRUCTURAL_CONSERVATION_THEOREM.md ss 10")
    print("*" * 65)

    charge = demo_charge_drift()
    quality = demo_balance_quality()
    sectors = demo_sector_decomposition()
    demo_cross_topology()
    scaling = demo_scaling_analysis()
    energy = demo_candidate_energy_monotonicity()

    print("\n" + "=" * 65)
    print("  SUMMARY")
    print("=" * 65)
    print("\n  Finite protocol outcomes:")
    print(
        f"  1. Charge-drift target:  "
        f"{'PASS' if charge['target_met'] else 'FAIL'} "
        f"({charge['passed']}/{charge['total']} fixtures)"
    )
    print(
        f"  2. Legacy q band:        "
        f"{'PASS' if quality['target_met'] else 'FAIL'} "
        f"(mean q = {quality['mean_quality']:.4f})"
    )
    print(
        f"  3. Sector-ratio target:  "
        f"{'PASS' if sectors['target_met'] else 'FAIL'} "
        f"(ratio = {sectors['ratio']:.4f})"
    )
    print("  4. Cross-topology table: descriptive finite comparison")
    print(
        f"  5. Scaling-law direction: "
        f"{'PASS' if scaling['target_met'] else 'FAIL'} "
        f"(coefficient = {scaling['slope']:+.4f})"
    )
    print(
        f"  6. Sampled energy target: "
        f"{'PASS' if energy['target_met'] else 'FAIL'} "
        f"({energy['steps'] - energy['positive_derivative_steps']}/"
        f"{energy['steps']} steps)"
    )

    targets = (charge, quality, sectors, scaling, energy)
    all_targets_met = all(result["target_met"] for result in targets)
    print(
        "\n  Overall legacy protocol result: "
        + ("PASS" if all_targets_met else "NEGATIVE — advertised targets fail")
    )
    print("  No canonical operator word was executed, so this run cannot infer")
    print("  conservation or Lyapunov stability from grammar compliance.")


if __name__ == "__main__":
    main()
