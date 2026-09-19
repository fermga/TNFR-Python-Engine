#!/usr/bin/env python3
r"""Example 26: auxiliary U(1) coordinates of the complex field Ψ.

Demonstrates the algebraic invariants of the nodewise transformation
Ψ(i) → e^{iα(i)}Ψ(i).  This is an auxiliary field-coordinate action, not an
engine operator or a derived symmetry of the nodal dynamics.

Gauge-invariant quantities:
  - Energy density  ℰ(i) = Φ_s² + |∇φ|² + |Ψ|² + J_ΔNFR²
  - Field magnitude |Ψ(i)|² = K_φ² + J_φ²
  - Topological norm |𝒯|² = 𝒬² + 𝒬̃²
  - Chirality norm   |𝒳|² = χ² + χ̃²
  - Global coherence C(t)

NOT gauge-invariant:
  - Noether charge    Q = Σ(Φ_s + K_φ): K_φ rotates
  - Symmetry breaking 𝒮: K_φ² and J_φ² individually change
  - arg(Ψ): the phase itself is the gauge degree of freedom

Derived edge diagnostics:
  - Connection A_ij = d(arg Ψ)_ij is an exact, pure-gauge one-form
  - F_C = Σ_cycle A_ij is zero analytically modulo 2π
  - Returned F_C values are floating-point cycle-closure residuals
  - |D_ij Ψ| is invariant when A is reconstructed from the rotated Ψ

The historical em_like, weak_like, strong_like and gravity_like labels remain
as an API-compatible snapshot heuristic.  They are not derived fundamental
interactions; strong_like contains only an anomalous closure-residual score.

TNFR physics basis:
  Nodal equation   ∂EPI/∂t = νf · ΔNFR(t)
  Complex field    Ψ = K_φ + i·J_φ (phase curvature + phase current)
  Grammar rule U3  Separately gates coupling by the node phase φ.

Usage:
    python examples/02_physics_regimes/26_gauge_structure_demo.py
"""

from __future__ import annotations

import networkx as nx

from tnfr.mathematics.unified_numerical import np
from tnfr.physics.gauge import (
    apply_gauge_transformation,
    capture_gauge_snapshot,
    classify_network_regimes,
    compute_covariant_derivative_magnitude,
    compute_gauge_connection,
    compute_gauge_curvature,
    compute_gauge_energy_decomposition,
    compute_yang_mills_action,
    verify_gauge_invariance,
)


# ── Helpers ──────────────────────────────────────────────────────────
def section(title: str) -> None:
    print(f"\n{'=' * 64}")
    print(f"  {title}")
    print(f"{'=' * 64}\n")


def _make_tnfr_graph(n: int = 20, p: float = 0.3, seed: int = 42) -> nx.Graph:
    """Create a TNFR network with randomised structural attributes."""
    rng = np.random.default_rng(seed)
    G = nx.watts_strogatz_graph(n, 4, p, seed=seed)
    for node in G.nodes():
        G.nodes[node]["EPI"] = float(rng.uniform(0.5, 2.0))
        G.nodes[node]["nu_f"] = float(rng.uniform(0.1, 1.0))
        G.nodes[node]["theta"] = float(rng.uniform(0.0, 2.0 * np.pi))
        G.nodes[node]["delta_nfr"] = float(rng.uniform(-0.5, 0.5))
    return G


# ── Main demonstration ──────────────────────────────────────────────
def main() -> None:
    G = _make_tnfr_graph(n=20, p=0.3, seed=42)
    print(f"Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # ------------------------------------------------------------------
    # 1. Gauge Snapshot (before transformation)
    # ------------------------------------------------------------------
    section("1. Gauge Snapshot — Pre-Transformation Telemetry")
    snap = capture_gauge_snapshot(G)
    psi_mags = list(snap.psi_magnitude.values())
    print(f"  |Ψ| range         : [{min(psi_mags):.4f}, {max(psi_mags):.4f}]")
    phases = list(snap.psi_phase.values())
    print(f"  arg(Ψ) range      : [{min(phases):.4f}, {max(phases):.4f}] rad")
    energies = list(snap.energy_density.values())
    print(f"  ℰ(i) range        : [{min(energies):.4f}, {max(energies):.4f}]")
    topo_vals = list(snap.topological_norm.values())
    print(f"  |𝒯|² range        : [{min(topo_vals):.6f}, {max(topo_vals):.6f}]")

    # ------------------------------------------------------------------
    # 2. Gauge Invariance Verification
    # ------------------------------------------------------------------
    section("2. Gauge Invariance — Local U(1) Test")
    result = verify_gauge_invariance(G, seed=77)
    print(f"  All invariants OK : {result.is_invariant}")
    print(f"  ℰ  max deviation  : {result.energy_max_deviation:.2e}")
    print(f"  |Ψ| max deviation : {result.magnitude_max_deviation:.2e}")
    print(f"  |𝒯|² deviation    : {result.topological_norm_max_deviation:.2e}")
    print(f"  |𝒳|² deviation    : {result.chirality_norm_max_deviation:.2e}")
    print(f"  C(t) deviation    : {result.coherence_deviation:.2e}")
    print()
    print("  NON-invariant diagnostics:")
    print(f"  Q   deviation     : {result.noether_charge_deviation:.4f}")
    print(f"  𝒮   max deviation : {result.symmetry_breaking_max_deviation:.4f}")

    # ------------------------------------------------------------------
    # 3. Exact connection and cycle closure
    # ------------------------------------------------------------------
    section("3. Exact Connection A_ij & Cycle Closure F_C")
    conn = compute_gauge_connection(G)
    if conn:
        conn_vals = list(conn.values())
        print(f"  A_ij range      : [{min(conn_vals):.4f}, {max(conn_vals):.4f}]")
        print(f"  Edges with A     : {len(conn)}")

    curv = compute_gauge_curvature(G)
    if curv:
        curv_vals = list(curv.values())
        max_closure = max(abs(value) for value in curv_vals)
        print(f"  max |F_C|       : {max_closure:.2e} (numerical residual)")
        print(f"  Cycles detected  : {len(curv)}")
    else:
        print("  No short cycles found for curvature computation.")

    s_ym = compute_yang_mills_action(G)
    print(f"  legacy S_YM       : {s_ym:.2e} (squared closure residual)")

    # ------------------------------------------------------------------
    # 4. Covariant Derivative |D_ij Ψ|
    # ------------------------------------------------------------------
    section("4. Covariant Derivative — Gauge-Invariant Transport")
    cov_mag = compute_covariant_derivative_magnitude(G)
    if cov_mag:
        cov_vals = list(cov_mag.values())
        print(f"  |D_ij Ψ| range  : [{min(cov_vals):.4f}, {max(cov_vals):.4f}]")
        print(f"  Mean |D_ij Ψ|   : {sum(cov_vals) / len(cov_vals):.4f}")

    # Verify |D_ij Ψ| by reconstructing A from the rotated auxiliary field.
    rng = np.random.default_rng(123)
    alpha = {n: float(rng.uniform(0, 2 * np.pi)) for n in G.nodes()}
    cov_before = compute_covariant_derivative_magnitude(G)
    rotated = apply_gauge_transformation(G, alpha)
    covariant_deviation = 0.0
    for u, v in G.edges():
        psi_u = rotated["psi"][u]
        psi_v = rotated["psi"][v]
        phase_u = float(np.angle(psi_u)) if abs(psi_u) > 1e-15 else 0.0
        phase_v = float(np.angle(psi_v)) if abs(psi_v) > 1e-15 else 0.0
        link = phase_v - phase_u
        transported = psi_u * complex(np.cos(link), np.sin(link))
        covariant_deviation = max(
            covariant_deviation,
            abs(abs(psi_v - transported) - cov_before[(u, v)]),
        )
    print(f"  |D_ij Ψ| rotation deviation: {covariant_deviation:.2e}")

    # ------------------------------------------------------------------
    # 5. Energy Decomposition
    # ------------------------------------------------------------------
    section("5. Gauge Energy Decomposition (4 Sectors)")
    decomp = compute_gauge_energy_decomposition(G)
    for key in ["potential_sector", "gradient_sector", "gauge_sector", "flux_sector"]:
        frac_key = key.replace("_sector", "_fraction")
        frac = decomp.get(frac_key, 0.0)
        total = decomp.get(key, 0.0)
        print(f"  {key:20s}: E = {total:.4f}  ({frac * 100:.1f}%)")
    print(f"  {'total':20s}: E = {decomp['total_energy']:.4f}")
    print(f"  S_YM               : {decomp['yang_mills_action']:.6f}")

    # ------------------------------------------------------------------
    # 6. Interaction Regime Classification
    # ------------------------------------------------------------------
    section("6. Historical Field-Coordinate Labels")
    regimes = classify_network_regimes(G)
    print(f"  Dominant regime   : {regimes['dominant_regime']}")
    print(f"  Mean closure error: {regimes['mean_gauge_curvature']:.2e}")
    print(f"  Closure fraction  : {regimes['gauge_flatness']:.4f}")
    print("  Regime counts:")
    for regime, count in sorted(regimes["regime_distribution"].items()):
        pct = 100 * count / G.number_of_nodes()
        print(f"    {regime:15s}: {count:3d} nodes ({pct:.0f}%)")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    section("Summary")
    print("  The auxiliary field coordinates admit nodewise U(1) rotations.")
    print("  Under Ψ(i) → e^{iα(i)} Ψ(i):")
    print()
    print("  INVARIANT: ℰ, |Ψ|², |𝒯|², |𝒳|², C(t), |D_ij Ψ|")
    print("  VARIANT  : Q, 𝒮, arg(Ψ), (𝒬,𝒬̃) and (χ,χ̃) rotate")
    print()
    print("  A_ij=d(arg Ψ)_ij is pure gauge and every cycle closes exactly")
    print("  modulo 2π; nonzero F_C reports floating-point residual only.")
    print()
    print("  Physics basis: nodal equation ∂EPI/∂t = νf · ΔNFR(t)")
    print("  Grammar relation: U3 separately constrains node-phase coupling.")
    print("  U6 requires a separate before/after structural-potential drift;")
    print("  this single-snapshot gauge demo does not assess it.")


if __name__ == "__main__":
    main()
