"""Finite variational diagnostics and the auxiliary quadratic model.

This example computes graph-field energy/action read-outs and the momentum
residual of a declared harmonic coordinate model. The nodal equation
``∂EPI/∂t = νf · ΔNFR(t)`` has not been derived as its Euler–Lagrange or
overdamped equation.

Key results shown:
1. Lagrangian density ℒ(i) = T(i) − V(i), Hamiltonian density H(i) = T(i) + V(i)
2. Auxiliary coordinate pairs: geometric (K_φ, J_φ) and potential (Φ_s, J_ΔNFR)
3. Harmonic-model momentum residual between two engine snapshots
4. Finite sampled action S = Σ dt · Σ_i ℒ(i)
5. Snapshot energy trends; symplecticity remains inconclusive without a Jacobian
6. Heuristic field comparisons carrying historical U1–U6 labels
7. Algebraically consistent finite sector read-outs

See: theory/TNFR_VARIATIONAL_PRINCIPLE.md for definitions and scope.
"""

from __future__ import annotations

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import networkx as nx
import numpy as np

from tnfr.constants import inject_defaults
from tnfr.operators import apply_glyph
from tnfr.physics.variational import (
    analyze_grammar_stationarity,
    analyze_potential_critical_points,
    capture_lagrangian_snapshot,
    check_symplectic_preservation,
    compute_action_functional,
    compute_euler_lagrange_residual,
    compute_phase_space_volume,
    identify_conjugate_pairs,
    translate_sectors,
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


def _evolve_step(G: nx.Graph) -> None:
    """One structural evolution step via Coherence (IL) on all nodes."""
    for node in G.nodes():
        apply_glyph(G, node, "IL")


def _evolve(G: nx.Graph, steps: int = 5, dt: float = 0.05) -> list[nx.Graph]:
    """Evolve the network and return snapshots at each step."""
    import copy

    snapshots = [copy.deepcopy(G)]
    for _ in range(steps):
        _evolve_step(G)
        snapshots.append(copy.deepcopy(G))
    return snapshots


# ------------------------------------------------------------------
# 1. Lagrangian snapshot — T, V, ℒ, H at a single instant
# ------------------------------------------------------------------


def demo_lagrangian_snapshot() -> None:
    """Decompose the structural energy into kinetic and potential sectors."""
    print("=" * 60)
    print("1. LAGRANGIAN SNAPSHOT  —  ℒ = T − V,  H = T + V")
    print("=" * 60)

    G = _build_graph()
    snap = capture_lagrangian_snapshot(G)

    print(f"  Total kinetic   T = {snap.total_kinetic:.6f}")
    print(f"  Total potential  V = {snap.total_potential:.6f}")
    print(f"  Lagrangian       L = T − V = {snap.total_lagrangian:.6f}")
    print(f"  Hamiltonian      H = T + V = {snap.total_hamiltonian:.6f}")
    print()

    # Per-node extremes
    nodes = list(G.nodes())
    max_T_node = max(nodes, key=lambda n: snap.kinetic[n])
    max_V_node = max(nodes, key=lambda n: snap.potential[n])
    print(f"  Node with max T: {max_T_node}  (T = {snap.kinetic[max_T_node]:.6f})")
    print(f"  Node with max V: {max_V_node}  (V = {snap.potential[max_V_node]:.6f})")
    print()


# ------------------------------------------------------------------
# 2. Auxiliary coordinate pairs — (K_φ, J_φ) and (Φ_s, J_ΔNFR)
# ------------------------------------------------------------------


def demo_conjugate_pairs() -> None:
    """Read the two coordinate pairs declared by the auxiliary model."""
    print("=" * 60)
    print("2. AUXILIARY COORDINATE PAIRS  —  (K_φ, J_φ) and (Φ_s, J_ΔNFR)")
    print("=" * 60)

    G = _build_graph()
    geo, pot = identify_conjugate_pairs(G)

    vol_geo = compute_phase_space_volume(geo)
    vol_pot = compute_phase_space_volume(pot)

    print(f"  Geometric sector ({geo.sector}):")
    print(f"    q = K_φ,  p = J_φ")
    print(f"    Legacy snapshot product = {vol_geo:.6f}")

    print(f"  Potential sector ({pot.sector}):")
    print(f"    q = Φ_s,  p = J_ΔNFR")
    print(f"    Legacy snapshot product = {vol_pot:.6f}")
    print()


# ------------------------------------------------------------------
# 3. Auxiliary momentum-equation residual
# ------------------------------------------------------------------


def demo_euler_lagrange() -> None:
    """Measure the declared harmonic momentum residual between snapshots."""
    print("=" * 60)
    print("3. AUXILIARY MOMENTUM RESIDUAL  —  finite snapshot comparison")
    print("=" * 60)

    G = _build_graph()
    dt = 0.05

    # Compare two engine snapshots with the auxiliary momentum equation.
    snap_before = capture_lagrangian_snapshot(G)
    for _ in range(3):
        _evolve_step(G)
    snap_after = capture_lagrangian_snapshot(G)

    el = compute_euler_lagrange_residual(snap_before, snap_after, dt=dt)
    print(f"  After 3 steps:")
    print(f"    RMS residual       = {el.rms_residual:.6f}")
    print(f"    Max residual       = {el.max_residual:.6f}")
    print(f"    Below model cutoff? = {el.is_stationary}")
    print(f"    Residual score      = {el.stationarity_quality:.4f}")

    # Repeat over a later pair; this is still a model-residual comparison.
    snap_mid = capture_lagrangian_snapshot(G)
    for _ in range(30):
        _evolve_step(G)
    snap_late = capture_lagrangian_snapshot(G)

    el2 = compute_euler_lagrange_residual(snap_mid, snap_late, dt=dt)
    print(f"  After 33 steps:")
    print(f"    RMS residual       = {el2.rms_residual:.6f}")
    print(f"    Max residual       = {el2.max_residual:.6f}")
    print(f"    Below model cutoff? = {el2.is_stationary}")
    print(f"    Residual score      = {el2.stationarity_quality:.4f}")
    print()


# ------------------------------------------------------------------
# 4. Action functional along an evolution path
# ------------------------------------------------------------------


def demo_action_functional() -> None:
    """Compute the total action S = ∫ dt Σ_i ℒ(i) along a trajectory."""
    print("=" * 60)
    print("4. ACTION FUNCTIONAL  —  S = ∫ dt Σ_i ℒ(i)")
    print("=" * 60)

    G = _build_graph()
    snaps = []
    for _ in range(20):
        snaps.append(capture_lagrangian_snapshot(G))
        _evolve_step(G)
    snaps.append(capture_lagrangian_snapshot(G))

    action = compute_action_functional(snaps, dt=0.05)
    print(f"  Steps:  {len(snaps) - 1}")
    print(f"  Action: S = {action:.6f}")
    print(f"  Mean Lagrangian per step: {action / (0.05 * (len(snaps) - 1)):.6f}")
    print()


# ------------------------------------------------------------------
# 5. Snapshot trend heuristic; no tangent-map certificate
# ------------------------------------------------------------------


def demo_symplectic() -> None:
    """Show endpoint heuristics and their inconclusive symplectic status."""
    print("=" * 60)
    print("5. ENDPOINT HEURISTICS  —  symplecticity requires a Jacobian")
    print("=" * 60)

    glyphs = [
        ("Coherence (IL)", "IL"),
        ("Dissonance (OZ)", "OZ"),
        ("Coupling (UM)", "UM"),
        ("Resonance (RA)", "RA"),
        ("Silence (SHA)", "SHA"),
        ("Emission (AL)", "AL"),
    ]

    for label, glyph in glyphs:
        G = _build_graph()
        try:
            snap_before = capture_lagrangian_snapshot(G)
            for node in G.nodes():
                apply_glyph(G, node, glyph)
            snap_after = capture_lagrangian_snapshot(G)
            check = check_symplectic_preservation(
                snap_before,
                snap_after,
                operator_name=glyph,
            )
            print(
                f"  {label:25s}  class={check.classification:12s}  "
                f"legacy_product_ratio={check.volume_ratio:.4f}  "
                f"local_symplectic={check.is_canonical}"
            )
        except Exception as e:
            print(f"  {label:25s}  error: {e}")
    print()


# ------------------------------------------------------------------
# 6. Heuristic field comparisons with historical grammar labels
# ------------------------------------------------------------------


def demo_grammar_stationarity() -> None:
    """Show advisory comparisons; these are not grammar validation."""
    print("=" * 60)
    print("6. HISTORICAL U1-U6 LABELS  —  advisory field comparisons")
    print("=" * 60)

    G = _build_graph()
    results = analyze_grammar_stationarity(G)

    print("  Per-label heuristic summary (grammar not validated):")
    for r in results:
        status = "within" if r.is_satisfied else "outside"
        print(f"    {r.rule:4s}  [{status:9s}]  diag={r.diagnostic_value:.4f}")
        print(f"          {r.variational_interpretation[:72]}")
    print()


# ------------------------------------------------------------------
# 7. Finite sector read-outs from the same fields
# ------------------------------------------------------------------


def demo_sector_translation() -> None:
    """Show algebraically related same-snapshot read-outs."""
    print("=" * 60)
    print("7. SECTOR READ-OUTS  —  quadratic | balance | complex coordinate")
    print("=" * 60)

    G = _build_graph()
    sectors = translate_sectors(G)

    # Variational
    T_total = sum(sectors["variational"]["T"].values())
    V_total = sum(sectors["variational"]["V"].values())

    # Conservation
    rho_total = sum(sectors["conservation"]["rho"].values())
    j_phi_mean = np.mean(list(sectors["conservation"]["J_phi"].values()))

    # Unified
    psi_vals = list(sectors["unified_psi"].values())
    psi_mean_mag = np.mean([abs(p) for p in psi_vals])

    print(f"  Variational:   T = {T_total:.6f},  V = {V_total:.6f}")
    print(f"  Conservation:  Σρ = {rho_total:.6f},  <J_φ> = {j_phi_mean:.6f}")
    print(f"  Unified Ψ:     <|Ψ|> = {psi_mean_mag:.6f}")
    print(f"  Consistency max|T+V − ½ℰ| = {sectors['consistency_check']:.2e}")
    print()


# ------------------------------------------------------------------
# 8. Critical points of the structural potential
# ------------------------------------------------------------------


def demo_critical_points() -> None:
    """Analyze critical points of V where the restoring force vanishes."""
    print("=" * 60)
    print("8. CRITICAL POINTS  —  ∂V/∂x = 0 analysis")
    print("=" * 60)

    G = _build_graph()
    cps = analyze_potential_critical_points(G)

    for cp in cps:
        grad_str = f"{cp.gradient_at_threshold:+.4f}"
        curv_str = f"{cp.curvature_at_threshold:+.4f}"
        print(
            f"  {cp.field_name:12s}  threshold={cp.threshold_value:.4f}  "
            f"∂V/∂x={grad_str}  ∂²V/∂x²={curv_str}  "
            f"type={cp.critical_type:8s}  critical={cp.is_critical}"
        )
    print()


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------


def main() -> None:
    print()
    print("TNFR FINITE VARIATIONAL DIAGNOSTICS")
    print("The quadratic auxiliary model is initialized from graph read-outs.")
    print("It is not a derivation of the full nodal equation or operator maps.")
    print()

    demo_lagrangian_snapshot()
    demo_conjugate_pairs()
    demo_euler_lagrange()
    demo_action_functional()
    demo_symplectic()
    demo_grammar_stationarity()
    demo_sector_translation()
    demo_critical_points()

    print("=" * 60)
    print("CONCLUSION: the identities and finite diagnostics above are scoped")
    print("to snapshots or the declared auxiliary model; grammar requires its")
    print("own operator-history validator and symplecticity requires a Jacobian.")
    print("See: theory/TNFR_VARIATIONAL_PRINCIPLE.md")
    print("=" * 60)


if __name__ == "__main__":
    main()
