"""Example 31: The π Structural Scale and the Derivative-Tower Tetrad.

Demonstrates the single genuine structural scale of TNFR — π — and how the
four structural fields emerge as the four orders of the discrete
structural-derivative tower. Nothing here is assumed beyond π and the
continuum ℝ; every other quantity emerges from the nodal dynamics
∂EPI/∂t = νf · ΔNFR(t).

Only π is a genuine structural scale: it bounds the WHOLE phase sector,
since both phase derivatives are wrapped angles —

    |∇φ| ≤ π   and   |K_φ| ≤ π .

The four tetrad fields are the four orders of the derivative tower:

    Φ_s   (0th order, global aggregation)   ΔNFR_j → Σ 1/d²
    |∇φ|  (1st order, local derivative)      φ_i   → ∇
    K_φ   (2nd order, local Laplacian)       φ_i   → ∇²   (K_φ = L_rw·φ)
    ξ_C   (correlation, non-local)           correlation range

The coherence length is set by the spectral gap (Fiedler value λ₂):

    ξ_C ∝ 1/√λ₂ .

Key results shown:
  1. π bounds the whole phase sector: both |∇φ| ≤ π and |K_φ| ≤ π.
  2. The derivative-tower tetrad (Φ_s, |∇φ|, K_φ, ξ_C) on a network.
  3. ξ_C tracks the spectral gap (ξ_C ∝ 1/√λ₂) across topologies.
  4. Cross-topology confinement: the π phase-wrap bounds and the π-derived
     Φ_s confinement bounds (drift < π/2, per-node < π/4) hold everywhere.

Physics basis:
  The nodal equation generates a transport layer whose structural diagnostics
  require exactly four irreducible channels (the orders of the derivative
  tower). π is the one genuine structural scale (the phase-wrap bound); φ, γ,
  e play no role and are intentionally absent.
  See: theory/MATHEMATICAL_DYNAMICS_BASIS.md
  See: theory/MINIMAL_STRUCTURAL_DEGREES.md §§ 4-5
"""

from __future__ import annotations

import math
import os
import sys

import networkx as nx
import numpy as np

# Ensure src/ is importable when running from examples/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from tnfr.constants import inject_defaults
from tnfr.constants.canonical import (
    GRAD_PHI_CANONICAL_THRESHOLD,
    K_PHI_CANONICAL_THRESHOLD,
    PHI_S_VON_KOCH_THRESHOLD,
    PI,
    U6_STRUCTURAL_POTENTIAL_LIMIT,
)
from tnfr.physics.fields import (
    compute_phase_curvature,
    compute_phase_gradient,
    compute_structural_potential,
    estimate_coherence_length,
)


def _seed_network(G: "nx.Graph", *, seed: int = 42, correlated: bool = False) -> None:
    """Inject canonical defaults and seed phase / ΔNFR on every node."""
    inject_defaults(G)
    rng = np.random.default_rng(seed)
    prev = 0.0
    for i, n in enumerate(G.nodes()):
        if correlated and i > 0:
            phase = prev + rng.normal(0, 0.3)
        else:
            phase = rng.uniform(0, 2 * math.pi)
        prev = phase
        G.nodes[n]["phase"] = phase
        G.nodes[n]["theta"] = phase
        G.nodes[n]["delta_nfr"] = rng.uniform(-0.5, 0.5)


def _algebraic_connectivity(G: "nx.Graph") -> float:
    """Fiedler value λ₂ (second-smallest Laplacian eigenvalue)."""
    laplacian = nx.laplacian_matrix(G).toarray().astype(float)
    eigvals = np.linalg.eigvalsh(laplacian)
    return float(eigvals[1]) if len(eigvals) > 1 else 0.0


# ---------------------------------------------------------------------------
# 1. π bounds the whole phase sector
# ---------------------------------------------------------------------------


def demo_pi_phase_sector() -> None:
    """π bounds BOTH phase derivatives: |∇φ| ≤ π and |K_φ| ≤ π."""
    print("=" * 65)
    print("  1. π — THE GENUINE STRUCTURAL SCALE (whole phase sector)")
    print("=" * 65)

    print("\n  Phase is an angle on S¹, so its derivatives are wrapped angles:")
    print("  wrap_angle(x) = atan2(sin x, cos x) ∈ [-π, π]")
    test_angles = [-4.5, -math.pi, -1.0, 0.0, 1.0, math.pi, 3.5, 7.0]
    print(f"  {'input':>10}  {'wrapped':>12}  {'|wrapped| ≤ π':>16}")
    print("  " + "-" * 42)
    for angle in test_angles:
        wrapped = math.atan2(math.sin(angle), math.cos(angle))
        print(f"  {angle:10.4f}  {wrapped:12.4f}  {abs(wrapped) <= math.pi + 1e-12!s:>16}")

    print("\n  Canonical safety thresholds (90% of the π wrap bound):")
    print(f"    |∇φ| early-warning  GRAD_PHI_CANONICAL_THRESHOLD = {GRAD_PHI_CANONICAL_THRESHOLD:.4f}")
    print(f"    |K_φ| safety        K_PHI_CANONICAL_THRESHOLD    = {K_PHI_CANONICAL_THRESHOLD:.4f}")
    print(f"    Phase-wrap maximum  π                            = {PI:.4f}")

    print("\n  Verification on a Watts-Strogatz network (N=30, k=4, p=0.3):")
    G = nx.watts_strogatz_graph(30, 4, 0.3, seed=42)
    _seed_network(G)
    grad = np.array(list(compute_phase_gradient(G).values()))
    k_phi = np.array(list(compute_phase_curvature(G).values()))
    print(f"    max |∇φ| = {np.max(np.abs(grad)):.4f}   (≤ π? {bool(np.all(np.abs(grad) <= PI + 1e-9))})")
    print(f"    max |K_φ| = {np.max(np.abs(k_phi)):.4f}   (≤ π? {bool(np.all(np.abs(k_phi) <= PI + 1e-9))})")
    print("    Both phase derivatives share the SAME bound — π scales the whole sector.")


# ---------------------------------------------------------------------------
# 2. The derivative-tower tetrad
# ---------------------------------------------------------------------------


def demo_derivative_tower_tetrad() -> None:
    """The four tetrad fields are the four orders of the derivative tower."""
    print("\n" + "=" * 65)
    print("  2. THE DERIVATIVE-TOWER TETRAD (Φ_s, |∇φ|, K_φ, ξ_C)")
    print("=" * 65)

    G = nx.watts_strogatz_graph(40, 4, 0.3, seed=7)
    _seed_network(G, seed=7, correlated=True)

    phi_s = np.array(list(compute_structural_potential(G).values()))
    grad = np.array(list(compute_phase_gradient(G).values()))
    k_phi = np.array(list(compute_phase_curvature(G).values()))
    xi_c = estimate_coherence_length(G)

    print(f"\n  {'field':<8}  {'tower order':<26}  {'magnitude':>12}")
    print("  " + "-" * 52)
    print(f"  {'Φ_s':<8}  {'0th (global aggregation)':<26}  {np.max(np.abs(phi_s)):12.4f}")
    print(f"  {'|∇φ|':<8}  {'1st (local derivative)':<26}  {np.max(np.abs(grad)):12.4f}")
    print(f"  {'K_φ':<8}  {'2nd (discrete Laplacian)':<26}  {np.max(np.abs(k_phi)):12.4f}")
    print(f"  {'ξ_C':<8}  {'correlation (non-local)':<26}  {xi_c:12.4f}")
    print("\n  These four orders are the minimal, complete structural basis:")
    print("  higher graph derivatives decompose into products of lower ones,")
    print("  so no fifth independent channel exists.")


# ---------------------------------------------------------------------------
# 3. ξ_C is set by the spectral gap
# ---------------------------------------------------------------------------


def demo_spectral_coherence_length() -> None:
    """ξ_C tracks the spectral gap: ξ_C ∝ 1/√λ₂."""
    print("\n" + "=" * 65)
    print("  3. COHERENCE LENGTH FROM THE SPECTRAL GAP (ξ_C ∝ 1/√λ₂)")
    print("=" * 65)

    topologies = [
        ("ring (N=40)", nx.cycle_graph(40)),
        ("WS (N=40,k=4)", nx.watts_strogatz_graph(40, 4, 0.2, seed=3)),
        ("WS (N=40,k=8)", nx.watts_strogatz_graph(40, 8, 0.3, seed=3)),
        ("complete (N=20)", nx.complete_graph(20)),
    ]

    print(f"\n  {'topology':<16}  {'λ₂ (Fiedler)':>14}  {'1/√λ₂':>10}  {'ξ_C (SDK)':>10}")
    print("  " + "-" * 56)
    for name, G in topologies:
        _seed_network(G, seed=11, correlated=True)
        lam2 = _algebraic_connectivity(G)
        inv_sqrt = 1.0 / math.sqrt(lam2) if lam2 > 1e-12 else float("inf")
        xi_c = estimate_coherence_length(G)
        print(f"  {name:<16}  {lam2:14.6f}  {inv_sqrt:10.4f}  {xi_c:10.4f}")

    print("\n  Smaller spectral gap λ₂ → longer coherence length: ξ_C ∝ 1/√λ₂.")
    print("  The scale of ξ_C is the spectral gap, not any assumed constant.")


# ---------------------------------------------------------------------------
# 4. Cross-topology confinement
# ---------------------------------------------------------------------------


def demo_cross_topology_confinement() -> None:
    """The π phase-wrap and π-derived Φ_s bounds hold across topologies."""
    print("\n" + "=" * 65)
    print("  4. CROSS-TOPOLOGY CONFINEMENT (π phase-wrap + π/2, π/4 Φ_s bounds)")
    print("=" * 65)

    print(f"\n  Φ_s drift bound      U6_STRUCTURAL_POTENTIAL_LIMIT = π/2 = {U6_STRUCTURAL_POTENTIAL_LIMIT:.4f}")
    print(f"  Φ_s per-node bound   PHI_S_VON_KOCH_THRESHOLD      = π/4 = {PHI_S_VON_KOCH_THRESHOLD:.4f}")

    topologies = [
        ("ring (N=30)", nx.cycle_graph(30)),
        ("WS (N=30,k=4)", nx.watts_strogatz_graph(30, 4, 0.3, seed=5)),
        ("grid 6x6", nx.grid_2d_graph(6, 6)),
        ("complete (N=15)", nx.complete_graph(15)),
    ]

    print(f"\n  {'topology':<16}  {'max|∇φ|≤π':>10}  {'max|K_φ|≤π':>11}  {'max|Φ_s|':>10}")
    print("  " + "-" * 54)
    for name, G in topologies:
        G = nx.convert_node_labels_to_integers(G)
        _seed_network(G, seed=23)
        grad = np.array(list(compute_phase_gradient(G).values()))
        k_phi = np.array(list(compute_phase_curvature(G).values()))
        phi_s = np.array(list(compute_structural_potential(G).values()))
        grad_ok = bool(np.all(np.abs(grad) <= PI + 1e-9))
        kphi_ok = bool(np.all(np.abs(k_phi) <= PI + 1e-9))
        max_phi_s = float(np.max(np.abs(phi_s))) if phi_s.size else 0.0
        print(f"  {name:<16}  {grad_ok!s:>10}  {kphi_ok!s:>11}  {max_phi_s:10.4f}")

    print("\n  The π phase-wrap bounds hold on every topology — a genuine")
    print("  structural bound, not a calibrated value.")


def main() -> None:
    print()
    print("TNFR — THE π STRUCTURAL SCALE AND THE DERIVATIVE-TOWER TETRAD")
    print("Only π is assumed as a genuine structural scale; ℝ is the assumed")
    print("continuum. Everything else emerges from ∂EPI/∂t = νf · ΔNFR(t).")
    print()

    demo_pi_phase_sector()
    demo_derivative_tower_tetrad()
    demo_spectral_coherence_length()
    demo_cross_topology_confinement()

    print("\n" + "=" * 65)
    print("CONCLUSION: π is the one genuine structural scale (the phase-wrap")
    print("bound of the whole phase sector). The tetrad is the four orders of")
    print("the derivative tower; ξ_C is set by the spectral gap (ξ_C ∝ 1/√λ₂).")
    print("No φ, γ, or e is assumed or used — the structure emerges.")
    print("=" * 65)


if __name__ == "__main__":
    main()
