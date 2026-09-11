"""Example 31: Exact π phase bounds and tetrad scale diagnostics.

Demonstrates the exact wrapped-angle bound π and reports the four canonical
structural diagnostics. The example organizes these read-outs by derivative
role; it does not prove that they form a minimal or state-complete basis.

Here π is the exact geometric scale of the whole wrapped phase sector,
since both phase derivatives are wrapped angles:

    |∇φ| ≤ π   and   |K_φ| ≤ π .

The four tetrad fields can be organized by these diagnostic roles:

    Φ_s   (0th order, global aggregation)   ΔNFR_j → Σ 1/d²
    |∇φ|  (1st order, local derivative)      φ_i   → ∇
    K_φ   (2nd order, local curvature)       φ_i   → circular curvature
    ξ_C   (correlation, non-local)           correlation range

The coherence-length estimator first attempts a state-dependent correlation
fit. When that fit is unavailable, it uses 1/√λ₂ as a spectral fallback on a
connected undirected graph.

Key results shown:
  1. π bounds the whole phase sector: both |∇φ| ≤ π and |K_φ| ≤ π.
  2. The derivative-tower tetrad (Φ_s, |∇φ|, K_φ, ξ_C) on a network.
  3. Fitted and spectral-fallback ξ_C values retain their provenance.
  4. Exact phase bounds are separated from selected Φ_s and phase-warning
     policies across several finite graph examples.

Physics basis:
  π is the exact phase-wrap scale. The π/16 phase-gradient warning, 0.9π
  curvature margin, π/4 potential magnitude warning, and π/2 U6 drift value
  are selected monitoring policies. A fixed graph spectrum supplies a useful
  length scale, while a fitted correlation length remains state-dependent.
  See: theory/MATHEMATICAL_DYNAMICS_BASIS.md
  See: theory/MINIMAL_STRUCTURAL_DEGREES.md §§ 4-5
"""

from __future__ import annotations

import math
import os
import sys

import networkx as nx
import numpy as np

# Preserve mathematical symbols when this script runs in a legacy Windows
# console whose inherited text encoding cannot represent them.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

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
    estimate_coherence_length_with_provenance,
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


def _structural_spectral_gap(G: "nx.Graph") -> float:
    """Emergent structural spectral gap λ₂ — the second-smallest eigenvalue of
    the canonical symmetric normalized Laplacian L_sym (same spectrum as the
    random-walk diffusion operator L_rw = I − D⁻¹W that the ΔNFR realises).
    Its inverse square root is the estimator's spectral fallback and a model
    comparison, distinct from a state-dependent correlation fit."""
    from tnfr.physics.structural_diffusion import symmetric_normalized_laplacian

    _, l_sym = symmetric_normalized_laplacian(G)
    eigvals = np.sort(np.clip(np.linalg.eigvalsh(l_sym), 0.0, None))
    return float(eigvals[1]) if len(eigvals) > 1 else 0.0


# ---------------------------------------------------------------------------
# 1. π bounds the whole phase sector
# ---------------------------------------------------------------------------


def demo_pi_phase_sector() -> None:
    """π bounds BOTH phase derivatives: |∇φ| ≤ π and |K_φ| ≤ π."""
    print("=" * 65)
    print("  1. π — THE EXACT WRAPPED-PHASE SCALE")
    print("=" * 65)

    print("\n  Phase is an angle on S¹, so its derivatives are wrapped angles:")
    print("  wrap_angle(x) = atan2(sin x, cos x) ∈ [-π, π]")
    test_angles = [-4.5, -math.pi, -1.0, 0.0, 1.0, math.pi, 3.5, 7.0]
    print(f"  {'input':>10}  {'wrapped':>12}  {'|wrapped| ≤ π':>16}")
    print("  " + "-" * 42)
    for angle in test_angles:
        wrapped = math.atan2(math.sin(angle), math.cos(angle))
        print(f"  {angle:10.4f}  {wrapped:12.4f}  {abs(wrapped) <= math.pi + 1e-12!s:>16}")

    print("\n  Exact phase bound and selected monitoring policies:")
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
    """Report the tetrad organized by derivative and correlation role."""
    print("\n" + "=" * 65)
    print("  2. TETRAD DIAGNOSTIC ROLES (Φ_s, |∇φ|, K_φ, ξ_C)")
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
    print(
        f"  {'K_φ':<8}  {'2nd (circular curvature)':<26}"
        f"  {np.max(np.abs(k_phi)):12.4f}"
    )
    print(f"  {'ξ_C':<8}  {'correlation (non-local)':<26}  {xi_c:12.4f}")
    print("\n  The tetrad exposes complementary aggregation, phase-derivative,")
    print("  curvature, and correlation read-outs. This snapshot does not")
    print("  establish complete reconstruction of graph state or dynamics.")


# ---------------------------------------------------------------------------
# 3. ξ_C estimator provenance and spectral comparison
# ---------------------------------------------------------------------------


def demo_spectral_coherence_length() -> None:
    """Compare ξ_C estimates with the 1/√λ₂ fallback scale."""
    print("\n" + "=" * 65)
    print("  3. COHERENCE-LENGTH PROVENANCE AND SPECTRAL COMPARISON")
    print("=" * 65)

    topologies = [
        ("ring (N=40)", nx.cycle_graph(40)),
        ("WS (N=40,k=4)", nx.watts_strogatz_graph(40, 4, 0.2, seed=3)),
        ("WS (N=40,k=8)", nx.watts_strogatz_graph(40, 8, 0.3, seed=3)),
        ("complete (N=20)", nx.complete_graph(20)),
    ]

    print(
        f"\n  {'topology':<16}  {'λ₂ (L_sym gap)':>14}  {'1/√λ₂':>10}"
        f"  {'ξ_C':>10}  {'method':>19}"
    )
    print("  " + "-" * 78)
    for name, G in topologies:
        _seed_network(G, seed=11, correlated=True)
        lam2 = _structural_spectral_gap(G)
        inv_sqrt = 1.0 / math.sqrt(lam2) if lam2 > 1e-12 else float("inf")
        estimate = estimate_coherence_length_with_provenance(G)
        print(
            f"  {name:<16}  {lam2:14.6f}  {inv_sqrt:10.4f}"
            f"  {estimate.value:10.4f}  {estimate.method:>19}"
        )

    print("\n  The spectral column is an exact graph-derived comparison scale.")
    print("  ξ_C equals it only when the estimator reports spectral_gap;")
    print("  autocorrelation_fit values also depend on the seeded state.")


# ---------------------------------------------------------------------------
# 4. Cross-topology exact bounds and selected policies
# ---------------------------------------------------------------------------


def demo_cross_topology_confinement() -> None:
    """Separate exact phase bounds from selected field-warning policies."""
    print("\n" + "=" * 65)
    print("  4. CROSS-TOPOLOGY PHASE BOUNDS AND MONITORING POLICIES")
    print("=" * 65)

    print(
        f"\n  U6 drift policy       U6_STRUCTURAL_POTENTIAL_LIMIT = π/2 = "
        f"{U6_STRUCTURAL_POTENTIAL_LIMIT:.4f}"
    )
    print(
        f"  Φ_s magnitude policy PHI_S_VON_KOCH_THRESHOLD      = π/4 = "
        f"{PHI_S_VON_KOCH_THRESHOLD:.4f}"
    )

    topologies = [
        ("ring (N=30)", nx.cycle_graph(30)),
        ("WS (N=30,k=4)", nx.watts_strogatz_graph(30, 4, 0.3, seed=5)),
        ("grid 6x6", nx.grid_2d_graph(6, 6)),
        ("complete (N=15)", nx.complete_graph(15)),
    ]

    print(
        f"\n  {'topology':<16}  {'max|∇φ|≤π':>10}  {'max|K_φ|≤π':>11}"
        f"  {'max|Φ_s|':>10}  {'<π/4 policy':>12}"
    )
    print("  " + "-" * 70)
    for name, G in topologies:
        G = nx.convert_node_labels_to_integers(G)
        _seed_network(G, seed=23)
        grad = np.array(list(compute_phase_gradient(G).values()))
        k_phi = np.array(list(compute_phase_curvature(G).values()))
        phi_s = np.array(list(compute_structural_potential(G).values()))
        grad_ok = bool(np.all(np.abs(grad) <= PI + 1e-9))
        kphi_ok = bool(np.all(np.abs(k_phi) <= PI + 1e-9))
        max_phi_s = float(np.max(np.abs(phi_s))) if phi_s.size else 0.0
        phi_policy = max_phi_s < PHI_S_VON_KOCH_THRESHOLD
        print(
            f"  {name:<16}  {grad_ok!s:>10}  {kphi_ok!s:>11}"
            f"  {max_phi_s:10.4f}  {phi_policy!s:>12}"
        )

    print("\n  The phase columns test exact wrapped-angle bounds.")
    print("  The Φ_s column tests a selected magnitude policy; it may be crossed.")
    print("  U6 drift is not tested because this example has no reference snapshot.")


def main() -> None:
    print()
    print("TNFR — EXACT π PHASE BOUNDS AND TETRAD SCALE DIAGNOSTICS")
    print("The nodal equation anchors the state update; this example separates")
    print("exact identities, selected policies, and estimator-dependent output.")
    print()

    demo_pi_phase_sector()
    demo_derivative_tower_tetrad()
    demo_spectral_coherence_length()
    demo_cross_topology_confinement()

    print("\n" + "=" * 65)
    print("CONCLUSION: π exactly bounds the wrapped phase sector. The tetrad")
    print("provides complementary diagnostics, while its warning levels are")
    print("selected policies and ξ_C reports its fitted or spectral provenance.")
    print("=" * 65)


if __name__ == "__main__":
    main()
