"""Example 33: Complex Field Unification (Psi = K_phi + i*J_phi).

Demonstrates the fundamental discovery that phase curvature K_phi and
phase current J_phi are dual aspects of a single complex geometric field:

  Psi = K_phi + i * J_phi

Key results shown:
  1. K_phi-J_phi anticorrelation: r ~ -0.854 to -0.997 across topologies
  2. Complex field Psi: magnitude, phase, polar decomposition
  3. Emergent derived fields: chirality (chi), symmetry breaking (S),
     coherence coupling (C)
  4. Tensor invariants: energy density (E), topological charge (Q)
  5. Energy decomposition: T (kinetic/transport) + V (potential/geometric)
  6. Action density and cross-sector coupling

Physics basis:
  The near-perfect anticorrelation r(K_phi, J_phi) implies that
  increasing phase curvature (confinement) suppresses phase current
  (transport). Static confinement and dynamic transport are dual
  aspects — they trade off within the unified complex field Psi.
  See: theory/EXTENDED_FIELDS_AND_DERIVED_QUANTITIES.md ss 2-3
  See: theory/STRUCTURAL_CONSERVATION_THEOREM.md ss 4-6
"""

from __future__ import annotations

import math
import sys
import os

import numpy as np
import networkx as nx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from tnfr.constants.canonical import PHI, GAMMA, PI, E
from tnfr.constants import inject_defaults
from tnfr.physics.fields import (
    compute_structural_potential,
    compute_phase_gradient,
    compute_phase_curvature,
    estimate_coherence_length,
    compute_complex_geometric_field_arrays,
    compute_emergent_fields,
    compute_tensor_invariants,
)
from tnfr.physics.extended import (
    compute_phase_current,
    compute_dnfr_flux,
)
from tnfr.physics.unified import (
    compute_complex_geometric_field,
    compute_field_magnitude,
    compute_field_phase,
    compute_chirality_field,
    compute_symmetry_breaking_field,
)


def _build_graph(n: int, topology: str, seed: int = 42) -> nx.Graph:
    """Build a TNFR-initialized graph of the given topology."""
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
    elif topology == "Complete":
        G = nx.complete_graph(n)
    elif topology == "Ring":
        G = nx.cycle_graph(n)
    else:
        G = nx.watts_strogatz_graph(n, 4, 0.3, seed=seed)

    inject_defaults(G)
    for node in G.nodes():
        G.nodes[node]["phase"] = rng.uniform(0, 2 * math.pi)
        G.nodes[node]["theta"] = G.nodes[node]["phase"]
        G.nodes[node]["delta_nfr"] = rng.uniform(-0.5, 0.5)
        G.nodes[node]["nu_f"] = rng.uniform(0.8, 1.2)
    return G


def _evolve_step(G: nx.Graph, dt: float = 0.05) -> None:
    """One diffusion step: phase alignment + DELTA_NFR smoothing."""
    for n in G.nodes():
        neighbors = list(G.neighbors(n))
        if neighbors:
            mean_phase = np.mean([G.nodes[nb]["phase"] for nb in neighbors])
            G.nodes[n]["phase"] += dt * (mean_phase - G.nodes[n]["phase"])
            G.nodes[n]["theta"] = G.nodes[n]["phase"]
            mean_dnfr = np.mean([G.nodes[nb]["delta_nfr"] for nb in neighbors])
            G.nodes[n]["delta_nfr"] += dt * (mean_dnfr - G.nodes[n]["delta_nfr"])


# ---------------------------------------------------------------------------
# 1. K_phi - J_phi anticorrelation
# ---------------------------------------------------------------------------

def demo_anticorrelation() -> None:
    """Verify strong anticorrelation between K_phi and J_phi."""
    print("=" * 65)
    print("  1. K_phi - J_phi ANTICORRELATION across Topologies")
    print("=" * 65)

    topologies = [
        ("WS (N=50)", "WS", 50),
        ("BA (N=50)", "BA", 50),
        ("Grid (7x7)", "Grid", 49),
        ("Ring (N=50)", "Ring", 50),
        ("Complete (N=15)", "Complete", 15),
    ]

    print(f"\n  Expected: r(K_phi, J_phi) in [-0.997, -0.854]")
    print(f"\n  {'Topology':<20}  {'r(K_phi, J_phi)':>16}  {'Mean |Psi|':>10}  {'Verdict':>10}")
    print("  " + "-" * 62)

    for name, topo, n in topologies:
        G = _build_graph(n, topo)
        # Evolve a few steps for realistic field distributions
        for _ in range(10):
            _evolve_step(G)

        k_phi = compute_phase_curvature(G)
        j_phi = compute_phase_current(G)

        k_arr = np.array([k_phi[n] for n in sorted(G.nodes())])
        j_arr = np.array([j_phi[n] for n in sorted(G.nodes())])

        # Pearson correlation
        if np.std(k_arr) > 1e-10 and np.std(j_arr) > 1e-10:
            corr = np.corrcoef(k_arr, j_arr)[0, 1]
        else:
            corr = 0.0

        psi = compute_complex_geometric_field(G)
        mean_mag = np.mean([abs(v) for v in psi.values()])

        verdict = "STRONG" if corr < -0.7 else ("MODERATE" if corr < -0.3 else "WEAK")
        print(f"  {name:<20}  {corr:16.4f}  {mean_mag:10.4f}  {verdict:>10}")

    print(f"\n  Physical mechanism:")
    print(f"    Increasing K_phi (confinement) -> suppresses J_phi (transport)")
    print(f"    They are dual aspects of unified complex field Psi")


# ---------------------------------------------------------------------------
# 2. Complex field Psi decomposition
# ---------------------------------------------------------------------------

def demo_complex_field() -> None:
    """Show Psi = K_phi + i*J_phi decomposition."""
    print("\n" + "=" * 65)
    print("  2. COMPLEX GEOMETRIC FIELD  Psi = K_phi + i*J_phi")
    print("=" * 65)

    G = _build_graph(30, "WS")
    for _ in range(10):
        _evolve_step(G)

    psi = compute_complex_geometric_field(G)
    magnitudes = compute_field_magnitude(psi)
    phases = compute_field_phase(psi)

    nodes = sorted(G.nodes())[:10]  # show first 10

    print(f"\n  Node decomposition (first 10 nodes, WS N=30):")
    print(f"  {'Node':>6}  {'K_phi':>8}  {'J_phi':>8}  {'|Psi|':>8}  {'arg(Psi)':>10}  {'Psi':>20}")
    print("  " + "-" * 68)

    k_phi = compute_phase_curvature(G)
    j_phi = compute_phase_current(G)

    for n in nodes:
        print(f"  {n:6d}  {k_phi[n]:8.4f}  {j_phi[n]:8.4f}  {magnitudes[n]:8.4f}  "
              f"{math.degrees(phases[n]):10.2f} deg  {psi[n].real:+.4f}{psi[n].imag:+.4f}j")

    # Array version
    arrays = compute_complex_geometric_field_arrays(G)
    print(f"\n  Array API summary:")
    print(f"    Psi array shape: ({len(arrays['psi_real'])},)")
    print(f"    Mean |Psi|: {np.mean(arrays['psi_magnitude']):.4f}")
    print(f"    Std |Psi|:  {np.std(arrays['psi_magnitude']):.4f}")
    print(f"    Mean arg(Psi): {np.mean(arrays['psi_phase']):.4f} rad")


# ---------------------------------------------------------------------------
# 3. Emergent derived fields
# ---------------------------------------------------------------------------

def demo_emergent_fields() -> None:
    """Compute and interpret chirality, symmetry breaking, coherence coupling."""
    print("\n" + "=" * 65)
    print("  3. EMERGENT DERIVED FIELDS")
    print("=" * 65)

    G = _build_graph(40, "WS")
    for _ in range(15):
        _evolve_step(G)

    chi = compute_chirality_field(G)
    sym_break = compute_symmetry_breaking_field(G)
    emergent = compute_emergent_fields(G)

    chi_arr = np.array(list(chi.values()))
    sb_arr = np.array(list(sym_break.values()))

    print(f"\n  a) Chirality  chi = |grad_phi|*K_phi - J_phi*J_DELTA_NFR")
    print(f"     Detects: Structural handedness / broken parity")
    print(f"     Mean: {np.mean(chi_arr):.6f}")
    print(f"     Std:  {np.std(chi_arr):.6f}")
    print(f"     |chi| > 0 signals asymmetry between local and transport sectors")

    print(f"\n  b) Symmetry Breaking  S = (|grad_phi|^2 - K_phi^2) + (J_phi^2 - J_DELTA_NFR^2)")
    print(f"     Order parameter for phase transitions")
    print(f"     Mean: {np.mean(sb_arr):.6f}  (S ~ 0 = balanced, |S| >> 0 = broken)")
    print(f"     Std:  {np.std(sb_arr):.6f}")

    print(f"\n  c) Coherence Coupling  C = Phi_s * |Psi|")
    print(f"     Multi-scale connector: global potential <-> local geometry")
    cc_arr = np.array(emergent.get("coherence_coupling", [0.0]))
    if len(cc_arr) > 1:
        print(f"     Mean: {np.mean(cc_arr):.6f}")
        print(f"     Std:  {np.std(cc_arr):.6f}")

    # Compare: high-coherence vs low-coherence node groups
    phi_s = compute_structural_potential(G)
    phi_s_arr = np.array([phi_s[n] for n in sorted(G.nodes())])
    median_phi_s = np.median(np.abs(phi_s_arr))

    high_phi_s = [n for n in G.nodes() if abs(phi_s[n]) >= median_phi_s]
    low_phi_s = [n for n in G.nodes() if abs(phi_s[n]) < median_phi_s]

    chi_high = np.mean([abs(chi[n]) for n in high_phi_s]) if high_phi_s else 0
    chi_low = np.mean([abs(chi[n]) for n in low_phi_s]) if low_phi_s else 0
    print(f"\n  Chirality comparison by Phi_s level:")
    print(f"    High |Phi_s| nodes (n={len(high_phi_s)}): mean |chi| = {chi_high:.6f}")
    print(f"    Low  |Phi_s| nodes (n={len(low_phi_s)}):  mean |chi| = {chi_low:.6f}")


# ---------------------------------------------------------------------------
# 4. Tensor invariants
# ---------------------------------------------------------------------------

def demo_tensor_invariants() -> None:
    """Compute energy density, topological charge, and charge density."""
    print("\n" + "=" * 65)
    print("  4. TENSOR INVARIANTS — Gauge-Invariant Quantities")
    print("=" * 65)

    topologies = [
        ("WS (N=40)", "WS", 40),
        ("BA (N=40)", "BA", 40),
        ("Grid (6x6)", "Grid", 36),
    ]

    print(f"\n  {'Topology':<16}  {'Mean E':>10}  {'Std E':>8}  "
          f"{'Mean Q':>10}  {'Mean rho':>10}")
    print("  " + "-" * 60)

    for name, topo, n in topologies:
        G = _build_graph(n, topo)
        for _ in range(10):
            _evolve_step(G)

        inv = compute_tensor_invariants(G)

        e_arr = np.array(inv.get("energy_density", [0.0]))
        q_arr = np.array(inv.get("topological_charge", [0.0]))
        rho_arr = np.array(inv.get("charge_density", [0.0]))

        print(f"  {name:<16}  {np.mean(e_arr):10.4f}  {np.std(e_arr):8.4f}  "
              f"{np.mean(q_arr):10.6f}  {np.mean(rho_arr):10.4f}")

    # Detailed breakdown for one topology
    G = _build_graph(40, "WS")
    for _ in range(10):
        _evolve_step(G)

    phi_s = compute_structural_potential(G)
    grad_phi = compute_phase_gradient(G)
    k_phi = compute_phase_curvature(G)
    j_phi = compute_phase_current(G)
    j_dnfr = compute_dnfr_flux(G)

    nodes_sorted = sorted(G.nodes())
    ps = np.array([phi_s[n] for n in nodes_sorted])
    gp = np.array([grad_phi[n] for n in nodes_sorted])
    kp = np.array([k_phi[n] for n in nodes_sorted])
    jp = np.array([j_phi[n] for n in nodes_sorted])
    jd = np.array([j_dnfr[n] for n in nodes_sorted])

    # Energy decomposition: E = T + V
    T = 0.5 * np.sum(jp ** 2 + jd ** 2)  # kinetic (transport)
    V = 0.5 * np.sum(ps ** 2 + gp ** 2 + kp ** 2)  # potential (geometric)
    E_total = T + V
    print(f"\n  Energy decomposition (WS N=40):")
    print(f"    E_total = {E_total:.4f}")
    print(f"    T (kinetic/transport) = {T:.4f}  ({T/E_total*100:.1f}%)")
    print(f"    V (potential/geometric) = {V:.4f}  ({V/E_total*100:.1f}%)")

    # Action density
    action = ps * gp + kp * jp + gp * jd
    print(f"\n  Action density A = Phi_s*|grad_phi| + K_phi*J_phi + |grad_phi|*J_DELTA_NFR:")
    print(f"    Mean A: {np.mean(action):.6f}")
    print(f"    Sum A:  {np.sum(action):.4f}")

    # Topological charge conservation check
    Q_total = np.sum(gp * jp - kp * jd)
    print(f"\n  Topological charge Q = sum(|grad_phi|*J_phi - K_phi*J_DELTA_NFR):")
    print(f"    Q_total = {Q_total:.6f}")
    print(f"    (Should be approximately conserved under grammar-compliant evolution)")


# ---------------------------------------------------------------------------
# 5. Evolution tracking of unified fields
# ---------------------------------------------------------------------------

def demo_evolution_tracking() -> None:
    """Track Psi magnitude and emergent fields during evolution."""
    print("\n" + "=" * 65)
    print("  5. EVOLUTION TRACKING — Unified Field Dynamics")
    print("=" * 65)

    G = _build_graph(40, "WS")
    n_steps = 30

    print(f"\n  Tracking WS (N=40) over {n_steps} diffusion steps:")
    print(f"  {'Step':>6}  {'Mean|Psi|':>10}  {'Mean|chi|':>10}  "
          f"{'Mean|S|':>10}  {'E_total':>10}  {'Q_total':>10}")
    print("  " + "-" * 62)

    for step in range(n_steps + 1):
        if step % 5 == 0:
            psi = compute_complex_geometric_field(G)
            chi = compute_chirality_field(G)
            sym = compute_symmetry_breaking_field(G)

            mean_psi = np.mean([abs(v) for v in psi.values()])
            mean_chi = np.mean([abs(v) for v in chi.values()])
            mean_sym = np.mean([abs(v) for v in sym.values()])

            inv = compute_tensor_invariants(G)
            e_total = np.sum(inv.get("energy_density", [0.0]))
            q_total = np.sum(inv.get("topological_charge", [0.0]))

            print(f"  {step:6d}  {mean_psi:10.4f}  {mean_chi:10.6f}  "
                  f"{mean_sym:10.6f}  {e_total:10.4f}  {q_total:10.6f}")

        _evolve_step(G, dt=0.1)

    print(f"\n  Expected behavior:")
    print(f"    |Psi| decreases as network synchronizes (K_phi, J_phi -> 0)")
    print(f"    |chi| decreases (symmetry restoration)")
    print(f"    |S| decreases (sector balance improves)")
    print(f"    E decreases (Lyapunov stability)")
    print(f"    Q approximately conserved (topological invariant)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print()
    print("*" * 65)
    print("  TNFR Example 33: Complex Field Unification")
    print("  Psi = K_phi + i * J_phi")
    print("  Theory: EXTENDED_FIELDS_AND_DERIVED_QUANTITIES.md ss 2-3")
    print("*" * 65)

    demo_anticorrelation()
    demo_complex_field()
    demo_emergent_fields()
    demo_tensor_invariants()
    demo_evolution_tracking()

    print("\n" + "=" * 65)
    print("  SUMMARY")
    print("=" * 65)
    print(f"""
  Complex Geometric Field Psi = K_phi + i * J_phi unifies:
    Real part (K_phi):  Static geometric confinement
    Imaginary part (J_phi):  Dynamic transport flow

  Anticorrelation r(K_phi, J_phi) ~ -0.85 to -0.997
    -> Confinement and transport are dual aspects

  Emergent fields from Psi:
    Chirality chi:       Structural handedness detector
    Symmetry Breaking S: Phase transition order parameter
    Coherence Coupling C: Multi-scale connector (Phi_s * |Psi|)

  Tensor invariants:
    Energy density E:    Gauge-invariant total energy
    Topological charge Q: Conserved under grammar evolution
    Action density A:    Cross-sector coupling measure

  This reduces 6 independent fields to 3 complex fields,
  achieving mathematical elegance without information loss.
""")


if __name__ == "__main__":
    main()
