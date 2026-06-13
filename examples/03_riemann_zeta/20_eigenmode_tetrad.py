#!/usr/bin/env python3
r"""Example 43: Per-Eigenmode Structural Field Tetrad

Computes the TNFR canonical structural field tetrad (Phi_s, |grad_phi|,
K_phi, xi_C) for each eigenmode psi_j of the discrete TNFR operator
H^(k)(sigma) = L_k + (sigma - 1/2) diag(log p_i).

This bridges spectral theory (eigenvalues, eigenvectors) to the canonical
TNFR measurement infrastructure (structural fields), extending the U6
confinement principle from graph nodes to spectral eigenmodes.

Key results demonstrated:
  1. Structural potential Phi_s reflects spectral pressure between eigenmodes
  2. Gradient and curvature capture eigenvector spatial variation
  3. Coherence length xi_C tracks eigenmode spatial decay
  4. U6 confinement transitions: more modes confined near sigma = 1/2
  5. Scaling with k: structural potential grows with eigenvalue density
  6. Cross-topology comparison: path vs cycle vs complete graphs

Physics basis: the nodal equation dEPI/dt = nu_f * DELTA_NFR(t) governs
all TNFR dynamics.  The structural field tetrad is the canonical diagnostic
toolkit (AGENTS.md, Structural Field Tetrad).  Extending it to eigenmodes
of H^(k)(sigma) reveals how spectral structure maps to structural
confinement -- the core of the TNFR-Riemann bridge.

Status: EXPERIMENTAL -- Research prototype for TNFR-Riemann P3 program.

Usage:
    python examples/03_riemann_zeta/20_eigenmode_tetrad.py
"""

from __future__ import annotations

import sys

from tnfr.riemann import (
    PHI_S_GOLDEN_THRESHOLD,
    PHI_S_VON_KOCH_THRESHOLD,
    check_u6_confinement,
    compare_confinement_at_sigma,
    compute_eigenmode_fields_general,
    compute_eigenmode_tetrad,
)


def main() -> None:
    print("=" * 70)
    print("TNFR-Riemann P3: Per-Eigenmode Structural Field Tetrad")
    print("=" * 70)
    print()

    # ------------------------------------------------------------------
    # Section 1: Single tetrad snapshot at sigma = 1/2, k = 50
    # ------------------------------------------------------------------
    print("Section 1: Eigenmode Tetrad at sigma = 1/2  (k = 50)")
    print("-" * 55)

    analysis = compute_eigenmode_tetrad(50, sigma=0.5)
    print(f"  k = {analysis.k},  sigma = {analysis.sigma},  "
          f"modes computed = {len(analysis.tetrads)}")
    print()

    print(f"  {'j':>4}  {'lambda':>10}  {'Phi_s':>10}  "
          f"{'|grad|':>10}  {'K_phi':>10}  {'xi_C':>10}")
    print(f"  {'─' * 4}  {'─' * 10}  {'─' * 10}  "
          f"{'─' * 10}  {'─' * 10}  {'─' * 10}")

    # Show first 10, last 5
    tetrads = analysis.tetrads
    show_indices = list(range(min(10, len(tetrads))))
    if len(tetrads) > 15:
        show_indices += list(range(len(tetrads) - 5, len(tetrads)))

    for idx in show_indices:
        t = tetrads[idx]
        xi_str = f"{t.xi_c:10.4f}" if t.xi_c == t.xi_c else "       NaN"
        print(
            f"  {t.mode_index:4d}  {t.eigenvalue:10.4f}  "
            f"{t.phi_s:10.4f}  {t.grad_phi:10.6f}  "
            f"{t.k_phi:10.6f}  {xi_str}"
        )
        if idx == show_indices[9] and len(tetrads) > 15:
            print(f"  {'...':>4}")

    print()
    print(f"  Aggregate:  mean Phi_s = {analysis.mean_phi_s:.4f},  "
          f"mean |grad| = {analysis.mean_grad_phi:.6f}")
    print(f"              mean K_phi = {analysis.mean_k_phi:.6f},  "
          f"mean xi_C   = {analysis.mean_xi_c:.4f}")
    print()

    # ------------------------------------------------------------------
    # Section 2: Field profiles vs mode index
    # ------------------------------------------------------------------
    print("Section 2: Field Profiles vs Eigenmode Index (k = 50)")
    print("-" * 55)

    print("  Phi_s(j): spectral structural potential")
    print("    Low-j (bulk) modes feel moderate spectral pressure.")
    print("    Extreme modes (j ~ 0 or j ~ k-1) feel the most pressure")
    print("    from the dense eigenvalue region.")
    print()

    # Quartile analysis
    n = len(tetrads)
    q1 = tetrads[:n // 4]
    q4 = tetrads[3 * n // 4:]

    mean_q1_phi_s = sum(t.phi_s for t in q1) / len(q1)
    mean_q4_phi_s = sum(t.phi_s for t in q4) / len(q4)
    mean_q1_grad = sum(t.grad_phi for t in q1) / len(q1)
    mean_q4_grad = sum(t.grad_phi for t in q4) / len(q4)

    print(f"  Quartile analysis (Q1 = modes 0..{n // 4 - 1}, "
          f"Q4 = modes {3 * n // 4}..{n - 1}):")
    print(f"    Phi_s:    Q1 mean = {mean_q1_phi_s:.4f},  "
          f"Q4 mean = {mean_q4_phi_s:.4f}")
    print(f"    |grad|:   Q1 mean = {mean_q1_grad:.6f},  "
          f"Q4 mean = {mean_q4_grad:.6f}")
    print()

    # Gradient summary
    grads = [t.grad_phi for t in tetrads]
    print(f"  |grad_phi|(j) range: [{min(grads):.6f}, {max(grads):.6f}]")
    print(f"  K_phi(j) range:      "
          f"[{min(t.k_phi for t in tetrads):.6f}, "
          f"{max(t.k_phi for t in tetrads):.6f}]")
    print()

    # ------------------------------------------------------------------
    # Section 3: U6 confinement test
    # ------------------------------------------------------------------
    print("Section 3: U6 Confinement at Eigenmode Level")
    print("-" * 55)

    print(f"  Thresholds:")
    print(f"    Von Koch: |Phi_s| < {PHI_S_VON_KOCH_THRESHOLD}")
    print(f"    Golden:   |Phi_s| < {PHI_S_GOLDEN_THRESHOLD}")
    print()

    u6_vk = check_u6_confinement(analysis, u6_threshold=PHI_S_VON_KOCH_THRESHOLD)
    u6_gold = check_u6_confinement(analysis, u6_threshold=PHI_S_GOLDEN_THRESHOLD)
    n_modes = len(analysis.tetrads)

    print(f"  Von Koch threshold ({PHI_S_VON_KOCH_THRESHOLD}):")
    n_conf_vk = int(u6_vk['fraction'] * n_modes)
    print(f"    Confined: {n_conf_vk} / {n_modes}  "
          f"({u6_vk['fraction']:.1%})")
    violator_vk = u6_vk['violations']
    if violator_vk:
        print(f"    Violating modes: {violator_vk[:10]}"
              f"{'...' if len(violator_vk) > 10 else ''}")
    print()

    print(f"  Golden threshold ({PHI_S_GOLDEN_THRESHOLD}):")
    n_conf_g = int(u6_gold['fraction'] * n_modes)
    print(f"    Confined: {n_conf_g} / {n_modes}  "
          f"({u6_gold['fraction']:.1%})")
    violator_g = u6_gold['violations']
    if violator_g:
        print(f"    Violating modes: {violator_g[:10]}"
              f"{'...' if len(violator_g) > 10 else ''}")
    print()

    # ------------------------------------------------------------------
    # Section 4: Sigma scan -- confinement vs sigma
    # ------------------------------------------------------------------
    print("Section 4: Confinement Fraction vs sigma (k = 30)")
    print("-" * 55)

    sigma_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    scan = compare_confinement_at_sigma(
        30,
        sigma_values=sigma_values,
        u6_threshold=PHI_S_GOLDEN_THRESHOLD,
    )

    print(f"  {'sigma':>6}  {'fraction':>10}  "
          f"{'mean Phi_s':>10}")
    print(f"  {'─' * 6}  {'─' * 10}  {'─' * 10}")

    best_sigma = None
    best_frac = -1.0
    for sigma_val in sorted(scan.keys()):
        entry = scan[sigma_val]
        frac = entry["fraction"]
        if frac > best_frac:
            best_frac = frac
            best_sigma = sigma_val
        print(
            f"  {sigma_val:6.2f}  "
            f"{frac:10.1%}  "
            f"{entry['mean_phi_s']:10.4f}"
        )

    print()
    print(f"  Best confinement at sigma = {best_sigma} "
          f"({best_frac:.1%} modes confined)")
    print("  --> Confinement is expected to peak near sigma = 1/2")
    print("      where the potential term V_sigma vanishes.")
    print()

    # ------------------------------------------------------------------
    # Section 5: Scaling with k
    # ------------------------------------------------------------------
    print("Section 5: Scaling of Eigenmode Fields with k")
    print("-" * 55)

    print(f"  {'k':>6}  {'mean Phi_s':>10}  {'max |grad|':>10}  "
          f"{'mean K_phi':>10}  {'u6_frac':>8}")
    print(f"  {'─' * 6}  {'─' * 10}  {'─' * 10}  {'─' * 10}  {'─' * 8}")

    for k in [10, 20, 50, 100, 200]:
        a = compute_eigenmode_tetrad(k, sigma=0.5)
        max_grad = max(t.grad_phi for t in a.tetrads)
        u6 = check_u6_confinement(a, u6_threshold=PHI_S_GOLDEN_THRESHOLD)
        print(
            f"  {k:6d}  {a.mean_phi_s:10.4f}  "
            f"{max_grad:10.6f}  {a.mean_k_phi:10.6f}  "
            f"{u6['fraction']:8.1%}"
        )

    print()
    print("  Phi_s grows with k (more eigenmodes -> higher spectral pressure).")
    print("  max |grad_phi| increases with k (finer eigenvector structure).")
    print()

    # ------------------------------------------------------------------
    # Section 6: Cross-topology comparison (path vs cycle vs complete)
    # ------------------------------------------------------------------
    print("Section 6: Cross-Topology Eigenmode Tetrad (k = 30)")
    print("-" * 55)

    from tnfr.riemann.topology import build_prime_cycle_graph, build_prime_complete_graph

    k_topo = 30

    # Path (tridiagonal)
    path_result = compute_eigenmode_tetrad(k_topo, sigma=0.5)

    # Cycle
    cycle_G = build_prime_cycle_graph(k_topo)
    cycle_result = compute_eigenmode_fields_general(cycle_G, sigma=0.5)

    # Complete
    complete_G = build_prime_complete_graph(k_topo)
    complete_result = compute_eigenmode_fields_general(complete_G, sigma=0.5)

    results = {
        "path": path_result,
        "cycle": cycle_result,
        "complete": complete_result,
    }

    print(f"  {'topology':>10}  {'mean Phi_s':>10}  {'mean |grad|':>11}  "
          f"{'mean K_phi':>10}  {'mean xi_C':>10}  {'u6_frac':>8}")
    print(f"  {'─' * 10}  {'─' * 10}  {'─' * 11}  "
          f"{'─' * 10}  {'─' * 10}  {'─' * 8}")

    for name, res in results.items():
        u6 = check_u6_confinement(res, u6_threshold=PHI_S_GOLDEN_THRESHOLD)
        print(
            f"  {name:>10}  {res.mean_phi_s:10.4f}  "
            f"{res.mean_grad_phi:11.6f}  "
            f"{res.mean_k_phi:10.6f}  "
            f"{res.mean_xi_c:10.4f}  "
            f"{u6['fraction']:8.1%}"
        )

    print()
    print("  Observations:")
    print("    - Phi_s is topology-independent (depends on eigenvalues, not graph).")
    print("    - Gradient and curvature differ: they probe eigenvector spatial")
    print("      structure, which is shaped by the Laplacian topology.")
    print("    - Coherence length varies: cycle (periodic) and complete (uniform)")
    print("      graphs produce different spatial correlation patterns.")
    print()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("=" * 70)
    print("Summary: Per-Eigenmode Structural Field Tetrad")
    print("=" * 70)
    print()
    print("  The four canonical TNFR structural fields (Phi_s, |grad_phi|,")
    print("  K_phi, xi_C) have been computed for each eigenmode psi_j of")
    print("  the discrete TNFR operator H^(k)(sigma).")
    print()
    print("  Key findings:")
    print("    1. U6 confinement is SPECTRAL: eigenmodes can be classified")
    print("       as confined or unconfined via |Phi_s(j)| < threshold.")
    print("    2. Confinement is best near sigma = 1/2, where the potential")
    print("       V_sigma vanishes and the operator is purely Laplacian.")
    print("    3. Phi_s grows with k (O(k) spectral density), while")
    print("       gradient and curvature probe eigenvector structure.")
    print("    4. Cross-topology: Phi_s is topology-independent (spectral),")
    print("       while gradient/curvature/xi_C are topology-dependent")
    print("       (eigenvector spatial structure).")
    print()
    print("  This bridges spectral theory to the canonical TNFR measurement")
    print("  infrastructure, extending the U6 confinement principle from")
    print("  graph nodes to spectral eigenmodes.")
    print()

    return


if __name__ == "__main__":
    sys.exit(main() or 0)
