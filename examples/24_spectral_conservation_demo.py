#!/usr/bin/env python3
r"""Example 47: P7 Conservation Laws and Grammar Compliance at Criticality.

Demonstrates the structural conservation theorem applied to prime-graph
eigenmodes of the discrete TNFR operator H^(k)(sigma) = L_k + V_sigma.

Two new spectral transport fields complete the conservation five-tuple:

  J_phi(j)    = <psi_j | L | psi_j> / k        phase current
  J_DNFR(j)   = sum_i |psi_j(i)|^2 log(p_i)    eigenvalue velocity

With the existing tetrad (Phi_s, |grad_phi|, K_phi), these give:

  Energy density   E(j) = Phi_s^2 + |grad_phi|^2 + K_phi^2 + J_phi^2 + J_DNFR^2
  Topological charge Q(j) = |grad_phi| * J_phi - K_phi * J_DNFR
  Charge density   rho(j) = Phi_s(j) + K_phi(j)

Key result: Grammar-compliant evolution (smooth sigma-perturbation) preserves
the total Noether charge Q_total better than grammar-violating evolution
(abrupt sigma-jumps), verifying dρ/dt + div(J) = S_grammar with
S_grammar -> 0 under U1-U6.

TNFR physics basis:
  Nodal equation: dEPI/dt = nu_f * DELTA_NFR(t)
  Conservation:   d(rho)/dt + div(J) = S_grammar
  Criticality:    sigma = 0.5 where V = 0 (pure Laplacian regime)

Usage:
    python examples/24_spectral_conservation_demo.py
"""

from __future__ import annotations

from tnfr.riemann.spectral_conservation import (
    compute_spectral_j_phi,
    compute_spectral_j_dnfr,
    compute_eigenmode_conservation,
    scan_conservation_vs_sigma,
    test_grammar_conservation,
    run_critical_conservation_analysis,
)
from tnfr.mathematics.unified_numerical import np


def section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


def main() -> None:
    print("P7: Conservation Laws and Grammar Compliance at Criticality")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Conservation fields at sigma = 0.5, k = 15
    # ------------------------------------------------------------------
    section("1. Conservation Fields at Criticality (k=15, sigma=0.5)")
    snap = compute_eigenmode_conservation(15, sigma=0.5)

    print(f"  Graph size k = {snap.k}")
    print(f"  Structural parameter sigma = {snap.sigma}")
    print(f"  Total energy E = {snap.total_energy:.6f}")
    print(f"  Total charge Q = {snap.total_charge:.6f}")
    print(f"  Total charge density rho = {snap.total_charge_density:.6f}")
    print(f"  Mean energy density = {snap.mean_energy_density:.6f}")
    print(f"  Mean charge = {snap.mean_charge:.6f}")

    # ------------------------------------------------------------------
    # 2. Per-eigenmode conservation table
    # ------------------------------------------------------------------
    section("2. Per-Eigenmode Conservation Fields (first 8 modes)")
    print("  j   lambda_j    Phi_s    |grad|    K_phi    J_phi   J_DNFR"
          "      E(j)      Q(j)")
    print("  " + "-" * 78)

    for m in snap.modes[:8]:
        print(
            f"  {m.mode_index:2d}  {m.eigenvalue:8.4f}  "
            f"{m.phi_s:7.4f}  {m.grad_phi:7.4f}  "
            f"{m.k_phi:7.4f}  {m.j_phi:7.4f}  {m.j_dnfr:7.4f}  "
            f"{m.energy_density:8.4f}  {m.topological_charge:8.4f}"
        )

    # ------------------------------------------------------------------
    # 3. Sigma scan of total energy and charge
    # ------------------------------------------------------------------
    section("3. Sigma Scan: E(sigma) and Q(sigma)")
    sigmas = np.linspace(0.2, 0.8, 13)
    scan = scan_conservation_vs_sigma(15, sigmas)

    print("  sigma    E_total     Q_total     |dQ/ds|    drift")
    print("  " + "-" * 55)
    for i, s in enumerate(scan.sigma_values):
        print(
            f"  {s:.2f}    {scan.total_energy[i]:9.4f}   "
            f"{scan.total_charge[i]:9.4f}   "
            f"{scan.charge_gradient[i]:8.4f}   "
            f"{scan.charge_drift_from_half[i]:8.5f}"
        )

    print(f"\n  Energy minimum at sigma = {scan.energy_minimum_sigma:.3f}")
    print(f"  Minimal |dQ/ds| at sigma = {scan.min_gradient_sigma:.3f}")

    # ------------------------------------------------------------------
    # 4. Grammar compliance test
    # ------------------------------------------------------------------
    section("4. Grammar Compliance: Conservation of Noether Charge")
    results = test_grammar_conservation(15, sigma=0.5)

    print("  Protocol            Valid   Q_drift    Quality")
    print("  " + "-" * 48)
    for r in results:
        tag = "YES" if r.is_grammar_compliant else "NO "
        print(
            f"  {r.protocol:<20s}  {tag}   "
            f"{r.charge_drift:9.6f}  {r.conservation_quality:.6f}"
        )

    compliant = [r for r in results if r.is_grammar_compliant]
    violating = [r for r in results if not r.is_grammar_compliant]
    cq = sum(r.conservation_quality for r in compliant) / len(compliant)
    vq = sum(r.conservation_quality for r in violating) / len(violating)

    print(f"\n  Mean quality (compliant):  {cq:.6f}")
    print(f"  Mean quality (violating):  {vq:.6f}")
    print(f"  Quality ratio (c/v):       {cq / vq:.4f}")
    print(f"  Compliant preserves charge better: {cq > vq}")

    # ------------------------------------------------------------------
    # 5. Full critical conservation analysis
    # ------------------------------------------------------------------
    section("5. Full Critical Conservation Analysis (k=20)")
    analysis = run_critical_conservation_analysis(
        k=20,
        sigma_values=np.linspace(0.2, 0.8, 25),
    )

    print(f"  Graph size k = {analysis.k}")
    print(f"  Energy minimum near sigma=0.5: "
          f"{analysis.sigma_half_is_energy_min}")
    print(f"  Minimal residual near sigma=0.5: "
          f"{analysis.sigma_half_has_min_residual}")
    print(f"  Compliant mean quality: "
          f"{analysis.compliant_mean_quality:.6f}")
    print(f"  Violating mean quality: "
          f"{analysis.violating_mean_quality:.6f}")
    print(f"  Quality ratio: {analysis.quality_ratio:.4f}")

    # ------------------------------------------------------------------
    # 6. Hellmann-Feynman verification
    # ------------------------------------------------------------------
    section("6. Hellmann-Feynman Verification (k=10)")
    from tnfr.riemann.operator import build_prime_path_graph, build_h_tnfr

    k = 10
    sigma, ds = 0.5, 1e-5
    G = build_prime_path_graph(k)
    nodes = sorted(G.nodes())
    log_p = np.array([np.log(float(G.nodes[n]["label"])) for n in nodes])

    Hc, _ = build_h_tnfr(G, sigma=sigma)
    Hp, _ = build_h_tnfr(G, sigma=sigma + ds)
    Hm, _ = build_h_tnfr(G, sigma=sigma - ds)

    evals_c, vecs_c = np.linalg.eigh(Hc)
    evals_p, _ = np.linalg.eigh(Hp)
    evals_m, _ = np.linalg.eigh(Hm)

    numerical = (evals_p - evals_m) / (2 * ds)
    analytic = compute_spectral_j_dnfr(vecs_c, log_p)

    print("  j   Numerical      Analytic       Rel Error")
    print("  " + "-" * 48)
    for j in range(k):
        rel = abs(analytic[j] - numerical[j]) / max(abs(numerical[j]), 1e-30)
        print(f"  {j:2d}  {numerical[j]:12.6f}   {analytic[j]:12.6f}   "
              f"{rel:10.2e}")

    max_err = float(np.max(np.abs(analytic - numerical)
                           / np.maximum(np.abs(numerical), 1e-30)))
    print(f"\n  Max relative error: {max_err:.2e}")
    print(f"  Hellmann-Feynman verified: {max_err < 1e-3}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    section("Summary")
    print("  P7 establishes the structural conservation theorem at")
    print("  criticality (sigma = 0.5) for prime-graph eigenmodes:")
    print()
    print("  1. Energy density E and topological charge Q are computed")
    print("     per eigenmode using five structural fields.")
    print("  2. Grammar-compliant evolution (smooth sigma steps)")
    print("     preserves Noether charge Q better than grammar-violating")
    print("     evolution (abrupt sigma jumps).")
    print("  3. The Hellmann-Feynman theorem accurately predicts")
    print("     eigenvalue velocity J_DNFR = dlambda/dsigma.")
    print("  4. Conservation residual |dQ/dsigma| tends to be minimal")
    print("     near sigma = 0.5, confirming S_grammar -> 0 at")
    print("     criticality under U1-U6.")
    print()
    print("  Physics: d(rho)/dt + div(J) = S_grammar")
    print("  At criticality: S_grammar -> 0 under grammar compliance.")


if __name__ == "__main__":
    main()
