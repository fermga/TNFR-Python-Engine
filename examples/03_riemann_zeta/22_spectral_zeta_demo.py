#!/usr/bin/env python3
r"""Example 45: P5 Discrete Spectral Zeta and Trace Formula.

Demonstrates the discrete spectral zeta function

    zeta_{H^(k)}(sigma, u) = sum_{j: lambda_j > 0} lambda_j(sigma)^{-u}

and the heat kernel trace

    Theta(beta) = tr(e^{-beta H}) = sum_j e^{-beta lambda_j}

together with the Mellin transform bridge that connects them:

    zeta_H(u) = (1/Gamma(u)) int_0^infty beta^{u-1} Theta(beta) dbeta

The Conjecture 10.1 framework asks whether

    zeta_{H^(k)}(1/2, u) -> C * zeta_R(u + delta) as k -> infty.

TNFR physics basis: the spectral zeta connects the TNFR eigenvalue
spectrum to partition functions and structural thermodynamics.  The
Mellin transform bridges heat-kernel dynamics (nodal-equation time
evolution) to zeta values, linking structural conservation with the
Riemann zeta function.

Usage:
    python examples/03_riemann_zeta/22_spectral_zeta_demo.py
"""

from __future__ import annotations

import sys
import time

import numpy as np


def main() -> None:
    """Run P5 spectral zeta demonstration."""
    from tnfr.riemann.spectral_zeta import (
        compute_positive_eigenvalues,
        compute_spectral_zeta,
        compute_spectral_zeta_derivative,
        compute_heat_kernel_trace,
        verify_mellin_bridge,
        riemann_zeta_approx,
        test_conjecture_10_1,
        test_conjecture_10_1_sequence,
        run_spectral_zeta_analysis,
        RIEMANN_ZETA_KNOWN_VALUES,
    )

    print("=" * 72)
    print("TNFR-Riemann P5: Discrete Spectral Zeta & Trace Formula")
    print("=" * 72)
    print()
    print("Spectral zeta:  zeta_H(sigma, u) = sum_{j: lam_j>0} lam_j^{-u}")
    print("Heat kernel:    Theta(beta) = tr(e^{-beta H})")
    print("Mellin bridge:  zeta_H(u) = (1/Gamma(u)) int beta^{u-1} Theta dbeta")
    print()

    # ------------------------------------------------------------------
    # 1. Positive eigenvalue extraction
    # ------------------------------------------------------------------
    print("-" * 72)
    print("1. POSITIVE EIGENVALUE EXTRACTION")
    print("-" * 72)

    for k in [5, 10, 20, 50, 100]:
        evals = compute_positive_eigenvalues(k, 0.5)
        lam_min = float(evals[0])
        lam_max = float(evals[-1])
        print(
            f"  k = {k:3d}:  n_pos = {len(evals):3d},  "
            f"lam_min = {lam_min:.6f},  lam_max = {lam_max:.4f}"
        )

    # ------------------------------------------------------------------
    # 2. Spectral zeta function
    # ------------------------------------------------------------------
    print()
    print("-" * 72)
    print("2. SPECTRAL ZETA FUNCTION at sigma = 1/2")
    print("-" * 72)

    u_values = np.array([0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0])
    k = 30
    result = compute_spectral_zeta(k, 0.5, u_values=u_values)
    print(f"\n  k = {k}, n_positive = {result.n_positive}")
    print(f"  {'u':>6s}  {'zeta_H(u)':>14s}")
    print("  " + "-" * 24)
    for u, z in zip(result.u_values, result.zeta_values):
        print(f"  {u:6.1f}  {z:14.6f}")

    print(f"\n  Check: zeta(0) = {result.zeta_values[0]:.1f} "
          f"(expected {result.n_positive})")

    # ------------------------------------------------------------------
    # 3. Spectral zeta derivative
    # ------------------------------------------------------------------
    print()
    print("-" * 72)
    print("3. SPECTRAL ZETA DERIVATIVE")
    print("-" * 72)

    u_d = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    deriv = compute_spectral_zeta_derivative(k, 0.5, u_values=u_d)
    print(f"\n  d/du zeta_H({k}, 1/2, u):")
    print(f"  {'u':>6s}  {'dzeta/du':>14s}")
    print("  " + "-" * 24)
    for u, d in zip(u_d, deriv):
        print(f"  {u:6.1f}  {d:14.6f}")

    # ------------------------------------------------------------------
    # 4. Heat kernel trace and thermodynamics
    # ------------------------------------------------------------------
    print()
    print("-" * 72)
    print("4. HEAT KERNEL TRACE & THERMODYNAMICS")
    print("-" * 72)

    betas = np.array([0.001, 0.01, 0.1, 1.0, 10.0, 100.0])
    hk = compute_heat_kernel_trace(k, 0.5, beta_values=betas)
    print(f"\n  k = {k}, sigma = 0.5")
    print(f"  {'beta':>8s}  {'Theta':>12s}  {'F(beta)':>12s}  {'S(beta)':>12s}")
    print("  " + "-" * 50)
    for i, beta in enumerate(hk.beta_values):
        print(
            f"  {beta:8.3f}  {hk.theta_values[i]:12.4f}  "
            f"{hk.free_energy[i]:12.4f}  {hk.entropy[i]:12.4f}"
        )

    print(f"\n  Theta(beta->0) = {hk.theta_values[0]:.2f} (expected ~{k})")
    print(f"  Theta(beta->inf) = {hk.theta_values[-1]:.6f} "
          f"(expected ~1 at sigma=1/2)")

    # ------------------------------------------------------------------
    # 5. Mellin bridge verification
    # ------------------------------------------------------------------
    print()
    print("-" * 72)
    print("5. MELLIN BRIDGE VERIFICATION")
    print("-" * 72)

    u_bridge = np.array([1.5, 2.0, 3.0, 4.0, 5.0])
    mb = verify_mellin_bridge(k, 0.5, u_values=u_bridge)
    print(f"\n  k = {k}, sigma = 0.5")
    print(
        f"  {'u':>5s}  {'direct':>12s}  {'Mellin':>12s}  {'rel_err':>12s}"
    )
    print("  " + "-" * 46)
    for i, u in enumerate(mb.u_values):
        print(
            f"  {u:5.1f}  {mb.zeta_direct[i]:12.6f}  "
            f"{mb.zeta_mellin[i]:12.6f}  {mb.relative_error[i]:12.4e}"
        )
    status = "VALID" if mb.bridge_valid else "FAILED"
    print(f"\n  Max relative error: {mb.max_relative_error:.4e} [{status}]")

    # ------------------------------------------------------------------
    # 6. Riemann zeta approximation check
    # ------------------------------------------------------------------
    print()
    print("-" * 72)
    print("6. RIEMANN ZETA APPROXIMATION CHECK")
    print("-" * 72)

    print(f"\n  {'u':>5s}  {'approx':>14s}  {'exact':>14s}  {'rel_err':>12s}")
    print("  " + "-" * 50)
    for u, exact in sorted(RIEMANN_ZETA_KNOWN_VALUES.items()):
        if u > 1.0:
            approx = riemann_zeta_approx(u, n_terms=100_000)
            err = abs(approx - exact) / abs(exact)
            print(f"  {u:5.1f}  {approx:14.8f}  {exact:14.8f}  {err:12.4e}")

    # ------------------------------------------------------------------
    # 7. Conjecture 10.1 test
    # ------------------------------------------------------------------
    print()
    print("-" * 72)
    print("7. CONJECTURE 10.1: zeta_H(1/2,u) ~ C * zeta_R(u + delta)?")
    print("-" * 72)

    k_vals = [10, 20, 50, 100]
    print(
        f"\n  {'k':>5s}  {'C_fit':>12s}  {'delta_fit':>10s}  "
        f"{'residual':>12s}  {'correlation':>12s}"
    )
    print("  " + "-" * 55)
    t_start = time.time()
    results = test_conjecture_10_1_sequence(k_vals)
    elapsed = time.time() - t_start

    for r in results:
        print(
            f"  {r.k:5d}  {r.C_fit:12.4e}  {r.delta_fit:10.4f}  "
            f"{r.residual:12.4e}  {r.correlation:12.6f}"
        )
    print(f"\n  Conjecture 10.1 sequence computed in {elapsed:.2f}s")

    # ------------------------------------------------------------------
    # 8. Integrated analysis
    # ------------------------------------------------------------------
    print()
    print("-" * 72)
    print("8. INTEGRATED P5 ANALYSIS")
    print("-" * 72)

    t_start = time.time()
    analysis = run_spectral_zeta_analysis(30)
    elapsed = time.time() - t_start

    print()
    print(analysis.summary)
    print(f"\n  Analysis completed in {elapsed:.2f}s")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print()
    print("=" * 72)
    print("P5 SUMMARY")
    print("=" * 72)
    print()
    print("  Spectral zeta:     zeta_H(sigma, u) computed and validated")
    print("  Heat kernel:       Theta, Z, F, S computed across beta range")
    print(f"  Mellin bridge:     {'VERIFIED' if mb.bridge_valid else 'NEEDS REVIEW'}"
          f" (max err = {mb.max_relative_error:.2e})")
    print(f"  Conjecture 10.1:   Tested for k in {k_vals}")
    if results:
        best = max(results, key=lambda r: r.correlation)
        print(f"    Best fit at k={best.k}: "
              f"C={best.C_fit:.4e}, delta={best.delta_fit:.4f}, "
              f"r={best.correlation:.4f}")
    print()
    print("TNFR-Riemann P5 demonstration complete.")
    print("=" * 72)


if __name__ == "__main__":
    main()
