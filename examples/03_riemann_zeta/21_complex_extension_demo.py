#!/usr/bin/env python3
r"""Example 44: P4 Complex-s Extension — Non-Hermitian TNFR-Riemann Operator.

Demonstrates the extension of the TNFR-Riemann operator to complex s:

    H^(k)(s) = L_k + (s - 1/2) diag(log p_1, ..., log p_k),  s in C.

When s = sigma + it with t != 0, H(s) is non-Hermitian and has complex
eigenvalues.  The Riemann zeros lie at s = 1/2 + i*t_n, making the
critical line the natural domain for studying connections to zeta.

Key questions investigated:
  1. Do eigenvalues lambda_j(s) have zero-crossings near s = 1/2 + i*t_n?
  2. What is the pseudo-spectrum sigma_eps(H(s)) near the critical line?
  3. Does the resolvent show poles related to Riemann zeros?

TNFR physics basis: complex structural frequencies in the nodal equation
dEPI/dt = nu_f * DELTA_NFR(t) support oscillatory (quantum-like) dynamics.
The imaginary part of s encodes this oscillatory regime.

Usage:
    python examples/03_riemann_zeta/21_complex_extension_demo.py
"""

from __future__ import annotations

import sys
import time

import numpy as np


def main() -> None:
    """Run P4 complex-s extension demonstration."""
    from tnfr.riemann.complex_extension import (
        KNOWN_RIEMANN_ZEROS,
        analyze_non_hermiticity,
        analyze_resolvent_along_critical_line,
        compare_with_riemann_zeros,
        compute_complex_eigenspectrum,
        compute_pseudospectrum,
        find_eigenvalue_zero_crossings,
        run_complex_plane_analysis,
        scan_critical_line,
    )

    print("=" * 72)
    print("TNFR-Riemann P4: Complex-s Extension (Non-Hermitian Operator)")
    print("=" * 72)
    print()
    print("Operator: H^(k)(s) = L_k + (s - 1/2) diag(log p_i),  s in C")
    print("Physics:  Complex nu_f => oscillatory TNFR dynamics")
    print()

    # ------------------------------------------------------------------
    # 1. Non-Hermiticity characterization
    # ------------------------------------------------------------------
    print("-" * 72)
    print("1. NON-HERMITICITY ANALYSIS")
    print("-" * 72)

    k = 20
    print(f"\nGraph size: k = {k} primes")
    print(f"{'s':>20s}  {'||H-H†||/||H||':>16s}  {'cond(V)':>10s}  {'min|λ|':>10s}")
    print("-" * 62)

    for t_val in [0.0, 1.0, 5.0, 14.13, 21.02, 50.0]:
        s = 0.5 + t_val * 1j
        result = analyze_non_hermiticity(k, s)
        print(
            f"  1/2 + {t_val:6.2f}i"
            f"  {result.non_hermiticity:16.6e}"
            f"  {result.condition_number:10.2f}"
            f"  {result.min_abs_eigenvalue:10.6f}"
        )

    # ------------------------------------------------------------------
    # 2. Critical line scan
    # ------------------------------------------------------------------
    print()
    print("-" * 72)
    print("2. CRITICAL LINE SCAN: s = 1/2 + it")
    print("-" * 72)

    for k_scan in [10, 20, 50]:
        t0 = time.perf_counter()
        scan = scan_critical_line(k_scan, t_max=50.0, n_points=300)
        elapsed = time.perf_counter() - t0
        print(
            f"\n  k = {k_scan}: scanned t in [0, 50], {len(scan.t_values)} points"
            f" ({elapsed:.2f}s)"
        )
        print(f"    Local minima in min|λ|: {len(scan.local_minima_t)}")
        if len(scan.local_minima_t) > 0:
            print(f"    {'t':>10s}  {'min|λ|':>10s}")
            for t_m, v_m in zip(scan.local_minima_t[:8], scan.local_minima_val[:8]):
                print(f"    {t_m:10.4f}  {v_m:10.6f}")

    # ------------------------------------------------------------------
    # 3. Eigenvalue zero-crossings vs Riemann zeros
    # ------------------------------------------------------------------
    print()
    print("-" * 72)
    print("3. EIGENVALUE ZERO-CROSSINGS vs KNOWN RIEMANN ZEROS")
    print("-" * 72)

    print(f"\n  First 10 known Riemann zeros (Im part):")
    for i, tz in enumerate(KNOWN_RIEMANN_ZEROS[:10], 1):
        print(f"    t_{i} = {tz:.6f}")

    for k_cmp in [10, 20, 50]:
        print(f"\n  k = {k_cmp}:")
        comparison = compare_with_riemann_zeros(
            k_cmp,
            t_max=55.0,
            n_points=400,
            n_zeros=10,
            threshold=2.0,
        )
        if comparison:
            print(
                f"    {'t_candidate':>12s}  {'nearest ζ zero':>14s}  {'distance':>10s}"
            )
            for t_c, t_r, dist in comparison[:8]:
                print(f"    {t_c:12.4f}  {t_r:14.6f}  {dist:10.4f}")
        else:
            print("    No near-zero eigenvalue crossings detected.")

    # ------------------------------------------------------------------
    # 4. Pseudo-spectrum snapshot
    # ------------------------------------------------------------------
    print()
    print("-" * 72)
    print("4. PSEUDO-SPECTRUM ANALYSIS")
    print("-" * 72)

    k_ps = 10
    for s_val in [0.5 + 0j, 0.5 + 14.13j, 0.5 + 21.02j]:
        ps = compute_pseudospectrum(
            k_ps,
            s_val,
            z_radius=4.0,
            n_grid=30,
        )
        sigma_min_floor = float(np.min(ps.sigma_min_grid))
        sigma_min_max = float(np.max(ps.sigma_min_grid))
        evals = ps.eigenvalues
        spread_re = float(np.ptp(evals.real))
        spread_im = float(np.ptp(evals.imag))
        print(f"\n  s = {s_val}:")
        print(f"    σ_min range: [{sigma_min_floor:.6f}, {sigma_min_max:.4f}]")
        print(
            f"    Eigenvalue spread: Re [{evals.real.min():.3f}, {evals.real.max():.3f}]"
            f"  Im [{evals.imag.min():.3f}, {evals.imag.max():.3f}]"
        )

    # ------------------------------------------------------------------
    # 5. Resolvent analysis
    # ------------------------------------------------------------------
    print()
    print("-" * 72)
    print("5. RESOLVENT ANALYSIS: ||(zI - H(1/2+it))^{-1}||")
    print("-" * 72)

    k_res = 15
    for z_probe in [0.0 + 0j, 0.5 + 0j]:
        res = analyze_resolvent_along_critical_line(
            k_res,
            z_probe=z_probe,
            t_max=50.0,
            n_points=300,
        )
        print(f"\n  z_probe = {z_probe}, k = {k_res}:")
        print(f"    Resolvent peaks: {len(res.peak_t_values)}")
        if len(res.peak_t_values) > 0:
            print(f"    {'t_peak':>10s}  {'||R||':>12s}")
            # Show top peaks by norm
            top_idx = np.argsort(res.peak_norms)[::-1][:6]
            for i in top_idx:
                print(f"    {res.peak_t_values[i]:10.4f}  {res.peak_norms[i]:12.2f}")

    # ------------------------------------------------------------------
    # 6. Integrated analysis
    # ------------------------------------------------------------------
    print()
    print("-" * 72)
    print("6. INTEGRATED P4 ANALYSIS")
    print("-" * 72)

    for k_int in [10, 30]:
        analysis = run_complex_plane_analysis(
            k_int,
            t_max=50.0,
            n_points=300,
        )
        print(f"\n{analysis.summary}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print()
    print("=" * 72)
    print("P4 SUMMARY")
    print("=" * 72)
    print(
        """
Key findings from the non-Hermitian extension:

1. H(s) transitions smoothly from Hermitian (Im(s)=0) to non-Hermitian.
   Non-Hermiticity grows monotonically with |Im(s)|.

2. Along the critical line s = 1/2 + it, eigenvalues become complex with
   oscillatory structure reflecting the prime distribution.

3. The pseudo-spectrum of the non-Hermitian operator can be significantly
   larger than the epsilon-disc union (non-normal sensitivity).

4. Eigenvalue near-zero events and resolvent peaks provide structural
   markers that can be compared with known Riemann zeros.

TNFR physics interpretation:
   The imaginary part Im(s) activates oscillatory dynamics in the nodal
   equation.  At sigma = 1/2 (critical line), the real part is at
   structural equilibrium (P1), while the imaginary part drives quantum-
   like oscillations whose spectral structure encodes prime distribution
   information — the core of the TNFR-Riemann connection.
"""
    )


if __name__ == "__main__":
    main()
