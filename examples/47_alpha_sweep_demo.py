"""TNFR-Riemann P18 demo - admissibility / gauge sweep of alpha(sigma).

Tests the robustness of the P17 Weil-TNFR positivity bridge:

* Dense Gaussian-width grid: log-spaced over [0.5, 12].
* Six structural gauges parameterising how h_sigma is encoded into
  (dnfr, phase, EPI) on the P14 prime-ladder graph.
* Per-cell ratio alpha(sigma; gauge) = W[sigma] / E_TNFR[sigma; gauge],
  reusing the gauge-invariant Weil functional once per sigma.

If alpha > 0 across the whole table, the P17 bridge is robust under
canonical-mapping ambiguity (strengthens the §14 numerical evidence).
A negative entry would falsify the bridge *as parameterised* (it would
NOT disprove RH; RH depends only on W).

Run from the repository root with PYTHONPATH=src:

    $env:PYTHONPATH=(Resolve-Path ./src).Path
    python examples/47_alpha_sweep_demo.py
"""

from __future__ import annotations

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np

from tnfr.riemann import (
    build_prime_ladder_hamiltonian,
    sweep_alpha,
)


def main() -> None:
    print("=" * 76)
    print("TNFR-Riemann P18 demo - admissibility / gauge sweep of alpha")
    print("=" * 76)

    # ---- Build P14 bundle (small for fast demo runtime) --------------
    n_primes = 18
    max_power = 5
    print(
        f"\nBuilding P14 prime-ladder Hamiltonian "
        f"(n_primes={n_primes}, max_power={max_power}) ..."
    )
    bundle = build_prime_ladder_hamiltonian(
        n_primes=n_primes, max_power=max_power, coupling=0.0
    )
    print(f"  Hilbert dimension = {n_primes * max_power}")

    # ---- Dense sigma grid (log-spaced) --------------------------------
    n_sigma = 12
    sigmas = np.logspace(np.log10(0.5), np.log10(12.0), n_sigma).tolist()
    print(f"\nSigma grid (log-spaced, n={n_sigma}):")
    print("  " + "  ".join(f"{s:.3f}" for s in sigmas))

    # ---- Run sweep ----------------------------------------------------
    print(
        f"\nProbing {n_sigma} sigma values across the default gauge "
        "family (6 gauges) ..."
    )
    cert = sweep_alpha(
        bundle, sigmas, n_zeros=50, max_zeros=180
    )

    # ---- Summary line -------------------------------------------------
    print("\n" + cert.summary())

    # ---- Weil functional per sigma (gauge-invariant) -----------------
    print("\n" + "-" * 76)
    print("W[sigma] (gauge-invariant)")
    print("-" * 76)
    print("  sigma     W[sigma]")
    for s, w in zip(cert.sigmas, cert.weil_values):
        sign = "+" if w >= 0 else "-"
        print(f"  {float(s):>6.3f}  {sign}{abs(float(w)):.4e}")

    # ---- Alpha table --------------------------------------------------
    print("\n" + "-" * 76)
    print("alpha(sigma; gauge) = W[sigma] / E_TNFR[sigma; gauge]")
    print("-" * 76)

    header = "  sigma   " + "  ".join(f"{g:>14s}" for g in cert.gauges)
    print(header)
    print("  " + "-" * (len(header) - 2))

    for j, s in enumerate(cert.sigmas):
        row_parts = [f"{float(s):>6.3f}  "]
        for i in range(len(cert.gauges)):
            a = float(cert.alpha_table[i, j])
            if np.isnan(a):
                row_parts.append(f"{'nan':>14s}")
            elif np.isinf(a):
                sign = "+inf" if a > 0 else "-inf"
                row_parts.append(f"{sign:>14s}")
            else:
                sign = "+" if a >= 0 else "-"
                row_parts.append(f" {sign}{abs(a):.4e}")
        print("  ".join(row_parts))

    # ---- Energy table (auxiliary) ------------------------------------
    print("\n" + "-" * 76)
    print("E_TNFR[sigma; gauge]  (canonical Lyapunov energy)")
    print("-" * 76)
    header = "  sigma   " + "  ".join(f"{g:>14s}" for g in cert.gauges)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for j, s in enumerate(cert.sigmas):
        row_parts = [f"{float(s):>6.3f}  "]
        for i in range(len(cert.gauges)):
            E = float(cert.energy_table[i, j])
            row_parts.append(f"   {E:>10.4e}")
        print("  ".join(row_parts))

    # ---- Interpretation ----------------------------------------------
    print("\n" + "=" * 76)
    print("Interpretation")
    print("=" * 76)
    if cert.weil_all_positive and cert.alpha_all_positive:
        print(
            "  - W[sigma] >= 0 across the dense grid (Weil positivity\n"
            "    holds on the tested admissible family).\n"
            "  - alpha(sigma; gauge) > 0 across the full "
            f"{len(cert.gauges)} x {len(cert.sigmas)} table.\n"
            f"  - Tightest entry: alpha_min = {cert.alpha_min:.4e}\n"
            f"    at sigma = {cert.alpha_min_sigma:.3f}, "
            f"gauge = '{cert.alpha_min_gauge}'.\n"
            "  The P17 Weil-TNFR bridge is ROBUST under the probed\n"
            "  canonical-mapping ambiguity.  This strengthens the\n"
            "  numerical evidence consistent with RH but does NOT prove\n"
            "  RH: closing gap G4 still requires an analytic lower bound\n"
            "  alpha(sigma; gauge) >= c > 0 over a dense admissible class."
        )
    else:
        print(
            f"  - W[sigma] >= 0 across the grid: "
            f"{cert.weil_all_positive}.\n"
            f"  - alpha > 0 across the (sigma, gauge) table: "
            f"{cert.alpha_all_positive}.\n"
            f"  - Worst entry: alpha_min = {cert.alpha_min:.4e}\n"
            f"    at sigma = {cert.alpha_min_sigma:.3f}, "
            f"gauge = '{cert.alpha_min_gauge}'.\n"
            "  The P17 bridge does NOT survive the probed gauge family\n"
            "  as currently parameterised.  RH itself is not affected;\n"
            "  the negative result localises where structural and\n"
            "  analytic positivities decouple."
        )
    print("=" * 76)


if __name__ == "__main__":
    main()
