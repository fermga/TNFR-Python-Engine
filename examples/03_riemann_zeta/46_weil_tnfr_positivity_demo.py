"""TNFR-Riemann P17 demo — Weil positivity bridge.

Tabulates two quantities across a Gaussian-width grid:

* The RH-equivalent Weil functional W[sigma] = sum_gamma h_sigma(gamma),
  computed both from the classical zero side and via the Weil-Guinand
  explicit formula using the canonical TNFR P14 prime-ladder
  Hamiltonian.
* The canonical TNFR Lyapunov energy E_TNFR[sigma] of a structural
  test state derived from h_sigma on the prime-ladder graph.

It reports the ratio alpha(sigma) = W[sigma] / E_TNFR[sigma].  If
alpha(sigma) > 0 across the grid, this is the operational TNFR-native
witness for Weil positivity (and hence, via Weil's equivalence,
numerical evidence for RH on this admissible family).

This is an *experimental research diagnostic*, not a proof.
Run from the repository root with PYTHONPATH=src:

    $env:PYTHONPATH=(Resolve-Path ./src).Path
    python examples/03_riemann_zeta/46_weil_tnfr_positivity_demo.py
"""

from __future__ import annotations

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from tnfr.riemann import (
    build_prime_ladder_hamiltonian,
    verify_weil_positivity,
    verify_weil_tnfr_bridge,
)


def main() -> None:
    print("=" * 72)
    print("TNFR-Riemann P17 demo - Weil-TNFR positivity bridge")
    print("=" * 72)

    # -- Build a modest P14 Hamiltonian (small for fast demo runtime) ----
    n_primes = 20
    max_power = 6
    print(
        f"\nBuilding P14 prime-ladder Hamiltonian "
        f"(n_primes={n_primes}, max_power={max_power}) ..."
    )
    bundle = build_prime_ladder_hamiltonian(
        n_primes=n_primes, max_power=max_power, coupling=0.0
    )
    print(f"  Hilbert dimension = {n_primes * max_power}")

    # -- Part 1: single-sigma Weil positivity certificate ----------------
    print("\n" + "-" * 72)
    print("Part 1: Weil positivity certificate (single sigma)")
    print("-" * 72)
    sigma_solo = 2.0
    print(f"\nVerifying W[sigma={sigma_solo}] >= 0 ...")
    cert = verify_weil_positivity(
        bundle, sigma=sigma_solo, n_zeros=60, max_zeros=200
    )
    print("  " + cert.summary())
    print(f"  W (zero side)        = {cert.weil_functional_zero_side:+.10f}")
    print(
        f"  W (explicit formula) = "
        f"{cert.weil_functional_explicit_formula:+.10f}"
    )
    print(f"  consistency residual = {cert.explicit_formula_residual:.3e}")
    print(f"  zeros used           = {cert.n_zeros_used}")
    print(f"  Weil positive?       = {cert.positive}")

    # -- Part 2: bridge across a sigma grid ------------------------------
    print("\n" + "-" * 72)
    print("Part 2: TNFR-Weil bridge across a Gaussian-width grid")
    print("-" * 72)
    sigmas = [1.0, 1.5, 2.0, 3.0, 5.0, 8.0]
    print(f"\nGrid: sigmas = {sigmas}")
    print("Computing bridge certificate ...")
    bridge = verify_weil_tnfr_bridge(
        bundle, sigmas, n_zeros=60, max_zeros=200
    )

    print("\n  sigma        W[sigma]          E_TNFR[sigma]       "
          "alpha(sigma)   W>=0   alpha>0")
    print("  " + "-" * 78)
    for i, s in enumerate(bridge.sigmas):
        W = bridge.weil_functional[i]
        E = bridge.tnfr_lyapunov_energy[i]
        a = bridge.alpha[i]
        wp = bridge.weil_positive[i]
        bp = bridge.bridge_positive[i]
        print(
            f"  {float(s):>5.2f}  {float(W):+.6e}  "
            f"{float(E):+.6e}  {float(a):+.6e}    "
            f"{str(bool(wp)):>5}   {str(bool(bp)):>5}"
        )

    print("\n  " + bridge.summary())

    # -- Honest framing --------------------------------------------------
    print("\n" + "=" * 72)
    print("Interpretation")
    print("=" * 72)
    if bridge.weil_positive_all and bridge.bridge_positive_all:
        print(
            "  - Weil positivity W[sigma] >= 0 holds across the grid (OK).\n"
            "  - The TNFR-native lower bound alpha(sigma) > 0 holds across\n"
            "    the grid with alpha_min = "
            f"{bridge.alpha_min:.4e} (positive).\n"
            "  This is numerical evidence consistent with RH on the\n"
            "  tested admissible family.  It is NOT a proof: closing\n"
            "  gap G4 (RH itself) would require promoting alpha > 0 to\n"
            "  a theorem on a dense admissible class."
        )
    else:
        print(
            "  - Weil positivity W[sigma] >= 0 holds: "
            f"{bridge.weil_positive_all}.\n"
            "  - TNFR lower bound alpha(sigma) > 0 holds: "
            f"{bridge.bridge_positive_all}.\n"
            "  The numerical bridge does not hold uniformly on this grid;\n"
            "  the failing regime localises where structural and analytic\n"
            "  positivities decouple under the present test-state mapping."
        )
    print("=" * 72)


if __name__ == "__main__":
    main()
