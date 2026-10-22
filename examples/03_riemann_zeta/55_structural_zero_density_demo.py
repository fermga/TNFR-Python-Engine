"""Finite comparison of a supplied classical smooth zero-count approximation.

The gamma factor of the classical completed zeta function supplies the
Riemann-Siegel theta function and the approximate targets. Reference ordinates
come independently from mpmath.zetazero for this arithmetic comparison. Neither
input is derived from a TNFR nodal trajectory. The target residual is a finite
ordinate error, not an identity with S(T), a proof of RH, or an equivalent RH
criterion. Current scope: theory/TNFR_RIEMANN_RESEARCH_NOTES.md.

Usage:
    python examples/03_riemann_zeta/55_structural_zero_density_demo.py
"""

from __future__ import annotations

import io
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
else:  # pragma: no cover - very old runtimes
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from tnfr.riemann import (
    compute_structural_zero_density_certificate,
    derive_smooth_zero_position,
    fetch_zero_imaginary_parts,
    riemann_siegel_theta,
    smooth_zero_count,
    smooth_zero_density,
)


def section(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def main() -> None:
    section("P28 — Section 1: Archimedean ingredients")
    print(
        "theta(T) = Im log Gamma(1/4 + iT/2) - (T/2) log pi\n"
        "bar N(T) = theta(T)/pi + 1     (Backlund smooth zero count)\n"
        "bar N'(T) approx (1/2 pi) log(T/2 pi) at large T"
    )
    for T in (14.13, 25.0, 50.0, 100.0, 500.0):
        theta = riemann_siegel_theta(T)
        count = smooth_zero_count(T)
        density = smooth_zero_density(T)
        print(
            f"  T = {T:>7.2f}   theta = {theta:>10.4f}   "
            f"bar N = {count:>8.4f}   bar N' = {density:.6f}"
        )

    section("P28 — Section 2: Smooth zero positions (Newton inversion)")
    print("For each n, Newton-solve bar N(T) = n.  Compare to true gamma_n.\n")
    actual_30 = fetch_zero_imaginary_parts(15)
    print(
        f"  {'n':>3} | {'tilde gamma_n':>14} | {'gamma_n':>14} | "
        f"{'r_n = gamma_n - tilde gamma_n':>30}"
    )
    print("  " + "-" * 73)
    for n in range(1, 16):
        smooth = derive_smooth_zero_position(n)
        actual = float(actual_30[n - 1])
        residual = actual - smooth
        print(f"  {n:>3} | {smooth:>14.6f} | {actual:>14.6f} | " f"{residual:>+30.6f}")

    section("P28 — Section 3: Operator-level comparison vs P27")
    print(
        "Compute W_1 distance between:\n"
        "  * spec(P14)             -- P27 baseline\n"
        "  * spec(tilde T_HP)      -- P28 supplied smooth approximation\n"
        "  * spec(T_HP) = {gamma_n} -- benchmark\n"
    )
    for n_zeros in (30, 60, 100):
        cert = compute_structural_zero_density_certificate(
            n_zeros=n_zeros, p14_n_primes=50, p14_max_power=8
        )
        print(f"\n  >>> n_zeros = {n_zeros}")
        print(f"  W_1(spec(P14),         spec(T_HP)) = " f"{cert.w1_p14_vs_actual:.4e}")
        print(
            f"  W_1(spec(tilde T_HP),  spec(T_HP)) = "
            f"{cert.w1_structural_vs_actual:.4e}"
        )
        print(
            f"  improvement ratio                  = " f"{cert.improvement_ratio:.2f}x"
        )
        print(f"  max |r_n|                          = " f"{cert.max_residual:.4e}")
        print(f"  selected finite C=2 check passed   = " f"{cert.bound_satisfied}")

    section("P28 — Section 4: Full certificate (N=80)")
    cert = compute_structural_zero_density_certificate(n_zeros=80)
    print(cert.summary())

    section("P28 — Section 5: Honest scope")
    print(
        "The smooth targets reuse classical archimedean input; reference zeros\n"
        "are supplied comparison data. Displayed distances and selected bounds\n"
        "are finite diagnostics. They do not derive those inputs from nodal\n"
        "dynamics, identify the ordinate error with S(T), or prove RH.\n"
        "See theory/TNFR_RIEMANN_RESEARCH_NOTES.md for the current scope."
    )

    print()
    print("=" * 78)
    print("Demo complete.")
    print("=" * 78)


if __name__ == "__main__":
    main()
