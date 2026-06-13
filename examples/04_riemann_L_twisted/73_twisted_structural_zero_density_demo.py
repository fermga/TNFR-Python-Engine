"""Demo 73: chi-twisted structural zero density (TNFR-Riemann P46).

L-track analogue of demo 55 (P28 structural zero density).  For each
primitive real Dirichlet character chi in {chi_3, chi_4, chi_5}:

  1. Shows the archimedean ingredients (theta_chi, N-bar_chi,
     N-bar'_chi) at a few representative T.
  2. Compares the structurally-derived smooth positions
     ``tilde gamma_n^(chi)`` (from theta_chi) against the actual
     ``gamma_n^(chi)`` from Hardy-Z bisection of L(s, chi).
  3. Computes the W_1 reduction
     ``W_1(spec(tilde T_HP^(chi)), spec(T_HP^(chi)))`` vs the P45
     baseline ``W_1(spec(P34|p!|q), spec(T_HP^(chi)))``.
  4. Prints the full per-character P46 certificate.
  5. Recalls the honest-scope statement.

Run:

    $env:PYTHONPATH = (Resolve-Path ./src).Path
    $env:PYTHONIOENCODING = "utf-8"
    & ./.venv312/Scripts/python.exe \
        examples/04_riemann_L_twisted/73_twisted_structural_zero_density_demo.py

Status: EXPERIMENTAL -- TNFR-Riemann P46 (May 2026).  Does NOT prove
GRH for any L(s, chi) and does NOT advance G4 = RH.
"""

from __future__ import annotations

import sys

from tnfr.riemann.dirichlet_l import (
    real_character_mod_3,
    real_character_mod_4,
    real_character_mod_5,
)
from tnfr.riemann.twisted_structural_zero_density import (
    compute_twisted_structural_zero_density_certificate,
    derive_twisted_smooth_zero_position,
    twisted_smooth_zero_count,
    twisted_smooth_zero_density,
    twisted_theta,
)
from tnfr.riemann.twisted_hilbert_polya import (
    fetch_chi_zero_imaginary_parts,
)
from tnfr.riemann.twisted_weil_explicit_formula import character_parity


def _ensure_utf8_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass


def main() -> int:
    _ensure_utf8_stdout()

    characters = [
        real_character_mod_3(),
        real_character_mod_4(),
        real_character_mod_5(),
    ]

    n_zeros = 18
    p34_n_primes = 30
    p34_max_power = 6

    print("=" * 78)
    print("P46 CHI-TWISTED STRUCTURAL ZERO DENSITY DEMO")
    print("(L-track analogue of P28; after Martinez Gamo,")
    print(" Zenodo 17665853 v2, 2025)")
    print("=" * 78)
    print(f"n_zeros        = {n_zeros}")
    print(f"p34_n_primes   = {p34_n_primes}")
    print(f"p34_max_power  = {p34_max_power}")
    print()

    # ------------------------------------------------------------------
    # Section 1: archimedean ingredients
    # ------------------------------------------------------------------
    print("-" * 78)
    print("Section 1: archimedean ingredients (theta_chi, N-bar_chi, "
          "N-bar'_chi)")
    print("-" * 78)
    sample_T = [10.0, 25.0, 50.0, 100.0]
    for chi in characters:
        a = character_parity(chi)
        q = int(chi.modulus)
        print()
        print(f"[chi = {chi.name}, q = {q}, parity a = {a}]")
        print(
            f"  {'T':>8} | {'theta_chi':>12} | "
            f"{'N-bar_chi':>10} | {'N-bar_chi prime':>14}"
        )
        for T in sample_T:
            theta = twisted_theta(T, chi)
            count = twisted_smooth_zero_count(T, chi)
            density = twisted_smooth_zero_density(T, chi)
            print(
                f"  {T:>8.2f} | {theta:>12.6f} | "
                f"{count:>10.6f} | {density:>14.6f}"
            )

    # ------------------------------------------------------------------
    # Section 2: derived vs actual zeros
    # ------------------------------------------------------------------
    print()
    print("-" * 78)
    print("Section 2: tilde gamma_n^(chi) (Newton on N-bar_chi) "
          "vs gamma_n^(chi) (Hardy-Z)")
    print("-" * 78)
    for chi in characters:
        print()
        print(f"[chi = {chi.name}]")
        actual_full = fetch_chi_zero_imaginary_parts(chi, 12)
        print(
            f"  {'n':>3} | {'tilde gamma_n':>14} | "
            f"{'gamma_n':>10} | {'residual r_n':>14}"
        )
        for n in range(1, 13):
            tilde = derive_twisted_smooth_zero_position(n, chi)
            actual = float(actual_full[n - 1])
            r = actual - tilde
            print(
                f"  {n:>3} | {tilde:>14.6f} | "
                f"{actual:>10.6f} | {r:>+14.6f}"
            )

    # ------------------------------------------------------------------
    # Section 3 + 4: full P46 certificate per character
    # ------------------------------------------------------------------
    print()
    print("-" * 78)
    print(f"Section 3+4: full P46 certificate (n_zeros = {n_zeros})")
    print("-" * 78)
    certs = []
    for chi in characters:
        print()
        cert = compute_twisted_structural_zero_density_certificate(
            chi,
            n_zeros=n_zeros,
            p34_n_primes=p34_n_primes,
            p34_max_power=p34_max_power,
        )
        certs.append(cert)
        print(cert.summary())

    # ------------------------------------------------------------------
    # Section 5: honest scope
    # ------------------------------------------------------------------
    print()
    print("-" * 78)
    print("Section 5: honest scope")
    print("-" * 78)
    print(
        "P46 derives the smooth chi-twisted zero density from\n"
        "TNFR archimedean ingredients alone (theta_chi, q, parity).\n"
        "The residuals r_n^(chi) = gamma_n - tilde gamma_n encode\n"
        "S_chi(T) = (1/pi) arg L(1/2 + iT, chi).  Bounding |S_chi(T)|\n"
        "uniformly is EQUIVALENT to GRH for L(s, chi) and is OPEN.\n"
        "P46 closes the structural origin of the smooth half of the\n"
        "chi-twisted spectral density; it does NOT prove GRH for any\n"
        "L(s, chi) and does NOT advance G4 = RH."
    )

    # Compact summary table
    print()
    print("-" * 78)
    print("Cross-character summary")
    print("-" * 78)
    print(
        f"  {'chi':>8} | {'q':>2} | {'a':>1} | {'max|r_n|':>10} | "
        f"{'W_1 struct':>11} | {'W_1 P34':>10} | "
        f"{'ratio':>7} | bound"
    )
    for cert in certs:
        print(
            f"  {cert.character_name:>8} | "
            f"{cert.character_modulus:>2} | "
            f"{cert.character_parity:>1} | "
            f"{cert.max_residual:>10.3e} | "
            f"{cert.w1_structural_vs_actual:>11.3e} | "
            f"{cert.w1_p34_vs_actual:>10.3e} | "
            f"{cert.improvement_ratio:>7.2f} | "
            f"{cert.bound_satisfied}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
