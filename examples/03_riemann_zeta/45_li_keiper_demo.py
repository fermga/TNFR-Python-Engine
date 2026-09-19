"""Example 45: finite Li-Keiper zero-sum comparisons (P16).

Li's criterion (Xian-Jin Li, 1997) states that the Riemann Hypothesis is
equivalent to the positivity of every Li-Keiper coefficient

    lambda_n = sum_rho [1 - (1 - 1/rho)^n],   n = 1, 2, 3, ...

This demo computes truncated sums for n = 1..N using:

  * the classical zeros provided by mpmath.zetazero (reference);
  * optionally, peak ordinates from the P13 classical analytic-continuation
    scan, explicitly placed on the critical line.

For rho = 1/2 + i*t, abs(1 - 1/rho) = 1. Every exact conjugate-pair
contribution is therefore nonnegative, even if t is not a zero ordinate.
The reported signs do not independently validate the scan or zero location.

Scope
-----
No omitted-zero or rounding enclosure is supplied. A finite sum's sign is
not a certified sign of the complete Li coefficient. Negative numerical
output would require an input/arithmetic investigation, not refute RH.
Compatibility labels such as certificate and lambda_n retain this finite
scope. No nodal trajectory, independent zero prediction or physical model
is implemented here.
"""

from __future__ import annotations

import sys

sys.stdout.reconfigure(encoding="utf-8")

from tnfr.riemann.li_keiper import LiKeiperCertificate, verify_li_keiper_criterion


def _print_section(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def section_1_classical_reference() -> LiKeiperCertificate:
    _print_section("Section 1 - Classical Li-Keiper coefficients (n = 1..30)")
    print("Computing lambda_n from 150 mpmath zetazero values at dps=40.")
    cert = verify_li_keiper_criterion(n_max=30, n_zeros=150, dps=40)
    print()
    print(cert.summary())
    print()
    print("First 10 truncated Li-Keiper sums:")
    for n in range(1, 11):
        print(f"  lambda_{n:<2d} = {cert.lambda_classical[n - 1]:+.6e}")
    print()
    print(
        "Computed truncated sums positive for n in [1, 30]: "
        f"{cert.positivity_classical}"
    )
    return cert


def section_2_extended_range() -> LiKeiperCertificate:
    _print_section("Section 2 - Extended range (n = 1..60) with deeper truncation")
    print("Computing lambda_n from 250 mpmath zetazero values at dps=50.")
    cert = verify_li_keiper_criterion(n_max=60, n_zeros=250, dps=50)
    print()
    print(cert.summary())
    print()
    print(
        "lambda_n grows asymptotically as ~(n/2) log(n/2pi); the values "
        "below are the truncated estimates."
    )
    print()
    for n in (1, 5, 10, 20, 30, 40, 50, 60):
        print(f"  lambda_{n:<2d} = {cert.lambda_classical[n - 1]:+.6e}")
    print()
    print(f"min_n lambda_n = {float(cert.lambda_classical.min()):+.6e}")
    print(
        "These critical-line inputs give nonnegative exact paired terms; "
        "the finite sign check does not independently test RH."
    )
    return cert


def section_3_tnfr_comparison() -> None:
    _print_section("Section 3 - TNFR resonance peaks vs classical zeros (optional)")
    print(
        "Re-running with compare_tnfr=True: P13 peak ordinates from the\n"
        "critical-line scan on t in [10, 80] are assigned real part 1/2;\n"
        "the finite conjugate-pair sums are recomputed from those coordinates."
    )
    print(
        "NOTE: the TNFR scan covers a much smaller t-window than mpmath\n"
        "(the supplied zero list), so values reflect different truncations.\n"
        "Nonnegative exact terms follow from the imposed real part, not a zero test."
    )
    cert = verify_li_keiper_criterion(
        n_max=20,
        n_zeros=80,
        dps=40,
        compare_tnfr=True,
        tnfr_t_min=10.0,
        tnfr_t_max=80.0,
        tnfr_n_samples=4001,
    )
    print()
    print(cert.summary())
    if cert.lambda_tnfr is not None:
        print()
        print("Side-by-side (first 10):")
        print("    n   lambda_classical    lambda_tnfr      |Δ|")
        for n in range(1, 11):
            cls = cert.lambda_classical[n - 1]
            tnf = cert.lambda_tnfr[n - 1]
            print(f"  {n:>3d}   {cls:+.6e}     {tnf:+.6e}   " f"{abs(cls - tnf):.3e}")


def section_4_interpretation() -> None:
    _print_section("Section 4 - Operational interpretation")
    print(
        "What this finite comparison reports:\n"
        "  * Li's classical criterion concerns complete coefficients\n"
        "    at every positive integer index.\n"
        "  * This example reports truncated sums from supplied zeros\n"
        "    and separately from peak ordinates placed on Re(rho)=1/2.\n"
        "  * Each exact paired term is nonnegative on that line, even\n"
        "    without zero membership; signs do not validate the scan.\n"
        "\n"
        "What P16 does NOT establish:\n"
        "  * No certified omitted-zero tail or rounding enclosure is\n"
        "    returned, even for a single complete coefficient.\n"
        "  * RH, an independently derived Hilbert-Polya bridge and\n"
        "    autonomous arithmetic NFR formation remain open.\n"
        "  * A negative numerical sum requires checking its inputs\n"
        "    and arithmetic; it is not a refutation of RH.\n"
    )


def main() -> None:
    print("TNFR-Riemann Programme - Example 45")
    print("Finite Li-Keiper zero-sum comparisons (P16)")
    section_1_classical_reference()
    section_2_extended_range()
    section_3_tnfr_comparison()
    section_4_interpretation()
    print()
    print("=" * 72)
    print("Done.")
    print("=" * 72)


if __name__ == "__main__":
    main()
