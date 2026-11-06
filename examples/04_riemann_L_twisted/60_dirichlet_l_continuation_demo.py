"""P33 demo: analytic continuation of chi-twisted prime-ladder L-series.

Shows that the chi-twisted TNFR prime-ladder Dirichlet trace built in
:mod:`tnfr.riemann.dirichlet_l` (P32) admits the same kind of analytic
continuation as the ordinary von Mangoldt series for ``-zeta'/zeta``
(P13).  Two checks are performed for every demo character:

1.  ``verify_twisted_continuation_agreement`` -- compares the
    chi-twisted prime-ladder partial sum to the mpmath continuation
    ``-L'(s, chi) / L(s, chi)`` on Re(s) > 1.
2.  ``scan_critical_line_for_l_poles`` -- locates the non-trivial
    zeros of ``L(s, chi)`` as resonance poles of ``-L'/L`` on the
    critical line Re(s) = 1/2, then compares against the LMFDB
    reference values.

The demo does NOT prove the generalised Riemann hypothesis.  It
illustrates that the structural P32 prime-ladder representation, once
continued analytically, exposes the same critical-line resonance
spectrum that the classical L-function does.

Tip: this script takes 30-90 seconds to run depending on the machine,
since each critical-line sample evaluates two high-precision mpmath
Dirichlet L calls.
"""

from __future__ import annotations

import numpy as np

from tnfr.riemann.analytic_continuation_dirichlet import (
    dirichlet_l_continued,
    dirichlet_log_l_derivative_continued,
    scan_critical_line_for_l_poles,
    verify_twisted_continuation_agreement,
)
from tnfr.riemann.dirichlet_l import (
    build_twisted_prime_ladder_spectrum,
    real_character_mod_3,
    real_character_mod_4,
    real_character_mod_5,
)

# LMFDB-tabulated first imaginary parts of non-trivial zeros, used as
# a manual cross-check for the critical-line scan.  Only the
# characters whose first zeros have been verified against LMFDB are
# listed here.
LMFDB_FIRST_ZEROS = {
    "chi_real_mod_3": [
        8.0397376938782723,
        11.249163239605370,
        15.704619170895680,
        18.261996003884728,
        20.466739547486924,
        24.061509200416502,
    ],
    "chi_real_mod_4": [
        6.0209489046975965,
        10.243770304166600,
        12.988098018343661,
        16.342535860090340,
        18.291998361050880,
        21.448296247574580,
        23.283290051908552,
    ],
}


def _print_agreement_report(label: str, ag) -> None:
    print(f"  Character   : {ag.chi_name}")
    print(f"  Samples     : {ag.s_values.size}")
    print(f"  Quality     : {ag.agreement_quality}")
    print(f"  max |abs err|: {ag.max_abs_diff:.3e}")
    print(f"  max |rel err|: {ag.max_rel_diff:.3e}")
    print(f"  Per-point summary ({label}):")
    for s, zpl, zco, rel in zip(
        ag.s_values, ag.z_prime_ladder, ag.z_continued, ag.rel_diff
    ):
        print(
            f"    s = {s.real:+.2f}{s.imag:+.2f}j   "
            f"Z_TNFR = {zpl.real:+.6f}{zpl.imag:+.6f}j   "
            f"-L'/L = {zco.real:+.6f}{zco.imag:+.6f}j   "
            f"rel = {rel:.2e}"
        )


def _print_pole_scan_report(scan, reference: list[float]) -> None:
    print(f"  Character        : {scan.chi_name}")
    print(f"  Range t          : [{scan.t_values[0]:.1f}, {scan.t_values[-1]:.1f}]")
    print(f"  Samples          : {scan.t_values.size}")
    print(f"  Detected peaks   : {scan.n_peaks}")
    print(f"  Median spacing   : {scan.median_spacing:.3f}")
    print(f"  LMFDB cross-check:")
    detected = list(scan.detected_peaks)
    pairs = list(zip(detected, reference))
    for det, ref in pairs:
        diff = abs(det - ref)
        marker = "OK" if diff < 0.05 else "??"
        print(
            f"    detected t = {det:>8.4f}   LMFDB t = {ref:>8.4f}"
            f"   |dt| = {diff:.4f}   [{marker}]"
        )
    if len(detected) > len(reference):
        for extra in detected[len(reference) :]:
            print(f"    extra peak t = {extra:>8.4f}   (beyond LMFDB shortlist)")
    if len(reference) > len(detected):
        for missing in reference[len(detected) :]:
            print(
                f"    missing peak t = {missing:>8.4f}   (not detected in scan range)"
            )


def main() -> None:
    print("=" * 78)
    print("P33: ANALYTIC CONTINUATION OF chi-TWISTED PRIME-LADDER L-SERIES")
    print("=" * 78)
    print()
    print("Goal: extend the chi-twisted prime-ladder Dirichlet trace built in")
    print("P32 (dirichlet_l.py) to the whole complex plane via the unique")
    print("meromorphic continuation -L'(s, chi)/L(s, chi).  For non-principal")
    print("primitive characters this is entire, so the only poles of -L'/L are")
    print("the non-trivial zeros of L(s, chi) on the critical line.")
    print()

    characters = [
        ("chi_3 (Legendre symbol mod 3)", real_character_mod_3()),
        ("chi_4 (Dirichlet beta function)", real_character_mod_4()),
        ("chi_5 (Legendre symbol mod 5)", real_character_mod_5()),
    ]

    # ------------------------------------------------------------------
    # Step 1: high-precision smoke test of the continuation evaluators
    # ------------------------------------------------------------------
    print("-" * 78)
    print("Step 1: smoke test of L(s, chi) and -L'(s, chi) / L(s, chi) at s = 2")
    print("-" * 78)
    for label, chi in characters:
        l_val = dirichlet_l_continued(chi, 2.0, dps=30)
        log_d = dirichlet_log_l_derivative_continued(chi, 2.0, dps=30)
        print(f"  {label}")
        print(f"     L(2, chi)      = {l_val.real:+.12f}{l_val.imag:+.12e}j")
        print(f"     -L'(2)/L(2)    = {log_d.real:+.12f}{log_d.imag:+.12e}j")
    print()

    # ------------------------------------------------------------------
    # Step 2: agreement between prime ladder and continuation on Re(s) > 1
    # ------------------------------------------------------------------
    print("-" * 78)
    print("Step 2: chi-twisted prime ladder vs analytic continuation on Re(s) > 1")
    print("-" * 78)
    s_check = [2 + 0j, 2 + 1j, 3 + 0j, 3 + 2j, 5 + 0j]
    for label, chi in characters:
        spec = build_twisted_prime_ladder_spectrum(chi, n_primes=400, max_power=14)
        agreement = verify_twisted_continuation_agreement(spec, chi, s_check, dps=30)
        print(f"  {label}")
        _print_agreement_report(label, agreement)
        print()

    # ------------------------------------------------------------------
    # Step 3: critical-line resonance pole scan (vs LMFDB tabulation)
    # ------------------------------------------------------------------
    print("-" * 78)
    print("Step 3: critical-line scan locating zeros of L(s, chi)")
    print("        (resonance poles of -L'/L on Re(s) = 1/2)")
    print("-" * 78)

    scan_configs = [
        (
            "chi_3 (Legendre symbol mod 3)",
            real_character_mod_3(),
            (5.0, 25.0, 2001, 3.0),
        ),
        (
            "chi_4 (Dirichlet beta function)",
            real_character_mod_4(),
            (5.0, 25.0, 2001, 3.0),
        ),
    ]
    for label, chi, (t_min, t_max, n, prom) in scan_configs:
        scan = scan_critical_line_for_l_poles(
            chi,
            t_min=t_min,
            t_max=t_max,
            n_samples=n,
            dps=20,
            peak_prominence=prom,
        )
        print(f"  {label}")
        _print_pole_scan_report(scan, LMFDB_FIRST_ZEROS[scan.chi_name])
        print()

    print("=" * 78)
    print("STATUS: P33 extends P13 (analytic continuation of the prime-ladder")
    print("        Dirichlet trace) from zeta to every Dirichlet L-function.")
    print("        It does NOT prove the generalised Riemann hypothesis.")
    print("        Same arithmetic obstruction as G4 for zeta.")
    print("=" * 78)


if __name__ == "__main__":  # pragma: no cover
    main()
