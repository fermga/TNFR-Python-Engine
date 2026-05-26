"""Example 78 — νf-Type Signature Diagnostic (§13triginta-prima).

Runs the νf-Type Signature on canonical P14 prime-ladder spectra at two
resolutions and reports the diagnostic verdicts.

Scope (mandatory honesty)
-------------------------
This script is a *diagnostic only*.  It does **not** construct, promote,
or modify any canonical operator.  It does **not** advance G4 = RH.
It is a *necessary-condition* check on whether the canonical P14 data
carries irreducible measure-valued structure for :math:`\\nu_f`.

References
----------
- ``theory/TNFR_RIEMANN_RESEARCH_NOTES.md`` §13triginta-prima
- ``src/tnfr/riemann/nuf_type_signature.py``
"""

from __future__ import annotations

from tnfr.riemann import compute_nuf_type_signature


def main() -> None:
    print("=" * 72)
    print("νf-Type Signature Diagnostic — §13triginta-prima.6")
    print("(Diagnostic only. Does NOT advance G4 = RH.)")
    print("=" * 72)

    print()
    print("--- Resolution 1: n_primes=200, max_power=8, n_bins=64 ---")
    cert1 = compute_nuf_type_signature(n_primes=200, max_power=8, n_bins=64)
    print(cert1.summary())

    print()
    print("--- Resolution 2: n_primes=400, max_power=12, n_bins=128 ---")
    cert2 = compute_nuf_type_signature(n_primes=400, max_power=12, n_bins=128)
    print(cert2.summary())

    print()
    print("=" * 72)
    print("Interpretation (§13triginta-prima.6–.7):")
    print("  - SCALAR_ADEQUATE  : signature < 0.15 → scalar νf suffices")
    print("  - INDETERMINATE    : 0.15 ≤ signature ≤ 0.5")
    print("  - MEASURE_VALUED_NECESSARY : signature > 0.5 → scalar νf loses info")
    print()
    print("Verdicts at the two resolutions:")
    print(f"  resolution 1: {cert1.verdict}  (S_νf = {cert1.signature:.4f})")
    print(f"  resolution 2: {cert2.verdict}  (S_νf = {cert2.signature:.4f})")
    print()
    print("Necessary-condition check, NOT proof of the νf-Type Conjecture.")
    print("Honest scope: does NOT advance G4 = RH; does NOT promote any operator.")
    print("=" * 72)


if __name__ == "__main__":
    main()
