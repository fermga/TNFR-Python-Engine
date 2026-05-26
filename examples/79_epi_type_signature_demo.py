"""Example 79 — EPI-Type Signature Diagnostic (§13triginta-quarta).

Runs the EPI-Type Signature on a canonical TNFR ring evolution at two
resolutions and reports the diagnostic verdicts.

Scope (mandatory honesty)
-------------------------
This script is a *diagnostic only*.  It does **not** construct, promote,
or modify any canonical operator.  It does **not** advance G4 = RH.
It is a *necessary-condition* check on whether canonical EPI evolution
on a TNFR graph carries irreducible BEPIElement-valued structure that
a single-mode scalar reading cannot represent without loss.

References
----------
- ``theory/TNFR_RIEMANN_RESEARCH_NOTES.md`` §13triginta-quarta
- ``src/tnfr/riemann/epi_type_signature.py``
- ``src/tnfr/mathematics/epi.py`` (BEPIElement definition)
"""

from __future__ import annotations

from tnfr.riemann import compute_epi_type_signature


def main() -> None:
    print("=" * 72)
    print("EPI-Type Signature Diagnostic — §13triginta-quarta.6")
    print("(Diagnostic only. Does NOT advance G4 = RH.)")
    print("=" * 72)

    print()
    print("--- Resolution 1: n_nodes=24, n_steps=64, n_bins=32 ---")
    cert1 = compute_epi_type_signature(
        n_nodes=24, n_steps=64, n_bins=32, seed=13
    )
    print(cert1.summary())

    print()
    print("--- Resolution 2: n_nodes=48, n_steps=128, n_bins=64 ---")
    cert2 = compute_epi_type_signature(
        n_nodes=48, n_steps=128, n_bins=64, seed=29
    )
    print(cert2.summary())

    print()
    print("=" * 72)
    print("Interpretation (§13triginta-quarta.6–.7):")
    print("  - SCALAR_ADEQUATE       : signature < 0.15 AND zero BEPI storage")
    print("                            → scalar EPI suffices for canonical evolution")
    print("  - INDETERMINATE         : in between")
    print("  - BEPI_VALUED_NECESSARY : signature > 0.5 OR non-zero BEPI storage")
    print("                            → scalar EPI loses canonical content")
    print()
    print("Verdicts at the two resolutions:")
    print(f"  resolution 1: {cert1.verdict}  (S_EPI = {cert1.signature:.4f},"
          f" BEPI fraction = {cert1.storage_bepi_fraction:.4f})")
    print(f"  resolution 2: {cert2.verdict}  (S_EPI = {cert2.signature:.4f},"
          f" BEPI fraction = {cert2.storage_bepi_fraction:.4f})")
    print()
    print("Necessary-condition check, NOT proof of the T-EPI Conjecture.")
    print("Honest scope: does NOT advance G4 = RH; does NOT promote any operator.")
    print("=" * 72)


if __name__ == "__main__":
    main()
