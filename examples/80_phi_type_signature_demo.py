"""Example 80 — phi-Type Signature Diagnostic (§13triginta-octava).

Runs the phi-Type Signature on a canonical TNFR ring evolution at two
resolutions and reports the diagnostic verdicts.

Scope (mandatory honesty)
-------------------------
This script is a *diagnostic only*.  It does **not** construct,
promote, or modify any canonical operator.  It does **not** advance
G4 = RH.  It is a *necessary-condition* check on whether canonical
phase evolution on a TNFR graph carries irreducible covering-space
structure that a single-sheet ``phi in [-pi, pi]`` reading cannot
represent without loss.

References
----------
- ``theory/TNFR_RIEMANN_RESEARCH_NOTES.md`` §13triginta-octava
- ``src/tnfr/riemann/phi_type_signature.py``
- ``src/tnfr/physics/_helpers.py::wrap_angle`` (canonical wrap)
"""

from __future__ import annotations

from tnfr.riemann import compute_phi_type_signature


def main() -> None:
    print("=" * 72)
    print("phi-Type Signature Diagnostic — §13triginta-octava.5")
    print("(Diagnostic only. Does NOT advance G4 = RH.)")
    print("=" * 72)

    print()
    print("--- Resolution 1: n_nodes=24, n_steps=64, n_bins=32 ---")
    cert1 = compute_phi_type_signature(
        n_nodes=24, n_steps=64, n_bins=32, seed=13
    )
    print(cert1.summary())

    print()
    print("--- Resolution 2: n_nodes=48, n_steps=128, n_bins=64 ---")
    cert2 = compute_phi_type_signature(
        n_nodes=48, n_steps=128, n_bins=64, seed=29
    )
    print(cert2.summary())

    print()
    print("=" * 72)
    print("Interpretation (§13triginta-octava.5–.6):")
    print("  - SCALAR_S1_ADEQUATE     : signature < 0.15 AND zero winding fraction")
    print("                             -> single-sheet phi in [-pi, pi] suffices")
    print("  - INDETERMINATE          : in between")
    print("  - COVER_LIFT_NECESSARY   : signature > 0.5 OR non-zero winding fraction")
    print("                             -> single-sheet phi loses canonical content")
    print()
    print("Verdicts at the two resolutions:")
    print(f"  res 1 (24/64/32): {cert1.verdict}")
    print(f"  res 2 (48/128/64): {cert2.verdict}")


if __name__ == "__main__":
    main()
