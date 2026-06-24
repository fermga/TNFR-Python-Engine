"""Example 81 — DeltaNFR-Type Signature Diagnostic (§13quadraginta).

Runs the DeltaNFR-Type Signature on a canonical TNFR ring evolution at
two resolutions and reports the diagnostic verdicts.

Scope (mandatory honesty)
-------------------------
This script is a *diagnostic only*.  It does **not** construct,
promote, or modify any canonical operator.  It does **not** advance
G4 = RH.  It is a *necessary-condition* check on whether canonical
gradient accumulation on a TNFR graph carries irreducible
multi-channel tensor content that a rank-1 scalar reading necessarily
compresses.

References
----------
- ``theory/TNFR_RIEMANN_RESEARCH_NOTES.md`` §13quadraginta
- ``src/tnfr/riemann/dnfr_type_signature.py``
- ``src/tnfr/dynamics/dnfr.py:2387::default_compute_delta_nfr``
  (canonical scalar-writing implementation)
"""

from __future__ import annotations

from tnfr.riemann import compute_dnfr_type_signature


def main() -> None:
    print("=" * 72)
    print("DeltaNFR-Type Signature Diagnostic — §13quadraginta.5")
    print("(Diagnostic only. Does NOT advance G4 = RH.)")
    print("=" * 72)

    print()
    print("--- Resolution 1: n_nodes=24, n_steps=64 ---")
    cert1 = compute_dnfr_type_signature(n_nodes=24, n_steps=64, seed=17)
    print(cert1.summary())

    print()
    print("--- Resolution 2: n_nodes=48, n_steps=128 ---")
    cert2 = compute_dnfr_type_signature(n_nodes=48, n_steps=128, seed=31)
    print(cert2.summary())

    print()
    print("=" * 72)
    print("Interpretation (§13quadraginta.5–.6):")
    print(
        "  - SCALAR_DNFR_ADEQUATE   : signature < 0.15 AND zero tensor storage fraction"
    )
    print("                             -> rank-1 scalar DeltaNFR suffices")
    print("  - INDETERMINATE          : in between")
    print(
        "  - TENSOR_LIFT_NECESSARY  : signature > 0.5 OR non-zero tensor storage fraction"
    )
    print(
        "                             -> rank-1 scalar DeltaNFR loses canonical content"
    )
    print()
    print("Verdicts at the two resolutions:")
    print(f"  res 1 (24/64): {cert1.verdict}")
    print(f"  res 2 (48/128): {cert2.verdict}")


if __name__ == "__main__":
    main()
