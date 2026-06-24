"""Example 83 — Delta-Phi-Max-Type Signature Diagnostic (§13quadraginta-sexta).

Runs the Delta-Phi-Max-Type Signature on canonical U3
(resonant-coupling) checks at two resolutions and reports the
diagnostic verdicts.

Scope (mandatory honesty)
-------------------------
This script is a *diagnostic only*.  It does **not** construct,
promote, or modify any canonical operator.  It does **not** advance
G4 = RH.  It is a *necessary-condition* check on whether the
canonical resonant-coupling threshold
:math:`\\Delta\\phi_{\\max} \\in [0, \\pi]` (scalar) carries an
irreducible edge-dependent or angle-of-attack-dependent functional
lift that the canonical scalar comparison necessarily collapses.

References
----------
- ``theory/TNFR_RIEMANN_RESEARCH_NOTES.md`` §13quadraginta-sexta
- ``src/tnfr/riemann/delta_phi_max_type_signature.py``
- ``src/tnfr/constants/canonical.py:506`` (canonical default
  ``DELTA_PHI_MAX = PI / 2``)
- ``src/tnfr/operators/grammar_dynamics.py:178-193`` (canonical U3
  check)
"""

from __future__ import annotations

from tnfr.riemann import compute_delta_phi_max_type_signature


def main() -> None:
    print("=" * 72)
    print("Delta-Phi-Max-Type Signature Diagnostic — §13quadraginta-sexta.5")
    print("(Diagnostic only. Does NOT advance G4 = RH.)")
    print("=" * 72)

    print()
    print("--- Resolution 1: n_nodes=24, anchors=9, offsets=8, seed=19 ---")
    cert1 = compute_delta_phi_max_type_signature(
        n_nodes=24,
        n_pair_anchors=9,
        n_offsets_per_anchor=8,
        seed=19,
    )
    print(cert1.summary())

    print()
    print("--- Resolution 2: n_nodes=48, anchors=17, offsets=16, seed=29 ---")
    cert2 = compute_delta_phi_max_type_signature(
        n_nodes=48,
        n_pair_anchors=17,
        n_offsets_per_anchor=16,
        seed=29,
    )
    print(cert2.summary())

    print()
    print("=" * 72)
    print("Interpretation (§13quadraginta-sexta.5-.6):")
    print(
        "  - SCALAR_THRESHOLD_ADEQUATE          : signature < 0.05 AND scalar storage fraction == 1.0"
    )
    print(
        "                                         -> scalar Delta_phi_max in [0, pi] suffices"
    )
    print("  - INDETERMINATE                      : in between")
    print(
        "  - EDGE_DEPENDENT_THRESHOLD_NECESSARY : signature > 0.25 OR scalar storage fraction < 1.0"
    )
    print(
        "                                         -> edge-dependent matrix or angle-of-attack lift may be required"
    )
    print()
    print("Verdicts at the two resolutions:")
    print(f"  res 1 (24/9/8/19):   {cert1.verdict}")
    print(f"  res 2 (48/17/16/29): {cert2.verdict}")


if __name__ == "__main__":
    main()
