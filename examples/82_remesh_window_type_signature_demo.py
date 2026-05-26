"""Example 82 — REMESH-Window-Type Signature Diagnostic (§13quadraginta-tertia).

Runs the REMESH-Window-Type Signature on a canonical TNFR ring
evolution at two resolutions and reports the diagnostic verdicts.

Scope (mandatory honesty)
-------------------------
This script is a *diagnostic only*.  It does **not** construct,
promote, or modify any canonical operator.  It does **not** advance
G4 = RH.  It is a *necessary-condition* check on whether canonical
REMESH evolution on a TNFR graph carries an irreducible continuous-
time or fractional-order kernel that the canonical integer-indexed
window (τ_l, τ_g) ∈ ℕ × ℕ necessarily discretises away.

References
----------
- ``theory/TNFR_RIEMANN_RESEARCH_NOTES.md`` §13quadraginta-tertia
- ``theory/REMESH_INFINITY_DERIVATION.md`` §§1–8 (N15 closure)
- ``src/tnfr/riemann/remesh_window_type_signature.py``
- ``src/tnfr/operators/remesh.py:1212::apply_network_remesh``
  (canonical integer-indexed implementation)
"""

from __future__ import annotations

from tnfr.riemann import compute_remesh_window_type_signature


def main() -> None:
    print("=" * 72)
    print("REMESH-Window-Type Signature Diagnostic — §13quadraginta-tertia.5")
    print("(Diagnostic only. Does NOT advance G4 = RH.)")
    print("=" * 72)

    print()
    print("--- Resolution 1: n_nodes=24, warmup=16, (tau_l, tau_g)=(4, 8), events=8 ---")
    cert1 = compute_remesh_window_type_signature(
        n_nodes=24,
        warmup_steps=16,
        tau_l=4,
        tau_g=8,
        remesh_events_per_window=8,
        seed=17,
    )
    print(cert1.summary())

    print()
    print("--- Resolution 2: n_nodes=48, warmup=24, (tau_l, tau_g)=(6, 12), events=12 ---")
    cert2 = compute_remesh_window_type_signature(
        n_nodes=48,
        warmup_steps=24,
        tau_l=6,
        tau_g=12,
        remesh_events_per_window=12,
        seed=31,
    )
    print(cert2.summary())

    print()
    print("=" * 72)
    print("Interpretation (§13quadraginta-tertia.5-.6):")
    print("  - INTEGER_WINDOW_ADEQUATE     : signature < 0.15 AND integer storage fraction == 1.0")
    print("                                  -> integer-indexed (tau_l, tau_g) in N x N suffices")
    print("  - INDETERMINATE               : in between")
    print("  - CONTINUOUS_KERNEL_NECESSARY : signature > 0.5 OR integer storage fraction < 1.0")
    print("                                  -> continuous kernel K(t,s) may be required")
    print()
    print("Verdicts at the two resolutions:")
    print(f"  res 1 (24/16/4-8/8):  {cert1.verdict}")
    print(f"  res 2 (48/24/6-12/12): {cert2.verdict}")


if __name__ == "__main__":
    main()
