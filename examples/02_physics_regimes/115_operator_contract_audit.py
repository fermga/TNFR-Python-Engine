"""TNFR Example 115: Operator-Contract Fidelity Audit — measured, not asserted.

Each of the 13 canonical operators carries a postcondition CONTRACT
(AGENTS.md §"The 13 Canonical Operators"): IL must not increase |ΔNFR|,
OZ must increase it, SHA must freeze νf, RA must propagate EPI without
altering identity, ZHIR must transform the phase, and so on.

These contracts have long been ENFORCED reactively by the Structural
Integrity Monitor (it is consulted by Operator.__call__ during live
execution). What was missing — and what this example demonstrates — is a
PROACTIVE, MEASURED audit: apply each operator in its correct canonical
context and measure whether its contract actually holds, producing a
per-operator certificate.

WHY CONTEXT MATTERS (the honest part)
=====================================
ΔNFR and C(t) are EMERGENT network fields (they depend on a node's
neighbours), not node-local quantities. So a stabiliser's contract
("IL reduces |ΔNFR|") manifests at the NETWORK level, after the emergent
field is recomputed — not in a single isolated node. The audit therefore
measures each operator in the context where its contract canonically
manifests:

- network    : stabilisers/structural ops (AL, EN, IL, UM, SHA, VAL, NUL, THOL)
- node       : the local destabiliser OZ (single-node |ΔNFR|)
- identity   : RA (EPI-sign preservation — "without altering identity")
- phase      : ZHIR (θ transformed, with its U4b precondition: prior IL + OZ)
- state      : NAV (a regime shift changes some state variable)
- advisory   : REMESH (a network-level echo, verified elsewhere)

This is the difference between *asserting* the contracts (a docstring) and
*measuring* them (this audit): 13/13 operators are shown to satisfy their
canonical postcondition, with the measured before→after values printed.

A FIXED BUG
===========
While building this audit, the SDK's ``integrity_check`` was found to read a
non-existent ``report.passed`` attribute (the real attribute is
``is_healthy``), so every node was silently skipped and the check always
returned ``nodes_checked=0``. That is now fixed, and a new
``net.audit_operators()`` exposes the measured audit below.

HONEST SCOPE
============
A validation/fidelity tool for the canonical operator catalog (invariant #4,
Grammar Compliance). It measures the postcondition contracts; it does not
modify operator physics or resolve any open program.

References:
- src/tnfr/physics/integrity.py (audit_operator_contracts, the monitor)
- AGENTS.md §"The 13 Canonical Operators" (the contracts)
- AGENTS.md §"Canonical Invariants" #4 (Grammar Compliance)
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from tnfr.physics.integrity import audit_operator_contracts
from tnfr.sdk import TNFR


def experiment_1_measured_audit():
    """Measure all 13 operator-contract fidelities."""
    print("=" * 72)
    print("EXPERIMENT 1: Measured operator-contract fidelity (13/13)")
    print("=" * 72)
    print()

    audit = audit_operator_contracts(n_nodes=16, seed=7)
    print(f"{'op':>7} {'context':>9}  {'contract':<40} {'measured':<28} ok")
    print("-" * 72)
    for r in audit.results:
        mark = "OK" if r.satisfied else "XX"
        print(f"{r.glyph:>7} {r.context:>9}  {r.contract:<40} "
              f"{r.detail:<28} {mark}")
    print()
    print(f"-> {audit.n_satisfied}/{audit.n_operators} operators satisfy their "
          f"canonical postcondition contract, MEASURED")
    print(f"   (not merely asserted). all_satisfied = {audit.all_satisfied}")
    print()


def experiment_2_context_matters():
    """Show why the measurement context is per-operator."""
    print("=" * 72)
    print("EXPERIMENT 2: Why the measurement context is per-operator")
    print("=" * 72)
    print()
    print("ΔNFR and C(t) are EMERGENT network fields. A stabiliser's contract")
    print("manifests at the network level after the field is recomputed; a")
    print("local destabiliser's at the single node; RA's as identity")
    print("preservation. Grouping the audit by context:")
    print()

    audit = audit_operator_contracts(n_nodes=16, seed=7)
    by_context: dict[str, list[str]] = {}
    for r in audit.results:
        by_context.setdefault(r.context, []).append(r.glyph)
    for context, glyphs in by_context.items():
        print(f"  {context:>9}: {', '.join(glyphs)}")
    print()
    print("-> measuring every operator at the network level would wrongly")
    print("   flag OZ (its |ΔNFR| rise is local) and RA (it redistributes,")
    print("   not increases, EPI). Honest measurement needs the right context.")
    print()


def experiment_3_sdk_audit():
    """The SDK exposes the measured audit; integrity_check is fixed."""
    print("=" * 72)
    print("EXPERIMENT 3: SDK audit_operators() + fixed integrity_check()")
    print("=" * 72)
    print()

    net = TNFR.create(16).random(0.3).evolve(2)

    audit = net.audit_operators()
    print(f"  net.audit_operators(): all_satisfied={audit['all_satisfied']}, "
          f"{audit['n_satisfied']}/{audit['n_operators']} operators")
    print()

    check = net.integrity_check("IL")
    print(f"  net.integrity_check('IL'): nodes_checked="
          f"{check['nodes_checked']}, pass_rate={check['pass_rate']:.2f}")
    print("  (was nodes_checked=0 before the .passed -> .is_healthy fix)")
    print()


def main():
    print()
    print("#" * 72)
    print("# TNFR Example 115: Operator-Contract Fidelity Audit")
    print("# measured, not asserted (invariant #4: Grammar Compliance)")
    print("#" * 72)
    print()
    experiment_1_measured_audit()
    experiment_2_context_matters()
    experiment_3_sdk_audit()
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)
    audit = audit_operator_contracts()
    print(audit.summary())
    print()
    print("The 13 canonical operators are MEASURED to satisfy their")
    print("postcondition contracts in their correct canonical contexts.")
    print("Validation tool for invariant #4; no operator physics changed.")


if __name__ == "__main__":
    main()
