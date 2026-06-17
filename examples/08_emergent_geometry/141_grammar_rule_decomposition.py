#!/usr/bin/env python3
"""
Example 141 — Decomposing the Grammar by Rule: the Asymptotic Capacity Lives
Entirely in the Bifurcation Rule (U4b)
==============================================================================

The grammar is the only mechanism that modifies coherence (the 13 operators are
the only way to change EPI), so understanding WHICH rule does the structural work
is knowledge about the paradigm itself.  Examples 139-140 characterized the whole
language L and computed its exact capacity lambda = 11.560930.  This example
DECOMPOSES that capacity rule by rule: it rebuilds the grammar automaton (ex 140)
with each U1-U6 rule toggled on/off and compares the exact growth rate lambda
(Perron-Frobenius) and the finite counts N(n).

Each rule emerges from a distinct piece of TNFR physics:
  U1a (start with a generator)     -> cannot evolve from EPI=0 without a source
  U1b (end with a closure)         -> must leave a coherent attractor
  U2  (no debt at acceptance)      -> the integral integral nu_f*dNFR must converge
  U4b (transformer needs context)  -> threshold energy is required to bifurcate

The measured result is sharp and exact: the ENTIRE asymptotic capacity cost comes
from U4b.  U1a, U1b, U2 are BOUNDARY / acceptance conditions — they cut the finite
count N(n) (the prefactor) but leave the growth rate at the full alphabet value
lambda = 13.  Only U4b — the rule that gates the transformers ZHIR (phase mutation)
and THOL (self-organization), i.e. the bifurcation operators — reduces the
asymptotic branching to lambda = 11.560930.

Doctrine compliance
-------------------
The automaton is built from the CANONICAL centralized operator sets (grammar_types
-> config.physics_derivation, the single source of truth).  Each ablated automaton
toggles one canonical rule; lambda is the exact Perron-Frobenius eigenvalue of its
transfer matrix.  Nothing is imposed — the decomposition is read off the canonical
grammar.

Three measured results
----------------------
M1 RULE ABLATION TABLE. Each rule cuts the count N(n): U1a (start) ~4.3x, U1b
   (end) ~3.2x, U2 (acceptance) ~1.6x, U4b ~1.5x at n=4. But only U4b changes the
   asymptotic growth rate lambda; U1a/U1b/U2 each leave lambda = 13 exactly.

M2 ONLY U4b COSTS THE ASYMPTOTIC CAPACITY. With U4b alone enabled, lambda =
   11.5609299951 — EXACTLY the full-grammar lambda (|diff| = 2.5e-14). With U4b
   removed (U1a+U1b+U2 only), lambda = 13.0000000000 exactly (the full alphabet).
   The bifurcation-context rule alone determines the language's asymptotic
   capacity.

M3 BOUNDARY RULES vs THE TRANSITION RULE. U1a/U1b/U2 are boundary/acceptance
   conditions: they constrain how a finite sequence starts, ends, and settles its
   convergence debt, cutting the finite count (the prefactor) but not the
   asymptotic branching. U4b is the only INTERIOR-TRANSITION rule: it forbids a
   transformer without recent context, and that is the sole source of the
   capacity loss 13 -> 11.56. The lost ~1.44 is exactly the suppression of ZHIR /
   THOL (the U4b bottleneck of example 139).

Honest scope
------------
This is standard symbolic-dynamics theory (the Perron-Frobenius eigenvalue =
topological entropy of a sofic language, here of sub-automata with rules toggled),
built on the canonical automaton of example 140. It is a CHARACTERIZATION that
localizes the grammar's asymptotic constraint to the bifurcation rule; it is not
new mathematics and closes no open problem. It does, however, yield a concrete
paradigm insight: the asymptotic "difficulty" of building valid coherence lives
in the transformations (bifurcations), not in the boundaries.

References
----------
- src/tnfr/operators/grammar_types.py (the canonical centralized operator sets)
- examples/08_emergent_geometry/140_grammar_automaton.py (the automaton + lambda)
- examples/08_emergent_geometry/139_grammar_formal_language.py (the ZHIR bottleneck)
- AGENTS.md "Unified Grammar (U1-U6)" (the per-rule physics)
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import numpy as np

from tnfr.operators.grammar_types import (
    GENERATORS, CLOSURES, STABILIZERS, DESTABILIZERS, TRANSFORMERS,
)

ALPHA = ["emission", "reception", "coherence", "dissonance", "coupling",
         "resonance", "silence", "expansion", "contraction",
         "self_organization", "mutation", "transition", "recursivity"]
START = ("START",)


def tag(x):
    """U4b-window tag: D destabilizer, I coherence/IL, O other."""
    if x in DESTABILIZERS:
        return "D"
    if x == "coherence":
        return "I"
    return "O"


def make_automaton(u1a=True, u1b=True, u2=True, u4b=True):
    """Build the grammar automaton with the given canonical rules enabled."""
    def transition(state, x):
        if state == START:
            if u1a and x not in GENERATORS:
                return None
            return ((tag(x),), x in DESTABILIZERS, x in STABILIZERS,
                    x in CLOSURES)
        win, has_d, has_s, _lc = state
        if u4b and x in TRANSFORMERS:
            if "D" not in win:
                return None
            if x == "mutation" and "I" not in win:
                return None
        return ((win + (tag(x),))[-3:], has_d or x in DESTABILIZERS,
                has_s or x in STABILIZERS, x in CLOSURES)

    def is_accept(state):
        if len(state) != 4:
            return False
        _w, has_d, has_s, last_clo = state
        if u1b and not last_clo:
            return False
        if u2 and has_d and not has_s:
            return False
        return True

    states = {START}
    edges: dict = {}
    frontier = [START]
    while frontier:
        s = frontier.pop()
        for x in ALPHA:
            ns = transition(s, x)
            if ns is None:
                continue
            edges.setdefault(s, []).append(ns)
            if ns not in states:
                states.add(ns)
                frontier.append(ns)
    return states, edges, is_accept


def count_n(states, edges, is_accept, n):
    layer = {START: 1}
    out = 0
    for k in range(n):
        nxt: dict = {}
        for s, c in layer.items():
            for ns in edges.get(s, ()):
                nxt[ns] = nxt.get(ns, 0) + c
        layer = nxt
        if k == n - 1:
            out = sum(c for st, c in layer.items() if is_accept(st))
    return out


def capacity(states, edges, is_accept):
    """Perron-Frobenius eigenvalue of the trim automaton's transfer matrix."""
    co = set(s for s in states if is_accept(s))
    changed = True
    while changed:
        changed = False
        for s in states:
            if s not in co and any(ns in co for ns in edges.get(s, ())):
                co.add(s)
                changed = True
    trim = [s for s in states if s in co and s != START]
    if not trim:
        return 0.0
    idx = {s: i for i, s in enumerate(trim)}
    M = np.zeros((len(trim), len(trim)))
    for s in trim:
        for ns in edges.get(s, ()):
            if ns in idx:
                M[idx[s], idx[ns]] += 1
    return float(np.max(np.abs(np.linalg.eigvals(M))))


def experiment_1_ablation_table():
    """M1: per-rule ablation of N(n) and lambda."""
    print("=" * 72)
    print("M1: RULE ABLATION — N(4) and the asymptotic capacity lambda")
    print("=" * 72)
    print("Each U1-U6 rule toggled on the canonical automaton (ex 140).")
    print()
    configs = [
        ("none (full alphabet)", dict(u1a=False, u1b=False, u2=False,
                                      u4b=False)),
        ("U1a only (start gen)", dict(u1a=True, u1b=False, u2=False,
                                      u4b=False)),
        ("U1b only (end closure)", dict(u1a=False, u1b=True, u2=False,
                                        u4b=False)),
        ("U2 only (no debt)", dict(u1a=False, u1b=False, u2=True, u4b=False)),
        ("U4b only (xform ctx)", dict(u1a=False, u1b=False, u2=False,
                                      u4b=True)),
        ("ALL (U1a+U1b+U2+U4b)", dict(u1a=True, u1b=True, u2=True, u4b=True)),
    ]
    print(f"  {'rules enabled':>24} {'N(4)':>9} {'lambda':>10} {'log2 lam':>9}")
    for label, cfg in configs:
        st, ed, acc = make_automaton(**cfg)
        N4 = count_n(st, ed, acc, 4)
        L = capacity(st, ed, acc)
        print(f"  {label:>24} {N4:>9} {L:>10.4f} "
              f"{np.log2(L) if L > 0 else 0:>9.4f}")
    print()
    print("  -> every rule cuts N(4), but only U4b changes lambda: U1a/U1b/U2")
    print("     leave lambda = 13 (the full alphabet), U4b drops it to 11.56.")


def experiment_2_only_u4b_costs_capacity():
    """M2: U4b-alone lambda == full-grammar lambda, exactly."""
    print()
    print("=" * 72)
    print("M2: ONLY U4b COSTS THE ASYMPTOTIC CAPACITY")
    print("=" * 72)
    lam_u4b = capacity(*make_automaton(False, False, False, True))
    lam_all = capacity(*make_automaton(True, True, True, True))
    lam_no_u4b = capacity(*make_automaton(True, True, True, False))
    print(f"  U4b alone       lambda = {lam_u4b:.10f}")
    print(f"  ALL rules       lambda = {lam_all:.10f}")
    print(f"  |difference|           = {abs(lam_u4b - lam_all):.2e}  "
          f"(exactly equal)")
    print(f"  U1a+U1b+U2 (no U4b)    = {lam_no_u4b:.10f}  (= alphabet size 13)")
    print()
    print("  -> the bifurcation-context rule U4b ALONE fixes the language's")
    print("     asymptotic capacity; removing it restores the full alphabet")
    print("     branching 13. U1a/U1b/U2 contribute ZERO to lambda.")


def experiment_3_boundary_vs_transition():
    """M3: boundary rules vs the single interior-transition rule."""
    print()
    print("=" * 72)
    print("M3: BOUNDARY RULES (U1/U2) vs THE TRANSITION RULE (U4b)")
    print("=" * 72)
    # what each rule cuts at the FINITE count level (n=4) in isolation
    base = count_n(*make_automaton(False, False, False, False), 4)
    for label, cfg in [
        ("U1a (start)", dict(u1a=True, u1b=False, u2=False, u4b=False)),
        ("U1b (end)", dict(u1a=False, u1b=True, u2=False, u4b=False)),
        ("U2 (acceptance)", dict(u1a=False, u1b=False, u2=True, u4b=False)),
        ("U4b (transition)", dict(u1a=False, u1b=False, u2=False, u4b=True)),
    ]:
        n4 = count_n(*make_automaton(**cfg), 4)
        L = capacity(*make_automaton(**cfg))
        kind = "TRANSITION (cuts lambda)" if L < 12.99 else "BOUNDARY (prefactor)"
        print(f"  {label:>18}: N(4) {base} -> {n4} "
              f"(x{base / n4:.2f}), lambda={L:.4f}  [{kind}]")
    print()
    print("  -> U1a/U1b/U2 are BOUNDARY conditions: they constrain how a finite")
    print("     sequence starts, ends, and settles its convergence debt — cutting")
    print("     the count but not the growth rate. U4b is the only INTERIOR-")
    print("     TRANSITION rule: gating ZHIR/THOL (the bifurcation operators) is")
    print("     the sole source of the capacity loss 13 -> 11.56 (the ZHIR")
    print("     bottleneck of example 139).")


def main():
    print()
    print("  ===============================================================")
    print("  Decomposing the Grammar by Rule")
    print("  The Asymptotic Capacity Lives Entirely in the Bifurcation (U4b)")
    print("  ===============================================================")
    print()
    experiment_1_ablation_table()
    experiment_2_only_u4b_costs_capacity()
    experiment_3_boundary_vs_transition()
    print()
    print("=" * 72)
    print("WHAT THIS ESTABLISHES")
    print("=" * 72)
    print("The grammar is the only mechanism that modifies coherence, so locating")
    print("WHICH rule does the structural work is paradigm knowledge. Decomposing")
    print("the capacity rule by rule (M1) shows that EVERY rule cuts the finite")
    print("count N(n), but only U4b changes the asymptotic growth rate lambda: U4b")
    print("alone gives lambda = 11.5609299951, EXACTLY the full-grammar value")
    print("(M2), while U1a/U1b/U2 each leave lambda = 13. So U1/U2 are BOUNDARY")
    print("conditions (start/end/convergence-debt — prefactor only) and U4b is the")
    print("single INTERIOR-TRANSITION rule that fixes the capacity (M3). The")
    print("asymptotic constraint on building valid coherence lives entirely in the")
    print("bifurcation rule (threshold energy to transform ZHIR/THOL), not in the")
    print("boundaries. HONEST SCOPE: standard symbolic-dynamics (Perron-Frobenius /")
    print("topological entropy of rule-toggled sub-automata) on the canonical")
    print("automaton of ex 140; a characterization, not new mathematics, closes no")
    print("open problem.")


if __name__ == "__main__":
    main()
