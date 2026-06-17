#!/usr/bin/env python3
"""
Example 142 — The Grammatical Quotient of the Operator Alphabet: the Static
Grammar Distinguishes 9 Roles, and Four Operators Are Grammatically Free
==============================================================================

The grammar is the only mechanism that modifies coherence (the 13 operators are
the only way to change EPI), so knowing exactly which operators the grammar can
and cannot tell apart is knowledge about the paradigm itself. Examples 139-141
characterized the language L, its automaton, and the rule (U4b) that fixes its
capacity. This example computes the SYMBOL-LEVEL QUOTIENT: the partition of the
13 operators into grammatical equivalence classes.

DEFINITION (static grammatical equivalence). Two operators a, b are equivalent,
a ~ b, iff replacing any occurrence of a by b (and vice versa) in EVERY
grammar-valid sequence preserves validity. Exactly: a ~ b iff they induce
identical transitions on every state of the canonical automaton (the same
START-legality, the same next state from every interior state, the same effect
on the acceptance flags). This is the symbol-level Myhill-Nerode quotient.

The measured result is sharp: the 13 operators collapse to exactly 9 classes,
and one class — {EN, UM, RA, NUL} (Reception, Coupling, Resonance, Contraction)
— is a FREE INTERIOR class: the static grammar carries no constraint that
separates them. Their distinguishing canonical constraints are RUNTIME, not
static-sequence rules: U3 (phase compatibility |phi_i - phi_j| <= dphi_max for
UM/RA) is a phase-STATE check at the moment of coupling, and the Reception /
Contraction contracts are telemetry/physics effects, none of which constrains
the operator SEQUENCE. So the quotient precisely delineates the SCOPE of the
static sequence grammar versus what it delegates to the runtime/phase/telemetry
layer.

Doctrine compliance
-------------------
The quotient is computed on the CANONICAL automaton, whose transitions are built
entirely from the centralized operator sets (grammar_types -> config.
physics_derivation, the single source of truth). The equivalence relation is
exact (identity of transition functions) and is independently confirmed by the
canonical validate_grammar oracle (51206 in-class substitutions, zero broken).
Nothing is imposed; the partition is read off the canonical grammar. This does
NOT claim the four operators have the same physical function — their nodal
dynamics are distinct (that is precisely why their constraints live in the
runtime layer); it states that the STATIC SEQUENCE grammar cannot resolve them.

Three measured results
----------------------
M1 NINE GRAMMATICAL CLASSES. The 13 operators partition into exactly 9 classes,
   one per realized role-combination: {EN, UM, RA, NUL} (free interior),
   {NAV, REMESH} (generator+closure), and seven singletons {AL} (generator),
   {IL} (stabilizer), {OZ} (closure+destabilizer), {SHA} (closure),
   {VAL} (destabilizer), {THOL} (stabilizer+transformer),
   {ZHIR} (destabilizer+transformer). The grammar's RESOLUTION is 9, not 13.

M2 IN-CLASS SUBSTITUTION IS EXACT, CROSS-CLASS BREAKS. Enumerating all 10343
   valid sequences of length <= 5, every one of the 51206 in-class symbol
   substitutions preserves validity (0 broken) — the classes are exact. A
   cross-class substitution (e.g. AL -> EN at a generator position) breaks
   validity, so the 9 classes are genuinely distinct.

M3 THE REDUNDANCY GAP. The symbol-counting capacity (ex 140, 13 symbols) is
   lambda_sym = 11.560930 = 3.531 bits/op; the role-counting capacity (9
   classes, collapsing the free-interior class to one) is lambda_cls = 8.752927
   = 3.130 bits/role. The gap 0.401 bits/op is the per-operator entropy the
   static grammar leaves UNCONSTRAINED — the free choice among grammatically
   interchangeable operators, almost all of it inside {EN, UM, RA, NUL}.

Honest scope
------------
This is the standard Myhill-Nerode symbol quotient (an exact equivalence
relation) plus the Perron-Frobenius capacity, computed on the canonical
automaton of example 140. It is a CHARACTERIZATION that delineates the scope of
the static sequence grammar — which operator differences it constrains (9 role
signatures) and which it delegates to the runtime phase/telemetry layer
({EN, UM, RA, NUL}). It is not new mathematics and closes no open problem. The
paradigm insight is precise: the static grammar resolves operators only up to
their grammatical role, and the operators whose canonical constraints are phase-
or telemetry-based (U3 coupling, reception, contraction) are invisible to it.

References
----------
- src/tnfr/operators/grammar_types.py (the canonical centralized operator sets)
- src/tnfr/operators/grammar_validate.py (the static validate_grammar oracle)
- examples/08_emergent_geometry/140_grammar_automaton.py (the automaton + lambda)
- examples/08_emergent_geometry/141_grammar_rule_decomposition.py (U4b capacity)
- AGENTS.md "Unified Grammar (U1-U6)" (U3 is a runtime phase check, not static)
"""

import os
import sys
import itertools

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import numpy as np

from tnfr.operators.grammar_types import (
    GENERATORS, CLOSURES, STABILIZERS, DESTABILIZERS, TRANSFORMERS,
)
from tnfr.operators.definitions import (
    Emission, Reception, Coherence, Dissonance, Coupling, Resonance,
    Silence, Expansion, Contraction, SelfOrganization, Mutation,
    Transition, Recursivity,
)
from tnfr.operators.grammar_validate import validate_grammar

ALPHA = ["emission", "reception", "coherence", "dissonance", "coupling",
         "resonance", "silence", "expansion", "contraction",
         "self_organization", "mutation", "transition", "recursivity"]
SHORT = {"emission": "AL", "reception": "EN", "coherence": "IL",
         "dissonance": "OZ", "coupling": "UM", "resonance": "RA",
         "silence": "SHA", "expansion": "VAL", "contraction": "NUL",
         "self_organization": "THOL", "mutation": "ZHIR", "transition": "NAV",
         "recursivity": "REMESH"}
NAME2INST = {
    "emission": Emission(), "reception": Reception(), "coherence": Coherence(),
    "dissonance": Dissonance(), "coupling": Coupling(), "resonance": Resonance(),
    "silence": Silence(), "expansion": Expansion(), "contraction": Contraction(),
    "self_organization": SelfOrganization(), "mutation": Mutation(),
    "transition": Transition(), "recursivity": Recursivity(),
}
START = ("START",)


def tag(x):
    """U4b-window tag: D destabilizer, I coherence/IL, O other."""
    if x in DESTABILIZERS:
        return "D"
    if x == "coherence":
        return "I"
    return "O"


def transition(state, x):
    """Canonical automaton transition (U1a start, U4b interior gate)."""
    if state == START:
        if x not in GENERATORS:
            return None
        return ((tag(x),), x in DESTABILIZERS, x in STABILIZERS, x in CLOSURES)
    win, has_d, has_s, _lc = state
    if x in TRANSFORMERS:
        if "D" not in win:
            return None
        if x == "mutation" and "I" not in win:
            return None
    return ((win + (tag(x),))[-3:], has_d or x in DESTABILIZERS,
            has_s or x in STABILIZERS, x in CLOSURES)


def is_accept(state):
    """U1b closure + U2 convergence acceptance condition."""
    if len(state) != 4:
        return False
    _w, has_d, has_s, last_clo = state
    return last_clo and (not has_d or has_s)


def reachable_states():
    states = {START}
    frontier = [START]
    while frontier:
        s = frontier.pop()
        for x in ALPHA:
            ns = transition(s, x)
            if ns is not None and ns not in states:
                states.add(ns)
                frontier.append(ns)
    return sorted(states, key=lambda s: (len(s), str(s)))


def role_label(x):
    roles = []
    if x in GENERATORS:
        roles.append("generator")
    if x in CLOSURES:
        roles.append("closure")
    if x in STABILIZERS:
        roles.append("stabilizer")
    if x in DESTABILIZERS:
        roles.append("destabilizer")
    if x in TRANSFORMERS:
        roles.append("transformer")
    return "+".join(roles) if roles else "free interior"


def equivalence_classes(states):
    """a ~ b iff identical transitions on every state (exact quotient)."""
    classes = []
    assigned = set()
    for a in ALPHA:
        if a in assigned:
            continue
        cls = [a]
        assigned.add(a)
        for b in ALPHA:
            if b in assigned:
                continue
            if all(transition(s, a) == transition(s, b) for s in states):
                cls.append(b)
                assigned.add(b)
        classes.append(cls)
    return classes


def capacity(alpha, states):
    """Perron-Frobenius eigenvalue of the trim transfer matrix over alpha."""
    co = set(s for s in states if is_accept(s))
    changed = True
    while changed:
        changed = False
        for s in states:
            if s in co:
                continue
            for x in alpha:
                ns = transition(s, x)
                if ns in co:
                    co.add(s)
                    changed = True
                    break
    trim = [s for s in states if s in co and s != START]
    idx = {s: i for i, s in enumerate(trim)}
    n = len(trim)
    M = np.zeros((n, n))
    for s in trim:
        for x in alpha:
            ns = transition(s, x)
            if ns in idx:
                M[idx[s], idx[ns]] += 1
    return float(max(np.linalg.eigvals(M).real))


def experiment_1_classes(states):
    print("=" * 72)
    print("E1: the grammatical equivalence classes (exact symbol quotient)")
    print("=" * 72)
    classes = equivalence_classes(states)
    print(f"  {len(ALPHA)} operators -> {len(classes)} grammatical classes "
          f"(the static grammar's resolution)")
    print()
    for cls in sorted(classes, key=lambda c: (-len(c), SHORT[c[0]])):
        members = ", ".join(SHORT[x] for x in cls)
        print(f"    {{{members:22s}}}  [{role_label(cls[0])}]")
    print()
    print("  KEY: {EN, UM, RA, NUL} is the FREE INTERIOR class -- four operators")
    print("  with distinct nodal dynamics (reception, coupling, resonance,")
    print("  contraction) the STATIC grammar cannot separate, because their")
    print("  canonical constraints are RUNTIME (U3 phase for UM/RA; telemetry")
    print("  contracts for EN/NUL), not static-sequence rules.")
    return classes


def experiment_2_substitution(classes):
    print()
    print("=" * 72)
    print("E2: in-class substitution is exact; cross-class breaks validity")
    print("=" * 72)
    valid_seqs = []
    for length in range(1, 6):
        for combo in itertools.product(ALPHA, repeat=length):
            if validate_grammar([NAME2INST[s] for s in combo], 0.0):
                valid_seqs.append(combo)
    print(f"  enumerated {len(valid_seqs)} grammar-valid sequences (len <= 5)")

    class_of = {x: c for c in classes for x in c}
    checked = 0
    broken = 0
    for seq in valid_seqs:
        for i, s in enumerate(seq):
            cls = class_of[s]
            if len(cls) == 1:
                continue
            for repl in cls:
                if repl == s:
                    continue
                seq2 = list(seq)
                seq2[i] = repl
                checked += 1
                if not validate_grammar([NAME2INST[x] for x in seq2], 0.0):
                    broken += 1
    print(f"  in-class substitutions: checked {checked}, broken {broken}  "
          f"-> classes exact: {broken == 0}")

    print("  cross-class substitutions that BREAK validity (classes distinct):")
    shown = 0
    for ca, cb in itertools.combinations(classes, 2):
        if shown >= 4:
            break
        a, b = ca[0], cb[0]
        done = False
        for seq in valid_seqs:
            for i, s in enumerate(seq):
                if s != a:
                    continue
                seq2 = list(seq)
                seq2[i] = b
                if not validate_grammar([NAME2INST[x] for x in seq2], 0.0):
                    label = " ".join(SHORT[x] for x in seq)
                    print(f"    [{label}]  swap {SHORT[a]}->{SHORT[b]} @ pos {i}"
                          f"  -> invalid")
                    done = True
                    shown += 1
                    break
            if done:
                break


def experiment_3_redundancy(classes, states):
    print()
    print("=" * 72)
    print("E3: the redundancy gap -- bits the static grammar leaves free")
    print("=" * 72)
    lam_sym = capacity(ALPHA, states)
    # collapse the free-interior class to a single representative
    free = max(classes, key=len)
    reps = [x for x in ALPHA if x == free[0] or x not in free]
    lam_cls = capacity(reps, states)
    print(f"  symbol-counting capacity (13 symbols)  lambda_sym = {lam_sym:.6f}"
          f"  = {np.log2(lam_sym):.3f} bits/op")
    print(f"  role-counting capacity   ( 9 classes)  lambda_cls = {lam_cls:.6f}"
          f"  = {np.log2(lam_cls):.3f} bits/role")
    gap = np.log2(lam_sym) - np.log2(lam_cls)
    print(f"  redundancy gap = {gap:.3f} bits/op")
    print("  -> the static grammar constrains operators up to their role")
    print("     (3.13 bits/role); the remaining 0.40 bits/op is free choice")
    print(f"     among interchangeable operators, almost all inside "
          f"{{{', '.join(SHORT[x] for x in free)}}}.")


def main():
    print()
    print("#" * 72)
    print("# Example 142 - The Grammatical Quotient of the Operator Alphabet")
    print("#" * 72)
    print()
    states = reachable_states()
    classes = experiment_1_classes(states)
    experiment_2_substitution(classes)
    experiment_3_redundancy(classes, states)
    print()
    print("=" * 72)
    print("Summary")
    print("=" * 72)
    print("  The static grammar U1-U6 resolves the 13 operators into 9 role")
    print("  classes. Four operators -- EN, UM, RA, NUL -- are grammatically")
    print("  free: their constraints (U3 phase coupling, reception/contraction")
    print("  contracts) live in the runtime/telemetry layer, not in the static")
    print("  sequence grammar. The quotient delineates exactly what the static")
    print("  grammar constrains; characterization, no open problem closed.")
    print()


if __name__ == "__main__":
    main()
