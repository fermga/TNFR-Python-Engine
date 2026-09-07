#!/usr/bin/env python3
"""Example 142 -- Symbol quotient of the flat grammar automaton.

Two operator names are equivalent here when they induce the same transformation
on every state of the *minimal* DFA from example 140. This is equivalence for the
flat symbolic/history language only. It does not identify operator physics.

Runtime phase checks (U3), reference-dependent potential drift (U6), and nested
parent/child coherence (U5) can distinguish operators or executions that this
symbol quotient cannot observe.
"""

import itertools
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import numpy as np

from tnfr.operators.grammar_types import (
    CLOSURES,
    DESTABILIZERS,
    GENERATORS,
    STABILIZERS,
    TRANSFORMERS,
)
from tnfr.operators.grammar_validate import validate_grammar

from _flat_grammar_model import (
    ALPHA,
    NAME2INST,
    SHORT,
    build_automaton,
    minimize_dfa,
    spectral_radius,
    transfer_matrix,
)


def role_label(symbol):
    roles = []
    if symbol in GENERATORS:
        roles.append("generator")
    if symbol in CLOSURES:
        roles.append("closure")
    if symbol in STABILIZERS:
        roles.append("stabilizer")
    if symbol in DESTABILIZERS:
        roles.append("destabilizer")
    if symbol in TRANSFORMERS:
        roles.append("transformer")
    return "+".join(roles) if roles else "no flat-history role"


def reachable_states():
    """Compatibility helper returning the centralized reachable state list."""

    states, _edges = build_automaton()
    return sorted(states, key=str)


def equivalence_classes(states):
    """Partition symbols by identical action on the minimal complete DFA."""

    partition, delta = minimize_dfa(states)
    representatives = {}
    for state, block in partition.items():
        representatives.setdefault(block, state)

    classes = []
    assigned = set()
    for left in ALPHA:
        if left in assigned:
            continue
        cls = []
        for right in ALPHA:
            if right in assigned:
                continue
            same_action = all(
                partition[delta(state, left)] == partition[delta(state, right)]
                for state in representatives.values()
            )
            if same_action:
                cls.append(right)
        assigned.update(cls)
        classes.append(cls)
    return classes


def _growth_radius(alpha):
    states, edges = build_automaton(alpha=alpha)
    _trim, _index, matrix = transfer_matrix(states, edges)
    return spectral_radius(matrix)


def experiment_1_classes(states):
    print("=" * 72)
    print("M1: EXACT SYMBOL ACTIONS ON THE MINIMAL FLAT DFA")
    print("=" * 72)
    classes = equivalence_classes(states)
    print(f"  {len(ALPHA)} operator names -> {len(classes)} DFA-action classes\n")
    for cls in sorted(classes, key=lambda value: (-len(value), SHORT[value[0]])):
        members = ", ".join(SHORT[symbol] for symbol in cls)
        print(f"    {{{members:22s}}} [{role_label(cls[0])}]")
    print("\n  {EN, UM, RA, NUL} shares one flat-history action; {NAV, REMESH}")
    print("  shares another. Their graph effects and runtime contracts remain")
    print("  distinct and are not quotiented by this experiment.")
    return classes


def experiment_2_finite_oracle_crosscheck(classes):
    print("\n" + "=" * 72)
    print("M2: FINITE DIRECT-VALIDATOR CROSS-CHECK")
    print("=" * 72)
    valid_words = []
    for length in range(1, 6):
        for word in itertools.product(ALPHA, repeat=length):
            if validate_grammar([NAME2INST[s] for s in word], 0.0):
                valid_words.append(word)

    class_of = {symbol: cls for cls in classes for symbol in cls}
    checked = 0
    broken = 0
    for word in valid_words:
        for position, symbol in enumerate(word):
            for replacement in class_of[symbol]:
                if replacement == symbol:
                    continue
                candidate = list(word)
                candidate[position] = replacement
                checked += 1
                if not validate_grammar([NAME2INST[s] for s in candidate], 0.0):
                    broken += 1

    print(f"  accepted words of length <= 5 = {len(valid_words)}")
    print(f"  in-class substitutions checked = {checked}")
    print(f"  changed validator outcomes      = {broken}")
    assert broken == 0
    print("\n  The finite oracle sample agrees with the exact minimal-DFA action.")


def experiment_3_alphabet_collapse(classes):
    print("\n" + "=" * 72)
    print("M3: GROWTH-RATE CHANGE AFTER COLLAPSING SYMBOL CLASSES")
    print("=" * 72)
    representatives = [cls[0] for cls in classes]
    symbol_radius = _growth_radius(ALPHA)
    class_radius = _growth_radius(representatives)
    log_gap = np.log2(symbol_radius) - np.log2(class_radius)
    print(f"  13-symbol radius = {symbol_radius:.10f}")
    print(f"   9-class radius  = {class_radius:.10f}")
    print(f"  log2 difference  = {log_gap:.6f} bits/operator")
    print("\n  This difference measures alphabet multiplicity in these two finite")
    print("  counting models. It is not a physical entropy or an operator-energy")
    print("  redundancy theorem.")


def main():
    print("\n" + "#" * 72)
    print("# Example 142 - Symbol Quotient of the Flat Grammar Automaton")
    print("#" * 72 + "\n")
    states = reachable_states()
    classes = experiment_1_classes(states)
    experiment_2_finite_oracle_crosscheck(classes)
    experiment_3_alphabet_collapse(classes)
    print("\n  Scope: flat, non-nested operator histories. U3, U5 hierarchy, and U6")
    print("  telemetry remain outside the quotient and require runtime evidence.")


if __name__ == "__main__":
    main()
