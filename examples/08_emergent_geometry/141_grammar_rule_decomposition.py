#!/usr/bin/env python3
"""Example 141 -- Rule ablation in the flat grammar automaton.

The finite automaton from example 140 permits a controlled ablation study of
history-only constraints. Earlier versions attributed the entire reduction in
word-growth rate to U4b. That conclusion became false when U2 acquired causal
prefix-debt accounting: U2 and U4b both restrict interior transitions, and their
combination has a different spectral radius from either rule alone.

This is an automata comparison, not a causal or physical energy decomposition.
U3 needs runtime phase state, U6 needs a potential reference, and U5's unbounded
nested form is stack-like/context-free. They cannot be toggled as DFA bits here.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import numpy as np

from _flat_grammar_model import (
    DEFAULT_RULES,
    FlatRules,
    accepted_counts,
    build_automaton,
    spectral_radius,
    transfer_matrix,
)


def make_automaton(
    u1a=True,
    u1b=True,
    u2=True,
    u4b=True,
    u4a=True,
    u2_remesh=True,
):
    """Compatibility wrapper returning a rule-configured flat automaton."""

    rules = FlatRules(u1a, u1b, u2, u4a, u4b, u2_remesh)
    states, labelled_edges = build_automaton(rules)
    edges = {
        state: [next_state for _symbol, next_state in outgoing]
        for state, outgoing in labelled_edges.items()
    }
    return states, edges, lambda state: _accept_with_rules(state, rules)


def _accept_with_rules(state, rules):
    from _flat_grammar_model import is_accept

    return is_accept(state, rules)


def _configured(rules):
    return (*build_automaton(rules), rules)


def count_n(states, edges, is_accept, n):
    """Compatibility count for legacy callers using unlabelled edges."""

    layer = {("START",): 1}
    for _ in range(n):
        next_layer = {}
        for state, count in layer.items():
            for next_state in edges.get(state, ()):
                next_layer[next_state] = next_layer.get(next_state, 0) + count
        layer = next_layer
    return sum(count for state, count in layer.items() if is_accept(state))


def capacity(states, edges, is_accept):
    """Numerically evaluate the growth radius for a legacy edge mapping."""

    coreachable = {state for state in states if is_accept(state)}
    changed = True
    while changed:
        changed = False
        for state in states - coreachable:
            if any(next_state in coreachable for next_state in edges.get(state, ())):
                coreachable.add(state)
                changed = True
    trim = sorted((state for state in coreachable if state != ("START",)), key=str)
    index = {state: i for i, state in enumerate(trim)}
    matrix = np.zeros((len(trim), len(trim)))
    for state in trim:
        for next_state in edges.get(state, ()):
            if next_state in index:
                matrix[index[state], index[next_state]] += 1
    return spectral_radius(matrix)


def metrics(rules):
    states, edges = build_automaton(rules)
    n4 = accepted_counts(edges, 4, rules)[-1]
    _trim, _index, matrix = transfer_matrix(states, edges, rules)
    return n4, spectral_radius(matrix)


def experiment_1_ablation_table():
    print("=" * 72)
    print("M1: RULE ABLATION IN THE FLAT SYMBOLIC/HISTORY MODEL")
    print("=" * 72)
    configurations = [
        ("none", FlatRules(False, False, False, False, False, False)),
        ("U1a start only", FlatRules(True, False, False, False, False, False)),
        ("U1b closure only", FlatRules(False, True, False, False, False, False)),
        ("U2 debt only", FlatRules(False, False, True, False, False, False)),
        ("U4a presence only", FlatRules(False, False, False, True, False, False)),
        ("U4b context only", FlatRules(False, False, False, False, True, False)),
        ("all flat checks", DEFAULT_RULES),
    ]
    print(f"  {'enabled checks':>20} {'N(4)':>9} {'radius':>13} {'log2':>9}")
    results = {}
    for label, rules in configurations:
        n4, radius = metrics(rules)
        results[label] = (n4, radius)
        print(f"  {label:>20} {n4:>9} {radius:>13.8f} {np.log2(radius):>9.5f}")
    print("\n  U1a, U1b, and U4a alter finite acceptance without changing the")
    print("  dominant radius in these isolated ablations. U2 prefix debt and")
    print("  U4b context each reduce it because both reject interior prefixes.")
    return results


def experiment_2_joint_constraint():
    print("\n" + "=" * 72)
    print("M2: U2 AND U4b ARE DISTINCT, INTERACTING INTERIOR CONSTRAINTS")
    print("=" * 72)
    u2_only = FlatRules(False, False, True, False, False, False)
    u4b_only = FlatRules(False, False, False, False, True, False)
    no_u4b = FlatRules(u4b=False)
    rows = [
        ("U2 only", metrics(u2_only)[1]),
        ("U4b only", metrics(u4b_only)[1]),
        ("all except U4b", metrics(no_u4b)[1]),
        ("all flat checks", metrics(DEFAULT_RULES)[1]),
    ]
    for label, radius in rows:
        print(f"  {label:>18}: radius = {radius:.10f}")
    assert rows[-1][1] < min(rows[0][1], rows[1][1])
    print("\n  The combined radius is lower than either isolated radius. This is")
    print("  a finite-model interaction, so additive 'capacity cost by rule' is")
    print("  not inferred from separate ablations.")


def experiment_3_scope():
    print("\n" + "=" * 72)
    print("M3: WHAT THE DFA CANNOT ABLATE")
    print("=" * 72)
    print("  U3: requires phases and the selected coupling pair at runtime.")
    print("  U5: unbounded THOL[...] nesting requires a stack/tree representation.")
    print("  U6: requires Phi_s before/after or another explicit reference state.")
    print("\n  Their absence from this table means 'not observable from a flat word',")
    print("  not 'zero effect' or 'redundant physical rule'.")


def main():
    print("\n" + "#" * 72)
    print("# Example 141 - Rule Ablation in the Flat Grammar Projection")
    print("#" * 72 + "\n")
    experiment_1_ablation_table()
    experiment_2_joint_constraint()
    experiment_3_scope()
    print("\n  These numerical radii characterize the constructed finite automata;")
    print("  they do not measure C(t), DeltaNFR, convergence, or equilibrium.")


if __name__ == "__main__":
    main()
