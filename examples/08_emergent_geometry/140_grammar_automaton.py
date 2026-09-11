#!/usr/bin/env python3
"""Example 140 -- A finite automaton for the flat grammar projection.

This example constructs the finite-state model of the constraints that can be
decided from a flat stream of default-depth operator names. The model covers U1,
U2, U4a, U4b, and the U2 REMESH presence check. It includes the current U2 prefix
debt capacity and distinguishes U4b's two memories: recent destabilization and
lifetime prior Coherence (IL).

The result is intentionally narrower than the full TNFR grammar:

* U3 phase compatibility is checked by operators against runtime node state;
* U6 compares structural potential with a reference field; and
* U5 nesting carries an explicit hierarchy. Unbounded THOL[...] nesting is
  stack-like/context-free and is not represented by this DFA.

For this flat projection, accepted-word counts are exact integers. The transfer
matrix and DFA are finite and exact; reported eigenvalues are floating-point
evaluations of that matrix. No automata result is a claim about TNFR physical
stability or equilibrium.
"""

import itertools
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import numpy as np

from tnfr.operators.grammar_validate import validate_grammar

from _flat_grammar_model import (
    ALPHA,
    NAME2INST,
    accepted_counts,
    build_automaton,
    minimize_dfa,
    spectral_radius,
    transfer_matrix,
)


def automaton_counts(edges, maxn):
    """Compatibility wrapper returning flat-model counts through ``maxn``."""

    return accepted_counts(edges, maxn)


def oracle_counts(maxn):
    """Return direct-validator counts for default-depth flat words."""

    from tnfr.operators.grammar_types import CLOSURES, GENERATORS

    out = []
    for n in range(1, maxn + 1):
        if n == 1:
            out.append(
                sum(
                    validate_grammar([NAME2INST[symbol]], 0.0)
                    for symbol in ALPHA
                )
            )
            continue
        count = 0
        # U1 pruning changes only enumeration cost, not the oracle result.
        for first in sorted(GENERATORS):
            for last in sorted(CLOSURES):
                for middle in itertools.product(ALPHA, repeat=n - 2):
                    sequence = [first, *middle, last]
                    if validate_grammar([NAME2INST[s] for s in sequence], 0.0):
                        count += 1
        out.append(count)
    return out


def experiment_1_automaton_reproduces_flat_oracle(states, edges):
    """Compare the centralized model with finite direct-oracle enumeration."""

    print("=" * 72)
    print("M1: FLAT AUTOMATON VS THE DIRECT VALIDATOR")
    print("=" * 72)
    print(f"  reachable flat-history states = {len(states)}")
    print(f"  labelled transitions          = {sum(map(len, edges.values()))}")
    auto = automaton_counts(edges, 6)
    oracle = oracle_counts(6)
    print(f"\n  {'n':>3} {'automaton':>12} {'validator':>12} {'match':>7}")
    for n, (left, right) in enumerate(zip(auto, oracle), start=1):
        print(f"  {n:>3} {left:>12} {right:>12} {str(left == right):>7}")
    assert auto == oracle
    print("\n  Exact finite cross-check: [2, 9, 84, 852, 9378, 109920].")
    print("  This checks the default-depth symbolic/history projection only.")


def experiment_2_minimal_dfa(states):
    """Minimize the complete flat DFA by partition refinement."""

    print("\n" + "=" * 72)
    print("M2: MINIMAL DFA OF THE FLAT PROJECTION")
    print("=" * 72)
    partition, _delta = minimize_dfa(states)
    classes = len(set(partition.values()))
    print(f"  raw reachable states        = {len(states)}")
    print(f"  minimal complete DFA states = {classes} (including the dead sink)")
    print("\n  The finite flat history language is regular by construction.")
    print("  This does not include the stack needed for unbounded U5 nesting.")
    return classes


def experiment_3_growth_rate(states, edges):
    """Compute the numerical spectral radius of the exact transfer matrix."""

    print("\n" + "=" * 72)
    print("M3: GROWTH RATE OF THE FINITE FLAT MODEL")
    print("=" * 72)
    trim, _index, matrix = transfer_matrix(states, edges)
    lam = spectral_radius(matrix)
    print(f"  co-reachable non-start states = {len(trim)}")
    print(f"  numerical spectral radius     = {lam:.10f}")
    print(f"  log2(radius)                  = {np.log2(lam):.6f} bits/operator")

    counts = automaton_counts(edges, 100)
    print("\n  finite count ratios approach the dominant growth rate:")
    print(f"  {'n':>4} {'N(n)/N(n-1)':>16} {'distance':>14}")
    for n in (6, 10, 20, 50, 100):
        ratio = counts[n - 1] / counts[n - 2]
        print(f"  {n:>4} {ratio:>16.8f} {abs(ratio - lam):>14.2e}")
    return lam


def main():
    print("\n" + "#" * 72)
    print("# Example 140 - Finite Automaton for the Flat Grammar Projection")
    print("#" * 72 + "\n")
    states, edges = build_automaton()
    experiment_1_automaton_reproduces_flat_oracle(states, edges)
    classes = experiment_2_minimal_dfa(states)
    lam = experiment_3_growth_rate(states, edges)

    print("\n" + "=" * 72)
    print("SCOPE")
    print("=" * 72)
    print(f"  The constructed flat model has a {classes}-state minimal DFA and")
    print(f"  numerical growth radius {lam:.10f}. These are properties of the")
    print("  finite symbolic/history projection. Runtime U3, reference-dependent")
    print("  U6, and nested U5 coherence remain outside this automaton; grammar")
    print("  labels alone do not certify a physical trajectory.")


if __name__ == "__main__":
    main()
