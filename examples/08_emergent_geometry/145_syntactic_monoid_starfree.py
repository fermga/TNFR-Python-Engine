#!/usr/bin/env python3
"""Example 145 -- Syntactic monoid of the flat grammar projection.

The minimal DFA from example 140 has a finite transition monoid. Exhaustive
closure and power checks show that this particular monoid is aperiodic. Standard
Schuetzenberger and McNaughton--Papert results then classify the constructed flat
language as star-free and FO[<]-definable.

The conclusion does not classify the explicit nested U5 language or runtime U3
and U6 acceptance. Those layers are absent from the DFA and may require a stack,
graph state, and reference telemetry.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from _flat_grammar_model import ALPHA, build_automaton, minimize_dfa


def transition_monoid(partition, delta):
    """Generate the transition monoid of a minimal complete DFA."""

    classes = sorted(set(partition.values()))
    class_index = {cls: i for i, cls in enumerate(classes)}
    representatives = {}
    for state, cls in partition.items():
        representatives.setdefault(cls, state)

    def generator(symbol):
        return tuple(
            class_index[partition[delta(representatives[cls], symbol)]]
            for cls in classes
        )

    identity = tuple(range(len(classes)))
    generators = {symbol: generator(symbol) for symbol in ALPHA}

    def compose(left, right):
        return tuple(right[left[i]] for i in range(len(classes)))

    monoid = {identity, *generators.values()}
    frontier = list(monoid)
    while frontier:
        element = frontier.pop()
        for generator_map in generators.values():
            product = compose(element, generator_map)
            if product not in monoid:
                monoid.add(product)
                frontier.append(product)
    return monoid, compose, identity, len(classes)


def stability_index(monoid, compose):
    """Return max minimal n satisfying x**n = x**(n+1), or ``None``."""

    largest = 0
    for element in monoid:
        power = element
        for exponent in range(1, len(monoid) + 2):
            next_power = compose(power, element)
            if next_power == power:
                largest = max(largest, exponent)
                break
            power = next_power
        else:
            return None
    return largest


def experiment_1_monoid(monoid, compose, number_of_states):
    print("=" * 72)
    print("M1: TRANSITION MONOID OF THE MINIMAL FLAT DFA")
    print("=" * 72)
    idempotents = sum(compose(element, element) == element for element in monoid)
    print(f"  minimal DFA states       = {number_of_states}")
    print(f"  transition-monoid size   = {len(monoid)}")
    print(f"  idempotent transformations = {idempotents}")
    print("\n  Because the DFA is minimal and complete, this transition monoid is")
    print("  the syntactic monoid of the constructed flat language.")
    return idempotents


def experiment_2_aperiodicity(monoid, compose):
    print("\n" + "=" * 72)
    print("M2: EXHAUSTIVE APERIODICITY CHECK")
    print("=" * 72)
    index = stability_index(monoid, compose)
    print(f"  every element stabilizes under powers: {index is not None}")
    print(f"  maximum first stabilization exponent: {index}")
    assert index is not None

    # A two-state parity automaton supplies a small periodic contrast.
    identity = (0, 1)
    toggle = (1, 0)

    def parity_compose(left, right):
        return tuple(right[left[i]] for i in range(2))

    parity_index = stability_index({identity, toggle}, parity_compose)
    print(f"  parity contrast stabilizes: {parity_index is not None}")
    assert parity_index is None
    print("\n  The finite monoid is aperiodic; the parity monoid exposes the power")
    print("  cycle that the same exhaustive criterion rejects.")
    return index


def experiment_3_classification(index):
    print("\n" + "=" * 72)
    print("M3: CLASSIFICATION OF THE CONSTRUCTED FLAT LANGUAGE")
    print("=" * 72)
    print(f"  computed premise: finite syntactic monoid is aperiodic (index {index})")
    print("  Schuetzenberger: aperiodic syntactic monoid iff star-free language")
    print("  McNaughton--Papert: star-free iff definable in FO[<]")
    print("\n  Therefore the flat language is star-free and FO[<]-definable.")
    print("  This theorem application does not include unbounded brackets or any")
    print("  numerical phase/potential condition.")


def main():
    print("\n" + "#" * 72)
    print("# Example 145 - Syntactic Monoid of the Flat Grammar Projection")
    print("#" * 72 + "\n")
    states, _edges = build_automaton()
    partition, delta = minimize_dfa(states)
    monoid, compose, _identity, state_count = transition_monoid(partition, delta)
    experiment_1_monoid(monoid, compose, state_count)
    index = experiment_2_aperiodicity(monoid, compose)
    experiment_3_classification(index)
    print("\n  Scope: the finite non-nested symbolic/history language only; U3,")
    print("  hierarchy-aware U5, and reference-dependent U6 remain external.")


if __name__ == "__main__":
    main()
