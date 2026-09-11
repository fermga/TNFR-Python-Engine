#!/usr/bin/env python3
"""
Example 139 — Flat Grammar Projection as a Formal Language
==========================================================

This example enumerates **flat**, non-nested operator sequences over the 13
canonical glyphs and classifies them with ``validate_grammar`` from the
standalone initial condition ``epi_initial=0.0``. For the current fixed
policies, that flat projection can be represented with finite validator state
and therefore admits a DFA/regular-language interpretation.

The enumeration does not execute operators on a graph. In particular, U3's
graph-dependent phase-compatibility precondition is enforced by the operator
pipeline and is not decided by this symbol-only validator projection.

This scope excludes U5 syntax such as ``THOL[body]``. Arbitrarily nested,
balanced ``THOL[...]`` bodies require a stack or recursive production, so the
complete nested glyph language is context-free rather than regular. Nothing in
the flat enumeration proves otherwise.

The program reports exact counts only for lengths 1 through 6, adjacent finite
ratios, finite per-symbol log-counts, and the length-5 glyph distribution. These
measurements do not determine the asymptotic growth rate or topological entropy.
``log2(13)`` is merely the unconstrained alphabet upper bound; convergence to it
and sub-extensive constraint cost are not inferred from six lengths.

References
----------
- src/tnfr/operators/grammar_validate.py (the canonical U1-U6 validator)
- src/tnfr/operators/definitions.py (the 13 canonical operators)
- theory/UNIFIED_GRAMMAR_RULES.md (U1-U6 and nested U5 syntax)
- AGENTS.md "Unified Grammar (U1-U6)"
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import itertools
import math
from collections import Counter

from tnfr.operators.definitions import (
    Coherence,
    Contraction,
    Coupling,
    Dissonance,
    Emission,
    Expansion,
    Mutation,
    Reception,
    Recursivity,
    Resonance,
    SelfOrganization,
    Silence,
    Transition,
)
from tnfr.operators.grammar_validate import validate_grammar

OPS = [
    ("AL", Emission()),
    ("EN", Reception()),
    ("IL", Coherence()),
    ("OZ", Dissonance()),
    ("UM", Coupling()),
    ("RA", Resonance()),
    ("SHA", Silence()),
    ("VAL", Expansion()),
    ("NUL", Contraction()),
    ("THOL", SelfOrganization()),
    ("ZHIR", Mutation()),
    ("NAV", Transition()),
    ("REMESH", Recursivity()),
]
NAMES = [n for n, _ in OPS]
INST = [o for _, o in OPS]
A = len(OPS)
GENERATORS = [0, 11, 12]  # AL, NAV, REMESH  (U1a)
CLOSURES = [6, 11, 12, 3]  # SHA, NAV, REMESH, OZ  (U1b)
_CACHE = {}


def valid_sequences(n):
    """Enumerate valid flat length-n sequences with the U1 boundary prune.

    Every valid sequence must start with a generator and end with a closure
    (U1), so we only enumerate those; the canonical validator then decides the
    remaining flat constraints. Length one is tested over the full alphabet
    because the same glyph must satisfy both boundaries.
    """
    out = []
    if n == 1:
        for i in range(A):
            if validate_grammar([INST[i]], 0.0):
                out.append((i,))
        return out
    for first in GENERATORS:
        for last in CLOSURES:
            for mid in itertools.product(range(A), repeat=n - 2):
                combo = (first, *mid, last)
                if validate_grammar([INST[i] for i in combo], 0.0):
                    out.append(combo)
    return out


def experiment_1_regular_language():
    """M1: enumerate the flat regular projection and its U1 boundary."""
    print("=" * 70)
    print("M1: FLAT GRAMMAR PROJECTION AND THE U1 BOUNDARY")
    print("=" * 70)
    print("L_flat contains non-nested operator sequences accepted by the")
    print("canonical validator with epi_initial=0.0 and current fixed policies.")
    print("Valid sequences must start with a U1a generator {AL, NAV, REMESH}")
    print("and end with a U1b closure {SHA, NAV, REMESH, OZ}; the enumeration")
    print("applies these necessary boundaries before the canonical validator.")
    print()
    _CACHE.clear()
    print(f"  {'n':>3} {'N(n)':>9}")
    for n in range(1, 7):
        v = valid_sequences(n)
        _CACHE[n] = v
        print(f"  {n:>3} {len(v):>9}")
    print()
    print("  The current flat validator state is finite (bounded recency/debt and")
    print("  boundary flags), so this projection admits a DFA/regular-language")
    print("  representation. Full U5 syntax with nested THOL[...] is context-free")
    print("  and is outside this enumeration. Runtime U3 phase compatibility also")
    print("  needs a graph state and is not a symbol-only DFA condition here.")


def experiment_2_finite_counts():
    """M2: report finite log-counts without inferring asymptotic capacity."""
    print()
    print("=" * 70)
    print("M2: FINITE GROWTH RATIOS AND PER-SYMBOL LOG-COUNTS")
    print("=" * 70)
    print(f"  unconstrained upper bound = log2(13) = {math.log2(A):.3f} bits/operator")
    print()
    print(
        f"  {'n':>3} {'N(n)':>9} {'lambda_n':>9} {'log2 lambda':>12} "
        f"{'log2N/n':>11}"
    )
    prev = None
    for n in range(1, 7):
        N = len(_CACHE[n])
        cap_sym = math.log2(N) / n
        if prev is None:
            print(f"  {n:>3} {N:>9} {'--':>9} {'--':>12} {cap_sym:>11.3f}")
        else:
            lam = N / prev
            print(
                f"  {n:>3} {N:>9} {lam:>9.3f} {math.log2(lam):>12.3f} "
                f"{cap_sym:>11.3f}"
            )
        prev = N
    print()
    print("  -> These are exact finite-n values for n <= 6. Their upward trend does")
    print("     not establish a limit, convergence to 13, sub-extensive constraint")
    print("     cost, or the topological entropy of L_flat. Resolving an asymptotic")
    print("     rate requires a proven transition matrix/DFA or longer exact counts.")


def experiment_3_length5_frequencies():
    """M3: measure the glyph distribution at the single length n=5."""
    print()
    print("=" * 70)
    print("M3: LENGTH-5 GLYPH FREQUENCIES")
    print("=" * 70)
    v = _CACHE[5]
    freq = Counter()
    for combo in v:
        for i in combo:
            freq[i] += 1
    total = sum(freq.values())
    uniform = 1.0 / A
    print("  operator frequencies across all valid length-5 sequences:")
    print(f"  {'op':>7} {'fraction':>9} {'vs uniform':>11}")
    for i in sorted(range(A), key=lambda j: -freq[j]):
        frac = freq[i] / total
        print(f"  {NAMES[i]:>7} {frac:>9.4f} {frac / uniform:>10.2f}x")
    print()
    start = Counter(combo[0] for combo in v)
    end = Counter(combo[-1] for combo in v)
    print(
        f"  START set = {{{', '.join(NAMES[i] for i in sorted(start))}}} "
        f"(= U1a generators)"
    )
    print(
        f"  END set   = {{{', '.join(NAMES[i] for i in sorted(end))}}} "
        f"(= U1b closures)"
    )
    print()
    least = min(range(A), key=lambda index: freq[index])
    most = max(range(A), key=lambda index: freq[index])
    print(
        f"  -> At n=5, {NAMES[most]} is most frequent and {NAMES[least]} is "
        "least frequent."
    )
    print("     Boundary roles and local preconditions help interpret this finite")
    print("     distribution, but it is not an asymptotic frequency hierarchy.")


def main():
    print()
    print("  ===============================================================")
    print("  Flat Grammar Projection as a Formal Language")
    print("  Finite Counts, Glyph Frequencies, and the U1 Boundary")
    print("  ===============================================================")
    print()
    experiment_1_regular_language()
    experiment_2_finite_counts()
    experiment_3_length5_frequencies()
    print()
    print("=" * 70)
    print("SCOPED FINDINGS")
    print("=" * 70)
    print("1. L_flat is the finite-state, non-nested projection accepted by the")
    print("   current validator; its U1 start/end boundary is explicit.")
    print("2. Full U5 expressions with arbitrarily nested THOL[...] belong to the")
    print("   context-free glyph language and are not recognized by this flat DFA.")
    print("3. Runtime U3 phase compatibility remains a graph-state precondition;")
    print("   this enumeration measures only the static symbol validator.")
    print("4. N(n), adjacent ratios, and glyph frequencies are exact only for the")
    print("   printed finite lengths. They establish no asymptotic entropy, limit")
    print("   toward the alphabet size, or persistent operator hierarchy.")


if __name__ == "__main__":
    main()
