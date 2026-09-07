#!/usr/bin/env python3
"""Example 144 -- Branch notation as finite union of flat words.

For a fixed number of alternatives, ``X[A|B]Y`` denotes the finite union
``{XAY, XBY}``. Union preserves regularity of the flat language from example
140. This syntactic fact does not prove that a Dissonance execution crossed a
physical bifurcation threshold, nor that the alternatives are dynamical basins;
those claims require trajectory telemetry.

Unbounded U5 nesting remains stack-like/context-free. Runtime U3 phase gates and
reference-dependent U6 potential drift are also outside this expansion check.
"""

import itertools
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from tnfr.operators.grammar_validate import validate_grammar

from _flat_grammar_model import NAME2INST, SHORT


def valid(names):
    """Validate one expanded default-depth flat operator history."""

    return validate_grammar([NAME2INST[name] for name in names], 0.0)


def expand(prefix, choices):
    """Expand a finite sequence of alternative blocks."""

    words = []
    for selection in itertools.product(*choices):
        word = list(prefix)
        for block in selection:
            word.extend(block)
        words.append(tuple(word))
    return words


def glyphs(word):
    return " ".join(SHORT[name] for name in word)


def experiment_1_oz_alternatives():
    print("=" * 72)
    print("M1: OZ [ZHIR | NUL] EXPANDS TO TWO FLAT HISTORIES")
    print("=" * 72)
    prefix = ["emission", "coherence", "dissonance"]
    suffix = ["coherence", "silence"]
    words = [
        tuple(prefix + ["mutation"] + suffix),
        tuple(prefix + ["contraction"] + suffix),
    ]
    for word in words:
        print(f"  {glyphs(word):36s} flat-history accepted={valid(word)}")
    assert all(valid(word) for word in words)
    print("\n  ZHIR and NUL retain different operator contracts. Acceptance of both")
    print("  symbol histories does not establish an observed physical branch.")


def experiment_2_finite_union():
    print("\n" + "=" * 72)
    print("M2: k BINARY CHOICES EXPAND TO A FINITE UNION")
    print("=" * 72)
    choices = [
        (("dissonance", "mutation"), ("dissonance", "contraction")),
        (("reception",), ("contraction",)),
        (("silence",), ("transition",)),
    ]
    for count in range(1, len(choices) + 1):
        selected = choices[:count]
        # Add a closure while the closure choice is not yet present.
        if count < len(choices):
            selected = [*selected, (("silence",),)]
        words = expand(("emission", "coherence"), selected)
        distinct = set(words)
        accepted = sum(valid(word) for word in distinct)
        print(
            f"  k={count}: {len(distinct)} distinct expansions; "
            f"{accepted} flat-history accepted"
        )
        assert accepted == len(distinct)
    print("\n  A finite union of regular flat-word sets is regular. This result is")
    print("  about expanded notation, independently of runtime graph behavior.")


def experiment_3_language_boundary():
    print("\n" + "=" * 72)
    print("M3: BRANCHING AND NESTING HAVE DIFFERENT MEMORY REQUIREMENTS")
    print("=" * 72)
    print("  fixed finite alternatives : expansion + finite-state validation")
    print("  unbounded THOL[...] depth : balanced stack/tree syntax (U5)")
    print("  UM/RA execution           : runtime phase compatibility (U3)")
    print("  Phi_s confinement         : before/after reference telemetry (U6)")
    print("\n  Only the first row is characterized by this finite-union example.")


def main():
    print("\n" + "#" * 72)
    print("# Example 144 - Branch Notation as Finite Union")
    print("#" * 72 + "\n")
    experiment_1_oz_alternatives()
    experiment_2_finite_union()
    experiment_3_language_boundary()


if __name__ == "__main__":
    main()
