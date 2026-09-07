#!/usr/bin/env python3
"""Example 143 -- Flat fragments and explicit U5 nesting syntax.

The operator-history validator consumes a flat sequence. A nested glyphic form
such as ``THOL[body]`` contains extra syntax that flattening destroys. This
example keeps those layers separate:

* flat fragments and registry words are checked by ``validate_grammar``;
* balanced bracket structure is checked with a stack; and
* parent/child U5 coherence is left to hierarchy-aware runtime telemetry.

The bracket-only skeleton constructed here is a Dyck language and therefore
deterministic context-free and non-regular. That classification applies to the
syntactic skeleton, not automatically to every numerical U5 acceptance policy.
U3 and U6 likewise remain runtime/reference checks outside the flat validator.
"""

import os
import sys
from math import comb

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from tnfr.operators.canonical_patterns import CANONICAL_SEQUENCES
from tnfr.operators.grammar_types import GLYPH_TO_FUNCTION
from tnfr.operators.grammar_validate import validate_grammar

from _flat_grammar_model import NAME2INST

GLYPH_TO_NAME = {glyph.value: name for glyph, name in GLYPH_TO_FUNCTION.items()}


def valid(seq):
    """Check the default-depth flat history represented by glyph strings."""

    return validate_grammar([NAME2INST[GLYPH_TO_NAME[glyph]] for glyph in seq], 0.0)


def flatten(node):
    """Project a nested list to a flat operator stream, discarding brackets."""

    out = []
    for item in node:
        if isinstance(item, tuple):
            head, body = item
            out.append(head)
            out.extend(flatten(body))
        else:
            out.append(item)
    return out


def bracket_tokens(node):
    """Encode explicit nesting without discarding open/close boundaries."""

    out = []
    for item in node:
        if isinstance(item, tuple):
            head, body = item
            out.extend((head, "["))
            out.extend(bracket_tokens(body))
            out.append("]")
        else:
            out.append(item)
    return out


def balanced(tokens):
    """Recognize the bracket skeleton with an explicit stack counter."""

    depth = 0
    for token in tokens:
        if token == "[":
            depth += 1
        elif token == "]":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


def catalan(n):
    """Number of balanced one-kind bracket strings with ``n`` pairs."""

    return comb(2 * n, n) // (n + 1)


def experiment_1_fragments_and_registry():
    print("=" * 72)
    print("M1: FRAGMENTS ARE DISTINCT FROM STANDALONE FLAT WORDS")
    print("=" * 72)
    fragments = [
        ("Bootstrap", ["AL", "UM", "IL"]),
        ("Stabilize", ["IL", "SHA"]),
        ("Explore", ["OZ", "ZHIR", "IL"]),
        ("Propagate", ["RA", "UM"]),
    ]
    for name, word in fragments:
        print(f"  {name:10s} {' '.join(word):20s} accepted standalone={valid(word)}")
    print("\n  The fragments omit some start, closure, or transformer context by")
    print("  design; they become candidates for a word only after composition.")

    accepted = 0
    print("\n  canonical_sequences registry (read dynamically):")
    for name, specification in CANONICAL_SEQUENCES.items():
        glyphs = [glyph.value for glyph in specification.glyphs]
        ok = valid(glyphs)
        accepted += int(ok)
        print(f"    {name:22s} flat-history accepted={ok}")
    print(f"  registry result: {accepted}/{len(CANONICAL_SEQUENCES)} accepted")
    assert accepted == len(CANONICAL_SEQUENCES)
    print("  This says nothing yet about U3 phase gates or U6 field drift.")


def experiment_2_composition():
    print("\n" + "=" * 72)
    print("M2: COMPOSITION SUPPLIES THE MISSING FLAT-HISTORY CONTEXT")
    print("=" * 72)
    examples = [
        (["AL", "UM", "IL"], ["AL", "UM", "IL", "SHA"]),
        (["IL", "SHA"], ["AL", "IL", "SHA"]),
        (
            ["OZ", "ZHIR", "IL"],
            ["AL", "IL", "OZ", "ZHIR", "IL", "SHA"],
        ),
    ]
    for fragment, composed in examples:
        print(
            f"  {' '.join(fragment):20s} {valid(fragment)!s:5s} -> "
            f"{' '.join(composed):36s} {valid(composed)}"
        )
    print("\n  Acceptance here is the flat symbolic/history result only.")


def experiment_3_nesting():
    print("\n" + "=" * 72)
    print("M3: U5 BRACKETS REQUIRE STACK-LIKE SYNTAX")
    print("=" * 72)
    nested = [
        "AL",
        "IL",
        "OZ",
        ("THOL", ["IL", "OZ", ("THOL", ["IL"]), "IL"]),
        "SHA",
    ]
    tokens = bracket_tokens(nested)
    projected = flatten(nested)
    print(f"  explicit tokens: {' '.join(tokens)}")
    print(f"  balanced brackets: {balanced(tokens)}")
    print(f"  flat projection: {' '.join(projected)}")
    print(f"  flat-history accepted: {valid(projected)}")
    print("\n  Flattening can be accepted while still losing the parent/child")
    print("  boundaries needed to evaluate U5 coherence at every level.")

    print("\n  balanced one-kind bracket skeletons:")
    print(f"  {'pairs':>5} {'Catalan count':>14}")
    for pairs in range(0, 9):
        print(f"  {pairs:>5} {catalan(pairs):>14}")
    print("\n  The constructed Dyck skeleton is context-free and non-regular.")
    print("  Full U5 acceptance also needs hierarchy data and a specified alpha/")
    print("  normalization, so this syntactic result is only one necessary layer.")


def main():
    print("\n" + "#" * 72)
    print("# Example 143 - Flat Fragments and Explicit U5 Nesting")
    print("#" * 72 + "\n")
    experiment_1_fragments_and_registry()
    experiment_2_composition()
    experiment_3_nesting()
    print("\n  The flat DFA, nested syntax, and runtime telemetry are separate")
    print("  interfaces. Combining them requires retaining all three layers.")


if __name__ == "__main__":
    main()
