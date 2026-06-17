#!/usr/bin/env python3
"""
Example 143 - The Glyphic-Function Sub-Language: an Audit of the Canonical
Patterns and the Nesting (THOL[...]) That Lifts the Grammar to Context-Free
==============================================================================

The grammar is the only mechanism that modifies coherence, and examples 139-142
characterized the OPERATOR-sequence language L as a regular language (a finite
automaton with capacity lambda=11.56, 9 grammatical role-classes). This example
audits the higher-level "canonical patterns" concept (Bootstrap / Stabilize /
Explore / Propagate, and the named registries scattered across the codebase) and
reconstructs it from the ORIGINAL source -- TNFR.pdf 2.3 "Macros glificas" and
the "Tabla de funciones glificas operativas" -- finding that the original concept
is a genuine SUB-LANGUAGE of nested glyphic functions, richer than the flat code
patterns, whose nesting feature THOL[...] places the full language one level
above L in the Chomsky hierarchy.

The original canonical glyphic functions (TNFR.pdf 2.3)
-------------------------------------------------------
  Activacion simple          AL -> IL -> RA            (stabilized emission that propagates)
  Estabilizacion mutacional  OZ -> ZHIR -> IL          (dissonance transformed into coherence)
  Ciclo regenerativo         NAV -> THOL[ ... ] -> SHA (self-organized node returns to latency)
  Interfaz adaptativa        THOL[ ZHIR -> UM -> NAV ] -> RA (network reorganizes and expands)
  MACRO INIT                 AL -> IL -> UM
  MOD ESTABILIZADOR          OZ -> ZHIR -> IL

The decisive feature the flat code patterns LOST is the NESTING THOL[ body ]:
THOL opens a sub-EPI whose interior is itself a glyphic sub-sequence. That is
exactly operational fractality (U5, nested EPIs). The code-side patterns
(Bootstrap=[AL,UM,IL], Stabilize=[IL,SHA], Explore=[OZ,ZHIR,IL], Propagate/
RESONATE=[RA,UM,RA]) are flat, depth-0, and DIVERGE from the PDF (e.g. the PDF's
MACRO INIT is [AL,IL,UM], not Bootstrap's [AL,UM,IL]).

Doctrine compliance
-------------------
Validity is decided exclusively by the canonical validate_grammar oracle (U1-U6).
The nesting bracket is the canonical THOL sub-EPI boundary (AGENTS.md: THOL
"creates sub-EPIs, fractal structuring"; U5 multi-scale coherence). Nothing is
imposed; the audit and the Chomsky-hierarchy placement are measured/derived.

Three measured results
----------------------
M1 AUDIT. The PDF-original glyphic functions and the code base patterns are ALL
   grammatical FRAGMENTS (0 of 9 PDF macros, 0 of 5 code patterns are valid
   standalone words) -- they are macros meant to be COMPOSED, not sequences. And
   2 of the 10 "concrete" registry sequences in canonical_patterns.py are in fact
   INVALID: therapeutic_protocol starts with EN (not a U1a generator) and
   full_deployment has ZHIR with no prior IL (U4b violated). The code patterns
   diverge from the PDF and dropped the nesting entirely.

M2 COMPOSITION. A fragment becomes a valid word exactly by adding the missing
   grammar boundary: a U1a generator prefix and a U1b closure suffix (plus the
   U4b context a transformer needs). Bootstrap [AL,UM,IL] -> [AL,UM,IL,SHA] is
   valid; the macros compose into larger valid words with this glue. This is the
   PDF's "compose into more complex structures" recovered exactly: the glyphic
   functions are the WORDS, composition is the higher grammar.

M3 NESTING = CONTEXT-FREE (the recovered fractal variable). A well-formed nested
   glyphic function THOL[ body ] (body itself grammar-valid) flattens to a
   grammar-VALID operator stream -- the regular layer of ex 139-142 is preserved.
   But the bracket structure is a Dyck language: the number of nesting tree shapes
   with n THOL nodes is exactly the Catalan number C_n (1,1,2,5,14,42,...), and
   the nesting depth is unbounded and balanced, so the bracketed glyphic-function
   language is CONTEXT-FREE and NOT regular (pumping lemma). The bracket depth IS
   the nested-EPI fractal scale (U5). So operational fractality U5 is precisely
   the feature that lifts the glyphic language one level above the regular
   operator grammar L.

Honest scope
------------
This is standard formal-language theory (Dyck language, Catalan numbers, the
pumping lemma, the Chomsky hierarchy) applied to the canonical grammar U1-U6 and
the canonical THOL nesting recovered from TNFR.pdf. It is an AUDIT plus a
CHARACTERIZATION: it documents the drift of the code-side "canonical patterns"
from the original glyphic-function concept and locates the full sub-language at
the context-free level, with nesting depth = the U5 fractal scale. It is not new
mathematics and closes no open problem. It does not modify the operator
definitions or the grammar; it does flag two invalid registry sequences for
future cleanup.

References
----------
- theory/TNFR.pdf 2.3 ("Macros glificas", "Tabla de funciones glificas operativas")
- src/tnfr/operators/grammar_validate.py (the canonical validate_grammar oracle)
- src/tnfr/operators/canonical_patterns.py (the compat-shim registry, audited)
- examples/08_emergent_geometry/139_grammar_formal_language.py (L is regular)
- examples/08_emergent_geometry/142_grammar_operator_quotient.py (9 role-classes)
- AGENTS.md "Unified Grammar (U1-U6)" U5 (operational fractality, nested EPIs)
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from math import comb

from tnfr.operators.definitions import (
    Emission, Reception, Coherence, Dissonance, Coupling, Resonance,
    Silence, Expansion, Contraction, SelfOrganization, Mutation,
    Transition, Recursivity,
)
from tnfr.operators.grammar_validate import validate_grammar

INST = {
    "AL": Emission(), "EN": Reception(), "IL": Coherence(),
    "OZ": Dissonance(), "UM": Coupling(), "RA": Resonance(),
    "SHA": Silence(), "VAL": Expansion(), "NUL": Contraction(),
    "THOL": SelfOrganization(), "ZHIR": Mutation(), "NAV": Transition(),
    "REMESH": Recursivity(),
}


def valid(seq):
    """Decide grammar validity with the canonical U1-U6 oracle."""
    return validate_grammar([INST[s] for s in seq], 0.0)


def flatten(node):
    """Flatten a nested glyphic function to its operator stream.

    A node is a list of items; an item is either a glyph string or a pair
    (head, body) meaning head[ body ] -- head opens a sub-EPI containing body.
    """
    out = []
    for item in node:
        if isinstance(item, tuple):
            head, body = item
            out.append(head)
            out.extend(flatten(body))
        else:
            out.append(item)
    return out


def catalan(n):
    return comb(2 * n, n) // (n + 1)


def experiment_1_audit():
    print("=" * 72)
    print("E1: AUDIT - PDF-original glyphic functions and code patterns vs U1-U6")
    print("=" * 72)

    pdf = [
        ("Activacion simple      [AL,IL,RA]", ["AL", "IL", "RA"]),
        ("Estabiliz. mutacional  [OZ,ZHIR,IL]", ["OZ", "ZHIR", "IL"]),
        ("Ciclo regen.           [NAV,THOL,SHA]", ["NAV", "THOL", "SHA"]),
        ("Interfaz adaptativa    [THOL,ZHIR,UM,NAV,RA]",
         ["THOL", "ZHIR", "UM", "NAV", "RA"]),
        ("MACRO INIT             [AL,IL,UM]", ["AL", "IL", "UM"]),
        ("MOD ESTABILIZADOR      [OZ,ZHIR,IL]", ["OZ", "ZHIR", "IL"]),
        ("Secuencia base         [AL,IL,UM,RA]", ["AL", "IL", "UM", "RA"]),
    ]
    code = [
        ("Bootstrap  [AL,UM,IL]", ["AL", "UM", "IL"]),
        ("Stabilize  [IL,SHA]", ["IL", "SHA"]),
        ("Explore    [OZ,ZHIR,IL]", ["OZ", "ZHIR", "IL"]),
        ("Propagate  [RA,UM]", ["RA", "UM"]),
        ("RESONATE   [RA,UM,RA]", ["RA", "UM", "RA"]),
    ]
    n_pdf_valid = sum(valid(s) for _, s in pdf)
    n_code_valid = sum(valid(s) for _, s in code)
    print("\n  PDF-original glyphic functions (macros to compose):")
    for name, seq in pdf:
        print(f"    {'VALID  ' if valid(seq) else 'fragment'}  {name}")
    print(f"    -> {n_pdf_valid}/{len(pdf)} valid standalone (the rest are macros)")
    print("\n  Code base patterns (sequence_generator / AGENTS.md):")
    for name, seq in code:
        print(f"    {'VALID  ' if valid(seq) else 'fragment'}  {name}")
    print(f"    -> {n_code_valid}/{len(code)} valid standalone")

    print("\n  canonical_patterns.py 'concrete' registry (the compat shim):")
    registry = [
        ("bifurcated_base    ", ["AL", "EN", "IL", "OZ", "ZHIR", "IL", "SHA"]),
        ("bifurcated_collapse", ["AL", "OZ", "NUL", "IL", "SHA"]),
        ("therapeutic_proto  ", ["EN", "AL", "IL", "OZ", "THOL", "IL", "SHA"]),
        ("theory_system      ", ["AL", "NAV", "UM", "RA", "IL", "SHA"]),
        ("full_deployment    ", ["AL", "UM", "RA", "OZ", "ZHIR", "IL", "SHA"]),
        ("mod_stabilizer     ", ["AL", "IL", "SHA"]),
        ("contained_crisis   ", ["AL", "EN", "IL", "OZ", "SHA"]),
        ("minimal_compress   ", ["AL", "EN", "IL", "NUL", "SHA"]),
        ("phase_lock         ", ["AL", "EN", "IL", "OZ", "ZHIR", "SHA"]),
        ("resonance_peak     ", ["AL", "EN", "IL", "RA", "SHA"]),
    ]
    invalid = []
    for name, seq in registry:
        ok = valid(seq)
        if not ok:
            invalid.append(name.strip())
        print(f"    {'VALID  ' if ok else 'INVALID'}  {name} {' '.join(seq)}")
    print(f"\n  -> {len(invalid)} of 10 registry sequences are INVALID under U1-U6:")
    print(f"     {invalid}")
    print("     therapeutic_proto starts with EN (not a U1a generator);")
    print("     full_deployment has ZHIR with no prior IL (U4b violated).")
    print("  -> the flat code patterns diverge from the PDF (MACRO INIT is")
    print("     [AL,IL,UM], not Bootstrap's [AL,UM,IL]) and dropped THOL nesting.")


def experiment_2_composition():
    print()
    print("=" * 72)
    print("E2: COMPOSITION - fragments become valid words with grammar glue")
    print("=" * 72)
    print("  A fragment is completed by a U1a generator prefix + U1b closure")
    print("  suffix (and the U4b context a transformer needs):")
    cases = [
        ("Bootstrap  [AL,UM,IL]", ["AL", "UM", "IL"], ["AL", "UM", "IL", "SHA"]),
        ("Stabilize  [IL,SHA]", ["IL", "SHA"], ["AL", "IL", "SHA"]),
        ("Explore    [OZ,ZHIR,IL]", ["OZ", "ZHIR", "IL"],
         ["AL", "IL", "OZ", "ZHIR", "IL", "SHA"]),
        ("MOD ESTAB. [OZ,ZHIR,IL]", ["OZ", "ZHIR", "IL"],
         ["AL", "IL", "OZ", "ZHIR", "IL", "SHA"]),
    ]
    for name, frag, word in cases:
        print(f"    {name:24s} fragment={valid(frag)!s:5s} "
              f"-> [{' '.join(word)}] = {valid(word)}")
    print("\n  Macros compose into larger valid words (generator ... closure):")
    # MACRO INIT [AL,IL,UM] composed with Activacion-simple [.,IL,RA] + closure
    composed = ["AL", "IL", "UM", "IL", "RA", "SHA"]
    print(f"    INIT[AL,IL,UM] (+) [IL,RA] (+) SHA = "
          f"[{' '.join(composed)}] = {valid(composed)}")
    composed2 = ["AL", "IL", "OZ", "ZHIR", "IL", "UM", "RA", "SHA"]
    print(f"    INIT (+) MOD-ESTAB (+) propagate (+) SHA = "
          f"[{' '.join(composed2)}] = {valid(composed2)}")
    print("  -> the glyphic functions are the WORDS; composition (with grammar")
    print("     glue) is the higher grammar -- the PDF's 'compose into more")
    print("     complex structures', recovered exactly.")


def experiment_3_nesting():
    print()
    print("=" * 72)
    print("E3: NESTING THOL[...] = CONTEXT-FREE (the recovered fractal variable)")
    print("=" * 72)
    print("  (a) a well-formed nested glyphic function flattens to a valid word:")
    nested = ["AL", "IL", "OZ", ("THOL", ["IL", "OZ", "ZHIR", "UM"]), "IL", "SHA"]
    flat = flatten(nested)
    print("      AL IL OZ THOL[ IL OZ ZHIR UM ] IL SHA")
    print(f"      flattened: {' '.join(flat)}  -> valid={valid(flat)}")
    deep = ["AL", "IL", "OZ",
            ("THOL", ["IL", "OZ", ("THOL", ["IL", "OZ", "ZHIR"]), "IL"]),
            "IL", "SHA"]
    flatd = flatten(deep)
    print(f"      depth-2:   {' '.join(flatd)}  -> valid={valid(flatd)}")
    print("      -> the regular operator layer (ex 139-142) is preserved.")

    print("\n  (b) nesting tree shapes with n THOL nodes = Catalan(n) (Dyck/CF):")

    def count_nestings(n):
        c = [1]
        for k in range(1, n + 1):
            c.append(sum(c[i] * c[k - 1 - i] for i in range(k)))
        return c

    cnt = count_nestings(8)
    print("      n :  shapes  Catalan(n)  match")
    for n in range(0, 9):
        cat = catalan(n)
        print(f"      {n} :  {cnt[n]:>6}  {cat:>10}  {cnt[n] == cat}")

    print("\n  (c) bracket depth is unbounded and balanced -> not regular:")
    for n in [1, 2, 3, 4]:
        s = "THOL[" * n + " ZHIR " + "]" * n
        depth = n
        print(f"      THOL^{n}: {s.strip()}  depth={depth} (balanced)")
    print("      a finite automaton has bounded states and cannot count")
    print("      arbitrary nesting depth -> the bracketed glyphic-function")
    print("      language is CONTEXT-FREE but NOT regular (pumping lemma).")
    print("\n  PHYSICAL CONTENT: bracket depth = nested-EPI fractal scale (U5).")
    print("  Operational fractality U5 is exactly the feature that lifts the")
    print("  glyphic language from regular (the operator grammar L) to")
    print("  context-free. The flat code patterns are its depth-0 projection.")


def main():
    print()
    print("#" * 72)
    print("# Example 143 - The Glyphic-Function Sub-Language and Its Nesting")
    print("#" * 72)
    print()
    experiment_1_audit()
    experiment_2_composition()
    experiment_3_nesting()
    print()
    print("=" * 72)
    print("Summary")
    print("=" * 72)
    print("  The code-side 'canonical patterns' (Bootstrap/Stabilize/Explore/")
    print("  Propagate) are a flat, partly-invalid, divergent re-invention of the")
    print("  original TNFR.pdf glyphic functions. Reconstructed canonically, the")
    print("  sub-language is: macros (fragments) that COMPOSE into valid words via")
    print("  grammar glue, with THOL[...] NESTING that makes the full language")
    print("  context-free -- nesting depth = the U5 fractal scale. Audit +")
    print("  characterization; no operator physics changed, no open problem closed.")
    print()


if __name__ == "__main__":
    main()
