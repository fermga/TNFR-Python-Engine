#!/usr/bin/env python3
"""
Example 144 - The Branching Combinator: the OZ Bifurcation as Alternation, the
Union Operation of the Glyphic Regular-Expression Algebra
==============================================================================

Example 143 recovered the glyphic-function sub-language from TNFR.pdf 2.3 and
found that its NESTING THOL[...] lifts the language to context-free. This example
recovers the OTHER feature the flat code patterns dropped: BRANCHING [ZHIR|NUL].
TNFR.pdf 2.3 "Bifurcacion y mutacion": a sequence can include BIFURCATION POINTS,
generally triggered by OZ (dissonance), giving possible ramifications
OZ -> [ZHIR / NUL]; the mutation can be directed (ZHIR) or lead to collapse /
withdrawal (NUL). The PDF is explicit and decisive for doctrine:

  "Estas opciones NO son alternativas simbolicas, sino trayectorias
   estructurales reales en el campo."

So the branch is a REAL physical bifurcation (U4a: d2EPI/dt2 > tau triggers it),
not mere syntax: OZ generates a bifurcation threshold and the node either
reorganizes (ZHIR) or collapses to latency (NUL).

The grammar/formal content
--------------------------
Branching is the ALTERNATION combinator [A|B]. A branched program X[A|B]Y denotes
the set XAY union XBY -- a UNION of words, a REGULAR operation. Regular languages
are closed under union, so branching alone does NOT raise the Chomsky class:
unlike nesting (ex 143, context-free), branching stays REGULAR. It is a compact
notation for a set of words already in the regular operator language L (ex 139).

The capstone is the canonical "Tabla comparativa de estructuras glificas"
(TNFR.pdf 2.3), a typology of five glyphic structure shapes:
  Lineal     AL -> IL -> RA -> SHA              (simple trajectory)        = concatenation
  Bifurcada  OZ -> [ZHIR | NUL]                 (double trajectory)        = UNION  <- this ex
  Fractal    NAV -> IL -> UM -> NAV             (self-similar cycle)       = Kleene star
  Ciclica    THOL[...] -> NAV -> THOL[...]      (nodal feedback)           = star of nests
  Jerarquica THOL[ AL -> ZHIR -> IL ]           (encapsulated process)     = nesting (Dyck)
The three NON-nesting operations -- concatenation, union, Kleene star -- are
EXACTLY the three regular-expression operations (Kleene's theorem): they generate
precisely the regular languages. Only nesting (Jerarquica) escapes to
context-free (ex 143). So the glyphic structural typology IS a regular-expression
algebra plus nesting, and branching is its UNION operation.

Doctrine compliance
-------------------
Validity is decided by the canonical validate_grammar oracle (U1-U6). The branch
sits at OZ, the canonical bifurcation trigger (U4a). The two-channel measurement
applies the canonical operators (Dissonance, Coherence, Mutation, Contraction) in
their grammar-correct order and reads the canonical node state -- nothing is
imposed.

Three measured results
----------------------
M1 THE BRANCH IS A REAL BIFURCATION INTO TWO ORTHOGONAL BASINS. Both branches
   OZ->ZHIR and OZ->NUL are grammar-valid (10 valid continuations after [AL,IL,
   OZ]; ZHIR and NUL are two of them). From the SAME post-[IL,OZ] state the two
   branches move ORTHOGONAL channels: ZHIR reorganizes the PHASE (d-theta=0.236,
   d|EPI|=0.000), NUL contracts the STRUCTURE (d-theta=0.000, d|EPI|=0.058). This
   is the PDF's "trayectorias estructurales reales": reorganize vs collapse, two
   physically distinct basins of the OZ-triggered bifurcation.

M2 ALTERNATION IS A REGULAR OPERATION. X[A|B]Y = XAY union XBY; regular languages
   are closed under union, so branching does NOT raise the Chomsky class
   (contrast nesting THOL[...] = context-free, ex 143). A branched program with k
   binary choice points denotes exactly 2^k concrete words; expanding and
   validating them confirms all 2^k are grammar-valid words already in L. Branching
   is exponential COMPRESSION -- a compact name for a set of already-valid words.

M3 THE GLYPHIC TYPOLOGY IS A REGULAR-EXPRESSION ALGEBRA PLUS NESTING. The PDF
   "Tabla comparativa de estructuras glificas" has five types; their combinators
   are concatenation (Lineal), union (Bifurcada <- this ex), Kleene star (Fractal/
   Ciclica), and nesting (Jerarquica). The three non-nesting operations are
   exactly the regular-expression operations (Kleene's theorem) and generate the
   regular languages; only nesting escapes to context-free. Branching is the UNION
   operation of this algebra.

Honest scope
------------
This is standard formal-language theory (closure of regular languages under
union; Kleene's theorem that regular = {concatenation, union, star}; the Chomsky
hierarchy) applied to the canonical grammar U1-U6 and the canonical OZ/U4a
bifurcation recovered from TNFR.pdf. The two-channel table is a robust
measurement of the canonical operator contracts (ZHIR = phase, NUL = structure),
honoring the PDF's "trayectorias reales". It is an audit + characterization; it
does not modify the operators or the grammar, and it closes no open problem. It
completes the glyphic combinator set begun in ex 143 (nesting): sequence, branch,
cycle (regular) + nest (context-free).

References
----------
- theory/TNFR.pdf 2.3 ("Bifurcacion y mutacion", "Estructuras bifurcadas",
  "Tabla comparativa de estructuras glificas")
- src/tnfr/operators/grammar_validate.py (the canonical validate_grammar oracle)
- src/tnfr/operators/definitions.py (Dissonance/Coherence/Mutation/Contraction)
- examples/08_emergent_geometry/143_glyphic_function_sublanguage.py (nesting = CF)
- examples/08_emergent_geometry/139_grammar_formal_language.py (L is regular)
- AGENTS.md "Unified Grammar (U1-U6)" U4a (OZ bifurcation trigger), operator cards
  (ZHIR phase transformation, NUL structural contraction)
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import copy
import itertools
import math
import warnings

import networkx as nx

from tnfr.constants import inject_defaults
from tnfr.operators.definitions import (
    Coherence,
    Contraction,
    Coupling,
    Dissonance,
    Emission,
    Mutation,
    Resonance,
    SelfOrganization,
    Silence,
    Transition,
)
from tnfr.operators.grammar_validate import validate_grammar

INST = {
    "AL": Emission(),
    "IL": Coherence(),
    "OZ": Dissonance(),
    "ZHIR": Mutation(),
    "NUL": Contraction(),
    "THOL": SelfOrganization(),
    "RA": Resonance(),
    "SHA": Silence(),
    "NAV": Transition(),
    "UM": Coupling(),
}


def valid(seq):
    """Decide grammar validity with the canonical U1-U6 oracle."""
    return validate_grammar([INST[s] for s in seq], 0.0)


def _epi_mag(v):
    """Robust scalar magnitude of an EPI value (scalar or structured)."""
    if isinstance(v, dict):
        c = v.get("continuous")
        return abs(c[0]) if c else 0.0
    return abs(v)


def _apply(G, op, nd=0):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        op(G, nd)


def _build_node(seed=7):
    """Canonical small network with a populated node state (nothing imposed)."""
    G = nx.erdos_renyi_graph(12, 0.3, seed=seed)
    if not nx.is_connected(G):
        comps = list(nx.connected_components(G))
        for i in range(1, len(comps)):
            G.add_edge(next(iter(comps[i - 1])), next(iter(comps[i])))
    inject_defaults(G)
    import numpy as np

    rng = np.random.default_rng(seed)
    for nd in G.nodes():
        G.nodes[nd]["EPI"] = rng.uniform(0.4, 0.7)
        G.nodes[nd]["theta"] = rng.uniform(0, 2 * math.pi)
        G.nodes[nd]["nu_f"] = rng.uniform(0.8, 1.2)
    return G


def experiment_1_real_bifurcation():
    print("=" * 72)
    print("M1: the branch is a real bifurcation into two ORTHOGONAL basins")
    print("=" * 72)
    pre = ["AL", "IL", "OZ"]
    branch_z = pre + ["ZHIR", "IL", "SHA"]
    branch_n = pre + ["NUL", "IL", "SHA"]
    print("  PDF: OZ -> [ZHIR | NUL]  (bifurcation triggered by OZ, U4a)")
    print(f"  ZHIR branch {branch_z}: valid={valid(branch_z)}")
    print(f"  NUL  branch {branch_n}: valid={valid(branch_n)}")
    conts = [op for op in INST if valid(pre + [op, "IL", "SHA"])]
    print(f"  valid continuations after [AL,IL,OZ]: {conts}")
    print(f"  -> {len(conts)} valid; ZHIR and NUL are two physically meaningful ones")

    # two-channel table from the SAME post-[IL,OZ] state
    gz = _build_node()
    _apply(gz, Coherence())  # prior IL = stable base (U4b)
    _apply(gz, Dissonance())  # recent OZ = elevated dNFR (bifurcation trigger)
    gn = copy.deepcopy(gz)
    th0 = gz.nodes[0]["theta"]
    ep0 = _epi_mag(gz.nodes[0]["EPI"])
    _apply(gz, Mutation())  # ZHIR branch
    _apply(gn, Contraction())  # NUL branch
    dz_th = abs(gz.nodes[0]["theta"] - th0)
    dz_ep = abs(_epi_mag(gz.nodes[0]["EPI"]) - ep0)
    dn_th = abs(gn.nodes[0]["theta"] - th0)
    dn_ep = abs(_epi_mag(gn.nodes[0]["EPI"]) - ep0)
    print()
    print(f"  from the SAME post-[IL,OZ] state (theta={th0:.4f}, |EPI|={ep0:.4f}):")
    print("    branch   d(theta)   d|EPI|    channel moved")
    print(
        f"    ZHIR     {dz_th:.4f}     {dz_ep:.4f}    PHASE (reorganize, theta->theta')"
    )
    print(
        f"    NUL      {dn_th:.4f}     {dn_ep:.4f}    STRUCTURE (collapse, dim EPI down)"
    )
    print("  -> the branches move ORTHOGONAL channels: a real bifurcation into")
    print("     two distinct basins (PDF: NOT symbolic alternatives but real")
    print("     structural trajectories).")


def experiment_2_alternation_regular():
    print()
    print("=" * 72)
    print("M2: alternation [A|B] is a REGULAR operation (branching stays regular)")
    print("=" * 72)
    print("  X[A|B]Y = XAY union XBY; regular languages are closed under union,")
    print("  so branching does NOT raise the Chomsky class (contrast nesting")
    print("  THOL[...] = context-free, ex 143). Branching = compact notation for")
    print("  a set of words already in the regular operator language L (ex 139).")

    # a branched program with k binary choice points -> 2^k concrete words
    print("\n  exponential compression: k binary choices -> 2^k concrete words")
    # program: AL IL [OZ ZHIR | OZ NUL] IL [RA | NUL] [SHA | NAV-closure...]
    # use canonical binary bifurcations, each pair grammar-interchangeable here.
    choice_sets = [
        ("c1", [["OZ", "ZHIR"], ["OZ", "NUL"]]),
        ("c2", [["IL"], ["IL", "IL"]]),
        ("c3", [["SHA"], ["OZ"]]),  # both are valid U1b closures
    ]
    for k in (1, 2, 3):
        prefix = ["AL", "IL"]
        sets = choice_sets[:k]
        words = []
        for combo in itertools.product(*[opts for _, opts in sets]):
            w = list(prefix)
            for part in combo:
                w += part
            # ensure a closure end for k<3 cases
            if w[-1] not in ("SHA", "OZ", "NAV", "REMESH"):
                w += ["SHA"]
            words.append(tuple(w))
        n_expanded = len(set(words))
        n_valid = sum(valid(list(w)) for w in set(words))
        print(
            f"    k={k} binary choices: 2^{k}={2**k} expansions, "
            f"{n_expanded} distinct, {n_valid} grammar-valid"
        )
    print("  -> a single branched glyphic program is a compact generator of an")
    print("     exponential set of concrete words, all already valid in L.")


def experiment_3_regexp_algebra():
    print()
    print("=" * 72)
    print("M3: the glyphic structural typology = regular-expression algebra + nest")
    print("=" * 72)
    typology = [
        ("Lineal    ", ["AL", "IL", "RA", "SHA"], "concatenation", "regular"),
        ("Bifurcada ", None, "union (alternation)  <- this example", "regular"),
        ("Fractal   ", ["NAV", "IL", "UM", "NAV"], "Kleene star (repeat)", "regular"),
        ("Ciclica   ", None, "star of nests", "context-free"),
        ("Jerarquica", None, "nesting (Dyck)  (ex 143)", "context-free"),
    ]
    print("  TNFR.pdf 'Tabla comparativa de estructuras glificas' (5 types):")
    print(f"    {'type':11s} {'combinator':36s} {'class'}")
    for name, seq, comb, cls in typology:
        print(f"    {name:11s} {comb:36s} {cls}")
    # verify the two concrete regular examples are valid words
    lineal = ["AL", "IL", "RA", "SHA"]
    fractal = ["NAV", "IL", "UM", "NAV"]
    print()
    print(f"  Lineal  {lineal}  valid={valid(lineal)}")
    print(
        f"  Fractal {fractal}  valid={valid(fractal)} "
        f"(NAV is both generator U1a and closure U1b)"
    )
    print()
    print("  Kleene's theorem: the regular languages are generated by exactly")
    print("  three operations -- concatenation, union, Kleene star. The glyphic")
    print("  typology's non-nesting types (Lineal=concat, Bifurcada=union,")
    print("  Fractal/Ciclica-repeat=star) ARE those three operations; only")
    print("  nesting (Jerarquica, THOL[...]) escapes to context-free (ex 143).")
    print("  -> branching is the UNION operation of the glyphic regexp algebra.")


def main():
    print()
    print("#" * 72)
    print("# Example 144 - The Branching Combinator: OZ Bifurcation as Alternation")
    print("#" * 72)
    print()
    experiment_1_real_bifurcation()
    experiment_2_alternation_regular()
    experiment_3_regexp_algebra()
    print()
    print("=" * 72)
    print("Summary")
    print("=" * 72)
    print("  Branching [ZHIR|NUL] is the OZ bifurcation made syntactic: a real")
    print("  bifurcation into two orthogonal basins (ZHIR=phase, NUL=structure),")
    print("  written as the ALTERNATION combinator. Alternation is a regular")
    print("  operation (union), so branching keeps the language regular -- only")
    print("  nesting (ex 143) is context-free. The glyphic structural typology is")
    print("  a regular-expression algebra (concat/union/star) + nesting, and")
    print("  branching is its union operation. Audit + characterization; no")
    print("  operator physics changed, no open problem closed.")
    print()


if __name__ == "__main__":
    main()
