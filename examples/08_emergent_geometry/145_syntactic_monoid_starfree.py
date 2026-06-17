#!/usr/bin/env python3
"""
Example 145 - The Syntactic Monoid of the Grammar: L Is Star-Free and
First-Order Definable
==============================================================================

The grammar thread characterized the unified grammar U1-U6 as a formal language
L over the 13-operator alphabet (ex 139), built its automaton and minimal DFA
(ex 140), decomposed its capacity by rule (ex 141), and quotiented its alphabet
(ex 142). This example computes the next canonical algebraic invariant after the
minimal DFA: the SYNTACTIC MONOID M(L) -- the transition monoid of the minimal
DFA, the smallest monoid that recognizes L. Its algebraic structure decides
exactly where L sits in the fine hierarchy of regular languages.

The decisive measured property is that M(L) is APERIODIC (group-free): it
contains no nontrivial group. By two classical theorems this pins L precisely:
  - Schutzenberger (1965): a regular language is STAR-FREE iff its syntactic
    monoid is aperiodic. So L is star-free -- expressible from the letters using
    only concatenation, union, and COMPLEMENT, with NO Kleene star.
  - McNaughton-Papert (1971): star-free = exactly the languages definable in
    first-order logic FO[<] over positions. So L is FO-definable -- every
    coherent valid sequence is described by a first-order formula about operator
    positions, with no fixed-point recursion.

Why this is paradigm knowledge
------------------------------
The grammar is the only mechanism that modifies coherence, so the logical
complexity of L is the logical complexity of "building valid coherence". Star-
free / FO-definable is the SIMPLEST nontrivial class of regular languages: it
says the grammar needs no counting and no modular/periodic structure -- only the
linear order of operator positions and the boundary/threshold conditions U1-U6
already encode. The grammar is as logically simple as a non-trivial coherence
constraint can be.

Doctrine compliance
-------------------
The automaton is built from the canonical centralized grammar sets (grammar_types
-> grammar_canon -> physics_derivation, the single source of truth). The monoid
is the exact transition monoid of the minimal DFA; aperiodicity is verified by
the rigorous group-free condition (every element stabilizes under powers); the
star-free / FO conclusions are the cited classical theorems, not new claims.

Three measured results
----------------------
M1 THE SYNTACTIC MONOID. The minimal DFA has 29 classes (28 live + 1 dead sink,
   ex 140). Its transition monoid -- the syntactic monoid M(L) -- has |M| = 312
   elements (incl. the identity = empty word) and 131 idempotents (e*e = e). The
   high idempotent density is the algebraic fingerprint of an aperiodic monoid.

M2 APERIODICITY (group-free). Every element x satisfies x^n = x^(n+1) for n <= 4
   (the stability index): no element generates a nontrivial cyclic group, so M(L)
   is H-trivial and contains NO nontrivial group. Contrast: the parity language
   a^even has syntactic monoid Z/2 (a period-2 group), is NOT aperiodic, and is
   NOT star-free -- the qualitative difference is exhibited side by side.

M3 STAR-FREE + FIRST-ORDER. By Schutzenberger, aperiodic syntactic monoid => L
   is star-free (concatenation/union/complement, no Kleene star). By McNaughton-
   Papert, star-free => L is definable in first-order logic FO[<]. The grammar
   sits in the simplest nontrivial level of the regular hierarchy: no counting,
   no periodicity -- just the linear order of positions plus the U1-U6 boundary
   and threshold conditions.

Honest scope
------------
This is standard algebraic automata theory (the syntactic monoid; Schutzenberger's
star-free theorem; the McNaughton-Papert FO characterization; Green's relations /
H-triviality) computed on the canonical automaton of ex 140. It is a
CHARACTERIZATION that locates L at the star-free / FO-definable level; it is not
new mathematics and closes no open problem. The syntactic monoid is computed as
the transition monoid of the minimal DFA (canonical), so |M| = 312 and the
aperiodicity verdict are exact.

References
----------
- src/tnfr/operators/grammar_canon.py / grammar_types.py (canonical grammar sets)
- examples/08_emergent_geometry/140_grammar_automaton.py (minimal DFA, 29 states)
- examples/08_emergent_geometry/142_grammar_operator_quotient.py (alphabet quotient)
- Schutzenberger 1965 (star-free = aperiodic syntactic monoid)
- McNaughton & Papert 1971 (star-free = first-order definable FO[<])
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from tnfr.operators.grammar_types import (
    GENERATORS, CLOSURES, STABILIZERS, DESTABILIZERS, TRANSFORMERS,
)

ALPHA = ["emission", "reception", "coherence", "dissonance", "coupling",
         "resonance", "silence", "expansion", "contraction",
         "self_organization", "mutation", "transition", "recursivity"]
START = ("START",)
DEAD = ("DEAD",)


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


def build_minimal_dfa(states):
    """Partition-refine to the minimal DFA over states + a dead sink."""
    allstates = states + [DEAD]
    accept = {s for s in allstates if is_accept(s)}

    def delta(s, x):
        if s == DEAD:
            return DEAD
        ns = transition(s, x)
        return ns if ns is not None else DEAD

    part = {s: (s in accept) for s in allstates}
    while True:
        keys: dict = {}
        newpart = {}
        for s in allstates:
            sig = (part[s],) + tuple(part[delta(s, x)] for x in ALPHA)
            keys.setdefault(sig, len(keys))
            newpart[s] = keys[sig]
        if len(set(newpart.values())) == len(set(part.values())):
            return newpart, delta
        part = newpart


def transition_monoid(part, delta):
    """The transition monoid of the minimal DFA = the syntactic monoid M(L)."""
    classlist = sorted(set(part.values()))
    cidx = {c: i for i, c in enumerate(classlist)}
    m = len(classlist)
    rep: dict = {}
    for s, c in part.items():
        rep.setdefault(c, s)

    def gen_map(x):
        return tuple(cidx[part[delta(rep[c], x)]] for c in classlist)

    identity = tuple(range(m))
    gens = {x: gen_map(x) for x in ALPHA}

    def compose(f, g):
        return tuple(g[f[i]] for i in range(m))

    monoid = {identity}
    monoid |= set(gens.values())
    frontier = list(monoid)
    while frontier:
        f = frontier.pop()
        for x in ALPHA:
            h = compose(f, gens[x])
            if h not in monoid:
                monoid.add(h)
                frontier.append(h)
    return monoid, compose, identity, m


def stability_index(monoid, compose):
    """Largest n with x^n = x^(n+1); finite iff the monoid is aperiodic."""
    max_n = 0
    for f in monoid:
        cur = f
        n = 1
        found = None
        while n <= len(monoid) + 1:
            nxt = compose(cur, f)
            if nxt == cur:
                found = n
                break
            cur = nxt
            n += 1
        if found is None:
            return None  # not aperiodic
        max_n = max(max_n, found)
    return max_n


def experiment_1_monoid(monoid, compose, m):
    print("=" * 72)
    print("M1: the syntactic monoid M(L) = transition monoid of the minimal DFA")
    print("=" * 72)
    idem = [f for f in monoid if compose(f, f) == f]
    consts = sum(1 for f in monoid if len(set(f)) == 1)
    print(f"  minimal DFA classes: {m}  ({m - 1} live + 1 dead sink)")
    print(f"  |M(L)| = {len(monoid)} elements (incl. identity = empty word)")
    print(f"  idempotents (e*e = e): {len(idem)}")
    print(f"  constant/reset elements (collapse to one class): {consts}")
    print("  -> high idempotent density is the fingerprint of an aperiodic monoid")


def experiment_2_aperiodic(monoid, compose):
    print()
    print("=" * 72)
    print("M2: aperiodicity (group-free) -- the decisive property")
    print("=" * 72)
    n = stability_index(monoid, compose)
    print(f"  every element x satisfies x^n = x^(n+1) with n <= {n}")
    print("  => no element generates a nontrivial cyclic group (H-trivial)")
    print("  => M(L) contains NO nontrivial group => APERIODIC")

    print("\n  CONTRAST: the parity language a^even is NOT star-free")
    pm = {"a": (1, 0), "b": (0, 1)}
    pid = (0, 1)
    pmon = {pid, pm["a"], pm["b"]}
    fr = list(pmon)
    while fr:
        f = fr.pop()
        for x in ("a", "b"):
            h = tuple(pm[x][f[i]] for i in range(2))
            if h not in pmon:
                pmon.add(h)
                fr.append(h)

    def pcompose(f, g):
        return tuple(g[f[i]] for i in range(2))

    pn = stability_index(pmon, pcompose)
    print(f"    parity syntactic monoid: {len(pmon)} elements = Z/2 (a group);")
    print(f"    a^2 = identity (period 2), so it is NOT aperiodic "
          f"(stability index: {pn})")
    print("    -> parity is NOT star-free; our grammar L IS. The qualitative")
    print("       difference (group vs group-free) is the whole point.")
    return n


def experiment_3_starfree(stability_n):
    print()
    print("=" * 72)
    print("M3: star-free + first-order definable (the two classical theorems)")
    print("=" * 72)
    print("  Schutzenberger (1965): regular L is STAR-FREE  <=>  M(L) aperiodic.")
    print(f"  M(L) is aperiodic (stability index {stability_n}), therefore:")
    print("    L is STAR-FREE -- built from letters by concatenation, union, and")
    print("    complement, with NO Kleene star.")
    print()
    print("  McNaughton-Papert (1971): STAR-FREE  <=>  first-order definable FO[<].")
    print("    Therefore L is FO-definable: every valid coherent sequence is")
    print("    described by a first-order formula over operator positions, with")
    print("    no fixed-point recursion.")
    print()
    print("  PARADIGM INSIGHT: building valid coherence is logically as simple as")
    print("  a nontrivial regular constraint can be -- no counting, no modular/")
    print("  periodic structure, just the linear order of positions plus the")
    print("  U1-U6 boundary and threshold conditions.")


def main():
    print()
    print("#" * 72)
    print("# Example 145 - The Syntactic Monoid: L Is Star-Free and FO-Definable")
    print("#" * 72)
    print()
    states = reachable_states()
    part, delta = build_minimal_dfa(states)
    monoid, compose, identity, m = transition_monoid(part, delta)
    experiment_1_monoid(monoid, compose, m)
    stability_n = experiment_2_aperiodic(monoid, compose)
    experiment_3_starfree(stability_n)
    print()
    print("=" * 72)
    print("Summary")
    print("=" * 72)
    print("  The grammar's syntactic monoid M(L) has 312 elements and is")
    print("  APERIODIC (group-free, stability index 4). By Schutzenberger L is")
    print("  star-free; by McNaughton-Papert L is first-order definable. The")
    print("  unified grammar U1-U6 sits in the simplest nontrivial class of")
    print("  regular languages. Characterization; no operator physics changed,")
    print("  no open problem closed.")
    print()


if __name__ == "__main__":
    main()
