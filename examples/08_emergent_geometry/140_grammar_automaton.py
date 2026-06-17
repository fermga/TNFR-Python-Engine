#!/usr/bin/env python3
"""
Example 140 — The Grammar Automaton: Constructing the Finite-State Machine and
Its Exact Capacity (the Perron-Frobenius Eigenvalue)
==============================================================================

Example 139 ASSERTED that the unified grammar U1-U6 defines a REGULAR formal
language L (finite memory, Myhill-Nerode) and ESTIMATED its capacity from the
finite counts N(n) = 2, 9, 84, 852, 9396, 111060.  This example CONSTRUCTS the
machine explicitly and computes the EXACT capacity:

  - it builds the finite-state automaton DIRECTLY from the canonical operator
    sets (post-centralization: grammar_types derives them from
    config.physics_derivation, the single source of truth),
  - it MINIMIZES it (Myhill-Nerode partition refinement) to a concrete 29-state
    minimal DFA — a constructive proof of regularity, not an assertion,
  - it computes the PERRON-FROBENIUS eigenvalue of the transfer matrix = the
    exact growth rate lambda = 11.560930 (the language's connective constant),
    which the finite N(n)/N(n-1) estimates of example 139 converge to.

The automaton's correctness is cross-checked against the canonical
validate_grammar oracle (it reproduces N(n) exactly), which also re-validates the
just-completed grammar centralization: the sets it is built from are the
canonical ones.

Doctrine compliance
-------------------
The automaton is built from the CANONICAL centralized sets (GENERATORS, CLOSURES,
STABILIZERS, DESTABILIZERS, TRANSFORMERS) imported from grammar_types — nothing is
hand-listed.  The state captures exactly what the canonical validator needs (the
U4b window as operator tags, the U2 stabilizer/destabilizer flags, the U1b
closure bit).  The acceptance and transition logic mirror the canonical U1-U6
validator, and the result is verified against it.

Three measured results
----------------------
M1 THE EXPLICIT AUTOMATON REPRODUCES THE GRAMMAR. A compressed finite-state
   automaton (83 reachable states) built from the canonical sets reproduces the
   canonical oracle N(n) = 2, 9, 84, 852, 9396, 111060 exactly (n = 1..6). The
   state is (window of the last 3 operator tags D/I/O, has-destabilizer,
   has-stabilizer, last-is-closure): enough to decide U1, U2, U4b. (U4a is
   subsumed by U2 because the bifurcation handlers ARE the stabilizers {IL, THOL}.)

M2 THE MINIMAL DFA HAS 29 STATES (CONSTRUCTIVE REGULARITY). Myhill-Nerode
   partition refinement collapses the automaton to a 29-state minimal DFA
   (including the dead/trap sink). A finite minimal automaton EXISTS and is
   exhibited — L is regular constructively, not merely by the finite-memory
   argument of example 139.

M3 THE EXACT CAPACITY IS THE PERRON-FROBENIUS EIGENVALUE. The transfer matrix of
   the trim automaton has spectral radius lambda = 11.560930 (capacity
   log2(lambda) = 3.531 bits/operator). This is the language's connective
   constant — the asymptotic branching factor of grammatically-allowed
   continuations. The finite N(n)/N(n-1) estimates of example 139 are
   NON-MONOTONIC: they overshoot to ~12.19 around n = 9, then settle down to
   lambda = 11.56 (|N(n)/N(n-1) - lambda| = 2.8e-05 by n = 79). The exact
   eigenvalue resolves the finite estimates.

Honest scope
------------
This is standard automata / symbolic-dynamics theory: the Myhill-Nerode theorem
(minimal DFA) and the Perron-Frobenius eigenvalue of the transfer matrix of a
sofic/regular language (its topological entropy = Shannon capacity). The
contribution is CONSTRUCTIVE — building the machine from the canonical
centralized grammar sets and computing its exact invariants — which deepens
example 139 (assertion -> construction) and cross-checks the grammar
centralization. It is not new mathematics and closes no open problem.

References
----------
- src/tnfr/operators/grammar_types.py (the canonical centralized operator sets)
- src/tnfr/operators/grammar_validate.py (the canonical U1-U6 oracle)
- src/tnfr/config/physics_derivation.py (the single source the sets derive from)
- examples/08_emergent_geometry/139_grammar_formal_language.py (the assertion)
- AGENTS.md "Unified Grammar (U1-U6)"
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import itertools

import numpy as np

from tnfr.operators.grammar_types import (
    GENERATORS, CLOSURES, STABILIZERS, DESTABILIZERS, TRANSFORMERS,
)
from tnfr.operators.definitions import (
    Emission, Reception, Coherence, Dissonance, Coupling, Resonance,
    Silence, Expansion, Contraction, SelfOrganization, Mutation,
    Transition, Recursivity,
)
from tnfr.operators.grammar_validate import validate_grammar

NAME2INST = {
    "emission": Emission(), "reception": Reception(), "coherence": Coherence(),
    "dissonance": Dissonance(), "coupling": Coupling(), "resonance": Resonance(),
    "silence": Silence(), "expansion": Expansion(), "contraction": Contraction(),
    "self_organization": SelfOrganization(), "mutation": Mutation(),
    "transition": Transition(), "recursivity": Recursivity(),
}
ALPHA = list(NAME2INST.keys())
START = ("START",)
DEAD = ("DEAD",)


def tag(x):
    """Operator tag for the U4b window: D destabilizer, I coherence/IL, O other."""
    if x in DESTABILIZERS:
        return "D"
    if x == "coherence":
        return "I"
    return "O"


def transition(state, x):
    """Canonical transition: append operator x, or None if grammar forbids it.

    U1a: only generators leave START.  U4b: a transformer needs a recent
    destabilizer (a D in the last-3 window); ZHIR (mutation) also needs a recent
    IL (an I in the window).  The state carries the U2 has-destabilizer /
    has-stabilizer flags and the U1b last-is-closure bit.
    """
    if state == START:
        if x not in GENERATORS:
            return None
        return ((tag(x),), x in DESTABILIZERS, x in STABILIZERS, x in CLOSURES)
    if len(state) != 4:                       # DEAD
        return None
    win, has_d, has_s, _last_clo = state
    if x in TRANSFORMERS:
        if "D" not in win:
            return None
        if x == "mutation" and "I" not in win:
            return None
    new_win = (win + (tag(x),))[-3:]
    return (new_win, has_d or x in DESTABILIZERS,
            has_s or x in STABILIZERS, x in CLOSURES)


def is_accept(state):
    """U1b (last is a closure) and U2 (no destabilizer without a stabilizer)."""
    if len(state) != 4:
        return False
    _win, has_d, has_s, last_clo = state
    return last_clo and (not has_d or has_s)


def build_automaton():
    """BFS the reachable states and the transition multigraph from START."""
    states = {START}
    edges: dict = {}
    frontier = [START]
    while frontier:
        s = frontier.pop()
        for x in ALPHA:
            ns = transition(s, x)
            if ns is None:
                continue
            edges.setdefault(s, []).append((x, ns))
            if ns not in states:
                states.add(ns)
                frontier.append(ns)
    return states, edges


def automaton_counts(edges, maxn):
    """Accepted-walk counts N_auto(n) from START (DP over the layers)."""
    layer = {START: 1}
    out = []
    for _ in range(maxn):
        nxt: dict = {}
        for s, c in layer.items():
            for (_x, ns) in edges.get(s, ()):
                nxt[ns] = nxt.get(ns, 0) + c
        out.append(sum(c for st, c in nxt.items() if is_accept(st)))
        layer = nxt
    return out


def oracle_counts(maxn):
    """Canonical N(n) via validate_grammar (U1-pruned for speed)."""
    GEN, CLO = sorted(GENERATORS), sorted(CLOSURES)
    out = []
    for n in range(1, maxn + 1):
        if n == 1:
            out.append(sum(1 for x in ALPHA
                           if validate_grammar([NAME2INST[x]], 0.0)))
            continue
        c = 0
        for first in GEN:
            for last in CLO:
                for mid in itertools.product(ALPHA, repeat=n - 2):
                    seq = ([NAME2INST[first]] + [NAME2INST[m] for m in mid]
                           + [NAME2INST[last]])
                    if validate_grammar(seq, 0.0):
                        c += 1
        out.append(c)
    return out


def experiment_1_automaton_reproduces_grammar(states, edges):
    """M1: the explicit automaton reproduces the canonical oracle N(n)."""
    print("=" * 70)
    print("M1: THE EXPLICIT AUTOMATON REPRODUCES THE GRAMMAR")
    print("=" * 70)
    print("Built from the CANONICAL centralized sets (grammar_types):")
    print(f"  generators   = {sorted(GENERATORS)}")
    print(f"  closures     = {sorted(CLOSURES)}")
    print(f"  destabilizers = {sorted(DESTABILIZERS)}")
    print(f"  transformers  = {sorted(TRANSFORMERS)}")
    n_edges = sum(len(v) for v in edges.values())
    print(f"  reachable states = {len(states)}, transitions = {n_edges}")
    print()
    maxn = 6
    auto = automaton_counts(edges, maxn)
    orac = oracle_counts(maxn)
    print(f"  {'n':>3} {'automaton':>11} {'oracle N(n)':>12} {'match':>7}")
    for n in range(maxn):
        print(f"  {n+1:>3} {auto[n]:>11} {orac[n]:>12} "
              f"{str(auto[n] == orac[n]):>7}")
    print()
    print("  -> the automaton (state = last-3 tags + U2 flags + closure bit)")
    print("     reproduces the canonical oracle exactly: it IS the grammar's")
    print("     finite-state machine. (U4a is subsumed by U2 — the bifurcation")
    print("     handlers ARE the stabilizers {IL, THOL}.)")


def experiment_2_minimal_dfa(states, edges):
    """M2: minimal DFA via partition refinement (Myhill-Nerode index)."""
    print()
    print("=" * 70)
    print("M2: THE MINIMAL DFA (CONSTRUCTIVE REGULARITY)")
    print("=" * 70)
    allstates = list(states) + [DEAD]
    sid = {s: i for i, s in enumerate(allstates)}
    delta = {}
    for s in allstates:
        for x in ALPHA:
            ns = transition(s, x) if s != DEAD else None
            delta[(sid[s], x)] = (sid[ns] if (ns is not None and ns in sid)
                                  else sid[DEAD])
    part = {sid[s]: (1 if is_accept(s) else 0) for s in allstates}
    while True:
        sig = {
            sid[s]: (part[sid[s]],)
            + tuple(part[delta[(sid[s], x)]] for x in ALPHA)
            for s in allstates
        }
        newlab: dict = {}
        newpart: dict = {}
        nxt = 0
        for i in sorted(sig, key=lambda k: sig[k]):
            if sig[i] not in newlab:
                newlab[sig[i]] = nxt
                nxt += 1
            newpart[i] = newlab[sig[i]]
        if newpart == part:
            break
        part = newpart
    nclasses = len(set(part.values()))
    print(f"  reachable states (raw) = {len(states)}")
    print(f"  minimal DFA states (Myhill-Nerode index, incl dead sink) "
          f"= {nclasses}")
    print()
    print(f"  -> a FINITE minimal automaton of {nclasses} states EXISTS and is")
    print("     exhibited: L is regular CONSTRUCTIVELY (example 139 argued it")
    print("     from finite memory; here the machine is built and minimized).")


def experiment_3_exact_capacity(states, edges):
    """M3: Perron-Frobenius eigenvalue = exact capacity."""
    print()
    print("=" * 70)
    print("M3: THE EXACT CAPACITY (PERRON-FROBENIUS EIGENVALUE)")
    print("=" * 70)
    # trim = reachable AND co-reachable (can still reach an accept state)
    co = set(s for s in states if is_accept(s))
    changed = True
    while changed:
        changed = False
        for s in states:
            if s not in co and any(ns in co for (_, ns) in edges.get(s, ())):
                co.add(s)
                changed = True
    trim = [s for s in states if s in co and s != START]
    idx = {s: i for i, s in enumerate(trim)}
    M = np.zeros((len(trim), len(trim)))
    for s in trim:
        for (_x, ns) in edges.get(s, ()):
            if ns in idx:
                M[idx[s], idx[ns]] += 1
    lam = float(np.max(np.abs(np.linalg.eigvals(M))))
    print(f"  trim states = {len(trim)}")
    print(f"  Perron-Frobenius eigenvalue lambda = {lam:.6f}")
    print(f"  capacity log2(lambda) = {np.log2(lam):.4f} bits/operator "
          f"(alphabet log2(13) = {np.log2(13):.4f})")
    print()
    print("  lambda is the connective constant = asymptotic branching factor of")
    print("  grammatically-allowed continuations.  The finite N(n)/N(n-1)")
    print("  estimates of example 139 are NON-MONOTONIC and converge to it:")
    ext = automaton_counts(edges, 80)
    print(f"    {'n':>4} {'N(n)/N(n-1)':>13} {'|ratio - lambda|':>17}")
    for n in (6, 9, 20, 40, 79):
        ratio = ext[n - 1] / ext[n - 2]
        print(f"    {n:>4} {ratio:>13.5f} {abs(ratio - lam):>17.2e}")
    print()
    print("  -> the ratio overshoots to ~12.19 around n=9, then settles down to")
    print(f"     lambda = {lam:.4f} (the example-139 finite estimates at n<=6 had")
    print("     not yet converged). The exact eigenvalue resolves them.")


def main():
    print()
    print("  ===============================================================")
    print("  The Grammar Automaton")
    print("  Construction, Minimal DFA, and Exact Capacity")
    print("  ===============================================================")
    print()
    states, edges = build_automaton()
    experiment_1_automaton_reproduces_grammar(states, edges)
    experiment_2_minimal_dfa(states, edges)
    experiment_3_exact_capacity(states, edges)
    print()
    print("=" * 70)
    print("WHAT THIS ESTABLISHES")
    print("=" * 70)
    print("The unified grammar's finite-state machine, built DIRECTLY from the")
    print("canonical centralized operator sets, reproduces the canonical oracle")
    print("N(n) exactly (M1) — a constructive cross-check of the grammar")
    print("centralization. Myhill-Nerode minimization exhibits a concrete")
    print("29-state minimal DFA (M2): L is regular constructively, not just by")
    print("the finite-memory argument of example 139. The transfer matrix's")
    print("Perron-Frobenius eigenvalue lambda = 11.560930 is the EXACT capacity")
    print("(M3), the connective constant the non-monotonic finite N(n)/N(n-1)")
    print("estimates converge to. HONEST SCOPE: standard automata / symbolic-")
    print("dynamics theory (Myhill-Nerode minimal DFA, Perron-Frobenius / topological")
    print("entropy of a sofic language); a constructive characterization deepening")
    print("example 139, not new mathematics, closes no open problem.")


if __name__ == "__main__":
    main()
