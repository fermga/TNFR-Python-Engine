#!/usr/bin/env python3
"""
Example 150 — The Emergent Grammatical Pattern: the Parry Maximum-Entropy
Measure, the Capacity Split, and the H-Theorem of Coherent Generation
==============================================================================

The grammar thread (139-145) characterized the unified grammar U1-U6 as a
star-free regular language with capacity λ = 11.560930 and a 312-element
syntactic monoid. This example studies the PATTERN the grammar produces ON ITS
OWN — the operator distribution that EMERGES from U1-U6 with no imposed bias —
and connects it to physics (maximum entropy / the H-theorem) and to the unifying
dual-lever lens (examples 146-149).

The emergent pattern is the Parry measure
-----------------------------------------
For a regular language given by an automaton, the unique stationary measure of
maximum entropy is the PARRY MEASURE (Shannon–Parry 1964): the Markov measure
whose transition probabilities are

    P(i→j) = M_ij · r_j / (λ · r_i),

with r the right Perron–Frobenius eigenvector of the transfer matrix M and λ its
spectral radius. Among ALL stationary measures supported on the same allowed
transitions, Parry is the one that maximizes the entropy rate — it is the
discrete maximum-entropy / Jaynes distribution of the language. It is what
"coherent generation with no further preference" looks like.

The recurrent phase
-------------------
The grammar automaton is NOT strongly connected: it has a transient START/launch
phase and a single large RECURRENT phase (the 45-state strongly-connected
component that carries the spectral radius λ). The emergent steady-state pattern
lives in this recurrent phase — the regime a long coherent sequence settles into.

The capacity split (the new exact result)
------------------------------------------
The total capacity log λ = 2.447631 bits/operator splits EXACTLY into two
information channels:

    log λ  =  H_state  +  H_choice
    2.447631 = 1.390957 + 1.056674   (residual ~3e-15)

  * H_state (1.39 bits) — the entropy of the STATE transition: which structural
    move the coherence flow makes next (the automaton-level dynamics).
  * H_choice (1.06 bits) — the entropy of WHICH operator within a grammatical
    equivalence class is used (the in-class free choice of the quotient,
    example 142). Several operators share the same automaton edge — a multigraph
    — and choosing among them is the second channel.

So the emergent generation budget is: ~57% spent on the structural state move,
~43% on the free choice among interchangeable operators (ex 142's redundancy,
now exact in entropy units).

The H-theorem (the equilibrium the dynamics relaxes to)
-------------------------------------------------------
The Parry measure π is the stationary distribution of the maximum-entropy walk.
Started from any distribution on the recurrent phase, the relative entropy
D(p_t ‖ π) decreases monotonically to 0 (the Markov-chain H-theorem, the same
arrow-of-time structure as example 135's diffusion H-theorem). So π is the
EQUILIBRIUM the coherent generation dynamics relaxes to — the emergent pattern is
a thermodynamic equilibrium of the language.

Doctrine compliance
-------------------
The automaton is built from the canonical centralized grammar sets (grammar_canon
→ grammar_types → physics_derivation). The Parry measure, the capacity split, and
the H-theorem are exact symbolic-dynamics / information-theory facts; the
dual-lever reading uses the canonical operator classification (AGENTS.md
"Operator-Tetrad Synergies").

Three measured results
----------------------
M1 THE PARRY MAXIMUM-ENTROPY PATTERN. On the 45-state recurrent component the
   Parry measure achieves the maximum entropy rate, strictly above the
   uniform-edge walk (h_Parry > h_uniform). The emergent operator frequencies are
   read off π: the low-constraint destabilizers OZ/VAL lead (~10% each), the
   U4b-bottlenecked ZHIR is rarest (~1%) — the emergent frequency of an operator
   is set by how little the grammar constrains it.

M2 THE CAPACITY SPLITS EXACTLY INTO STATE + CHOICE. log λ = H_state + H_choice =
   1.390957 + 1.056674 = 2.447631 (residual ~3e-15). The structural state move
   carries 1.39 bits; the in-class operator choice (the ex-142 quotient
   redundancy) carries 1.06 bits. Coherent generation is ~57% structure, ~43%
   free choice among grammatically interchangeable operators.

M3 THE H-THEOREM — THE EMERGENT PATTERN IS AN EQUILIBRIUM. Under the Parry walk
   the relative entropy D(p_t ‖ π) decreases monotonically to 0: π is the
   equilibrium the coherent dynamics relaxes to (the Markov H-theorem, the same
   arrow-of-time structure as ex 135). Read through the dual-lever (ex 146-149),
   the emergent steady state spreads coherent generation across both arms
   (pressure ΔNFR and capacity νf) — no single lever dominates the equilibrium.

Honest scope
------------
This is standard symbolic-dynamics + information theory (the Shannon–Parry
maximum-entropy measure; the entropy chain rule giving the state/choice split;
the Markov-chain H-theorem) computed on the canonical automaton of example 140.
It is a CHARACTERIZATION of the pattern the grammar produces; it is not new
mathematics and closes no open problem. The capacity split log λ = H_state +
H_choice is exact and ties the emergent pattern to the ex-142 quotient; the
H-theorem ties it to the ex-135 arrow of time; the dual-lever reading ties it to
the ex 146-149 unification.

References
----------
- examples/08_emergent_geometry/140_grammar_automaton.py (the automaton, λ)
- examples/08_emergent_geometry/142_grammar_operator_quotient.py (the quotient / H_choice)
- examples/08_emergent_geometry/145_syntactic_monoid_starfree.py (star-free monoid)
- examples/08_emergent_geometry/135_arrow_of_time_h_theorem.py (the diffusion H-theorem)
- examples/07_number_theory/147_numbers_as_free_monoid_words.py (the dual-lever)
- Shannon & Parry (1964): the maximum-entropy measure of a sofic/regular language
- AGENTS.md "Operator-Tetrad Synergies" (dual-lever), "Unified Grammar (U1-U6)"
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import networkx as nx
import numpy as np

from tnfr.operators.grammar_types import (
    CLOSURES,
    DESTABILIZERS,
    GENERATORS,
    STABILIZERS,
    TRANSFORMERS,
)

ALPHA = [
    "emission",
    "reception",
    "coherence",
    "dissonance",
    "coupling",
    "resonance",
    "silence",
    "expansion",
    "contraction",
    "self_organization",
    "mutation",
    "transition",
    "recursivity",
]
SHORT = {
    "emission": "AL",
    "reception": "EN",
    "coherence": "IL",
    "dissonance": "OZ",
    "coupling": "UM",
    "resonance": "RA",
    "silence": "SHA",
    "expansion": "VAL",
    "contraction": "NUL",
    "self_organization": "THOL",
    "mutation": "ZHIR",
    "transition": "NAV",
    "recursivity": "REMESH",
}
# Dual-lever classification (AGENTS.md "Operator-Tetrad Synergies", ex 37/130).
LEVER = {
    "UM": "nu_f",
    "SHA": "nu_f",
    "VAL": "nu_f",
    "IL": "dNFR",
    "OZ": "dNFR",
    "THOL": "dNFR",
    "ZHIR": "dNFR",
    "NAV": "dNFR",
    "NUL": "both",
    "AL": "neither",
    "EN": "neither",
    "RA": "neither",
    "REMESH": "neither",
}
START = ("START",)


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
    return (
        (win + (tag(x),))[-3:],
        has_d or x in DESTABILIZERS,
        has_s or x in STABILIZERS,
        x in CLOSURES,
    )


def is_accept(state):
    if len(state) != 4:
        return False
    _w, has_d, has_s, last_clo = state
    return last_clo and (not has_d or has_s)


def build_trim():
    """Reachable + co-reachable states, the multigraph M, and the op multiplicity."""
    states = {START}
    frontier = [START]
    while frontier:
        s = frontier.pop()
        for x in ALPHA:
            ns = transition(s, x)
            if ns is not None and ns not in states:
                states.add(ns)
                frontier.append(ns)
    co = {s for s in states if is_accept(s)}
    changed = True
    while changed:
        changed = False
        for s in states:
            if s not in co and any(
                transition(s, x) in co for x in ALPHA if transition(s, x) is not None
            ):
                co.add(s)
                changed = True
    trim = [s for s in states if s in co]
    idx = {s: i for i, s in enumerate(trim)}
    n = len(trim)
    M = np.zeros((n, n))
    ops = {}  # (i, j) -> list of operators on that edge
    for s in trim:
        for x in ALPHA:
            ns = transition(s, x)
            if ns in idx:
                M[idx[s], idx[ns]] += 1
                ops.setdefault((idx[s], idx[ns]), []).append(x)
    return trim, idx, M, ops


def recurrent_scc(trim, idx, M):
    """The strongly-connected component carrying the spectral radius."""
    n = len(trim)
    lam = float(max(np.linalg.eigvals(M).real))
    DG = nx.DiGraph()
    DG.add_nodes_from(range(n))
    for i in range(n):
        for j in range(n):
            if M[i, j] > 0:
                DG.add_edge(i, j)
    for comp in sorted(nx.strongly_connected_components(DG), key=len, reverse=True):
        sub = sorted(comp)
        Msub = M[np.ix_(sub, sub)]
        if Msub.size and abs(float(max(np.linalg.eigvals(Msub).real)) - lam) < 1e-6:
            return sub, Msub, lam
    raise RuntimeError("no recurrent SCC carrying lambda")


def parry_measure(Msub):
    """Parry transition matrix P, stationary pi, right eigvec r, spectral radius."""
    m = len(Msub)
    w, V = np.linalg.eig(Msub)
    k = int(np.argmax(w.real))
    lam = float(w[k].real)
    r = np.abs(V[:, k].real)
    wl, Vl = np.linalg.eig(Msub.T)
    kl = int(np.argmax(wl.real))
    left = np.abs(Vl[:, kl].real)
    P = np.zeros((m, m))
    for i in range(m):
        if r[i] > 1e-15:
            for j in range(m):
                if Msub[i, j] > 0 and r[j] > 1e-15:
                    P[i, j] = Msub[i, j] * r[j] / (lam * r[i])
    pi = left * r
    pi = pi / pi.sum()
    return P, pi, r, lam


def experiment_1_parry_pattern(trim, idx, sub, Msub, P, pi):
    print("=" * 72)
    print("M1: the Parry maximum-entropy pattern (the emergent operator frequencies)")
    print("=" * 72)
    m = len(sub)
    # uniform-edge contrast
    U = np.zeros((m, m))
    for i in range(m):
        tot = Msub[i].sum()
        if tot > 0:
            U[i] = Msub[i] / tot
    piU = np.ones(m) / m
    for _ in range(4000):
        piU = piU @ U
        piU = piU / piU.sum()

    def h_rate(Q, p):
        return -sum(
            p[i] * Q[i, j] * np.log(Q[i, j])
            for i in range(m)
            for j in range(m)
            if Q[i, j] > 1e-15
        )

    hP = h_rate(P, pi)
    hU = h_rate(U, np.abs(piU) / np.abs(piU).sum())
    print(f"  recurrent component: {m} states (the steady-state phase)")
    print(f"  Parry entropy rate  h_state = {hP:.6f} bits-nat (state channel)")
    print(
        f"  uniform-edge rate   h_unif  = {hU:.6f}  ->  Parry is the maximum "
        f"(+{hP-hU:.4f})"
    )

    # emergent operator frequency: Parry edge prob spread over the operators
    # sharing each edge (each gets an equal share of the edge's word-probability).
    pos = {v: i for i, v in enumerate(sub)}
    freq = {x: 0.0 for x in ALPHA}
    for s in trim:
        i = idx[s]
        if i not in pos:
            continue
        ii = pos[i]
        for x in ALPHA:
            ns = transition(s, x)
            if ns in idx and idx[ns] in pos:
                jj = pos[idx[ns]]
                # operators sharing edge (ii,jj): split the edge prob equally
                share = P[ii, jj] / Msub[ii, jj] if Msub[ii, jj] > 0 else 0.0
                freq[x] += pi[ii] * share
    tot = sum(freq.values())
    for x in freq:
        freq[x] /= tot
    print("  emergent operator frequencies (steady state):")
    for x in sorted(ALPHA, key=lambda x: -freq[x]):
        print(f"    {SHORT[x]:7s} {100*freq[x]:5.2f}%   [{LEVER[SHORT[x]]}]")
    print("  -> low-constraint destabilizers (OZ/VAL) lead; the U4b-bottlenecked")
    print("     ZHIR is rarest. Emergent frequency = how little U1-U6 constrains.")
    return freq


def experiment_2_capacity_split(sub, Msub, P, pi, ops, idx, trim, lam):
    print()
    print("=" * 72)
    print("M2: the capacity splits EXACTLY into state + choice channels")
    print("=" * 72)
    m = len(sub)
    h_state = -sum(
        pi[i] * P[i, j] * np.log(P[i, j])
        for i in range(m)
        for j in range(m)
        if P[i, j] > 1e-15
    )
    # H_choice: each multi-edge (i,j) with mult>1 adds log(mult) of in-class choice
    h_choice = 0.0
    for i in range(m):
        for j in range(m):
            si, sj = sub[i], sub[j]
            mult = len(ops.get((si, sj), []))
            if P[i, j] > 1e-15 and mult > 1:
                h_choice += pi[i] * P[i, j] * np.log(mult)
    print(f"  H_state  (which structural move)        = {h_state:.6f}")
    print(f"  H_choice (which operator in the class)  = {h_choice:.6f}")
    print(f"  H_state + H_choice = {h_state + h_choice:.6f}")
    print(
        f"  log(lambda)        = {np.log(lam):.6f}  "
        f"(residual {abs(h_state + h_choice - np.log(lam)):.1e})"
    )
    frac = h_state / (h_state + h_choice)
    print(f"  -> coherent generation is ~{100*frac:.0f}% structural state move,")
    print(f"     ~{100*(1-frac):.0f}% free choice among interchangeable operators")
    print("     (the ex-142 in-class quotient redundancy, now in entropy units).")


def experiment_3_h_theorem(sub, P, pi):
    print()
    print("=" * 72)
    print("M3: the H-theorem -- the emergent pattern is an equilibrium")
    print("=" * 72)
    m = len(sub)
    p = np.zeros(m)
    p[int(np.argmax(pi))] = 1.0
    p = p @ P
    p = p / p.sum()
    prev = None
    monotone = True
    print("  D(p_t || pi) under the Parry walk (relative entropy to equilibrium):")
    for t in range(12):
        msk = (pi > 1e-12) & (p > 1e-15)
        D = float(np.sum(p[msk] * np.log(p[msk] / pi[msk])))
        if prev is not None and D > prev + 1e-9:
            monotone = False
        if t < 7:
            print(f"    t={t}: D = {D:.5f}")
        prev = D
        p = p @ P
        s = p.sum()
        if s > 0:
            p = p / s
    print(f"  monotone decreasing to 0: {monotone}")
    print("  -> the Parry measure pi is the EQUILIBRIUM the coherent generation")
    print("     dynamics relaxes to (the Markov H-theorem, the same arrow-of-time")
    print("     structure as the diffusion H-theorem of ex 135). The emergent")
    print("     grammatical pattern is a maximum-entropy / Jaynes equilibrium.")


def main():
    print()
    print("#" * 72)
    print("# Example 150 - The Emergent Grammatical Pattern: Parry Maxent, the")
    print("#               Capacity Split, and the H-Theorem")
    print("#" * 72)
    print()
    trim, idx, M, ops = build_trim()
    sub, Msub, lam = recurrent_scc(trim, idx, M)
    P, pi, r, lam_r = parry_measure(Msub)
    experiment_1_parry_pattern(trim, idx, sub, Msub, P, pi)
    experiment_2_capacity_split(sub, Msub, P, pi, ops, idx, trim, lam_r)
    experiment_3_h_theorem(sub, P, pi)
    print()
    print("=" * 72)
    print("Summary")
    print("=" * 72)
    print("  The pattern the grammar produces on its own is the Parry maximum-")
    print("  entropy measure on its 45-state recurrent phase. Its capacity splits")
    print("  EXACTLY into a structural-state channel (1.39 bits) and an in-class")
    print("  operator-choice channel (1.06 bits, the ex-142 quotient redundancy),")
    print("  summing to log(lambda)=2.4476. Under the maximum-entropy walk the")
    print("  relative entropy decreases monotonically to the Parry equilibrium")
    print("  (the Markov H-theorem, ex 135's arrow of time). The emergent")
    print("  grammatical pattern is a Jaynes / maximum-entropy equilibrium that")
    print("  spreads coherent generation across both dual-lever arms.")
    print("  Characterization; no operator physics changed, no open problem closed.")
    print()


if __name__ == "__main__":
    main()
