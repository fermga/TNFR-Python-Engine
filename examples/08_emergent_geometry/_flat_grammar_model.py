"""Finite-state model of TNFR's flat operator-history constraints.

This support module centralizes the automata used by examples 140--145 and 150.
It models the part of ``validate_grammar`` that is decidable from a flat stream
of default-depth operator names with ``epi_initial=0``:

* U1 start and closure conditions;
* U2 prefix debt and final stabilizer-presence checks;
* U4a trigger/handler presence;
* U4b recent-destabilizer and lifetime-prior-IL context; and
* the U2 REMESH amplification check.

The projection deliberately does not certify runtime U3 phase compatibility,
canonical U6 potential drift, or U5 nested parent/child coherence. Those checks
need graph state, a reference field, or an explicit nested representation. The
default ``Recursivity()`` symbol used by the oracle has depth one, so U5 is
inactive in the flat cross-checks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np

from tnfr.config.operator_names import BIFURCATION_WINDOW, U2_DEBT_CAPACITY
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
from tnfr.operators.grammar_debt import advance_debt
from tnfr.operators.grammar_types import (
    BIFURCATION_HANDLERS,
    BIFURCATION_TRIGGERS,
    CLOSURES,
    DESTABILIZERS,
    GENERATORS,
    STABILIZERS,
    TRANSFORMERS,
)

ALPHA: tuple[str, ...] = (
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
)

SHORT: Mapping[str, str] = {
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

NAME2INST = {
    "emission": Emission(),
    "reception": Reception(),
    "coherence": Coherence(),
    "dissonance": Dissonance(),
    "coupling": Coupling(),
    "resonance": Resonance(),
    "silence": Silence(),
    "expansion": Expansion(),
    "contraction": Contraction(),
    "self_organization": SelfOrganization(),
    "mutation": Mutation(),
    "transition": Transition(),
    "recursivity": Recursivity(),
}

START = ("START",)
DEAD = ("DEAD",)


@dataclass(frozen=True)
class FlatRules:
    """Switches for the history-only rules represented by the automaton."""

    u1a: bool = True
    u1b: bool = True
    u2: bool = True
    u4a: bool = True
    u4b: bool = True
    u2_remesh: bool = True


DEFAULT_RULES = FlatRules()


def _initial_state() -> tuple:
    return ((), False, False, False, 0, False, False, False, False)


def transition(state: tuple, symbol: str, rules: FlatRules = DEFAULT_RULES):
    """Return the next flat-history state, or ``None`` for a forbidden prefix."""

    if symbol not in ALPHA:
        raise KeyError(f"unknown operator symbol: {symbol!r}")
    if state == DEAD:
        return None
    if state == START:
        if rules.u1a and symbol not in GENERATORS:
            return None
        state = _initial_state()

    (
        recent_destabilizers,
        prior_il,
        has_destabilizer,
        has_stabilizer,
        debt,
        has_trigger,
        has_handler,
        has_remesh,
        _last_closure,
    ) = state

    if rules.u4b and symbol in TRANSFORMERS:
        if not any(recent_destabilizers):
            return None
        if symbol == "mutation" and not prior_il:
            return None

    next_debt = advance_debt(debt, symbol) if rules.u2 else 0
    if rules.u2 and next_debt > U2_DEBT_CAPACITY:
        return None

    recent = (recent_destabilizers + (symbol in DESTABILIZERS,))[
        -BIFURCATION_WINDOW:
    ]
    return (
        recent,
        prior_il or symbol == "coherence",
        has_destabilizer or symbol in DESTABILIZERS,
        has_stabilizer or symbol in STABILIZERS,
        next_debt,
        has_trigger or symbol in BIFURCATION_TRIGGERS,
        has_handler or symbol in BIFURCATION_HANDLERS,
        has_remesh or symbol == "recursivity",
        symbol in CLOSURES,
    )


def is_accept(state: tuple, rules: FlatRules = DEFAULT_RULES) -> bool:
    """Return whether a reachable state satisfies the enabled final checks."""

    if state in (START, DEAD) or len(state) != 9:
        return False
    (
        _recent,
        _prior_il,
        has_destabilizer,
        has_stabilizer,
        _debt,
        has_trigger,
        has_handler,
        has_remesh,
        last_closure,
    ) = state
    if rules.u1b and not last_closure:
        return False
    if rules.u2 and has_destabilizer and not has_stabilizer:
        return False
    if rules.u4a and has_trigger and not has_handler:
        return False
    if rules.u2_remesh and has_remesh and has_destabilizer and not has_stabilizer:
        return False
    return True


def build_automaton(
    rules: FlatRules = DEFAULT_RULES,
    alpha: Sequence[str] = ALPHA,
) -> tuple[set[tuple], dict[tuple, list[tuple[str, tuple]]]]:
    """Build all reachable states and labelled edges by breadth-first search."""

    states = {START}
    edges: dict[tuple, list[tuple[str, tuple]]] = {}
    frontier = [START]
    while frontier:
        state = frontier.pop()
        for symbol in alpha:
            next_state = transition(state, symbol, rules)
            if next_state is None:
                continue
            edges.setdefault(state, []).append((symbol, next_state))
            if next_state not in states:
                states.add(next_state)
                frontier.append(next_state)
    return states, edges


def accepted_counts(
    edges: Mapping[tuple, Iterable[tuple[str, tuple]]],
    max_length: int,
    rules: FlatRules = DEFAULT_RULES,
) -> list[int]:
    """Return exact accepted-word counts for lengths 1 through ``max_length``."""

    layer = {START: 1}
    counts: list[int] = []
    for _ in range(max_length):
        next_layer: dict[tuple, int] = {}
        for state, count in layer.items():
            for _symbol, next_state in edges.get(state, ()):
                next_layer[next_state] = next_layer.get(next_state, 0) + count
        counts.append(
            sum(
                count
                for state, count in next_layer.items()
                if is_accept(state, rules)
            )
        )
        layer = next_layer
    return counts


def coreachable_states(
    states: Iterable[tuple],
    edges: Mapping[tuple, Iterable[tuple[str, tuple]]],
    rules: FlatRules = DEFAULT_RULES,
) -> set[tuple]:
    """Return reachable states from which an accepting state is reachable."""

    state_set = set(states)
    coreachable = {state for state in state_set if is_accept(state, rules)}
    changed = True
    while changed:
        changed = False
        for state in state_set - coreachable:
            if any(next_state in coreachable for _, next_state in edges.get(state, ())):
                coreachable.add(state)
                changed = True
    return coreachable


def transfer_matrix(
    states: Iterable[tuple],
    edges: Mapping[tuple, Iterable[tuple[str, tuple]]],
    rules: FlatRules = DEFAULT_RULES,
    *,
    include_start: bool = False,
) -> tuple[list[tuple], dict[tuple, int], np.ndarray]:
    """Return the exact integer transfer matrix on the trimmed automaton."""

    coreachable = coreachable_states(states, edges, rules)
    trim = sorted(
        (
            state
            for state in states
            if state in coreachable and (include_start or state != START)
        ),
        key=str,
    )
    index = {state: i for i, state in enumerate(trim)}
    matrix = np.zeros((len(trim), len(trim)), dtype=float)
    for state in trim:
        for _symbol, next_state in edges.get(state, ()):
            if next_state in index:
                matrix[index[state], index[next_state]] += 1.0
    return trim, index, matrix


def spectral_radius(matrix: np.ndarray) -> float:
    """Return the numerically evaluated spectral radius of a finite matrix."""

    if matrix.size == 0:
        return 0.0
    return float(np.max(np.abs(np.linalg.eigvals(matrix))))


def minimize_dfa(
    states: Iterable[tuple],
    rules: FlatRules = DEFAULT_RULES,
    alpha: Sequence[str] = ALPHA,
) -> tuple[dict[tuple, int], Callable[[tuple, str], tuple]]:
    """Minimize the complete DFA by stable partition refinement."""

    all_states = list(states) + [DEAD]

    def delta(state: tuple, symbol: str) -> tuple:
        if state == DEAD:
            return DEAD
        next_state = transition(state, symbol, rules)
        return next_state if next_state is not None else DEAD

    partition = {state: int(is_accept(state, rules)) for state in all_states}
    while True:
        labels: dict[tuple, int] = {}
        next_partition: dict[tuple, int] = {}
        for state in all_states:
            signature = (partition[state],) + tuple(
                partition[delta(state, symbol)] for symbol in alpha
            )
            labels.setdefault(signature, len(labels))
            next_partition[state] = labels[signature]
        if all(next_partition[s] == partition[s] for s in all_states):
            return next_partition, delta
        # Numeric labels can be permuted even after the equivalence classes are
        # stable. Canonicalize by state-membership sets before the next round.
        old_classes = {
            frozenset(s for s in all_states if partition[s] == label)
            for label in set(partition.values())
        }
        new_classes = {
            frozenset(s for s in all_states if next_partition[s] == label)
            for label in set(next_partition.values())
        }
        if old_classes == new_classes:
            return next_partition, delta
        partition = next_partition
