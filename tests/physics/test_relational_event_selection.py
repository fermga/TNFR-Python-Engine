"""Static event-choice restrictions under a declared exact finite action.

Candidate vertices are an incidence encoding for the existing selector owner,
not physical nodes or a new selection API. Exact phase-turn labels describe the
prepared ideal states; native fields independently use materialized radians.
No trajectory, event clock, stochastic law or winding recovery is inferred.
"""

import math
from copy import deepcopy
from fractions import Fraction as Q
from itertools import product

import networkx as nx
import pytest

from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.physics.relational_observations import observe_relational_relocation
from tnfr.physics.selector_symmetry import derive_selector_symmetry

EPSILON = Q(1, 256)
OLD_BRIDGE = (0, 5)
MODEL = RelationalExchangeModel(1)


def _prepared(winding=0):
    graph = nx.disjoint_union(nx.cycle_graph(5), nx.cycle_graph(5))
    graph.add_edge(*OLD_BRIDGE)
    labels = []
    for node in graph:
        form = EPSILON if node == 0 else Q(0)
        turns = Q(winding * (node % 5), 5)
        graph.nodes[node].update(
            EPI=float(form), theta=2 * math.pi * winding * (node % 5) / 5, nu_f=1.0
        )
        labels.append(("node", ("form", form), ("phase_turns", turns), ("capacity", 1)))
    return graph, tuple(labels)


def _reflection(left, right):
    return tuple(
        5 * (node // 5) + (-(node % 5) % 5 if (left, right)[node // 5] else node % 5)
        for node in range(10)
    )


def _incidence_action(graph, labels, candidates, actions, *, abstention=False):
    """Declare complete primitive/support labels and unordered event incidences."""
    size = len(graph)
    event_indices = tuple(range(size, size + len(candidates)))
    all_labels = labels + (("bridge_exchange",),) * len(candidates)
    if abstention:
        all_labels += (("abstention",),)
    total = len(all_labels)
    relations = [[("absent",) for _ in range(total)] for _ in range(total)]
    for i, j in product(range(size), repeat=2):
        present = graph.has_edge(i, j)
        relations[i][j] = ("support", present, Q(1) if present else Q(0))
    for index, edge in zip(event_indices, candidates, strict=True):
        for node in graph:
            incidence = ("event_incidence", node in OLD_BRIDGE, node in edge)
            relations[index][node] = relations[node][index] = incidence
    by_edge = {frozenset(edge): index for index, edge in zip(event_indices, candidates)}
    lifted = []
    for action in actions:
        extension = tuple(
            by_edge[frozenset(action[node] for node in edge)] for edge in candidates
        )
        lifted.append(action + extension + ((total - 1,) if abstention else ()))
    return dict(
        state_labels=all_labels,
        relation_labels=relations,
        permutations=tuple(lifted),
        candidates=event_indices + ((total - 1,) if abstention else ()),
    )


def _synchronized_action(*, abstention=False):
    graph, labels = _prepared()
    # The candidate-universe premise is "both new endpoints neighbor the old
    # ports". It is support-defined and invariant under independent reflections.
    neighbors = tuple(
        tuple(n for n in graph[port] if n != other)
        for port, other in (OLD_BRIDGE, tuple(reversed(OLD_BRIDGE)))
    )
    candidates = tuple(product(*neighbors))
    assert set(candidates) == {(1, 6), (1, 9), (4, 6), (4, 9)}
    actions = tuple(
        _reflection(left, right) for left, right in product((False, True), repeat=2)
    )
    return (
        graph,
        candidates,
        _incidence_action(graph, labels, candidates, actions, abstention=abstention),
    )


def test_full_state_incidence_stabilizer_obstructs_a_single_bridge_choice():
    _, _, declaration = _synchronized_action()
    report = derive_selector_symmetry(**declaration)
    assert len(report.stabilizer_permutations) == 4
    assert report.candidate_orbits == ((10, 11, 12, 13),)
    assert report.fixed_candidates == ()
    assert report.unique_equivariant_selection_obstructed
    # A proposed deterministic choice must equal its image for EVERY symmetry.
    # Independently verify that each supplied event violates this condition.
    for candidate in declaration["candidates"]:
        assert any(
            action[candidate] != candidate for action in declaration["permutations"]
        )


def test_tied_candidates_are_strictly_passive_and_have_distinct_native_responses():
    graph, candidates, _ = _synchronized_action()
    original = deepcopy(graph)
    responses = {}
    for new in candidates:
        report = observe_relational_relocation(
            graph, model=MODEL, remove_bridge=OLD_BRIDGE, add_bridge=new
        )
        assert report.storage_change == report.form_storage_change == -(EPSILON**2) / 2
        assert report.phase_storage_change == 0
        assert report.represented_zero_supply_passive
        assert report.before.form_storage == 3 * EPSILON**2 / 2
        assert report.after.form_storage == EPSILON**2
        # At phase zero, g=0, H=pi*d. The new left port still has q=-eps,
        # but degree changes from two to three. The former receiver loses q.
        a, b = new
        assert float(report.form_rate_change[a]) == pytest.approx(
            -float(EPSILON) / 12, rel=0, abs=1e-15
        )
        assert float(report.phase_rate_change[a]) == pytest.approx(
            float(EPSILON) / (12 * math.pi), rel=0, abs=1e-15
        )
        assert float(report.form_rate_change[5]) == pytest.approx(
            -float(EPSILON) / 6, rel=0, abs=1e-15
        )
        assert report.form_rate_change[b] == report.phase_rate_change[b] == 0
        responses[new] = report
    # Choosing which left port acquires the bridge changes the observed field.
    assert responses[(1, 6)].after.form_rate != responses[(4, 6)].after.form_rate
    # Choosing a right port is still a different support event when its current
    # local rates happen to agree: the fresh phase metric changes there.
    assert responses[(1, 6)].after.phase_metric != responses[(1, 9)].after.phase_metric
    assert graph.graph == original.graph
    assert dict(graph.nodes(data=True)) == dict(original.nodes(data=True))
    assert set(graph.edges) == set(original.edges)


def test_candidate_order_and_an_undeclared_singleton_cannot_break_the_tie():
    _, _, declaration = _synchronized_action()
    expected = derive_selector_symmetry(**declaration)
    reordered = dict(declaration)
    reordered["candidates"] = tuple(reversed(declaration["candidates"]))
    reordered["permutations"] = tuple(reversed(declaration["permutations"]))
    assert derive_selector_symmetry(**reordered) == expected
    restricted = dict(declaration, candidates=(declaration["candidates"][0],))
    with pytest.raises(ValueError, match="invariant"):
        derive_selector_symmetry(**restricted)


def test_abstention_is_fixed_but_does_not_supply_an_occurrence_law():
    _, _, declaration = _synchronized_action(abstention=True)
    report = derive_selector_symmetry(**declaration)
    assert report.candidate_orbits == ((10, 11, 12, 13), (14,))
    assert report.fixed_candidates == (14,)
    assert not report.unique_equivariant_selection_obstructed
    # This fixed outcome is a caller-declared abstention card, not a graph node,
    # a probability distribution or evidence that an event ever occurs.
    assert report.state_labels[14] == ("abstention",)


def test_equal_winding_state_costs_do_not_imply_full_state_symmetry():
    graph, labels = _prepared(winding=1)
    candidates = tuple((k, k + 5) for k in range(1, 5))
    actions = (_reflection(False, False), _reflection(True, True))
    declaration = _incidence_action(graph, labels, candidates, actions)
    report = derive_selector_symmetry(**declaration)
    assert report.stabilizer_permutations == (tuple(range(14)),)
    assert report.fixed_candidates == (10, 11, 12, 13)
    assert not report.unique_equivariant_selection_obstructed

    # Check the FULL support automorphism group, not just absence of a subgroup
    # obstruction. Complete exact primitive colors eliminate every nonidentity
    # automorphism. This does not prove that any selector is unique or canonical.
    colored = graph.copy()
    nx.set_node_attributes(colored, dict(enumerate(labels)), "primitive_color")
    matcher = nx.algorithms.isomorphism.GraphMatcher(
        colored,
        colored,
        node_match=lambda left, right: left["primitive_color"]
        == right["primitive_color"],
    )
    assert tuple(matcher.isomorphisms_iter()) == (dict(enumerate(range(10))),)
    for new in candidates:
        observation = observe_relational_relocation(
            graph, model=MODEL, remove_bridge=OLD_BRIDGE, add_bridge=new
        )
        assert observation.storage_change == -(EPSILON**2) / 2
