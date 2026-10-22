"""Exact counterexample to completeness of fifteen marked winding values."""

from fractions import Fraction as Q

import networkx as nx
import pytest

from tests.sine_evidence_helpers import forbid_sine_regeneration
from tnfr.physics._organized_external_interaction import (
    _derive_organized_external_interaction,
)


@pytest.fixture(scope="module", autouse=True)
def no_trajectory_or_frozen_execution():
    with forbid_sine_regeneration():
        yield


@pytest.fixture(scope="module")
def geometry():
    return _derive_organized_external_interaction()


def _principal_turns(value):
    return value - (value + Q(1, 2)).numerator // (value + Q(1, 2)).denominator


def _cycle_gaps(state, cycle):
    return tuple(
        _principal_turns(state[b] - state[a])
        for a, b in zip(cycle, cycle[1:] + cycle[:1])
    )


@pytest.fixture(scope="module")
def states():
    signs = (1, -1, 1, -1, 1)
    profile = (1, -1, 1, 0, 0, 0, 0, 0, 0)
    product = tuple(
        Q(j, 9) + (Q(c, 5) if c < 5 else Q(0)) for c in range(6) for j in range(9)
    )
    changed = tuple(
        product[9 * c + j] + (Q(9, 25) * signs[c] * profile[j] if c < 5 else Q(0))
        for c in range(6)
        for j in range(9)
    )
    return product, changed


def test_marked_graph_has_more_cycle_information_than_fifteen_observations(geometry):
    edges = geometry.internal_edges + geometry.outer_edges
    graph = nx.Graph()
    graph.add_nodes_from(geometry.node_order)
    graph.add_edges_from((a, b) for a, b, _ in edges)
    assert sorted(map(len, nx.connected_components(graph))) == [9, 45]
    assert graph.number_of_edges() == 99
    assert len(nx.cycle_basis(graph)) == 47

    # Disjoint nonzero cycle vectors are linearly independent, not merely
    # fifteen labels for a potentially redundant collection of observations.
    covered = set()
    cycles = geometry.internal_cycles + geometry.outer_cycles
    assert len(cycles) == 15
    for cycle in cycles:
        cycle_edges = {frozenset((a, b)) for a, b in zip(cycle, cycle[1:] + cycle[:1])}
        assert cycle_edges and covered.isdisjoint(cycle_edges)
        covered.update(cycle_edges)
    assert covered == {frozenset((a, b)) for a, b, _ in edges}


def test_same_fifteen_windings_do_not_fix_mixed_cycle_class(geometry, states):
    product, changed = states
    for state in states:
        for a, b, _ in geometry.internal_edges + geometry.outer_edges:
            assert abs(_principal_turns(state[b] - state[a])) < Q(1, 2)
        assert (
            tuple(
                sum(_cycle_gaps(state, cycle))
                for cycle in geometry.internal_cycles + geometry.outer_cycles
            )
            == (Q(1),) * 15
        )

    square = (2, 11, 12, 3)
    assert _cycle_gaps(changed, square) == (
        Q(12, 25),
        Q(106, 225),
        Q(-1, 5),
        Q(56, 225),
    )
    assert sum(_cycle_gaps(changed, square)) == 1
    assert sum(_cycle_gaps(product, square)) == 0


def test_vertex_lifts_recentring_and_donor_rotation_preserve_marked_class(
    geometry, states
):
    _, changed = states
    shifted = tuple(
        value + Q((7 * node) % 11 - 5) for node, value in enumerate(changed)
    )
    mean = sum(shifted) / len(shifted)
    shifted = tuple(value - mean for value in shifted)
    donor_rotated = shifted[:45] + tuple(value + Q(7, 13) for value in shifted[45:])
    for state in (shifted, donor_rotated):
        for a, b, _ in geometry.internal_edges + geometry.outer_edges:
            assert _principal_turns(state[b] - state[a]) == _principal_turns(
                changed[b] - changed[a]
            )


def test_counterexample_is_not_a_stationary_target(geometry, states):
    _, changed = states
    node = 3
    neighbors = {
        b if a == node else a
        for a, b, _ in geometry.weighted_edges
        if a == node or b == node
    }
    gaps = {b: _principal_turns(changed[b] - changed[node]) for b in neighbors}
    assert gaps == {
        2: Q(56, 225),
        4: Q(1, 9),
        12: Q(1, 5),
        39: Q(-1, 5),
        48: Q(0),
    }
    # Opposite wheel sines cancel. Both remaining internal angles lie in
    # (0, pi), so the node's exact sine current is strictly positive.
    assert 0 < gaps[2] < Q(1, 2)
    assert 0 < gaps[4] < Q(1, 2)
