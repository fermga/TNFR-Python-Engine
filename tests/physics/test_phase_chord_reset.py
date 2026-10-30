"""Integer cycle resets distinguish support birth, phase writes and coordinates."""

from dataclasses import replace
from fractions import Fraction as F

import networkx as nx
import pytest

from tests.joint_phase_helpers import exact_phase_cycle_state as _state
from tnfr.physics._exact_linear_algebra import exact_matrix_inverse
from tnfr.physics.phase_cycle_geometry import (
    derive_phase_chord_extension,
    derive_phase_cycle_geometry,
    observe_phase_chord_reset,
)


def _extended(graph, edge):
    result = graph.copy()
    result.add_edge(*edge)
    return result


def _row(geometry, cycle):
    result = [0] * len(geometry.edges)
    for i, j in zip(cycle, cycle[1:] + cycle[:1]):
        result[geometry.edges.index((min(i, j), max(i, j)))] += 1 if i < j else -1
    return tuple(result)


@pytest.mark.parametrize(
    "graph,edge",
    [
        (nx.path_graph(5), (0, 4)),
        (nx.cycle_graph(6), (0, 3)),
        (nx.barbell_graph(3, 0), (1, 4)),
    ],
)
def test_integer_basis_maps_preserve_inherited_cycles_and_append_one_chord(graph, edge):
    before = derive_phase_cycle_geometry(graph)
    after = derive_phase_cycle_geometry(_extended(graph, edge))
    extension = derive_phase_chord_extension(before, after)
    assert after.cycle_rank == before.cycle_rank + 1
    assert after.edges[extension.added_edge_index] == tuple(sorted(edge))
    assert (
        tuple(after.edges[i] for i in extension.inherited_edge_indices) == before.edges
    )
    adapted_rows = []
    for cycle in before.fundamental_cycles:
        adapted_rows.append(_row(after, cycle))
    adapted_rows.append(_row(after, extension.created_cycle))
    to_after = extension.inherited_cycle_coordinates + (
        extension.created_cycle_coordinates,
    )
    for coordinates, row in zip(to_after, adapted_rows, strict=True):
        assert (
            tuple(
                sum(c * basis[e] for c, basis in zip(coordinates, after.cycle_rows))
                for e in range(len(after.edges))
            )
            == row
        )
    for coordinates, row in zip(
        extension.after_cycle_coordinates, after.cycle_rows, strict=True
    ):
        assert (
            tuple(
                sum(c * basis[e] for c, basis in zip(coordinates, adapted_rows))
                for e in range(len(after.edges))
            )
            == row
        )
    # Independent rational elimination verifies the integer inverse map.
    inverse = exact_matrix_inverse(tuple(tuple(F(c) for c in row) for row in to_after))
    assert inverse == extension.after_cycle_coordinates
    assert all(value.denominator == 1 for row in inverse for value in row)


@pytest.mark.parametrize("orientation", [1, -1])
def test_path_closure_creates_signed_winding_without_changing_phase(orientation):
    path = nx.path_graph(5)
    phases = {i: F(orientation * i, 5) for i in path}
    before, after = _state(path, phases), _state(_extended(path, (0, 4)), phases)
    reset = observe_phase_chord_reset(before, after)
    assert reset.inherited_periods_before == reset.inherited_periods_after == ()
    assert reset.inherited_period_changes == ()
    # Owner orientation is new edge 0->4, then the tree return path 4->...->0.
    assert reset.extension.created_cycle == (0, 4, 3, 2, 1)
    assert reset.created_edge_turn_before_phase == F(-orientation, 5)
    assert (
        reset.created_period_before_phase == reset.created_period_after == -orientation
    )
    assert reset.created_period_phase_change == 0
    assert reset.phase_unchanged_modulo_rotation
    assert after.sine_balance_status == "proved_by_odd_cancellation"
    assert before.sine_balance_status == "unresolved"


def test_phase_writes_can_preserve_or_change_new_period_without_being_hidden():
    path = nx.path_graph(5)
    ring = _extended(path, (0, 4))
    phases = {i: F(i, 5) for i in path}
    before = _state(path, phases)
    shifted = dict(phases)
    shifted[0] += F(1, 100)
    same_sector = observe_phase_chord_reset(before, _state(ring, shifted))
    assert not same_sector.phase_unchanged_modulo_rotation
    assert same_sector.created_period_after == -1
    assert same_sector.created_period_phase_change == 0
    assert same_sector.after.sine_balance_status == "unresolved"
    changed = observe_phase_chord_reset(before, _state(ring, {i: 0 for i in ring}))
    assert changed.created_period_before_phase == -1
    assert changed.created_period_after == 0
    assert changed.created_period_phase_change == 1
    # These are endpoint differences; neither call supplies a continuous path.
    assert not changed.phase_unchanged_modulo_rotation


def test_an_existing_sector_is_preserved_while_an_independent_cycle_is_added():
    graph = nx.cycle_graph(5)
    graph.add_edges_from([(0, 5), (5, 6), (6, 7), (7, 8)])
    phases = {i: F(i, 5) for i in range(5)}
    phases.update({i: F(i - 4, 5) for i in range(5, 9)})
    reset = observe_phase_chord_reset(
        _state(graph, phases), _state(_extended(graph, (0, 8)), phases)
    )
    assert reset.inherited_periods_before == reset.inherited_periods_after
    assert tuple(map(abs, reset.inherited_periods_before)) == (1,)
    assert reset.inherited_period_changes == (0,)
    assert reset.created_period_after == -1
    assert reset.after.geometry.cycle_rank == 2
    adapted = reset.inherited_periods_after + (reset.created_period_after,)
    assert (
        tuple(
            sum(a * p for a, p in zip(row, adapted))
            for row in reset.extension.after_cycle_coordinates
        )
        == reset.after.cycle_periods
    )


def test_phase_writes_can_change_old_periods_and_antipodal_counterfactual_abstains():
    graph = nx.cycle_graph(6)
    reset = observe_phase_chord_reset(
        _state(graph, {i: F(i, 6) for i in graph}),
        _state(_extended(graph, (0, 3)), {i: 0 for i in graph}),
    )
    assert reset.inherited_period_changes == tuple(
        -w for w in reset.inherited_periods_before
    )
    assert reset.inherited_periods_after == (0,)
    assert reset.created_period_after == 0
    assert reset.created_edge_turn_before_phase is None
    assert reset.created_period_before_phase is None
    assert reset.created_period_phase_change is None
    assert reset.after.geometry.short_cycle_consensus_only


def test_nonacute_old_phase_chord_is_only_a_counterfactual_coordinate():
    graph = nx.cycle_graph(10)
    phases = {i: F(i, 10) for i in graph}
    reset = observe_phase_chord_reset(
        _state(graph, phases), _state(_extended(graph, (0, 4)), {i: 0 for i in graph})
    )
    assert reset.created_edge_turn_before_phase == F(2, 5) > F(1, 4)
    assert reset.created_period_before_phase == 0
    # A different return path differs by the old unit-winding cycle. The
    # retained-tree 'born period' is not a path-independent new charge.
    alternate_cycle = (0, 4, 5, 6, 7, 8, 9)
    alternate_period = sum(
        (phases[j] - phases[i] + F(1, 2)) % 1 - F(1, 2)
        for i, j in zip(alternate_cycle, alternate_cycle[1:] + alternate_cycle[:1])
    )
    assert alternate_period == 1
    assert abs(alternate_period - reset.created_period_before_phase) == abs(
        reset.inherited_periods_before[0]
    )


def test_common_phase_rotation_and_relabeling_preserve_the_declared_reset():
    graph = nx.path_graph(5)
    ring = _extended(graph, (0, 4))
    phases = {i: F(i, 5) for i in graph}
    reset = observe_phase_chord_reset(
        _state(graph, phases), _state(ring, {i: p + F(2, 7) for i, p in phases.items()})
    )
    assert reset.phase_unchanged_modulo_rotation
    labels = {i: ("node", 5 - i) for i in graph}
    renamed = observe_phase_chord_reset(
        _state(
            nx.relabel_nodes(graph, labels), {labels[i]: p for i, p in phases.items()}
        ),
        _state(
            nx.relabel_nodes(ring, labels), {labels[i]: p for i, p in phases.items()}
        ),
    )
    assert renamed.created_period_after == reset.created_period_after
    assert (
        renamed.extension.inherited_edge_indices
        == reset.extension.inherited_edge_indices
    )


@pytest.mark.parametrize(
    "change", ["none", "remove", "replace", "two", "reorder", "extra_node"]
)
def test_unsupported_support_changes_are_not_silently_single_chord_events(change):
    graph = nx.path_graph(5)
    after = graph.copy()
    if change == "remove":
        graph = nx.cycle_graph(5)
    elif change == "replace":
        after.remove_edge(1, 2)
        after.add_edge(0, 4)
    elif change == "two":
        after.add_edges_from([(0, 4), (0, 3)])
    elif change == "reorder":
        after = nx.Graph()
        after.add_nodes_from(reversed(list(graph)))
        after.add_edges_from([*graph.edges, (0, 4)])
    elif change == "extra_node":
        after.add_edge(4, 5)
    with pytest.raises(ValueError):
        derive_phase_chord_extension(
            derive_phase_cycle_geometry(graph), derive_phase_cycle_geometry(after)
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("cycle_periods", (99,)),
        ("nodal_turns", (F(0),) * 5),
        ("sine_balance_status", "forged"),
    ],
)
def test_state_caches_are_rebuilt_before_observing_a_reset(field, value):
    graph = nx.path_graph(5)
    phases = {i: F(i, 5) for i in graph}
    before = _state(graph, phases)
    after = _state(_extended(graph, (0, 4)), phases)
    with pytest.raises(ValueError, match="match|rebuild|differ|primitive"):
        observe_phase_chord_reset(replace(before, **{field: value}), after)


def test_geometry_caches_cannot_forge_a_basis_map():
    graph = nx.path_graph(5)
    before = derive_phase_cycle_geometry(graph)
    after = derive_phase_cycle_geometry(_extended(graph, (0, 4)))
    with pytest.raises(ValueError, match="match|rebuild|differ|primitive"):
        derive_phase_chord_extension(before, replace(after, cycle_rank=99))
