"""The retained prism with strict all-edge U3 excludes winding sectors.

Angles in the detached controls are exact multiples of mathematical pi;
no rounded angle proves a circulation identity. These tests constrain a
fixture domain, not TNFR in general or operator admission: UM/RA may admit
only a compatible subset, whereas the premise here covers every support
edge. No graph state is changed and no evolution or new phase law is run.
"""

from fractions import Fraction as Q
from itertools import combinations

import networkx as nx
import pytest

from tests.physics._internal_mode_fixture import NODES, _graph
from tnfr.physics.coupling_winding import observe_coupling_gap_step


def _wrap_pi_units(value):
    """Shortest displacement in [-1,1), in units of exact pi."""
    return (value + 1) % 2 - 1


@pytest.fixture(scope="module")
def prism():
    s = pytest.importorskip("sympy")
    graph = _graph()
    assert tuple(graph) == NODES
    index = {node: i for i, node in enumerate(NODES)}
    support = nx.relabel_nodes(graph, index)
    edges = tuple(sorted(tuple(sorted(edge)) for edge in support.edges))
    incidence = s.Matrix(
        6, len(edges), lambda i, j: int(i == edges[j][1]) - int(i == edges[j][0])
    )
    cycles = ((0, 1, 2), (3, 4, 5), (0, 1, 4, 3), (1, 2, 5, 4))
    cycle_rows = []
    for cycle in cycles:
        row = [0] * len(edges)
        for left, right in zip(cycle, cycle[1:] + cycle[:1], strict=True):
            edge = tuple(sorted((left, right)))
            assert edge in edges
            row[edges.index(edge)] += 1 if left < right else -1
        cycle_rows.append(row)
    return s, support, edges, incidence, cycles, s.Matrix(cycle_rows)


def test_two_triangles_and_two_squares_span_the_entire_prism_cycle_space(prism):
    s, support, edges, incidence, cycles, circulation = prism
    assert len(edges) - len(support) + 1 == 4
    assert incidence.rank() == 5 and circulation.rank() == 4
    assert incidence * circulation.T == s.zeros(6, 4)
    assert tuple(map(len, cycles)) == (3, 3, 4, 4)
    assert nx.diameter(support) == 2
    # Every closed circular phase cycle sums to 2*pi times an integer.
    # Strict |edge gap|<pi/2 gives |cycle sum|<length*pi/2<=2*pi.
    # The smallest nonzero integer circulation is 2*pi, so all four
    # independent basis circulations must be zero. This uses a strict
    # bound, including on the four-edge cycles, not a floating tolerance.
    smallest_nonzero_integer_circulation = Q(2)  # In exact pi units.
    for cycle, row in zip(cycles, circulation.tolist(), strict=True):
        bound = sum(abs(value) for value in row) * Q(1, 2)
        assert bound == Q(len(cycle), 2)
        assert bound <= smallest_nonzero_integer_circulation


def test_zero_cycle_circulation_gives_a_common_open_semicircle_on_this_support(prism):
    s, support, edges, incidence, _, circulation = prism
    tree_edges = ((0, 1), (0, 2), (0, 3), (1, 4), (2, 5))
    tree_columns = [edges.index(edge) for edge in tree_edges]
    chord_columns = [j for j in range(len(edges)) if j not in tree_columns]
    cycle_chords = circulation[:, chord_columns]
    assert abs(cycle_chords.det()) == 1
    tree_gaps = s.Matrix(s.symbols("d01 d02 d03 d14 d25", real=True))
    chord_gaps = -cycle_chords.inv() * circulation[:, tree_columns] * tree_gaps
    gap = s.zeros(len(edges), 1)
    for column, value in zip(tree_columns, tree_gaps, strict=True):
        gap[column] = value
    for column, value in zip(chord_columns, chord_gaps, strict=True):
        gap[column] = value
    assert circulation * gap == s.zeros(4, 1)
    d01, d02, d03, d14, d25 = tree_gaps
    lift = s.Matrix((0, d01, d02, d03, d01 + d14, d02 + d25))
    assert (incidence.T * lift - gap).applyfunc(s.expand) == s.zeros(len(edges), 1)

    # This constructs the global real lift rather than assuming a global
    # chart at the outset. Each pair has a path of at most two edges; if
    # every |gap|<1/2, telescoping and the triangle inequality give a
    # strict |lift[j]-lift[i]|<1. The maximum pair difference is therefore
    # <pi in physical angles: one common open semicircle contains all nodes.
    for left, right in combinations(support, 2):
        path = nx.shortest_path(support, left, right)
        path_sum = 0
        for start, end in zip(path, path[1:]):
            edge = tuple(sorted((start, end)))
            path_sum += (1 if start < end else -1) * gap[edges.index(edge)]
        assert s.expand(path_sum - lift[right] + lift[left]) == 0
        assert Q(len(path) - 1, 2) <= 1


def test_strictness_is_essential_and_semicircle_does_not_mean_half_pi_width(prism):
    s, _, edges, _, _, circulation = prism
    # A detached domain countercontrol, not a new prepared runtime graph.
    strict_phases = (Q(0), Q(0), Q(3, 8), Q(3, 8), Q(3, 8), Q(3, 4))
    strict_gaps = s.Matrix(
        [_wrap_pi_units(strict_phases[j] - strict_phases[i]) for i, j in edges]
    )
    assert max(abs(value) for value in strict_gaps) == Q(3, 8) < Q(1, 2)
    assert circulation * strict_gaps == s.zeros(4, 1)
    assert Q(1, 2) < max(strict_phases) - min(strict_phases) == Q(3, 4) < 1
    distinct = sorted(set(strict_phases))
    empty_arcs = [
        right - left for left, right in zip(distinct, distinct[1:] + [distinct[0] + 2])
    ]
    assert 2 - max(empty_arcs) == Q(3, 4)

    # Inclusive |gap|<=pi/2 permits square winding. This does not assert
    # regularity of every phasor mean or admission of a particular word.
    boundary_phases = (Q(0), Q(1, 2), Q(0), Q(3, 2), Q(1), Q(3, 2))
    boundary_gaps = s.Matrix(
        [_wrap_pi_units(boundary_phases[j] - boundary_phases[i]) for i, j in edges]
    )
    assert max(abs(value) for value in boundary_gaps) == Q(1, 2)
    assert circulation * boundary_gaps / 2 == s.Matrix((0, 0, 1, -1))


def test_existing_uniform_cycle_twist_is_a_positive_sector_outside_this_fixture():
    # Reuse the existing target-only cycle companion on normalized gap
    # coordinates. Its map is linear: multiplying the fixed gap vector by
    # exact pi proves the same identity for physical gaps pi/4. The numeric
    # call is an algebraic companion, not a live U3/winding certificate.
    phase_pi_units = tuple(Q(i, 4) for i in range(8))
    gaps = tuple(
        _wrap_pi_units(right - left)
        for left, right in zip(
            phase_pi_units, phase_pi_units[1:] + phase_pi_units[:1], strict=True
        )
    )
    assert gaps == (Q(1, 4),) * 8
    assert sum(gaps) / 2 == 1 and max(gaps) < Q(1, 2)
    comparison = observe_coupling_gap_step(gaps, phase_gate=Q(1, 2))
    assert comparison.output_gaps == gaps
    assert comparison.spread_before == comparison.spread_after == 0
    assert comparison.sum_residual == 0
    # Equal adjacent gaps make canonical midpoint phase pressure zero.
    # With uniform EPI/capacity and cycle degree, all four pressure channels
    # vanish. This preserves a prepared pattern within that restricted map;
    # it does not prove formation, localized EPI, or a full-runtime law.
    assert all(gaps[i] - gaps[i - 1] == 0 for i in range(len(gaps)))
