"""Independent geometric obstructions and the separate sine-balance boundary."""

from dataclasses import replace
from fractions import Fraction as F

import networkx as nx
import pytest

from tnfr.physics.phase_cycle_geometry import (
    assess_acute_cycle_periods,
    derive_phase_cycle_geometry,
    reconstruct_phase_cycle_state,
)
from tnfr.sdk import export_to_json
from tnfr.utils.io import json_loads


def _chain(geometry, cycle):
    row = [0] * len(geometry.edges)
    for i, j in zip(cycle, cycle[1:] + cycle[:1]):
        row[geometry.edges.index((min(i, j), max(i, j)))] += 1 if i < j else -1
    return tuple(row)


def _coordinates(geometry, chain):
    # Fundamental chord columns form the identity; no numerical inverse.
    return tuple(
        chain[i]
        for i in range(len(geometry.edges))
        if i not in geometry.tree_edge_indices
    )


def test_joint_cycle_obstruction_survives_individually_admitted_period_bounds():
    # Three paths of lengths 3, 2, 2; the two fundamental cycles have length 5.
    graph = nx.Graph()
    graph.add_nodes_from(range(6))
    graph.add_edges_from(((0, 1), (0, 2), (2, 3), (1, 4), (3, 4), (1, 5), (3, 5)))
    geometry = derive_phase_cycle_geometry(graph)
    assert tuple(map(len, geometry.fundamental_cycles)) == (5, 5)
    for combination in ((1, 0), (0, 1)):
        report = assess_acute_cycle_periods(
            geometry, cycle_periods=(1, -1), cycle_combination=combination
        )
        assert report.strict_bound_margin == F(1, 4)
        assert report.status == "necessary_bound_passed"
        assert not report.obstruction_certified
    report = assess_acute_cycle_periods(
        geometry, cycle_periods=(1, -1), cycle_combination=(1, -1)
    )
    square = _chain(geometry, (3, 4, 1, 5))
    assert report.combined_edge_chain == square
    assert report.combined_period == 2
    assert report.strict_period_bound == 1
    assert report.strict_bound_margin == -1
    assert report.obstruction_certified and report.status == "obstructed"
    # A rational rescaling changes the witness size but never its verdict.
    scaled = assess_acute_cycle_periods(
        geometry, cycle_periods=(1, -1), cycle_combination=(F(-2, 3), F(2, 3))
    )
    assert scaled.combined_period == F(-2, 3) * report.combined_period
    assert scaled.strict_bound_margin == F(2, 3) * report.strict_bound_margin
    assert scaled.obstruction_certified


def test_prism_requires_equal_ring_periods_but_does_not_force_consensus():
    geometry = derive_phase_cycle_geometry(nx.circular_ladder_graph(5))
    assert geometry.short_cycle_span_rank == 5 < geometry.cycle_rank == 6
    assert not geometry.short_cycle_consensus_only
    left = _chain(geometry, tuple(range(5)))
    right = _chain(geometry, tuple(range(5, 10)))
    squares = tuple(
        _chain(geometry, (i, (i + 1) % 5, (i + 1) % 5 + 5, i + 5)) for i in range(5)
    )
    assert tuple(map(sum, zip(*squares))) == tuple(a - b for a, b in zip(left, right))
    assert all(sum(map(abs, row)) == 4 for row in squares)
    for handedness in (1, -1):
        phases = tuple(F(i, 5) for i in range(5)) + tuple(
            F(handedness * i, 5) % 1 for i in range(5)
        )
        gaps = tuple(
            (phases[j] - phases[i] + F(1, 2)) % 1 - F(1, 2) for i, j in geometry.edges
        )
        periods = tuple(
            sum((c * t for c, t in zip(row, gaps)), F(0)) for row in geometry.cycle_rows
        )
        assert all(k.denominator == 1 for k in periods)
        reports = tuple(
            assess_acute_cycle_periods(
                geometry,
                cycle_periods=tuple(int(k) for k in periods),
                cycle_combination=_coordinates(geometry, square),
            )
            for square in squares
        )
        assert sum(report.combined_period for report in reports) == 1 - handedness
        if handedness == -1:
            # Equality at the bound is excluded by STRICT acute admission.
            assert any(r.obstruction_certified for r in reports)
            assert any(r.strict_bound_margin == 0 for r in reports)
        else:
            assert not any(r.obstruction_certified for r in reports)
            state = reconstruct_phase_cycle_state(geometry, edge_turns=gaps)
            assert state.sine_balance_status == "proved_by_odd_cancellation"
            assert max(map(abs, state.edge_turns)) == F(1, 5)
            assert not hasattr(reports[0], "equilibrium_certified")


def test_physical_witness_is_independent_of_node_order_and_fundamental_basis():
    graph = nx.circular_ladder_graph(5)
    phases = {i: F(i, 5) for i in range(5)}
    phases.update({i + 5: -F(i, 5) for i in range(5)})
    results = []
    for order in (tuple(range(10)), (7, 4, 9, 2, 8, 0, 5, 1, 6, 3)):
        ordered = nx.Graph()
        ordered.add_nodes_from(order)
        ordered.add_edges_from(graph.edges())
        geometry = derive_phase_cycle_geometry(ordered)
        positions = {node: i for i, node in enumerate(order)}
        gaps = tuple(
            (phases[order[j]] - phases[order[i]] + F(1, 2)) % 1 - F(1, 2)
            for i, j in geometry.edges
        )
        periods = tuple(
            sum(c * t for c, t in zip(row, gaps)) for row in geometry.cycle_rows
        )
        assert all(k.denominator == 1 for k in periods)
        square = _chain(geometry, tuple(positions[node] for node in (1, 2, 7, 6)))
        report = assess_acute_cycle_periods(
            geometry,
            cycle_periods=tuple(int(k) for k in periods),
            cycle_combination=_coordinates(geometry, square),
        )
        results.append(
            (report.combined_period, report.strict_period_bound, report.status)
        )
    assert results == [(F(1), F(1), "obstructed")] * 2


def test_acute_phase_feasibility_does_not_prove_any_equilibrium_in_its_sector():
    # Two four-edge A->B paths and a direct A->B chord. Each long path minus
    # the chord has period +1. Exact geometry exists; sine balance cannot.
    graph = nx.Graph()
    graph.add_nodes_from(range(8))
    paths = ((0, 2, 3, 4, 1), (0, 5, 6, 7, 1))
    graph.add_edge(0, 1)
    phases = {0: F(0), 1: F(-1, 6)}
    for path in paths:
        graph.add_edges_from(zip(path, path[1:]))
        phases.update(
            {node: index * F(5, 24) for index, node in enumerate(path[1:-1], 1)}
        )
    geometry = derive_phase_cycle_geometry(graph)
    gaps = tuple(
        (phases[j] - phases[i] + F(1, 2)) % 1 - F(1, 2) for i, j in geometry.edges
    )
    state = reconstruct_phase_cycle_state(geometry, edge_turns=gaps)
    assert max(map(abs, gaps)) == F(5, 24) < F(1, 4)
    for path in paths:
        assert sum(c * t for c, t in zip(_chain(geometry, path), gaps)) == 1
    assert state.sine_balance_status == "unresolved"
    report = assess_acute_cycle_periods(
        geometry, cycle_periods=state.cycle_periods, cycle_combination=(1, 0)
    )
    assert report.status == "necessary_bound_passed"
    # At ANY acute equilibrium, degree-two balances and sine injectivity force
    # constant gaps on each long path. For chord angle phi in (-pi/2,0), their
    # angle (2*pi+phi)/4 exceeds 3*pi/8. The source sine sum is strictly greater
    # than 2*sin(3*pi/8)-1 > 0, an analytic obstruction independent of this state.
    s = pytest.importorskip("sympy")
    assert s.simplify(2 * s.sin(3 * s.pi / 8) - 1).is_positive
    source_current = 2 * s.sin(2 * s.pi * F(5, 24)) + s.sin(-s.pi / 3)
    assert s.simplify(source_current).is_positive


@pytest.mark.parametrize(
    "periods,combination",
    [
        ((True,), (1,)),
        ((1.0,), (1,)),
        ((F(1),), (1,)),
        ((1, 2), (1,)),
        ({1}, (1,)),
        ((1,), (True,)),
        ((1,), (float("nan"),)),
        ((1,), (float("inf"),)),
        ((1,), (1j,)),
        ((1,), (0,)),
        ((1,), ()),
        ((1,), (1, 2)),
        ((1,), {1}),
    ],
)
def test_period_or_witness_admission_rejects_invalid_evidence(periods, combination):
    geometry = derive_phase_cycle_geometry(nx.cycle_graph(5))
    with pytest.raises((TypeError, ValueError)):
        assess_acute_cycle_periods(
            geometry, cycle_periods=periods, cycle_combination=combination
        )


def test_revalidation_tree_boundary_and_exact_export(tmp_path):
    geometry = derive_phase_cycle_geometry(nx.cycle_graph(5))
    for forged in (
        replace(geometry, cycle_rank=2),
        replace(geometry, cycle_rows=((0,) * 5,)),
    ):
        with pytest.raises(ValueError, match="derived fields"):
            assess_acute_cycle_periods(
                forged, cycle_periods=(1,), cycle_combination=(1,)
            )
    with pytest.raises(ValueError):
        assess_acute_cycle_periods(
            derive_phase_cycle_geometry(nx.path_graph(3)),
            cycle_periods=(),
            cycle_combination=(),
        )
    report = assess_acute_cycle_periods(
        geometry, cycle_periods=(1,), cycle_combination=(F(3, 7),)
    )
    assert report.strict_period_bound == F(15, 28)
    assert report.strict_bound_margin == F(3, 28)
    payload = report.to_dict()
    assert payload["schema"] == "tnfr.acute-cycle-period-assessment.v1"
    path = tmp_path / "periods.json"
    export_to_json(payload, path)
    assert json_loads(path.read_text(encoding="utf-8")) == payload
