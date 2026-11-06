"""Target-free whole-sector barriers, independent bounds and abstention controls."""

from dataclasses import replace
from fractions import Fraction as Q

import mpmath as mp
import networkx as nx
import pytest

from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.mathematics._rational_interval import I
from tnfr.physics._sine_sector_boundary import _sector_boundary_bounds
from tnfr.physics.phase_cycle_geometry import derive_phase_cycle_geometry
from tnfr.physics.relational_sine_comparison import bound_relational_sine_exchange
from tnfr.physics.relational_sine_forecast import SineForecast
from tnfr.physics.relational_sine_pattern import (
    SineRelativeForecast,
    bound_relational_sine_pattern,
)
from tnfr.physics.relational_sine_recovery import certify_sine_sector_capture
from tnfr.sdk import export_to_json
from tnfr.utils.io import json_loads

MODEL = RelationalExchangeModel(1, phase_domain="regular")


def _mp(value):
    value = Q(value)
    return mp.mpf(value.numerator) / value.denominator


def _graph(*, pair=False, return_edge=False):
    graph = nx.cycle_graph(5)
    if pair:
        graph = nx.disjoint_union(graph, nx.cycle_graph(5))
        graph.add_edges_from(((0, 10), (5, 10)))
        if return_edge:
            graph.add_edge(1, 6)
    for node in graph:
        graph.nodes[node].update(
            EPI=Q(1, 1024) if node == 0 else 0,
            theta=Q(5, 4) * (node % 5) if node != 10 else 0,
            nu_f=Q(2) if node == 1 else Q(1),
        )
    graph.graph["GAMMA"] = {"type": "none"}
    return graph


def _offsets(graph):
    geometry = derive_phase_cycle_geometry(graph)
    return tuple(-1 if edge in ((0, 4), (5, 9)) else 0 for edge in geometry.edges)


def _source(graph=None, *, model=MODEL):
    return bound_relational_sine_exchange(
        _graph() if graph is None else graph, reference_model=model
    )


def _pattern(graph=None, *, error=Q(1, 65536)):
    graph = _graph() if graph is None else graph
    return bound_relational_sine_pattern(
        graph,
        reference_node=0,
        reference_model=MODEL,
        form_error_bounds=(error,) * len(graph),
        phase_error_bounds=(error,) * len(graph),
    )


def _forecast_source():
    # Interface-only fixture, not independently authenticated trajectory data.
    pattern = _pattern(error=Q(0))
    endpoint = tuple(I(v) for v in pattern.nominal_form + pattern.nominal_phase) + (
        I(pattern.capacity[-1]),
    )
    full = SineForecast(
        model=MODEL,
        neighbors=pattern.neighbors,
        visible_capacity=pattern.capacity[:-1],
        initial_box=endpoint,
        observation_time=Q(0),
        end_time=Q(2),
        time_step=Q(1, 4),
        order=4,
        steps=(),
        validated_end_time=Q(1, 4),
        endpoint=endpoint,
        failed_tube=None,
        status="unavailable",
        reasons=("fixture",),
    )
    return SineRelativeForecast(pattern, full, (), (), I(0), I(0))


def test_cycle_boundary_matches_independent_jensen_minimum_on_every_feasible_face():
    geometry = derive_phase_cycle_geometry(nx.cycle_graph(5))
    bounds = _sector_boundary_bounds(geometry, (1,))
    assert len(bounds) == 10
    with mp.workdps(90):
        exact = 1 + 4 * (1 - mp.cos(3 * mp.pi / 8))
        for edge, orientation in enumerate(geometry.cycle_rows[0]):
            # Other sign forces the remaining four gaps past their closed
            # limits and has an empty face. On the feasible face Jensen is exact.
            face = 2 * edge + int(orientation == 1)
            assert 0 <= exact - _mp(bounds[face]) < mp.mpf("1e-30")
        assert _mp(min(bounds)) > 5 * (1 - mp.cos(2 * mp.pi / 5))
        # Independent non-minimal boundary point with the same period.
        turns = (Q(1, 4), Q(1, 8), Q(5, 24), Q(5, 24), Q(5, 24))
        assert sum(turns) == 1
        energy = sum(1 - mp.cos(2 * mp.pi * _mp(t)) for t in turns)
        assert energy > exact >= _mp(min(bounds))
    tree = derive_phase_cycle_geometry(nx.path_graph(4))
    assert _sector_boundary_bounds(tree, ()) == (Q(1),) * 6


@pytest.mark.parametrize("pair", (False, True))
def test_moving_nonsymmetric_source_is_captured_without_any_equilibrium_target(pair):
    graph = _graph(pair=pair)
    source = _source(graph)
    result = certify_sine_sector_capture(source, edge_turn_offsets=_offsets(graph))
    assert result.admitted and result.energy_margin > 0
    assert any(bound.abs_max > 0 for bound in source.phase_rates)
    assert not hasattr(result, "target_phase_turns")
    assert not hasattr(result, "radius")
    assert len(result.boundary_face_lower_bounds) == 2 * graph.number_of_edges()
    assert result.cycle_periods == (1,) * (2 if pair else 1)
    assert result.hypothesis_failures == result.unresolved_conditions == ()
    weights = tuple(Q(graph.degree[i]) / graph.nodes[i]["nu_f"] for i in graph)
    mean = sum(w * graph.nodes[i]["EPI"] for i, w in enumerate(weights)) / sum(weights)
    assert result.weighted_form_mean == mean
    with mp.workdps(90):
        expected = sum(
            (_mp(source.epi[j]) - _mp(source.epi[i])) ** 2 / 2
            + 1
            - mp.cos(_mp(source.phase[j]) - _mp(source.phase[i]))
            for i, j in graph.edges
        )
        assert (
            _mp(result.storage_bounds.lo) <= expected <= _mp(result.storage_bounds.hi)
        )
    # Cached work and storage are not admission evidence.
    forged = replace(source, storage=I(10**6), form_storage=Q(-1), phase_storage=I(-1))
    again = certify_sine_sector_capture(forged, edge_turn_offsets=_offsets(graph))
    assert again.storage_bounds == result.storage_bounds and again.admitted


def test_uncertain_observations_keep_common_origins_unknown_and_cover_source_corners():
    graph = _graph(pair=True)
    source = _pattern(graph)
    report = source.certify_sector_capture(edge_turn_offsets=_offsets(graph))
    assert report.admitted
    assert report.weighted_form_mean is report.weighted_phase_mean is None
    with mp.workdps(90):
        for sign in (-1, 1):
            x = tuple(
                _mp(v) + 7 + sign * (-1) ** i * _mp(source.form_error_bounds[i])
                for i, v in enumerate(source.nominal_form)
            )
            theta = tuple(
                _mp(v) - 3 - sign * (-1) ** i * _mp(source.phase_error_bounds[i])
                for i, v in enumerate(source.nominal_phase)
            )
            energy = sum(
                (x[j] - x[i]) ** 2 / 2 + 1 - mp.cos(theta[j] - theta[i])
                for i, j in graph.edges
            )
            assert (
                energy
                <= _mp(report.storage_upper_bound)
                < _mp(report.boundary_storage_lower_bound)
            )
    broad = replace(source, phase_error_bounds=(Q(1),) * len(graph))
    failed = broad.certify_sector_capture(edge_turn_offsets=_offsets(graph))
    assert not failed.admitted
    assert (
        "whole_source_set_strict_acute_sector_not_certified"
        in failed.unresolved_conditions
    )


def test_feasible_sector_without_sine_equilibrium_never_receives_capture():
    graph = nx.Graph()
    graph.add_nodes_from(range(8))
    graph.add_edge(0, 1)
    turns = {0: Q(0), 1: Q(-1, 6)}
    for path in ((0, 2, 3, 4, 1), (0, 5, 6, 7, 1)):
        graph.add_edges_from(zip(path, path[1:]))
        turns.update({node: j * Q(5, 24) for j, node in enumerate(path[1:-1], 1)})
    for node in graph:
        graph.nodes[node].update(
            EPI=0, theta=float(2 * mp.pi * _mp(turns[node])), nu_f=1
        )
    geometry = derive_phase_cycle_geometry(graph)
    offsets = tuple(
        -int((turns[j] - turns[i] + Q(1, 2)) // 1) for i, j in geometry.edges
    )
    report = certify_sine_sector_capture(_source(graph), edge_turn_offsets=offsets)
    assert all(value.lo > 0 for value in report.edge_acute_margin_bounds)
    assert not report.admitted and report.energy_margin < 0
    assert report.hypothesis_failures == ()


def test_conservative_bound_can_be_unavailable_despite_an_existing_equilibrium():
    # Same-handed rings with a return edge have a known acute sine equilibrium.
    # This bound's deterministic tangent hints are not a complete optimizer.
    graph = _graph(pair=True, return_edge=True)
    report = certify_sine_sector_capture(
        _source(graph), edge_turn_offsets=_offsets(graph)
    )
    assert all(value.lo > 0 for value in report.edge_acute_margin_bounds)
    assert not report.admitted and report.hypothesis_failures == ()
    assert (
        "strict_total_storage_boundary_barrier_not_certified"
        in report.unresolved_conditions
    )


def test_cycle_closing_triangle_preserves_joint_capture_with_all_incident_edges():
    graph = _graph(pair=True)
    graph.add_edge(0, 5)
    source = _source(graph)
    report = certify_sine_sector_capture(source, edge_turn_offsets=_offsets(graph))
    assert report.geometry.cycle_rank == 3
    assert report.cycle_periods == (1, 0, 1)
    assert len(report.boundary_face_lower_bounds) == 26
    assert report.admitted and report.energy_margin > 0
    # The altered degrees change the conserved mean even though the added
    # connection has zero initial phase and form differences only cost storage.
    weights = tuple(Q(graph.degree[i]) / graph.nodes[i]["nu_f"] for i in graph)
    expected = sum(w * source.epi[i] for i, w in enumerate(weights)) / sum(weights)
    assert report.weighted_form_mean == expected


def test_lossless_and_frozen_node_premises_cannot_certify_convergence():
    graph = _graph()
    lossless = RelationalExchangeModel(
        1, epi_weight=0, phase_weight=1, phase_domain="regular"
    )
    result = certify_sine_sector_capture(
        _source(graph, model=lossless), edge_turn_offsets=_offsets(graph)
    )
    assert (
        not result.admitted
        and "positive_epi_weight_required" in result.hypothesis_failures
    )
    graph.nodes[2]["nu_f"] = 0
    result = certify_sine_sector_capture(
        _source(graph), edge_turn_offsets=_offsets(graph)
    )
    assert (
        not result.admitted
        and "strictly_positive_held_capacity_required" in result.hypothesis_failures
    )
    graph.nodes[2]["nu_f"] = Q(1, 2**200)
    result = certify_sine_sector_capture(
        _source(graph), edge_turn_offsets=_offsets(graph)
    )
    assert result.admitted and result.capacity_bounds[2].lo == 0


@pytest.mark.parametrize(
    "field,value",
    [
        ("capacity", (True, 1, 1, 1, 1)),
        ("phase", (float("nan"), 0, 0, 0, 0)),
        ("epi", (0, 0)),
        ("degrees", (3, 2, 2, 2, 2)),
        ("edges", ((0, 1),) * 5),
        ("law", "another_law"),
    ],
)
def test_invalid_consumed_primitives_cannot_be_certified(field, value):
    with pytest.raises((TypeError, ValueError)):
        certify_sine_sector_capture(
            replace(_source(), **{field: value}), edge_turn_offsets=_offsets(_graph())
        )


@pytest.mark.parametrize("offsets", [(0,) * 4, (False,) * 5, (0.0,) * 5, {0}])
def test_edge_chart_requires_exact_ordered_integer_offsets(offsets):
    with pytest.raises((TypeError, ValueError)):
        certify_sine_sector_capture(_source(), edge_turn_offsets=offsets)


def test_missing_wrap_and_large_form_energy_remain_unavailable():
    result = certify_sine_sector_capture(_source(), edge_turn_offsets=(0,) * 5)
    assert not result.admitted
    graph = _graph()
    graph.nodes[0]["EPI"] = 10
    result = certify_sine_sector_capture(
        _source(graph), edge_turn_offsets=_offsets(graph)
    )
    assert not result.admitted and result.energy_margin < 0


def test_exact_report_export_does_not_install_a_target_or_execute_a_graph(tmp_path):
    graph = _graph()
    before = nx.node_link_data(graph, edges="edges")
    report = certify_sine_sector_capture(
        _source(graph), edge_turn_offsets=_offsets(graph)
    )
    assert nx.node_link_data(graph, edges="edges") == before
    payload = report.to_dict()
    assert payload["schema"] == "tnfr.relational-sine-sector-capture.v1"
    path = tmp_path / "sector.json"
    export_to_json(payload, path)
    assert json_loads(path.read_text(encoding="utf-8")) == payload


def test_forecast_reader_uses_full_endpoint_and_actual_time_without_solver_replay():
    source = _forecast_source()
    full = source.full_forecast
    report = source.certify_sector_capture(edge_turn_offsets=_offsets(_graph()))
    assert report.admitted and report.input_forecast_admitted is False
    assert report.observation_time == Q(1, 4) < report.input_forecast_requested_end_time
    enlarged = list(full.endpoint)
    enlarged[0] = I(-10, 10)
    failed = replace(source, full_forecast=replace(full, endpoint=tuple(enlarged)))
    assert not failed.certify_sector_capture(
        edge_turn_offsets=_offsets(_graph())
    ).admitted
    with pytest.raises(ValueError):
        replace(
            source, full_forecast=replace(full, freeze_hidden=True)
        ).certify_sector_capture(edge_turn_offsets=_offsets(_graph()))
    with pytest.raises(ValueError):
        replace(
            source, full_forecast=replace(full, validated_end_time=Q(3))
        ).certify_sector_capture(edge_turn_offsets=_offsets(_graph()))


@pytest.mark.parametrize(
    "change",
    ("visible", "pattern_visible", "pattern_final", "initial", "layout", "endpoint"),
)
def test_forecast_capacity_declarations_must_describe_the_same_held_law(change):
    source = _forecast_source()
    pattern, full = source.pattern, source.full_forecast
    if change == "visible":
        full = replace(full, visible_capacity=(Q(3),) + full.visible_capacity[1:])
    elif change == "pattern_visible":
        pattern = replace(pattern, capacity=(Q(3),) + pattern.capacity[1:])
    elif change == "pattern_final":
        pattern = replace(pattern, capacity=pattern.capacity[:-1] + (Q(2),))
    elif change == "initial":
        full = replace(full, initial_box=full.initial_box[:-1] + (I(0, 2),))
    elif change == "layout":
        full = replace(full, initial_box=full.initial_box[:-1])
    else:
        full = replace(full, endpoint=full.endpoint[:-1] + (I(2),))
    with pytest.raises(ValueError, match="must match|must contain"):
        replace(source, pattern=pattern, full_forecast=full).certify_sector_capture(
            edge_turn_offsets=_offsets(_graph())
        )


def test_forecast_capacity_enclosure_can_widen_without_changing_its_held_value():
    source = _forecast_source()
    full = source.full_forecast
    wide = I(1, 2)
    source = replace(
        source, full_forecast=replace(full, endpoint=full.endpoint[:-1] + (wide,))
    )
    report = source.certify_sector_capture(edge_turn_offsets=_offsets(_graph()))
    assert report.admitted
    assert report.capacity_bounds[-1] == wide
