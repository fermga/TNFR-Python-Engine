"""Independent geometry and analytic controls for the storage-handoff obstruction."""

from fractions import Fraction as Q
from inspect import signature
from typing import get_type_hints

import mpmath
import networkx as nx
import numpy as np
import pytest

from tnfr.physics import relational_sine_two_port_compatibility as owner
from tnfr.sdk import export_to_json, relational_report_to_dict
from tnfr.utils.io import json_loads


@pytest.fixture(autouse=True)
def no_compatibility_root_or_transcendental_evaluation(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("the analytic obstruction must use exact rational work only")

    for name in (
        "assess_sine_two_port_compatibility",
        "_root_enclosures",
        "_enclose_decreasing_phase_root",
        "sin",
        "cos",
        "pi_interval",
    ):
        monkeypatch.setattr(owner, name, forbidden)


@pytest.fixture(scope="module")
def mp():
    context = mpmath.mp.clone()
    context.dps = 85
    return context


def _mp(mp, value):
    value = Q(value)
    return mp.mpf(value.numerator) / value.denominator


def _assess(radius=Q(1, 65536)):
    return owner.assess_sine_two_port_handoff_obstruction(phase_error_radius=radius)


def _graph():
    graph = nx.disjoint_union(nx.cycle_graph(9), nx.cycle_graph(9))
    graph.add_edges_from(((0, 9), (1, 10)))
    return graph


def _storage(mp, graph, phase, form=None):
    if form is None:
        form = (mp.mpf(0),) * len(graph)
    return sum(
        (form[j] - form[i]) ** 2 / 2 + 1 - mp.cos(phase[j] - phase[i])
        for i, j in graph.edges
    )


def _witness_phases(mp, report):
    return tuple(2 * mp.pi * _mp(mp, value) for value in report.witness_nodal_turns)


def test_witness_is_a_full_support_boundary_state_with_the_actual_degree_gauge():
    report = _assess()
    graph = _graph()
    assert report.geometry.nodes == tuple(graph)
    assert report.geometry.edges == tuple(sorted(tuple(sorted(e)) for e in graph.edges))
    assert report.geometry.cycle_rank == graph.number_of_edges() - len(graph) + 1 == 3
    assert report.degrees == tuple(graph.degree[i] for i in graph)
    assert report.invariant_weights == report.degrees
    assert report.weighted_coordinate_mass == 40
    assert {i for i, degree in enumerate(report.degrees) if degree == 3} == {
        0,
        1,
        9,
        10,
    }
    assert report.witness_uncentered_weighted_phase_mean_turns == Q(819, 1280)
    assert sum(report.witness_uncentered_nodal_turns[:9]) == Q(259, 32)
    assert sum(report.witness_uncentered_nodal_turns[9:]) == Q(287, 64)
    nodes = report.witness_nodal_turns
    assert sum(graph.degree[i] * nodes[i] for i in graph) == 0
    assert all(value == 0 for value in report.witness_epi)
    actual_boundary = []
    for index, ((i, j), offset, turn) in enumerate(
        zip(
            report.geometry.edges,
            report.witness_edge_integer_offsets,
            report.witness_edge_turns,
        )
    ):
        assert offset == (2 if (i, j) == (0, 8) else int((i, j) == (9, 17)))
        assert nodes[j] - nodes[i] - offset == turn
        if j < 9:
            expected = Q(1, 4) if (i, j) == (1, 2) else Q(7, 32)
            expected *= -1 if (i, j) == (0, 8) else 1
        elif i >= 9:
            expected = -Q(1, 9) if (i, j) == (9, 17) else Q(1, 9)
        else:
            expected = Q(31, 576) if i == 0 else -Q(31, 576)
        assert turn == expected
        assert abs(turn) <= Q(1, 4)
        if abs(turn) == Q(1, 4):
            actual_boundary.append(index)
    assert report.witness_boundary_edge_indices == tuple(actual_boundary)
    assert tuple(report.geometry.edges[i] for i in actual_boundary) == ((1, 2),)


def test_all_integer_periods_follow_from_correlated_lifts_and_incidence():
    report = _assess()
    edges = report.geometry.edges
    turns = dict(zip(edges, report.witness_edge_turns))

    def period(cycle):
        return sum(
            turns[tuple(sorted((i, j)))] * (1 if i < j else -1)
            for i, j in zip(cycle, cycle[1:] + cycle[:1])
        )

    assert tuple(map(period, report.named_cycles)) == (2, 1, 0)
    assert report.named_cycle_periods == tuple(map(period, report.named_cycles))
    assert report.fundamental_cycle_periods == tuple(
        map(period, report.geometry.fundamental_cycles)
    )
    # Incidence reconstructs every unwrapped edge from the same nodal lift;
    # the closing integers then supply the correct circular branches.
    for edge, (i, j) in enumerate(edges):
        raw = sum(
            report.geometry.incidence[node][edge] * report.witness_nodal_turns[node]
            for node in report.geometry.nodes
        )
        assert raw == report.witness_nodal_turns[j] - report.witness_nodal_turns[i]
        assert raw - report.witness_edge_integer_offsets[edge] == turns[i, j]


def test_independent_full_storage_and_derivative_bound_cover_the_boundary_witness(mp):
    report = _assess(0)
    q, c, delta = 4 * mp.pi / 9, 2 * mp.pi / 9, mp.pi / 9

    def f(value):
        return 1 - mp.cos(value)

    initial_minimum = 9 * f(q) + 9 * f(c) + 2 * f(delta)
    boundary_energy = 1 + 8 * f(7 * mp.pi / 16) + 9 * f(c) + 2 * f(31 * mp.pi / 288)
    assert abs(
        _storage(mp, _graph(), _witness_phases(mp, report)) - boundary_energy
    ) < mp.mpf("1e-78")
    assert initial_minimum - boundary_energy > _mp(
        mp, report.ideal_storage_gap_lower_bound
    )
    # Reconstruct the independent analytic path, including its derivative.
    # Its endpoint is the report's witness, but it is not a dynamical path.
    for fraction in (Q(0), Q(1, 7), Q(1, 2), Q(6, 7), Q(1)):
        s = _mp(mp, fraction) * mp.pi / 144
        derivative = 8 * mp.sin(q + 8 * s) - 8 * mp.sin(q - s) - mp.sin(delta - s / 2)
        assert derivative < _mp(mp, -Q(17, 288))
        phases = (mp.mpf(0), q - s) + tuple(j * q + (9 - j) * s for j in range(2, 9))
        phases += tuple(delta - s / 2 + j * c for j in range(9))
        potential = f(q + 8 * s) + 8 * f(q - s) + 9 * f(c) + 2 * f(delta - s / 2)
        assert abs(_storage(mp, _graph(), phases) - potential) < mp.mpf("1e-78")
    # Elementary bounds used by the proof give the displayed rational gap
    # without evaluating a root or selecting a preparation from a response.
    derivative_upper = Q(8 * 10, 2 * 16**2) - Q(2 * 31, 288)
    assert derivative_upper < 0
    assert report.ideal_storage_gap_lower_bound == -derivative_upper * Q(3, 144)
    assert report.phase_storage_lipschitz == 2 * _graph().number_of_edges()


@pytest.mark.parametrize("origin_turns", (Q(-3, 5), Q(0), Q(1, 18), Q(1, 7), Q(11, 6)))
def test_all_origin_minimum_and_perturbed_families_keep_the_storage_obstruction(
    origin_turns, mp
):
    report = _assess()
    graph = _graph()
    phases = tuple(2 * mp.pi * k * j / 9 for k in (2, 1) for j in range(9))
    phases = phases[:9] + tuple(
        value + 2 * mp.pi * _mp(mp, origin_turns) for value in phases[9:]
    )
    ideal = _storage(mp, graph, phases)
    zero_origin_minimum = (
        9 * (1 - mp.cos(4 * mp.pi / 9))
        + 9 * (1 - mp.cos(2 * mp.pi / 9))
        + 2 * (1 - mp.cos(mp.pi / 9))
    )
    # Equality at the independently known minimizing relative origin.
    assert ideal >= zero_origin_minimum - mp.mpf("1e-78")
    radius = _mp(mp, report.phase_error_radius)
    errors = tuple(radius * (1 if i % 3 == 0 else -1) for i in graph)
    perturbed = tuple(value + error for value, error in zip(phases, errors))
    perturbed_potential = _storage(mp, graph, perturbed)
    assert abs(perturbed_potential - ideal) <= _mp(
        mp, report.phase_storage_error_allowance
    )
    witness_energy = _storage(mp, graph, _witness_phases(mp, report))
    assert perturbed_potential - witness_energy > _mp(
        mp, report.storage_gap_lower_bound
    )
    forms = tuple(mp.mpf(i - 8) / 7 for i in graph)
    assert _storage(mp, graph, perturbed, forms) > perturbed_potential
    assert report.handoff_obstruction_certified


def test_witness_matches_arbitrary_source_mean_leaf_without_storage_change(mp):
    report = _assess()
    graph = _graph()
    phases = _witness_phases(mp, report)
    shifted = tuple(value + _mp(mp, Q(7, 13)) for value in phases)
    forms = (mp.mpf(11) / 7,) * len(graph)
    assert abs(
        _storage(mp, graph, shifted, forms) - _storage(mp, graph, phases)
    ) < mp.mpf("1e-78")
    phase_mean = sum(graph.degree[i] * shifted[i] for i in graph) / 40
    form_mean = sum(graph.degree[i] * forms[i] for i in graph) / 40
    assert abs(phase_mean - _mp(mp, Q(7, 13))) < mp.mpf("1e-78")
    assert abs(form_mean - _mp(mp, Q(11, 7))) < mp.mpf("1e-78")


def test_positive_error_family_uses_a_strict_rational_margin():
    report = _assess()
    assert report.storage_gap_lower_bound == Q(137, 221184)
    assert report.max_certifying_phase_error_radius == Q(17, 552960)
    assert report.handoff_obstruction_certified
    assert report.status == "certified_handoff_obstruction"
    assert report.unavailable_reasons == ()
    assert 2 * report.max_certifying_phase_error_radius < Q(3, 18)
    assert report.reference_model.effective_weights == (Q(1023, 1024), Q(1, 1024))
    assert report.reference_model.storage_scale == 1
    assert report.capacity == (Q(1),) * 18


@pytest.mark.parametrize("radius", (Q(17, 552960), Q(1, 100), Q(10**400)))
def test_zero_or_negative_declared_margin_abstains_without_capture_inference(radius):
    report = _assess(radius)
    assert report.phase_error_radius == radius
    assert report.storage_gap_lower_bound <= 0
    assert not report.handoff_obstruction_certified
    assert report.status == "unavailable"
    assert report.unavailable_reasons == ("strict_storage_gap_not_certified",)
    assert report.witness_edge_turns[report.witness_boundary_edge_indices[0]] == Q(1, 4)


@pytest.mark.parametrize("radius", (Q(0), Q(1, 10**400), np.int64(0), -0.0, 1e-6))
def test_shared_admission_preserves_exact_rationals_and_represented_real_values(radius):
    report = _assess(radius)
    expected = Q.from_float(radius) if type(radius) is float else Q(radius)
    assert type(report.phase_error_radius) is Q
    assert report.phase_error_radius == expected
    assert report.handoff_obstruction_certified


class _UnderflowingReal(float):
    def __float__(self):
        return 0.0


@pytest.mark.parametrize(
    "radius",
    (
        True,
        False,
        np.bool_(True),
        float("nan"),
        float("inf"),
        -float("inf"),
        -1,
        -Q(1, 10**400),
        "0.0001",
        None,
        1j,
        _UnderflowingReal(1),
    ),
)
def test_invalid_primitives_reject_before_geometry(radius, monkeypatch):
    monkeypatch.setattr(
        owner, "_derive", lambda *args: pytest.fail("invalid radius reached geometry")
    )
    with pytest.raises((TypeError, ValueError), match="phase_error_radius"):
        _assess(radius)


def test_exact_projection_and_sdk_export_do_not_evaluate_other_research(tmp_path):
    report = _assess()
    direct = report.to_dict()
    generic = relational_report_to_dict(report)
    assert direct["schema"] == "tnfr.sine-two-port-handoff-obstruction.v1"
    assert generic["report_type"] == "SineTwoPortHandoffObstruction"
    assert generic["report"] == direct["report"]
    assert direct["report"]["storage_gap_lower_bound"] == {
        "numerator": 137,
        "denominator": 221184,
    }
    path = tmp_path / "handoff-obstruction.json"
    export_to_json(report, path)
    assert json_loads(path.read_text(encoding="utf-8")) == direct
    assert get_type_hints(owner.SineTwoPortHandoffObstruction)
    parameters = signature(owner.assess_sine_two_port_handoff_obstruction).parameters
    assert tuple(parameters) == ("phase_error_radius",)
    assert (
        parameters["phase_error_radius"].default
        is parameters["phase_error_radius"].empty
    )
