"""Independent current-source controls of the frozen geometry/response question.

The declaration predates response evaluation. These tests do not rewrite its
artifacts, infer the coefficient from the response, or certify a trajectory.
"""

from dataclasses import replace
from fractions import Fraction as Q
from pathlib import Path

import mpmath as mp
import networkx as nx
import pytest

from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.mathematics._rational_interval import I
from tnfr.physics.phase_cycle_geometry import assess_return_path_geometry_response
from tnfr.physics.relational_sine_comparison import bound_relational_sine_exchange
from tnfr.utils.io import json_loads


@pytest.fixture(scope="module")
def declaration():
    path = (
        Path(__file__).resolve().parents[2]
        / "docs/assets/return_path_geometry_response/declaration.json"
    )
    return json_loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def graph(declaration):
    graph = nx.Graph()
    graph.add_nodes_from(declaration["nodes"])
    for name in ("left_cycle", "right_cycle"):
        cycle = declaration[name]
        graph.add_edges_from(zip(cycle, cycle[1:] + cycle[:1]))
    graph.add_edges_from(declaration["additional_edges"])
    graph.graph["GAMMA"] = {"type": "none"}
    for node in graph:
        graph.nodes[node].update(EPI=0, theta=0, nu_f=1)
    return graph


@pytest.fixture(scope="module")
def source(graph):
    return bound_relational_sine_exchange(
        graph,
        reference_model=RelationalExchangeModel(
            1, epi_weight=0, phase_weight=1, phase_domain="regular"
        ),
    )


def _root(declaration):
    coefficient = declaration["source_coefficient"]

    def current(angle):
        sine = mp.sin(angle)
        return sine + coefficient * sine**3

    return mp.findroot(
        lambda v: current(mp.pi / 2 - v / 4) - current(v) - current(2 * v / 3),
        (mp.pi / 6, 2 * mp.pi / 5),
    )


@pytest.fixture(scope="module")
def observation(declaration):
    with mp.workdps(declaration["geometry_observation"]["decimal_places"]):
        denominator = declaration["geometry_observation"]["turn_grid_denominator"]
        scaled = _root(declaration) * denominator / (2 * mp.pi)
        return Q(int(mp.floor(scaled)), denominator), Q(
            int(mp.ceil(scaled)), denominator
        )


def _predict(source, declaration, observation, **changes):
    arguments = dict(
        left_cycle=tuple(declaration["left_cycle"]),
        right_cycle=tuple(declaration["right_cycle"]),
        mediator=declaration["mediator"],
        special_turn_bounds=observation,
        form_direction=tuple(declaration["form_direction"]),
        observation_origin=declaration["geometry_observation"]["origin"],
    )
    arguments.update(changes)
    return assess_return_path_geometry_response(source, **arguments)


@pytest.fixture(scope="module")
def prediction(source, declaration, observation):
    return _predict(source, declaration, observation)


def test_geometry_infers_a_nonzero_coefficient_without_consuming_the_response(
    declaration, observation, prediction
):
    assert prediction.coefficient_status == "bounded"
    assert 0 < prediction.coefficient_lower < declaration["source_coefficient"]
    assert declaration["source_coefficient"] < prediction.coefficient_upper
    assert prediction.special_turn_bounds == observation
    assert prediction.named_cycle_periods == tuple(declaration["named_periods"])
    assert prediction.observation_origin == "supplied_mathematical_interval"
    assert prediction.form_direction == tuple(declaration["form_direction"])


def test_reserved_acceleration_from_both_full_nodal_rows_is_enclosed(
    declaration, graph, prediction
):
    with mp.workdps(declaration["geometry_observation"]["decimal_places"]):
        v = _root(declaration)
        bulk, connecting = mp.pi / 2 - v / 4, 2 * v / 3
        phase = (
            0,
            v,
            v + bulk,
            v + 2 * bulk,
            v + 3 * bulk,
            2 * connecting,
            2 * connecting - v,
            2 * connecting - v - bulk,
            2 * connecting - v - 2 * bulk,
            2 * connecting - v - 3 * bulk,
            connecting,
        )
        coefficient = declaration["source_coefficient"]
        direction = declaration["form_direction"]

        def complete_rows(form, angles):
            form_rows = tuple(
                mp.fsum(
                    mp.sin(angles[j] - angles[i])
                    + coefficient * mp.sin(angles[j] - angles[i]) ** 3
                    for j in graph[i]
                )
                / graph.degree[i]
                for i in graph
            )
            phase_rows = tuple(
                mp.fsum(form[i] - form[j] for j in graph[i]) / graph.degree[i]
                for i in graph
            )
            return form_rows + phase_rows

        zero = (mp.mpf(0),) * len(graph)
        assert max(map(abs, complete_rows(zero, phase))) < mp.mpf("1e-85")
        phase_velocity = tuple(
            mp.diff(
                lambda amplitude: complete_rows(
                    tuple(amplitude * item for item in direction), phase
                )[len(graph) + i],
                0,
            )
            for i in graph
        )
        acceleration = tuple(
            mp.diff(
                lambda time: complete_rows(
                    zero,
                    tuple(
                        angle + time * rate
                        for angle, rate in zip(phase, phase_velocity)
                    ),
                )[i],
                0,
            )
            for i in graph
        )
        assert len(prediction.response_acceleration_bounds) == len(graph)
        for enclosure, observed in zip(
            prediction.response_acceleration_bounds, acceleration
        ):
            assert enclosure.contains(Q(mp.nstr(observed, 85)))
        # Reciprocity preserves the degree-weighted form rate. This is checked
        # against independently differentiated rows, not a cached balance flag.
        assert abs(mp.fsum(graph.degree[i] * acceleration[i] for i in graph)) < mp.mpf(
            "1e-85"
        )
        assert any(value != 0 for value in acceleration)


def test_interval_spanning_entire_branch_has_no_finite_coefficient_upper_bound(
    source, declaration
):
    report = _predict(source, declaration, (Q(1, 12), Q(1, 5)))
    assert report.coefficient_status == "unbounded_above"
    assert report.coefficient_lower == 0
    assert report.coefficient_upper is None
    assert report.response_acceleration_bounds is None


def test_pure_cubic_limit_has_quadratically_ill_conditioned_inverse(declaration):
    # A finite check of the separately derived asymptotic expansion. This is
    # geometry-only evidence and does not add another reserved response.
    with mp.workdps(95):

        def first(angle):
            return mp.cos(angle / 4) - mp.sin(angle) - mp.sin(2 * angle / 3)

        def third(angle):
            return (
                mp.cos(angle / 4) ** 3 - mp.sin(angle) ** 3 - mp.sin(2 * angle / 3) ** 3
            )

        limit = mp.findroot(third, (mp.pi / 6, 2 * mp.pi / 5))
        constant = first(limit) / mp.diff(third, limit)
        assert constant > 0
        previous = None
        for coefficient in (1000, 10000):
            angle = _root({**declaration, "source_coefficient": coefficient})
            derivative = mp.diff(lambda value: -first(value) / third(value), angle)
            errors = (
                abs(coefficient * (limit - angle) / constant - 1),
                abs(derivative * constant / coefficient**2 - 1),
            )
            assert angle < limit
            assert derivative > 0
            assert max(errors) < mp.mpf("0.01")
            if previous is not None:
                assert all(new < old for new, old in zip(errors, previous))
            previous = errors


def test_source_derived_caches_do_not_supply_the_inferred_response(
    source, declaration, observation, prediction
):
    altered = replace(
        source,
        storage=I(999),
        form_rates=(I(999),) * len(source.nodes),
        phase_rates=(I(999),) * len(source.nodes),
    )
    rebuilt = _predict(altered, declaration, observation)
    assert rebuilt.coefficient_lower == prediction.coefficient_lower
    assert rebuilt.coefficient_upper == prediction.coefficient_upper
    assert (
        rebuilt.response_acceleration_bounds == prediction.response_acceleration_bounds
    )


def test_boolean_capacity_cannot_pass_as_the_declared_unit_capacity(
    source, declaration, observation
):
    altered = replace(source, capacity=(True,) + source.capacity[1:])
    with pytest.raises((TypeError, ValueError)):
        _predict(altered, declaration, observation)
