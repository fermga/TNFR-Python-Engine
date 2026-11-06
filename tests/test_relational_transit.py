"""Continuous enclosure contracts independent of the long proof computation."""

import math
import pickle
from fractions import Fraction as Q

import networkx as nx
import pytest

from tnfr.dynamics.relational import (
    RelationalExchangeModel,
    evaluate_relational_exchange,
)
from tnfr.mathematics._rational_interval import I
from tnfr.physics import relational_transit as owner

CYCLES = (tuple(range(5)), tuple(range(5, 10)))


def graph_at(A=0.0, B=0.0, a=0.0, b=0.0):
    graph = nx.Graph()
    for row in CYCLES:
        graph.add_edges_from(zip(row, row[1:] + row[:1]), weight=1.0)
    graph.add_edges_from(((0, 5), (1, 6)), weight=1.0)
    graph.graph.update(GAMMA={"type": "none"}, vectorized_dnfr=True)
    for row in CYCLES:
        for node, x, phase in zip(row, (A, -A, -B, 0, B), (a, -a, -b, 0, b)):
            graph.nodes[node].update(EPI=x, theta=phase, nu_f=1.0)
    return graph


MODEL = RelationalExchangeModel(1.0, phase_domain="positive_resultant")


def owned(graph):
    return pickle.dumps(
        (graph.graph, tuple(graph.nodes(data=True)), tuple(graph.edges(data=True)))
    )


def test_reduction_matches_full_native_rows_without_changing_graph():
    graph = graph_at(A=0.125, B=-0.0625, a=0.75, b=0.625)
    saved = owned(graph)
    field = evaluate_relational_exchange(graph, model=MODEL)
    lookup = {node: index for index, node in enumerate(field.nodes)}
    x = {node: Q(field.epi[i]) for node, i in lookup.items()}
    theta = {node: Q(field.phase[i]) for node, i in lookup.items()}
    box = tuple(I(v) for v in (3 * x[0] - x[4], 2 * x[4] - x[0], theta[0], theta[4]))
    rates = owner._flow(box, Q(1, 2), Q(1, 2), Q(1))
    dx = {node: field.form_rate[i] for node, i in lookup.items()}
    expected = (
        3 * dx[0] - dx[4],
        2 * dx[4] - dx[0],
        field.phase_rate[lookup[0]],
        field.phase_rate[lookup[4]],
    )
    # The engine reads binary64 trigonometry; it is not an exact-real oracle.
    assert all(
        abs(float(value.midpoint) - ref) < 2e-15 for value, ref in zip(rates, expected)
    )
    assert owned(graph) == saved


@pytest.fixture(scope="module")
def short_capture():
    graph = graph_at(a=4 * math.pi / 5, b=2 * math.pi / 5)
    saved = owned(graph)
    result = owner.certify_relational_transit_capture(
        graph, model=MODEL, cycles=CYCLES, horizon=Q(1, 8), time_step=Q(1, 8), order=4
    )
    assert owned(graph) == saved
    return result


def test_whole_endpoint_capture_and_all_resultants(short_capture):
    result = short_capture
    assert result.admitted, result.unavailable_reasons
    assert result.validated_horizon == Q(1, 8)
    assert result.endpoint_storage.hi < 7
    assert all(value.lo > 0 for value in result.positive_rectangle_margins)
    assert all(value > 0 for value in result.steps[0].resultant_real_lower_bounds)
    assert result.steps[0].picard_interior_margin > 0
    assert not result.initial_winding_zero
    assert result.requested_sector == result.target_sector == 1
    assert result.rectangle_kind == "positive_winding"
    assert result.rectangle_margin_bounds == result.positive_rectangle_margins
    assert tuple(sector for sector, _ in result.candidate_rectangle_margin_bounds) == (
        1,
        0,
        -1,
    )


def test_consensus_zero_solution_enclosed_even_without_positive_capture():
    result = owner.certify_relational_transit_capture(
        graph_at(),
        model=MODEL,
        cycles=CYCLES,
        horizon=Q(1, 8),
        time_step=Q(1, 8),
        order=4,
    )
    assert result.validated_horizon == Q(1, 8)
    assert all(value.contains(0) for value in result.endpoint)
    assert result.initial_winding_zero
    assert not result.admitted
    assert (
        "whole_endpoint_positive_rectangle_not_certified" in result.unavailable_reasons
    )
    assert result.rectangle_kind is None and result.rectangle_margin_bounds == ()
    assert result.target_sector is None and result.requested_sector == 1
    consensus = dict(result.candidate_rectangle_margin_bounds)[0]
    assert all(value.lo > 0 for value in consensus)


@pytest.mark.parametrize("sector", (-1, 0, 1))
def test_any_sector_admission_retains_the_entire_matching_endpoint(sector):
    graph = graph_at(a=sector * 4 * math.pi / 5, b=sector * 2 * math.pi / 5)
    saved = owned(graph)
    result = owner.certify_relational_transit_capture(
        graph,
        model=MODEL,
        cycles=CYCLES,
        horizon=Q(1, 8),
        time_step=Q(1, 8),
        order=4,
        requested_sector=None,
    )
    assert owned(graph) == saved
    assert result.admitted, result.unavailable_reasons
    assert result.requested_sector is None and result.target_sector == sector
    assert (
        result.rectangle_kind
        == {
            -1: "negative_winding",
            0: "consensus",
            1: "positive_winding",
        }[sector]
    )
    assert (
        result.rectangle_margin_bounds
        == dict(result.candidate_rectangle_margin_bounds)[sector]
    )
    assert all(value.lo > 0 for value in result.rectangle_margin_bounds)
    assert result.endpoint_storage.hi < 7
    if sector == 0:
        assert all(value.contains(0) for value in result.endpoint)


@pytest.mark.parametrize("requested", (-1, 0))
def test_declared_sector_is_required_instead_of_substituting_a_different_basin(
    requested,
):
    result = owner.certify_relational_transit_capture(
        graph_at(),
        model=MODEL,
        cycles=CYCLES,
        horizon=Q(1, 8),
        time_step=Q(1, 8),
        order=4,
        requested_sector=requested,
    )
    assert result.validated_horizon == result.horizon
    assert result.requested_sector == requested
    assert result.admitted == (requested == 0)
    if requested == 0:
        assert result.target_sector == 0 and result.rectangle_kind == "consensus"
    else:
        assert result.target_sector is None and result.rectangle_kind is None
        assert result.rectangle_margin_bounds == ()
        assert result.unavailable_reasons == (
            "whole_endpoint_requested_rectangle_not_certified",
        )


def test_central_resultant_blocks_formal_four_row_continuation():
    box = tuple(I(value) for value in (0, 0, Q(8, 5), Q(8, 5)))
    bounds = owner._regular_bounds(box)
    assert bounds[0] > 0 and bounds[1] > 0 and bounds[2] < 0
    step, tube, failure = owner._validated_step(
        box, Q(1, 8), Q(1, 2), Q(1, 2), Q(1), 4, Q(0)
    )
    assert step is None and tube == box
    assert failure == "whole_tube_resultant_not_positive"


def test_exact_symmetry_rejection_never_projects_or_attempts_transit(monkeypatch):
    graph = graph_at(a=0.75, b=0.625)
    graph.nodes[5]["EPI"] = 2.0**-60

    def forbidden(*args, **kwargs):
        raise AssertionError("inadmissible symmetry attempted")

    monkeypatch.setattr(owner, "_validated_step", forbidden)
    result = owner.certify_relational_transit_capture(
        graph, model=MODEL, cycles=CYCLES, horizon=1, time_step=Q(1, 8)
    )
    assert not result.admitted and not result.steps
    assert "exact_copy_reflection_required" in result.unavailable_reasons


def test_unresolved_step_is_retained_and_not_retried(monkeypatch):
    calls = []

    def fail(box, *args):
        calls.append(box)
        return None, box, "synthetic_unresolved_bound"

    monkeypatch.setattr(owner, "_validated_step", fail)
    result = owner.certify_relational_transit_capture(
        graph_at(),
        model=MODEL,
        cycles=CYCLES,
        horizon=1,
        time_step=Q(1, 8),
        requested_sector=None,
    )
    assert len(calls) == 1
    assert result.failed_tube == result.initial_box
    assert result.validated_horizon == 0
    assert "synthetic_unresolved_bound" in result.unavailable_reasons
    assert result.rectangle_kind == "consensus"
    assert all(value.lo > 0 for value in result.rectangle_margin_bounds)
    assert not result.admitted and result.target_sector is None


@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({"horizon": True}, TypeError),
        ({"time_step": 0.125}, TypeError),
        ({"horizon": 0}, ValueError),
        ({"order": True}, ValueError),
        ({"order": 3}, ValueError),
        ({"order": 17}, ValueError),
        ({"horizon": 4097, "time_step": 1}, ValueError),
        ({"requested_sector": True}, ValueError),
        ({"requested_sector": False}, ValueError),
        ({"requested_sector": 1.0}, ValueError),
        ({"requested_sector": 2}, ValueError),
        ({"requested_sector": "0"}, ValueError),
    ],
)
def test_invalid_numerical_policy(kwargs, error):
    policy = dict(horizon=1, time_step=Q(1, 8), order=4)
    policy.update(kwargs)
    with pytest.raises(error):
        owner.certify_relational_transit_capture(
            graph_at(), model=MODEL, cycles=CYCLES, **policy
        )
