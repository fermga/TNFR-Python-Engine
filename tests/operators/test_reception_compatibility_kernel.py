"""Finite source decisions exercise admission, exact contrast and sealed reads."""

from __future__ import annotations

import math
import sys
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
from fractions import Fraction

import networkx as nx
import pytest

from tnfr.constants.aliases import ALIAS_EPI, ALIAS_THETA, ALIAS_VF
from tnfr.operators._reception_compatibility import (
    ReceptionCompatibilityPolicy,
    observe_reception_compatibility,
    reception_compatibility_matches_graph,
)
from tnfr.utils.numeric import angle_diff


def _graph(graph_type=nx.Graph):
    graph = graph_type()
    for node, phase, capacity in ((0, 0.0, 1.0), (1, 0.2, 1.0), (2, 2.0, 4.0)):
        graph.add_node(node, **{ALIAS_THETA[0]: phase, ALIAS_VF[0]: capacity})
    graph.add_edges_from([(2, 0), (1, 0)])
    return graph


def _state(graph):
    return deepcopy((dict(graph.nodes(data=True)), tuple(graph.edges), graph.graph))


@pytest.mark.parametrize(
    "graph_type", [nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph]
)
def test_select_reads_all_candidates_and_keeps_incoming_order_without_writes(
    graph_type,
):
    graph = _graph(graph_type)
    if graph.is_directed():
        graph.add_node(3, **{ALIAS_THETA[0]: float("nan"), ALIAS_VF[0]: -1})
        graph.add_edge(0, 3)
    if graph.is_multigraph():
        graph.add_edge(1, 0)
    before = _state(graph)
    decision = observe_reception_compatibility(
        graph, 0, ReceptionCompatibilityPolicy(0.5, 0.1)
    )

    assert decision.candidates == (2, 1)
    assert decision.candidate_phases == (2.0, 0.2)
    assert decision.candidate_capacities == (4.0, 1.0)
    assert decision.selected_sources == (1,)
    assert decision.eligible and decision.reason == "selected"
    assert decision.abstention_reason is None
    assert decision._proof_fields_are_intact()
    assert reception_compatibility_matches_graph(graph, decision)
    assert _state(graph) == before


@pytest.mark.parametrize(
    "mode, expected, reason",
    [("select", (1,), "selected"), ("require_all", (), "not_all_compatible")],
)
def test_modes_distinguish_selection_from_all_source_restriction(
    mode, expected, reason
):
    decision = observe_reception_compatibility(
        _graph(), 0, ReceptionCompatibilityPolicy(0.5, 1, mode)
    )
    assert decision.selected_sources == expected
    assert decision.reason == reason


@pytest.mark.parametrize("mode", ["select", "require_all"])
def test_maximal_policy_selects_complete_valid_support_without_u3_restriction(mode):
    graph = _graph()
    decision = observe_reception_compatibility(
        graph, 0, ReceptionCompatibilityPolicy(math.pi, 1, mode)
    )
    assert decision.selected_sources == (2, 1)
    assert decision.candidate_phases[0] > math.pi / 2


@pytest.mark.parametrize("mode", ["select", "require_all"])
def test_isolated_valid_target_abstains_and_does_not_make_reception_event(mode):
    graph = _graph()
    graph.remove_edges_from(list(graph.edges))
    before = _state(graph)
    decision = observe_reception_compatibility(
        graph, 0, ReceptionCompatibilityPolicy(0, 0, mode)
    )
    assert decision.selected_sources == ()
    assert decision.abstention_reason == "no_candidates"
    assert not decision.eligible
    assert _state(graph) == before


def test_nonempty_incompatible_support_has_distinct_abstention_reason():
    decision = observe_reception_compatibility(
        _graph(), 0, ReceptionCompatibilityPolicy(0, 0)
    )
    assert decision.abstention_reason == "no_compatible_sources"
    assert decision.candidates == (2, 1)


@pytest.mark.parametrize(
    "target,source,eta,compatible",
    [
        (0, 0, 0, True),
        (0, 1, 0, False),
        (0, 1, 1, True),
        (1, 0, math.nextafter(1.0, 0.0), False),
        (1, 0, 1, True),
        (1, 1, 0, True),
        (1, 2, 0.5, True),
        (2, 1, 0.5, True),
        (1, 2, math.nextafter(0.5, 0.0), False),
    ],
)
def test_capacity_contrast_zero_cases_symmetry_and_closed_boundary(
    target, source, eta, compatible
):
    graph = nx.Graph()
    graph.add_node(0, **{ALIAS_THETA[0]: 0, ALIAS_VF[0]: target})
    graph.add_node(1, **{ALIAS_THETA[0]: 0, ALIAS_VF[0]: source})
    graph.add_edge(0, 1)
    decision = observe_reception_compatibility(
        graph, 0, ReceptionCompatibilityPolicy(0, eta)
    )
    assert decision.eligible is compatible


@pytest.mark.parametrize("scale", [math.ulp(0.0), 1.0, sys.float_info.max / 2])
def test_exact_capacity_boundary_does_not_accept_rounded_underflow_product(scale):
    graph = nx.Graph()
    graph.add_node(0, **{ALIAS_THETA[0]: 0, ALIAS_VF[0]: scale})
    graph.add_node(1, **{ALIAS_THETA[0]: 0, ALIAS_VF[0]: 2 * scale})
    graph.add_edge(0, 1)
    exact_boundary = observe_reception_compatibility(
        graph, 0, ReceptionCompatibilityPolicy(0, 0.5)
    )
    just_below = observe_reception_compatibility(
        graph, 0, ReceptionCompatibilityPolicy(0, math.nextafter(0.5, 0.0))
    )
    assert exact_boundary.selected_sources == (1,)
    assert just_below.selected_sources == ()
    if scale == math.ulp(0.0):
        # A floating product falsely makes the just-below inequality inclusive.
        assert scale <= math.nextafter(0.5, 0.0) * (2 * scale)


def test_phase_uses_circular_difference_and_explicit_represented_boundary():
    graph = _graph()
    graph.remove_node(2)
    graph.nodes[0][ALIAS_THETA[0]] = math.pi - 0.1
    graph.nodes[1][ALIAS_THETA[0]] = -math.pi + 0.1
    gap = abs(
        angle_diff(graph.nodes[1][ALIAS_THETA[0]], graph.nodes[0][ALIAS_THETA[0]])
    )
    assert gap < 0.3
    at = observe_reception_compatibility(graph, 0, ReceptionCompatibilityPolicy(gap, 0))
    below = observe_reception_compatibility(
        graph, 0, ReceptionCompatibilityPolicy(math.nextafter(gap, 0), 0)
    )
    assert at.eligible and not below.eligible


@pytest.mark.parametrize("field", ["max_phase_gap", "max_relative_capacity_gap"])
@pytest.mark.parametrize(
    "invalid",
    [True, "0.1", None, float("nan"), float("inf"), -0.1, Fraction(1, 10**1000)],
)
def test_policy_rejects_invalid_or_lost_values_before_materialization(field, invalid):
    kwargs = dict(max_phase_gap=0.5, max_relative_capacity_gap=0.5)
    kwargs[field] = invalid
    with pytest.raises((TypeError, ValueError)):
        ReceptionCompatibilityPolicy(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"max_phase_gap": 0},
        {"max_relative_capacity_gap": 0},
        {
            "max_phase_gap": math.nextafter(math.pi, math.inf),
            "max_relative_capacity_gap": 0,
        },
        {
            "max_phase_gap": 0,
            "max_relative_capacity_gap": Fraction(1) + Fraction(1, 10**100),
        },
        {"max_phase_gap": 0, "max_relative_capacity_gap": 0, "mode": "automatic"},
        {"max_phase_gap": 0, "max_relative_capacity_gap": 0, "mode": True},
    ],
)
def test_required_tolerances_original_bounds_and_mode_are_explicit(kwargs):
    with pytest.raises((TypeError, ValueError)):
        ReceptionCompatibilityPolicy(**kwargs)


@pytest.mark.parametrize("endpoint", [0, 1, 2])
@pytest.mark.parametrize("aliases", [ALIAS_THETA, ALIAS_VF])
@pytest.mark.parametrize(
    "invalid", [True, "1.0", None, float("nan"), float("inf"), Fraction(1, 10**1000)]
)
def test_every_consumed_coordinate_rejects_invalid_authoritative_alias(
    endpoint, aliases, invalid
):
    graph = _graph()
    graph.nodes[endpoint][aliases[0]] = invalid
    if len(aliases) > 1:
        graph.nodes[endpoint][aliases[1]] = 1.0
    before = _state(graph)
    with pytest.raises((TypeError, ValueError)):
        observe_reception_compatibility(graph, 0, ReceptionCompatibilityPolicy(0, 0))
    assert _state(graph) == before


@pytest.mark.parametrize("endpoint", [0, 1, 2])
@pytest.mark.parametrize("aliases", [ALIAS_THETA, ALIAS_VF])
def test_missing_coordinate_is_not_defaulted_even_for_excluded_source(
    endpoint, aliases
):
    graph = _graph()
    for alias in aliases:
        graph.nodes[endpoint].pop(alias, None)
    with pytest.raises(ValueError, match="explicit"):
        observe_reception_compatibility(graph, 0, ReceptionCompatibilityPolicy(0, 0))


def test_valid_secondary_aliases_and_negative_phase_are_admitted():
    graph = _graph()
    for node in graph:
        for aliases in (ALIAS_THETA, ALIAS_VF):
            value = graph.nodes[node].pop(aliases[0])
            graph.nodes[node][aliases[-1]] = value
    graph.nodes[1][ALIAS_THETA[-1]] = -0.2
    decision = observe_reception_compatibility(
        graph, 0, ReceptionCompatibilityPolicy(0.5, 0.1)
    )
    assert decision.selected_sources == (1,)


@pytest.mark.parametrize("value", [-1.0, -Fraction(1, 10**1000)])
def test_negative_capacity_is_rejected_even_if_phase_would_exclude(value):
    graph = _graph()
    graph.nodes[2][ALIAS_VF[0]] = value
    with pytest.raises(ValueError):
        observe_reception_compatibility(graph, 0, ReceptionCompatibilityPolicy(0, 0))


def test_finite_phases_with_unrepresentable_difference_are_invalid_not_incompatible():
    graph = _graph()
    graph.nodes[0][ALIAS_THETA[0]] = -sys.float_info.max
    graph.nodes[2][ALIAS_THETA[0]] = sys.float_info.max
    with pytest.raises(ValueError):
        observe_reception_compatibility(graph, 0, ReceptionCompatibilityPolicy(0, 0))


def test_observer_consumes_no_form_kind_pressure_or_history():
    graph = _graph()
    graph.nodes[1][ALIAS_EPI[0]] = object()
    graph.graph["history"] = object()
    decision = observe_reception_compatibility(
        graph, 0, ReceptionCompatibilityPolicy(0.5, 0.1)
    )
    assert decision.eligible
    graph.nodes[1][ALIAS_EPI[0]] = float("nan")
    graph.graph["history"] = None
    assert reception_compatibility_matches_graph(graph, decision)


@pytest.mark.parametrize(
    "mutation",
    [
        "target_phase",
        "target_capacity",
        "excluded_phase",
        "excluded_capacity",
        "selected_phase",
        "remove_edge",
        "add_edge",
        "order",
    ],
)
def test_full_candidate_read_is_revalidated_even_when_selection_would_stay_same(
    mutation,
):
    graph = _graph()
    decision = observe_reception_compatibility(
        graph, 0, ReceptionCompatibilityPolicy(0.5, 0.1)
    )
    if mutation in (
        "target_phase",
        "target_capacity",
        "excluded_phase",
        "excluded_capacity",
        "selected_phase",
    ):
        endpoint, alias = {
            "target_phase": (0, ALIAS_THETA[0]),
            "target_capacity": (0, ALIAS_VF[0]),
            "excluded_phase": (2, ALIAS_THETA[0]),
            "excluded_capacity": (2, ALIAS_VF[0]),
            "selected_phase": (1, ALIAS_THETA[0]),
        }[mutation]
        graph.nodes[endpoint][alias] += 0.01
    elif mutation == "remove_edge":
        graph.remove_edge(2, 0)
    elif mutation == "add_edge":
        graph.add_node(3, **{ALIAS_THETA[0]: 3.0, ALIAS_VF[0]: 10.0})
        graph.add_edge(3, 0)
    else:
        graph.remove_edge(2, 0)
        graph.add_edge(2, 0)
    assert not reception_compatibility_matches_graph(graph, decision)


def test_detached_equivalent_graph_matches_and_actual_self_loop_is_one_candidate():
    graph = _graph(nx.MultiDiGraph)
    graph.add_edge(0, 0)
    graph.add_edge(0, 0)
    decision = observe_reception_compatibility(
        graph, 0, ReceptionCompatibilityPolicy(0, 0)
    )
    assert decision.selected_sources == (0,)
    assert reception_compatibility_matches_graph(deepcopy(graph), decision)


@pytest.mark.parametrize(
    "field,value",
    [
        ("reason", "no_candidates"),
        ("selected_sources", (2,)),
        ("candidate_phases", (0.0, 0.2)),
        ("candidate_capacities", (1.0, 1.0)),
    ],
)
def test_tampered_decision_is_not_revalidated(field, value):
    graph = _graph()
    decision = observe_reception_compatibility(
        graph, 0, ReceptionCompatibilityPolicy(0.5, 0.1)
    )
    object.__setattr__(decision, field, value)
    assert not decision._proof_fields_are_intact()
    assert not reception_compatibility_matches_graph(graph, decision)


def test_policy_tampering_and_unsealed_replacement_are_detected():
    graph = _graph()
    decision = observe_reception_compatibility(
        graph, 0, ReceptionCompatibilityPolicy(0.5, 0.1)
    )
    with pytest.raises(FrozenInstanceError):
        decision.policy.mode = "require_all"
    unsealed = replace(decision, _proof_stamp=())
    assert not unsealed._proof_fields_are_intact()
    object.__setattr__(decision.policy, "max_phase_gap", 0.6)
    assert not decision._proof_fields_are_intact()
    assert not reception_compatibility_matches_graph(graph, decision)


def test_payload_rejects_misaligned_source_coordinates_and_invented_result():
    decision = observe_reception_compatibility(
        _graph(), 0, ReceptionCompatibilityPolicy(0.5, 0.1)
    )
    with pytest.raises(ValueError, match="align"):
        replace(decision, candidate_phases=(0.2,))
    with pytest.raises(ValueError, match="contradicts"):
        replace(decision, selected_sources=(2,))
    with pytest.raises(TypeError, match="tuples"):
        replace(decision, candidates=[2, 1])
