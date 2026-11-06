"""Real finite THOL/UM/EN/RA composition with a connected supplied parent.

The child's identity comes from this THOL execution. The explicit UM request
supplies access, not authenticated ancestry or autonomous candidate discovery.
"""

from __future__ import annotations

import math
from copy import deepcopy

import networkx as nx
import pytest

from tnfr.alias import get_attr
from tnfr.constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_THETA, ALIAS_VF
from tnfr.operators._thol_constants import THOL_CHILD_VF_DAMPING, THOL_SUB_EPI_SCALING
from tnfr.operators.definitions import Coupling, Reception, Resonance, SelfOrganization
from tnfr.operators.grammar_debt import read_grammar_state
from tnfr.operators.network_stage import (
    STAGE_SCHEDULE_KEY,
    execute_coupling_stage,
    execute_neighbor_stage,
    execute_self_organization_stage,
)
from tnfr.operators.nodal_equation import observe_structural_acceleration
from tnfr.operators.preconditions import OperatorPreconditionError
from tnfr.operators.word_execution import execute_network_operator_stage
from tnfr.types import require_finite_real_scalar_epi

_PARENT = "parent"
_PEER = "old-peer"


def _source(graph_type=nx.Graph):
    graph = graph_type(
        THOL_METABOLIC_ENABLED=False,
        THOL_PROPAGATION_ENABLED=False,
        THOL_ALLOW_ISOLATED=False,
        VALIDATE_OPERATOR_PRECONDITIONS=True,
        GRAMMAR_REJECTION_MODE="raise",
        _t=2.0,
        RANDOM_SEED=17,
        UM_FUNCTIONAL_LINKS=True,
        UM_BIDIRECTIONAL=True,
        UM_SYNC_VF=True,
        UM_STABILIZE_DNFR=True,
        UM_COMPAT_THRESHOLD=0.5,
        GLYPH_FACTORS={
            "THOL_accel": 0.25,
            "UM_theta_push": 0.5,
            "UM_vf_sync": 0.25,
            "UM_dnfr_reduction": 0.25,
            "EN_mix": 0.25,
            "RA_epi_diff": 0.5,
        },
    )
    graph.add_node(
        _PARENT,
        EPI=0.875,
        nu_f=1.0,
        theta=0.25,
        delta_nfr=0.05,
        Si=0.5,
        epi_kind="prepared-form",
        epi_time_history=[(0.0, 0.5), (1.0, 0.625), (2.0, 0.875)],
        glyph_history=["IL", "OZ"],
        _grammar_u2_debt=1,
        _grammar_prior_coherence=True,
        _operator_step=2,
    )
    graph.add_node(
        _PEER,
        EPI=0.125,
        nu_f=3.0,
        theta=0.75,
        delta_nfr=0.05,
        Si=0.5,
        epi_kind="prepared-form",
        glyph_history=["AL", "IL"],
        _grammar_u2_debt=0,
        _grammar_prior_coherence=True,
        _operator_step=2,
    )
    graph.add_edge(_PARENT, _PEER, weight=0.37, supplied="old-support")
    if graph.is_multigraph():
        graph.add_edge(_PARENT, _PEER, weight=0.0, supplied="parallel-zero")
    return graph


def _act(graph, operator, node, route, **kwargs):
    if route == "public":
        assert operator(graph, node, **kwargs) is None
        return None
    if isinstance(operator, SelfOrganization):
        # Source preparation uses the concrete THOL stage so its explicitly
        # supplied tau is retained; the generic dispatcher has no tau keyword.
        executor = execute_self_organization_stage
    elif route == "dispatcher":
        executor = execute_network_operator_stage
    elif isinstance(operator, Coupling):
        executor = execute_coupling_stage
    else:
        executor = execute_neighbor_stage
    result = executor(graph, operator, (node,), **kwargs)
    assert result.nodes_processed == 1
    return result


def _form(graph, node):
    return require_finite_real_scalar_epi(get_attr(graph.nodes[node], ALIAS_EPI))


def _edge_records(graph):
    if graph.is_multigraph():
        return {
            (u, v, key): deepcopy(data)
            for u, v, key, data in graph.edges(keys=True, data=True)
        }
    return {(u, v): deepcopy(data) for u, v, data in graph.edges(data=True)}


def _plain(graph):
    metadata = dict(graph.graph)
    if "_node_cache" in metadata:
        metadata["_node_cache"] = {
            node: id(wrapper) for node, wrapper in metadata["_node_cache"].items()
        }
    return deepcopy(
        (
            tuple(graph.nodes(data=True)),
            _edge_records(graph),
            metadata,
            hasattr(graph, "_last_operator_applied"),
            getattr(graph, "_last_operator_applied", None),
        )
    )


def _birth(graph, route):
    old_support = _edge_records(graph)
    _act(graph, SelfOrganization(), _PARENT, route, tau=0.1)
    (child,) = graph.nodes[_PARENT]["sub_nodes"]
    assert graph.nodes[_PARENT]["sub_epis"][0]["node_id"] == child
    assert graph.nodes[child]["parent_node"] == _PARENT
    assert graph.degree(child) == 0 and graph.degree(_PARENT) > 0
    assert _edge_records(graph) == old_support
    # The source is supplied; the two measured slopes imply A=1/8>1/10.
    assert _form(graph, child) == THOL_SUB_EPI_SCALING * 0.875
    assert get_attr(graph.nodes[child], ALIAS_VF) == THOL_CHILD_VF_DAMPING
    assert get_attr(graph.nodes[child], ALIAS_THETA) == 0.25
    assert read_grammar_state(graph.nodes[child]).history == ()
    assert "epi_time_history" not in graph.nodes[child]
    assert not observe_structural_acceleration(graph, child).available
    return child


@pytest.mark.parametrize("route", ["public", "atomic", "dispatcher"])
@pytest.mark.parametrize(
    "graph_type", [nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph]
)
def test_real_birth_joins_augmented_um_neighborhood_without_replacing_old_relations(
    route, graph_type
):
    graph = _source(graph_type)
    child = _birth(graph, route)
    old_support = _edge_records(graph)
    old_contexts = {
        node: read_grammar_state(graph.nodes[node]) for node in (_PEER, child)
    }
    time_history = deepcopy(graph.nodes[_PARENT]["epi_time_history"])
    forms = {node: _form(graph, node) for node in graph}
    pressure = get_attr(graph.nodes[_PARENT], ALIAS_DNFR)
    _act(graph, Coupling(), _PARENT, route, potential_partner=child)

    # Independent three-participant circular mean; the old parallel edge does
    # not duplicate its node in the mean. These phases lie in one regular arc.
    consensus = math.atan2(
        2.0 * math.sin(0.25) + math.sin(0.75), 2.0 * math.cos(0.25) + math.cos(0.75)
    )
    target_phase = 0.5 * (0.25 + consensus)
    peer_phase = 0.5 * (0.75 + consensus)
    assert get_attr(graph.nodes[_PARENT], ALIAS_THETA) == pytest.approx(
        target_phase, abs=2e-15
    )
    assert get_attr(graph.nodes[child], ALIAS_THETA) == pytest.approx(
        target_phase, abs=2e-15
    )
    assert get_attr(graph.nodes[_PEER], ALIAS_THETA) == pytest.approx(
        peer_phase, abs=2e-15
    )
    expected_capacity = 1.0 + 0.25 * ((3.0 + THOL_CHILD_VF_DAMPING) / 2.0 - 1.0)
    assert get_attr(graph.nodes[_PARENT], ALIAS_VF) == expected_capacity
    assert get_attr(graph.nodes[child], ALIAS_VF) == THOL_CHILD_VF_DAMPING
    assert get_attr(graph.nodes[_PEER], ALIAS_VF) == 3.0
    mean_alignment = ((1.0 - 0.25 / math.pi) + 1.0) / 2.0
    assert get_attr(graph.nodes[_PARENT], ALIAS_DNFR) == pytest.approx(
        pressure * (1.0 - 0.25 * mean_alignment), abs=2e-15
    )
    assert {node: _form(graph, node) for node in graph} == forms
    after_support = _edge_records(graph)
    assert len(after_support) == len(old_support) + 1
    assert all(after_support[edge] == data for edge, data in old_support.items())
    assert graph.has_edge(_PARENT, child)
    if graph.is_directed():
        assert not graph.has_edge(child, _PARENT)
    assert read_grammar_state(graph.nodes[_PARENT]).history[-2:] == ("THOL", "UM")
    assert {
        node: read_grammar_state(graph.nodes[node]) for node in old_contexts
    } == old_contexts
    assert graph.nodes[_PARENT]["epi_time_history"] == time_history
    assert "epi_time_history" not in graph.nodes[child]
    assert not observe_structural_acceleration(graph, child).available
    assert graph.graph["_t"] == 2.0
    assert graph.graph[STAGE_SCHEDULE_KEY]["potential_pair"] == {
        "target": _PARENT,
        "partner": child,
        "origin": "explicit_request",
    }


@pytest.mark.parametrize("route", ["public", "atomic", "dispatcher"])
def test_ordinary_um_can_create_same_edge_but_uses_a_different_phase_capacity_map(
    route,
):
    outcomes = []
    for explicit in (False, True):
        graph = _source()
        child = _birth(graph, route)
        graph.graph["_node_sample"] = [child]
        graph.graph["UM_CANDIDATE_COUNT"] = 0
        _act(
            graph,
            Coupling(),
            _PARENT,
            route,
            **({"potential_partner": child} if explicit else {}),
        )
        assert graph.has_edge(_PARENT, child)
        assert graph.edges[_PARENT, _PEER] == {
            "weight": 0.37,
            "supplied": "old-support",
        }
        outcomes.append(
            (
                get_attr(graph.nodes[_PARENT], ALIAS_THETA),
                get_attr(graph.nodes[_PARENT], ALIAS_VF),
                get_attr(graph.nodes[child], ALIAS_THETA),
            )
        )
        assert ("potential_pair" in graph.graph.get(STAGE_SCHEDULE_KEY, {})) is explicit
    assert outcomes[0] == pytest.approx((0.375, 1.5, 0.25), abs=2e-15)
    assert outcomes[1][0] < outcomes[0][0]
    assert outcomes[1][1] < outcomes[0][1]
    assert outcomes[1][2] > outcomes[0][2]


@pytest.mark.parametrize("route", ["public", "atomic", "dispatcher"])
def test_birth_contact_changes_actual_en_then_ra_against_no_um_control(route):
    responses = []
    for connect in (False, True):
        graph = _source()
        child = _birth(graph, route)
        child_form = THOL_SUB_EPI_SCALING * 0.875
        if connect:
            _act(graph, Coupling(), _PARENT, route, potential_partner=child)
        else:
            assert graph.degree(child) == 0
        # EN's mean is unweighted: the unchanged old edge weight is not a
        # license to erase the newly realized child from its incoming support.
        neighbor_mean = (0.125 + child_form) / 2.0 if connect else 0.125
        expected_received = 0.875 * 0.75 + 0.25 * neighbor_mean
        if route == "dispatcher":
            # The ordinary dispatcher keeps telemetry enabled. Its detector
            # finds no emitter here, but the existing incoming EN form is valid.
            with pytest.warns(UserWarning, match="no emission sources detected"):
                _act(graph, Reception(), _PARENT, route)
        else:
            _act(
                graph,
                Reception(),
                _PARENT,
                route,
                **({"track_sources": False} if route != "dispatcher" else {}),
            )
        assert _form(graph, _PARENT) == expected_received
        _act(graph, Resonance(), _PEER, route)
        expected_response = 0.125 + 0.5 * (expected_received - 0.125)
        assert _form(graph, _PEER) == expected_response
        assert read_grammar_state(graph.nodes[_PARENT]).history[-1] == "EN"
        assert read_grammar_state(graph.nodes[_PEER]).history[-1] == "RA"
        assert read_grammar_state(graph.nodes[child]).history == ()
        assert "epi_time_history" not in graph.nodes[child]
        assert graph.graph["_t"] == 2.0
        responses.append(expected_response)
    assert responses[1] > responses[0]


@pytest.mark.parametrize("route", ["public", "atomic", "dispatcher"])
def test_directed_created_arc_enables_child_reception_and_parent_resonance_only(route):
    graph = _source(nx.DiGraph)
    child = _birth(graph, route)
    with pytest.raises((ValueError, OperatorPreconditionError)):
        _act(graph, Resonance(), child, route)
    _act(graph, Coupling(), _PARENT, route, potential_partner=child)
    # parent->child is incoming to the child, not to the parent.
    assert tuple(graph.predecessors(_PARENT)) == ()
    assert tuple(graph.predecessors(child)) == (_PARENT,)
    with pytest.raises((ValueError, OperatorPreconditionError)):
        _act(graph, Resonance(), child, route)
    before_parent = _form(graph, _PARENT)
    with pytest.raises((ValueError, OperatorPreconditionError)):
        _act(graph, Reception(), _PARENT, route, source_nodes=(child,))
    assert _form(graph, _PARENT) == before_parent
    before_child = _form(graph, child)
    _act(
        graph,
        Reception(),
        child,
        route,
        source_nodes=(_PARENT,),
        **({"track_sources": False} if route != "dispatcher" else {}),
    )
    assert _form(graph, child) == before_child * 0.75 + before_parent * 0.25
    assert read_grammar_state(graph.nodes[child]).history == ("EN",)
    expected_parent = before_parent + 0.5 * (
        (0.125 + _form(graph, child)) / 2.0 - before_parent
    )
    _act(graph, Resonance(), _PARENT, route)
    assert _form(graph, _PARENT) == expected_parent
    assert read_grammar_state(graph.nodes[child]).history == ("EN",)
    assert "epi_time_history" not in graph.nodes[child]


@pytest.mark.parametrize("route", ["public", "atomic", "dispatcher"])
def test_failed_connected_creation_restores_birth_support_context_and_phase_changes(
    route,
):
    graph = _source()
    child = _birth(graph, route)
    before = _plain(graph)

    def fail(subject):
        assert subject.has_edge(_PARENT, child)
        assert read_grammar_state(subject.nodes[_PARENT]).history[-1] == "UM"
        subject.nodes[_PEER]["EPI"] = 9.0
        subject.graph["late_marker"] = True
        raise RuntimeError("synthetic connected UM failure")

    with pytest.raises(RuntimeError, match="connected UM failure"):
        _act(
            graph,
            Coupling(),
            _PARENT,
            route,
            potential_partner=child,
            compute_delta_nfr=fail,
        )
    assert _plain(graph) == before
    assert graph.degree(child) == 0


@pytest.mark.parametrize("mode", ["raise", "fallback"])
def test_connected_parent_must_execute_um_without_substituting_its_creating_request(
    mode,
):
    graph = _source()
    child = _birth(graph, "atomic")
    graph.graph["GRAMMAR_REJECTION_MODE"] = mode
    graph.nodes[_PARENT]["_grammar_u2_debt"] = 3
    before = _plain(graph)
    with pytest.raises((ValueError, RuntimeError, OperatorPreconditionError)):
        _act(graph, Coupling(), _PARENT, "atomic", potential_partner=child)
    assert _plain(graph) == before


@pytest.mark.parametrize("route", ["public", "atomic", "dispatcher"])
def test_requested_child_supplies_u3_when_all_old_neighbors_are_incompatible(route):
    graph = _source()
    graph.nodes[_PEER]["theta"] = 2.75
    child = _birth(graph, route)
    peer_before = deepcopy(graph.nodes[_PEER])
    with pytest.raises((ValueError, OperatorPreconditionError)):
        _act(graph, Coupling(), _PARENT, route)
    _act(graph, Coupling(), _PARENT, route, potential_partner=child)
    assert graph.has_edge(_PARENT, child)
    assert graph.nodes[_PEER] == peer_before
    assert graph.edges[_PARENT, _PEER] == {"weight": 0.37, "supplied": "old-support"}
    assert read_grammar_state(graph.nodes[_PARENT]).history[-2:] == ("THOL", "UM")
