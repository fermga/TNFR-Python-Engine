"""Actual AL history, eligible THOL birth and continued child participation.

Times, coefficients and access are declared interventions. Birth does not
transfer the parent's grammar, physical trajectory or Emission activation.
"""

from copy import deepcopy
from fractions import Fraction as Q

import networkx as nx
import pytest

from tnfr.alias import get_attr, set_attr
from tnfr.constants import inject_defaults
from tnfr.constants.aliases import (
    ALIAS_D2EPI,
    ALIAS_DEPI,
    ALIAS_DNFR,
    ALIAS_EPI,
    ALIAS_THETA,
    ALIAS_VF,
)
from tnfr.errors import TNFRValueError
from tnfr.operators.coherence import Coherence
from tnfr.operators.coupling import Coupling
from tnfr.operators.dissonance import Dissonance
from tnfr.operators.emission_episode import execute_emission_episode
from tnfr.operators.grammar_debt import read_grammar_state
from tnfr.operators.network_stage import (
    STAGE_SCHEDULE_KEY,
    execute_coupling_stage,
    execute_dissonance_stage,
    execute_neighbor_stage,
    execute_pointwise_stage,
)
from tnfr.operators.nodal_equation import observe_structural_acceleration
from tnfr.operators.preconditions.reception import validate_reception_strict
from tnfr.operators.reception import Reception
from tnfr.operators.resonance import Resonance
from tnfr.operators.self_organization_selection import (
    execute_eligible_self_organization_stage,
)
from tnfr.operators.word_execution import execute_network_operator_stage
from tnfr.types import require_finite_real_scalar_epi

PARENT, PEER = "parent", "peer"
NODES = (PARENT, PEER)


def _form(graph, node):
    return Q.from_float(
        require_finite_real_scalar_epi(
            get_attr(
                graph.nodes[node], ALIAS_EPI, strict=True, conv=lambda value: value
            )
        )
    )


def _pressure(graph, node):
    return Q.from_float(get_attr(graph.nodes[node], ALIAS_DNFR))


def _cohere(graph, targets):
    execute_pointwise_stage(
        graph,
        Coherence(),
        targets,
        phase_locking_coefficient=0.0,
        require_exact_operator=True,
        validate_preconditions=True,
    )


def _prepare_pressure(graph, targets, factor):
    _cohere(graph, targets)
    graph.graph["GLYPH_FACTORS"]["OZ_dnfr_factor"] = factor
    execute_dissonance_stage(
        graph,
        Dissonance(),
        targets,
        propagate_to_network=False,
        validate_preconditions=True,
    )


def _episode(graph):
    return execute_emission_episode(
        graph, NODES, duration=0.125, method="euler", n_jobs=1
    )


def _source():
    graph = nx.Graph()
    graph.add_edge(PARENT, PEER, weight=1.0)
    inject_defaults(graph)
    graph.graph.update(
        _t=0.0,
        DT_MIN=0.0,
        INTEGRATOR_METHOD="euler",
        GAMMA={"type": "none"},
        RANDOM_SEED=71,
        VALIDATE_OPERATOR_PRECONDITIONS=True,
        GRAMMAR_CANON={"enabled": True},
        GRAMMAR_REJECTION_MODE="raise",
        EDGE_AWARE_ENABLED=True,
        OZ_NOISE_MODE=False,
        IL_PRECONDITIONS={"warn_zero_dnfr": False},
        CLIP_MODE="hard",
        EPI_MIN=-1.0,
        EPI_MAX=1.0,
        THOL_METABOLIC_ENABLED=False,
        THOL_PROPAGATION_ENABLED=False,
        UM_FUNCTIONAL_LINKS=True,
        UM_BIDIRECTIONAL=True,
        UM_SYNC_VF=True,
        UM_STABILIZE_DNFR=True,
        UM_COMPAT_THRESHOLD=0.5,
        history={},
    )
    graph.graph["GLYPH_FACTORS"].update(
        IL_dnfr_factor=0.5,
        OZ_dnfr_factor=2.0,
        THOL_accel=0.125,
        UM_theta_push=0.5,
        UM_vf_sync=0.25,
        UM_dnfr_reduction=0.25,
        EN_mix=0.25,
        RA_epi_diff=0.5,
        RA_vf_amplification=0.125,
    )
    for node in NODES:
        graph.nodes[node].update(
            EPI=0.25,
            nu_f=1.0,
            theta=0.0,
            delta_nfr=0.5,
            Si=0.0,
            epi_kind="form",
        )
        set_attr(graph.nodes[node], ALIAS_DEPI, 0.0)
        set_attr(graph.nodes[node], ALIAS_D2EPI, 0.0)
    _prepare_pressure(graph, NODES, 2.0)
    assert all("epi_time_history" not in graph.nodes[n] for n in NODES)
    first = _episode(graph)
    assert first.exact_rates == (Q(1, 2), Q(1, 2))
    assert first.exact_epi_after == (Q(5, 16), Q(5, 16))
    _prepare_pressure(graph, (PARENT,), 2.5)
    second = _episode(graph)
    assert second.exact_rates == (Q(5, 8), Q(1, 2))
    assert second.exact_epi_after == (Q(25, 64), Q(3, 8))
    for result in (first, second):
        assert result.flow_result.events == ()
        assert result.flow_result.pressure_refresh_callback_invocations == 0
        assert (
            result.flow_result.all_positive_flow_intervals_binary64_held_pressure_identified
        )
    for node, endpoint, acceleration, history in (
        (PARENT, Q(25, 64), 1.0, ("IL", "OZ", "AL", "IL", "OZ", "AL")),
        (PEER, Q(3, 8), 0.0, ("IL", "OZ", "AL", "AL")),
    ):
        data = graph.nodes[node]
        assert _form(graph, node) == endpoint
        assert tuple(data["epi_time_history"]) == (
            (0.0, 0.25),
            (0.125, 0.3125),
            (0.25, float(endpoint)),
        )
        observation = observe_structural_acceleration(graph, node)
        assert observation.available and observation.source == "epi_time_history"
        assert observation.value == acceleration
        grammar = read_grammar_state(data)
        assert grammar.history == history and grammar.operator_step == len(history)
        assert grammar.u2_debt == 1 and grammar.prior_coherence
        assert data["_structural_lineage"]["activation_count"] == 2
    assert graph.graph["_t"] == 0.25
    return graph


def _birth(graph):
    physical = {node: deepcopy(graph.nodes[node]["epi_time_history"]) for node in NODES}
    dispatch = execute_eligible_self_organization_stage(
        graph, execution_kwargs={"tau": 0.5, "validate_preconditions": True}
    )
    assert dispatch.policy == "all_eligible_once"
    assert dispatch.eligibility.eligible_nodes == (PARENT,)
    assert tuple(row.node for row in dispatch.eligibility.candidates) == NODES
    parent_candidate, peer_candidate = dispatch.eligibility.candidates
    assert parent_candidate.acceleration.value == 1.0 and parent_candidate.eligible
    assert (
        peer_candidate.acceleration.available
        and peer_candidate.acceleration.value == 0.0
    )
    assert peer_candidate.threshold_crossed is False and not peer_candidate.eligible
    assert dispatch.stage_result.nodes_processed == 1
    ((parent, child),) = dispatch.parent_children
    assert parent == PARENT and tuple(graph) == (*NODES, child)
    assert graph.degree(child) == 0 and tuple(graph.edges) == ((PARENT, PEER),)
    assert graph.nodes[PARENT]["sub_nodes"] == [child]
    assert graph.graph["hierarchy"][PARENT] == [child]
    record = graph.nodes[PARENT]["sub_epis"][0]
    assert record["node_id"] == child and record["timestamp"] == 7
    assert record["d2_epi"] == 1.0 and record["tau"] == 0.5
    assert record["metabolized"] is False
    assert _pressure(graph, PARENT) == Q(3, 4)
    assert _pressure(graph, PEER) == Q(1, 2)
    for node in NODES:
        assert graph.nodes[node]["epi_time_history"] == physical[node]
        assert graph.nodes[node]["_structural_lineage"]["activation_count"] == 2
    assert read_grammar_state(graph.nodes[PARENT]).operator_step == 7
    assert read_grammar_state(graph.nodes[PARENT]).u2_debt == 0
    data = graph.nodes[child]
    assert data["parent_node"] == PARENT
    assert _form(graph, child) == Q(15, 128)
    assert get_attr(data, ALIAS_VF) == 0.95
    assert _pressure(graph, child) == 0 and get_attr(data, ALIAS_THETA) == 0.0
    assert data["epi_history"] == [15 / 128]
    grammar = read_grammar_state(data)
    assert grammar.history == () and grammar.operator_step == grammar.u2_debt == 0
    assert not grammar.prior_coherence
    assert "epi_time_history" not in data and "_structural_lineage" not in data
    observation = observe_structural_acceleration(graph, child)
    assert not observation.available and observation.history_length == 1
    assert graph.graph["_t"] == 0.25
    return child, physical


def _act(graph, operator, node, route, **kwargs):
    if route == "public":
        if isinstance(operator, Reception):
            kwargs["track_sources"] = False
        return operator(graph, node, **kwargs)
    if route == "dispatcher":
        return execute_network_operator_stage(graph, operator, (node,), **kwargs)
    if isinstance(operator, Coupling):
        return execute_coupling_stage(graph, operator, (node,), **kwargs)
    if isinstance(operator, Reception):
        kwargs["track_sources"] = False
    return execute_neighbor_stage(graph, operator, (node,), **kwargs)


def _contact_and_admit(graph, child, physical, route, *, connect):
    if connect:
        _act(graph, Coupling(), PARENT, route, potential_partner=child)
        assert graph.has_edge(PARENT, child)
        assert graph.graph[STAGE_SCHEDULE_KEY]["potential_pair"] == {
            "target": PARENT,
            "partner": child,
            "origin": "explicit_request",
        }
        # Independent affinity formula; scaled production evaluation may differ
        # by a few rounding units, so this ancillary weight gets a stated bound.
        x, c = 25 / 64, 15 / 128
        expected_weight = 0.75 + 0.25 * (1.0 - abs(x - c) / (x + c + 1e-9))
        assert graph.edges[PARENT, child]["weight"] == pytest.approx(
            expected_weight, rel=0.0, abs=2e-15
        )
        assert Q.from_float(get_attr(graph.nodes[PARENT], ALIAS_VF)) == Q(
            8950904259398861, 2**53
        )
        assert _pressure(graph, PARENT) == Q(9, 16)
    else:
        assert graph.degree(child) == 0
        assert get_attr(graph.nodes[PARENT], ALIAS_VF) == 1.0
        assert _pressure(graph, PARENT) == Q(3, 4)
    assert graph.edges[PARENT, PEER] == {"weight": 1.0}
    initial_pressure = Q(9, 16) if connect else Q(3, 4)
    for count in range(1, 4):
        _cohere(graph, (PARENT,))
        assert _pressure(graph, PARENT) == initial_pressure / 2**count
        if count == 2:
            if connect:
                validate_reception_strict(graph, PARENT)
            else:
                with pytest.raises(TNFRValueError, match="DNFR"):
                    validate_reception_strict(graph, PARENT)
    validate_reception_strict(graph, PARENT)
    assert _form(graph, PARENT) == Q(25, 64) and _form(graph, PEER) == Q(3, 8)
    assert _form(graph, child) == Q(15, 128)
    for node in graph:
        assert get_attr(graph.nodes[node], ALIAS_THETA) == 0.0
    for node in NODES:
        assert graph.nodes[node]["epi_time_history"] == physical[node]
    assert read_grammar_state(graph.nodes[PARENT]).operator_step == 10 + int(connect)
    assert read_grammar_state(graph.nodes[child]).history == ()
    assert graph.graph["_t"] == 0.25


@pytest.mark.parametrize("route", ["public", "atomic", "dispatcher"])
def test_actual_birth_contact_child_input_changes_parent_and_old_peer(route):
    results = {}
    for arm in ("no_contact", "all_inputs", "peer_only"):
        graph = _source()
        child, physical = _birth(graph)
        connect = arm != "no_contact"
        _contact_and_admit(graph, child, physical, route, connect=connect)
        child_before = deepcopy(graph.nodes[child])
        kwargs = {"source_nodes": (PEER,)} if arm == "peer_only" else {}
        _act(graph, Reception(), PARENT, route, **kwargs)
        expected_parent = Q(363, 1024) if arm == "all_inputs" else Q(99, 256)
        assert _form(graph, PARENT) == expected_parent
        assert _pressure(graph, PARENT) == (Q(9, 128) if connect else Q(3, 32))
        _act(graph, Resonance(), PEER, route)
        expected_peer = Q(747, 2048) if arm == "all_inputs" else Q(195, 512)
        assert _form(graph, PEER) == expected_peer
        assert get_attr(graph.nodes[PEER], ALIAS_VF) == 9 / 8
        assert get_attr(graph.nodes[PEER], ALIAS_THETA) == 0.0
        assert _pressure(graph, PEER) == Q(1, 2)
        assert read_grammar_state(graph.nodes[PARENT]).operator_step == 11 + int(
            connect
        )
        assert read_grammar_state(graph.nodes[PARENT]).history[-1] == "EN"
        assert read_grammar_state(graph.nodes[PEER]).operator_step == 5
        assert read_grammar_state(graph.nodes[PEER]).history[-1] == "RA"
        # Merely being read by other targets supplies no child occurrence.
        assert graph.nodes[child] == child_before
        for node in NODES:
            assert graph.nodes[node]["_structural_lineage"]["activation_count"] == 2
            assert graph.nodes[node]["epi_time_history"] == physical[node]
            with pytest.raises(TNFRValueError, match="final EPI"):
                observe_structural_acceleration(graph, node)
        assert graph.graph["_t"] == 0.25
        results[arm] = (_form(graph, PARENT), _form(graph, PEER))
    assert results["peer_only"] == results["no_contact"]
    assert results["all_inputs"][0] - results["peer_only"][0] == -Q(33, 1024)
    assert results["all_inputs"][1] - results["peer_only"][1] == -Q(33, 2048)


@pytest.mark.parametrize("route", ["public", "atomic", "dispatcher"])
def test_actual_born_child_executes_own_reception_then_resonance(route):
    graph = _source()
    child, physical = _birth(graph)
    _contact_and_admit(graph, child, physical, route, connect=True)
    _act(graph, Reception(), child, route, source_nodes=(PARENT,))
    assert _form(graph, child) == Q(95, 512)
    assert read_grammar_state(graph.nodes[child]).history == ("EN",)
    assert get_attr(graph.nodes[child], ALIAS_VF) == 0.95
    _act(graph, Resonance(), child, route)
    assert _form(graph, child) == Q(295, 1024)
    expected_capacity = float(Q.from_float(0.95) * Q(9, 8))
    assert get_attr(graph.nodes[child], ALIAS_VF) == expected_capacity
    assert _pressure(graph, child) == 0
    assert get_attr(graph.nodes[child], ALIAS_THETA) == 0.0
    grammar = read_grammar_state(graph.nodes[child])
    assert grammar.history == ("EN", "RA") and grammar.operator_step == 2
    assert "epi_time_history" not in graph.nodes[child]
    assert "_structural_lineage" not in graph.nodes[child]
    assert not graph.nodes[child].get("_emission_activated", False)
    assert _form(graph, PARENT) == Q(25, 64) and _form(graph, PEER) == Q(3, 8)
    for node in NODES:
        assert graph.nodes[node]["epi_time_history"] == physical[node]
        assert graph.nodes[node]["_structural_lineage"]["activation_count"] == 2
    assert observe_structural_acceleration(graph, PARENT).value == 1.0
    assert observe_structural_acceleration(graph, PEER).value == 0.0
    assert graph.graph["_t"] == 0.25
