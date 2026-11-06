"""Independent finite controls for explicitly requested potential UM pairs.

The pair is supplied access, not acquired support or an autonomous selector.
No continuous trajectory, sampled candidate search, or physical mapping runs.
"""

from __future__ import annotations

import math
from copy import deepcopy
from itertools import product

import networkx as nx
import pytest

from tnfr.alias import get_attr
from tnfr.constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_THETA, ALIAS_VF
from tnfr.operators import apply_glyph
from tnfr.operators.definitions import (
    Coherence,
    Coupling,
    Emission,
    Reception,
    Resonance,
    Silence,
)
from tnfr.operators.definitions_base import Operator
from tnfr.operators.grammar_debt import read_grammar_state
from tnfr.operators.grammar_execution import ValidatedSequence
from tnfr.operators.network_stage import (
    STAGE_SCHEDULE_KEY,
    TWO_PHASE_JACOBI,
    execute_coupling_stage,
)
from tnfr.operators.preconditions import OperatorPreconditionError
from tnfr.operators.word_execution import execute_network_operator_stage
from tnfr.physics.integrity import MonitorMode, enable_integrity_monitor
from tnfr.types import require_finite_real_scalar_epi


def _source(graph_type=nx.Graph):
    graph = graph_type(
        UM_FUNCTIONAL_LINKS=True,
        UM_BIDIRECTIONAL=True,
        UM_SYNC_VF=True,
        UM_STABILIZE_DNFR=True,
        UM_COMPAT_THRESHOLD=0.5,
        VALIDATE_OPERATOR_PRECONDITIONS=True,
        GRAMMAR_REJECTION_MODE="raise",
        RANDOM_SEED=17,
        GLYPH_FACTORS={
            "UM_theta_push": 0.5,
            "UM_vf_sync": 0.25,
            "UM_dnfr_reduction": 0.25,
            "RA_epi_diff": 0.25,
            "RA_vf_amplification": 0.125,
            "RA_phase_coupling": 0.5,
            "EN_mix": 0.25,
        },
    )
    for node, form, capacity, phase in ((0, 0.25, 1.0, 0.125), (1, 0.75, 2.0, 0.375)):
        graph.add_node(
            node,
            EPI=form,
            nu_f=capacity,
            theta=phase,
            delta_nfr=0.4,
            Si=0.5,
            epi_kind="prepared-form",
            glyph_history=["AL", "IL"],
            _grammar_u2_debt=0,
            _grammar_prior_coherence=True,
            _operator_step=2,
        )
    return graph


def _plain(graph):
    edges = (
        tuple(
            (u, v, key, deepcopy(data))
            for u, v, key, data in graph.edges(keys=True, data=True)
        )
        if graph.is_multigraph()
        else tuple((u, v, deepcopy(data)) for u, v, data in graph.edges(data=True))
    )
    return (
        tuple((node, deepcopy(data)) for node, data in graph.nodes(data=True)),
        edges,
        deepcopy(graph.graph),
        hasattr(graph, "_last_operator_applied"),
        getattr(graph, "_last_operator_applied", None),
    )


def _apply(graph, route, *, partner=1, operator=None, **kwargs):
    selected = Coupling() if operator is None else operator
    if route == "public":
        assert selected(graph, 0, potential_partner=partner, **kwargs) is None
        return None
    executor = (
        execute_coupling_stage if route == "stage" else execute_network_operator_stage
    )
    result = executor(graph, selected, (0,), potential_partner=partner, **kwargs)
    assert result.schedule == TWO_PHASE_JACOBI
    assert result.nodes_processed == 1
    return result


def _form(data):
    return require_finite_real_scalar_epi(get_attr(data, ALIAS_EPI), "test form")


@pytest.mark.parametrize("route", ["public", "stage", "dispatcher"])
@pytest.mark.parametrize(
    "graph_type", [nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph]
)
def test_supplied_pair_creates_only_its_relation_and_target_occurrence(
    route, graph_type
):
    graph = _source(graph_type)
    partner_history = read_grammar_state(graph.nodes[1])
    _apply(graph, route)

    assert tuple(graph) == (0, 1)
    assert graph.number_of_edges() == 1
    assert graph.has_edge(0, 1)
    assert (_form(graph.nodes[0]), _form(graph.nodes[1])) == (0.25, 0.75)
    assert graph.nodes[0]["epi_kind"] == graph.nodes[1]["epi_kind"] == "prepared-form"
    assert read_grammar_state(graph.nodes[0]).history == ("AL", "IL", "UM")
    assert read_grammar_state(graph.nodes[0]).operator_step == 3
    assert read_grammar_state(graph.nodes[1]) == partner_history
    assert graph.graph[STAGE_SCHEDULE_KEY]["potential_pair"] == {
        "target": 0,
        "partner": 1,
        "origin": "explicit_request",
    }


@pytest.mark.parametrize("route", ["public", "stage"])
@pytest.mark.parametrize(
    "bidirectional, sync_vf, stabilize", list(product((False, True), repeat=3))
)
def test_pair_channels_and_affinity_match_independent_two_point_algebra(
    route, bidirectional, sync_vf, stabilize
):
    graph = _source()
    graph.graph.update(
        UM_BIDIRECTIONAL=bidirectional,
        UM_SYNC_VF=sync_vf,
        UM_STABILIZE_DNFR=stabilize,
    )
    _apply(graph, route)

    # Interior phases avoid wrapping. With bidirectionality the two-point
    # circular mean is 1/4; otherwise the source approaches its partner's 3/8.
    target_phase, partner_phase = (0.1875, 0.3125) if bidirectional else (0.25, 0.375)
    assert get_attr(graph.nodes[0], ALIAS_THETA) == pytest.approx(
        target_phase, abs=1e-14
    )
    assert get_attr(graph.nodes[1], ALIAS_THETA) == pytest.approx(
        partner_phase, abs=1e-14
    )
    assert get_attr(graph.nodes[0], ALIAS_VF) == (1.25 if sync_vf else 1.0)
    assert get_attr(graph.nodes[1], ALIAS_VF) == 2.0
    affinity = 1.0 - (partner_phase - target_phase) / math.pi
    expected_pressure = 0.4 * (1.0 - 0.25 * affinity) if stabilize else 0.4
    assert get_attr(graph.nodes[0], ALIAS_DNFR) == pytest.approx(
        expected_pressure, abs=1e-14
    )
    assert get_attr(graph.nodes[1], ALIAS_DNFR) == 0.4
    epi_similarity = 1.0 - 0.5 / (1.0 + 1e-9)
    expected_weight = 0.5 * affinity + 0.25 * epi_similarity + 0.25
    assert graph.edges[0, 1]["weight"] == pytest.approx(expected_weight, abs=1e-14)


@pytest.mark.parametrize("route", ["public", "stage", "glyph"])
def test_omitting_potential_access_does_not_bypass_the_old_isolate_gate(route):
    graph = _source()
    before = deepcopy(dict(graph.nodes(data=True)))
    with pytest.raises((ValueError, OperatorPreconditionError)):
        if route == "glyph":
            apply_glyph(graph, 0, "UM")
        elif route == "public":
            Coupling()(graph, 0)
        else:
            execute_coupling_stage(graph, Coupling(), (0,))
    assert dict(graph.nodes(data=True)) == before
    assert graph.number_of_edges() == 0


@pytest.mark.parametrize("route", ["public", "stage"])
def test_omitted_and_explicit_none_preserve_the_existing_connected_domain(route):
    omitted, explicit = _source(), _source()
    for graph in (omitted, explicit):
        graph.add_edge(0, 1, weight=0.37, supplied=True)
    if route == "public":
        Coupling()(omitted, 0)
        Coupling()(explicit, 0, potential_partner=None)
    else:
        execute_coupling_stage(omitted, Coupling(), (0,))
        execute_coupling_stage(explicit, Coupling(), (0,), potential_partner=None)
    for node in omitted:
        for aliases in (ALIAS_EPI, ALIAS_VF, ALIAS_THETA, ALIAS_DNFR):
            assert get_attr(omitted.nodes[node], aliases) == get_attr(
                explicit.nodes[node], aliases
            )
        assert read_grammar_state(omitted.nodes[node]) == read_grammar_state(
            explicit.nodes[node]
        )
    assert (
        omitted.edges[0, 1]
        == explicit.edges[0, 1]
        == {"weight": 0.37, "supplied": True}
    )
    assert "potential_pair" not in explicit.graph.get(STAGE_SCHEDULE_KEY, {})


@pytest.mark.parametrize("route", ["public", "stage"])
@pytest.mark.parametrize(
    "failure",
    [
        "unknown",
        "same",
        "partner_edge",
        "incompatible",
        "tight_limit",
        "bad_phase",
        "bad_capacity",
        "low_partner_activity",
        "links_disabled",
        "affinity",
    ],
)
def test_rejected_creation_preserves_the_complete_owned_graph(route, failure):
    graph = _source()
    partner = 1
    if failure == "unknown":
        partner = 99
    elif failure == "same":
        partner = 0
    elif failure == "partner_edge":
        graph.add_node(2, **deepcopy(graph.nodes[0]))
        graph.add_edge(1, 2, weight=0.25)
    elif failure == "incompatible":
        graph.nodes[1]["theta"] = math.pi
    elif failure == "tight_limit":
        graph.graph["UM_MAX_PHASE_DIFF"] = 0.1
    elif failure == "bad_phase":
        graph.nodes[1][ALIAS_THETA[0]] = True
        graph.nodes[1][ALIAS_THETA[-1]] = 0.25
    elif failure == "bad_capacity":
        graph.nodes[1][ALIAS_VF[0]] = -1.0
    elif failure == "low_partner_activity":
        graph.nodes[1]["EPI"] = 0.001
    elif failure == "links_disabled":
        graph.graph["UM_FUNCTIONAL_LINKS"] = False
    else:
        graph.graph["UM_COMPAT_THRESHOLD"] = 1.0
    before = _plain(graph)
    with pytest.raises((ValueError, TypeError, KeyError, OperatorPreconditionError)):
        _apply(graph, route, partner=partner)
    assert _plain(graph) == before


@pytest.mark.parametrize("endpoint", [0, 1])
def test_incoming_directed_support_excludes_only_the_required_isolated_partner(
    endpoint,
):
    graph = _source(nx.DiGraph)
    graph.add_node(2, **deepcopy(graph.nodes[0]))
    graph.add_edge(2, endpoint, weight=0.6)
    assert tuple(graph.successors(endpoint)) == ()
    before = _plain(graph)
    if endpoint == 0:
        # Incoming-only target access is not a consumed UM neighbor. The new
        # domain permits it while preserving that existing directed relation.
        _apply(graph, "stage")
        assert graph.edges[2, 0] == {"weight": 0.6}
        assert graph.has_edge(0, 1) and graph.number_of_edges() == 2
        assert graph.nodes[2] == before[0][2][1]
        return
    with pytest.raises((ValueError, OperatorPreconditionError)):
        _apply(graph, "stage")
    assert _plain(graph) == before


@pytest.mark.parametrize("route", ["public", "stage"])
@pytest.mark.parametrize("rejection_mode", ["raise", "fallback"])
def test_potential_pair_cannot_replace_blocked_grammar_with_a_successful_fallback(
    route, rejection_mode
):
    graph = _source()
    graph.graph["GRAMMAR_REJECTION_MODE"] = rejection_mode
    graph.nodes[0]["_grammar_u2_debt"] = 3
    before = _plain(graph)
    with pytest.raises((ValueError, RuntimeError)):
        _apply(graph, route)
    assert _plain(graph) == before


@pytest.mark.parametrize("route", ["public", "dispatcher"])
def test_validated_word_context_is_bound_to_the_requested_um_step(route):
    graph = _source()
    word = ValidatedSequence(
        [Emission(), Coupling(), Resonance(), Coherence(), Silence()]
    )
    before = _plain(graph)
    with pytest.raises(ValueError, match="context"):
        _apply(graph, route, sequence_context=word.step(2))
    assert _plain(graph) == before
    _apply(graph, route, sequence_context=word.step(1))
    assert graph.has_edge(0, 1)
    assert read_grammar_state(graph.nodes[0]).history[-1] == "UM"


@pytest.mark.parametrize("route", ["public", "stage"])
def test_late_callback_failure_rolls_back_pair_channels_history_and_provenance(route):
    graph = _source()
    before = _plain(graph)

    def fail_after_commit(subject):
        assert subject.has_edge(0, 1)
        assert subject.nodes[0]["glyph_history"][-1] == "UM"
        subject.nodes[1]["EPI"] = 9.0
        subject.graph["late_marker"] = True
        raise RuntimeError("late potential refresh failed")

    with pytest.raises(RuntimeError, match="late potential refresh failed"):
        _apply(graph, route, compute_delta_nfr=fail_after_commit)
    assert _plain(graph) == before


@pytest.mark.parametrize("route", ["public", "stage"])
def test_explicit_partner_does_not_consume_the_sampled_candidate_inventory(route):
    graph = _source()
    graph.graph.update(
        _node_sample=[99], UM_CANDIDATE_COUNT="unused", UM_CANDIDATE_MODE="unused"
    )
    _apply(graph, route)
    assert graph.has_edge(0, 1)
    assert graph.graph["RANDOM_SEED"] == 17
    assert graph.graph["_node_sample"] == [99]


@pytest.mark.parametrize("route", ["public", "stage"])
def test_custom_um_implementation_is_not_silently_replaced_by_the_pair_kernel(route):
    class DifferentCoupling(Coupling):
        def _execute(self, graph, node, **kwargs):
            pytest.fail(
                "custom implementation must not enter the canonical potential branch"
            )

    graph = _source()
    before = _plain(graph)
    with pytest.raises(ValueError, match="canonical"):
        _apply(graph, route, operator=DifferentCoupling())
    assert _plain(graph) == before


@pytest.mark.parametrize("targets", [(), (0, 1)])
def test_potential_stage_requires_one_target(targets):
    graph = _source()
    before = _plain(graph)
    with pytest.raises(ValueError, match="one target"):
        execute_coupling_stage(graph, Coupling(), targets, potential_partner=1)
    assert _plain(graph) == before


@pytest.mark.parametrize("graph_type", [nx.Graph, nx.DiGraph])
def test_actual_relation_enables_existing_ra_without_changing_its_support_rule(
    graph_type,
):
    graph = _source(graph_type)
    for data in graph.nodes.values():
        data["theta"] = 0.0
    with pytest.raises((ValueError, OperatorPreconditionError)):
        Resonance()(graph, 0)
    _apply(graph, "public")
    Resonance()(graph, 0)
    assert _form(graph.nodes[0]) == 0.25 + 0.25 * (0.75 - 0.25)
    assert graph.number_of_edges() == 1
    assert read_grammar_state(graph.nodes[0]).history[-2:] == ("UM", "RA")
    if graph.is_directed():
        # UM creates 0 -> 1, not its inverse. RA reads outgoing neighbors.
        with pytest.raises((ValueError, OperatorPreconditionError)):
            Resonance()(graph, 1)


def test_directed_reception_uses_the_new_incoming_relation_without_an_added_filter():
    graph = _source(nx.DiGraph)
    graph.nodes[1]["delta_nfr"] = 0.1
    _apply(graph, "stage")
    Reception()(graph, 1, track_sources=False)
    assert _form(graph.nodes[1]) == 0.75 + 0.25 * (0.25 - 0.75)
    assert _form(graph.nodes[0]) == 0.25
    assert tuple(graph.edges) == ((0, 1),)
    assert read_grammar_state(graph.nodes[1]).history[-1] == "EN"


def test_dispatcher_rejects_potential_access_for_ra_before_any_mutation():
    graph = _source()
    before = _plain(graph)
    with pytest.raises(ValueError, match="only for Coupling"):
        execute_network_operator_stage(graph, Resonance(), (0,), potential_partner=1)
    assert _plain(graph) == before


@pytest.mark.parametrize("route", ["public", "stage"])
@pytest.mark.parametrize("mode", tuple(MonitorMode))
def test_potential_creation_retains_the_real_integrity_monitor(route, mode):
    graph = _source()
    monitor = enable_integrity_monitor(
        graph,
        mode=mode,
        conservation_threshold=0.0,
        lyapunov_tolerance=1e6,
        charge_drift_threshold=1e6,
    )
    _apply(graph, route)
    assert graph.has_edge(0, 1)
    assert graph.graph["integrity_monitor"] is monitor
    assert monitor.mode is mode
    if mode is MonitorMode.OFF:
        assert monitor.latest_report is None
    else:
        assert monitor.latest_report.postcondition_evaluated
        assert monitor.latest_report.postcondition_ok
        assert monitor.summary.total_operators == 1


@pytest.mark.parametrize("route", ["public", "stage"])
def test_changed_base_execute_cannot_be_bypassed_by_the_pair_stage(route, monkeypatch):
    graph = _source()
    before = _plain(graph)

    def changed_execute(self, subject, node, **kwargs):
        pytest.fail("changed execution hook must not be silently omitted")

    monkeypatch.setattr(Operator, "_execute", changed_execute)
    with pytest.raises(ValueError, match="canonical"):
        _apply(graph, route)
    assert _plain(graph) == before
