"""Independent finite controls for configured EN compatibility and occurrence.

The request time and thresholds are supplied. These are operator events, not
an acquired source, a continuous trajectory, or an autonomous selector.
"""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import replace

import networkx as nx
import pytest

from tnfr.alias import get_attr
from tnfr.constants.aliases import ALIAS_EPI, ALIAS_EPI_KIND, ALIAS_THETA, ALIAS_VF
from tnfr.operators.definitions import Coupling, Reception, Resonance, Silence
from tnfr.operators.grammar_debt import read_grammar_state
from tnfr.operators.metrics_basic import reception_metrics
from tnfr.operators.network_stage import execute_neighbor_stage
from tnfr.operators.preconditions import OperatorPreconditionError
from tnfr.operators.reception_selection import (
    EligibleReceptionDispatch,
    ReceptionCompatibilityPolicy,
    execute_eligible_reception_stage,
    observe_reception_compatibility,
)
from tnfr.types import require_finite_real_scalar_epi
from tnfr.utils.io import json_dumps, json_loads


def _graph(graph_type=nx.DiGraph):
    graph = graph_type(
        GLYPH_FACTORS={"EN_mix": 0.25, "RA_epi_diff": 0.5},
        GRAMMAR_REJECTION_MODE="raise",
        VALIDATE_OPERATOR_PRECONDITIONS=True,
        EPI_MIN=-2.0,
        EPI_MAX=2.0,
    )
    for node, value in enumerate((0.25, 0.75, -0.25, 0.25)):
        graph.add_node(
            node,
            EPI=value,
            nu_f=1.0,
            theta=0.0,
            delta_nfr=0.05,
            epi_kind=f"kind-{node}",
            glyph_history=["AL", "IL"],
            _grammar_u2_debt=0,
            _grammar_prior_coherence=True,
            _operator_step=2,
        )
    graph.nodes[2]["theta"] = math.pi
    graph.nodes[3]["nu_f"] = 0.5
    for source in (1, 2, 3):
        graph.add_edge(source, 0, weight=1.0)
    return graph


def _policy(mode="select"):
    return ReceptionCompatibilityPolicy(0.25, 0.0, mode=mode)


def _form(graph, node=0):
    return require_finite_real_scalar_epi(get_attr(graph.nodes[node], ALIAS_EPI))


def _plain(graph):
    edges = (
        tuple(graph.edges(keys=True, data=True))
        if graph.is_multigraph()
        else tuple(graph.edges(data=True))
    )
    metadata = dict(graph.graph)
    if "_node_cache" in metadata:
        # Live NodeNX wrappers have identity semantics; deep-copying two
        # snapshots would manufacture unequal wrappers despite no mutation.
        metadata["_node_cache"] = {
            node: id(wrapper) for node, wrapper in metadata["_node_cache"].items()
        }
    return deepcopy(
        (
            tuple(graph.nodes(data=True)),
            edges,
            metadata,
            hasattr(graph, "_last_operator_applied"),
            getattr(graph, "_last_operator_applied", None),
        )
    )


def _attempt(graph, policy=None, **kwargs):
    return execute_eligible_reception_stage(
        graph,
        0,
        _policy() if policy is None else policy,
        execution_kwargs={"track_sources": False, **kwargs.pop("options", {})},
        **kwargs,
    )


@pytest.mark.parametrize(
    "graph_type", [nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph]
)
def test_selected_phase_capacity_sources_have_real_form_kind_and_history_effect(
    graph_type,
):
    graph = _graph(graph_type)
    peers = {node: deepcopy(graph.nodes[node]) for node in (1, 2, 3)}
    result = _attempt(graph, options={"collect_metrics": True})
    assert result.executed
    assert result.compatibility.selected_sources == (1,)
    assert result.compatibility.candidates == (1, 2, 3)
    assert result.compatibility.candidate_capacities == (1.0, 1.0, 0.5)
    assert result.stage_result.nodes_processed == 1
    assert _form(graph) == 0.375
    assert (
        get_attr(graph.nodes[0], ALIAS_EPI_KIND, strict=True, conv=lambda value: value)
        == "kind-1"
    )
    assert {node: graph.nodes[node] for node in peers} == peers
    assert read_grammar_state(graph.nodes[0]).history == ("AL", "IL", "EN")
    assert read_grammar_state(graph.nodes[0]).operator_step == 3
    (observation,) = result.stage_result.reception_observations
    assert observation.source_nodes == observation.neighbors == (1,)
    assert observation.source_selection_origin == "compatibility_policy"
    assert observation.compatibility_decision == result.compatibility
    assert observation._proof_fields_are_intact()
    metrics = graph.graph["operator_metrics"][-1]
    assert metrics["source_selection_origin"] == "compatibility_policy"
    # Metrics are ordinary serializable diagnostics, while observation above
    # retains the sealed decision. JSON metadata must not pretend to be proof.
    decoded = json_loads(json_dumps(metrics, allow_nan=False))
    selection = decoded["compatibility_decision"]
    assert selection["node"] == 0
    assert selection["candidates"] == [1, 2, 3]
    assert selection["candidate_phases"] == [0.0, math.pi, 0.0]
    assert selection["candidate_capacities"] == [1.0, 1.0, 0.5]
    assert selection["selected_sources"] == [1]
    assert selection["policy"] == {
        "max_phase_gap": 0.25,
        "max_relative_capacity_gap": 0.0,
        "mode": "select",
    }
    assert "_proof_stamp" not in selection
    live = reception_metrics(graph, 0, 0.25)
    assert live["compatibility_decision"] is None


@pytest.mark.parametrize(
    "case,mode,reason",
    [
        ("empty", "select", "no_candidates"),
        ("empty", "require_all", "no_candidates"),
        ("incompatible", "select", "no_compatible_sources"),
        ("partial", "require_all", "not_all_compatible"),
    ],
)
def test_source_abstention_has_no_operator_callback_history_or_schedule(
    case, mode, reason
):
    graph = _graph()
    if case == "empty":
        graph.remove_edges_from(tuple(graph.edges))
    elif case == "incompatible":
        graph.nodes[1]["theta"] = math.pi
    before = _plain(graph)
    calls = []
    result = _attempt(
        graph, _policy(mode), compute_delta_nfr=lambda subject: calls.append(subject)
    )
    assert not result.executed and result.stage_result is None
    assert result.compatibility.abstention_reason == reason
    assert calls == []
    assert _plain(graph) == before


@pytest.mark.parametrize("mode", ["select", "require_all"])
def test_permissive_policy_matches_legacy_row_but_keeps_distinct_domain_and_certificate(
    mode,
):
    graph, legacy, explicit = _graph(), _graph(), _graph()
    policy = ReceptionCompatibilityPolicy(math.pi, 1.0, mode=mode)
    result = _attempt(graph, policy, include_epi_jump_certificate=True)
    execute_neighbor_stage(legacy, Reception(), (0,), track_sources=False)
    explicit_stage = execute_neighbor_stage(
        explicit, Reception(), (0,), source_nodes=(1, 2, 3), track_sources=False
    )
    assert _form(graph) == _form(legacy) == _form(explicit) == 0.25
    assert result.compatibility.selected_sources == (1, 2, 3)
    assert result.stage_result.epi_jump_certificate is None
    assert (
        "explicit_reception_sources"
        in result.stage_result.epi_jump_certificate_abstention_reason
    )
    assert (
        result.stage_result.reception_observations[0].source_selection_origin
        == "compatibility_policy"
    )
    assert (
        explicit_stage.reception_observations[0].source_selection_origin
        == "explicit_request"
    )


@pytest.mark.parametrize("node", [0, 2])
@pytest.mark.parametrize("aliases", [ALIAS_THETA, ALIAS_VF], ids=["phase", "capacity"])
@pytest.mark.parametrize(
    "invalid", [None, True, math.inf], ids=["missing", "boolean", "infinite"]
)
def test_every_primitive_input_is_admitted_even_for_an_excluded_candidate(
    node, aliases, invalid
):
    graph = _graph()
    if invalid is None:
        for alias in aliases:
            graph.nodes[node].pop(alias, None)
    else:
        graph.nodes[node][aliases[0]] = invalid
        graph.nodes[node][aliases[-1]] = 0.0 if aliases == ALIAS_THETA else 1.0
    before = _plain(graph)
    with pytest.raises((TypeError, ValueError, OperatorPreconditionError)):
        _attempt(graph)
    assert _plain(graph) == before


def test_selected_form_error_is_not_abstention_but_excluded_form_is_not_a_filter_operand():
    graph = _graph()
    graph.nodes[2]["EPI"] = math.inf
    result = _attempt(graph)
    assert result.executed and _form(graph) == 0.375
    graph = _graph()
    graph.nodes[1]["EPI"] = math.inf
    before = _plain(graph)
    with pytest.raises((TypeError, ValueError, OperatorPreconditionError)):
        _attempt(graph)
    assert _plain(graph) == before


@pytest.mark.parametrize("mode", ["raise", "fallback"])
def test_nonempty_sources_with_blocked_grammar_raise_instead_of_abstaining_or_substituting(
    mode,
):
    graph = _graph()
    graph.graph["GRAMMAR_REJECTION_MODE"] = mode
    graph.nodes[0]["_grammar_u2_debt"] = 3
    before = _plain(graph)
    assert observe_reception_compatibility(graph, 0, _policy()).eligible
    with pytest.raises((ValueError, RuntimeError, OperatorPreconditionError)):
        _attempt(graph)
    assert _plain(graph) == before


def test_failed_optional_precondition_is_not_a_source_abstention():
    graph = _graph()
    graph.nodes[0]["delta_nfr"] = 1.0
    before = _plain(graph)
    assert observe_reception_compatibility(graph, 0, _policy()).eligible
    with pytest.raises(ValueError, match="EN precondition"):
        _attempt(graph)
    assert _plain(graph) == before


def test_late_callback_failure_restores_form_context_and_decision_metadata():
    graph = _graph()
    before = _plain(graph)
    calls = []

    def reject(subject):
        calls.append(1)
        assert _form(subject) == 0.375
        assert read_grammar_state(subject.nodes[0]).history[-1] == "EN"
        subject.nodes[2]["theta"] = 0.0
        subject.graph["late_marker"] = True
        raise RuntimeError("synthetic compatibility callback rejection")

    with pytest.raises(RuntimeError, match="compatibility callback rejection"):
        _attempt(graph, compute_delta_nfr=reject)
    assert calls == [1]
    assert _plain(graph) == before


@pytest.mark.parametrize(
    "change", ["rejected_phase", "rejected_capacity", "source_order", "removed_support"]
)
def test_stale_full_candidate_association_cannot_be_reused_for_the_same_selected_tuple(
    change,
):
    graph = _graph()
    decision = observe_reception_compatibility(graph, 0, _policy())
    if change == "rejected_phase":
        graph.nodes[2]["theta"] = 2.0
    elif change == "rejected_capacity":
        graph.nodes[3]["nu_f"] = 0.25
    elif change == "source_order":
        graph.remove_edge(2, 0)
        graph.add_edge(2, 0, weight=1.0)
    else:
        graph.remove_edge(2, 0)
    assert observe_reception_compatibility(graph, 0, _policy()).selected_sources == (1,)
    before = _plain(graph)
    with pytest.raises(RuntimeError, match="compatibility decision is stale"):
        execute_neighbor_stage(
            graph,
            Reception(),
            (0,),
            source_nodes=(1,),
            track_sources=False,
            _reception_compatibility_decision=decision,
        )
    assert _plain(graph) == before


def test_decision_and_dispatch_cannot_reassociate_a_policy_after_the_action():
    graph = _graph()
    other_policy = ReceptionCompatibilityPolicy(0.5, 0.0, mode="select")
    other_decision = observe_reception_compatibility(graph, 0, other_policy)
    result = _attempt(graph)
    altered = replace(
        result.compatibility,
        policy=ReceptionCompatibilityPolicy(0.5, 0.0, mode="select"),
    )
    assert altered.selected_sources == result.compatibility.selected_sources
    assert not altered._proof_fields_are_intact()
    with pytest.raises(ValueError):
        EligibleReceptionDispatch(altered, result.stage_result)
    with pytest.raises(ValueError):
        EligibleReceptionDispatch(result.compatibility, None)
    assert other_decision._proof_fields_are_intact()
    assert other_decision.selected_sources == result.compatibility.selected_sources
    with pytest.raises(ValueError, match="retain its compatibility decision"):
        EligibleReceptionDispatch(other_decision, result.stage_result)


def test_candidate_identity_phase_association_changes_the_actual_received_form():
    forms = []
    for exchange in (False, True):
        graph = _graph()
        if exchange:
            graph.nodes[1]["theta"], graph.nodes[2]["theta"] = (
                graph.nodes[2]["theta"],
                graph.nodes[1]["theta"],
            )
        result = _attempt(graph)
        assert result.compatibility.selected_sources == ((2,) if exchange else (1,))
        forms.append(_form(graph))
    assert forms == [0.375, 0.125]


def test_incoming_selected_order_retains_semantic_kind_tie_without_numeric_change():
    outputs = []
    for reorder in (False, True):
        graph = _graph()
        graph.nodes[0]["EPI"] = 0.1
        graph.nodes[1]["EPI"] = 0.6
        graph.nodes[2]["EPI"] = -0.6
        graph.nodes[2]["theta"] = 0.0
        graph.graph["GLYPH_FACTORS"]["EN_mix"] = 0.5
        if reorder:
            graph.remove_edge(1, 0)
            graph.add_edge(1, 0, weight=1.0)
        result = _attempt(graph)
        assert result.compatibility.selected_sources == ((2, 1) if reorder else (1, 2))
        outputs.append(
            (
                _form(graph),
                get_attr(
                    graph.nodes[0],
                    ALIAS_EPI_KIND,
                    strict=True,
                    conv=lambda value: value,
                ),
            )
        )
    assert outputs == [(0.05, "kind-1"), (0.05, "kind-2")]


@pytest.mark.parametrize(
    "option,value",
    [
        ("source_nodes", (1,)),
        ("sequence_context", None),
        ("track_sources", "false"),
        ("max_distance", -1),
    ],
)
def test_invalid_options_are_rejected_even_when_sources_would_abstain(option, value):
    graph = _graph()
    graph.remove_edges_from(tuple(graph.edges))
    before = _plain(graph)
    with pytest.raises((TypeError, ValueError, OperatorPreconditionError)):
        _attempt(graph, options={option: value})
    assert _plain(graph) == before


def test_zero_capacity_pair_can_admit_an_instantaneous_jump_without_claiming_active_flow():
    graph = _graph()
    graph.nodes[0]["nu_f"] = graph.nodes[1]["nu_f"] = 0.0
    result = _attempt(graph)
    assert result.compatibility.selected_sources == (1,)
    assert result.executed and _form(graph) == 0.375
    assert get_attr(graph.nodes[0], ALIAS_VF) == 0.0


def test_successful_en_preserves_filter_inputs_but_does_not_establish_future_admission():
    graph = _graph()
    before = observe_reception_compatibility(graph, 0, _policy())
    result = _attempt(graph)
    after = observe_reception_compatibility(graph, 0, _policy())
    assert result.executed and after == before
    assert after._proof_fields_are_intact()
    assert _form(graph) == 0.375


def test_actual_coupling_phase_alignment_can_enable_a_later_reception():
    graph = _graph(nx.Graph)
    graph.remove_nodes_from((2, 3))
    graph.nodes[1]["theta"] = 0.4
    graph.graph.update(
        UM_BIDIRECTIONAL=True,
        UM_SYNC_VF=False,
        UM_STABILIZE_DNFR=False,
        UM_FUNCTIONAL_LINKS=False,
    )
    graph.graph["GLYPH_FACTORS"]["UM_theta_push"] = 0.75
    policy = ReceptionCompatibilityPolicy(0.2, 0.0, mode="select")
    assert not _attempt(graph, policy).executed
    Coupling()(graph, 0)
    assert get_attr(graph.nodes[0], ALIAS_THETA) == pytest.approx(0.15)
    assert get_attr(graph.nodes[1], ALIAS_THETA) == pytest.approx(0.25)
    assert read_grammar_state(graph.nodes[0]).history[-1] == "UM"
    result = _attempt(graph, policy)
    assert result.executed and _form(graph) == 0.375
    assert read_grammar_state(graph.nodes[0]).history[-2:] == ("UM", "EN")


def test_actual_silence_capacity_change_can_disable_a_later_reception():
    graph = _graph(nx.Graph)
    graph.remove_nodes_from((2, 3))
    graph.graph.update(EPI_LATENT_MAX=1.0, SHA_MIN_VF=0.1)
    graph.graph["GLYPH_FACTORS"]["SHA_vf_factor"] = 0.5
    policy = ReceptionCompatibilityPolicy(0.25, 0.1, mode="select")
    assert observe_reception_compatibility(graph, 0, policy).selected_sources == (1,)
    forms = (_form(graph, 0), _form(graph, 1))
    Silence()(graph, 1)
    assert get_attr(graph.nodes[1], ALIAS_VF) == 0.5
    assert (_form(graph, 0), _form(graph, 1)) == forms
    assert read_grammar_state(graph.nodes[1]).history[-1] == "SHA"
    before = _plain(graph)
    assert not _attempt(graph, policy).executed
    assert _plain(graph) == before


def test_selected_reception_changes_the_next_actual_resonance_response():
    outputs = []
    for route in ("policy", "legacy", "require_all"):
        graph = _graph()
        for node in graph:
            graph.nodes[node]["epi_kind"] = "same"
        if route == "legacy":
            execute_neighbor_stage(graph, Reception(), (0,), track_sources=False)
        else:
            result = _attempt(
                graph, _policy("require_all" if route == "require_all" else "select")
            )
            assert result.executed is (route == "policy")
        received = 0.375 if route == "policy" else 0.25
        assert _form(graph) == received
        Resonance()(graph, 3)
        # RA consumes node 3's outgoing neighbor 0; its phase gate is admitted.
        assert _form(graph, 3) == 0.5 * 0.25 + 0.5 * received
        assert read_grammar_state(graph.nodes[3]).history[-1] == "RA"
        assert set(graph.edges) == {(1, 0), (2, 0), (3, 0)}
        outputs.append(_form(graph, 3))
    assert outputs == [0.3125, 0.25, 0.25]
