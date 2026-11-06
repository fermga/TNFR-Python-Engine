"""Bounded addressing, admission and rollback of actual retained Reception.

These small states carry explicitly supplied unit-test grammar contexts. They
are not claimed birth histories, physical observations or autonomous sources.
"""

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import replace

import networkx as nx
import pytest

from tnfr.alias import get_attr
from tnfr.constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_THETA, ALIAS_VF
from tnfr.operators import retained_reception as owner
from tnfr.operators.grammar_debt import read_grammar_state
from tnfr.operators.grammar_types import StructuralGrammarError
from tnfr.operators.network_stage import execute_neighbor_stage
from tnfr.operators.preconditions import OperatorPreconditionError
from tnfr.operators.reception import Reception
from tnfr.operators.retained_organization import (
    capture_retained_organization,
    retained_organization_matches_graph,
)
from tnfr.operators.retained_reception import (
    RetainedInputPort,
    execute_retained_reception_stage,
    retained_input_ports,
)
from tnfr.types import require_finite_real_scalar_epi
from tnfr.utils._structural_signature import structural_proof_signature

CORE, MEMBER, DONOR, PEER = "core", "member", "donor", "peer"
MEMBERS = (CORE, MEMBER)


def _graph(graph_type=nx.DiGraph):
    graph = graph_type(
        _t=0.0,
        RANDOM_SEED=17,
        VALIDATE_OPERATOR_PRECONDITIONS=True,
        GRAMMAR_CANON={"enabled": True},
        GRAMMAR_REJECTION_MODE="raise",
        EPI_MIN=-1.0,
        EPI_MAX=1.0,
        CLIP_MODE="hard",
        GLYPH_FACTORS={"EN_mix": 0.5},
    )
    for node, form in ((CORE, 0.25), (MEMBER, 0.125), (DONOR, 0.5), (PEER, 0.375)):
        graph.add_node(
            node,
            EPI=form,
            nu_f=1.0,
            theta=0.0,
            delta_nfr=0.0625,
            epi_kind="form",
            glyph_history=["IL"],
            private_memory={"node": node},
        )
    graph.add_edge(CORE, MEMBER, weight=1.0, role="interior")
    if graph.is_directed():
        graph.add_edge(MEMBER, CORE, weight=1.0, role="interior-return")
    graph.add_edge(DONOR, MEMBER, weight=0.25, role="external-input")
    graph.add_edge(PEER, CORE, weight=0.0, role="other-input")
    return graph


def _record(graph):
    return capture_retained_organization(graph, MEMBERS)


def _form(graph, node):
    return require_finite_real_scalar_epi(
        get_attr(graph.nodes[node], ALIAS_EPI, strict=True, conv=lambda value: value)
    )


def _state(graph, node):
    return (
        _form(graph, node),
        get_attr(graph.nodes[node], ALIAS_VF),
        get_attr(graph.nodes[node], ALIAS_THETA),
        get_attr(graph.nodes[node], ALIAS_DNFR),
        read_grammar_state(graph.nodes[node]),
    )


def _edges(graph):
    return (
        tuple(graph.edges(keys=True, data=True))
        if graph.is_multigraph()
        else tuple(graph.edges(data=True))
    )


def _checkpoint(graph):
    return structural_proof_signature(
        (
            graph.graph,
            tuple(graph.nodes(data=True)),
            _edges(graph),
            hasattr(graph, "_last_operator_applied"),
            getattr(graph, "_last_operator_applied", None),
        ),
        opaque_references=(graph,),
    )


@pytest.mark.parametrize(
    "graph_type", [nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph]
)
def test_passive_table_uses_retained_incoming_order_not_live_donor_values(graph_type):
    graph = _graph(graph_type)
    graph.add_edge(MEMBER, MEMBER, weight=0.0)
    if graph.is_multigraph():
        graph.add_edge(DONOR, MEMBER, weight=2.0, role="parallel")
    if graph.is_directed():
        graph.add_node("outgoing-only")
        graph.add_edge(MEMBER, "outgoing-only", weight=1.0)
    graph.nodes[DONOR]["EPI"] = None
    record = _record(graph)
    before = _checkpoint(graph)
    assert retained_input_ports(record) == (
        RetainedInputPort(0, CORE, PEER),
        RetainedInputPort(1, MEMBER, DONOR),
    )
    assert _checkpoint(graph) == before
    # The passive table is not a refreshed permission after a support change.
    graph.remove_edge(DONOR, MEMBER)
    assert retained_input_ports(record)[1].source == DONOR
    assert not retained_organization_matches_graph(graph, record)


@pytest.mark.parametrize(
    "graph_type", [nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph]
)
def test_address_resolves_real_single_source_en_and_returns_fresh_record(graph_type):
    graph = _graph(graph_type)
    ordinary = deepcopy(graph)
    record = _record(graph)
    protected = {
        node: structural_proof_signature(graph.nodes[node])
        for node in graph
        if node != MEMBER
    }
    edges = deepcopy(_edges(graph))
    result = execute_retained_reception_stage(graph, record, 1)
    expected = execute_neighbor_stage(
        ordinary, Reception(), (MEMBER,), source_nodes=(DONOR,), track_sources=False
    )
    assert result.port == RetainedInputPort(1, MEMBER, DONOR)
    assert result.stage_result.glyph == expected.glyph == "EN"
    assert result.stage_result.nodes_processed == 1
    assert _state(graph, MEMBER) == _state(ordinary, MEMBER)
    assert _form(graph, MEMBER) == 5 / 16
    observed = result.stage_result.reception_observations[0]
    assert observed.node == MEMBER and observed.neighbors == (DONOR,)
    assert observed.source_nodes == (DONOR,)
    assert observed.source_selection_origin == "explicit_request"
    assert not observed.source_tracking_enabled
    assert observed.neighbor_epi_values == (0.5,)
    assert result.stage_result.neighbor_epi_jump_certificate is None
    for node, signature in protected.items():
        assert structural_proof_signature(graph.nodes[node]) == signature
    assert _edges(graph) == edges and graph.graph["_t"] == 0.0
    assert retained_organization_matches_graph(graph, result.after)
    assert not retained_organization_matches_graph(graph, record)
    before_stale_attempt = _checkpoint(graph)
    with pytest.raises(ValueError, match="stale"):
        execute_retained_reception_stage(graph, record, 1)
    assert _checkpoint(graph) == before_stale_attempt
    second = execute_retained_reception_stage(graph, result.after, 1)
    assert _form(graph, MEMBER) == 13 / 32
    assert retained_organization_matches_graph(graph, second.after)


@pytest.mark.parametrize("kind", [None, "absent", "other-form"])
def test_donor_form_and_actual_kind_are_live_after_record_capture(kind):
    graph = _graph()
    record = _record(graph)
    graph.nodes[DONOR]["EPI"] = 0.75
    if kind == "absent":
        del graph.nodes[DONOR]["epi_kind"]
        actual_kind = ""
    else:
        graph.nodes[DONOR]["epi_kind"] = kind
        actual_kind = str(kind)
    assert retained_organization_matches_graph(graph, record)
    result = execute_retained_reception_stage(graph, record, 1)
    observed = result.stage_result.reception_observations[0]
    assert observed.neighbor_epi_values == (0.75,)
    assert observed.neighbor_epi_kinds == (actual_kind,)
    assert _form(graph, MEMBER) == 7 / 16


@pytest.mark.parametrize("kind", [None, "absent"])
def test_target_actual_kind_keeps_none_distinct_from_normalized_record(kind):
    graph = _graph()
    if kind == "absent":
        del graph.nodes[MEMBER]["epi_kind"]
    else:
        graph.nodes[MEMBER]["epi_kind"] = None
    record = _record(graph)
    assert record.member_states[1].epi_kind == ""
    result = execute_retained_reception_stage(graph, record, 1)
    observed = result.stage_result.reception_observations[0]
    assert observed.target_epi_kind_before == ("" if kind == "absent" else "None")


@pytest.mark.parametrize("bad", [None, True, float("nan"), "missing"])
def test_live_bad_donor_form_rejects_instead_of_using_fallback_or_cached_value(bad):
    graph = _graph()
    record = _record(graph)
    for key in ALIAS_EPI:
        graph.nodes[DONOR].pop(key, None)
    if bad != "missing":
        graph.nodes[DONOR][ALIAS_EPI[0]] = bad
        if len(ALIAS_EPI) > 1:
            graph.nodes[DONOR][ALIAS_EPI[1]] = 0.5
    before = _checkpoint(graph)
    with pytest.raises((TypeError, ValueError)):
        execute_retained_reception_stage(graph, record, 1)
    assert _checkpoint(graph) == before


@pytest.mark.parametrize("bad", [True, False, 1.0, "1", None, -1, 2])
def test_exact_ordinal_domain_rejects_without_dispatch(bad):
    graph = _graph()
    record = _record(graph)
    before = _checkpoint(graph)
    with pytest.raises((TypeError, ValueError, IndexError)):
        execute_retained_reception_stage(graph, record, bad)
    assert _checkpoint(graph) == before


def test_empty_passive_table_does_not_create_successful_empty_stage():
    graph = _graph()
    graph.remove_edge(DONOR, MEMBER)
    graph.remove_edge(PEER, CORE)
    record = _record(graph)
    assert retained_input_ports(record) == ()
    before = _checkpoint(graph)
    with pytest.raises(IndexError, match="does not exist"):
        execute_retained_reception_stage(graph, record, 0)
    assert _checkpoint(graph) == before


@pytest.mark.parametrize(
    "change", ["member", "history", "edge", "order", "foreign", "tampered"]
)
def test_whole_member_support_association_is_readmitted_before_addressing(change):
    graph = _graph()
    record = _record(graph)
    if change == "member":
        graph.nodes[CORE]["EPI"] = 0.375
    elif change == "history":
        graph.nodes[CORE]["glyph_history"].append("EN")
    elif change == "edge":
        graph.edges[DONOR, MEMBER]["weight"] = 0.5
    elif change == "order":
        data = dict(graph.edges[CORE, MEMBER])
        graph.remove_edge(CORE, MEMBER)
        graph.add_edge(CORE, MEMBER, **data)
    elif change == "foreign":
        record = _record(_graph())
    else:
        record = replace(record, incoming_ports=record.incoming_ports[::-1])
        with pytest.raises(ValueError, match="intact"):
            retained_input_ports(record)
    before = _checkpoint(graph)
    with pytest.raises(ValueError, match="foreign, stale, invalid or unsealed"):
        execute_retained_reception_stage(graph, record, 1)
    assert _checkpoint(graph) == before


@pytest.mark.parametrize("policy", ["disabled", "fallback", "local-false"])
def test_required_policy_is_admitted_before_empty_address(policy):
    graph = _graph()
    graph.remove_edge(DONOR, MEMBER)
    graph.remove_edge(PEER, CORE)
    options = None
    if policy == "disabled":
        graph.graph["GRAMMAR_CANON"] = {"enabled": False}
    elif policy == "fallback":
        graph.graph["GRAMMAR_REJECTION_MODE"] = "fallback"
    else:
        options = {"validate_preconditions": False}
    record = _record(graph)
    before = _checkpoint(graph)
    with pytest.raises(ValueError, match="requires"):
        execute_retained_reception_stage(graph, record, 0, execution_kwargs=options)
    assert _checkpoint(graph) == before


@pytest.mark.parametrize(
    "options",
    [
        {"source_nodes": (PEER,)},
        {"track_sources": True},
        {"sequence_context": None},
        {"compute_delta_nfr": None},
        {"collect_metrics": 1},
        {"window": True},
        {"dt": float("nan")},
    ],
)
def test_invalid_or_forbidden_request_options_preserve_source(options):
    graph = _graph()
    record = _record(graph)
    before = _checkpoint(graph)
    with pytest.raises((TypeError, ValueError, OperatorPreconditionError)):
        execute_retained_reception_stage(graph, record, 1, execution_kwargs=options)
    assert _checkpoint(graph) == before


def test_target_own_debt_is_not_replaced_by_other_member_credit():
    graph = _graph()
    graph.nodes[MEMBER]["_grammar_u2_debt"] = 3
    record = _record(graph)
    before = _checkpoint(graph)
    with pytest.raises(StructuralGrammarError) as caught:
        execute_retained_reception_stage(graph, record, 1)
    assert caught.value.rule == "U2"
    assert _checkpoint(graph) == before


@pytest.mark.parametrize("hook", ["__call__", "_execute"])
def test_noncanonical_reception_methods_cannot_supply_addressed_result(
    monkeypatch, hook
):
    graph = _graph()
    record = _record(graph)
    before = _checkpoint(graph)
    monkeypatch.setattr(Reception, hook, lambda *args, **kwargs: None)
    with pytest.raises(TypeError, match="canonical Reception"):
        execute_retained_reception_stage(graph, record, 1)
    assert _checkpoint(graph) == before


@pytest.mark.parametrize(
    "change",
    ["other-member", "donor", "edge", "phase", "pressure", "capacity", "kind", "raise"],
)
def test_post_dispatch_changes_or_failure_restore_the_entire_invocation(
    monkeypatch, change
):
    graph = _graph()
    record = _record(graph)
    before = _checkpoint(graph)
    original = owner.execute_neighbor_stage

    def corrupt(*args, **kwargs):
        result = original(*args, **kwargs)
        assert _form(graph, MEMBER) == 5 / 16
        if change == "other-member":
            graph.nodes[CORE]["private_memory"]["changed"] = True
        elif change == "donor":
            graph.nodes[DONOR]["EPI"] = 0.875
        elif change == "edge":
            graph.edges[DONOR, MEMBER]["weight"] = 0.875
        elif change == "phase":
            graph.nodes[MEMBER][ALIAS_THETA[0]] = 0.25
        elif change == "pressure":
            graph.nodes[MEMBER][ALIAS_DNFR[0]] = 0.125
        elif change == "capacity":
            graph.nodes[MEMBER][ALIAS_VF[0]] = 2.0
        elif change == "kind":
            graph.nodes[MEMBER]["epi_kind"] = "corrupted"
        else:
            raise RuntimeError("after actual addressed EN")
        return result

    monkeypatch.setattr(owner, "execute_neighbor_stage", corrupt)
    with pytest.raises((RuntimeError, ValueError)):
        execute_retained_reception_stage(graph, record, 1)
    assert _checkpoint(graph) == before
    assert retained_organization_matches_graph(graph, record)


def test_post_observation_failure_restores_completed_en(monkeypatch):
    graph = _graph()
    record = _record(graph)
    before = _checkpoint(graph)
    original = owner.capture_retained_organization

    def fail(*args, **kwargs):
        result = original(*args, **kwargs)
        assert result.member_states[1].epi == 5 / 16
        raise RuntimeError("after retained observation")

    monkeypatch.setattr(owner, "capture_retained_organization", fail)
    with pytest.raises(RuntimeError, match="after retained observation"):
        execute_retained_reception_stage(graph, record, 1)
    assert _checkpoint(graph) == before


@pytest.mark.parametrize("raises", [False, True])
def test_options_mapping_cannot_change_the_source_address_or_partial_state(raises):
    graph = _graph()
    record = _record(graph)
    before = _checkpoint(graph)

    class SideEffectOptions(Mapping):
        def __len__(self):
            return 1

        def __getitem__(self, key):
            if key != "collect_metrics":
                raise KeyError(key)
            return False

        def __iter__(self):
            graph.nodes[MEMBER]["EPI"] = 0.875
            graph.remove_edge(DONOR, MEMBER)
            graph.add_edge(PEER, MEMBER)
            if raises:
                raise RuntimeError("options materialization")
            yield "collect_metrics"

    if raises:
        with pytest.raises(RuntimeError, match="options materialization"):
            execute_retained_reception_stage(
                graph, record, 1, execution_kwargs=SideEffectOptions()
            )
        assert _checkpoint(graph) == before
    else:
        result = execute_retained_reception_stage(
            graph, record, 1, execution_kwargs=SideEffectOptions()
        )
        assert result.port == RetainedInputPort(1, MEMBER, DONOR)
        assert _form(graph, MEMBER) == 5 / 16
        assert graph.has_edge(DONOR, MEMBER) and not graph.has_edge(PEER, MEMBER)


def test_dispatch_summary_rejects_unrelated_address_and_target_channels():
    graph = _graph()
    result = execute_retained_reception_stage(graph, _record(graph), 1)
    with pytest.raises(ValueError, match="port"):
        replace(result, port=RetainedInputPort(0, MEMBER, DONOR))
    graph.nodes[MEMBER][ALIAS_DNFR[0]] = 0.125
    altered = _record(graph)
    with pytest.raises(ValueError, match="capacity and pressure"):
        replace(result, after=altered)
