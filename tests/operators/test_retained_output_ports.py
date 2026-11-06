"""Bounded grouped output using actual EN and declared unit-test contexts.

These admission states are supplied fixtures, not acquired THOL histories.
The separate participation fixture owns actual construction and re-expression.
"""

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import replace

import networkx as nx
import pytest

from tnfr.alias import get_attr
from tnfr.constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_VF
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
    RetainedOutputPort,
    execute_retained_output_reception_stage,
    retained_output_ports,
)
from tnfr.types import require_finite_real_scalar_epi
from tnfr.utils._structural_signature import structural_proof_signature

A, C, R, S, U = "a", "c", "receiver", "other-receiver", "input"
MEMBERS = (A, C)


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
    for node, value in ((A, 0.25), (C, 0.125), (R, 0.375), (S, 0.5), (U, 0.5)):
        graph.add_node(
            node,
            EPI=value,
            nu_f=1.0,
            theta=0.0,
            delta_nfr=0.0625,
            epi_kind="form",
            glyph_history=["IL"],
            private_memory={"node": node},
        )
    graph.add_edge(A, C, weight=1.0, role="interior")
    if graph.is_directed():
        graph.add_edge(C, A, weight=1.0, role="interior-return")
    graph.add_edge(A, R, weight=0.25, role="first-output")
    graph.add_edge(A, S, weight=0.0, role="second-output")
    graph.add_edge(C, S, weight=2.0, role="second-output")
    graph.add_edge(C, R, weight=0.5, role="first-output")
    graph.add_edge(U, A, weight=1.0, role="input")
    return graph


def _record(graph, members=MEMBERS):
    return capture_retained_organization(graph, members)


def _form(graph, node):
    return require_finite_real_scalar_epi(
        get_attr(graph.nodes[node], ALIAS_EPI, strict=True, conv=lambda value: value)
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
def test_passive_output_intersection_uses_complete_directed_unique_support(graph_type):
    graph = _graph(graph_type)
    graph.add_edge(A, A, weight=1.0)
    if graph.is_multigraph():
        graph.add_edge(A, R, weight=7.0, role="parallel")
    if graph.is_directed():
        graph.add_node("reverse-only")
        graph.add_edge("reverse-only", A)
        graph.add_edge("reverse-only", C)
    # A passive port is not a payload or numeric receiver-admission verdict.
    graph.nodes[R]["EPI"] = None
    record = _record(graph)
    before = _checkpoint(graph)
    assert retained_output_ports(record) == (
        RetainedOutputPort(0, R, MEMBERS),
        RetainedOutputPort(1, S, MEMBERS),
    )
    assert _checkpoint(graph) == before
    assert retained_output_ports(_record(graph, (C, A)))[0].receiver == S
    graph.remove_edge(C, R)
    assert retained_output_ports(record)[0].receiver == R
    assert not retained_organization_matches_graph(graph, record)
    assert all(port.receiver != R for port in retained_output_ports(_record(graph)))


@pytest.mark.parametrize("members", [(A,), MEMBERS])
def test_actual_output_matches_explicit_en_and_preserves_the_source(members):
    graph = _graph()
    graph.nodes[A]["epi_kind"] = None
    del graph.nodes[C]["epi_kind"]
    ordinary = deepcopy(graph)
    record = _record(graph, members)
    before_sources = tuple(
        structural_proof_signature(graph.nodes[node]) for node in members
    )
    index = next(
        port.ordinal for port in retained_output_ports(record) if port.receiver == R
    )
    result = execute_retained_output_reception_stage(graph, record, index)
    explicit = execute_neighbor_stage(
        ordinary, Reception(), (R,), source_nodes=members, track_sources=False
    )
    assert result.port == RetainedOutputPort(index, R, members)
    assert result.stage_result.glyph == explicit.glyph == "EN"
    observed = result.stage_result.reception_observations[0]
    assert observed.node == R and observed.source_nodes == members
    assert observed.neighbors == members
    assert observed.source_selection_origin == "explicit_request"
    assert observed.neighbor_epi_kinds == (
        ("None",) if len(members) == 1 else ("None", "")
    )
    assert observed.target_epi_before == 0.375
    assert _form(graph, R) == _form(ordinary, R)
    assert _form(graph, R) == (5 / 16 if len(members) == 1 else 9 / 32)
    assert read_grammar_state(graph.nodes[R]) == read_grammar_state(ordinary.nodes[R])
    assert get_attr(graph.nodes[R], ALIAS_VF) == 1.0
    assert get_attr(graph.nodes[R], ALIAS_DNFR) == 0.0625
    assert result.stage_result.neighbor_epi_jump_certificate is None
    assert not observed.source_tracking_enabled
    assert (
        tuple(structural_proof_signature(graph.nodes[node]) for node in members)
        == before_sources
    )
    assert retained_organization_matches_graph(graph, record)
    assert retained_organization_matches_graph(graph, result.after)


@pytest.mark.parametrize("kind", [None, 17])
def test_external_receiver_form_and_kind_are_live_after_capture(kind):
    graph = _graph()
    record = _record(graph)
    graph.nodes[R]["EPI"] = 0.75
    graph.nodes[R]["epi_kind"] = kind
    assert retained_organization_matches_graph(graph, record)
    result = execute_retained_output_reception_stage(graph, record, 0)
    observed = result.stage_result.reception_observations[0]
    assert observed.target_epi_before == 0.75
    assert observed.target_epi_kind_before == str(kind)
    assert _form(graph, R) == 15 / 32
    assert observed.neighbor_epi_values == (0.25, 0.125)


@pytest.mark.parametrize("change", ["form", "history", "edge"])
def test_source_member_and_support_changes_require_fresh_capture(change):
    graph = _graph()
    record = _record(graph)
    if change == "form":
        graph.nodes[C]["EPI"] = 0.375
    elif change == "history":
        graph.nodes[C]["epi_time_history"] = [(0.0, 0.125)]
    else:
        graph.edges[C, R]["weight"] = 0.75
    before = _checkpoint(graph)
    with pytest.raises(ValueError, match="foreign, stale, invalid or unsealed"):
        execute_retained_output_reception_stage(graph, record, 0)
    assert _checkpoint(graph) == before


def test_foreign_tampered_and_inconsistent_summaries_supply_no_permission():
    graph = _graph()
    record = _record(graph)
    for bad in (
        _record(_graph()),
        replace(record, outgoing_ports=record.outgoing_ports[::-1]),
    ):
        before = _checkpoint(graph)
        with pytest.raises(ValueError, match="foreign, stale, invalid or unsealed"):
            execute_retained_output_reception_stage(graph, bad, 0)
        assert _checkpoint(graph) == before
    result = execute_retained_output_reception_stage(graph, record, 0)
    with pytest.raises(ValueError, match="port"):
        replace(result, port=RetainedOutputPort(1, R, MEMBERS))
    graph.nodes[C]["private_memory"]["changed"] = True
    with pytest.raises(ValueError, match="source group's own state"):
        replace(result, after=_record(graph))


def test_actual_input_en_requires_refreshed_source_record_for_output():
    graph = _graph()
    old = _record(graph)
    execute_neighbor_stage(
        graph, Reception(), (A,), source_nodes=(U,), track_sources=False
    )
    assert _form(graph, A) == 3 / 8
    before = _checkpoint(graph)
    with pytest.raises(ValueError, match="stale"):
        execute_retained_output_reception_stage(graph, old, 0)
    assert _checkpoint(graph) == before
    result = execute_retained_output_reception_stage(graph, _record(graph), 0)
    assert result.stage_result.reception_observations[0].neighbor_epi_values == (
        3 / 8,
        1 / 8,
    )
    assert _form(graph, R) == 5 / 16


def test_exact_ordinal_admission_never_falls_back_to_another_receiver():
    graph = _graph()
    record = _record(graph)
    for bad in (True, -1, 0.0, "0", 2):
        before = _checkpoint(graph)
        with pytest.raises((TypeError, ValueError, IndexError)):
            execute_retained_output_reception_stage(graph, record, bad)
        assert _checkpoint(graph) == before


def test_common_policy_options_and_receiver_domain_precede_empty_or_actual_read():
    for change in ("disabled", "fallback", "local-false", "override", "bad-window"):
        graph = _graph()
        graph.remove_edge(C, R)
        graph.remove_edge(C, S)
        options = None
        if change == "disabled":
            graph.graph["GRAMMAR_CANON"] = {"enabled": False}
        elif change == "fallback":
            graph.graph["GRAMMAR_REJECTION_MODE"] = "fallback"
        elif change == "local-false":
            options = {"validate_preconditions": False}
        elif change == "override":
            options = {"source_nodes": (A,)}
        else:
            options = {"window": True}
        record = _record(graph)
        assert retained_output_ports(record) == ()
        before = _checkpoint(graph)
        with pytest.raises((TypeError, ValueError, OperatorPreconditionError)):
            execute_retained_output_reception_stage(
                graph, record, 0, execution_kwargs=options
            )
        assert _checkpoint(graph) == before
    graph = _graph()
    graph.remove_edge(C, R)
    graph.remove_edge(C, S)
    record = _record(graph)
    before = _checkpoint(graph)
    with pytest.raises(IndexError, match="does not exist"):
        execute_retained_output_reception_stage(graph, record, 0)
    assert _checkpoint(graph) == before
    graph = _graph()
    record = _record(graph)
    graph.nodes[R]["nu_f"] = True
    before = _checkpoint(graph)
    with pytest.raises((TypeError, ValueError), match="capacity"):
        execute_retained_output_reception_stage(graph, record, 0)
    assert _checkpoint(graph) == before


def test_receiver_uses_its_own_live_grammar_not_source_credit():
    graph = _graph()
    record = _record(graph)
    graph.nodes[R]["_grammar_u2_debt"] = 3
    assert retained_organization_matches_graph(graph, record)
    before = _checkpoint(graph)
    with pytest.raises(StructuralGrammarError) as caught:
        execute_retained_output_reception_stage(graph, record, 0)
    assert caught.value.rule == "U2"
    assert _checkpoint(graph) == before


def test_replaced_reception_binding_cannot_supply_a_grouped_output(monkeypatch):
    graph = _graph()
    record = _record(graph)
    before = _checkpoint(graph)
    monkeypatch.setattr(Reception, "_execute", lambda *args, **kwargs: None)
    with pytest.raises(TypeError, match="canonical Reception"):
        execute_retained_output_reception_stage(graph, record, 0)
    assert _checkpoint(graph) == before


@pytest.mark.parametrize("change", ["form", "kind", "pressure", "source", "raise"])
def test_late_actual_output_failure_or_corruption_restores_source_and_receiver(
    monkeypatch, change
):
    graph = _graph()
    record = _record(graph)
    before = _checkpoint(graph)
    original = owner.execute_neighbor_stage

    def corrupt(*args, **kwargs):
        result = original(*args, **kwargs)
        assert _form(graph, R) == 9 / 32
        if change == "form":
            graph.nodes[R][ALIAS_EPI[0]] = 0.875
        elif change == "kind":
            graph.nodes[R]["epi_kind"] = "corrupted"
        elif change == "pressure":
            graph.nodes[R][ALIAS_DNFR[0]] = 0.125
        elif change == "source":
            graph.nodes[C]["private_memory"]["changed"] = True
        else:
            raise RuntimeError("after actual grouped output")
        return result

    monkeypatch.setattr(owner, "execute_neighbor_stage", corrupt)
    with pytest.raises((RuntimeError, ValueError)):
        execute_retained_output_reception_stage(graph, record, 0)
    assert _checkpoint(graph) == before
    assert retained_organization_matches_graph(graph, record)


@pytest.mark.parametrize("raises", [False, True])
def test_options_mapping_cannot_change_resolved_output_or_leave_partial_state(raises):
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
            graph.nodes[R]["EPI"] = 0.875
            graph.remove_edge(C, R)
            if raises:
                raise RuntimeError("options materialization")
            yield "collect_metrics"

    if raises:
        with pytest.raises(RuntimeError, match="options materialization"):
            execute_retained_output_reception_stage(
                graph, record, 0, execution_kwargs=SideEffectOptions()
            )
        assert _checkpoint(graph) == before
    else:
        result = execute_retained_output_reception_stage(
            graph, record, 0, execution_kwargs=SideEffectOptions()
        )
        assert result.port == RetainedOutputPort(0, R, MEMBERS)
        assert _form(graph, R) == 9 / 32
        assert graph.has_edge(C, R)
