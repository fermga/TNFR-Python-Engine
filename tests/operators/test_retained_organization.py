"""Finite admission/transaction controls; no trajectory or producer fixture."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import networkx as nx
import pytest

from tnfr.constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_THETA, ALIAS_VF
from tnfr.operators.definitions import Contraction, Expansion, Reception
from tnfr.operators.definitions_base import Operator
from tnfr.operators.network_stage import TWO_PHASE_JACOBI, execute_pointwise_stage
from tnfr.operators.preconditions import OperatorPreconditionError
from tnfr.operators.retained_organization import (
    RetainedScaleDispatch,
    capture_retained_organization,
    execute_retained_scale_stage,
    retained_organization_matches_graph,
)
from tnfr.utils._structural_signature import structural_proof_signature


def _graph(graph_type=nx.Graph):
    graph = graph_type(
        GRAMMAR_REJECTION_MODE="raise",
        EDGE_AWARE_ENABLED=True,
        GLYPH_FACTORS={"VAL_scale": 1.25, "NUL_scale": 0.5},
        RANDOM_SEED=17,
    )
    for node, epi, capacity, history in (
        ("parent", 0.4, 2.0, ["AL", "IL"]),
        ("child", 0.2, 0.75, ["EN"]),
        ("boundary", -0.1, 3.0, ["AL"]),
    ):
        graph.add_node(
            node,
            EPI=epi,
            nu_f=capacity,
            theta=0.25,
            delta_nfr=0.125,
            epi_kind="same",
            glyph_history=history,
        )
    graph.add_edge("parent", "child", weight=0.375, label="internal")
    graph.add_edge("boundary", "parent", weight=0.0, label="incoming")
    if graph.is_multigraph():
        graph.add_edge("parent", "child", weight=0.75, label="parallel")
    return graph


def _plain(graph):
    return structural_proof_signature(
        (
            graph.graph,
            tuple(graph.nodes(data=True)),
            (
                tuple(graph.edges(keys=True, data=True))
                if graph.is_multigraph()
                else tuple(graph.edges(data=True))
            ),
        ),
        opaque_references=(graph,),
    )


@pytest.mark.parametrize(
    "graph_type", [nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph]
)
def test_capture_preserves_real_ports_without_reading_boundary_coordinates(graph_type):
    graph = _graph(graph_type)
    graph.nodes["boundary"]["EPI"] = float("nan")
    graph.nodes["boundary"]["theta"] = None
    graph.nodes["parent"]["epi_time_history"] = [(0.0, 99.0)]
    before = _plain(graph)
    record = capture_retained_organization(graph, ("child", "parent"))
    assert _plain(graph) == before
    assert retained_organization_matches_graph(graph, record)
    assert tuple(state.node for state in record.member_states) == ("child", "parent")
    assert record.member_states[0].grammar.history == ("EN",)
    assert record.member_states[0].temporal_context == ()
    assert record.member_states[1].temporal_context[0][0] == "epi_time_history"
    assert record.incoming_ports[0] == ("parent",)
    assert record.outgoing_ports[0] == (() if graph.is_directed() else ("parent",))
    assert record.incoming_ports[1] == (
        ("boundary",) if graph.is_directed() else ("child", "boundary")
    )
    assert len(record.incident_edges) == (3 if graph.is_multigraph() else 2)
    if graph.is_multigraph():
        assert tuple(edge[2] for edge in record.incident_edges[:2]) == (0, 1)


@pytest.mark.parametrize(
    "members", [(), [], ["parent"], ("parent", "parent"), ("absent",)]
)
def test_member_domain_rejected_without_writes(members):
    graph = _graph()
    before = _plain(graph)
    with pytest.raises((TypeError, ValueError, KeyError)):
        capture_retained_organization(graph, members)
    assert _plain(graph) == before


@pytest.mark.parametrize("aliases", [ALIAS_EPI, ALIAS_THETA, ALIAS_VF, ALIAS_DNFR])
@pytest.mark.parametrize("bad", [None, True, "0.25", float("nan"), float("inf")])
def test_authoritative_invalid_coordinate_cannot_use_valid_secondary_alias(
    aliases, bad
):
    graph = _graph()
    for key in aliases:
        graph.nodes["parent"].pop(key, None)
    graph.nodes["parent"][aliases[0]] = bad
    if len(aliases) > 1:
        graph.nodes["parent"][aliases[1]] = 0.25
    before = _plain(graph)
    with pytest.raises((TypeError, ValueError)):
        capture_retained_organization(graph, ("parent",))
    assert _plain(graph) == before


@pytest.mark.parametrize("aliases", [ALIAS_EPI, ALIAS_THETA, ALIAS_VF, ALIAS_DNFR])
def test_missing_coordinate_is_not_fabricated_and_valid_secondary_alias_works(aliases):
    graph = _graph()
    for key in aliases:
        graph.nodes["parent"].pop(key, None)
    with pytest.raises(ValueError, match="explicit"):
        capture_retained_organization(graph, ("parent",))
    graph.nodes["parent"][aliases[-1]] = 0.25
    assert capture_retained_organization(graph, ("parent",))._proof_fields_are_intact()


def test_negative_capacity_and_unadmitted_kind_reject():
    graph = _graph()
    graph.nodes["parent"]["nu_f"] = -0.25
    with pytest.raises(ValueError):
        capture_retained_organization(graph, ("parent",))
    graph.nodes["parent"]["nu_f"] = 0.0
    graph.nodes["parent"]["epi_kind"] = object()
    with pytest.raises(TypeError, match="kind"):
        capture_retained_organization(graph, ("parent",))
    del graph.nodes["parent"]["epi_kind"]
    record = capture_retained_organization(graph, ("parent",))
    assert record.member_states[0].epi_kind == ""
    assert "epi_kind" not in graph.nodes["parent"]


@pytest.mark.parametrize(
    "change",
    [
        "form",
        "phase",
        "capacity",
        "pressure",
        "kind",
        "history",
        "temporal",
        "alias",
        "edge",
        "port",
        "order",
    ],
)
def test_live_member_context_and_port_changes_make_record_stale(change):
    graph = _graph(nx.DiGraph)
    record = capture_retained_organization(graph, ("parent", "child"))
    if change in {"form", "phase", "capacity", "pressure", "kind"}:
        key = {
            "form": "EPI",
            "phase": "theta",
            "capacity": "nu_f",
            "pressure": "delta_nfr",
            "kind": "epi_kind",
        }[change]
        graph.nodes["parent"][key] = "other" if change == "kind" else 0.625
    elif change == "history":
        graph.nodes["child"]["glyph_history"].append("IL")
    elif change == "temporal":
        graph.nodes["child"]["epi_history"] = [0.2]
    elif change == "alias":
        graph.nodes["parent"][ALIAS_EPI[-1]] = 0.4
    elif change == "edge":
        graph.edges["parent", "child"]["weight"] = 0.125
    elif change == "port":
        graph.add_edge("child", "boundary")
    else:
        graph.add_edge("parent", "boundary")
        record = capture_retained_organization(graph, ("parent", "child"))
        data = dict(graph.edges["parent", "child"])
        graph.remove_edge("parent", "child")
        graph.add_edge("parent", "child", **data)
    before = _plain(graph)
    assert not retained_organization_matches_graph(graph, record)
    with pytest.raises(ValueError, match="stale"):
        execute_retained_scale_stage(graph, record, Expansion())
    assert _plain(graph) == before


def test_boundary_state_changes_are_allowed_but_foreign_and_tampered_records_reject():
    graph = _graph()
    record = capture_retained_organization(graph, ("parent",))
    graph.nodes["boundary"]["EPI"] = 0.9
    graph.graph["unrelated_report"] = "new"
    assert retained_organization_matches_graph(graph, record)
    assert not retained_organization_matches_graph(_graph(), record)
    modified = replace(
        record, member_states=(replace(record.member_states[0], epi=0.6),)
    )
    assert not retained_organization_matches_graph(graph, modified)


@pytest.mark.parametrize(
    "graph_type", [nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph]
)
@pytest.mark.parametrize("operator_type", [Expansion, Contraction])
@pytest.mark.parametrize("edge_aware", [False, True])
def test_action_matches_existing_stage_and_preserves_distinct_contexts(
    graph_type, operator_type, edge_aware
):
    graph = _graph(graph_type)
    graph.graph["EDGE_AWARE_ENABLED"] = edge_aware
    ordinary = deepcopy(graph)
    members = ("child", "parent")
    record = capture_retained_organization(graph, members)
    boundary = deepcopy(graph.nodes["boundary"])
    result = execute_retained_scale_stage(graph, record, operator_type())
    expected = execute_pointwise_stage(
        ordinary, operator_type(), members, require_exact_operator=True
    )
    assert result.stage_result.schedule == expected.schedule == TWO_PHASE_JACOBI
    assert result.stage_result.glyph == expected.glyph
    for node in members:
        for key in (
            "EPI",
            "nu_f",
            "theta",
            "delta_nfr",
            "epi_kind",
            "glyph_history",
            "_grammar_u2_debt",
            "_grammar_prior_coherence",
            "_operator_step",
        ):
            assert graph.nodes[node].get(key) == ordinary.nodes[node].get(key)
    assert graph.nodes["boundary"] == boundary
    assert result.before.member_states[0].grammar.history == ("EN",)
    assert result.after.member_states[0].grammar.history == (
        "EN",
        operator_type.glyph.value,
    )
    assert result.after.member_states[1].grammar.history == (
        "AL",
        "IL",
        operator_type.glyph.value,
    )
    assert retained_organization_matches_graph(graph, result.after)
    assert not retained_organization_matches_graph(graph, result.before)


@pytest.mark.parametrize(
    "options",
    [
        {"compute_delta_nfr": None},
        {"sequence_context": {}},
        {"require_exact_operator": False},
        {"dt": True, "validate_nodal_equation": True},
        {"collect_metrics": "yes"},
    ],
)
def test_unsupported_or_invalid_options_do_not_write(options):
    graph = _graph()
    record = capture_retained_organization(graph, ("parent",))
    before = _plain(graph)
    with pytest.raises((TypeError, ValueError, OperatorPreconditionError)):
        execute_retained_scale_stage(
            graph, record, Expansion(), execution_kwargs=options
        )
    assert _plain(graph) == before


def test_other_operator_and_subclass_reject():
    class Derived(Expansion):
        pass

    graph = _graph()
    record = capture_retained_organization(graph, ("parent",))
    for operator in (Derived(), Reception()):
        with pytest.raises(TypeError, match="exact"):
            execute_retained_scale_stage(graph, record, operator)


def test_equal_value_rebinding_cannot_erase_shared_member_history_association():
    graph = _graph()
    shared = [0.2, 0.3]
    graph.nodes["parent"]["epi_history"] = shared
    graph.nodes["child"]["epi_history"] = shared
    record = capture_retained_organization(graph, ("parent", "child"))
    assert retained_organization_matches_graph(graph, record)
    graph.nodes["child"]["epi_history"] = list(shared)
    assert not retained_organization_matches_graph(graph, record)


@pytest.mark.parametrize("owner", [Expansion, Contraction])
@pytest.mark.parametrize("hook", ["__call__", "_execute"])
def test_overridden_canonical_hooks_reject_before_writes(monkeypatch, owner, hook):
    graph = _graph()
    record = capture_retained_organization(graph, ("parent",))
    before = _plain(graph)
    monkeypatch.setattr(owner, hook, lambda *args, **kwargs: None)
    with pytest.raises(TypeError, match="canonical bindings"):
        execute_retained_scale_stage(graph, record, owner())
    assert _plain(graph) == before


@pytest.mark.parametrize("attribute", ["_execute", "name", "glyph"])
def test_instance_binding_overrides_reject(attribute):
    graph = _graph()
    record = capture_retained_organization(graph, ("parent",))
    operator = Expansion()
    setattr(operator, attribute, getattr(operator, attribute))
    with pytest.raises(TypeError, match="canonical bindings"):
        execute_retained_scale_stage(graph, record, operator)


@pytest.mark.parametrize("hook", ["__call__", "_execute"])
def test_overridden_inherited_base_hooks_reject(monkeypatch, hook):
    graph = _graph()
    record = capture_retained_organization(graph, ("parent",))
    before = _plain(graph)
    monkeypatch.setattr(Operator, hook, lambda *args, **kwargs: None)
    with pytest.raises(TypeError, match="canonical bindings"):
        execute_retained_scale_stage(graph, record, Expansion())
    assert _plain(graph) == before


@pytest.mark.parametrize("change", ["phase", "port", "edge"])
def test_dispatch_summary_rejects_inconsistent_retained_boundary(change):
    graph = _graph()
    record = capture_retained_organization(graph, ("parent",))
    result = execute_retained_scale_stage(graph, record, Expansion())
    if change == "phase":
        graph.nodes["parent"]["theta"] = 0.75
    elif change == "port":
        graph.add_edge("parent", "parent")
    else:
        graph.edges["parent", "child"]["weight"] = 0.25
    changed = capture_retained_organization(graph, ("parent",))
    with pytest.raises(ValueError, match="ports and member phase"):
        RetainedScaleDispatch(result.before, changed, result.stage_result)


def test_late_nonmember_mutation_is_detected_and_rolls_back(monkeypatch):
    import tnfr.operators.retained_organization as owner

    original = owner.execute_pointwise_stage
    graph = _graph()
    record = capture_retained_organization(graph, ("parent", "child"))
    before = _plain(graph)

    def mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        graph.nodes["boundary"]["EPI"] = 0.75
        return result

    monkeypatch.setattr(owner, "execute_pointwise_stage", mutate)
    with pytest.raises(RuntimeError, match="protected"):
        execute_retained_scale_stage(graph, record, Expansion())
    assert _plain(graph) == before
    assert retained_organization_matches_graph(graph, record)
