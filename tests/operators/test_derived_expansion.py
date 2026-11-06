"""Finite derived VAL map/materialization controls; no external producer."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from fractions import Fraction

import networkx as nx
import pytest

from tnfr.constants import DNFR_PRIMARY, EPI_PRIMARY, THETA_PRIMARY, VF_PRIMARY
from tnfr.operators._birth_materialization import materialize_isolated_member
from tnfr.operators.derived_expansion import (
    DerivedExpansionPolicy,
    execute_derived_expansion_stage,
)
from tnfr.operators.expansion import Expansion
from tnfr.operators.grammar_debt import read_grammar_state
from tnfr.operators.network_stage import (
    execute_pointwise_stage,
    execute_self_organization_stage,
)
from tnfr.operators.retained_organization import (
    capture_retained_organization,
    retained_organization_matches_graph,
)
from tnfr.operators.self_organization import SelfOrganization
from tnfr.utils._structural_signature import structural_proof_signature


def _graph(graph_type=nx.Graph, *, epi=0.5, capacity=1.0):
    graph = graph_type(
        GRAMMAR_REJECTION_MODE="raise",
        EDGE_AWARE_ENABLED=True,
        GLYPH_FACTORS={"VAL_scale": 1.5},
        RANDOM_SEED=17,
    )
    graph.add_node(
        "parent",
        EPI=epi,
        nu_f=capacity,
        theta=0.25,
        delta_nfr=0.125,
        epi_kind="form",
        glyph_history=["AL", "IL"],
        private_memory={"old": [1, 2]},
    )
    graph.add_node(
        "peer",
        EPI=0.125,
        nu_f=2.0,
        theta=0.25,
        delta_nfr=0.125,
        epi_kind="form",
        glyph_history=["AL", "IL"],
    )
    graph.add_edge("parent", "peer", weight=0.375)
    if graph.is_multigraph():
        graph.add_edge("parent", "peer", weight=0.0)
    return graph


def _policy(order="pre_val", **kwargs):
    return DerivedExpansionPolicy(
        source_order=order,
        form_ratio=kwargs.get("form_ratio", 0.5),
        capacity_ratio=kwargs.get("capacity_ratio", 0.5),
    )


def _plain(graph):
    edges = (
        tuple(graph.edges(keys=True, data=True))
        if graph.is_multigraph()
        else tuple(graph.edges(data=True))
    )
    return structural_proof_signature(
        (
            graph.graph,
            tuple(graph.nodes(data=True)),
            edges,
            getattr(graph, "_last_operator_applied", None),
        ),
        opaque_references=(graph,),
    )


@pytest.mark.parametrize(
    "graph_type", [nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph]
)
@pytest.mark.parametrize("order", ["pre_val", "post_val"])
@pytest.mark.parametrize("sign", [-1, 1])
def test_actual_val_parent_and_declared_signed_child_map(graph_type, order, sign):
    graph = _graph(graph_type, epi=sign * 0.5)
    ordinary = deepcopy(graph)
    source = capture_retained_organization(graph, ("parent",))
    peer = deepcopy(graph.nodes["peer"])
    result = execute_derived_expansion_stage(graph, source, "child", _policy(order))
    execute_pointwise_stage(
        ordinary, Expansion(), ("parent",), require_exact_operator=True
    )
    assert graph.nodes["parent"] == ordinary.nodes["parent"]
    assert graph.nodes["peer"] == peer
    assert tuple(graph) == ("parent", "peer", "child")
    assert graph.degree("child") == 0
    assert tuple(graph.edges) == tuple(ordinary.edges)
    child = graph.nodes["child"]
    assert child[EPI_PRIMARY] == sign * (0.25 if order == "pre_val" else 0.375)
    assert child[VF_PRIMARY] == (0.5 if order == "pre_val" else 0.75)
    assert child[THETA_PRIMARY] == 0.25
    assert child[DNFR_PRIMARY] == 0.0
    assert child["epi_kind"] == "form"
    assert child["derived_from"] == "parent"
    assert child["derivation_source_order"] == order
    assert child["epi_history"] == [child[EPI_PRIMARY]]
    assert child["glyph_history"] == []
    assert "epi_time_history" not in child
    assert "parent_node" not in child and "hierarchy_level" not in child
    assert "private_memory" not in child and "sub_nodes" not in graph.nodes["parent"]
    assert read_grammar_state(child).operator_step == 0
    assert not read_grammar_state(child).prior_coherence
    assert result.after.members == ("parent", "child")
    assert retained_organization_matches_graph(graph, result.after)
    assert result.scale_result.after.member_states[0].epi == sign * 0.75
    assert result.inheritance_source.epi == sign * (0.5 if order == "pre_val" else 0.75)


@pytest.mark.parametrize("field", ["form_ratio", "capacity_ratio"])
@pytest.mark.parametrize(
    "invalid", [0, -0.5, 1.5, True, "0.5", float("nan"), float("inf")]
)
def test_policy_ratios_reject_invalid_values(field, invalid):
    with pytest.raises((TypeError, ValueError)):
        _policy(**{field: invalid})


@pytest.mark.parametrize("order", [None, "pre", "post", True])
def test_order_is_explicit_and_admitted(order):
    with pytest.raises((TypeError, ValueError)):
        _policy(order)
    with pytest.raises(TypeError):
        DerivedExpansionPolicy(form_ratio=0.5, capacity_ratio=0.5)


def test_policy_is_readmitted_when_frozen_fields_are_tampered():
    graph = _graph()
    source = capture_retained_organization(graph, ("parent",))
    policy = _policy()
    object.__setattr__(policy, "capacity_ratio", float("nan"))
    before = _plain(graph)
    with pytest.raises(ValueError):
        execute_derived_expansion_stage(graph, source, "child", policy)
    assert _plain(graph) == before


def test_valid_policy_rebinding_is_normalized_at_invocation():
    graph = _graph()
    policy = _policy()
    object.__setattr__(policy, "form_ratio", Fraction(1, 2))
    result = execute_derived_expansion_stage(
        graph, capture_retained_organization(graph, ("parent",)), "child", policy
    )
    assert result.policy is not policy
    assert type(result.policy.form_ratio) is float
    assert result.policy.form_ratio == 0.5


@pytest.mark.parametrize("child", ["parent", None, [], {"id": 1}])
def test_invalid_or_colliding_identity_preserves_graph(child):
    graph = _graph()
    source = capture_retained_organization(graph, ("parent",))
    before = _plain(graph)
    with pytest.raises((TypeError, ValueError)):
        execute_derived_expansion_stage(graph, source, child, _policy())
    assert _plain(graph) == before


def test_bool_identity_retains_native_networkx_collision_semantics():
    graph = _graph()
    source = capture_retained_organization(graph, ("parent",))
    result = execute_derived_expansion_stage(graph, source, True, _policy())
    assert result.child is True and graph.degree(True) == 0
    new_source = capture_retained_organization(graph, ("parent",))
    before = _plain(graph)
    with pytest.raises(ValueError, match="already exists"):
        execute_derived_expansion_stage(graph, new_source, 1, _policy())
    assert _plain(graph) == before


@pytest.mark.parametrize("change", ["foreign", "multiple", "form", "history", "edge"])
def test_source_membership_and_consumed_boundary_are_live(change):
    graph = _graph()
    source = capture_retained_organization(
        _graph() if change == "foreign" else graph,
        ("parent", "peer") if change == "multiple" else ("parent",),
    )
    if change == "form":
        graph.nodes["parent"]["EPI"] = 0.25
    elif change == "history":
        graph.nodes["parent"]["glyph_history"].append("IL")
    elif change == "edge":
        graph.edges["parent", "peer"]["weight"] = 0.5
    before = _plain(graph)
    with pytest.raises(ValueError):
        execute_derived_expansion_stage(graph, source, "child", _policy())
    assert _plain(graph) == before


@pytest.mark.parametrize("channel", ["epi", "capacity"])
def test_nonzero_child_product_underflow_rolls_back_actual_parent_val(channel):
    graph = _graph(**{channel: float.fromhex("0x0.0000000000001p-1022")})
    source = capture_retained_organization(graph, ("parent",))
    before = _plain(graph)
    with pytest.raises(ValueError, match="underflow"):
        execute_derived_expansion_stage(graph, source, "child", _policy())
    assert _plain(graph) == before


def test_legitimate_zero_parent_coordinates_remain_zero():
    graph = _graph(epi=0.0, capacity=0.0)
    result = execute_derived_expansion_stage(
        graph, capture_retained_organization(graph, ("parent",)), "child", _policy()
    )
    assert result.after.member_states[1].epi == 0.0
    assert result.after.member_states[1].capacity == 0.0


@pytest.mark.parametrize("order", ["pre_val", "post_val"])
def test_out_of_bounds_child_rejects_without_extra_clipping(order):
    graph = _graph()
    graph.graph["EPI_MIN"] = 0.4
    source = capture_retained_organization(graph, ("parent",))
    before = _plain(graph)
    with pytest.raises(ValueError, match="outside configured"):
        execute_derived_expansion_stage(graph, source, "child", _policy(order))
    assert _plain(graph) == before


def test_materialization_failure_after_val_restores_all_graph_state(monkeypatch):
    import tnfr.operators.derived_expansion as owner

    graph = _graph()
    source = capture_retained_organization(graph, ("parent",))
    before = _plain(graph)
    original = owner.materialize_isolated_member

    def corrupt(graph, node, attributes):
        original(graph, node, attributes)
        graph.nodes["peer"]["EPI"] = 0.75

    monkeypatch.setattr(owner, "materialize_isolated_member", corrupt)
    with pytest.raises(RuntimeError, match="retained core"):
        execute_derived_expansion_stage(graph, source, "child", _policy())
    assert _plain(graph) == before


def test_result_rejects_mixed_inheritance_boundary():
    graph = _graph()
    result = execute_derived_expansion_stage(
        graph, capture_retained_organization(graph, ("parent",)), "child", _policy()
    )
    with pytest.raises(ValueError, match="source or member"):
        replace(result, inheritance_source=result.scale_result.after.member_states[0])


def test_summary_rejects_inherited_temporal_evidence_despite_equal_core():
    graph = _graph()
    result = execute_derived_expansion_stage(
        graph, capture_retained_organization(graph, ("parent",)), "child", _policy()
    )
    graph.nodes["child"]["epi_time_history"] = [(0.0, 0.25)]
    changed = capture_retained_organization(graph, ("parent", "child"))
    with pytest.raises(ValueError, match="initial EPI sample"):
        replace(result, after=changed)


def test_summary_rejects_changed_incoming_port_order_with_same_edge_sequence():
    graph = _graph(nx.DiGraph)
    graph.add_edge("peer", "parent")
    graph.add_edge("other", "parent")
    result = execute_derived_expansion_stage(
        graph, capture_retained_organization(graph, ("parent",)), "child", _policy()
    )
    edges = tuple(graph.edges)
    graph.remove_edge("peer", "parent")
    graph.add_edge("peer", "parent")
    assert tuple(graph.edges) == edges
    changed = capture_retained_organization(graph, ("parent", "child"))
    with pytest.raises(ValueError, match="retain its core"):
        replace(result, after=changed)


@pytest.mark.parametrize(
    "history_key,history",
    [
        ("glyph_history", ["AL"]),
        ("epi_history", [0.25, 0.5]),
        ("epi_time_history", [(0.0, 0.5)]),
        ("_grammar_prior_coherence", True),
    ],
)
def test_shared_materializer_rejects_inherited_context_before_insertion(
    history_key, history
):
    graph = _graph()
    attributes = {
        EPI_PRIMARY: 0.5,
        VF_PRIMARY: 1.0,
        THETA_PRIMARY: 0.0,
        DNFR_PRIMARY: 0.0,
        "epi_history": [0.5],
        "glyph_history": [],
    }
    attributes[history_key] = history
    before = _plain(graph)
    with pytest.raises(ValueError):
        materialize_isolated_member(graph, "child", attributes)
    assert _plain(graph) == before


def test_thol_child_attributes_keep_original_materialization_values_and_order():
    graph = _graph(epi=0.875)
    graph.graph.update(
        THOL_METABOLIC_ENABLED=False, THOL_PROPAGATION_ENABLED=False, _t=2.0
    )
    graph.nodes["parent"].update(
        glyph_history=["IL", "OZ"],
        epi_time_history=[(0.0, 0.5), (1.0, 0.625), (2.0, 0.875)],
    )
    execute_self_organization_stage(graph, SelfOrganization(), ("parent",), tau=0.1)
    child = graph.nodes["parent"]["sub_nodes"][0]
    expected = {
        EPI_PRIMARY: 0.3 * 0.875,
        VF_PRIMARY: 0.95,
        THETA_PRIMARY: 0.25,
        DNFR_PRIMARY: 0.0,
        "parent_node": "parent",
        "hierarchy_level": 1,
        "_bifurcation_level": 1,
        "_hierarchy_path": ["parent"],
        "epi_history": [0.3 * 0.875],
        "glyph_history": [],
    }
    assert graph.nodes[child] == expected
    assert tuple(graph.nodes[child]) == tuple(expected)
    assert graph.degree(child) == 0
    assert read_grammar_state(graph.nodes[child]).history == ()
