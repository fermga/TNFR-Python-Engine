"""Boundaries of association-derived access followed by actual Coupling.

The small association states here are explicitly supplied controls. Consistent
metadata is not authenticated past THOL execution. Separate participation
controls acquire their associations through actual Emission/THOL execution.
No scientific producer, candidate scan or trajectory search is imported.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy

import networkx as nx
import pytest

from tnfr.operators.birth_access_coupling import (
    execute_birth_access_coupling,
    observe_birth_association_access,
)
from tnfr.operators.grammar_debt import read_grammar_state
from tnfr.operators.network_stage import STAGE_SCHEDULE_KEY
from tnfr.operators.preconditions import OperatorPreconditionError

PARENT, CHILD, PEER, SECOND = "parent", "child", "peer", "second-child"


def _graph(graph_type=nx.Graph):
    """A declared association; only the parent's future UM is requested."""
    graph = graph_type(
        _t=0.25,
        UM_FUNCTIONAL_LINKS=True,
        UM_BIDIRECTIONAL=True,
        UM_SYNC_VF=True,
        UM_STABILIZE_DNFR=True,
        UM_COMPAT_THRESHOLD=0.5,
        VALIDATE_OPERATOR_PRECONDITIONS=True,
        GRAMMAR_CANON={"enabled": True},
        GRAMMAR_REJECTION_MODE="raise",
        RANDOM_SEED=17,
        GLYPH_FACTORS={
            "UM_theta_push": 0.5,
            "UM_vf_sync": 0.25,
            "UM_dnfr_reduction": 0.25,
        },
        hierarchy={PARENT: [CHILD]},
    )
    for node, form, capacity, phase in (
        (PARENT, 0.25, 1.0, 0.125),
        (CHILD, 0.75, 2.0, 0.375),
        (PEER, 0.5, 1.0, 0.125),
    ):
        graph.add_node(
            node,
            EPI=form,
            nu_f=capacity,
            theta=phase,
            delta_nfr=0.4,
            Si=0.5,
            epi_kind="supplied-form",
            glyph_history=[],
        )
    graph.nodes[PARENT].update(
        sub_nodes=[CHILD],
        glyph_history=["AL", "IL"],
        _grammar_u2_debt=0,
        _grammar_prior_coherence=True,
        _operator_step=2,
    )
    graph.nodes[CHILD]["parent_node"] = PARENT
    graph.add_edge(PARENT, PEER, weight=0.37, origin="supplied-old-support")
    return graph


def _state(graph):
    """Graph-owned entry state, retaining the existing monitor identity."""
    edges = (
        tuple(graph.edges(keys=True, data=True))
        if graph.is_multigraph()
        else tuple(graph.edges(data=True))
    )
    return (
        deepcopy(tuple(graph.nodes(data=True))),
        deepcopy(edges),
        deepcopy(
            {
                key: value
                for key, value in graph.graph.items()
                if key != "integrity_monitor"
            }
        ),
        graph.graph.get("integrity_monitor"),
        hasattr(graph, "_last_operator_applied"),
        getattr(graph, "_last_operator_applied", None),
    )


def _no_association(graph):
    graph.nodes[PARENT].pop("sub_nodes")
    graph.graph.pop("hierarchy")


def _second_association(graph):
    graph.add_node(SECOND, **deepcopy(graph.nodes[CHILD]))
    graph.nodes[PARENT]["sub_nodes"].append(SECOND)
    graph.graph["hierarchy"][PARENT].append(SECOND)


def _damage(graph, kind):
    if kind == "wrong_local_container":
        graph.nodes[PARENT]["sub_nodes"] = (CHILD,)
    elif kind == "wrong_hierarchy_container":
        graph.graph["hierarchy"] = None
    elif kind == "wrong_index_container":
        graph.graph["hierarchy"][PARENT] = (CHILD,)
    elif kind == "duplicate":
        graph.nodes[PARENT]["sub_nodes"].append(CHILD)
        graph.graph["hierarchy"][PARENT].append(CHILD)
    elif kind == "self":
        graph.nodes[PARENT]["sub_nodes"] = [PARENT]
        graph.graph["hierarchy"][PARENT] = [PARENT]
    elif kind == "missing_child":
        graph.nodes[PARENT]["sub_nodes"].append("missing-child")
        graph.graph["hierarchy"][PARENT].append("missing-child")
    elif kind == "missing_parent_reference":
        graph.nodes[CHILD].pop("parent_node")
    elif kind == "connected_stale_sibling":
        _second_association(graph)
        graph.add_edge(SECOND, PEER)
        graph.nodes[SECOND]["parent_node"] = PEER
    elif kind == "index_order_disagrees":
        _second_association(graph)
        graph.graph["hierarchy"][PARENT].reverse()
    elif kind == "unhashable_child":
        graph.nodes[PARENT]["sub_nodes"] = [[CHILD]]
        graph.graph["hierarchy"][PARENT] = [[CHILD]]
    else:
        raise AssertionError(kind)


@pytest.mark.parametrize(
    "kind, error",
    [
        ("wrong_local_container", TypeError),
        ("wrong_hierarchy_container", TypeError),
        ("wrong_index_container", TypeError),
        ("duplicate", ValueError),
        ("self", ValueError),
        ("missing_child", ValueError),
        ("missing_parent_reference", ValueError),
        ("connected_stale_sibling", ValueError),
        ("index_order_disagrees", ValueError),
        ("unhashable_child", TypeError),
    ],
)
def test_invalid_complete_association_never_becomes_abstention(kind, error):
    graph = _graph()
    _damage(graph, kind)
    before = _state(graph)
    with pytest.raises(error):
        execute_birth_access_coupling(graph, PARENT)
    assert _state(graph) == before
    assert not graph.has_edge(PARENT, CHILD)


def test_passive_reader_rejects_stale_connected_reference_without_repair():
    graph = _graph()
    _damage(graph, "connected_stale_sibling")
    before = _state(graph)
    with pytest.raises(ValueError):
        observe_birth_association_access(graph, PARENT)
    assert _state(graph) == before


@pytest.mark.parametrize("index_present", [False, True])
def test_legacy_missing_parent_index_keeps_local_association(index_present):
    graph = _graph()
    if index_present:
        # Unrelated malformed entries are outside this direct reader's domain.
        graph.graph["hierarchy"] = {PEER: "unconsumed-index-entry"}
    else:
        graph.graph.pop("hierarchy")
    before = _state(graph)
    access = observe_birth_association_access(graph, PARENT)
    assert access.status == "selected"
    assert access.selected_partner == CHILD
    assert access.associated_children == access.isolated_children == (CHILD,)
    assert access.policy == "unique_isolated_birth_child"
    assert _state(graph) == before


def test_passive_access_does_not_consume_grammar_or_numeric_um_readiness():
    graph = _graph()
    graph.graph["GRAMMAR_CANON"] = {"enabled": False}
    graph.graph["GRAMMAR_REJECTION_MODE"] = "fallback"
    graph.nodes[CHILD].update(EPI="not-a-form", theta=None)
    before = _state(graph)
    access = observe_birth_association_access(graph, PARENT)
    assert access.status == "selected"
    assert access.selected_partner == CHILD
    assert _state(graph) == before


def test_ambiguous_access_does_not_filter_on_phase_or_unused_numeric_fields():
    graph = _graph()
    _second_association(graph)
    graph.nodes[SECOND].update(theta=3.0, EPI="not-a-form")
    before = _state(graph)
    result = execute_birth_access_coupling(graph, PARENT)
    assert result.access.status == "ambiguous"
    assert result.access.isolated_children == (CHILD, SECOND)
    assert result.access.selected_partner is None
    assert result.stage_result is None
    assert not result.executed
    assert _state(graph) == before
    assert STAGE_SCHEDULE_KEY not in graph.graph


@pytest.mark.parametrize("support", ["incoming", "self_loop"])
def test_directed_nonisolated_child_is_excluded_without_global_candidate_search(
    support,
):
    graph = _graph(nx.MultiDiGraph)
    if support == "incoming":
        graph.add_edge(PEER, CHILD, key="incoming-only")
        assert graph.out_degree(CHILD) == 0
    else:
        graph.add_edge(CHILD, CHILD, key="self-support")
    graph.add_node("unlisted-isolate", parent_node=PARENT)
    before = _state(graph)
    result = execute_birth_access_coupling(graph, PARENT)
    assert result.access.status == "no_isolated_child"
    assert result.access.connected_children == (CHILD,)
    assert result.access.isolated_children == ()
    assert result.access.associated_children == (CHILD,)
    assert result.stage_result is None
    assert not result.executed
    assert _state(graph) == before


@pytest.mark.parametrize(
    "options, error",
    [
        ([], TypeError),
        ({"potential_partner": CHILD}, ValueError),
        ({"sequence_context": ()}, ValueError),
        ({"compute_delta_nfr": None}, ValueError),
        ({"validate_preconditions": False}, ValueError),
        ({"collect_metrics": 1}, OperatorPreconditionError),
        ({"window": True}, TypeError),
        ({"dt": float("nan")}, OperatorPreconditionError),
    ],
)
def test_wrapper_request_errors_precede_empty_access_abstention(options, error):
    graph = _graph()
    _no_association(graph)
    before = _state(graph)
    with pytest.raises(error):
        execute_birth_access_coupling(graph, PARENT, execution_kwargs=options)
    assert _state(graph) == before


@pytest.mark.parametrize("policy", ["disabled_grammar", "fallback"])
def test_execute_requires_strict_policy_even_without_access(policy):
    graph = _graph()
    _no_association(graph)
    if policy == "disabled_grammar":
        graph.graph["GRAMMAR_CANON"] = {"enabled": False}
    else:
        graph.graph["GRAMMAR_REJECTION_MODE"] = "fallback"
    before = _state(graph)
    with pytest.raises(ValueError):
        execute_birth_access_coupling(graph, PARENT)
    assert _state(graph) == before


def test_unique_access_cannot_replace_a_grammar_blocked_um():
    graph = _graph()
    graph.nodes[PARENT]["_grammar_u2_debt"] = 3
    before = _state(graph)
    with pytest.raises((ValueError, RuntimeError, OperatorPreconditionError)):
        execute_birth_access_coupling(graph, PARENT)
    assert _state(graph) == before
    assert read_grammar_state(graph.nodes[PARENT]).history == ("AL", "IL")


def test_missing_target_rejects_without_examining_an_unrelated_association():
    graph = _graph()
    before = _state(graph)
    with pytest.raises(KeyError):
        observe_birth_association_access(graph, "missing-parent")
    assert _state(graph) == before


class _LateCouplingFailure:
    def before_operator(self, graph, node):
        pass

    def after_operator(self, graph, node, operator):
        assert operator == "coupling"
        assert node == PARENT
        assert graph.has_edge(PARENT, CHILD)
        assert read_grammar_state(graph.nodes[PARENT]).history[-1] == "UM"
        assert graph.graph["_t"] == 0.25
        graph.nodes[CHILD]["parent_node"] = PEER
        graph.nodes[PEER]["EPI"] = 9.0
        graph.graph["declared_late_fault"] = True
        raise RuntimeError("declared birth-access UM completion failure")


def test_late_actual_um_failure_restores_preexisting_association_and_full_state():
    graph = _graph()
    monitor = _LateCouplingFailure()
    graph.graph["integrity_monitor"] = monitor
    before = _state(graph)
    with pytest.raises(
        RuntimeError, match="declared birth-access UM completion failure"
    ):
        execute_birth_access_coupling(graph, PARENT)
    assert _state(graph) == before
    assert graph.degree(CHILD) == 0
    assert graph.graph["integrity_monitor"] is monitor


class _FailingOptions(Mapping):
    def __init__(self, graph):
        self.graph = graph

    def __iter__(self):
        self.graph.nodes[CHILD]["parent_node"] = PEER
        self.graph.graph["declared_option_fault"] = True
        raise RuntimeError("declared option materialization failure")

    def __len__(self):
        return 1

    def __getitem__(self, key):
        raise KeyError(key)


def test_option_materialization_failure_is_inside_invocation_transaction():
    graph = _graph()
    before = _state(graph)
    with pytest.raises(RuntimeError, match="declared option materialization failure"):
        execute_birth_access_coupling(
            graph, PARENT, execution_kwargs=_FailingOptions(graph)
        )
    assert _state(graph) == before


class _RetargetingOptions(Mapping):
    def __init__(self, graph):
        self.graph = graph

    def __iter__(self):
        self.graph.nodes[PARENT]["sub_nodes"] = [SECOND]
        self.graph.graph["hierarchy"][PARENT] = [SECOND]
        return iter(("validate_preconditions",))

    def __len__(self):
        return 1

    def __getitem__(self, key):
        if key == "validate_preconditions":
            return True
        raise KeyError(key)


def test_successful_option_materialization_cannot_change_access_source():
    graph = _graph()
    # This declared isolate is not in the original direct association list.
    graph.add_node(SECOND, **deepcopy(graph.nodes[CHILD]))
    result = execute_birth_access_coupling(
        graph, PARENT, execution_kwargs=_RetargetingOptions(graph)
    )
    assert result.executed
    assert result.access.associated_children == (CHILD,)
    assert result.access.selected_partner == CHILD
    assert graph.has_edge(PARENT, CHILD)
    assert graph.degree(SECOND) == 0
    assert graph.nodes[PARENT]["sub_nodes"] == [CHILD]
    assert graph.graph["hierarchy"][PARENT] == [CHILD]
    assert read_grammar_state(graph.nodes[PARENT]).history == ("AL", "IL", "UM")
    assert graph.graph["_t"] == 0.25
