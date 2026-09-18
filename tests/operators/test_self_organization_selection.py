"""Set-valued THOL observations and explicit, revalidated public dispatch."""

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace

import networkx as nx
import pytest

from tnfr.constants.aliases import ALIAS_D2EPI, ALIAS_DNFR, ALIAS_EPI, ALIAS_VF
from tnfr.operators.definitions import SelfOrganization
from tnfr.operators.preconditions import OperatorPreconditionError
from tnfr.operators.self_organization_selection import (
    execute_eligible_self_organization_stage,
    observe_self_organization_eligibility,
)


def _graph(nodes=(0, 1)):
    graph = nx.Graph(THOL_METABOLIC_ENABLED=False, RANDOM_SEED=17)
    for node in nodes:
        graph.add_node(node, **{
            ALIAS_EPI[0]: 0.6, ALIAS_VF[0]: 1.0, ALIAS_DNFR[0]: 0.2,
            "theta": 0.0, "glyph_history": ["OZ"],
            "epi_time_history": [(0.0, 0.0), (1.0, 0.1), (2.0, 0.6)],
        })
    if len(nodes) > 1:
        graph.add_edges_from(zip(nodes, nodes[1:]))
    return graph


def _state(graph):
    return deepcopy((
        {key: value for key, value in graph.graph.items()
         if key != "integrity_monitor"},
        tuple((node, dict(data)) for node, data in graph.nodes(data=True)),
        tuple(graph.edges(data=True)),
        (hasattr(graph, "_last_operator_applied"),
         getattr(graph, "_last_operator_applied", None)),
    ))


def _rows(report):
    return {row.node: row for row in report.candidates}


def test_observation_preserves_all_candidates_without_selecting_one_parent():
    graph = _graph()
    before = _state(graph)
    report = observe_self_organization_eligibility(graph)
    assert report.eligible_nodes == (0, 1)
    assert report.joint_stage_viable is True
    for row in _rows(report).values():
        assert row.acceleration.available is True
        assert row.grammar_allowed is True
        assert row.application_preconditions_passed is True
        assert row.proposal_valid is True
        assert row.threshold_crossed is True
        assert row.birth_proposed is True
        assert row.depth_limit_reached is False
    assert _state(graph) == before


def test_grammar_denial_does_not_hide_a_separately_valid_birth_proposal():
    graph = _graph()
    graph.nodes[1]["glyph_history"] = []
    before = _state(graph)
    report = observe_self_organization_eligibility(graph)
    rows = _rows(report)
    assert report.eligible_nodes == (0,)
    assert rows[1].grammar_allowed is False
    assert rows[1].grammar_violations
    assert rows[1].proposal_valid is True
    assert rows[1].birth_proposed is True
    assert _state(graph) == before


@pytest.mark.parametrize("enabled, requested, eligible", (
    (False, True, True), (True, False, True), (True, True, False),
))
def test_optional_gate_switches_match_public_semantics(enabled, requested, eligible):
    graph = _graph()
    graph.graph.update(VALIDATE_OPERATOR_PRECONDITIONS=enabled, THOL_MIN_VF=2.0)
    before = _state(graph)
    report = observe_self_organization_eligibility(
        graph, execution_kwargs={"validate_preconditions": requested},
    )
    assert bool(report.eligible_nodes) is eligible
    for row in _rows(report).values():
        assert row.optional_gate_enabled is (enabled and requested)
        assert row.application_preconditions_passed is eligible
        assert row.proposal_valid is True
        assert row.birth_proposed is True
    assert _state(graph) == before


def test_short_authoritative_history_does_not_reuse_old_cache_or_legacy_values():
    graph = _graph()
    for node in graph:
        graph.nodes[node].update(
            epi_time_history=[(2.0, 0.6)], epi_history=[0.0, 0.0, 100.0],
        )
        graph.nodes[node][ALIAS_D2EPI[0]] = 100.0
    before = _state(graph)
    report = observe_self_organization_eligibility(graph)
    assert report.eligible_nodes == ()
    for row in _rows(report).values():
        assert row.acceleration.available is False
        assert row.acceleration.value is None
        assert row.birth_proposed is False
    assert _state(graph) == before


def test_stale_physical_history_is_reported_without_legacy_fallback():
    graph = _graph()
    graph.nodes[0]["epi_time_history"][-1] = (2.0, 0.5)
    graph.nodes[0]["epi_history"] = [0.0, 0.0, 100.0]
    before = _state(graph)
    report = observe_self_organization_eligibility(graph)
    row = _rows(report)[0]
    assert report.eligible_nodes == (1,)
    assert row.acceleration_error
    assert row.proposal_valid is False
    assert row.proposal_error
    assert row.birth_proposed is None
    assert _state(graph) == before


def test_threshold_crossing_at_maximum_depth_is_not_birth_eligibility():
    graph = _graph()
    graph.graph["THOL_MAX_BIFURCATION_DEPTH"] = 0
    before = _state(graph)
    report = observe_self_organization_eligibility(graph)
    assert report.eligible_nodes == ()
    for row in _rows(report).values():
        assert row.proposal_valid is True
        assert row.threshold_crossed is True
        assert row.depth_limit_reached is True
        assert row.birth_proposed is False
    assert _state(graph) == before


def test_complete_proposal_failure_is_separate_from_acceleration_and_grammar():
    graph = _graph()
    graph.graph["THOL_PROPAGATION_ENABLED"] = True
    before = _state(graph)
    report = observe_self_organization_eligibility(graph)
    assert report.eligible_nodes == ()
    for row in _rows(report).values():
        assert row.acceleration.available is True
        assert row.grammar_allowed is True
        assert row.application_preconditions_passed is True
        assert row.proposal_valid is False
        assert row.proposal_error
        assert row.birth_proposed is None
    assert _state(graph) == before


def test_empty_dispatch_is_a_true_noop_without_history_or_schedule_writes():
    graph = _graph()
    for node in graph:
        graph.nodes[node]["glyph_history"] = []
    before = _state(graph)
    execute_eligible_self_organization_stage(graph)
    assert _state(graph) == before


def test_dispatch_recomputes_live_state_and_never_trusts_a_previous_observation():
    graph = _graph()
    report = observe_self_organization_eligibility(graph)
    assert report.eligible_nodes == (0, 1)
    for node in graph:
        graph.nodes[node]["glyph_history"] = []
    before = _state(graph)
    execute_eligible_self_organization_stage(graph)
    assert _state(graph) == before


def test_collision_allocation_keeps_both_eligible_parents_and_creates_one_child_each():
    graph = _graph((1, "1"))
    report = observe_self_organization_eligibility(graph)
    assert report.eligible_nodes == (1, "1")
    assert report.joint_stage_viable is True
    execute_eligible_self_organization_stage(graph)
    assert graph.nodes[1]["sub_nodes"] == ["1_sub_0"]
    assert graph.nodes["1"]["sub_nodes"] == ["1_sub_1"]
    assert graph.nodes["1_sub_0"]["parent_node"] == 1
    assert graph.nodes["1_sub_1"]["parent_node"] == "1"
    assert graph.degree("1_sub_0") == graph.degree("1_sub_1") == 0
    assert graph.nodes[1]["glyph_history"][-1] == "THOL"
    assert graph.nodes["1"]["glyph_history"][-1] == "THOL"


class _RecordingMonitor:
    def __init__(self):
        self.events = []

    def before_operator(self, graph, node):
        self.events.append(("before", node))

    def after_operator(self, graph, node, operator):
        self.events.append(("after", node))


def test_observation_does_not_call_monitor_callbacks():
    graph = _graph()
    monitor = _RecordingMonitor()
    graph.graph["integrity_monitor"] = monitor
    before = _state(graph)
    report = observe_self_organization_eligibility(graph)
    assert report.eligible_nodes == (0, 1)
    assert monitor.events == []
    assert graph.graph["integrity_monitor"] is monitor
    assert _state(graph) == before


def test_late_monitor_failure_restores_every_selected_parent_and_monitor():
    class RejectingMonitor(_RecordingMonitor):
        def after_operator(self, graph, node, operator):
            super().after_operator(graph, node, operator)
            if node == 1:
                graph.graph["late_side_effect"] = True
                raise RuntimeError("second parent rejected")

    graph = _graph()
    monitor = RejectingMonitor()
    graph.graph["integrity_monitor"] = monitor
    before = _state(graph)
    with pytest.raises(RuntimeError, match="second parent rejected"):
        execute_eligible_self_organization_stage(graph)
    assert _state(graph) == before
    assert graph.graph["integrity_monitor"] is monitor
    assert monitor.events == []


@pytest.mark.parametrize("key", ("targets", "sequence_context", "compute_delta_nfr"))
@pytest.mark.parametrize("entry", (
    observe_self_organization_eligibility, execute_eligible_self_organization_stage,
))
def test_extra_selection_or_callback_keywords_are_not_silently_accepted(key, entry):
    graph = _graph()
    before = _state(graph)
    with pytest.raises((TypeError, ValueError)):
        entry(graph, execution_kwargs={key: None})
    assert _state(graph) == before


@pytest.mark.parametrize("entry", (
    observe_self_organization_eligibility, execute_eligible_self_organization_stage,
))
def test_adversarial_kwargs_materialization_cannot_leave_graph_side_effects(entry):
    graph = _graph()
    before = _state(graph)

    class MutatingKeywords(Mapping):
        def __iter__(self):
            graph.add_node("injected-parent")
            graph.graph["materialization_effect"] = True
            raise RuntimeError("kwargs materialization rejected")

        def __len__(self):
            return 1

        def __getitem__(self, key):
            return 0.1

    with pytest.raises((TypeError, ValueError, RuntimeError)):
        entry(graph, execution_kwargs=MutatingKeywords())
    assert _state(graph) == before


def test_late_metric_failure_rolls_back_an_otherwise_viable_joint_birth(monkeypatch):
    graph = _graph()
    before = _state(graph)
    original = SelfOrganization._collect_metrics

    def fail_second(self, target, node, state_before):
        if node == 1:
            raise RuntimeError("second metric rejected")
        return original(self, target, node, state_before)

    monkeypatch.setattr(SelfOrganization, "_collect_metrics", fail_second)
    with pytest.raises(RuntimeError, match="second metric rejected"):
        execute_eligible_self_organization_stage(
            graph, execution_kwargs={"collect_metrics": True},
        )
    assert _state(graph) == before


def test_empty_graph_observation_and_dispatch_do_not_create_schedule_metadata():
    graph = nx.Graph()
    before = _state(graph)
    report = observe_self_organization_eligibility(graph)
    result = execute_eligible_self_organization_stage(graph)
    assert report.candidates == ()
    assert report.eligible_nodes == ()
    assert result.stage_result is None
    assert result.parent_children == ()
    assert _state(graph) == before


def test_short_trace_retention_does_not_change_runtime_grammar_window():
    graph = _graph()
    for node in graph:
        graph.nodes[node]["glyph_history"] = ["OZ", "EN", "EN"]
    report = observe_self_organization_eligibility(
        graph, execution_kwargs={"window": 1},
    )
    assert report.eligible_nodes == (0, 1)
    result = execute_eligible_self_organization_stage(
        graph, execution_kwargs={"window": 1},
    )
    assert {parent for parent, child in result.parent_children} == {0, 1}
    for parent in (0, 1):
        assert list(graph.nodes[parent]["glyph_history"]) == ["THOL"]


def test_observation_is_immutable_and_does_not_retain_mutable_options():
    graph = _graph()
    options = {"tau": 0.1}
    report = observe_self_organization_eligibility(graph, execution_kwargs=options)
    options["tau"] = 99.0
    graph.nodes[0]["epi_time_history"][-1] = (2.0, -5.0)
    assert dict(report.execution_options)["tau"] == 0.1
    assert _rows(report)[0].acceleration.samples[-1] == (2.0, 0.6)
    with pytest.raises(FrozenInstanceError):
        report.tau = 99.0
    with pytest.raises(FrozenInstanceError):
        _rows(report)[0].proposal_valid = False


@pytest.mark.parametrize("renaming", (
    {0: "north", 1: "south", 2: "east", 3: "west"},
    {0: 20, 1: -1, 2: 0, 3: 7},
    {0: ("n", 4), 1: ("n", 1), 2: ("n", 9), 3: ("n", 3)},
))
def test_supplied_state_complete_eligible_set_covaries_under_finite_relabeling(renaming):
    # This is a finite supplied-history control. It establishes neither causal
    # history provenance nor arbitrary all-state runtime equivariance.
    graph = _graph((0, 1, 2, 3))
    renamed = _graph(tuple(renaming[index] for index in range(4)))
    for subject, labels in ((graph, dict(enumerate(range(4)))), (renamed, renaming)):
        subject.nodes[labels[1]]["glyph_history"] = []
        subject.nodes[labels[3]]["_bifurcation_level"] = 5
    before = _state(graph), _state(renamed)
    left = observe_self_organization_eligibility(graph)
    right = observe_self_organization_eligibility(renamed)
    assert set(left.eligible_nodes) == {0, 2}
    assert set(right.eligible_nodes) == {renaming[0], renaming[2]}
    for node, row in _rows(left).items():
        assert replace(_rows(right)[renaming[node]], node=node) == row
    assert left.joint_stage_viable is right.joint_stage_viable is True
    assert (_state(graph), _state(renamed)) == before


def test_joint_support_failure_is_reported_and_prevents_any_dispatch(monkeypatch):
    from tnfr.operators import self_organization_selection as selection

    graph = _graph()
    before = _state(graph)

    def reject_joint(snapshot, operator, proposals):
        assert {node for node, proposal in proposals} == {0, 1}
        snapshot.graph["detached_attempt"] = True
        raise ValueError("joint support rejected")

    monkeypatch.setattr(selection, "_merge_and_validate_self_organization_stage", reject_joint)
    report = observe_self_organization_eligibility(graph)
    assert report.eligible_nodes == (0, 1)
    assert report.joint_stage_viable is False
    assert "joint support rejected" in report.joint_error
    assert _state(graph) == before
    with pytest.raises(ValueError, match="joint THOL birth proposal"):
        execute_eligible_self_organization_stage(graph)
    assert _state(graph) == before


def test_runtime_grammar_replacement_is_rolled_back_instead_of_reported_as_birth(monkeypatch):
    from tnfr.operators import grammar_application

    graph = _graph()
    before = _state(graph)
    monkeypatch.setattr(grammar_application, "enforce_canonical_grammar",
                        lambda *args, **kwargs: "IL")
    with pytest.raises(RuntimeError, match="built-in two-phase"):
        execute_eligible_self_organization_stage(graph)
    assert _state(graph) == before


def test_successful_monitor_cannot_silently_delete_an_unselected_original_node():
    class DeletingMonitor(_RecordingMonitor):
        def after_operator(self, graph, node, operator):
            super().after_operator(graph, node, operator)
            graph.remove_node(1)

    graph = _graph()
    graph.nodes[1]["glyph_history"] = []
    monitor = DeletingMonitor()
    graph.graph["integrity_monitor"] = monitor
    before = _state(graph)
    assert observe_self_organization_eligibility(graph).eligible_nodes == (0,)
    with pytest.raises(RuntimeError, match="unexpected node support"):
        execute_eligible_self_organization_stage(graph)
    assert _state(graph) == before
    assert graph.graph["integrity_monitor"] is monitor
    assert monitor.events == []


def test_invalid_joint_pattern_sink_is_diagnosed_without_removing_individual_eligibility():
    graph = _graph()
    graph.graph["recognized_coherence_patterns"] = {}
    before = _state(graph)
    report = observe_self_organization_eligibility(graph)
    assert report.eligible_nodes == (0, 1)
    assert report.joint_stage_viable is False
    assert "recognized_coherence_patterns" in report.joint_error
    assert _state(graph) == before
    with pytest.raises(ValueError, match="joint THOL birth proposal"):
        execute_eligible_self_organization_stage(graph)
    assert _state(graph) == before


@pytest.mark.parametrize("graph_type", (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph))
def test_successful_monitor_cannot_remove_original_edge_support(graph_type):
    class DeletingEdgeMonitor(_RecordingMonitor):
        def after_operator(self, graph, node, operator):
            super().after_operator(graph, node, operator)
            if graph.is_multigraph():
                graph.remove_edge(0, 1, key=0)
            else:
                graph.remove_edge(0, 1)

    graph = graph_type(_graph())
    graph.nodes[1]["glyph_history"] = []
    if graph.is_multigraph():
        # The original pair survives, so a pair-only comparison would miss
        # the removed original multiedge key.
        graph.add_edge(0, 1, key="retained-parallel")
    monitor = DeletingEdgeMonitor()
    graph.graph["integrity_monitor"] = monitor
    before = _state(graph)
    original_edges = tuple(graph.edges(keys=True)) if graph.is_multigraph() else tuple(graph.edges)
    with pytest.raises(RuntimeError, match="edge support"):
        execute_eligible_self_organization_stage(graph)
    assert _state(graph) == before
    after_edges = tuple(graph.edges(keys=True)) if graph.is_multigraph() else tuple(graph.edges)
    assert after_edges == original_edges
    assert graph.graph["integrity_monitor"] is monitor
    assert monitor.events == []


@pytest.mark.parametrize("options", (
    {"tau": float("nan")}, {"tau": True}, {"window": False}, {"window": -1},
    {"collect_metrics": "false"}, {"validate_preconditions": 1},
    {"validate_nodal_equation": None}, {"dt": 0.0}, {"dt": float("inf")},
))
def test_invalid_execution_options_reject_before_graph_owned_effects(options):
    graph = _graph()
    before = _state(graph)
    for entry in (observe_self_organization_eligibility,
                  execute_eligible_self_organization_stage):
        with pytest.raises((TypeError, ValueError, OperatorPreconditionError)):
            entry(graph, execution_kwargs=options)
        assert _state(graph) == before
