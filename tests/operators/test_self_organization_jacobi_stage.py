"""Immutable all-target proposal, hierarchy merge and rollback tests for THOL."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import networkx as nx
import pytest

from tnfr.alias import get_attr
from tnfr.constants.aliases import (
    ALIAS_D2EPI,
    ALIAS_DNFR,
    ALIAS_EPI,
    ALIAS_THETA,
    ALIAS_VF,
)
from tnfr.operators.definitions import SelfOrganization
from tnfr.operators.metabolism import compute_subepi_amplitude_alignment
from tnfr.operators.network_stage import (
    OPERATOR_MAJOR_GAUSS_SEIDEL,
    STAGE_CONTRACT_KEY,
    STAGE_SCHEDULE_KEY,
    TWO_PHASE_JACOBI,
    execute_self_organization_stage,
)
from tnfr.operators.word_execution import run_network_sequence
from tnfr.operators.preconditions import OperatorPreconditionError


def _add_parent(
    graph: nx.Graph,
    node: Any,
    *,
    history: list[float],
    glyph_history: list[str] | None = None,
) -> None:
    graph.add_node(
        node,
        **{
            ALIAS_EPI[0]: 0.6,
            ALIAS_VF[0]: 1.0,
            ALIAS_DNFR[0]: 0.2,
            ALIAS_THETA[0]: 0.1,
            "epi_history": history,
            "glyph_history": list(glyph_history or ["OZ"]),
        },
    )


def _collision_graph(
    graph_type: type[nx.Graph] = nx.Graph,
) -> nx.Graph:
    graph = graph_type()
    # Distinct NetworkX identifiers with the same string rendering deliberately
    # propose the same legacy direct-call child identifier.
    _add_parent(graph, 1, history=[0.0, 0.1, 0.6])
    _add_parent(graph, "1", history=[0.0, 0.2, 0.9])
    graph.graph["THOL_METABOLIC_ENABLED"] = False
    return graph


def _plain_state(
    graph: nx.Graph,
    *,
    omit_graph_keys: frozenset[str] = frozenset(),
) -> tuple[Any, ...]:
    omitted = {
        "integrity_monitor",
        "_node_cache",
        "_node_cache_weak",
        *omit_graph_keys,
    }
    edges = (
        tuple(
            (left, right, key, deepcopy(dict(data)))
            for left, right, key, data in graph.edges(keys=True, data=True)
        )
        if graph.is_multigraph()
        else tuple(
            (left, right, deepcopy(dict(data)))
            for left, right, data in graph.edges(data=True)
        )
    )
    return (
        tuple(graph.nodes),
        {node: deepcopy(dict(data)) for node, data in graph.nodes(data=True)},
        edges,
        {
            key: deepcopy(value)
            for key, value in graph.graph.items()
            if key not in omitted
        },
        (
            hasattr(graph, "_last_operator_applied"),
            getattr(graph, "_last_operator_applied", None),
        ),
    )


def _structural_state(graph: nx.Graph) -> tuple[Any, ...]:
    return (
        tuple(graph.nodes),
        {
            node: (
                get_attr(data, ALIAS_D2EPI, None),
                get_attr(data, ALIAS_DNFR, None),
                deepcopy(data.get("sub_nodes")),
                deepcopy(data.get("sub_epis")),
                data.get("parent_node"),
                data.get("hierarchy_level"),
                data.get("_bifurcation_level"),
                deepcopy(data.get("_hierarchy_path")),
            )
            for node, data in graph.nodes(data=True)
        },
        deepcopy(graph.graph.get("hierarchy")),
    )


@pytest.mark.parametrize(
    "graph_type", (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)
)
def test_collision_safe_support_merge_is_target_order_invariant(
    graph_type: type[nx.Graph],
) -> None:
    forward = _collision_graph(graph_type)
    reverse = _collision_graph(graph_type)

    execute_self_organization_stage(
        forward, SelfOrganization(), (1, "1"), tau=0.1
    )
    execute_self_organization_stage(
        reverse, SelfOrganization(), ("1", 1), tau=0.1
    )

    assert _structural_state(reverse) == _structural_state(forward)
    assert tuple(forward.nodes) == (1, "1", "1_sub_0", "1_sub_1")
    assert forward.nodes[1]["sub_nodes"] == ["1_sub_0"]
    assert forward.nodes["1"]["sub_nodes"] == ["1_sub_1"]
    assert forward.nodes[1]["sub_epis"][-1]["node_id"] == "1_sub_0"
    assert forward.nodes["1"]["sub_epis"][-1]["node_id"] == "1_sub_1"
    assert forward.graph["hierarchy"] == {
        1: ["1_sub_0"],
        "1": ["1_sub_1"],
    }
    assert forward.graph[STAGE_SCHEDULE_KEY]["schedule"] == TWO_PHASE_JACOBI
    assert forward.graph[STAGE_CONTRACT_KEY][
        "structural_state_target_order_invariant"
    ] is True


def test_single_target_stage_matches_direct_prepared_transaction() -> None:
    direct = nx.Graph(THOL_METABOLIC_ENABLED=False)
    staged = nx.Graph(THOL_METABOLIC_ENABLED=False)
    for graph in (direct, staged):
        _add_parent(graph, 0, history=[0.0, 0.1, 0.6])

    SelfOrganization()(direct, 0, tau=0.1, collect_metrics=True)
    execute_self_organization_stage(
        staged,
        SelfOrganization(),
        (0,),
        tau=0.1,
        collect_metrics=True,
    )

    assert tuple(staged.nodes) == tuple(direct.nodes)
    assert {
        node: dict(data) for node, data in staged.nodes(data=True)
    } == {
        node: dict(data) for node, data in direct.nodes(data=True)
    }
    stage_only = {STAGE_SCHEDULE_KEY, STAGE_CONTRACT_KEY}
    assert {
        key: value
        for key, value in staged.graph.items()
        if key not in stage_only
    } == dict(direct.graph)


def test_every_target_is_planned_from_one_unchanged_stage_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = _collision_graph()
    original = SelfOrganization._prepare_execution
    observed: list[tuple[int, tuple[Any, ...]]] = []

    def recording_prepare(self, subject, node, kwargs):
        observed.append((id(subject), tuple(subject.nodes)))
        assert "1_sub_0" not in subject
        assert "1_sub_1" not in subject
        return original(self, subject, node, kwargs)

    monkeypatch.setattr(SelfOrganization, "_prepare_execution", recording_prepare)

    execute_self_organization_stage(
        graph, SelfOrganization(), ("1", 1), tau=0.1
    )

    assert len(observed) == 2
    assert len({identity for identity, _nodes in observed}) == 1
    assert [nodes for _identity, nodes in observed] == [(1, "1"), (1, "1")]


class _RecordingMonitor:
    def __init__(self) -> None:
        self.events: list[tuple[str, Any, int]] = []
        self.pending: Any = None

    def before_operator(self, graph: nx.Graph, node: Any) -> None:
        self.pending = node
        self.events.append(("before", node, graph.number_of_nodes()))

    def after_operator(self, graph: nx.Graph, node: Any, operator: str) -> None:
        assert operator == "self_organization"
        self.events.append(("after", node, graph.number_of_nodes()))
        self.pending = None

    def discard_pending_operator(self) -> None:
        self.pending = None


def test_monitor_and_metric_streams_retain_requested_target_order() -> None:
    graph = _collision_graph()
    monitor = _RecordingMonitor()
    graph.graph["integrity_monitor"] = monitor

    execute_self_organization_stage(
        graph,
        SelfOrganization(),
        ("1", 1),
        tau=0.1,
        collect_metrics=True,
    )

    assert [(kind, node) for kind, node, _count in monitor.events] == [
        ("before", "1"),
        ("after", "1"),
        ("before", 1),
        ("after", 1),
    ]
    assert [count for kind, _node, count in monitor.events if kind == "before"] == [
        2,
        2,
    ]
    assert [count for kind, _node, count in monitor.events if kind == "after"] == [
        4,
        4,
    ]
    assert [metric["d2epi"] for metric in graph.graph["operator_metrics"]] == [
        get_attr(graph.nodes["1"], ALIAS_D2EPI),
        get_attr(graph.nodes[1], ALIAS_D2EPI),
    ]


def test_merged_support_is_validated_before_any_live_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = _collision_graph()
    before = _plain_state(graph)
    original = SelfOrganization._validate_merged_stage_support

    def reject_candidate(self, candidate, proposals):
        original(self, candidate, proposals)
        assert {"1_sub_0", "1_sub_1"} <= set(candidate)
        assert candidate.graph["hierarchy"][1] == ["1_sub_0"]
        raise RuntimeError("rejected detached THOL hierarchy")

    monkeypatch.setattr(
        SelfOrganization, "_validate_merged_stage_support", reject_candidate
    )

    with pytest.raises(RuntimeError, match="rejected detached THOL hierarchy"):
        execute_self_organization_stage(
            graph, SelfOrganization(), ("1", 1), tau=0.1
        )

    assert _plain_state(graph) == before


def test_late_monitor_failure_restores_topology_state_and_monitor() -> None:
    graph = _collision_graph()

    class RejectingMonitor(_RecordingMonitor):
        def after_operator(self, subject, node, operator):
            super().after_operator(subject, node, operator)
            if node == 1:
                subject.graph["monitor_side_effect"] = operator
                subject.nodes[node]["monitor_side_effect"] = True
                raise RuntimeError("rejected second THOL target")

    monitor = RejectingMonitor()
    graph.graph["integrity_monitor"] = monitor
    before = _plain_state(graph)
    monitor_before = deepcopy(vars(monitor))

    with pytest.raises(RuntimeError, match="rejected second THOL target"):
        execute_self_organization_stage(
            graph, SelfOrganization(), ("1", 1), tau=0.1
        )

    assert _plain_state(graph) == before
    assert vars(monitor) == monitor_before


def test_late_metric_failure_restores_complete_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = _collision_graph()
    before = _plain_state(graph)
    original = SelfOrganization._collect_metrics

    def reject_second_metric(self, subject, node, state_before):
        if node == 1:
            raise RuntimeError("rejected second THOL metric")
        return original(self, subject, node, state_before)

    monkeypatch.setattr(SelfOrganization, "_collect_metrics", reject_second_metric)

    with pytest.raises(RuntimeError, match="rejected second THOL metric"):
        execute_self_organization_stage(
            graph,
            SelfOrganization(),
            ("1", 1),
            tau=0.1,
            collect_metrics=True,
        )

    assert _plain_state(graph) == before


def test_late_pressure_refresh_failure_restores_topology_and_metadata() -> None:
    graph = _collision_graph()
    before = _plain_state(graph)
    calls: list[nx.Graph] = []

    def reject_refresh(subject: nx.Graph) -> None:
        calls.append(subject)
        subject.add_node("callback-child", EPI=99.0)
        subject.nodes[1][ALIAS_DNFR[0]] = 99.0
        subject.graph["callback_side_effect"] = True
        raise RuntimeError("rejected THOL pressure refresh")

    with pytest.raises(RuntimeError, match="rejected THOL pressure refresh"):
        execute_self_organization_stage(
            graph,
            SelfOrganization(),
            (1, "1"),
            tau=0.1,
            compute_delta_nfr=reject_refresh,
        )

    assert calls == [graph]
    assert _plain_state(graph) == before


def test_primary_channels_use_alias_and_pressure_cache_boundaries() -> None:
    graph = _collision_graph()
    graph.graph.update(_dnfrmax=0.2, _dnfrmax_node="1")
    graph.nodes[1]["accel"] = -99.0
    graph.nodes["1"]["accel"] = -99.0

    execute_self_organization_stage(
        graph, SelfOrganization(), ("1", 1), tau=0.1
    )

    pressures = {
        node: abs(float(get_attr(graph.nodes[node], ALIAS_DNFR)))
        for node in (1, "1")
    }
    expected_node = max(pressures, key=pressures.get)
    assert graph.graph["_dnfrmax"] == pytest.approx(pressures[expected_node])
    assert graph.graph["_dnfrmax_node"] == expected_node
    assert graph.nodes[1]["accel"] != -99.0
    assert graph.nodes["1"]["accel"] != -99.0


def test_stage_alignment_uses_public_metabolism_readout() -> None:
    graph = nx.Graph(THOL_METABOLIC_ENABLED=False)
    _add_parent(graph, 0, history=[0.0, 0.1, 0.6])
    graph.add_node(
        "old",
        **{
            ALIAS_EPI[0]: 0.1,
            ALIAS_VF[0]: 0.5,
            ALIAS_DNFR[0]: 0.0,
            ALIAS_THETA[0]: 0.1,
            "parent_node": 0,
            "hierarchy_level": 1,
            "_bifurcation_level": 1,
            "_hierarchy_path": [0],
        },
    )
    graph.nodes[0]["sub_nodes"] = ["old"]
    graph.nodes[0]["sub_epis"] = [{"epi": 0.1, "node_id": "old"}]

    execute_self_organization_stage(
        graph, SelfOrganization(), (0,), tau=0.1
    )

    stored = graph.nodes[0]["_thol_subepi_amplitude_alignment"]
    assert stored == compute_subepi_amplitude_alignment(graph, 0)
    values = [record["epi"] for record in graph.nodes[0]["sub_epis"]]
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    assert stored == pytest.approx(1.0 / (1.0 + variance))
    assert graph.graph["hierarchy"][0] == ["old", "0_sub_1"]


def test_conflicting_graph_hierarchy_rejects_stage_before_live_write() -> None:
    graph = nx.Graph(
        THOL_METABOLIC_ENABLED=False,
        hierarchy={0: ["different-child"]},
    )
    _add_parent(graph, 0, history=[0.0, 0.1, 0.6])
    graph.add_node(
        "old",
        **{
            ALIAS_EPI[0]: 0.1,
            ALIAS_VF[0]: 0.5,
            ALIAS_DNFR[0]: 0.0,
            ALIAS_THETA[0]: 0.1,
            "parent_node": 0,
        },
    )
    graph.nodes[0]["sub_nodes"] = ["old"]
    graph.nodes[0]["sub_epis"] = [{"epi": 0.1, "node_id": "old"}]
    before = _plain_state(graph)

    with pytest.raises(OperatorPreconditionError, match="must match parent sub_nodes"):
        execute_self_organization_stage(
            graph, SelfOrganization(), (0,), tau=0.1
        )

    assert _plain_state(graph) == before

def test_noncanonical_override_uses_transactional_gauss_seidel_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = _collision_graph()
    original = SelfOrganization._execute
    calls: list[Any] = []

    def recording_execute(self, subject, node, **kwargs):
        calls.append(node)
        return original(self, subject, node, **kwargs)

    monkeypatch.setattr(SelfOrganization, "_execute", recording_execute)
    result = execute_self_organization_stage(
        graph, SelfOrganization(), ("1", 1), tau=0.1
    )

    assert calls == ["1", 1]
    assert result.schedule == OPERATOR_MAJOR_GAUSS_SEIDEL
    assert graph.graph[STAGE_SCHEDULE_KEY]["schedule"] == (
        OPERATOR_MAJOR_GAUSS_SEIDEL
    )


def test_grammar_replacement_uses_transactional_gauss_seidel_fallback() -> None:
    graph = nx.Graph(THOL_METABOLIC_ENABLED=False)
    _add_parent(
        graph,
        0,
        history=[0.0, 0.1, 0.6],
        glyph_history=["AL"],
    )

    result = execute_self_organization_stage(
        graph, SelfOrganization(), (0,), tau=0.1
    )

    assert result.schedule == OPERATOR_MAJOR_GAUSS_SEIDEL
    assert graph.nodes[0]["glyph_history"][-1] == "IL"
    assert "sub_nodes" not in graph.nodes[0]


def test_shared_word_executor_routes_canonical_thol_to_jacobi_stage() -> None:
    graph = _collision_graph()

    run_network_sequence(
        graph,
        ["self_organization"],
        cycles=1,
        validate=False,
    )

    assert graph.graph[STAGE_SCHEDULE_KEY]["schedule"] == TWO_PHASE_JACOBI
    assert graph.graph[STAGE_CONTRACT_KEY]["schedule_matches_contract"] is True
    assert graph.nodes[1]["sub_nodes"] == ["1_sub_0"]
    assert graph.nodes["1"]["sub_nodes"] == ["1_sub_1"]
