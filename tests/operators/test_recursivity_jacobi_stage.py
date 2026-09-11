"""Atomic immutable all-target execution for Recursivity advisories."""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import FrozenInstanceError

import networkx as nx
import pytest

from tnfr.operators import remesh
from tnfr.operators._recursivity_stage_kernel import (
    RecursivityAdvisoryProposal,
    propose_recursivity_advisory,
)
from tnfr.operators.definitions import Recursivity
from tnfr.operators.network_stage import (
    OPERATOR_MAJOR_GAUSS_SEIDEL,
    STAGE_CONTRACT_KEY,
    STAGE_SCHEDULE_KEY,
    TWO_PHASE_JACOBI,
    execute_recursivity_stage,
)
from tnfr.operators.word_execution import run_network_sequence
from tnfr.types import Glyph


def _graph() -> nx.Graph:
    graph = nx.path_graph(3)
    for node in graph:
        graph.nodes[node].update(
            EPI=0.2 + node,
            nu_f=1.0,
            theta=0.1 * node,
            DeltaNFR=0.4,
            glyph_history=[],
        )
    return graph


def _structural_state(graph: nx.Graph) -> tuple[object, ...]:
    nodes = tuple(
        (
            node,
            graph.nodes[node]["EPI"],
            graph.nodes[node]["nu_f"],
            graph.nodes[node]["theta"],
            graph.nodes[node]["DeltaNFR"],
        )
        for node in graph
    )
    return nodes, tuple(graph.edges(data=True))


def _events(graph: nx.Graph) -> list[object]:
    return list(graph.graph["history"]["events"])


def test_advisory_proposal_is_frozen_and_snapshot_bound() -> None:
    graph = _graph()
    graph.graph["history"] = {"C_steps": [0.7, 0.8]}

    proposal = propose_recursivity_advisory(graph)

    assert proposal == RecursivityAdvisoryProposal(
        step=2,
        had_warning_step=False,
        warning_step_before=None,
    )
    assert proposal.emits_advisory is True
    with pytest.raises(FrozenInstanceError):
        proposal.step = 3  # type: ignore[misc]


def test_stage_is_target_order_invariant_and_deduplicates_advisory() -> None:
    forward = _graph()
    reverse = _graph()
    structural_before = _structural_state(forward)

    execute_recursivity_stage(forward, Recursivity(), (0, 1, 2))
    execute_recursivity_stage(reverse, Recursivity(), (2, 1, 0))

    assert _structural_state(forward) == structural_before
    assert _structural_state(reverse) == structural_before
    assert _events(forward) == _events(reverse)
    assert len(_events(forward)) == 1
    assert _events(forward)[0][0] == "warn"
    for graph in (forward, reverse):
        assert [list(graph.nodes[node]["glyph_history"]) for node in graph] == [
            ["REMESH"],
            ["REMESH"],
            ["REMESH"],
        ]
        assert [
            graph.nodes[node]["source_glyph"] for node in graph
        ] == ["REMESH", "REMESH", "REMESH"]
        assert graph.graph[STAGE_SCHEDULE_KEY]["schedule"] == TWO_PHASE_JACOBI
        assert graph.graph[STAGE_CONTRACT_KEY][
            "executed_two_phase_contract_complete"
        ] is True


def test_repeated_stage_emits_once_per_telemetry_step() -> None:
    graph = _graph()
    graph.graph["history"] = {"C_steps": [0.9]}

    execute_recursivity_stage(graph, Recursivity(), tuple(graph))
    execute_recursivity_stage(graph, Recursivity(), tuple(graph))

    assert len(_events(graph)) == 1
    graph.graph["history"]["C_steps"].append(0.95)
    execute_recursivity_stage(graph, Recursivity(), tuple(graph))

    assert len(_events(graph)) == 2
    assert [event[1]["step"] for event in _events(graph)] == [1, 2]


def test_word_stage_never_invokes_explicit_delayed_epi_remesh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = _graph()
    graph.graph.update(
        REMESH_TAU_GLOBAL=1,
        REMESH_TAU_LOCAL=1,
        REMESH_ALPHA=0.5,
        REMESH_ALPHA_HARD=True,
        REMESH_LOG_EVENTS=False,
    )
    graph.graph["_epi_hist"] = deque(
        [
            {node: 0.0 for node in graph},
            {node: graph.nodes[node]["EPI"] for node in graph},
        ],
        maxlen=8,
    )
    epi_before = tuple(graph.nodes[node]["EPI"] for node in graph)

    def forbidden_explicit_remesh(_graph: nx.Graph) -> None:
        raise AssertionError("word execution invoked apply_network_remesh")

    monkeypatch.setattr(
        remesh,
        "apply_network_remesh",
        forbidden_explicit_remesh,
    )

    run_network_sequence(graph, ["recursivity"], validate=False)

    assert tuple(graph.nodes[node]["EPI"] for node in graph) == epi_before
    assert len(_events(graph)) == 1


def test_metrics_and_monitor_lifecycle_commit_in_target_order() -> None:
    graph = _graph()
    graph.graph["COLLECT_OPERATOR_METRICS"] = True

    class Monitor:
        def __init__(self) -> None:
            self.before: list[int] = []
            self.after: list[tuple[int, str]] = []

        def before_operator(self, _graph: nx.Graph, node: int) -> None:
            self.before.append(node)

        def after_operator(
            self, _graph: nx.Graph, node: int, operator: str
        ) -> None:
            self.after.append((node, operator))

    monitor = Monitor()
    graph.graph["integrity_monitor"] = monitor

    execute_recursivity_stage(graph, Recursivity(), (2, 0, 1))

    assert monitor.before == [2, 0, 1]
    assert monitor.after == [
        (2, "recursivity"),
        (0, "recursivity"),
        (1, "recursivity"),
    ]
    metrics = graph.graph["operator_metrics"]
    assert [metric["operator"] for metric in metrics] == [
        "Recursivity",
        "Recursivity",
        "Recursivity",
    ]
    assert all(metric["delta_epi"] == 0.0 for metric in metrics)


def test_late_pressure_refresh_failure_restores_complete_stage() -> None:
    graph = _graph()
    nodes_before = deepcopy(dict(graph.nodes(data=True)))
    graph_before = deepcopy(dict(graph.graph))

    def failed_refresh(live_graph: nx.Graph) -> None:
        live_graph.graph["partial_refresh"] = True
        live_graph.nodes[0]["EPI"] = 99.0
        raise RuntimeError("REMESH pressure refresh failed")

    graph.graph["compute_delta_nfr"] = failed_refresh

    with pytest.raises(RuntimeError, match="pressure refresh failed"):
        run_network_sequence(graph, ["recursivity"], validate=False)

    assert dict(graph.nodes(data=True)) == nodes_before
    assert {
        key: value
        for key, value in graph.graph.items()
        if key != "compute_delta_nfr"
    } == graph_before
    assert graph.graph["compute_delta_nfr"] is failed_refresh


def test_invalid_advisory_sink_rejects_before_node_lifecycle() -> None:
    graph = _graph()
    graph.graph["history"] = {"events": 7}
    nodes_before = deepcopy(dict(graph.nodes(data=True)))

    with pytest.raises(AttributeError):
        execute_recursivity_stage(graph, Recursivity(), tuple(graph))

    assert dict(graph.nodes(data=True)) == nodes_before
    assert graph.graph["history"] == {"events": 7}
    assert STAGE_SCHEDULE_KEY not in graph.graph


def test_overridden_recursivity_execution_uses_transactional_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = _graph()

    def overridden(self, live_graph, node, **kwargs):
        live_graph.graph.setdefault("ordered_override", []).append(node)
        live_graph.nodes[node]["EPI"] += len(
            live_graph.graph["ordered_override"]
        )

    monkeypatch.setattr(Recursivity, "_execute", overridden)

    result = execute_recursivity_stage(
        graph,
        Recursivity(),
        (2, 0, 1),
    )

    assert result.schedule == OPERATOR_MAJOR_GAUSS_SEIDEL
    assert graph.graph["ordered_override"] == [2, 0, 1]
    assert graph.graph[STAGE_CONTRACT_KEY][
        "executed_two_phase_contract_complete"
    ] is False


def test_grammar_replacement_uses_transactional_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = _graph()
    from tnfr.operators import grammar_application

    def select_coherence(_graph, _node, glyph, _ctx=None):
        return Glyph.IL if glyph is Glyph.REMESH else glyph

    monkeypatch.setattr(
        grammar_application,
        "enforce_canonical_grammar",
        select_coherence,
    )

    result = execute_recursivity_stage(
        graph,
        Recursivity(),
        tuple(graph),
    )

    assert result.schedule == OPERATOR_MAJOR_GAUSS_SEIDEL
    assert all(
        list(graph.nodes[node]["glyph_history"]) == ["IL"]
        for node in graph
    )
    assert graph.graph[STAGE_CONTRACT_KEY][
        "schedule_matches_contract"
    ] is False