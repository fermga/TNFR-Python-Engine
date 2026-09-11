"""Failure atomicity for the established operator-major network schedule."""

from __future__ import annotations

from copy import deepcopy
import threading

import networkx as nx
import pytest

from tnfr.errors import TNFRValueError
from tnfr.operators.definitions import Coherence, Dissonance, Emission
from tnfr.operators.network_stage import (
    OPERATOR_MAJOR_GAUSS_SEIDEL,
    STAGE_CONTRACT_KEY,
    STAGE_SCHEDULE_KEY,
    execute_operator_major_stage,
)
from tnfr.operators.stage_contracts import stage_schedule_metadata
from tnfr.operators.word_execution import run_network_sequence


def _graph() -> nx.Graph:
    graph = nx.path_graph(3)
    for node in graph:
        graph.nodes[node].update(
            EPI=0.1 * (node + 1),
            nu_f=1.0,
            phase=0.0,
            DeltaNFR=0.0,
            glyph_history=[],
        )
    graph.graph["RANDOM_SEED"] = 7
    return graph


def _plain_state(graph: nx.Graph) -> tuple[object, ...]:
    return (
        tuple((node, deepcopy(dict(data))) for node, data in graph.nodes(data=True)),
        tuple(
            (left, right, deepcopy(dict(data)))
            for left, right, data in graph.edges(data=True)
        ),
        deepcopy(dict(graph.graph)),
        getattr(graph, "_last_operator_applied", None),
    )


def test_operator_major_stage_rolls_back_a_late_target_failure(monkeypatch) -> None:
    graph = _graph()
    before = _plain_state(graph)

    def fail_on_second_target(self, live_graph, node, **kwargs):
        live_graph.nodes[node]["EPI"] += 10.0
        live_graph.graph.setdefault("stage_events", []).append(node)
        if node == 1:
            raise RuntimeError("late target failure")

    monkeypatch.setattr(Dissonance, "__call__", fail_on_second_target)

    with pytest.raises(RuntimeError, match="late target failure"):
        run_network_sequence(graph, ["dissonance"], validate=False)

    assert _plain_state(graph) == before


def test_operator_major_stage_rolls_back_pressure_refresh_failure(
    monkeypatch,
) -> None:
    graph = _graph()
    before = _plain_state(graph)

    def local_write(self, live_graph, node, **kwargs):
        live_graph.nodes[node]["EPI"] += 1.0

    def failed_refresh(live_graph):
        live_graph.graph["refresh_started"] = True
        raise RuntimeError("pressure refresh failure")

    monkeypatch.setattr(Dissonance, "__call__", local_write)
    graph.graph["compute_delta_nfr"] = failed_refresh
    before = _plain_state(graph)

    with pytest.raises(RuntimeError, match="pressure refresh failure"):
        run_network_sequence(graph, ["dissonance"], validate=False)

    assert _plain_state(graph) == before


def test_operator_major_stage_retains_sequential_reads_and_labels_schedule(
    monkeypatch,
) -> None:
    graph = _graph()

    def sequential_write(self, live_graph, node, **kwargs):
        predecessor = node - 1
        if predecessor in live_graph:
            live_graph.nodes[node]["EPI"] = live_graph.nodes[predecessor]["EPI"] + 1.0
        else:
            live_graph.nodes[node]["EPI"] = 1.0

    monkeypatch.setattr(Emission, "__call__", sequential_write)

    result = execute_operator_major_stage(graph, Emission(), tuple(graph))

    assert [graph.nodes[node]["EPI"] for node in graph] == [1.0, 2.0, 3.0]
    assert result.schedule == OPERATOR_MAJOR_GAUSS_SEIDEL
    expected_schedule = {
        "operator": "emission",
        "glyph": "AL",
        "schedule": OPERATOR_MAJOR_GAUSS_SEIDEL,
        "nodes_processed": 3,
    }
    assert graph.graph[STAGE_SCHEDULE_KEY] == expected_schedule
    assert graph.graph[STAGE_CONTRACT_KEY] == stage_schedule_metadata(
        "AL",
        observed_schedule=OPERATOR_MAJOR_GAUSS_SEIDEL,
    )


def test_operator_major_stage_rejects_duplicate_targets_before_writes(
    monkeypatch,
) -> None:
    graph = _graph()
    before = _plain_state(graph)
    calls: list[int] = []

    def record_call(self, live_graph, node, **kwargs):
        calls.append(node)

    monkeypatch.setattr(Emission, "__call__", record_call)

    with pytest.raises(TNFRValueError, match="targets must be unique"):
        execute_operator_major_stage(graph, Emission(), [0, 0])

    assert calls == []
    assert _plain_state(graph) == before


def test_operator_major_stage_preserves_unrelated_lock_metadata() -> None:
    graph = _graph()
    user_lock = threading.Lock()
    graph.graph["user_lock"] = user_lock

    run_network_sequence(graph, ["coherence"], validate=False)

    assert graph.graph["user_lock"] is user_lock


def test_operator_major_stage_rolls_back_with_unrelated_lock_metadata(
    monkeypatch,
) -> None:
    graph = _graph()
    user_lock = threading.Lock()
    graph.graph["user_lock"] = user_lock
    epi_before = tuple(graph.nodes[node]["EPI"] for node in graph)

    def fail_on_second_target(self, live_graph, node, **kwargs):
        live_graph.nodes[node]["EPI"] += 1.0
        if node == 1:
            raise RuntimeError("late target failure with lock")

    monkeypatch.setattr(Dissonance, "__call__", fail_on_second_target)

    with pytest.raises(RuntimeError, match="failure with lock"):
        run_network_sequence(graph, ["dissonance"], validate=False)

    assert tuple(graph.nodes[node]["EPI"] for node in graph) == epi_before
    assert graph.graph["user_lock"] is user_lock
