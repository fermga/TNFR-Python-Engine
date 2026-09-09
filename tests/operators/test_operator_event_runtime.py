"""Atomic runtime realization of exact operator-event schedules."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import FrozenInstanceError
from fractions import Fraction
from typing import Any

import networkx as nx
import pytest

import tnfr.operators as operators

from tnfr.dynamics.integrators import AbstractIntegrator
from tnfr.dynamics.runtime import _resolve_integrator_instance
from tnfr.errors import TNFRValueError
from tnfr.operators.event_runtime import (
    ExecutedGlyphStage,
    ExecutedNodalFlowInterval,
    ObservedRepresentedEPIScheduleComposition,
    OperatorEventExecutionResult,
    RepresentedEPIScheduleOperation,
    execute_operator_event_schedule,
)
from tnfr.operators.event_timing import build_operator_event_schedule
from tnfr.operators.network_stage import (
    OPERATOR_MAJOR_GAUSS_SEIDEL,
    TWO_PHASE_JACOBI,
    GraphTransactionSnapshot,
    NetworkStageResult,
)


_WORD = ("emission", "coupling", "coherence", "silence")


def _graph() -> nx.Graph:
    graph = nx.path_graph(2)
    graph.graph.update(
        _t=0.0,
        RANDOM_SEED=7,
        _gamma_spec={"type": "none"},
        EPI_MIN=-4.0,
        EPI_MAX=4.0,
    )
    for node in graph:
        graph.nodes[node].update(
            EPI=0.0,
            nu_f=1.0,
            theta=0.0,
            delta_nfr=0.2,
            glyph_history=[],
        )
    return graph


def _schedule(
    durations: tuple[float, ...] = (0.25, 0.0, 0.0, 0.0, 0.5),
):
    return build_operator_event_schedule(
        _WORD,
        start_time=0.0,
        flow_durations=durations,
    )


def _plain_state(graph: nx.Graph) -> tuple[Any, ...]:
    graph_data = {
        key: deepcopy(value)
        for key, value in graph.graph.items()
        if key not in {"integrator", "_integrator_cache"}
    }
    return (
        tuple(
            (node, deepcopy(dict(data)))
            for node, data in graph.nodes(data=True)
        ),
        tuple(
            (left, right, deepcopy(dict(data)))
            for left, right, data in graph.edges(data=True)
        ),
        graph_data,
        (
            hasattr(graph, "_last_operator_applied"),
            getattr(graph, "_last_operator_applied", None),
        ),
    )


class CountingIntegrator(AbstractIntegrator):
    """Minimal deterministic integrator with observable mutable state."""

    def __init__(self, *, wrong_endpoint: bool = False) -> None:
        self.calls = 0
        self.wrong_endpoint = wrong_endpoint

    def integrate(
        self,
        graph,
        *,
        dt,
        t,
        method,
        n_jobs,
    ) -> None:
        self.calls += 1
        graph.nodes[0]["EPI"] += 1.0
        multiplier = 2.0 if self.wrong_endpoint else 1.0
        graph.graph["_t"] = float(t) + multiplier * float(dt)


def test_schedule_executes_declared_flows_and_zero_duration_jumps() -> None:
    graph = _graph()

    result = execute_operator_event_schedule(graph, _schedule())

    assert isinstance(result, OperatorEventExecutionResult)
    assert result.target_nodes == (0, 1)
    assert result.flow_interval_indices == (0, 1, 2, 3, 4)
    assert result.positive_flow_interval_indices == (0, 4)
    assert result.final_time == 0.75
    assert result.integrator_name == "DefaultIntegrator"
    assert result.pressure_refresh_callback_invocations == 0
    assert result.flow_provenance.endswith("execute_operator_event_schedule")
    assert result.nodal_flow_inputs == (
        "live_nu_f_and_delta_nfr_at_each_interval_start"
    )
    assert result.runtime_clock_checked
    assert result.whole_schedule_graph_state_atomic
    assert result.operator_jumps_have_zero_duration
    assert not result.solver_accuracy_certified
    assert not result.adaptive_u2_u4_policy
    assert not result.external_side_effects_rolled_back
    assert graph.graph["_t"] == 0.75

    events = graph.graph["hybrid_event_log"]
    assert [event["event_index"] for event in events] == [0, 1, 2, 3]
    assert [event["operator_name"] for event in events] == list(_WORD)
    assert [event["event_time"] for event in events] == [
        0.25,
        0.25,
        0.25,
        0.25,
    ]
    assert all(event["stage_schedule"] == TWO_PHASE_JACOBI for event in events)
    assert all(event["zero_duration"] for event in events)
    assert all(not event["feeds_epi_time_history"] for event in events)
    assert events[0]["event_offset"] == Fraction(1, 4)

    for node in graph:
        history = list(graph.nodes[node]["epi_time_history"])
        assert [sample[0] for sample in history] == [0.25, 0.75]
        assert history[0][1] > 0.05
        assert history[1][1] >= history[0][1]

    with pytest.raises(FrozenInstanceError):
        result.final_time = 1.0  # type: ignore[misc]


def test_pure_flow_schedule_uses_current_pressure_without_event_log() -> None:
    graph = _graph()
    refresh_times: list[float] = []

    def refresh(live_graph: nx.Graph) -> None:
        refresh_times.append(live_graph.graph["_t"])

    graph.graph["compute_delta_nfr"] = refresh
    schedule = build_operator_event_schedule(
        (),
        start_time=0.0,
        flow_durations=(0.25,),
    )

    result = execute_operator_event_schedule(graph, schedule)

    assert result.events == ()
    assert result.target_nodes == (0, 1)
    assert result.positive_flow_interval_indices == (0,)
    assert result.pressure_refresh_callback_invocations == 0
    assert refresh_times == []
    assert graph.graph["_t"] == 0.25
    assert graph.nodes[0]["EPI"] == pytest.approx(0.05)
    assert "hybrid_event_log" not in graph.graph


def test_result_counts_completed_pressure_refresh_callbacks() -> None:
    graph = _graph()
    refresh_times: list[float] = []

    def refresh(live_graph: nx.Graph) -> None:
        refresh_times.append(live_graph.graph["_t"])

    graph.graph["compute_delta_nfr"] = refresh

    result = execute_operator_event_schedule(graph, _schedule())

    assert result.pressure_refresh_callback_invocations == len(_WORD)
    assert refresh_times == [0.25] * len(_WORD)


@pytest.mark.parametrize("precache", [False, True])
def test_late_failure_restores_integrator_identity_state_and_cache(
    precache: bool,
) -> None:
    graph = _graph()
    integrator = CountingIntegrator()
    graph.graph["integrator"] = integrator
    if precache:
        assert _resolve_integrator_instance(graph) is integrator
        cache_before = graph.graph["_integrator_cache"]
    else:
        cache_before = None

    def fail_pressure_refresh(live_graph) -> None:
        live_graph.graph["refresh_started"] = True
        raise RuntimeError("late pressure refresh failure")

    graph.graph["compute_delta_nfr"] = fail_pressure_refresh
    before = _plain_state(graph)

    with pytest.raises(RuntimeError, match="late pressure refresh failure"):
        execute_operator_event_schedule(
            graph,
            _schedule((0.25, 0.0, 0.0, 0.0, 0.0)),
        )

    assert _plain_state(graph) == before
    assert graph.graph["integrator"] is integrator
    assert integrator.calls == 0
    if precache:
        assert graph.graph["_integrator_cache"] is cache_before
        assert graph.graph["_integrator_cache"][1] is integrator
    else:
        assert "_integrator_cache" not in graph.graph


def test_wrong_integrator_endpoint_rolls_back_the_complete_flow() -> None:
    graph = _graph()
    integrator = CountingIntegrator(wrong_endpoint=True)
    graph.graph["integrator"] = integrator
    schedule = build_operator_event_schedule(
        (),
        start_time=0.0,
        flow_durations=(0.25,),
    )
    before = _plain_state(graph)

    with pytest.raises(TNFRValueError, match="scheduled boundary"):
        execute_operator_event_schedule(graph, schedule)

    assert _plain_state(graph) == before
    assert graph.graph["integrator"] is integrator
    assert integrator.calls == 0
    assert "_integrator_cache" not in graph.graph


def test_resized_runtime_array_rolls_back_without_masking_clock_error() -> None:
    np = pytest.importorskip("numpy")
    graph = _graph()
    graph.graph["ordinary_marker"] = "before"
    runtime_cache = np.array([1.0, 2.0])

    class ResizingWrongEndpointIntegrator(AbstractIntegrator):
        def __init__(self) -> None:
            self.calls = 0

        def integrate(
            self,
            live_graph,
            *,
            dt,
            t,
            method,
            n_jobs,
        ) -> None:
            self.calls += 1
            cache = live_graph.graph["_test_cache"]
            cache.resize((3,), refcheck=False)
            cache[:] = 9.0
            live_graph.graph["ordinary_marker"] = "after"
            live_graph.nodes[0]["EPI"] += 10.0
            live_graph.graph["_t"] = float(t) + 2.0 * float(dt)

    integrator = ResizingWrongEndpointIntegrator()
    graph.graph["integrator"] = integrator
    graph.graph["_test_cache"] = runtime_cache
    schedule = build_operator_event_schedule(
        (),
        start_time=0.0,
        flow_durations=(0.25,),
    )

    with pytest.raises(TNFRValueError, match="scheduled boundary"):
        execute_operator_event_schedule(graph, schedule)

    assert graph.graph["_test_cache"] is runtime_cache
    assert runtime_cache.shape == (2,)
    assert np.array_equal(runtime_cache, np.array([1.0, 2.0]))
    assert graph.graph["ordinary_marker"] == "before"
    assert graph.nodes[0]["EPI"] == 0.0
    assert graph.graph["_t"] == 0.0
    assert integrator.calls == 0
    assert "_integrator_cache" not in graph.graph
    assert "hybrid_event_log" not in graph.graph


def test_rollback_failure_is_attached_without_replacing_primary_error(
    monkeypatch,
) -> None:
    graph = _graph()
    integrator = CountingIntegrator(wrong_endpoint=True)
    graph.graph["integrator"] = integrator
    schedule = build_operator_event_schedule(
        (),
        start_time=0.0,
        flow_durations=(0.25,),
    )
    restore = GraphTransactionSnapshot.restore

    def restore_then_report_failure(
        snapshot: GraphTransactionSnapshot,
        live_graph: nx.Graph,
    ) -> None:
        restore(snapshot, live_graph)
        raise RuntimeError("secondary rollback failure")

    monkeypatch.setattr(
        GraphTransactionSnapshot,
        "restore",
        restore_then_report_failure,
    )

    with pytest.raises(TNFRValueError, match="scheduled boundary") as error:
        execute_operator_event_schedule(graph, schedule)

    rollback_failure = getattr(
        error.value,
        "_tnfr_rollback_failure",
        None,
    )
    assert isinstance(rollback_failure, RuntimeError)
    assert str(rollback_failure) == "secondary rollback failure"
    assert graph.graph["_t"] == 0.0
    assert graph.nodes[0]["EPI"] == 0.0
    assert integrator.calls == 0


def test_fixed_initial_targets_survive_support_growth(monkeypatch) -> None:
    graph = _graph()
    observed_targets: list[tuple[Any, ...]] = []

    def grow_support(
        live_graph,
        operator,
        targets,
        *,
        sequence_context,
        compute_delta_nfr,
    ):
        observed_targets.append(tuple(targets))
        if len(observed_targets) == 1:
            live_graph.add_node(
                "late",
                EPI=0.0,
                nu_f=0.0,
                theta=0.0,
                delta_nfr=0.0,
            )
        return NetworkStageResult(
            operator=operator.name,
            glyph=operator.glyph.value,
            schedule=TWO_PHASE_JACOBI,
            nodes_processed=len(tuple(targets)),
        )

    monkeypatch.setattr(
        "tnfr.operators.word_execution.execute_network_operator_stage",
        grow_support,
    )

    result = execute_operator_event_schedule(
        graph,
        _schedule((0.0, 0.0, 0.0, 0.0, 0.0)),
    )

    assert observed_targets == [(0, 1)] * 4
    assert result.target_nodes == (0, 1)
    assert "late" in graph


def test_nonexact_stage_result_rolls_back_instead_of_logging_request(
    monkeypatch,
) -> None:
    graph = _graph()
    before = _plain_state(graph)

    def replace_stage(
        live_graph,
        operator,
        targets,
        *,
        sequence_context,
        compute_delta_nfr,
    ):
        live_graph.nodes[0]["EPI"] = 9.0
        return NetworkStageResult(
            operator=operator.name,
            glyph=operator.glyph.value,
            schedule=OPERATOR_MAJOR_GAUSS_SEIDEL,
            nodes_processed=len(tuple(targets)),
        )

    monkeypatch.setattr(
        "tnfr.operators.word_execution.execute_network_operator_stage",
        replace_stage,
    )

    with pytest.raises(TNFRValueError, match="not realized"):
        execute_operator_event_schedule(
            graph,
            _schedule((0.0, 0.0, 0.0, 0.0, 0.0)),
        )

    assert _plain_state(graph) == before
    assert "hybrid_event_log" not in graph.graph


def test_invalid_word_is_rejected_before_its_initial_flow() -> None:
    graph = _graph()
    schedule = build_operator_event_schedule(
        ("reception",),
        start_time=0.0,
        flow_durations=(0.25, 0.0),
    )
    before = _plain_state(graph)

    with pytest.raises(TNFRValueError, match="Invalid operator-event sequence"):
        execute_operator_event_schedule(graph, schedule)

    assert _plain_state(graph) == before
    assert graph.graph["_t"] == 0.0
    assert all("epi_time_history" not in graph.nodes[node] for node in graph)


@pytest.mark.parametrize("runtime_time", [None, 1.0, 0, True])
def test_live_binary64_start_clock_is_required_before_writes(
    runtime_time,
) -> None:
    graph = _graph()
    if runtime_time is None:
        del graph.graph["_t"]
    else:
        graph.graph["_t"] = runtime_time
    before = _plain_state(graph)

    with pytest.raises(TNFRValueError, match="runtime clock|scheduled boundary"):
        execute_operator_event_schedule(graph, _schedule())

    assert _plain_state(graph) == before


def test_collapsed_positive_interval_is_rejected_before_writes() -> None:
    graph = _graph()
    graph.graph["_t"] = float(2**53)
    schedule = build_operator_event_schedule(
        ("emission",),
        start_time=float(2**53),
        flow_durations=(0.5, 0.0),
    )
    before = _plain_state(graph)

    with pytest.raises(TNFRValueError, match="cannot bind"):
        execute_operator_event_schedule(graph, schedule)

    assert _plain_state(graph) == before


def test_zhir_duration_mismatch_is_rejected_before_writes() -> None:
    graph = _graph()
    graph.graph["_t"] = float(2**52)
    schedule = build_operator_event_schedule(
        ("mutation",),
        start_time=float(2**52),
        flow_durations=(0.6, 0.0),
    )
    before = _plain_state(graph)

    with pytest.raises(TNFRValueError, match="cannot bind") as error:
        execute_operator_event_schedule(graph, schedule)

    assert error.value.context[
        "zhir_event_indices_with_duration_mismatch"
    ] == (0,)
    assert _plain_state(graph) == before
    assert graph.graph["_t"] == float(2**52)
    assert "hybrid_event_log" not in graph.graph
    assert all("epi_time_history" not in graph.nodes[node] for node in graph)


def test_nonadditive_positive_interval_is_rejected_before_writes() -> None:
    graph = _graph()
    schedule = build_operator_event_schedule(
        (
            "emission",
            "coherence",
            "dissonance",
            "mutation",
            "coherence",
            "silence",
        ),
        start_time=0.0,
        flow_durations=(0.1,) * 7,
    )
    before = _plain_state(graph)

    with pytest.raises(TNFRValueError, match="cannot bind") as error:
        execute_operator_event_schedule(graph, schedule)

    assert error.value.context[
        "nonadditive_positive_interval_indices"
    ] == (5,)
    assert _plain_state(graph) == before
    assert graph.graph["_t"] == 0.0
    assert "hybrid_event_log" not in graph.graph


def test_nonlist_hybrid_event_sink_is_rejected_before_writes() -> None:
    graph = _graph()
    graph.graph["hybrid_event_log"] = ()
    before = _plain_state(graph)

    with pytest.raises(TNFRValueError, match="must be a list"):
        execute_operator_event_schedule(graph, _schedule())

    assert _plain_state(graph) == before

def test_runtime_schedule_executor_is_public() -> None:
    assert (
        operators.execute_operator_event_schedule
        is execute_operator_event_schedule
    )
    assert operators.OperatorEventExecutionResult is OperatorEventExecutionResult
    assert operators.ExecutedNodalFlowInterval is ExecutedNodalFlowInterval
    assert operators.ExecutedGlyphStage is ExecutedGlyphStage
    assert (
        operators.ObservedRepresentedEPIScheduleComposition
        is ObservedRepresentedEPIScheduleComposition
    )
    assert (
        operators.RepresentedEPIScheduleOperation
        is RepresentedEPIScheduleOperation
    )


@pytest.mark.parametrize(
    ("keyword", "value"),
    [
        ("context", []),
        ("method", 1),
        ("n_jobs", True),
        ("suppress_birth_warnings", 1),
    ],
)
def test_execution_controls_are_strict_before_writes(
    keyword: str,
    value: Any,
) -> None:
    graph = _graph()
    before = _plain_state(graph)

    with pytest.raises(TypeError):
        execute_operator_event_schedule(
            graph,
            _schedule(),
            **{keyword: value},
        )

    assert _plain_state(graph) == before

def test_positive_preflow_supplies_fresh_timestamped_zhir_evidence() -> None:
    graph = _graph()
    graph.graph["ZHIR_THRESHOLD_XI"] = 0.05
    for node in graph:
        graph.nodes[node]["epi_history"] = [-0.2, -0.1, 0.0]
        graph.nodes[node]["epi_time_history"] = [
            (-1.0, 0.0),
            (0.0, 0.0),
        ]
    names = (
        "emission",
        "coherence",
        "dissonance",
        "mutation",
        "coherence",
        "silence",
    )
    schedule = build_operator_event_schedule(
        names,
        start_time=0.0,
        flow_durations=(0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0),
    )

    result = execute_operator_event_schedule(graph, schedule)

    mutation_event = result.events[3]
    assert mutation_event.operator_name == "mutation"
    assert mutation_event.event_time == 0.5
    assert graph.graph["hybrid_event_log"][3]["operator_name"] == "mutation"
    for node in graph:
        data = graph.nodes[node]
        history = list(data["epi_time_history"])
        assert [sample[0] for sample in history] == [0.0, 0.5]
        assert data["_zhir_gate_history_key"] == "epi_time_history"
        assert data["_zhir_gate_sample_interval"] == 0.5
        assert data["_zhir_gate_depi_dt"] > data["_zhir_gate_xi"]
        assert list(data["glyph_history"]) == [
            "AL",
            "IL",
            "OZ",
            "ZHIR",
            "IL",
            "SHA",
        ]


def test_zero_zhir_preflow_is_rejected_before_any_operator_jump() -> None:
    graph = _graph()
    names = (
        "emission",
        "coherence",
        "dissonance",
        "mutation",
        "coherence",
        "silence",
    )
    schedule = build_operator_event_schedule(
        names,
        start_time=0.0,
        flow_durations=(0.0,) * 7,
    )
    before = _plain_state(graph)

    with pytest.raises(TNFRValueError, match="cannot bind"):
        execute_operator_event_schedule(graph, schedule)

    assert _plain_state(graph) == before
    assert "hybrid_event_log" not in graph.graph
