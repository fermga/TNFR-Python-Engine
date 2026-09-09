"""Atomic event-schedule and delayed-REMESH cycle contract."""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import replace
from fractions import Fraction
import sys
from typing import Any

import networkx as nx
import pytest

import tnfr.operators.event_remesh_runtime as event_remesh_runtime
from tnfr.dynamics.integrators import AbstractIntegrator
from tnfr.errors import TNFRValueError
from tnfr.operators import (
    EventRemeshCycleResult,
    RemeshHistoryTransitionObservation,
    build_operator_event_schedule,
    execute_event_remesh_cycle,
)
from tnfr.utils import CallbackEvent, callback_manager


def _graph(
    *,
    current: tuple[float, float] = (2.0, 0.0),
    past: tuple[float, float] | None = (0.0, 2.0),
) -> nx.Graph:
    graph = nx.path_graph(2)
    graph.graph.update(
        _t=0.0,
        RANDOM_SEED=19,
        _gamma_spec={"type": "none"},
        REMESH_TAU_GLOBAL=1,
        REMESH_TAU_LOCAL=1,
        REMESH_ALPHA=0.5,
        REMESH_ALPHA_HARD=True,
        REMESH_LOG_EVENTS=False,
        EPI_MIN=-10.0,
        EPI_MAX=10.0,
        CLIP_MODE="hard",
    )
    for node, epi in enumerate(current):
        graph.nodes[node].update(
            EPI=epi,
            nu_f=1.0,
            theta=0.0,
            delta_nfr=0.0,
            glyph_history=[],
        )
    history = [] if past is None else [
        {node: value for node, value in enumerate(past)}
    ]
    graph.graph["_epi_hist"] = deque(history, maxlen=64)
    return graph


def _schedule(
    graph: nx.Graph,
    *,
    operators: tuple[str, ...] = (),
    durations: tuple[float, ...] = (0.0,),
):
    return build_operator_event_schedule(
        operators,
        start_time=graph.graph["_t"],
        flow_durations=durations,
    )


def _state(graph: nx.Graph) -> tuple[Any, ...]:
    graph_data = {
        key: deepcopy(value)
        for key, value in graph.graph.items()
        if key not in {"integrator", "_integrator_cache"}
    }
    return (
        tuple((node, deepcopy(dict(data))) for node, data in graph.nodes(data=True)),
        tuple(
            (left, right, deepcopy(dict(data)))
            for left, right, data in graph.edges(data=True)
        ),
        graph_data,
    )


def test_cycle_preserves_pre_jump_history_index_and_one_metric() -> None:
    graph = _graph()

    result = execute_event_remesh_cycle(
        graph,
        _schedule(graph),
        metric_weights={0: 1.0, 1: 3.0},
    )

    assert isinstance(result, EventRemeshCycleResult)
    assert result.metric_weights == (1.0, 3.0)
    assert result.remesh_applied
    assert result.remesh.evidence is not None
    assert result.remesh.evidence.metric_weights == result.metric_weights
    assert result.pre_schedule_epi.epi_values == (2.0, 0.0)
    assert result.pre_remesh_epi.epi_values == (2.0, 0.0)
    assert result.post_remesh_epi.epi_values == (0.5, 1.5)
    assert result.exact_schedule_weighted_mean_drift == 0
    assert result.exact_remesh_weighted_mean_drift == Fraction(3, 4)
    assert result.exact_total_weighted_mean_drift == Fraction(3, 4)
    assert result.history_length_before_cycle == 1
    assert result.history_length_before_append == 1
    assert result.history_length_after_append == 2
    assert not result.history_container_rebuilt
    assert not result.history_oldest_snapshot_evicted
    assert list(graph.graph["_epi_hist"]) == [
        {0: 0.0, 1: 2.0},
        {0: 2.0, 1: 0.0},
    ]
    assert result.history_convention.endswith(
        "history[-(tau+1)]"
    )
    assert result.schedule_left_history_unchanged
    assert result.whole_cycle_graph_state_atomic
    assert result.common_metric_is_frozen
    assert not result.remesh_history_repetition_certified
    assert not result.mixed_runtime_gain_certified
    assert not result.external_side_effects_rolled_back


def test_uniform_metric_is_materialized_for_insufficient_history() -> None:
    graph = _graph(past=None)
    calls: list[str] = []

    def refresh(_: nx.Graph) -> None:
        calls.append("refresh")

    graph.graph["compute_delta_nfr"] = refresh
    result = execute_event_remesh_cycle(
        graph,
        _schedule(graph),
        refresh_pressure_after_remesh=True,
    )

    assert result.metric_weights == (1.0, 1.0)
    assert result.remesh.status == "insufficient_history"
    assert not result.remesh_applied
    assert result.history_length_after_append == 1
    assert list(graph.graph["_epi_hist"]) == [{0: 2.0, 1: 0.0}]
    assert result.post_remesh_pressure_refresh_requested
    assert result.post_remesh_pressure_refresh_callback_invocations == 0
    assert not result.post_remesh_pressure_refresh_performed
    assert calls == []


def test_post_remesh_pressure_refresh_is_explicit_and_counted_after_return() -> None:
    graph = _graph()
    calls: list[tuple[float, float]] = []

    def refresh(live_graph: nx.Graph) -> None:
        values = tuple(live_graph.nodes[node]["EPI"] for node in live_graph)
        calls.append(values)
        for node in live_graph:
            live_graph.nodes[node]["delta_nfr"] = -live_graph.nodes[node]["EPI"]

    graph.graph["compute_delta_nfr"] = refresh
    result = execute_event_remesh_cycle(
        graph,
        _schedule(graph),
        refresh_pressure_after_remesh=True,
    )

    assert calls == [(0.5, 1.5)]
    assert result.schedule_pressure_refresh_callback_invocations == 0
    assert result.post_remesh_pressure_refresh_callback_invocations == 1
    assert result.post_remesh_pressure_refresh_performed
    assert result.pressure_after_remesh_before_refresh == (0.0, 0.0)
    assert result.pressure_after_optional_refresh == (-0.5, -1.5)


def test_valid_mixed_glyph_schedule_reports_capacity_separately() -> None:
    graph = _graph()
    word = ("emission", "coupling", "coherence", "silence")
    result = execute_event_remesh_cycle(
        graph,
        _schedule(
            graph,
            operators=word,
            durations=(0.0, 0.0, 0.0, 0.0, 0.0),
        ),
        metric_weights=(2.0, 1.0),
        suppress_birth_warnings=True,
    )

    assert tuple(event.operator_name for event in result.event_execution.events) == word
    assert result.remesh_applied
    assert result.schedule_capacity_changed
    assert not result.remesh_capacity_changed
    assert result.capacity_before_schedule == (1.0, 1.0)
    assert result.capacity_before_remesh != result.capacity_before_schedule
    assert result.capacity_after_remesh == result.capacity_before_remesh
    assert result.remesh.evidence is not None
    assert result.remesh.evidence.metric_weights == (2.0, 1.0)


class _MutatingIntegrator(AbstractIntegrator):
    def __init__(self, mutation: str) -> None:
        self.mutation = mutation
        self.calls = 0

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
        if self.mutation == "history":
            graph.graph["_epi_hist"].append({0: 9.0, 1: 9.0})
        elif self.mutation == "support":
            graph.add_node(
                "late",
                EPI=0.0,
                nu_f=1.0,
                theta=0.0,
                delta_nfr=0.0,
            )
        graph.nodes[0]["EPI"] += 0.25
        graph.graph["_t"] = float(t) + float(dt)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("history", "changed REMESH history"),
        ("support", "fixed ordered node support"),
    ],
)
def test_schedule_cannot_hide_history_or_support_mutation(
    mutation: str,
    message: str,
) -> None:
    graph = _graph()
    integrator = _MutatingIntegrator(mutation)
    graph.graph["integrator"] = integrator
    before = _state(graph)
    history = graph.graph["_epi_hist"]

    with pytest.raises(TNFRValueError, match=message):
        execute_event_remesh_cycle(
            graph,
            _schedule(graph, durations=(0.25,)),
        )

    assert _state(graph) == before
    assert graph.graph["_epi_hist"] is history
    assert integrator.calls == 0


def test_late_pressure_failure_rolls_back_schedule_history_and_remesh() -> None:
    graph = _graph()
    side_effects: list[str] = []

    def fail_refresh(live_graph: nx.Graph) -> None:
        side_effects.append("emitted")
        live_graph.graph["refresh_started"] = True
        raise RuntimeError("post-remesh refresh failed")

    graph.graph["compute_delta_nfr"] = fail_refresh
    before = _state(graph)
    history = graph.graph["_epi_hist"]

    with pytest.raises(RuntimeError, match="post-remesh refresh failed"):
        execute_event_remesh_cycle(
            graph,
            _schedule(graph),
            refresh_pressure_after_remesh=True,
        )

    assert _state(graph) == before
    assert graph.graph["_epi_hist"] is history
    assert side_effects == ["emitted"]


@pytest.mark.parametrize(
    "weights",
    [
        {0: 1.0},
        (1.0,),
        (1.0, 0.0),
        iter((1.0, 1.0)),
    ],
)
def test_declared_metric_is_strict_and_preflighted(weights: object) -> None:
    graph = _graph()
    before = _state(graph)

    with pytest.raises(TNFRValueError, match="metric_weights"):
        execute_event_remesh_cycle(
            graph,
            _schedule(graph),
            metric_weights=weights,
        )

    assert _state(graph) == before


def test_metric_input_is_materialized_before_execution() -> None:
    graph = _graph()
    supplied = [1.0, 3.0]

    result = execute_event_remesh_cycle(
        graph,
        _schedule(graph),
        metric_weights=supplied,
    )
    supplied[0] = 99.0

    assert result.metric_weights == (1.0, 3.0)
    assert result.pre_schedule_epi.metric_weights == (1.0, 3.0)
    assert result.remesh.evidence is not None
    assert result.remesh.evidence.metric_weights == (1.0, 3.0)


@pytest.mark.parametrize("delay", [True, 1.5, 0, -1])
def test_history_append_rejects_nonpositive_or_nonintegral_delay(
    delay: object,
) -> None:
    graph = _graph()
    graph.graph["REMESH_TAU_LOCAL"] = delay
    before = _state(graph)
    history = graph.graph["_epi_hist"]

    with pytest.raises(TNFRValueError, match="positive integer"):
        execute_event_remesh_cycle(graph, _schedule(graph))

    assert _state(graph) == before
    assert graph.graph["_epi_hist"] is history


def test_history_append_rejects_unmaterializable_delay_capacity() -> None:
    graph = _graph()
    integrator = _MutatingIntegrator("none")
    graph.graph["integrator"] = integrator
    graph.graph["REMESH_TAU_GLOBAL"] = sys.maxsize
    before = _state(graph)
    history = graph.graph["_epi_hist"]

    with pytest.raises(TNFRValueError, match="materializable history"):
        execute_event_remesh_cycle(
            graph,
            _schedule(graph, durations=(0.25,)),
        )

    assert _state(graph) == before
    assert graph.graph["_epi_hist"] is history
    assert integrator.calls == 0


def test_numpy_history_is_materialized_without_truth_value_coercion() -> None:
    np = pytest.importorskip("numpy")
    graph = _graph()
    raw_history = np.asarray([{0: 0.0, 1: 2.0}], dtype=object)
    graph.graph["_epi_hist"] = raw_history

    result = execute_event_remesh_cycle(graph, _schedule(graph))

    assert result.remesh_applied
    assert result.history_container_rebuilt
    assert isinstance(graph.graph["_epi_hist"], deque)
    assert list(graph.graph["_epi_hist"])[0] == {0: 0.0, 1: 2.0}


def test_optional_refresh_rejects_replaced_pressure_hook_atomically() -> None:
    graph = _graph()

    def original_refresh(_: nx.Graph) -> None:
        return None

    def replace_hook(target: nx.Graph, _context: object) -> None:
        target.graph["compute_delta_nfr"] = lambda _: None

    graph.graph["compute_delta_nfr"] = original_refresh
    callback_manager.register_callback(
        graph,
        CallbackEvent.ON_REMESH,
        replace_hook,
        name="replace-pressure-hook",
    )
    before = _state(graph)
    history = graph.graph["_epi_hist"]

    with pytest.raises(TNFRValueError, match="pressure-refresh hook"):
        execute_event_remesh_cycle(
            graph,
            _schedule(graph),
            refresh_pressure_after_remesh=True,
        )

    assert _state(graph) == before
    assert graph.graph["_epi_hist"] is history
    assert graph.graph["compute_delta_nfr"] is original_refresh


def test_public_stub_exposes_event_remesh_cycle_contract() -> None:
    import ast
    from pathlib import Path

    import tnfr.operators.event_remesh_runtime as module

    package = Path(module.__file__).resolve().parent
    module_stub = ast.parse(
        (package / "event_remesh_runtime.pyi").read_text(encoding="utf-8")
    )
    package_stub = ast.parse(
        (package / "__init__.pyi").read_text(encoding="utf-8")
    )
    functions = {
        node.name: node
        for node in module_stub.body
        if isinstance(node, ast.FunctionDef)
    }
    executor = functions["execute_event_remesh_cycle"]

    assert ast.unparse(executor.returns) == "EventRemeshCycleResult"
    assert [argument.arg for argument in executor.args.kwonlyargs] == [
        "metric_weights",
        "refresh_pressure_after_remesh",
        "context",
        "method",
        "n_jobs",
        "suppress_birth_warnings",
        "include_flow_certificates",
        "include_stage_certificates",
    ]
    imported = {
        alias.name
        for node in package_stub.body
        if isinstance(node, ast.ImportFrom)
        and node.module == "event_remesh_runtime"
        for alias in node.names
    }
    assert imported == {
        "EventRemeshCycleResult",
        "RemeshHistoryTransitionObservation",
        "WeightedEPIObservation",
        "execute_event_remesh_cycle",
    }

class _DoubleAppendDeque(deque):
    def append(self, value: object) -> None:
        super().append(value)
        super().append(value)


def test_history_subclass_cannot_duplicate_the_owned_pre_remesh_sample() -> None:
    graph = _graph()
    graph.graph["_epi_hist"] = _DoubleAppendDeque(
        graph.graph["_epi_hist"], maxlen=64
    )

    result = execute_event_remesh_cycle(graph, _schedule(graph))

    assert result.remesh_applied
    assert result.history_container_rebuilt
    assert type(graph.graph["_epi_hist"]) is deque
    assert list(graph.graph["_epi_hist"]) == [
        {0: 0.0, 1: 2.0},
        {0: 2.0, 1: 0.0},
    ]
    assert result.post_remesh_epi.epi_values == (0.5, 1.5)


def test_applied_remesh_restarts_same_time_mutation_history_at_right_endpoint() -> None:
    graph = _graph()
    for node in graph:
        graph.nodes[node]["epi_time_history"] = deque(
            [(0.0, graph.nodes[node]["EPI"])], maxlen=3
        )

    result = execute_event_remesh_cycle(graph, _schedule(graph))

    assert result.post_remesh_epi_time_boundary_recorded
    assert result.remesh.epi_time_boundary_recorded
    assert list(graph.nodes[0]["epi_time_history"]) == [(0.0, 0.5)]
    assert list(graph.nodes[1]["epi_time_history"]) == [(0.0, 1.5)]
    assert list(graph.graph["_epi_hist"]) == [
        {0: 0.0, 1: 2.0},
        {0: 2.0, 1: 0.0},
    ]


def test_invalid_remesh_configuration_preflights_before_positive_flow() -> None:
    graph = _graph()
    integrator = _MutatingIntegrator("none")
    graph.graph["integrator"] = integrator
    graph.graph["REMESH_ALPHA"] = 2.0
    before = _state(graph)

    with pytest.raises(ValueError, match="REMESH_alpha"):
        execute_event_remesh_cycle(
            graph,
            _schedule(graph, durations=(0.25,)),
        )

    assert _state(graph) == before
    assert integrator.calls == 0


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("clock", "runtime clock"),
        ("event_log", "hybrid_event_log"),
        ("phase", "capacity, pressure or phase"),
        ("time_history", "authoritative epi_time_history"),
    ],
)
def test_remesh_observer_cannot_corrupt_cycle_trace_surfaces(
    mutation: str,
    message: str,
) -> None:
    graph = _graph()

    def mutate(target: nx.Graph, _context: object) -> None:
        if mutation == "clock":
            target.graph["_t"] = 9.0
        elif mutation == "event_log":
            target.graph["hybrid_event_log"] = []
        elif mutation == "phase":
            target.nodes[0]["theta"] = 1.25
        else:
            target.nodes[0]["epi_time_history"].clear()

    callback_manager.register_callback(
        graph,
        CallbackEvent.ON_REMESH,
        mutate,
        name=f"mutate-{mutation}",
    )
    before = _state(graph)
    history = graph.graph["_epi_hist"]

    with pytest.raises(TNFRValueError, match=message):
        execute_event_remesh_cycle(graph, _schedule(graph))

    assert _state(graph) == before
    assert graph.graph["_epi_hist"] is history


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("clock", "runtime clock"),
        ("event_log", "hybrid_event_log"),
        ("phase", "changed phase"),
        ("time_history", "epi_time_history"),
        ("hook", "pressure-refresh hook"),
        ("edge", "changed edge state"),
        ("remesh_meta", "_REMESH_META"),
        ("alpha_source", "_REMESH_ALPHA_SRC"),
        ("telemetry", "history"),
    ],
)
def test_pressure_refresh_cannot_corrupt_cycle_trace_surfaces(
    mutation: str,
    message: str,
) -> None:
    graph = _graph()

    def refresh(target: nx.Graph) -> None:
        if mutation == "clock":
            target.graph["_t"] = 9.0
        elif mutation == "event_log":
            target.graph["hybrid_event_log"] = []
        elif mutation == "phase":
            target.nodes[0]["theta"] = 1.25
        elif mutation == "time_history":
            target.nodes[0]["epi_time_history"].clear()
        elif mutation == "hook":
            target.graph["compute_delta_nfr"] = lambda _: None
        elif mutation == "edge":
            target.edges[0, 1]["weight"] = 9.0
        elif mutation == "remesh_meta":
            target.graph["_REMESH_META"].clear()
        elif mutation == "alpha_source":
            target.graph["_REMESH_ALPHA_SRC"] = "forged"
        else:
            target.graph["history"]["remesh_events"].clear()

    graph.graph["REMESH_LOG_EVENTS"] = True
    graph.graph["compute_delta_nfr"] = refresh
    before = _state(graph)
    history = graph.graph["_epi_hist"]

    with pytest.raises(TNFRValueError, match=message):
        execute_event_remesh_cycle(
            graph,
            _schedule(graph),
            refresh_pressure_after_remesh=True,
        )

    assert _state(graph) == before
    assert graph.graph["_epi_hist"] is history


def test_unrepresentable_disagreement_keeps_exact_observation() -> None:
    graph = _graph(current=(1e308, -1e308), past=None)
    graph.graph["EPI_MIN"] = -1e308
    graph.graph["EPI_MAX"] = 1e308

    result = execute_event_remesh_cycle(graph, _schedule(graph))

    assert not result.remesh_applied
    assert result.pre_schedule_epi.exact_disagreement_energy > 0
    assert result.pre_schedule_epi.disagreement_energy is None

def test_cycle_forwards_detached_runtime_flow_certificates() -> None:
    graph = _graph()
    graph.graph["DT_MIN"] = 0.0

    result = execute_event_remesh_cycle(
        graph,
        _schedule(graph, durations=(0.25,)),
        include_flow_certificates=True,
    )

    execution = result.event_execution
    assert execution.flow_certification_requested
    assert len(execution.flow_interval_evidence) == 1
    evidence = execution.flow_interval_evidence[0]
    assert evidence.runtime_bound_binary64_interval_identified
    assert not evidence.runtime_bound_exact_affine_map_identified
    assert evidence.certificate is not None
    assert evidence.certificate.left_epi == (2.0, 0.0)
    assert evidence.certificate.right_epi == (2.0, 0.0)
    assert result.post_remesh_epi.epi_values == (0.5, 1.5)
    assert "flow_interval_evidence" not in graph.graph


@pytest.mark.parametrize("value", [None, 0, 1.0, "yes"])
def test_cycle_flow_certificate_flag_requires_a_strict_bool(value: object) -> None:
    graph = _graph()
    before = _state(graph)

    with pytest.raises(TypeError, match="include_flow_certificates"):
        execute_event_remesh_cycle(
            graph,
            _schedule(graph),
            include_flow_certificates=value,  # type: ignore[arg-type]
        )

    assert _state(graph) == before


def test_cycle_forwards_stage_certification_request() -> None:
    graph = _graph()
    graph.graph["DT_MIN"] = 0.0

    result = execute_event_remesh_cycle(
        graph,
        _schedule(graph, durations=(0.25,)),
        include_stage_certificates=True,
    )

    execution = result.event_execution
    assert execution.stage_certification_requested
    assert execution.flow_certification_requested
    assert execution.glyph_stage_evidence == ()
    assert execution.represented_epi_schedule_composition is not None
    assert "glyph_stage_evidence" not in graph.graph


@pytest.mark.parametrize("value", [None, 0, 1.0, "yes"])
def test_cycle_stage_certificate_flag_requires_a_strict_bool(value: object) -> None:
    graph = _graph()
    before = _state(graph)

    with pytest.raises(TypeError, match="include_stage_certificates"):
        execute_event_remesh_cycle(
            graph,
            _schedule(graph),
            include_stage_certificates=value,  # type: ignore[arg-type]
        )

    assert _state(graph) == before


def test_history_transition_records_exact_append_and_selected_lags() -> None:
    graph = _graph()

    result = execute_event_remesh_cycle(graph, _schedule(graph))
    transition = result.history_transition

    assert isinstance(transition, RemeshHistoryTransitionObservation)
    assert transition.nodes == (0, 1)
    assert transition.incoming_exact_history == (
        (Fraction(0), Fraction(2)),
    )
    assert transition.outgoing_exact_history == (
        (Fraction(0), Fraction(2)),
        (Fraction(2), Fraction(0)),
    )
    assert transition.appended_exact_pre_remesh_epi == (
        Fraction(2),
        Fraction(0),
    )
    assert transition.selected_local_delayed_epi == (
        Fraction(0),
        Fraction(2),
    )
    assert transition.selected_global_delayed_epi == (
        Fraction(0),
        Fraction(2),
    )
    assert transition.tau_local == 1
    assert transition.tau_global == 1
    assert transition.history_maxlen == 64
    assert transition.incoming_history_present
    assert transition.incoming_history_is_canonical_deque
    assert not transition.history_container_rebuilt
    assert not transition.oldest_snapshot_evicted
    assert not transition.incoming_history_truncated_during_rebuild
    assert transition.history_rebuild_truncation_count == 0
    assert transition.canonical_history_transition_certified
    assert result.phase_before_schedule == (0.0, 0.0)
    assert result._proof_fields_are_intact()


def test_history_transition_records_independent_insufficient_lags() -> None:
    graph = _graph()
    graph.graph["REMESH_TAU_LOCAL"] = 1
    graph.graph["REMESH_TAU_GLOBAL"] = 2

    result = execute_event_remesh_cycle(graph, _schedule(graph))
    transition = result.history_transition

    assert transition.selected_local_delayed_epi == (
        Fraction(0),
        Fraction(2),
    )
    assert transition.selected_global_delayed_epi is None
    assert result.remesh.status == "insufficient_history"
    assert not result.remesh_applied
    assert transition.canonical_history_transition_certified
    assert result._proof_fields_are_intact()


def test_history_transition_records_canonical_left_eviction() -> None:
    graph = _graph()
    incoming = [
        {0: float(index), 1: -float(index)}
        for index in range(64)
    ]
    graph.graph["_epi_hist"] = deque(incoming, maxlen=64)

    result = execute_event_remesh_cycle(graph, _schedule(graph))
    transition = result.history_transition

    assert transition.incoming_history_is_canonical_deque
    assert not transition.history_container_rebuilt
    assert transition.oldest_snapshot_evicted
    assert transition.history_rebuild_truncation_count == 0
    assert len(transition.incoming_exact_history) == 64
    assert len(transition.outgoing_exact_history) == 64
    assert transition.outgoing_exact_history[0] == (
        Fraction(1),
        Fraction(-1),
    )
    assert transition.outgoing_exact_history[-1] == (
        Fraction(2),
        Fraction(0),
    )
    assert result.history_length_before_append == 64
    assert result.history_length_after_append == 64
    assert result._proof_fields_are_intact()


def test_history_transition_records_rebuild_truncation_before_eviction() -> None:
    graph = _graph()
    graph.graph["_epi_hist"] = [
        {0: float(index), 1: -float(index)}
        for index in range(66)
    ]

    result = execute_event_remesh_cycle(graph, _schedule(graph))
    transition = result.history_transition

    assert not transition.incoming_history_is_canonical_deque
    assert transition.history_container_rebuilt
    assert transition.incoming_history_truncated_during_rebuild
    assert transition.history_rebuild_truncation_count == 2
    assert transition.oldest_snapshot_evicted
    assert len(transition.incoming_exact_history) == 66
    assert len(transition.outgoing_exact_history) == 64
    assert transition.outgoing_exact_history[0] == (
        Fraction(3),
        Fraction(-3),
    )
    assert transition.outgoing_exact_history[-1] == (
        Fraction(2),
        Fraction(0),
    )
    assert result.history_length_before_cycle == 66
    assert result.history_length_before_append == 64
    assert result._proof_fields_are_intact()


@pytest.mark.parametrize(
    "malformed",
    [
        [{0: 1.0}],
        [{0: 1.0, 1: 2.0, 2: 3.0}],
        [(1.0, 2.0)],
        [{0: complex(1.0, 1.0), 1: 2.0}],
    ],
)
def test_history_transition_rejects_malformed_incoming_history_atomically(
    malformed: object,
) -> None:
    graph = _graph()
    graph.graph["_epi_hist"] = malformed
    before = _state(graph)

    with pytest.raises(TNFRValueError):
        execute_event_remesh_cycle(graph, _schedule(graph))

    assert _state(graph) == before


def test_history_transition_and_cycle_proofs_fail_closed_after_tampering() -> None:
    graph = _graph()
    result = execute_event_remesh_cycle(graph, _schedule(graph))
    transition = result.history_transition

    with pytest.raises(ValueError):
        replace(
            transition,
            selected_local_delayed_epi=None,
        )

    object.__setattr__(
        transition,
        "selected_local_delayed_epi",
        None,
    )
    assert not transition.canonical_history_transition_certified
    assert not transition._proof_fields_are_intact()
    assert not result._proof_fields_are_intact()


def test_cycle_proof_detects_nested_remesh_plan_tampering_by_value() -> None:
    graph = _graph()
    result = execute_event_remesh_cycle(graph, _schedule(graph))
    proposal = result.remesh.plan.proposals[0]

    object.__setattr__(proposal, "epi_local", 9.0)

    assert not result._proof_fields_are_intact()


def test_cycle_proof_detects_nested_schedule_composition_tampering() -> None:
    graph = _graph()
    graph.graph["DT_MIN"] = 0.0
    result = execute_event_remesh_cycle(
        graph,
        _schedule(graph, durations=(0.25,)),
        include_stage_certificates=True,
    )
    composition = (
        result.event_execution.represented_epi_schedule_composition
    )
    assert composition is not None
    assert result._proof_fields_are_intact()

    object.__setattr__(
        composition,
        "exact_energy_gain_upper_bound",
        Fraction(0),
    )

    assert not composition._proof_fields_are_intact()
    assert not result._proof_fields_are_intact()


def test_cycle_proof_detects_boundary_phase_tampering() -> None:
    graph = _graph()
    result = execute_event_remesh_cycle(graph, _schedule(graph))

    object.__setattr__(result, "phase_before_schedule", (0.5, 0.0))

    assert not result._proof_fields_are_intact()

def test_history_transition_records_reverse_independent_lag_availability() -> None:
    graph = _graph()
    graph.graph["REMESH_TAU_LOCAL"] = 2
    graph.graph["REMESH_TAU_GLOBAL"] = 1

    result = execute_event_remesh_cycle(graph, _schedule(graph))
    transition = result.history_transition

    assert transition.selected_local_delayed_epi is None
    assert transition.selected_global_delayed_epi == (
        Fraction(0),
        Fraction(2),
    )
    assert result.remesh.status == "insufficient_history"
    assert transition.canonical_history_transition_certified
    assert result._proof_fields_are_intact()


class _CyclicHashableNode:
    def __init__(self, label: str) -> None:
        self.label = label
        self.self_reference = self

    def __hash__(self) -> int:
        return hash(self.label)

    def __eq__(self, other: object) -> bool:
        return (
            type(other) is _CyclicHashableNode
            and self.label == other.label
        )


def test_cycle_proof_serialization_accepts_cyclic_hashable_nodes() -> None:
    left = _CyclicHashableNode("left")
    right = _CyclicHashableNode("right")
    graph = _graph()
    nx.relabel_nodes(graph, {0: left, 1: right}, copy=False)
    graph.graph["_epi_hist"] = deque(
        [{left: 0.0, right: 2.0}],
        maxlen=64,
    )

    result = execute_event_remesh_cycle(graph, _schedule(graph))

    assert result.target_nodes == (left, right)
    assert result.history_transition.nodes == (left, right)
    assert result._proof_fields_are_intact()


def test_cycle_hard_false_claims_are_read_only_properties() -> None:
    graph = _graph()
    result = execute_event_remesh_cycle(graph, _schedule(graph))

    for name in (
        "remesh_history_repetition_certified",
        "mixed_runtime_gain_certified",
        "external_side_effects_rolled_back",
    ):
        assert getattr(result, name) is False
        with pytest.raises(AttributeError):
            object.__setattr__(result, name, True)

class _SlottedMutableNode:
    __slots__ = ("label", "payload", "self_reference")

    def __init__(self, label: str) -> None:
        self.label = label
        self.payload = "original"
        self.self_reference = self

    def __hash__(self) -> int:
        return hash(self.label)

    def __eq__(self, other: object) -> bool:
        return (
            type(other) is _SlottedMutableNode
            and self.label == other.label
        )


def test_cycle_proof_serializes_mutable_slotted_nodes_by_mro_state() -> None:
    left = _SlottedMutableNode("left")
    right = _SlottedMutableNode("right")
    graph = _graph()
    nx.relabel_nodes(graph, {0: left, 1: right}, copy=False)
    graph.graph["_epi_hist"] = deque(
        [{left: 0.0, right: 2.0}],
        maxlen=64,
    )

    result = execute_event_remesh_cycle(graph, _schedule(graph))

    assert result._proof_fields_are_intact()
    left.payload = "tampered"
    assert not result.history_transition._proof_fields_are_intact()
    assert not result._proof_fields_are_intact()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("capacity", "non-EPI alias channels"),
        ("pressure", "non-EPI alias channels"),
        ("phase", "non-EPI alias channels"),
        ("edge", "protected edge state"),
    ],
)
def test_outer_cycle_rejects_remesh_wrapper_channel_or_edge_mutation(
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    message: str,
) -> None:
    graph = _graph()
    before = _state(graph)
    history = graph.graph["_epi_hist"]
    original = event_remesh_runtime.apply_network_remesh

    def mutate_after_apply(target: nx.Graph, **kwargs: object) -> Any:
        result = original(target, **kwargs)
        if mutation == "capacity":
            target.nodes[0]["nu_f"] = 3.0
        elif mutation == "pressure":
            target.nodes[0]["delta_nfr"] = 4.0
        elif mutation == "phase":
            target.nodes[0]["theta"] = 1.25
        else:
            target.edges[0, 1]["weight"] = 9.0
        return result

    monkeypatch.setattr(
        event_remesh_runtime,
        "apply_network_remesh",
        mutate_after_apply,
    )
    with pytest.raises(TNFRValueError, match=message):
        execute_event_remesh_cycle(graph, _schedule(graph))

    assert _state(graph) == before
    assert graph.graph["_epi_hist"] is history


def test_outer_cycle_rejects_applied_remesh_without_exact_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = _graph()
    before = _state(graph)
    history = graph.graph["_epi_hist"]
    original = event_remesh_runtime.apply_network_remesh

    def remove_evidence(target: nx.Graph, **kwargs: object) -> Any:
        result = original(target, **kwargs)
        return replace(
            result,
            plan=replace(result.plan, evidence=None),
        )

    monkeypatch.setattr(
        event_remesh_runtime,
        "apply_network_remesh",
        remove_evidence,
    )
    with pytest.raises(RuntimeError, match="omitted exact stability evidence"):
        execute_event_remesh_cycle(graph, _schedule(graph))

    assert _state(graph) == before
    assert graph.graph["_epi_hist"] is history
