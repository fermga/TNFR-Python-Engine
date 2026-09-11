"""Atomic runtime realization of exact operator-event schedules."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from copy import deepcopy
from dataclasses import FrozenInstanceError, asdict, fields, replace
from fractions import Fraction
from types import MappingProxyType
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
from tnfr.operators.event_timing import (
    build_operator_event_schedule,
    build_physical_flow_partition,
)
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
        for node in graph:
            data = graph.nodes[node]
            data["EPI"] = float(data["EPI"]) + float(dt) * (
                float(data["nu_f"]) * float(data["delta_nfr"])
            )
        multiplier = 2.0 if self.wrong_endpoint else 1.0
        graph.graph["_t"] = float(t) + multiplier * float(dt)


class HeldInputIntegrator(AbstractIntegrator):
    """Custom integrator with an exact held-input nodal residual."""

    def __init__(self) -> None:
        self.calls = 0

    def integrate(self, graph, *, dt, t, method, n_jobs) -> None:
        self.calls += 1
        for node in graph:
            data = graph.nodes[node]
            data["EPI"] = float(data["EPI"]) + float(dt) * (
                float(data["nu_f"]) * float(data["delta_nfr"])
            )
            data["dEPI_dt"] = float(data["nu_f"]) * float(
                data["delta_nfr"]
            )
            data["d2EPI_dt2"] = 0.0
        graph.graph["_t"] = float(t) + float(dt)


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


def test_schedule_accepts_immutable_mapping_proxy_metadata() -> None:
    graph = _graph()
    configuration = MappingProxyType({"mode": "fixed", "scale": 1.0})
    graph.graph["immutable_configuration"] = configuration

    execute_operator_event_schedule(graph, _schedule())

    assert graph.graph["immutable_configuration"] is configuration


def test_event_and_result_seals_fail_closed_after_low_level_tampering() -> None:
    event_result = execute_operator_event_schedule(_graph(), _schedule())
    event = event_result.events[0]
    assert event._proof_fields_are_intact()

    object.__setattr__(event, "nodes_processed", event.nodes_processed + 1)

    assert not event._proof_fields_are_intact()
    with pytest.raises(ValueError, match="proof fields are not intact"):
        event.as_record()
    assert not event_result._proof_fields_are_intact()
    assert not event_result.runtime_clock_checked
    assert not event.zero_duration
    assert event.history_channel == "hybrid_event_log"
    assert not event.feeds_epi_time_history

    result = execute_operator_event_schedule(_graph(), _schedule())
    assert result._proof_fields_are_intact()
    object.__setattr__(result, "final_time", result.final_time + 1.0)

    assert not result._proof_fields_are_intact()
    assert not result.runtime_clock_checked
    assert not result.whole_schedule_graph_state_atomic
    assert not result.operator_jumps_have_zero_duration


def test_runtime_evidence_preserves_historical_dataclass_contract_schema() -> None:
    result = execute_operator_event_schedule(
        _graph(),
        _schedule(),
        include_stage_certificates=True,
    )
    composition = result.represented_epi_schedule_composition
    assert composition is not None
    stage = result.glyph_stage_evidence[0]
    values_and_contracts = (
        (
            result.events[0],
            {
                "zero_duration": True,
                "history_channel": "hybrid_event_log",
                "feeds_epi_time_history": False,
            },
        ),
        (
            result.flow_interval_evidence[0],
            {
                "solver_accuracy_certified": False,
                "future_or_repeated_schedule_stability_certified": False,
            },
        ),
        (
            stage,
            {
                "solver_accuracy_certified": False,
                "future_or_repeated_schedule_stability_certified": False,
                "scope": stage.scope,
            },
        ),
        (
            composition,
            {"scope": composition.scope},
        ),
        (
            result,
            {
                "runtime_clock_checked": True,
                "flow_provenance": (
                    "tnfr.operators.event_runtime."
                    "execute_operator_event_schedule"
                ),
                "nodal_flow_inputs": (
                    "live_nu_f_and_delta_nfr_at_each_interval_start"
                ),
                "whole_schedule_graph_state_atomic": True,
                "operator_jumps_have_zero_duration": True,
                "solver_accuracy_certified": False,
                "adaptive_u2_u4_policy": False,
                "external_side_effects_rolled_back": False,
                "future_or_repeated_schedule_stability_certified": False,
                "flow_scope": result.flow_scope,
            },
        ),
    )

    for value, contracts in values_and_contracts:
        declared = {item.name: item for item in fields(value)}
        payload = asdict(value)
        rendered = repr(value)
        for name, expected in contracts.items():
            assert name in declared
            assert not declared[name].init
            assert payload[name] == expected
            assert f"{name}=" in rendered
            with pytest.raises(ValueError, match="init=False"):
                replace(value, **{name: expected})

    historical_result_init_fields = (
        "schedule",
        "target_nodes",
        "flow_interval_indices",
        "positive_flow_interval_indices",
        "events",
        "final_time",
        "integrator_name",
        "pressure_refresh_callback_invocations",
        "flow_certification_requested",
        "flow_interval_evidence",
        "stage_certification_requested",
        "glyph_stage_evidence",
        "represented_epi_schedule_composition",
    )
    current_init_fields = tuple(
        item.name for item in fields(result) if item.init
    )
    assert current_init_fields[: len(historical_result_init_fields)] == (
        historical_result_init_fields
    )
    reconstructed = OperatorEventExecutionResult(
        *(getattr(result, name) for name in historical_result_init_fields)
    )
    assert reconstructed._proof_fields_are_intact() is False
    assert reconstructed.schedule is result.schedule
    assert reconstructed.stage_certification_requested


@pytest.mark.parametrize(
    "graph_type",
    (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph),
)
def test_valid_flow_does_not_retain_guard_only_networkx_cached_views(
    graph_type: type[nx.Graph],
) -> None:
    graph = graph_type()
    graph.add_edge(0, 1)
    graph.graph.update(
        _t=0.0,
        _gamma_spec={"type": "none"},
        EPI_MIN=-4.0,
        EPI_MAX=4.0,
    )
    for node, data in graph._node.items():
        data.update(
            EPI=float(node),
            nu_f=1.0,
            theta=0.0,
            delta_nfr=0.1,
            glyph_history=[],
        )
    cached_names = {
        "adj",
        "nodes",
        "edges",
        "degree",
        "succ",
        "pred",
        "out_edges",
        "in_edges",
        "in_degree",
        "out_degree",
    }
    for name in cached_names:
        graph.__dict__.pop(name, None)

    schedule = build_operator_event_schedule(
        (),
        start_time=0.0,
        flow_durations=(0.25,),
    )
    execute_operator_event_schedule(graph, schedule)

    # The integrator genuinely reads ``nodes``.  Every other cached surface was
    # created only to seal NetworkX bindings and must be removed afterward.
    assert cached_names.intersection(graph.__dict__) == {"nodes"}


def test_runtime_dataclass_contract_tampering_fails_closed() -> None:
    result = execute_operator_event_schedule(
        _graph(),
        _schedule(),
        include_stage_certificates=True,
    )
    composition = result.represented_epi_schedule_composition
    assert composition is not None
    stage = result.glyph_stage_evidence[0]
    stage_scope = stage.scope
    composition_scope = composition.scope
    flow_scope = result.flow_scope
    tamper_cases = (
        (result.events[0], "zero_duration", False, False),
        (
            result.events[0],
            "history_channel",
            "forged",
            "hybrid_event_log",
        ),
        (result.events[0], "feeds_epi_time_history", True, False),
        (
            result.flow_interval_evidence[0],
            "solver_accuracy_certified",
            True,
            False,
        ),
        (
            result.flow_interval_evidence[0],
            "future_or_repeated_schedule_stability_certified",
            True,
            False,
        ),
        (
            stage,
            "solver_accuracy_certified",
            True,
            False,
        ),
        (
            stage,
            "future_or_repeated_schedule_stability_certified",
            True,
            False,
        ),
        (
            stage,
            "scope",
            "forged",
            stage_scope,
        ),
        (
            composition,
            "scope",
            "forged",
            composition_scope,
        ),
        (result, "runtime_clock_checked", False, False),
        (
            result,
            "flow_provenance",
            "forged",
            "tnfr.operators.event_runtime.execute_operator_event_schedule",
        ),
        (
            result,
            "nodal_flow_inputs",
            "forged",
            "live_nu_f_and_delta_nfr_at_each_interval_start",
        ),
        (result, "whole_schedule_graph_state_atomic", False, False),
        (result, "operator_jumps_have_zero_duration", False, False),
        (result, "solver_accuracy_certified", True, False),
        (result, "adaptive_u2_u4_policy", True, False),
        (result, "external_side_effects_rolled_back", True, False),
        (
            result,
            "future_or_repeated_schedule_stability_certified",
            True,
            False,
        ),
        (
            result,
            "flow_scope",
            "forged",
            flow_scope,
        ),
    )

    for value, name, replacement, public_value in tamper_cases:
        original = object.__getattribute__(value, name)
        assert value._proof_fields_are_intact()
        object.__setattr__(value, name, replacement)
        assert not value._proof_fields_are_intact()
        assert getattr(value, name) == public_value
        object.__setattr__(value, name, original)
        assert value._proof_fields_are_intact()


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

    def refresh(_live_graph: nx.Graph) -> None:
        return None

    graph.graph["compute_delta_nfr"] = refresh

    result = execute_operator_event_schedule(graph, _schedule())

    assert result.pressure_refresh_callback_invocations == len(_WORD)
    assert result.stage_pressure_refresh_callback_invocations == len(_WORD)


def test_stage_pressure_callback_cannot_replace_its_binding() -> None:
    graph = _graph()

    def replacement(_live_graph: nx.Graph) -> None:
        return None

    def refresh(live_graph: nx.Graph) -> None:
        live_graph.graph["compute_delta_nfr"] = replacement

    graph.graph["compute_delta_nfr"] = refresh
    before = _plain_state(graph)

    with pytest.raises(RuntimeError, match="pressure callback changed"):
        execute_operator_event_schedule(graph, _schedule())

    assert _plain_state(graph) == before
    assert graph.graph["compute_delta_nfr"] is refresh


def test_stage_pressure_callback_cannot_change_phase() -> None:
    graph = _graph()

    def refresh(live_graph: nx.Graph) -> None:
        live_graph.nodes[0]["delta_nfr"] = 0.4
        live_graph.nodes[0]["theta"] = 0.5

    graph.graph["compute_delta_nfr"] = refresh
    before = _plain_state(graph)

    with pytest.raises(
        TNFRValueError,
        match="pressure callback changed non-pressure graph state",
    ):
        execute_operator_event_schedule(
            graph,
            _schedule((0.0, 0.0, 0.0, 0.0, 0.0)),
        )

    assert _plain_state(graph) == before
    assert graph.graph["compute_delta_nfr"] is refresh


def test_integrator_cannot_replace_hook_before_event_callback() -> None:
    graph = _graph()
    evil_calls: list[str] = []

    def expected_refresh(_live_graph: nx.Graph) -> None:
        return None

    def evil_refresh(_live_graph: nx.Graph) -> None:
        evil_calls.append("called")

    class HookReplacingIntegrator(AbstractIntegrator):
        def __init__(self) -> None:
            self.calls = 0

        def integrate(self, live_graph, *, dt, t, method, n_jobs) -> None:
            self.calls += 1
            live_graph.graph["compute_delta_nfr"] = evil_refresh
            live_graph.graph["_t"] = float(t) + float(dt)

    integrator = HookReplacingIntegrator()
    graph.graph["compute_delta_nfr"] = expected_refresh
    graph.graph["integrator"] = integrator
    before = _plain_state(graph)

    with pytest.raises(TNFRValueError, match="may update only EPI"):
        execute_operator_event_schedule(
            graph,
            _schedule((0.25, 0.0, 0.0, 0.0, 0.0)),
        )

    assert evil_calls == []
    assert _plain_state(graph) == before
    assert graph.graph["compute_delta_nfr"] is expected_refresh
    assert integrator.calls == 0


def test_integrator_cannot_hide_history_on_a_new_node() -> None:
    graph = _graph()

    class LateHistoryIntegrator(AbstractIntegrator):
        def integrate(self, live_graph, *, dt, t, method, n_jobs) -> None:
            endpoint = float(t) + float(dt)
            live_graph.add_node(
                "late",
                EPI=1.0,
                nu_f=1.0,
                theta=0.0,
                delta_nfr=0.0,
                epi_time_history=[(endpoint, 1.0)],
            )
            live_graph.graph["_t"] = endpoint

    graph.graph["integrator"] = LateHistoryIntegrator()
    schedule = build_operator_event_schedule(
        (),
        start_time=0.0,
        flow_durations=(0.25,),
    )

    with pytest.raises(TNFRValueError, match="must not write epi_time_history"):
        execute_operator_event_schedule(graph, schedule)

    assert "late" not in graph
    assert graph.graph["_t"] == 0.0
    assert all("epi_time_history" not in graph.nodes[node] for node in graph)


@pytest.mark.parametrize(
    ("mutation", "expected_check"),
    (
        ("phase", "phase_preserved"),
        ("capacity", "capacity_preserved"),
        ("pressure", "pressure_preserved"),
        ("edge", "edge_state_preserved"),
        ("topology", "node_support_preserved"),
        ("history", "protected_nodal_state_preserved"),
        ("configuration", "graph_configuration_preserved"),
    ),
)
def test_integrator_cannot_mutate_protected_flow_channels(
    mutation: str,
    expected_check: str,
) -> None:
    graph = _graph()

    class MutatingIntegrator(AbstractIntegrator):
        def __init__(self) -> None:
            self.calls = 0

        def integrate(self, live_graph, *, dt, t, method, n_jobs) -> None:
            self.calls += 1
            if mutation == "phase":
                live_graph.nodes[0]["theta"] = 0.5
            elif mutation == "capacity":
                live_graph.nodes[0]["nu_f"] = 2.0
            elif mutation == "pressure":
                live_graph.nodes[0]["delta_nfr"] = 0.5
            elif mutation == "edge":
                live_graph.edges[0, 1]["weight"] = 2.0
            elif mutation == "topology":
                live_graph.add_node("late")
            elif mutation == "history":
                live_graph.nodes[0]["glyph_history"].append("OZ")
            elif mutation == "configuration":
                live_graph.graph["INTEGRATOR_METHOD"] = "rk4"
            live_graph.graph["_t"] = float(t) + float(dt)

    integrator = MutatingIntegrator()
    graph.graph["integrator"] = integrator
    schedule = build_operator_event_schedule(
        (),
        start_time=0.0,
        flow_durations=(0.25,),
    )
    before = _plain_state(graph)

    with pytest.raises(TNFRValueError, match="may update only EPI") as failure:
        execute_operator_event_schedule(graph, schedule)

    assert expected_check in failure.value.context["failed_preservation_checks"]
    assert _plain_state(graph) == before
    assert integrator.calls == 0


def test_integrator_cannot_rebind_one_side_of_a_protected_alias() -> None:
    graph = _graph()
    shared_payload = {"marker": "shared"}
    graph.graph["shared_payload"] = shared_payload
    graph.nodes[0]["payload"] = shared_payload

    class AliasRebindingIntegrator(AbstractIntegrator):
        def __init__(self) -> None:
            self.calls = 0

        def integrate(self, live_graph, *, dt, t, method, n_jobs) -> None:
            self.calls += 1
            live_graph.nodes[0]["payload"] = {"marker": "shared"}
            for node in live_graph:
                data = live_graph.nodes[node]
                rate = float(data["nu_f"]) * float(data["delta_nfr"])
                data["EPI"] = float(data["EPI"]) + float(dt) * rate
                data["dEPI_dt"] = rate
                data["d2EPI_dt2"] = 0.0
            live_graph.graph["_t"] = float(t) + float(dt)

    integrator = AliasRebindingIntegrator()
    graph.graph["integrator"] = integrator
    schedule = build_operator_event_schedule(
        (),
        start_time=0.0,
        flow_durations=(0.25,),
    )

    with pytest.raises(TNFRValueError, match="may update only EPI") as failure:
        execute_operator_event_schedule(graph, schedule)

    assert "protected_identity_preserved" in failure.value.context[
        "failed_preservation_checks"
    ]
    assert graph.graph["shared_payload"] is graph.nodes[0]["payload"]
    assert graph.graph["shared_payload"] == shared_payload
    assert integrator.calls == 0


def test_integrator_cannot_break_undirected_edge_storage_alias() -> None:
    graph = _graph()
    original_edge_storage = graph._adj[0][1]

    class EdgeAliasRebindingIntegrator(AbstractIntegrator):
        def integrate(self, live_graph, *, dt, t, method, n_jobs) -> None:
            live_graph._adj[1][0] = dict(live_graph._adj[1][0])
            for node in live_graph:
                data = live_graph.nodes[node]
                rate = float(data["nu_f"]) * float(data["delta_nfr"])
                data["EPI"] = float(data["EPI"]) + float(dt) * rate
                data["dEPI_dt"] = rate
                data["d2EPI_dt2"] = 0.0
            live_graph.graph["_t"] = float(t) + float(dt)

    graph.graph["integrator"] = EdgeAliasRebindingIntegrator()
    schedule = build_operator_event_schedule(
        (),
        start_time=0.0,
        flow_durations=(0.25,),
    )

    with pytest.raises(TNFRValueError, match="may update only EPI") as failure:
        execute_operator_event_schedule(graph, schedule)

    assert "protected_identity_preserved" in failure.value.context[
        "failed_preservation_checks"
    ]
    assert graph._adj[0][1] is graph._adj[1][0]
    assert graph._adj[0][1] is original_edge_storage


def test_integrator_cannot_break_directed_root_adjacency_alias() -> None:
    source = _graph()
    graph = nx.DiGraph(source)
    graph.graph.update(source.graph)
    original_adjacency = graph._adj

    class RootAliasRebindingIntegrator(AbstractIntegrator):
        def integrate(self, live_graph, *, dt, t, method, n_jobs) -> None:
            live_graph.__dict__["_succ"] = dict(live_graph._succ)
            for node in live_graph:
                data = live_graph.nodes[node]
                rate = float(data["nu_f"]) * float(data["delta_nfr"])
                data["EPI"] = float(data["EPI"]) + float(dt) * rate
                data["dEPI_dt"] = rate
                data["d2EPI_dt2"] = 0.0
            live_graph.graph["_t"] = float(t) + float(dt)

    graph.graph["integrator"] = RootAliasRebindingIntegrator()
    schedule = build_operator_event_schedule(
        (),
        start_time=0.0,
        flow_durations=(0.25,),
    )

    with pytest.raises(TNFRValueError, match="may update only EPI") as failure:
        execute_operator_event_schedule(graph, schedule)

    assert "protected_identity_preserved" in failure.value.context[
        "failed_preservation_checks"
    ]
    assert graph._adj is graph._succ
    assert graph._adj is original_adjacency


def test_custom_integrator_requires_exact_held_input_nodal_residual() -> None:
    graph = _graph()

    class ArbitraryEPIIntegrator(AbstractIntegrator):
        def integrate(self, live_graph, *, dt, t, method, n_jobs) -> None:
            live_graph.nodes[0]["EPI"] += 1.0
            live_graph.graph["_t"] = float(t) + float(dt)

    graph.graph["integrator"] = ArbitraryEPIIntegrator()
    schedule = build_operator_event_schedule(
        (),
        start_time=0.0,
        flow_durations=(0.25,),
    )
    before = _plain_state(graph)

    with pytest.raises(TNFRValueError, match="exact held-input nodal equation"):
        execute_operator_event_schedule(graph, schedule)

    assert _plain_state(graph) == before


def test_custom_nodal_integrator_keeps_internal_state_and_abstains_provenance() -> None:
    graph = _graph()
    for node in graph:
        graph.nodes[node]["delta_nfr"] = 0.25
    integrator = HeldInputIntegrator()
    graph.graph["integrator"] = integrator
    schedule = build_operator_event_schedule(
        (),
        start_time=0.0,
        flow_durations=(0.25,),
    )

    result = execute_operator_event_schedule(
        graph,
        schedule,
        include_flow_certificates=True,
    )

    assert integrator.calls == 1
    assert all(graph.nodes[node]["EPI"] == 0.0625 for node in graph)
    evidence = result.flow_interval_evidence[0]
    assert evidence.certificate is not None
    assert evidence.certificate.exact_nodal_equation_realized
    assert not evidence.integrator_provenance_certified
    assert not evidence.runtime_bound_binary64_interval_identified


def test_custom_nodal_residual_is_required_for_physical_segments() -> None:
    graph = _graph()
    graph.graph["compute_delta_nfr"] = lambda _graph: None

    class ArbitraryEPIIntegrator(AbstractIntegrator):
        def integrate(self, live_graph, *, dt, t, method, n_jobs) -> None:
            live_graph.nodes[0]["EPI"] += 1.0
            live_graph.graph["_t"] = float(t) + float(dt)

    graph.graph["integrator"] = ArbitraryEPIIntegrator()
    schedule = build_operator_event_schedule(
        (),
        start_time=0.0,
        flow_durations=(0.25,),
    )
    partition = build_physical_flow_partition(
        schedule.intervals[0],
        (0.125, 0.125),
    )
    before = _plain_state(graph)

    with pytest.raises(TNFRValueError, match="exact held-input nodal equation"):
        execute_operator_event_schedule(
            graph,
            schedule,
            physical_flow_partitions=(partition,),
        )

    assert _plain_state(graph) == before


@pytest.mark.parametrize("hostile_attribute", ("graph", "nodes"))
def test_empty_schedule_avoids_virtual_graph_attribute_access(
    hostile_attribute: str,
    ) -> None:
    class HostileGraph(nx.Graph):
        armed = False
        hostile_attribute = ""
        touches = 0

        def __getattribute__(self, name):
            if (
                name == object.__getattribute__(self, "hostile_attribute")
                and object.__getattribute__(self, "armed")
            ):
                namespace = object.__getattribute__(self, "__dict__")
                graph_mapping = dict.__getitem__(namespace, "graph")
                dict.pop(graph_mapping, "marker", None)
                object.__setattr__(
                    self,
                    "touches",
                    object.__getattribute__(self, "touches") + 1,
                )
            return object.__getattribute__(self, name)

    graph = HostileGraph()
    graph.hostile_attribute = hostile_attribute
    graph.add_node(
        0,
        EPI=0.0,
        nu_f=1.0,
        theta=0.0,
        delta_nfr=0.0,
        glyph_history=[],
    )
    graph.graph.update(_t=0.0, marker="preserved")
    schedule = build_operator_event_schedule(
        (),
        start_time=0.0,
        flow_durations=(0.0,),
    )
    graph.armed = True

    result = execute_operator_event_schedule(graph, schedule)

    object.__setattr__(graph, "armed", False)
    assert result.target_nodes == (0,)
    assert graph.graph["marker"] == "preserved"
    assert graph.touches == 0


def test_certificate_capture_uses_complete_stored_node_support() -> None:
    class FilteringGraph(nx.Graph):
        def __iter__(self):
            return (
                node
                for node in nx.Graph.__iter__(self)
                if node != "marker"
            )

    graph = FilteringGraph()
    graph.add_edges_from(((0, 1), (1, "marker")))
    graph.graph.update(
        _t=0.0,
        GAMMA={"type": "none"},
        EPI_MIN=-4.0,
        EPI_MAX=4.0,
    )
    for node in (0, 1, "marker"):
        graph.nodes[node].update(
            EPI=0.0,
            nu_f=1.0,
            theta=0.0,
            delta_nfr=0.25,
            glyph_history=[],
        )
    schedule = build_operator_event_schedule(
        (),
        start_time=0.0,
        flow_durations=(0.25,),
    )

    result = execute_operator_event_schedule(
        graph,
        schedule,
        include_flow_certificates=True,
    )

    evidence = result.flow_interval_evidence[0]
    assert evidence.certificate is not None
    assert evidence.certificate.left.nodes == (0, 1, "marker")
    assert evidence.certificate.right.nodes == (0, 1, "marker")
    assert result.target_nodes == (0, 1, "marker")


def test_integrator_method_resolution_avoids_hostile_attribute_dispatch() -> None:
    graph = _graph()
    graph.add_node(
        "marker",
        EPI=0.0,
        nu_f=1.0,
        theta=0.0,
        delta_nfr=0.25,
        glyph_history=[],
    )
    for node in graph:
        graph.nodes[node]["delta_nfr"] = 0.25

    class HostileMethodIntegrator(HeldInputIntegrator):
        def __init__(self, live_graph) -> None:
            super().__init__()
            self.live_graph = live_graph
            self.integrate_lookups = 0

        def __getattribute__(self, name):
            if name == "integrate":
                live_graph = object.__getattribute__(self, "live_graph")
                live_graph.remove_node("marker")
                object.__setattr__(
                    self,
                    "integrate_lookups",
                    object.__getattribute__(self, "integrate_lookups") + 1,
                )
            return object.__getattribute__(self, name)

    integrator = HostileMethodIntegrator(graph)
    graph.graph["integrator"] = integrator
    schedule = build_operator_event_schedule(
        (),
        start_time=0.0,
        flow_durations=(0.25,),
    )

    execute_operator_event_schedule(graph, schedule)

    assert "marker" in graph
    assert integrator.integrate_lookups == 0
    assert integrator.calls == 1


@pytest.mark.parametrize("mutation", ("class", "namespace"))
def test_pressure_callback_cannot_replace_graph_identity_surfaces(
    mutation: str,
) -> None:
    class OriginalGraph(nx.Graph):
        pass

    class ReplacementGraph(nx.Graph):
        pass

    graph = OriginalGraph(_graph())
    graph.graph.update(_graph().graph)
    original_namespace = graph.__dict__

    def refresh(live_graph: nx.Graph) -> None:
        if mutation == "class":
            live_graph.__class__ = ReplacementGraph
        else:
            live_graph.__dict__ = dict(live_graph.__dict__)

    graph.graph["compute_delta_nfr"] = refresh
    schedule = build_operator_event_schedule(
        (),
        start_time=0.0,
        flow_durations=(0.25,),
    )
    partition = build_physical_flow_partition(
        schedule.intervals[0],
        (0.125, 0.125),
    )

    with pytest.raises(
        TNFRValueError,
        match="pressure callback changed non-pressure graph state",
    ):
        execute_operator_event_schedule(
            graph,
            schedule,
            physical_flow_partitions=(partition,),
        )

    assert type(graph) is OriginalGraph
    assert graph.__dict__ is original_namespace


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


def test_partition_iterable_failure_rolls_back_graph_owned_side_effects() -> None:
    graph = _graph()
    schedule = build_operator_event_schedule(
        (),
        start_time=0.0,
        flow_durations=(0.25,),
    )
    before = _plain_state(graph)

    def mutating_partitions():
        graph.nodes[0]["EPI"] = 99.0
        graph.graph["iterator_marker"] = "changed"
        raise RuntimeError("partition iterator failed")
        yield  # pragma: no cover

    with pytest.raises(RuntimeError, match="partition iterator failed"):
        execute_operator_event_schedule(
            graph,
            schedule,
            physical_flow_partitions=mutating_partitions(),
        )

    assert _plain_state(graph) == before


def test_partition_side_effect_rolls_back_on_later_preflight_failure() -> None:
    graph = _graph()
    schedule = build_operator_event_schedule(
        (),
        start_time=float(2**53),
        flow_durations=(0.5,),
    )

    def mutate_then_finish():
        graph.graph["iterator_marker"] = "leaked"
        return
        yield  # pragma: no cover - keeps this function an iterator

    with pytest.raises(TNFRValueError):
        execute_operator_event_schedule(
            graph,
            schedule,
            physical_flow_partitions=mutate_then_finish(),
        )

    assert "iterator_marker" not in graph.graph


def test_valid_partition_iterable_cannot_commit_graph_side_effects() -> None:
    graph = _graph()
    schedule = build_operator_event_schedule(
        (),
        start_time=0.0,
        flow_durations=(0.0,),
    )
    before = _plain_state(graph)

    def mutate_then_finish():
        graph.graph["iterator_marker"] = "leaked"
        return
        yield  # pragma: no cover - keeps this function an iterator

    with pytest.raises(TNFRValueError, match="materialization changed graph state"):
        execute_operator_event_schedule(
            graph,
            schedule,
            physical_flow_partitions=mutate_then_finish(),
        )

    assert _plain_state(graph) == before


def test_context_materialization_cannot_commit_graph_side_effects() -> None:
    graph = _graph()
    schedule = build_operator_event_schedule(
        _WORD,
        start_time=0.0,
        flow_durations=(0.0,) * (len(_WORD) + 1),
    )
    before = _plain_state(graph)

    class MutatingContext(Mapping[str, Any]):
        def __getitem__(self, key: str) -> Any:
            raise KeyError(key)

        def __iter__(self) -> Iterator[str]:
            graph.graph["context_side_effect"] = "persisted"
            return iter(())

        def __len__(self) -> int:
            return 0

    with pytest.raises(
        TNFRValueError,
        match="Schedule input materialization changed graph state",
    ):
        execute_operator_event_schedule(
            graph,
            schedule,
            context=MutatingContext(),
        )

    assert _plain_state(graph) == before
    assert "context_side_effect" not in graph.graph


def test_integrator_factory_cannot_commit_resolution_side_effects() -> None:
    graph = _graph()
    before = _plain_state(graph)

    def mutating_factory(live_graph: nx.Graph) -> HeldInputIntegrator:
        live_graph.graph["factory_side_effect"] = "persisted"
        return HeldInputIntegrator()

    graph.graph["integrator"] = mutating_factory
    schedule = build_operator_event_schedule(
        (),
        start_time=0.0,
        flow_durations=(0.25,),
    )

    with pytest.raises(
        TNFRValueError,
        match="integrator resolution changed graph state",
    ):
        execute_operator_event_schedule(graph, schedule)

    assert _plain_state(graph) == before
    assert graph.graph["integrator"] is mutating_factory
    assert "factory_side_effect" not in graph.graph
    assert "_integrator_cache" not in graph.graph


def test_flow_metadata_resolution_cannot_commit_mapping_side_effects() -> None:
    graph = _graph()

    class MutatingGamma(dict[str, str]):
        def __init__(self) -> None:
            dict.__init__(self, type="none")
            self.get_calls = 0

        def get(self, key: str, default: Any = None) -> Any:
            self.get_calls += 1
            graph.graph["gamma_side_effect"] = "persisted"
            return dict.get(self, key, default)

    gamma = MutatingGamma()
    graph.graph.pop("_gamma_spec", None)
    graph.graph["GAMMA"] = gamma
    before = _plain_state(graph)
    schedule = build_operator_event_schedule(
        (),
        start_time=0.0,
        flow_durations=(0.25,),
    )

    with pytest.raises(
        TNFRValueError,
        match="flow metadata resolution changed graph state",
    ):
        execute_operator_event_schedule(graph, schedule)

    assert _plain_state(graph) == before
    assert graph.graph["GAMMA"] is gamma
    assert gamma.get_calls == 0
    assert "gamma_side_effect" not in graph.graph


def test_physical_history_subclass_iteration_is_not_dispatched() -> None:
    graph = _graph()

    class HostileHistory(list[tuple[float, float]]):
        def __iter__(self) -> Iterator[tuple[float, float]]:
            graph.graph["history_side_effect"] = "persisted"
            return list.__iter__(self)

    histories = []
    for node, data in graph.nodes(data=True):
        history = HostileHistory([(0.0, float(data["EPI"]))])
        data["epi_time_history"] = history
        histories.append(history)
    schedule = build_operator_event_schedule(
        (),
        start_time=0.0,
        flow_durations=(0.25,),
    )

    execute_operator_event_schedule(graph, schedule)

    assert "history_side_effect" not in graph.graph
    assert all(
        tuple(list.__iter__(history)) == ((0.0, 0.0),)
        for history in histories
    )
    assert all(
        tuple(graph.nodes[node]["epi_time_history"])
        == ((0.0, 0.0), (0.25, 0.05))
        for node in graph
    )


@pytest.mark.parametrize("alias_key", ("settings", "_epi_hist"))
def test_hybrid_event_log_cannot_alias_another_graph_channel(
    alias_key: str,
) -> None:
    graph = _graph()
    sink: list[dict[str, Any]] = []
    graph.graph["hybrid_event_log"] = sink
    graph.graph[alias_key] = sink
    schedule = build_operator_event_schedule(
        _WORD,
        start_time=0.0,
        flow_durations=(0.0,) * (len(_WORD) + 1),
    )

    with pytest.raises(
        TNFRValueError,
        match="must not alias another graph-owned channel",
    ):
        execute_operator_event_schedule(graph, schedule)

    assert graph.graph["hybrid_event_log"] is sink
    assert graph.graph[alias_key] is sink
    assert sink == []


def test_partition_guard_detects_equal_value_rebinding_after_id_reuse() -> None:
    graph = _graph()
    attributes = graph._node[0]
    original: dict[str, object] = {}
    attributes["payload"] = original
    original_identity = id(original)
    schedule = build_operator_event_schedule(
        (),
        start_time=0.0,
        flow_durations=(0.0,),
    )

    def rebind_payload():
        del attributes["payload"]
        replacement: dict[str, object] = {}
        attributes["payload"] = replacement
        assert id(replacement) != original_identity
        return
        yield  # pragma: no cover - keeps this function an iterator

    with pytest.raises(TNFRValueError, match="materialization changed graph state"):
        execute_operator_event_schedule(
            graph,
            schedule,
            physical_flow_partitions=rebind_payload(),
        )

    assert graph._node[0]["payload"] is original


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
        transaction_snapshot,
    ):
        assert transaction_snapshot is not None
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
        transaction_snapshot,
    ):
        assert transaction_snapshot is not None
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
