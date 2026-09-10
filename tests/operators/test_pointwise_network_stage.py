"""Executable contract tests for pointwise all-target stages."""

from __future__ import annotations

from collections import deque
from copy import deepcopy
import math
import threading
import warnings
from typing import Any

import networkx as nx
import pytest

from tnfr.alias import get_attr
from tnfr.constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_THETA, ALIAS_VF
from tnfr.glyph_history import push_glyph
from tnfr.operators.definitions import (
    Contraction,
    Emission,
    Expansion,
    Mutation,
    Silence,
)
from tnfr.operators.factor_contracts import GlyphFactorValidationError
from tnfr.operators.network_stage import (
    OPERATOR_MAJOR_GAUSS_SEIDEL,
    STAGE_CONTRACT_KEY,
    STAGE_SCHEDULE_KEY,
    TWO_PHASE_JACOBI,
    execute_operator_major_stage,
    execute_pointwise_stage,
)
from tnfr.operators.preconditions import OperatorPreconditionError
from tnfr.operators.stage_contracts import stage_schedule_metadata
from tnfr.types import Glyph, real_scalar_epi


_OPERATOR_TYPES = {
    Glyph.AL: Emission,
    Glyph.SHA: Silence,
    Glyph.VAL: Expansion,
    Glyph.NUL: Contraction,
    Glyph.ZHIR: Mutation,
}
_FACTORS = {
    Glyph.AL: {"AL_boost": 0.15},
    Glyph.SHA: {"SHA_vf_factor": 0.5},
    Glyph.VAL: {"VAL_scale": 1.25},
    Glyph.NUL: {"NUL_scale": 0.5},
    Glyph.ZHIR: {"ZHIR_theta_shift_factor": 0.5},
}


def _graph(*, glyph: Glyph) -> nx.Graph:
    graph = nx.path_graph(3)
    graph.graph.update(
        GLYPH_FACTORS=deepcopy(_FACTORS[glyph]),
        RANDOM_SEED=17,
        EDGE_AWARE_ENABLED=True,
        EPI_MIN=-1.0,
        EPI_MAX=1.0,
        ZHIR_THRESHOLD_XI=0.1,
    )
    for node in graph:
        epi = 0.25 + 0.1 * node
        graph.nodes[node].update(
            {
                ALIAS_EPI[0]: epi,
                ALIAS_VF[0]: 1.0 + 0.25 * node,
                ALIAS_DNFR[0]: 0.2 + 0.05 * node,
                ALIAS_THETA[0]: 0.1 * node,
                "EPI_kind": "wave",
                "glyph_history": (
                    ["AL", "IL", "OZ"]
                    if glyph is Glyph.ZHIR
                    else ["AL", "IL"]
                ),
            }
        )
        if glyph is Glyph.ZHIR:
            graph.nodes[node]["epi_history"] = [0.0, 0.1, epi]
    return graph


def _primary_state(graph: nx.Graph, glyph: Glyph) -> dict[Any, tuple[float, ...]]:
    state: dict[Any, tuple[float, ...]] = {}
    for node, data in graph.nodes(data=True):
        epi = float(real_scalar_epi(get_attr(data, ALIAS_EPI)))
        vf = float(get_attr(data, ALIAS_VF))
        dnfr = float(get_attr(data, ALIAS_DNFR))
        if glyph is Glyph.AL:
            state[node] = (epi,)
        elif glyph is Glyph.SHA:
            state[node] = (vf,)
        elif glyph is Glyph.VAL:
            state[node] = (epi, vf)
        elif glyph is Glyph.ZHIR:
            state[node] = (float(get_attr(data, ALIAS_THETA)),)
        else:
            state[node] = (epi, vf, dnfr)
    return state


def _plain_state(
    graph: nx.Graph, *, omit_graph_keys: frozenset[str] = frozenset()
) -> tuple[object, ...]:
    return (
        tuple(
            (node, deepcopy(dict(data)))
            for node, data in graph.nodes(data=True)
        ),
        tuple(
            (left, right, deepcopy(dict(data)))
            for left, right, data in graph.edges(data=True)
        ),
        deepcopy(
            {
                key: value
                for key, value in graph.graph.items()
                if key not in omit_graph_keys
            }
        ),
        (
            hasattr(graph, "_last_operator_applied"),
            getattr(graph, "_last_operator_applied", None),
        ),
    )


def _assert_exact_schedule(
    graph: nx.Graph,
    operator: Any,
    *,
    schedule: str,
    count: int,
) -> None:
    assert graph.graph[STAGE_SCHEDULE_KEY] == {
        "operator": operator.name,
        "glyph": operator.glyph.value,
        "schedule": schedule,
        "nodes_processed": count,
    }
    assert graph.graph[STAGE_CONTRACT_KEY] == stage_schedule_metadata(
        operator,
        observed_schedule=schedule,
    )


@pytest.mark.parametrize("glyph", tuple(_OPERATOR_TYPES))
def test_pointwise_primary_state_is_exactly_target_order_invariant(
    glyph: Glyph,
) -> None:
    forward = _graph(glyph=glyph)
    reverse = _graph(glyph=glyph)
    operator_type = _OPERATOR_TYPES[glyph]

    forward_result = execute_pointwise_stage(
        forward, operator_type(), (0, 1, 2)
    )
    reverse_result = execute_pointwise_stage(
        reverse, operator_type(), (2, 1, 0)
    )

    assert _primary_state(forward, glyph) == _primary_state(reverse, glyph)
    assert forward_result.schedule == TWO_PHASE_JACOBI
    assert reverse_result.schedule == TWO_PHASE_JACOBI
    _assert_exact_schedule(
        forward, operator_type(), schedule=TWO_PHASE_JACOBI, count=3
    )
    _assert_exact_schedule(
        reverse, operator_type(), schedule=TWO_PHASE_JACOBI, count=3
    )
    for graph in (forward, reverse):
        for node in graph:
            assert tuple(graph.nodes[node]["glyph_history"])[-1] == glyph.value


@pytest.mark.parametrize("glyph", tuple(_OPERATOR_TYPES))
def test_pointwise_runtime_matches_the_existing_single_target_kernel(
    glyph: Glyph,
) -> None:
    pointwise = _graph(glyph=glyph)
    sequential = _graph(glyph=glyph)
    operator_type = _OPERATOR_TYPES[glyph]

    execute_pointwise_stage(pointwise, operator_type(), (0, 1, 2))
    execute_operator_major_stage(sequential, operator_type(), (0, 1, 2))

    assert _primary_state(pointwise, glyph) == _primary_state(sequential, glyph)
    if glyph is Glyph.NUL:
        assert pointwise.graph["nul_densification_log"] == (
            sequential.graph["nul_densification_log"]
        )


@pytest.mark.parametrize(
    ("glyph", "timestamp_key"),
    ((Glyph.AL, "emission_timestamp"), (Glyph.SHA, "latency_start_time")),
)
def test_al_and_sha_share_one_timestamp_across_the_stage(
    glyph: Glyph, timestamp_key: str
) -> None:
    graph = _graph(glyph=glyph)

    execute_pointwise_stage(graph, _OPERATOR_TYPES[glyph](), (2, 0, 1))

    timestamps = {graph.nodes[node][timestamp_key] for node in graph}
    assert len(timestamps) == 1


def test_nul_and_edge_intervention_logs_retain_requested_target_order() -> None:
    nul_graph = _graph(glyph=Glyph.NUL)
    nul_graph.graph["nul_densification_log"] = [{"node": "existing"}]
    execute_pointwise_stage(nul_graph, Contraction(), (2, 0, 1))

    assert [
        event["node"] for event in nul_graph.graph["nul_densification_log"]
    ] == ["existing", 2, 0, 1]

    val_graph = _graph(glyph=Glyph.VAL)
    val_graph.graph["GLYPH_FACTORS"] = {"VAL_scale": 2.0}
    for node, epi in enumerate((0.75, 0.8, 0.9)):
        val_graph.nodes[node][ALIAS_EPI[0]] = epi
    execute_pointwise_stage(val_graph, Expansion(), (1, 2, 0))

    interventions = val_graph.graph["edge_aware_interventions"]
    assert [event["node"] for event in interventions] == [1, 2, 0]
    assert [event["glyph"] for event in interventions] == ["VAL"] * 3


def test_grammar_replacement_falls_back_to_an_exact_gs_diagnostic() -> None:
    graph = _graph(glyph=Glyph.VAL)
    for node in graph:
        graph.nodes[node]["glyph_history"] = ["VAL", "VAL"]
    operator = Expansion()

    result = execute_pointwise_stage(graph, operator, (0, 1, 2))

    assert result.schedule == OPERATOR_MAJOR_GAUSS_SEIDEL
    _assert_exact_schedule(
        graph,
        operator,
        schedule=OPERATOR_MAJOR_GAUSS_SEIDEL,
        count=3,
    )
    contract = graph.graph[STAGE_CONTRACT_KEY]
    assert contract["declared_schedule"] == TWO_PHASE_JACOBI
    assert contract["schedule_matches_contract"] is False
    assert contract["executed_two_phase_contract_complete"] is False
    assert all(
        tuple(graph.nodes[node]["glyph_history"])[-1] == "IL"
        for node in graph
    )


def test_empty_stage_calls_pressure_callback_once_and_records_its_contract() -> None:
    graph = nx.Graph()
    callback_calls: list[nx.Graph] = []

    def pressure_callback(subject: nx.Graph) -> None:
        callback_calls.append(subject)
        subject.graph["empty_pressure_refresh"] = True

    operator = Emission()
    result = execute_pointwise_stage(
        graph,
        operator,
        (),
        compute_delta_nfr=pressure_callback,
    )

    assert callback_calls == [graph]
    assert graph.graph["empty_pressure_refresh"] is True
    assert result.nodes_processed == 0
    assert result.schedule == TWO_PHASE_JACOBI
    _assert_exact_schedule(
        graph, operator, schedule=TWO_PHASE_JACOBI, count=0
    )


def test_warning_promoted_to_error_rolls_back_prior_lifecycle_commits() -> None:
    graph = _graph(glyph=Glyph.AL)
    graph.graph["MAX_SILENCE_DURATION"] = 2.0
    graph.nodes[1].update(
        latent=True,
        latency_start_time="2030-01-01T00:00:00+00:00",
        preserved_epi=graph.nodes[1][ALIAS_EPI[0]],
        silence_duration=3.0,
        was_initial_on_silence=False,
    )
    before = _plain_state(graph)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(UserWarning, match="extended silence"):
            execute_pointwise_stage(graph, Emission(), (0, 1, 2))

    assert _plain_state(graph) == before


class _RejectingMonitor:
    def __init__(self) -> None:
        self.events: list[tuple[str, Any]] = []
        self.pending: Any = None

    def before_operator(self, graph: nx.Graph, node: Any) -> None:
        self.pending = node
        self.events.append(("before", node))

    def after_operator(self, graph: nx.Graph, node: Any, operator: str) -> None:
        self.events.append(("after", node))
        if node == 1:
            graph.graph["monitor_side_effect"] = operator
            graph.nodes[node]["monitor_side_effect"] = True
            raise RuntimeError("monitor rejected second target")
        self.pending = None

    def discard_pending_operator(self) -> None:
        self.pending = None


def test_monitor_failure_rolls_back_graph_and_monitor_state() -> None:
    graph = _graph(glyph=Glyph.SHA)
    monitor = _RejectingMonitor()
    graph.graph["integrity_monitor"] = monitor
    before = _plain_state(
        graph, omit_graph_keys=frozenset({"integrity_monitor"})
    )
    monitor_before = deepcopy(vars(monitor))

    with pytest.raises(RuntimeError, match="monitor rejected second target"):
        execute_pointwise_stage(graph, Silence(), (0, 1, 2))

    assert graph.graph["integrity_monitor"] is monitor
    assert _plain_state(
        graph, omit_graph_keys=frozenset({"integrity_monitor"})
    ) == before
    assert vars(monitor) == monitor_before


def test_pressure_callback_failure_rolls_back_state_topology_and_metadata() -> None:
    graph = _graph(glyph=Glyph.NUL)
    before = _plain_state(graph)
    callback_calls: list[nx.Graph] = []

    def rejected_callback(subject: nx.Graph) -> None:
        callback_calls.append(subject)
        subject.nodes[0][ALIAS_EPI[0]] = 99.0
        subject.edges[0, 1]["callback_side_effect"] = True
        subject.graph["callback_side_effect"] = True
        subject.add_node("callback-added", EPI=99.0)
        raise RuntimeError("pressure callback rejected stage")

    with pytest.raises(RuntimeError, match="pressure callback rejected stage"):
        execute_pointwise_stage(
            graph,
            Contraction(),
            (0, 1, 2),
            compute_delta_nfr=rejected_callback,
        )

    assert callback_calls == [graph]
    assert _plain_state(graph) == before


@pytest.mark.parametrize("location", ("graph", "node", "edge"))
def test_stage_supports_ordinary_lock_metadata_without_replacing_it(
    location: str,
) -> None:
    graph = _graph(glyph=Glyph.VAL)
    user_lock = threading.Lock()
    if location == "graph":
        graph.graph["user_lock"] = user_lock
    elif location == "node":
        graph.nodes[1]["user_lock"] = user_lock
    else:
        graph.edges[0, 1]["user_lock"] = user_lock

    execute_pointwise_stage(graph, Expansion(), (0, 1, 2))

    if location == "graph":
        observed = graph.graph["user_lock"]
    elif location == "node":
        observed = graph.nodes[1]["user_lock"]
    else:
        observed = graph.edges[0, 1]["user_lock"]
    assert observed is user_lock


def test_hostile_callable_introspection_is_not_run_before_failed_preflight() -> None:
    graph = _graph(glyph=Glyph.ZHIR)
    graph.nodes[2]["epi_history"] = [0.0, 0.1]

    class HostileCallback:
        __slots__ = ("calls", "graph")

        def __init__(self, live_graph: nx.Graph) -> None:
            object.__setattr__(self, "calls", 0)
            object.__setattr__(self, "graph", live_graph)

        def __getattribute__(self, name: str) -> Any:
            if name == "__dict__":
                live_graph = object.__getattribute__(self, "graph")
                live_graph.graph["hostile_introspection_marker"] = True
            return object.__getattribute__(self, name)

        def __call__(self, _graph: nx.Graph) -> None:
            object.__setattr__(
                self,
                "calls",
                object.__getattribute__(self, "calls") + 1,
            )

    callback = HostileCallback(graph)
    graph.graph["compute_delta_nfr"] = callback
    graph_keys_before = tuple(graph.graph)
    nodes_before = tuple(
        (node, deepcopy(dict(data))) for node, data in graph.nodes(data=True)
    )

    with pytest.raises(
        OperatorPreconditionError, match="signed dEPI/dt > xi"
    ):
        execute_pointwise_stage(
            graph,
            Mutation(),
            (0, 1, 2),
            compute_delta_nfr=callback,
        )

    assert tuple(graph.graph) == graph_keys_before
    assert "hostile_introspection_marker" not in graph.graph
    assert object.__getattribute__(callback, "calls") == 0
    assert tuple(
        (node, dict(data)) for node, data in graph.nodes(data=True)
    ) == nodes_before


def test_zhir_rejects_late_target_evidence_before_any_target_commit() -> None:
    graph = _graph(glyph=Glyph.ZHIR)
    graph.nodes[2]["epi_history"] = [0.0, 0.1]
    before = _plain_state(graph)

    with pytest.raises(
        OperatorPreconditionError, match="signed dEPI/dt > xi"
    ):
        execute_pointwise_stage(graph, Mutation(), (0, 1, 2))

    assert _plain_state(graph) == before


def test_zhir_bounded_histories_use_monotonic_steps_and_one_event_per_target(
) -> None:
    graph = _graph(glyph=Glyph.ZHIR)
    for node in graph:
        graph.nodes[node]["glyph_history"] = deque(
            ["IL", "OZ"], maxlen=2
        )
        graph.nodes[node]["_operator_step"] = 10

    execute_pointwise_stage(
        graph, Mutation(), (0, 1, 2), tau=0.01, window=2
    )

    first_events = graph.graph["zhir_bifurcation_events"]
    assert [event["node"] for event in first_events] == [0, 1, 2]
    assert [event["timestamp"] for event in first_events] == [11, 11, 11]
    assert [event["event_index"] for event in first_events] == [1, 2, 3]
    for node in graph:
        assert graph.nodes[node]["_operator_step"] == 11
        assert list(graph.nodes[node]["glyph_history"]) == ["OZ", "ZHIR"]
        push_glyph(graph.nodes[node], "IL", 2)
        push_glyph(graph.nodes[node], "IL", 2)
        push_glyph(graph.nodes[node], "OZ", 2)

    execute_pointwise_stage(
        graph, Mutation(), (2, 1, 0), tau=0.01, window=2
    )

    events = graph.graph["zhir_bifurcation_events"]
    assert [event["node"] for event in events] == [0, 1, 2, 2, 1, 0]
    assert [event["event_index"] for event in events] == list(range(1, 7))
    for node in graph:
        node_events = [event for event in events if event["node"] == node]
        assert [event["timestamp"] for event in node_events] == [11, 15]
        assert graph.nodes[node]["_operator_step"] == 15


def test_zhir_history_resize_does_not_rewind_the_proposed_operator_step() -> None:
    graph = _graph(glyph=Glyph.ZHIR)

    execute_pointwise_stage(
        graph, Mutation(), (0,), tau=0.01, window=2
    )

    assert graph.nodes[0]["_operator_step"] == 4
    assert list(graph.nodes[0]["glyph_history"]) == ["OZ", "ZHIR"]
    assert graph.graph["zhir_bifurcation_events"] == [
        {
            "node": 0,
            "d2_epi": pytest.approx(0.05),
            "tau": 0.01,
            "timestamp": 4,
            "event_index": 1,
        }
    ]


def test_zhir_tau_override_is_bound_to_metrics_and_event() -> None:
    graph = _graph(glyph=Glyph.ZHIR)
    graph.graph.update(
        BIFURCATION_THRESHOLD_TAU=1.0,
        COLLECT_OPERATOR_METRICS=True,
    )

    execute_pointwise_stage(graph, Mutation(), (0,), tau=0.01)

    metrics = graph.graph["operator_metrics"][-1]
    assert metrics["d2epi"] == pytest.approx(0.05)
    assert metrics["bifurcation_threshold_tau"] == 0.01
    assert metrics["bifurcation_potential"] is True
    assert metrics["bifurcation_triggered"] is True
    assert graph.nodes[0]["_zhir_tau"] == 0.01
    assert graph.graph["zhir_bifurcation_events"][0]["tau"] == 0.01


def test_zhir_strict_preflight_and_late_factor_rejection_are_read_only() -> None:
    graph = _graph(glyph=Glyph.ZHIR)
    graph.graph.update(
        VALIDATE_OPERATOR_PRECONDITIONS=True,
        GLYPH_FACTORS={"ZHIR_theta_shift_factor": 0.0},
    )
    before = _plain_state(graph)

    with pytest.raises(
        GlyphFactorValidationError, match="ZHIR_theta_shift_factor"
    ):
        execute_pointwise_stage(graph, Mutation(), (0, 1, 2))

    assert _plain_state(graph) == before
    assert all("_mutation_context" not in graph.nodes[node] for node in graph)


def test_nonbifurcating_zhir_replaces_stale_current_flags() -> None:
    graph = _graph(glyph=Glyph.ZHIR)
    graph.graph["COLLECT_OPERATOR_METRICS"] = True
    data = graph.nodes[0]
    data[ALIAS_EPI[0]] = 0.4
    data["epi_history"] = [0.0, 0.2, 0.4]
    data["_operator_step"] = 3
    data["_zhir_bifurcation_potential"] = True
    data["_zhir_d2epi"] = 9.0
    data["_zhir_tau"] = 0.01
    data["_zhir_operator_step"] = 3
    graph.graph["zhir_bifurcation_events"] = [
        {
            "node": 0,
            "d2_epi": 9.0,
            "tau": 0.01,
            "timestamp": 3,
            "event_index": 1,
        }
    ]

    execute_pointwise_stage(graph, Mutation(), (0,), tau=0.1)

    assert data["_zhir_bifurcation_potential"] is False
    assert data["_zhir_d2epi"] == pytest.approx(0.0)
    assert data["_zhir_tau"] == 0.1
    assert data["_zhir_operator_step"] == 4
    assert len(graph.graph["zhir_bifurcation_events"]) == 1
    metrics = graph.graph["operator_metrics"][-1]
    assert metrics["bifurcation_potential"] is False
    assert metrics["bifurcation_triggered"] is False


def test_fixed_zhir_metrics_ignore_stale_dynamic_branch_telemetry() -> None:
    graph = _graph(glyph=Glyph.ZHIR)
    graph.graph.update(
        GLYPH_FACTORS={"ZHIR_theta_shift": 0.2},
        COLLECT_OPERATOR_METRICS=True,
    )
    graph.nodes[0].update(
        _zhir_regime_changed=True,
        _zhir_regime_before=2,
        _zhir_regime_after=3,
        _zhir_theta_before=math.pi,
        _zhir_theta_after=1.5 * math.pi,
    )

    execute_pointwise_stage(graph, Mutation(), (0,), tau=1.0)

    metrics = graph.graph["operator_metrics"][-1]
    assert graph.nodes[0]["_zhir_fixed_mode"] is True
    assert graph.nodes[0]["_zhir_regime_changed"] is True
    assert metrics["transformation_mode"] == "fixed"
    assert metrics["regime_changed"] is False
    assert metrics["theta_regime_before"] == 0
    assert metrics["theta_regime_after"] == 0


@pytest.mark.parametrize("tau", (0.01, 1.0))
def test_zhir_variant_creation_mode_is_rejected_before_writes(
    tau: float,
) -> None:
    graph = _graph(glyph=Glyph.ZHIR)
    graph.graph["ZHIR_BIFURCATION_MODE"] = "variant_creation"
    before = _plain_state(graph)

    with pytest.raises(
        OperatorPreconditionError,
        match="supports 'detection' only; use THOL",
    ):
        execute_pointwise_stage(graph, Mutation(), (0, 1, 2), tau=tau)

    assert _plain_state(graph) == before
