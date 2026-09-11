r"""Transactional runtime policy for finite P2 Reception/REMESH invocations.

The global P2 certificate and the executed-sequence adapter remain separate:
the former proves a restricted numeric kernel family, while the latter binds
one completed finite execution.  This module supplies the missing execution
boundary.  It revalidates the restricted live preconditions for every call,
executes one canonical finite word sequence, and constructs the existing
finite certificate before its outer graph transaction can commit.

The returned object keeps the scope of the existing sequence certificate.  In
particular, successful use of this reusable entry point does not turn one
completed invocation into evidence about a future or unobserved invocation.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from fractions import Fraction
from typing import Any

import networkx as nx

from ..errors import TNFRValueError
from ..operators._delayed_remesh_kernel import _materialize_remesh_metric
from ..operators.event_remesh_causal_runtime import (
    EventRemeshCycleExecutionSpec,
    _materialize_specs_and_partitions,
    _require_schedule_chain,
    execute_event_remesh_cycle_sequence,
)
from ..operators.event_remesh_runtime import (
    _epi_values,
    _history_signature,
    _ordered_identity_is,
    _read_only_graph_state,
    _require_canonical_nodes_surface,
    _require_graph,
    _require_read_only_graph_state,
)
from ..operators.event_runtime import _nodal_flow_capture_graph
from ..operators.event_timing import OperatorEventSchedule
from ..operators.factor_contracts import resolve_runtime_operator_factors
from ..operators.network_stage import (
    GraphTransactionSnapshot,
    _networkx_runtime_layout,
    _runtime_mapping_items,
)
from ..operators.remesh import (
    _materialize_network_remesh_configuration,
)
from ..types import Glyph
from ..utils._structural_signature import binary64_vectors_are_identical
from .binary64_p2_reception_stability import (
    P2HalfReceptionRemeshStabilityCertificate,
)
from .runtime_flow_stability import capture_nodal_flow_state
from .runtime_p2_reception_remesh_sequence import (
    ExecutedP2HalfReceptionRemeshSequenceCertificate,
    certify_executed_p2_half_reception_remesh_sequence,
)
from .runtime_p2_reception_stage import (
    _exact_capacities,
    _exact_conductance,
    _metric_ray,
    _normalized_binary64_metric,
)

__all__ = ("execute_p2_half_reception_remesh_policy_invocation",)

_POLICY_WORD = ("reception", "coherence", "recursivity")


class _LiveP2GrammarContext(Mapping[str, bool]):
    """Materialize the U1a premise from the live pair at each cycle start."""

    __slots__ = ("_graph", "_nodes")

    def __init__(self, graph: nx.Graph, nodes: tuple[Any, Any]) -> None:
        self._graph = graph
        self._nodes = nodes

    def __getitem__(self, key: str) -> bool:
        if key != "initial_epi_nonzero":
            raise KeyError(key)
        return any(value != 0.0 for value in _epi_values(self._graph, self._nodes))

    def __iter__(self) -> Iterator[str]:
        return iter(("initial_epi_nonzero",))

    def __len__(self) -> int:
        return 1


def _require_kernel(
    value: Any,
) -> P2HalfReceptionRemeshStabilityCertificate:
    if type(value) is not P2HalfReceptionRemeshStabilityCertificate:
        raise TNFRValueError(
            "kernel_certificate must be an exact P2 half-Reception "
            "certificate"
        )
    if not value._proof_fields_are_intact() or not all(
        passed for _name, passed in value.conditions
    ):
        raise TNFRValueError(
            "kernel_certificate is unsealed, tampered, or inconsistent"
        )
    return value


def _require_exact_policy_specs(
    specs: Any,
    *,
    horizon: int,
) -> tuple[EventRemeshCycleExecutionSpec, ...]:
    if type(specs) is not tuple:
        raise TNFRValueError("specs must be an exact immutable tuple")
    if len(specs) < horizon:
        raise TNFRValueError(
            "spec count must reach the active-history extinction horizon"
        )
    for index, spec in enumerate(tuple.__iter__(specs)):
        if type(spec) is not EventRemeshCycleExecutionSpec:
            raise TypeError(
                f"specs[{index}] must be an exact EventRemeshCycleExecutionSpec"
            )
        schedule = object.__getattribute__(spec, "schedule")
        if type(schedule) is not OperatorEventSchedule:
            raise TypeError(
                f"specs[{index}].schedule must be an exact "
                "OperatorEventSchedule"
            )
        partition_source = object.__getattribute__(
            spec,
            "physical_flow_partitions",
        )
        if type(partition_source) is not tuple or partition_source:
            raise TNFRValueError(
                "policy cycle partition sources must be exact empty tuples"
            )

    materialized, partition_rows = _materialize_specs_and_partitions(specs)
    if any(type(row) is not tuple or row for row in partition_rows):
        raise RuntimeError("policy partition materialization was not empty")
    return materialized


def _require_zero_flow_policy_schedules(
    graph: nx.Graph,
    specs: tuple[EventRemeshCycleExecutionSpec, ...],
) -> None:
    schedules = tuple(spec.schedule for spec in specs)
    _require_schedule_chain(graph, schedules)
    for index, schedule in enumerate(schedules):
        if type(schedule) is not OperatorEventSchedule:
            raise TypeError(
                f"specs[{index}].schedule must be an exact "
                "OperatorEventSchedule"
            )
        if schedule.cycles != 1 or schedule.operator_names != _POLICY_WORD:
            raise TNFRValueError(
                "every policy schedule must contain one canonical "
                "Reception/Coherence/Recursivity word"
            )
        if (
            type(schedule.flow_durations) is not tuple
            or len(schedule.flow_durations) != 4
            or any(
                type(value) is not float or value != 0.0
                for value in schedule.flow_durations
            )
            or type(schedule.intervals) is not tuple
            or len(schedule.intervals) != 4
            or any(
                interval.duration != 0.0
                or interval.exact_duration != Fraction(0)
                for interval in schedule.intervals
            )
            or schedule.total_flow_duration != 0.0
            or schedule.exact_total_flow_duration != Fraction(0)
        ):
            raise TNFRValueError(
                "every policy schedule must have four exact zero-duration "
                "flows"
            )


def _require_metric(
    metric_weights: Any,
    nodes: tuple[Any, Any],
    kernel: P2HalfReceptionRemeshStabilityCertificate,
) -> tuple[float, float]:
    if (
        type(metric_weights) is not tuple
        or len(metric_weights) != 2
        or any(
            type(value) is not float
            or not math.isfinite(value)
            or value <= 0.0
            for value in metric_weights
        )
    ):
        raise TNFRValueError(
            "metric_weights must be an exact positive binary64 pair"
        )
    frozen = _materialize_remesh_metric(metric_weights, nodes)
    if type(frozen) is not tuple or len(frozen) != 2:
        raise RuntimeError("P2 metric materialization lost its pair shape")
    pair = (frozen[0], frozen[1])
    if not binary64_vectors_are_identical(pair, metric_weights):
        raise RuntimeError("P2 metric materialization changed binary64 values")
    if _normalized_binary64_metric(
        pair,
        label="policy metric",
    ) != object.__getattribute__(kernel, "exact_normalized_metric"):
        raise TNFRValueError(
            "metric_weights ray does not match the P2 kernel metric"
        )
    return pair


def _require_runtime_configuration(
    graph: nx.Graph,
    *,
    layout: Any,
    kernel: P2HalfReceptionRemeshStabilityCertificate,
) -> None:
    source = object.__getattribute__(kernel, "remesh_class_certificate")
    source_configuration = object.__getattribute__(source, "configuration")
    runtime = _materialize_network_remesh_configuration(
        graph,
        _layout=layout,
    )
    if not (
        runtime.tau_local == source.tau_local
        and runtime.tau_global == source.tau_global
        and runtime.history_maxlen == source.history_maxlen
        and type(runtime.alpha) is float
        and binary64_vectors_are_identical((runtime.alpha,), (1.0,))
        and runtime.clip_mode == "hard"
        and binary64_vectors_are_identical(
            (runtime.epi_min, runtime.epi_max),
            (source_configuration.epi_min, source_configuration.epi_max),
        )
    ):
        raise TNFRValueError(
            "runtime REMESH configuration does not match the P2 source class"
        )


def _require_half_reception_factor(layout: Any) -> None:
    graph_data = {
        key: value
        for key, value in _runtime_mapping_items(layout.graph_mapping)
        if type(key) is str
    }
    raw_factors = graph_data.get("GLYPH_FACTORS")
    factors = resolve_runtime_operator_factors(
        raw_factors,
        Glyph.EN,
        graph_data,
    )
    mix = factors.get("EN_mix")
    if (
        type(mix) is not float
        or not binary64_vectors_are_identical((mix,), (0.5,))
    ):
        raise TNFRValueError(
            "runtime Reception mix must be exact binary64 one half"
        )


def _require_live_p2_state(
    graph: nx.Graph,
    *,
    nodes: tuple[Any, Any],
    metric: tuple[float, float],
    kernel: P2HalfReceptionRemeshStabilityCertificate,
) -> tuple[float, ...]:
    capture_graph, capture_nodes = _nodal_flow_capture_graph(graph)
    if not _ordered_identity_is(capture_nodes, nodes):
        raise TNFRValueError("P2 capture changed ordered node support")
    snapshot = capture_nodal_flow_state(capture_graph, nodes=capture_nodes)
    capacities = _exact_capacities(snapshot)
    conductance = _exact_conductance(snapshot)
    diffusion_metric = _metric_ray(conductance, capacities)
    source_metric = object.__getattribute__(kernel, "exact_normalized_metric")
    declared_metric = _normalized_binary64_metric(
        metric,
        label="policy metric",
    )
    if diffusion_metric != source_metric or declared_metric != source_metric:
        raise TNFRValueError(
            "live P2 diffusion metric does not match the kernel metric"
        )

    source = object.__getattribute__(kernel, "remesh_class_certificate")
    current = _epi_values(graph, nodes)
    if any(
        value < source.configuration.epi_min
        or value > source.configuration.epi_max
        for value in current
    ):
        raise TNFRValueError("current EPI pair is outside the P2 source interval")

    _present, _history_object, history = _history_signature(graph, nodes)
    required_incoming = source.required_history_length - 1
    if not (
        required_incoming <= len(history) <= source.history_maxlen
    ):
        raise TNFRValueError(
            "incoming REMESH history or capacity cannot serve the first cycle"
        )
    active_incoming = history[-source.tau_global :]
    if len(active_incoming) != source.tau_global or any(
        value < source.configuration.epi_min
        or value > source.configuration.epi_max
        for row in active_incoming
        for value in row
    ):
        raise TNFRValueError(
            "active incoming REMESH history leaves the P2 source interval"
        )
    return current


def execute_p2_half_reception_remesh_policy_invocation(
    graph: nx.Graph,
    kernel_certificate: P2HalfReceptionRemeshStabilityCertificate,
    specs: tuple[EventRemeshCycleExecutionSpec, ...],
    *,
    metric_weights: tuple[float, float],
    suppress_birth_warnings: bool = False,
) -> ExecutedP2HalfReceptionRemeshSequenceCertificate:
    """Execute one fully revalidated finite P2 policy invocation atomically.

    A fresh outer graph snapshot precedes every inspection or materialization
    of ``specs`` and ``metric_weights``. Static preflight checks the entry EPI,
    and a live mapping rederives grammar admission at every cycle start. Causal
    execution without an affine telescope and finite P2 certification all remain
    inside that transaction. Any failure restores graph-owned state to the entry
    boundary.
    """

    graph = _require_graph(graph)
    transaction = GraphTransactionSnapshot(graph)
    try:
        preparation_state = _read_only_graph_state(graph)
        kernel = _require_kernel(kernel_certificate)
        if type(suppress_birth_warnings) is not bool:
            raise TypeError("suppress_birth_warnings must be a bool")

        _require_canonical_nodes_surface(graph)
        layout = _networkx_runtime_layout(graph)
        if layout.directed:
            raise TNFRValueError(
                "P2 policy preflight requires undirected graph support"
            )
        live_nodes = tuple(node for node, _data in layout.node_data)
        source_nodes = object.__getattribute__(kernel, "node_order")
        if (
            len(live_nodes) != 2
            or type(source_nodes) is not tuple
            or len(source_nodes) != 2
            or not _ordered_identity_is(live_nodes, source_nodes)
        ):
            raise TNFRValueError(
                "live graph must have exactly the ordered P2 kernel nodes"
            )
        nodes = (live_nodes[0], live_nodes[1])

        materialized_specs = _require_exact_policy_specs(
            specs,
            horizon=kernel.active_history_extinction_horizon,
        )
        _require_zero_flow_policy_schedules(graph, materialized_specs)
        metric = _require_metric(metric_weights, nodes, kernel)
        _require_runtime_configuration(
            graph,
            layout=layout,
            kernel=kernel,
        )
        _require_half_reception_factor(layout)
        current_epi = _require_live_p2_state(
            graph,
            nodes=nodes,
            metric=metric,
            kernel=kernel,
        )
        if not any(value != 0.0 for value in current_epi):
            raise TNFRValueError(
                "P2 Reception policy requires pre-existing nonzero EPI form"
            )
        _require_read_only_graph_state(
            graph,
            preparation_state,
            boundary="p2_policy_static_preflight",
        )

        execution = execute_event_remesh_cycle_sequence(
            graph,
            materialized_specs,
            metric_weights=metric,
            context=_LiveP2GrammarContext(graph, nodes),
            suppress_birth_warnings=suppress_birth_warnings,
            require_runtime_telescope=False,
        )
        post_execution_state = _read_only_graph_state(graph)
        certificate = certify_executed_p2_half_reception_remesh_sequence(
            kernel,
            execution,
        )
        if (
            type(certificate)
            is not ExecutedP2HalfReceptionRemeshSequenceCertificate
            or object.__getattribute__(certificate, "kernel_certificate")
            is not kernel
            or object.__getattribute__(certificate, "execution") is not execution
        ):
            raise TNFRValueError(
                "P2 policy post-certification returned an invalid certificate"
            )
        if not certificate._proof_fields_are_intact() or not all(
            passed
            for _name, passed in object.__getattribute__(
                certificate,
                "conditions",
            )
        ):
            raise TNFRValueError(
                "P2 policy post-certification returned an invalid certificate"
            )
        _require_read_only_graph_state(
            graph,
            post_execution_state,
            boundary="p2_policy_postcertification",
        )
        return certificate
    except BaseException as failure:
        transaction.restore_after_failure(graph, failure)
        raise
