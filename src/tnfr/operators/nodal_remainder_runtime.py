"""Explicit carried-EPI execution of restricted canonical event schedules.

This opt-in owner reads canonical pressure from the visible graph, preserves
the exact numerical remainder across EPI-preserving events and continuations,
and rolls back all graph-owned state on failure. DefaultIntegrator and the
existing visible-EPI custom-integrator contract are unchanged.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from fractions import Fraction
import math
from typing import Any

import networkx as nx

from ..alias import set_attr
from ..constants.aliases import ALIAS_D2EPI, ALIAS_DEPI, ALIAS_EPI, ALIAS_EPI_KIND, ALIAS_THETA
from ..dynamics._euler_kernel import (
    NodalRemainderState, NodalRemainderStep, advance_nodal_remainder, initialize_nodal_remainder,
)
from ..dynamics.dnfr import default_compute_delta_nfr
from ..errors import TNFRValueError
from ..physics.forcing_realization import _runtime_weights
from ..physics.nodal_remainder import NodalRemainderSequence, observe_nodal_remainder_sequence
from ..physics.runtime_flow_stability import NodalFlowStateSnapshot
from ..utils._structural_signature import proof_stamps_are_identical
from .event_timing import OperatorEventSchedule, PhysicalFlowPartition, StructuralFlowInterval
from . import event_runtime as events
from .network_stage import (
    GraphTransactionSnapshot, NetworkStageResult, _networkx_runtime_layout,
    _runtime_class_mro, _set_runtime_mapping_item,
)
from .word_execution import execute_network_operator_stage

__all__ = [
    "NodalRemainderRuntimeBinding", "ExecutedNodalRemainderFlow", "ExecutedNodalRemainderEvent",
    "NodalRemainderEventExecution", "execute_nodal_remainder_event_schedule",
]

_STATE_KEY = "_nodal_remainder_runtime_state"
_ALLOWED_NAMES = frozenset(("coupling", "coherence", "silence"))
_FLOW_OWNER = object()


def _intact(value, version):
    try:
        return bool(value._proof_stamp and proof_stamps_are_identical(
            value._proof_stamp, events._sealed_dataclass_stamp(value, version),
        ))
    except BaseException:
        return False


@dataclass(frozen=True)
class NodalRemainderRuntimeBinding:
    """Graph-owned numerical state, fixed support/chart, and live clock."""

    nodes: tuple[Any, ...]
    time: float
    state: NodalRemainderState
    graph_identity: int
    support_signature: tuple[Any, ...]
    kind_signature: tuple[Any, ...]
    binding_identity: int = 0
    _proof_stamp: tuple[Any, ...] = field(default=(), repr=False, compare=False)

    @property
    def intact(self):
        try:
            return self.binding_identity == id(self) and _intact(self, "nodal_remainder_binding_v1")
        except BaseException:
            return False


def _new_binding(graph, nodes, time, state, support, kind):
    value = NodalRemainderRuntimeBinding(nodes, time, state, id(graph), support, kind)
    object.__setattr__(value, "binding_identity", id(value))
    object.__setattr__(value, "_proof_stamp", events._sealed_dataclass_stamp(value, "nodal_remainder_binding_v1"))
    return value


@dataclass(frozen=True, slots=True)
class NodalRemainderPressureRefresh:
    time: float
    boundary: str
    before_snapshot: NodalFlowStateSnapshot
    after_snapshot: NodalFlowStateSnapshot
    binding: NodalRemainderRuntimeBinding
    checks: tuple[tuple[str, bool], ...]
    callback_identity: int


@dataclass(frozen=True, slots=True)
class ExecutedNodalRemainderFlow:
    parent_interval_index: int
    segment_index: int
    segment: StructuralFlowInterval
    before_snapshot: NodalFlowStateSnapshot
    after_snapshot: NodalFlowStateSnapshot
    before_binding: NodalRemainderRuntimeBinding
    after_binding: NodalRemainderRuntimeBinding
    step: NodalRemainderStep
    normalized_weights: tuple[tuple[str, Fraction], ...]
    epi_weight: Fraction
    pressure_refresh_checks: tuple[tuple[str, bool], ...]
    phase_before: tuple[float, ...]
    phase_after: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class ExecutedNodalRemainderEvent:
    event: events.ExecutedOperatorEvent
    stage: NetworkStageResult
    before_binding: NodalRemainderRuntimeBinding
    after_binding: NodalRemainderRuntimeBinding
    before_snapshot: NodalFlowStateSnapshot
    after_snapshot: NodalFlowStateSnapshot
    phase_before: tuple[float, ...]
    phase_after: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class NodalRemainderEventExecution:
    """One sealed finite live execution; no solver or future stability claim."""

    schedule: OperatorEventSchedule
    nodes: tuple[Any, ...]
    initial_binding: NodalRemainderRuntimeBinding
    final_binding: NodalRemainderRuntimeBinding
    flows: tuple[ExecutedNodalRemainderFlow, ...]
    events: tuple[ExecutedNodalRemainderEvent, ...]
    prefix_budget: NodalRemainderSequence | None
    refresh_records: tuple[NodalRemainderPressureRefresh, ...]
    grammar_context: tuple[tuple[str, bool], ...]
    exact_reconstructed_change: tuple[Fraction, ...]
    exact_nodal_area: tuple[Fraction, ...]
    exact_nodal_balance_residual: tuple[Fraction, ...]
    execution_identity: int = 0
    _proof_stamp: tuple[Any, ...] = field(default=(), repr=False, compare=False)

    @property
    def runtime_provenance_certified(self):
        try:
            return bool(
                self.execution_identity == id(self)
                and self.initial_binding.intact and self.final_binding.intact
                and not any(self.exact_nodal_balance_residual)
                and _intact(self, "nodal_remainder_event_execution_v1")
            )
        except BaseException:
            return False

    @property
    def solver_accuracy_certified(self):
        return False

    @property
    def future_or_repeated_schedule_stability_certified(self):
        return False

    @property
    def pressure_readout(self):
        return "canonical_default_pressure_from_visible_epi"


def _scalar(data, aliases, label, default=0.0):
    # Captures can occur outside a whole-graph read-only observation. Resolve
    # exact stored keys without dispatching a mapping subclass's read hooks.
    # A string-subclass core key would otherwise be read by ordinary aliases
    # but silently missed here; reject it through base string inspection.
    for key, _value in events._runtime_mapping_items(data):
        if type(key) is not str and any(owner is str for owner in _runtime_class_mro(type(key))):
            if any(str.__eq__(key, alias) is True for alias in aliases):
                raise TNFRValueError("runtime scalar aliases must use exact string keys")
    entries = events._selected_mapping_entries(data, aliases)
    value = entries[0][1] if entries else default
    if type(value) is not float or not math.isfinite(value):
        raise TNFRValueError(f"{label} must be an actual finite scalar float")
    return value


def _chart(graph):
    layout = _networkx_runtime_layout(graph)
    nodes = tuple(node for node, _data in layout.node_data)
    if not nodes:
        raise TNFRValueError("carried nodal execution requires nonempty ordered support")
    if any(left == right for left, right, *_rest in layout.edges):
        raise TNFRValueError("carried nodal execution requires loop-free simple support")
    epi = tuple(_scalar(data, ALIAS_EPI, "EPI") for _node, data in layout.node_data)
    for _node, data in layout.node_data:
        for name in ALIAS_EPI_KIND:
            if name in data and type(data[name]) is not str:
                raise TNFRValueError("EPI kind metadata must remain a string")
    support = events._edge_state_signature(graph, layout=layout)
    kind = events._node_alias_state_signature(graph, ALIAS_EPI_KIND, node_data=layout.node_data)
    return nodes, epi, support, kind


def _same_nodes(left, right):
    return len(left) == len(right) and all(a is b for a, b in zip(left, right, strict=True))


def _require_binding(graph, binding, *, lower, upper):
    _configuration(graph, lower, upper)
    if type(binding) is not NodalRemainderRuntimeBinding or not binding.intact:
        raise TNFRValueError("the persistent nodal remainder binding is invalid")
    if graph.graph.get(_STATE_KEY) is not binding or binding.graph_identity != id(graph):
        raise TNFRValueError("the nodal remainder binding belongs to a different graph or owner")
    nodes, epi, support, kind = _chart(graph)
    if (not _same_nodes(nodes, binding.nodes) or epi != binding.state.epi
            or support != binding.support_signature or kind != binding.kind_signature
            or binding.state.epi_lower != lower or binding.state.epi_upper != upper):
        raise TNFRValueError("the live graph no longer matches its carried scalar chart/support/band")
    binding.state.exact_epi
    events._require_runtime_clock(graph, binding.time, boundary="nodal_remainder_binding")


def _owned_binding_commit(graph, before, after):
    if before is None:
        if _STATE_KEY in graph.graph:
            raise TNFRValueError("initial carry attachment would replace an existing binding")
    elif graph.graph.get(_STATE_KEY) is not before:
        raise TNFRValueError("the carry binding changed before its executor-owned commit")
    layout = _networkx_runtime_layout(graph)
    opaque = events._canonical_graph_storage_references(graph, layout)
    retained, transient = [], []
    protected = events._protected_graph_identity_signature(
        graph, excluded_node_keys=frozenset(), excluded_graph_keys=frozenset((_STATE_KEY,)),
        retained_references=retained, opaque_references=opaque, layout=layout,
        transient_networkx_cached_views=transient,
    )
    _set_runtime_mapping_item(graph.graph, _STATE_KEY, after)
    if protected != events._protected_graph_identity_signature(
        graph, excluded_node_keys=frozenset(), excluded_graph_keys=frozenset((_STATE_KEY,)),
        retained_references=[], opaque_references=opaque,
    ):
        raise TNFRValueError("carry attachment changed another graph-owned channel")
    events._remove_transient_networkx_cached_views(graph, tuple(transient))


def _snapshot(graph):
    value, captured = events._capture_interval_endpoint(graph)
    if not captured or value is None:
        raise TNFRValueError("carried execution requires a capturable scalar nodal state")
    return value


def _phase_tuple(graph):
    """Capture stored phases in the executor's ordered scalar chart.

    NodalFlowStateSnapshot intentionally contains only the flow channels.
    Keep these separate raw binary64 observations instead of inferring
    phase from pressure, which is not an injective phase readout.
    """
    layout = _networkx_runtime_layout(graph)
    return tuple(_scalar(data, ALIAS_THETA, "phase") for _node, data in layout.node_data)


def _configuration(graph, lower, upper):
    mapping = graph.graph
    if mapping.get("integrator") is not None:
        raise TNFRValueError("an explicitly configured integrator conflicts with the carried executor")
    if mapping.get("INTEGRATOR_METHOD", "euler") != "euler":
        raise TNFRValueError("carried execution requires the explicit Euler method")
    if mapping.get("use_extended_dynamics", False):
        raise TNFRValueError("extended dynamics are outside the carried nodal execution contract")
    gamma = mapping.get("GAMMA", {"type": "none"})
    if not isinstance(gamma, Mapping) or gamma.get("type", "none") != "none":
        raise TNFRValueError("carried execution requires Gamma none")
    if mapping.get("CLIP_MODE", "hard") != "hard":
        raise TNFRValueError("soft clipping is outside the unclipped carried nodal contract")
    configured_lower, configured_upper = mapping.get("EPI_MIN", -1.0), mapping.get("EPI_MAX", 1.0)
    if (type(configured_lower) is not float or type(configured_upper) is not float
            or not math.isfinite(configured_lower) or not math.isfinite(configured_upper)
            or not configured_lower <= lower <= upper <= configured_upper):
        raise TNFRValueError("the declared carried band must lie inside configured EPI bounds")
    present = "compute_delta_nfr" in mapping
    if present and mapping["compute_delta_nfr"] is not default_compute_delta_nfr:
        raise TNFRValueError("carried execution accepts only canonical default pressure")
    return present


def execute_nodal_remainder_event_schedule(
    graph: nx.Graph, schedule: OperatorEventSchedule, *,
    physical_flow_partitions: tuple[PhysicalFlowPartition, ...] = (),
    epi_lower: float = .05, epi_upper: float = 1.0,
) -> NodalRemainderEventExecution:
    """Execute one finite fixed-support UM/IL/SHA schedule with carried EPI.

    Each declared positive segment is one exact represented-input nodal step.
    Canonical pressure is refreshed from visible EPI before every segment and
    at each positive interval's terminal boundary. The reserved graph-owned
    encoding persists across calls; stale state is rejected, never reset.
    Both reconstructed and visible EPI must stay in the declared configured
    band. The executor does not clip, accept custom solvers, or transfer carry
    across an EPI-changing event. Graph-owned rollback includes its encoding,
    histories, caches, aliases and canonical stage writes.
    Every executed flow and event retains its ordered phase tuple before
    and after the operation inside the complete executor seal. These are
    actual stored observations, not phases reconstructed from pressure.
    """
    if type(graph) is not nx.Graph:
        raise TypeError("carried event execution requires an exact simple undirected nx.Graph")
    transaction = GraphTransactionSnapshot(graph)
    try:
        def prepare():
            if type(schedule) is not OperatorEventSchedule:
                raise TypeError("schedule must be an OperatorEventSchedule")
            schedule.__post_init__()
            events._validate_schedule_clock(schedule)
            partitions = events._materialize_physical_flow_partitions(schedule, physical_flow_partitions)
            if any(name not in _ALLOWED_NAMES for name in schedule.operator_names):
                raise TNFRValueError("carried events support only coupling, coherence and silence")
            nodes, epi, support, kind = _chart(graph)
            initial_state = initialize_nodal_remainder(epi, epi_lower=epi_lower, epi_upper=epi_upper)
            callback_present = _configuration(graph, epi_lower, epi_upper)
            context = {"initial_epi_nonzero": bool(epi) and all(value > 0 for value in epi)}
            operators, word = events._prepare_word(schedule, context)
            events._require_runtime_clock(graph, schedule.start_time, boundary="schedule.start")
            events._validate_hybrid_log(graph)
            binding = graph.graph.get(_STATE_KEY)
            if _STATE_KEY in graph.graph:
                _require_binding(graph, binding, lower=epi_lower, upper=epi_upper)
            return nodes, support, kind, initial_state, callback_present, context, operators, word, partitions, binding

        prepared = events._run_readonly_graph_observation(graph, prepare, label="carried schedule preparation")
        nodes, support, kind, zero_state, callback_present, context, operators, word, partitions, binding = prepared
        if binding is None:
            binding = _new_binding(graph, nodes, schedule.start_time, zero_state, support, kind)
            _owned_binding_commit(graph, None, binding)
        initial_binding = binding
        partition_index = {item.parent_interval.index: item for item in partitions}
        flows, committed_events, refresh_records = [], [], []

        def refresh(label):
            _require_binding(graph, binding, lower=epi_lower, upper=epi_upper)
            before = _snapshot(graph)
            callback, checks, _label = events._invoke_restricted_pressure_refresh(
                graph, expected_present=callback_present, expected_callback=default_compute_delta_nfr,
                n_jobs=None, boundary_label=label,
            )
            _require_binding(graph, binding, lower=epi_lower, upper=epi_upper)
            record = NodalRemainderPressureRefresh(
                binding.time, label, before, _snapshot(graph), binding, tuple(checks.items()), id(callback),
            )
            refresh_records.append(record)
            return record

        def stage_pressure_refresh(live_graph):
            if live_graph is not graph:
                raise TNFRValueError("stage pressure refresh changed its live graph owner")
            refresh("operator_stage_pressure_refresh")

        for interval in schedule.intervals:
            events._require_runtime_clock(graph, interval.start_time, boundary="carried interval start")
            if interval.exact_duration > 0:
                partition = partition_index.get(interval.index)
                segments = (interval,) if partition is None else partition.segments
                for segment_index, segment in enumerate(segments):
                    current_refresh = refresh(f"interval[{interval.index}].segment[{segment_index}].start")
                    before = current_refresh.after_snapshot
                    phase_before = _phase_tuple(graph)
                    _require_binding(graph, binding, lower=epi_lower, upper=epi_upper)
                    if before.epi != binding.state.epi:
                        raise TNFRValueError("flow capture differs from the current visible carried state")
                    weights = events._run_readonly_graph_observation(
                        graph, lambda: _runtime_weights(graph), label="carried pressure weights",
                    )
                    events._record_guarded_mutation_flow_boundary(graph)
                    step = events._run_readonly_graph_observation(
                        graph, lambda: advance_nodal_remainder(
                            binding.state, timestep=segment.duration, capacity=before.nu_f, pressure=before.delta_nfr,
                        ), label="carried nodal proposal",
                    )
                    if (type(step) is not NodalRemainderStep or step.before is not binding.state
                            or step.timestep != segment.duration or step.capacity != before.nu_f
                            or step.pressure != before.delta_nfr):
                        raise TNFRValueError("carried nodal proposal does not bind its actual live inputs")
                    layout = _networkx_runtime_layout(graph)
                    rates = tuple(nu * pressure for nu, pressure in zip(before.nu_f, before.delta_nfr, strict=True))
                    previous = tuple(_scalar(data, ALIAS_DEPI, "dEPI") for _node, data in layout.node_data)
                    accelerations = tuple((rate - old) / segment.duration for rate, old in zip(rates, previous, strict=True))
                    if not all(math.isfinite(value) for value in (*rates, *accelerations)):
                        raise TNFRValueError("carried flow requires finite visible derivatives")
                    flow_guard = events._capture_integrator_flow_state(graph, _FLOW_OWNER)
                    for index, (_node, data) in enumerate(layout.node_data):
                        set_attr(data, ALIAS_EPI, step.after.epi[index])
                        set_attr(data, ALIAS_DEPI, rates[index])
                        set_attr(data, ALIAS_D2EPI, accelerations[index])
                    _set_runtime_mapping_item(graph.graph, "_t", segment.start_time + segment.duration)
                    events._require_runtime_clock(graph, segment.end_time, boundary="carried segment end")
                    events._require_integrator_flow_contract(graph, _FLOW_OWNER, flow_guard, interval_index=interval.index)
                    after = _snapshot(graph)
                    phase_after = _phase_tuple(graph)
                    if after.epi != step.after.epi:
                        raise TNFRValueError("the committed EPI differs from the carried nodal proposal")
                    next_binding = _new_binding(graph, nodes, segment.end_time, step.after, support, kind)
                    _owned_binding_commit(graph, binding, next_binding)
                    previous_binding, binding = binding, next_binding
                    _require_binding(graph, binding, lower=epi_lower, upper=epi_upper)
                    events._record_guarded_mutation_flow_boundary(graph)
                    flows.append(ExecutedNodalRemainderFlow(
                        interval.index, segment_index, segment, before, after,
                        previous_binding, binding, step, weights, dict(weights)["epi"], current_refresh.checks,
                        phase_before, phase_after,
                    ))
                refresh(f"interval[{interval.index}].terminal")
            if interval.index >= schedule.event_count:
                continue
            event = schedule.events[interval.index]
            before_binding, before = binding, _snapshot(graph)
            phase_before = _phase_tuple(graph)
            _require_binding(graph, binding, lower=epi_lower, upper=epi_upper)
            stage = execute_network_operator_stage(
                graph, operators[event.word_position], nodes,
                sequence_context=None if word is None else word.step(event.word_position),
                compute_delta_nfr=stage_pressure_refresh if callback_present else None,
                transaction_snapshot=transaction,
            )
            events._require_exact_stage(event, stage, target_count=len(nodes))
            _require_binding(graph, binding, lower=epi_lower, upper=epi_upper)
            events._require_pressure_callback_binding(
                graph, expected_present=callback_present, expected_callback=default_compute_delta_nfr,
            )
            events._require_runtime_clock(graph, event.event_time, boundary="carried event")
            after = _snapshot(graph)
            phase_after = _phase_tuple(graph)
            committed = events.ExecutedOperatorEvent.from_stage(event, stage)
            committed_events.append(ExecutedNodalRemainderEvent(
                committed, stage, before_binding, binding, before, after, phase_before, phase_after,
            ))
            events._record_guarded_mutation_flow_boundary(graph)
            events._append_guarded_hybrid_event_record(graph, committed.as_record())
        _require_binding(graph, binding, lower=epi_lower, upper=epi_upper)
        events._require_runtime_clock(graph, schedule.end_time, boundary="carried schedule end")
        if flows:
            prefix = events._run_readonly_graph_observation(
                graph, lambda: observe_nodal_remainder_sequence(
                    initial=initial_binding.state,
                    timesteps=tuple(flow.step.timestep for flow in flows),
                    capacities=tuple(flow.step.capacity for flow in flows),
                    pressures=tuple(flow.step.pressure for flow in flows),
                ), label="carried prefix certification",
            )
            if prefix.steps != tuple(flow.step for flow in flows) or prefix.endpoint != binding.state:
                raise TNFRValueError("live carried steps differ from their exact finite prefix replay")
            area = prefix.prefixes[-1].cumulative_nodal_area
        else:
            prefix, area = None, (Fraction(0),) * len(nodes)
        change = tuple(after - before for after, before in zip(
            binding.state.exact_epi, initial_binding.state.exact_epi, strict=True,
        ))
        residual = tuple(actual - supplied for actual, supplied in zip(change, area, strict=True))
        if any(residual):
            raise TNFRValueError("the complete event execution lost its reconstructed nodal balance")
        result = NodalRemainderEventExecution(
            schedule, nodes, initial_binding, binding, tuple(flows), tuple(committed_events), prefix,
            tuple(refresh_records), tuple(context.items()), change, area, residual,
        )
        object.__setattr__(result, "execution_identity", id(result))
        object.__setattr__(result, "_proof_stamp", events._sealed_dataclass_stamp(result, "nodal_remainder_event_execution_v1"))
        if not result.runtime_provenance_certified:
            raise TNFRValueError("carried runtime evidence failed its executor seal")
        return result
    except BaseException as failure:
        transaction.restore_after_failure(graph, failure)
        raise
