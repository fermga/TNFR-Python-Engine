"""Address live external Reception through a retained organization's ports.

Input ordinals resolve a member and one external donor; output ordinals resolve
an external receiver shared by every retained member. The interior remains
actual graph state with its own context; no macro node or reduced law is
supplied. External coordinates stay live until the canonical EN snapshot.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .._exact_time import bounded_represented_real, finite_represented_real
from ..config.parsing import parse_bool
from ..constants import get_param
from ..constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_VF
from ..utils._structural_signature import proof_stamps_are_identical
from ._argument_validation import finite_real, validate_common_execution_arguments
from ._epi_domain import require_real_scalar_epi
from ._reception_kernel import _node_kind, _raw_alias
from .network_stage import (
    TWO_PHASE_JACOBI,
    GraphTransactionSnapshot,
    NetworkStageResult,
    execute_neighbor_stage,
)
from .retained_organization import (
    _OPTIONS,
    RetainedOrganization,
    _member_payload,
    _protected_state,
    _readmit_retained_organization,
    capture_retained_organization,
)
from .retained_synthesis import (
    _CANONICAL_RECEPTION,
    _boundary,
    _canonical_reception,
    _same,
    _same_fields,
)

__all__ = [
    "RetainedInputPort",
    "RetainedReceptionDispatch",
    "retained_input_ports",
    "execute_retained_reception_stage",
    "RetainedOutputPort",
    "RetainedOutputReceptionDispatch",
    "retained_output_ports",
    "execute_retained_output_reception_stage",
]

_OPERATOR = "Retained Reception"


def _ordinal(value: Any) -> int:
    if type(value) is not int:
        raise TypeError("retained input port ordinal must be an exact integer")
    if value < 0:
        raise ValueError("retained input port ordinal must be nonnegative")
    return value


def _admit_record(
    record: RetainedOrganization, *, label: str = "retained input ports"
) -> None:
    if type(record) is not RetainedOrganization:
        raise TypeError(f"{label} require an exact RetainedOrganization")
    if not record._proof_fields_are_intact():
        raise ValueError(f"{label} require an intact organization")


@dataclass(frozen=True, slots=True)
class RetainedInputPort:
    """An organization-relative address, not a donor snapshot or permission."""

    ordinal: int
    member: Any
    source: Any

    def __post_init__(self) -> None:
        _ordinal(self.ordinal)


def retained_input_ports(record: RetainedOrganization) -> tuple[RetainedInputPort, ...]:
    """Enumerate retained external incoming neighbors without a live graph read.

    Member order precedes each retained incoming unique-neighbor order. Internal
    nodes, including self-loops, are excluded. Parallel edges count once here;
    their keyed data remain bound by the organization. An empty table is valid.
    Even an intact table may describe a record that no longer matches its graph.
    """
    _admit_record(record)
    members = set(record.members)
    ports = []
    for member, incoming in zip(record.members, record.incoming_ports, strict=True):
        for source in incoming:
            if source not in members:
                ports.append(RetainedInputPort(len(ports), member, source))
    return tuple(ports)


@dataclass(frozen=True, slots=True)
class RetainedReceptionDispatch:
    """Consistent addressed EN summary, not an authenticated invocation receipt.

    The shared EN observation retains its actual semantic-kind view. Generic
    retained member kinds normalize absent/None and cannot replace that view.
    External donor coordinates are deliberately absent from the before record.
    """

    before: RetainedOrganization
    after: RetainedOrganization
    port: RetainedInputPort
    stage_result: NetworkStageResult

    def __post_init__(self) -> None:
        _admit_record(self.before)
        _admit_record(self.after)
        if self.before._graph_owner is not self.after._graph_owner or not _same(
            _boundary(self.before), _boundary(self.after)
        ):
            raise ValueError(
                "retained Reception must preserve member identities and ports"
            )
        if type(self.port) is not RetainedInputPort:
            raise TypeError("retained Reception requires its resolved input port")
        self.port.__post_init__()
        ports = retained_input_ports(self.before)
        if self.port.ordinal >= len(ports) or not _same_fields(
            (self.port.member, self.port.source),
            (ports[self.port.ordinal].member, ports[self.port.ordinal].source),
        ):
            raise ValueError("retained Reception port contradicts its organization")
        index = self.before.members.index(self.port.member)
        for position, (before, after) in enumerate(
            zip(self.before.member_states, self.after.member_states, strict=True)
        ):
            if position == index:
                if not _same_fields(
                    (before.capacity, before.pressure),
                    (after.capacity, after.pressure),
                ):
                    raise ValueError(
                        "retained Reception must preserve target capacity and pressure"
                    )
            elif not _same_fields(
                (_member_payload(before), self.before.member_metadata[position]),
                (_member_payload(after), self.after.member_metadata[position]),
            ):
                raise ValueError(
                    "retained Reception changed another member's own state"
                )
        result = self.stage_result
        if type(result) is not NetworkStageResult:
            raise TypeError("retained Reception requires its actual stage result")
        result.__post_init__()
        if (
            result.operator != _CANONICAL_RECEPTION[2]
            or result.glyph != _CANONICAL_RECEPTION[3].value
            or result.schedule != TWO_PHASE_JACOBI
            or result.nodes_processed != 1
            or len(result.reception_observations) != 1
            or result.neighbor_epi_jump_certificate is not None
        ):
            raise ValueError("retained Reception requires one canonical EN stage")
        observed = result.reception_observations[0]
        if (
            not observed._proof_fields_are_intact()
            or observed.source_tracking_enabled
            or observed.source_selection_origin != "explicit_request"
            or observed.compatibility_decision is not None
            or not _same_fields(
                (
                    observed.node,
                    observed.source_nodes,
                    observed.neighbors,
                    observed.target_epi_before,
                    observed.target_epi_after,
                ),
                (
                    self.port.member,
                    (self.port.source,),
                    (self.port.source,),
                    self.before.member_states[index].epi,
                    self.after.member_states[index].epi,
                ),
            )
        ):
            raise ValueError("retained Reception read does not match its resolved port")
        for actual_kind, member in (
            (observed.target_epi_kind_before, self.before.member_states[index]),
            (observed.target_epi_kind_after, self.after.member_states[index]),
        ):
            # The record normalizes None/absence. The live consumer separately
            # checks EN's exact views, without inventing a second kind snapshot.
            allowed = ("", "None") if member.epi_kind == "" else (member.epi_kind,)
            if actual_kind not in allowed:
                raise ValueError("retained Reception kind contradicts its member")


def _materialize_options(raw: Mapping[str, Any] | None) -> dict[str, Any]:
    if raw is not None and not isinstance(raw, Mapping):
        raise TypeError("execution_kwargs must be a mapping or None")
    options = {} if raw is None else dict(raw)
    if any(type(key) is not str or key not in _OPTIONS for key in options):
        raise ValueError("unsupported retained Reception execution option")
    return options


def _admit_options(graph: Any, options: dict[str, Any]) -> dict[str, Any]:
    from . import _validated_execution_window

    options = dict(options)
    options.setdefault("validate_preconditions", True)
    validate_common_execution_arguments(graph.graph, options, operator=_OPERATOR)
    if not options["validate_preconditions"]:
        raise ValueError("retained Reception requires validate_preconditions=True")
    options["window"] = _validated_execution_window(graph, options.get("window"))
    if "dt" in options:
        options["dt"] = finite_real(
            options["dt"],
            operator=_OPERATOR,
            label="dt",
            lower=math.nextafter(0.0, math.inf),
        )
    canon = get_param(graph, "GRAMMAR_CANON")
    if not isinstance(canon, Mapping):
        raise TypeError("GRAMMAR_CANON must be a mapping")
    if not parse_bool(canon.get("enabled", False)) or (
        graph.graph.get("GRAMMAR_REJECTION_MODE") != "raise"
    ):
        raise ValueError("retained Reception requires enabled grammar and raise mode")
    return options


def execute_retained_reception_stage(
    graph: Any,
    record: RetainedOrganization,
    port_index: int,
    *,
    execution_kwargs: Mapping[str, Any] | None = None,
) -> RetainedReceptionDispatch:
    """Resolve one external input address and execute actual member EN atomically.

    Every member and port is re-admitted; external donor form/kind is live.
    Only the resolved member may change its form/kind and own event context.
    Capacity, pressure, phases, support and all non-target nodes are preserved.
    Ordinary graph telemetry may change. Optional activity gates retain their
    graph policy; local precondition validation and canonical raise grammar are
    mandatory. No source/context override, fallback or callback is accepted.

    Graph-owned failures restore this invocation, including options mapping
    materialization. External side effects are outside that transaction. The
    returned after record supports a later explicit call; no event is scheduled.
    """
    transaction = GraphTransactionSnapshot(graph)
    try:
        options = _materialize_options(execution_kwargs)
        transaction.restore(graph)
        options = _admit_options(graph, options)
        port_index = _ordinal(port_index)
        operator = _canonical_reception()
        before = _readmit_retained_organization(graph, record)
        if before is None:
            raise ValueError(
                "retained organization is foreign, stale, invalid or unsealed"
            )
        ports = retained_input_ports(before)
        if port_index >= len(ports):
            raise IndexError("retained input port does not exist")
        port = ports[port_index]
        protected = _protected_state(graph, (port.member,))
        kind_before = _node_kind(graph, port.member)
        result = execute_neighbor_stage(
            graph,
            operator,
            (port.member,),
            source_nodes=(port.source,),
            track_sources=False,
            transaction_snapshot=transaction,
            **options,
        )
        after = capture_retained_organization(graph, before.members)
        if not proof_stamps_are_identical(
            protected, _protected_state(graph, (port.member,))
        ):
            raise RuntimeError("retained Reception changed protected state or ports")
        if len(result.reception_observations) != 1 or not _same_fields(
            (
                result.reception_observations[0].target_epi_kind_before,
                result.reception_observations[0].target_epi_kind_after,
            ),
            (kind_before, _node_kind(graph, port.member)),
        ):
            raise RuntimeError("retained Reception changed its actual target kind")
        return RetainedReceptionDispatch(before, after, port, result)
    except BaseException as failure:
        transaction.restore_after_failure(graph, failure)
        raise


@dataclass(frozen=True, slots=True)
class RetainedOutputPort:
    """One external receiver shared by the complete retained source group.

    This address supplies neither a cached receiver state nor live permission.
    All sources remain individual graph nodes with their own contexts.
    """

    ordinal: int
    receiver: Any
    sources: tuple[Any, ...]

    def __post_init__(self) -> None:
        _ordinal(self.ordinal)
        if type(self.sources) is not tuple or not self.sources:
            raise TypeError("retained output sources must be a nonempty exact tuple")
        if len(set(self.sources)) != len(self.sources):
            raise ValueError("retained output sources must be unique")
        if self.receiver in self.sources:
            raise ValueError("retained output receiver must be external to its group")


def retained_output_ports(
    record: RetainedOrganization,
) -> tuple[RetainedOutputPort, ...]:
    """Observe external receivers reached by every retained member.

    Order follows the first member's outgoing unique-neighbor order. Each
    directed relation must point from member to receiver, supplying incoming
    support for actual EN. Parallel edges do not duplicate sources. Incomplete
    intersections yield no port; they never silently select a smaller group.
    This passive table can outlive its association with the actual graph.
    """
    _admit_record(record, label="retained output ports")
    members = set(record.members)
    remaining = tuple(set(neighbors) for neighbors in record.outgoing_ports[1:])
    ports = []
    for receiver in record.outgoing_ports[0]:
        if receiver not in members and all(receiver in nodes for nodes in remaining):
            ports.append(RetainedOutputPort(len(ports), receiver, record.members))
    return tuple(ports)


@dataclass(frozen=True, slots=True)
class RetainedOutputReceptionDispatch:
    """Checked grouped-output summary, not an authenticated invocation receipt.

    Before/after describe the unchanged source group. The shared EN observation
    describes the live receiver and its real read; generic retained kinds cannot
    reconstruct that observation's distinction between absent and None kinds.
    """

    before: RetainedOrganization
    after: RetainedOrganization
    port: RetainedOutputPort
    stage_result: NetworkStageResult

    def __post_init__(self) -> None:
        _admit_record(self.before, label="retained output ports")
        _admit_record(self.after, label="retained output ports")
        if self.before._graph_owner is not self.after._graph_owner or not _same(
            _boundary(self.before), _boundary(self.after)
        ):
            raise ValueError(
                "retained output must preserve source identities and ports"
            )
        if not _same_fields(
            tuple(
                (_member_payload(member), metadata)
                for member, metadata in zip(
                    self.before.member_states, self.before.member_metadata, strict=True
                )
            ),
            tuple(
                (_member_payload(member), metadata)
                for member, metadata in zip(
                    self.after.member_states, self.after.member_metadata, strict=True
                )
            ),
        ):
            raise ValueError("retained output changed its source group's own state")
        if type(self.port) is not RetainedOutputPort:
            raise TypeError("retained output requires its resolved output port")
        self.port.__post_init__()
        ports = retained_output_ports(self.before)
        if self.port.ordinal >= len(ports) or not _same_fields(
            (self.port.receiver, self.port.sources),
            (ports[self.port.ordinal].receiver, ports[self.port.ordinal].sources),
        ):
            raise ValueError("retained output port contradicts its organization")
        result = self.stage_result
        if type(result) is not NetworkStageResult:
            raise TypeError("retained output requires its actual stage result")
        result.__post_init__()
        if (
            result.operator != _CANONICAL_RECEPTION[2]
            or result.glyph != _CANONICAL_RECEPTION[3].value
            or result.schedule != TWO_PHASE_JACOBI
            or result.nodes_processed != 1
            or len(result.reception_observations) != 1
            or result.neighbor_epi_jump_certificate is not None
        ):
            raise ValueError("retained output requires one canonical EN stage")
        observed = result.reception_observations[0]
        if (
            not observed._proof_fields_are_intact()
            or observed.source_tracking_enabled
            or observed.source_selection_origin != "explicit_request"
            or observed.compatibility_decision is not None
            or not _same_fields(
                (
                    observed.node,
                    observed.source_nodes,
                    observed.neighbors,
                    observed.neighbor_epi_values,
                ),
                (
                    self.port.receiver,
                    self.before.members,
                    self.before.members,
                    tuple(member.epi for member in self.before.member_states),
                ),
            )
        ):
            raise ValueError("retained output read does not match its complete group")
        for actual_kind, member in zip(
            observed.neighbor_epi_kinds, self.before.member_states, strict=True
        ):
            allowed = ("", "None") if member.epi_kind == "" else (member.epi_kind,)
            if actual_kind not in allowed:
                raise ValueError("retained output kind contradicts its source member")


def execute_retained_output_reception_stage(
    graph: Any,
    record: RetainedOrganization,
    port_index: int,
    *,
    execution_kwargs: Mapping[str, Any] | None = None,
) -> RetainedOutputReceptionDispatch:
    """Deliver one actual source-group EN to a resolved external receiver.

    Re-admit all source members, own metadata/context and incident support.
    The receiver is live and receives exactly one canonical EN using all ordered
    members. Sources, other nodes, support, phases, and receiver capacity and
    pressure are preserved. Receiver form/kind and own event bookkeeping, plus
    ordinary graph telemetry, may change. No average replaces actual members.

    Options, canonical bindings and graph-owned rollback follow the retained
    input consumer. No caller-supplied target/source override, fallback or
    callback is accepted. Later RA remains a separate admitted invocation.
    """
    transaction = GraphTransactionSnapshot(graph)
    try:
        options = _materialize_options(execution_kwargs)
        transaction.restore(graph)
        options = _admit_options(graph, options)
        port_index = _ordinal(port_index)
        operator = _canonical_reception()
        before = _readmit_retained_organization(graph, record)
        if before is None:
            raise ValueError(
                "retained organization is foreign, stale, invalid or unsealed"
            )
        ports = retained_output_ports(before)
        if port_index >= len(ports):
            raise IndexError("retained output port does not exist")
        port = ports[port_index]
        protected = _protected_state(graph, (port.receiver,))
        receiver_before = _output_receiver_state(graph, port.receiver)
        source_kinds = tuple(_node_kind(graph, node) for node in before.members)
        kind_before = _node_kind(graph, port.receiver)
        result = execute_neighbor_stage(
            graph,
            operator,
            (port.receiver,),
            source_nodes=before.members,
            track_sources=False,
            transaction_snapshot=transaction,
            **options,
        )
        after = capture_retained_organization(graph, before.members)
        if not proof_stamps_are_identical(
            protected, _protected_state(graph, (port.receiver,))
        ):
            raise RuntimeError("retained output changed protected state or ports")
        receiver_after = _output_receiver_state(graph, port.receiver)
        if not _same_fields(
            receiver_before[1:],
            receiver_after[1:],
        ):
            raise RuntimeError("retained output changed receiver capacity or pressure")
        if len(result.reception_observations) != 1 or not _same_fields(
            (
                result.reception_observations[0].neighbor_epi_kinds,
                result.reception_observations[0].target_epi_before,
                result.reception_observations[0].target_epi_after,
                result.reception_observations[0].target_epi_kind_before,
                result.reception_observations[0].target_epi_kind_after,
            ),
            (
                source_kinds,
                receiver_before[0],
                receiver_after[0],
                kind_before,
                _node_kind(graph, port.receiver),
            ),
        ):
            raise RuntimeError("retained output changed its actual Reception read")
        return RetainedOutputReceptionDispatch(before, after, port, result)
    except BaseException as failure:
        transaction.restore_after_failure(graph, failure)
        raise


def _output_receiver_state(graph: Any, receiver: Any) -> tuple[float, float, float]:
    """Read consumed form and held capacity/pressure, without retaining a member.

    The external receiver keeps ordinary EN's live kind/context domain. No
    additional finite phase or retained-member metadata admission is imposed.
    """
    return (
        require_real_scalar_epi(
            _raw_alias(graph, receiver, ALIAS_EPI, 0.0),
            operator=_OPERATOR,
            label="receiver EPI",
        ),
        bounded_represented_real(
            _raw_alias(graph, receiver, ALIAS_VF, 0.0),
            "receiver capacity",
            minimum=0,
        )[0],
        finite_represented_real(
            _raw_alias(graph, receiver, ALIAS_DNFR, 0.0),
            "receiver pressure",
        )[0],
    )
