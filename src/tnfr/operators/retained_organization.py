"""Member-preserving organization observations and canonical scale actions.

Members remain actual graph nodes with their own ports and event contexts.
This interface neither constructs a macro node nor defines proliferation or
synthesis. The graph retains complete state; observations are not checkpoints.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any

from .._exact_time import bounded_represented_real, finite_represented_real
from ..alias import get_attr
from ..constants.aliases import (
    ALIAS_DNFR,
    ALIAS_EPI,
    ALIAS_EPI_KIND,
    ALIAS_THETA,
    ALIAS_VF,
)
from ..utils._structural_signature import (
    proof_stamps_are_identical,
    structural_proof_signature,
)
from ._epi_domain import require_real_scalar_epi
from .contraction import Contraction
from .expansion import Expansion
from .grammar_debt import GrammarState, read_grammar_state
from .network_stage import (
    TWO_PHASE_JACOBI,
    GraphTransactionSnapshot,
    NetworkStageResult,
    _detached_stage_graph,
    execute_pointwise_stage,
)

__all__ = [
    "RetainedMemberState",
    "RetainedOrganization",
    "RetainedScaleDispatch",
    "capture_retained_organization",
    "retained_organization_matches_graph",
    "execute_retained_scale_stage",
]

_VERSION = "retained_organization_v1"
_MISSING = object()
_TEMPORAL_KEYS = ("epi_time_history", "epi_history", "_epi_history")
_CANONICAL_SCALE_BINDINGS = {
    kind: (kind.__call__, kind._execute, kind.name, kind.glyph)
    for kind in (Expansion, Contraction)
}
_OPTIONS = frozenset(
    {
        "window",
        "validate_preconditions",
        "collect_metrics",
        "validate_nodal_equation",
        "dt",
    }
)


@dataclass(frozen=True, slots=True)
class RetainedMemberState:
    """Readable primitive coordinates and separately retained own context.

    Temporal entries are immutable signatures of present raw history values;
    absence stays absent. They assert neither endpoint currency nor acceleration.
    """

    node: Any
    epi: float
    epi_kind: str
    phase: float
    capacity: float
    pressure: float
    grammar: GrammarState
    temporal_context: tuple[tuple[str, Any], ...]


def _member_payload(member: RetainedMemberState) -> tuple[Any, ...]:
    if (
        type(member) is not RetainedMemberState
        or type(member.grammar) is not GrammarState
    ):
        raise TypeError("retained member state requires the canonical record types")
    grammar = member.grammar
    return (
        member.node,
        member.epi,
        member.epi_kind,
        member.phase,
        member.capacity,
        member.pressure,
        grammar.history,
        grammar.u2_debt,
        grammar.prior_coherence,
        grammar.operator_step,
        member.temporal_context,
    )


def _payload_stamp(record: RetainedOrganization) -> Any:
    return structural_proof_signature(
        (
            _VERSION,
            id(record._graph_owner),
            record.members,
            tuple(_member_payload(member) for member in record.member_states),
            record.directed,
            record.multigraph,
            record.incoming_ports,
            record.outgoing_ports,
            record.incident_edges,
            record.member_metadata,
            record._state_associations,
        ),
        opaque_references=(record._graph_owner,),
    )


@dataclass(frozen=True, slots=True)
class RetainedOrganization:
    """Sealed owner-bound member/port observation, not a collective identity proof.

    ``member_metadata`` and incident attribute entries are immutable signatures,
    not serialized state. All ports retain original neighbor order; parallel edges
    remain distinct in ``incident_edges`` but count once in each neighbor sequence.
    Boundary-node form, phase, capacity and history are not admitted or retained.
    """

    members: tuple[Any, ...]
    member_states: tuple[RetainedMemberState, ...]
    directed: bool
    multigraph: bool
    incoming_ports: tuple[tuple[Any, ...], ...]
    outgoing_ports: tuple[tuple[Any, ...], ...]
    incident_edges: tuple[tuple[Any, Any, Any, Any], ...]
    member_metadata: tuple[Any, ...]
    _graph_owner: Any = field(repr=False, compare=False)
    _state_associations: Any = field(default=(), repr=False, compare=False)
    _proof_stamp: Any = field(default=(), repr=False, compare=False)
    _retained_references: tuple[Any, ...] = field(default=(), repr=False, compare=False)

    def _proof_fields_are_intact(self) -> bool:
        try:
            return type(self) is RetainedOrganization and proof_stamps_are_identical(
                self._proof_stamp, _payload_stamp(self)
            )
        except BaseException:
            return False


@dataclass(frozen=True, slots=True)
class RetainedScaleDispatch:
    """Summary of one canonical scale stage, not an execution certificate.

    Construction checks internal retained-boundary consistency. Publicly
    constructing this summary does not authenticate an operator invocation.
    """

    before: RetainedOrganization
    after: RetainedOrganization
    stage_result: NetworkStageResult

    def __post_init__(self) -> None:
        for record in (self.before, self.after):
            if (
                type(record) is not RetainedOrganization
                or not record._proof_fields_are_intact()
            ):
                raise ValueError("retained scale dispatch requires intact observations")
        if (
            self.before._graph_owner is not self.after._graph_owner
            or not proof_stamps_are_identical(
                structural_proof_signature(self.before.members),
                structural_proof_signature(self.after.members),
            )
        ):
            raise ValueError("retained scale dispatch must preserve member identities")

        def retained_boundary(record: RetainedOrganization) -> Any:
            return structural_proof_signature(
                (
                    record.directed,
                    record.multigraph,
                    record.incoming_ports,
                    record.outgoing_ports,
                    record.incident_edges,
                    tuple(
                        (member.node, member.phase) for member in record.member_states
                    ),
                )
            )

        if not proof_stamps_are_identical(
            retained_boundary(self.before), retained_boundary(self.after)
        ):
            raise ValueError(
                "retained scale dispatch must preserve ports and member phase"
            )
        stage = self.stage_result
        if (
            type(stage) is not NetworkStageResult
            or stage.operator not in (Expansion.name, Contraction.name)
            or stage.glyph
            != (
                Expansion.glyph.value
                if stage.operator == Expansion.name
                else Contraction.glyph.value
            )
            or stage.schedule != TWO_PHASE_JACOBI
            or stage.nodes_processed != len(self.before.members)
        ):
            raise ValueError("retained scale dispatch requires a canonical scale stage")


def _members(graph: Any, members: tuple[Any, ...]) -> tuple[Any, ...]:
    if type(members) is not tuple or not members:
        raise TypeError("retained members must be a nonempty exact tuple")
    if len(set(members)) != len(members):
        raise ValueError("retained members must be unique")
    for node in members:
        if node not in graph:
            raise KeyError(node)
    return members


def _required(data: Mapping[str, Any], aliases: tuple[str, ...], label: str) -> Any:
    raw = get_attr(data, aliases, _MISSING, strict=True, conv=lambda value: value)
    if raw is _MISSING:
        raise ValueError(f"retained organization requires explicit member {label}")
    return raw


def _signature(graph: Any, value: Any, references: list[Any] | None = None) -> Any:
    return structural_proof_signature(
        value, opaque_references=(graph,), _retained_references=references
    )


def _edges(graph: Any):
    if graph.is_multigraph():
        yield from graph.edges(keys=True, data=True)
    else:
        for left, right, data in graph.edges(data=True):
            yield left, right, None, data


def _capture(graph: Any, members: tuple[Any, ...]) -> RetainedOrganization:
    members = _members(graph, members)
    selected = set(members)
    references: list[Any] = []
    metadata = tuple(
        _signature(graph, graph.nodes[node], references) for node in members
    )
    temporal = tuple(
        tuple(
            (key, _signature(graph, graph.nodes[node][key], references))
            for key in _TEMPORAL_KEYS
            if key in graph.nodes[node]
        )
        for node in members
    )
    incoming = tuple(
        tuple(
            graph.predecessors(node) if graph.is_directed() else graph.neighbors(node)
        )
        for node in members
    )
    outgoing = tuple(tuple(graph.neighbors(node)) for node in members)
    raw_incident = tuple(
        (left, right, key, data)
        for left, right, key, data in _edges(graph)
        if left in selected or right in selected
    )
    incident = tuple(
        (left, right, key, _signature(graph, data, references))
        for left, right, key, data in raw_incident
    )
    associations = _signature(
        graph,
        (tuple((node, graph.nodes[node]) for node in members), raw_incident),
        references,
    )
    snapshot = _detached_stage_graph(graph)
    states = []
    for node, context in zip(members, temporal, strict=True):
        data = snapshot.nodes[node]
        epi = require_real_scalar_epi(
            _required(data, ALIAS_EPI, "EPI"),
            operator="RetainedOrganization",
            label="member EPI",
        )
        phase = finite_represented_real(
            _required(data, ALIAS_THETA, "phase"), "member phase"
        )[0]
        capacity = bounded_represented_real(
            _required(data, ALIAS_VF, "capacity"), "member capacity", minimum=0
        )[0]
        pressure = finite_represented_real(
            _required(data, ALIAS_DNFR, "pressure"), "member pressure"
        )[0]
        kind = get_attr(
            data, ALIAS_EPI_KIND, None, strict=True, conv=lambda value: value
        )
        if kind is None:
            kind = ""
        if type(kind) is not str:
            raise TypeError("retained member EPI kind must be a string or None")
        states.append(
            RetainedMemberState(
                node,
                epi,
                kind,
                phase,
                capacity,
                pressure,
                read_grammar_state(data),
                context,
            )
        )
    record = RetainedOrganization(
        members,
        tuple(states),
        graph.is_directed(),
        graph.is_multigraph(),
        incoming,
        outgoing,
        incident,
        metadata,
        graph,
        _state_associations=associations,
        _retained_references=tuple(references),
    )
    return replace(record, _proof_stamp=_payload_stamp(record))


def capture_retained_organization(
    graph: Any, members: tuple[Any, ...]
) -> RetainedOrganization:
    """Observe existing members without writing defaults, history or telemetry.

    The shared detached reader admits core coordinates; original metadata is bound
    by shared signatures. Missing or stale temporal evidence is retained without
    repair or an acceleration claim. Failure restores graph-owned state.
    """
    transaction = GraphTransactionSnapshot(graph)
    try:
        record = _capture(graph, members)
        transaction.restore(graph)
        return record
    except BaseException as failure:
        transaction.restore_after_failure(graph, failure)
        raise


def _readmit_retained_organization(
    graph: Any, record: RetainedOrganization
) -> RetainedOrganization | None:
    """Return the same fresh read that establishes current record agreement."""
    if (
        type(record) is not RetainedOrganization
        or record._graph_owner is not graph
        or not record._proof_fields_are_intact()
    ):
        return None
    try:
        current = capture_retained_organization(graph, record.members)
        return (
            current
            if proof_stamps_are_identical(record._proof_stamp, current._proof_stamp)
            else None
        )
    except Exception:
        return None


def retained_organization_matches_graph(
    graph: Any, record: RetainedOrganization
) -> bool:
    """Re-admit the original member/port boundary; unrelated state may differ."""
    return _readmit_retained_organization(graph, record) is not None


def _protected_state(
    graph: Any,
    members: tuple[Any, ...],
    *,
    preserve_member_phase: bool = True,
) -> Any:
    """Bind support/nonmember state and, by default, selected member phase."""
    if type(preserve_member_phase) is not bool:
        raise TypeError("preserve_member_phase must be a bool")
    selected = set(members)
    return _signature(
        graph,
        (
            graph.is_directed(),
            graph.is_multigraph(),
            tuple(graph.nodes),
            tuple(
                (
                    node,
                    tuple(
                        graph.predecessors(node)
                        if graph.is_directed()
                        else graph.neighbors(node)
                    ),
                    tuple(graph.neighbors(node)),
                )
                for node in graph.nodes
            ),
            tuple((left, right, key, data) for left, right, key, data in _edges(graph)),
            tuple(
                (node, graph.nodes[node])
                for node in graph.nodes
                if node not in selected
            ),
            tuple(
                (
                    node,
                    tuple(
                        (key, graph.nodes[node][key])
                        for key in ALIAS_THETA
                        if key in graph.nodes[node]
                    ),
                )
                for node in (members if preserve_member_phase else ())
            ),
        ),
    )


def execute_retained_scale_stage(
    graph: Any,
    record: RetainedOrganization,
    operator: Expansion | Contraction,
    *,
    execution_kwargs: Mapping[str, Any] | None = None,
) -> RetainedScaleDispatch:
    """Execute the existing scale map on exactly the retained members once.

    No graph edit, pressure callback, pooled context or fallback glyph is admitted.
    Whole support, nonmember state and member phase must remain unchanged. Ordinary
    graph telemetry and each target's own event history may change. Graph-owned
    failures roll back; external side effects are outside the transaction contract.
    """
    transaction = GraphTransactionSnapshot(graph)
    try:
        kind = type(operator)
        if kind not in _CANONICAL_SCALE_BINDINGS:
            raise TypeError(
                "retained scale action requires exact Expansion or Contraction"
            )
        canonical_call, canonical_execute, canonical_name, canonical_glyph = (
            _CANONICAL_SCALE_BINDINGS[kind]
        )
        if (
            kind.__call__ is not canonical_call
            or kind._execute is not canonical_execute
            or operator.name != canonical_name
            or operator.glyph is not canonical_glyph
            or any(
                key in vars(operator)
                for key in ("__call__", "_execute", "name", "glyph")
            )
        ):
            raise TypeError(
                "retained scale action requires unmodified canonical bindings"
            )
        if execution_kwargs is not None and not isinstance(execution_kwargs, Mapping):
            raise TypeError("execution_kwargs must be a mapping or None")
        options = {} if execution_kwargs is None else dict(execution_kwargs)
        if any(type(key) is not str or key not in _OPTIONS for key in options):
            raise ValueError("unsupported retained scale execution option")
        transaction.restore(graph)
        before = _readmit_retained_organization(graph, record)
        if before is None:
            raise ValueError(
                "retained organization is foreign, stale, invalid or unsealed"
            )
        protected = _protected_state(graph, record.members)
        stage = execute_pointwise_stage(
            graph,
            operator,
            record.members,
            transaction_snapshot=transaction,
            require_exact_operator=True,
            **options,
        )
        if not proof_stamps_are_identical(
            protected, _protected_state(graph, record.members)
        ):
            raise RuntimeError("retained scale action changed protected state or ports")
        after = _capture(graph, record.members)
        return RetainedScaleDispatch(before, after, stage)
    except BaseException as failure:
        transaction.restore_after_failure(graph, failure)
        raise
