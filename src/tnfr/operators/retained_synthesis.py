"""Retained two-member concentration through actual NUL followed by EN.

This explicitly selected fragment retains both nodes and all ports. It is not
an additional glyph, a unique synthesis law, or historical reabsorption. Exact
mean/contrast observations never reconstruct or replace executable node state.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from .._exact_time import finite_represented_real
from ..utils._structural_signature import (
    proof_stamps_are_identical,
    structural_proof_signature,
)
from ._reception_kernel import _node_kind
from .contraction import Contraction
from .network_stage import (
    TWO_PHASE_JACOBI,
    GraphTransactionSnapshot,
    NetworkStageResult,
    execute_neighbor_stage,
)
from .reception import Reception
from .retained_organization import (
    _OPTIONS,
    RetainedOrganization,
    RetainedScaleDispatch,
    _member_payload,
    _protected_state,
    capture_retained_organization,
    execute_retained_scale_stage,
)

__all__ = [
    "RetainedPairState",
    "RetainedSynthesisDispatch",
    "capture_retained_pair",
    "execute_retained_synthesis_stage",
]

_CANONICAL_RECEPTION = (
    Reception.__call__,
    Reception._execute,
    Reception.name,
    Reception.glyph,
)
_CONTRACTION_NAME = Contraction.name
_CONTRACTION_GLYPH = Contraction.glyph.value


def _same(left: Any, right: Any) -> bool:
    return proof_stamps_are_identical(
        structural_proof_signature(left), structural_proof_signature(right)
    )


def _same_fields(left: tuple[Any, ...], right: tuple[Any, ...]) -> bool:
    """Compare separate semantic fields without imposing cross-field aliases."""
    return len(left) == len(right) and all(
        _same(a, b) for a, b in zip(left, right, strict=True)
    )


def _chart(organization: RetainedOrganization) -> tuple[Fraction, Fraction]:
    if (
        type(organization) is not RetainedOrganization
        or not organization._proof_fields_are_intact()
        or len(organization.members) != 2
    ):
        raise ValueError("retained pair requires an intact two-member organization")
    core, member = (
        finite_represented_real(state.epi, "retained pair form")[1]
        for state in organization.member_states
    )
    return (core + member) / 2, (core - member) / 2


@dataclass(frozen=True, slots=True)
class RetainedPairState:
    """Owner-bound pair with exact real mean/contrast of represented forms.

    The raw organization retains both identities, kinds, contexts and ports.
    Fraction coordinates omit signed-zero encoding and are not a checkpoint.
    They are observations only, never a floating-point reconstruction path.
    ``reception_kinds`` retains EN's actual string view: explicit None becomes
    'None', whereas absent kind becomes ''. The general member view normalizes
    both to ''. Consumption rechecks this additional view against live aliases.
    """

    organization: RetainedOrganization
    form_mean: Fraction
    form_contrast: Fraction
    reception_kinds: tuple[str, str]

    def __post_init__(self) -> None:
        mean, contrast = _chart(self.organization)
        if (
            type(self.form_mean) is not Fraction
            or type(self.form_contrast) is not Fraction
            or self.form_mean != mean
            or self.form_contrast != contrast
        ):
            raise ValueError("retained pair chart contradicts its represented forms")
        if (
            type(self.reception_kinds) is not tuple
            or len(self.reception_kinds) != 2
            or any(type(kind) is not str for kind in self.reception_kinds)
            or any(
                actual
                not in (("", "None") if state.epi_kind == "" else (state.epi_kind,))
                for actual, state in zip(
                    self.reception_kinds, self.organization.member_states, strict=True
                )
            )
        ):
            raise ValueError(
                "retained pair Reception kinds contradict member admission"
            )

    @property
    def members(self) -> tuple[Any, Any]:
        return self.organization.members

    @property
    def core(self) -> Any:
        return self.members[0]

    @property
    def member(self) -> Any:
        return self.members[1]


def _admit_pair(pair: RetainedPairState) -> None:
    if type(pair) is not RetainedPairState:
        raise TypeError("retained synthesis requires an exact RetainedPairState")
    pair.__post_init__()


def capture_retained_pair(graph: Any, members: tuple[Any, Any]) -> RetainedPairState:
    """Observe an ordered pair without requiring a link or creating state."""
    if type(members) is not tuple or len(members) != 2:
        raise ValueError("retained pair members must be an exact two-element tuple")
    transaction = GraphTransactionSnapshot(graph)
    try:
        organization = capture_retained_organization(graph, members)
        kinds = tuple(_node_kind(graph, node) for node in members)
        pair = RetainedPairState(organization, *_chart(organization), kinds)
        transaction.restore(graph)
        return pair
    except BaseException as failure:
        transaction.restore_after_failure(graph, failure)
        raise


def _boundary(organization: RetainedOrganization) -> tuple[Any, ...]:
    return (
        organization.members,
        organization.directed,
        organization.multigraph,
        organization.incoming_ports,
        organization.outgoing_ports,
        organization.incident_edges,
        tuple((state.node, state.phase) for state in organization.member_states),
    )


@dataclass(frozen=True, slots=True)
class RetainedSynthesisDispatch:
    """Consistent NUL/EN observations, not authenticated execution evidence.

    Both members remain separately addressable. Current form information may
    be lost by an EN reset even though this report retains earlier observations.
    """

    before: RetainedPairState
    contracted: RetainedPairState
    after: RetainedPairState
    contraction_result: RetainedScaleDispatch
    reception_result: NetworkStageResult

    def __post_init__(self) -> None:
        for pair in (self.before, self.contracted, self.after):
            _admit_pair(pair)
        original = self.before.organization
        contracted = self.contracted.organization
        after = self.after.organization
        for record in (contracted, after):
            if record._graph_owner is not original._graph_owner or not _same(
                _boundary(record), _boundary(original)
            ):
                raise ValueError(
                    "retained synthesis must preserve identities and ports"
                )
        scale = self.contraction_result
        if type(scale) is not RetainedScaleDispatch:
            raise TypeError("retained synthesis requires its Contraction result")
        scale.__post_init__()
        if (
            scale.stage_result.operator != _CONTRACTION_NAME
            or scale.stage_result.glyph != _CONTRACTION_GLYPH
            or not proof_stamps_are_identical(
                original._proof_stamp, scale.before._proof_stamp
            )
            or not proof_stamps_are_identical(
                contracted._proof_stamp, scale.after._proof_stamp
            )
        ):
            raise ValueError("retained synthesis Contraction boundaries do not match")
        if self.before.reception_kinds != self.contracted.reception_kinds:
            raise ValueError("Contraction must preserve both Reception kinds")
        if not _same(
            (
                _member_payload(contracted.member_states[1]),
                contracted.member_metadata[1],
            ),
            (_member_payload(after.member_states[1]), after.member_metadata[1]),
        ):
            raise ValueError("Reception must preserve the other member's own state")
        if self.contracted.reception_kinds[1] != self.after.reception_kinds[1]:
            raise ValueError("Reception must preserve the other member's kind")
        core_before, donor = contracted.member_states
        core_after = after.member_states[0]
        if not _same(
            (core_before.capacity, core_before.pressure),
            (core_after.capacity, core_after.pressure),
        ):
            raise ValueError("Reception must preserve core capacity and pressure")
        reception = self.reception_result
        if type(reception) is not NetworkStageResult:
            raise TypeError("retained synthesis requires its Reception result")
        reception.__post_init__()
        if (
            reception.operator != _CANONICAL_RECEPTION[2]
            or reception.glyph != _CANONICAL_RECEPTION[3].value
            or reception.schedule != TWO_PHASE_JACOBI
            or reception.nodes_processed != 1
            or len(reception.reception_observations) != 1
        ):
            raise ValueError("retained synthesis requires one canonical Reception")
        observed = reception.reception_observations[0]
        if (
            not observed._proof_fields_are_intact()
            or observed.source_tracking_enabled
            or observed.compatibility_decision is not None
            or not _same_fields(
                (
                    observed.node,
                    observed.source_nodes,
                    observed.neighbors,
                    observed.target_epi_before,
                    observed.target_epi_kind_before,
                    observed.neighbor_epi_values,
                    observed.neighbor_epi_kinds,
                    observed.target_epi_after,
                    observed.target_epi_kind_after,
                ),
                (
                    self.before.core,
                    (self.before.member,),
                    (self.before.member,),
                    core_before.epi,
                    self.contracted.reception_kinds[0],
                    (donor.epi,),
                    (self.contracted.reception_kinds[1],),
                    core_after.epi,
                    self.after.reception_kinds[0],
                ),
            )
        ):
            raise ValueError(
                "Reception read does not match the retained pair boundaries"
            )


def _options(raw: Mapping[str, Any] | None, label: str) -> dict[str, Any]:
    if raw is not None and not isinstance(raw, Mapping):
        raise TypeError(f"{label} must be a mapping or None")
    options = {} if raw is None else dict(raw)
    if any(type(key) is not str or key not in _OPTIONS for key in options):
        raise ValueError(f"unsupported retained synthesis {label}")
    return options


def _canonical_reception() -> Reception:
    call, execute, name, glyph = _CANONICAL_RECEPTION
    operator = Reception()
    if (
        type(operator) is not Reception
        or Reception.__call__ is not call
        or Reception._execute is not execute
        or operator.name != name
        or operator.glyph is not glyph
        or any(
            key in vars(operator) for key in ("__call__", "_execute", "name", "glyph")
        )
    ):
        raise TypeError("retained synthesis requires unmodified canonical Reception")
    return operator


def execute_retained_synthesis_stage(
    graph: Any,
    pair: RetainedPairState,
    *,
    contraction_kwargs: Mapping[str, Any] | None = None,
    reception_kwargs: Mapping[str, Any] | None = None,
) -> RetainedSynthesisDispatch:
    """Execute actual NUL(pair), then EN(core, member-only), atomically.

    Incoming realized member support is required; no reverse arc is supplied.
    Original support, nonmember state and phases remain intact. Each actual
    operator keeps its own grammar and factor domain. No context, source,
    callback or fallback override is admitted. Graph telemetry may change;
    external side effects remain outside the shared transaction contract.
    """
    transaction = GraphTransactionSnapshot(graph)
    try:
        nul_options = _options(contraction_kwargs, "contraction_kwargs")
        en_options = _options(reception_kwargs, "reception_kwargs")
        _admit_pair(pair)
        reception = _canonical_reception()
        transaction.restore(graph)
        if pair.organization._graph_owner is not graph:
            raise ValueError("retained pair is foreign, stale, invalid or unsealed")
        before = capture_retained_pair(graph, pair.members)
        if (
            not proof_stamps_are_identical(
                pair.organization._proof_stamp, before.organization._proof_stamp
            )
            or pair.reception_kinds != before.reception_kinds
        ):
            raise ValueError("retained pair is foreign, stale, invalid or unsealed")
        if pair.member not in before.organization.incoming_ports[0]:
            raise ValueError("retained member must be an incoming source of the core")
        protected = _protected_state(graph, pair.members)
        scale = execute_retained_scale_stage(
            graph, before.organization, Contraction(), execution_kwargs=nul_options
        )
        contracted = capture_retained_pair(graph, pair.members)
        en_protected = _protected_state(graph, (pair.core,))
        result = execute_neighbor_stage(
            graph,
            reception,
            (pair.core,),
            source_nodes=(pair.member,),
            track_sources=False,
            transaction_snapshot=transaction,
            **en_options,
        )
        after = capture_retained_pair(graph, pair.members)
        if not proof_stamps_are_identical(
            protected, _protected_state(graph, pair.members)
        ) or not proof_stamps_are_identical(
            en_protected, _protected_state(graph, (pair.core,))
        ):
            raise RuntimeError("retained synthesis changed protected state or ports")
        return RetainedSynthesisDispatch(before, contracted, after, scale, result)
    except BaseException as failure:
        transaction.restore_after_failure(graph, failure)
        raise
