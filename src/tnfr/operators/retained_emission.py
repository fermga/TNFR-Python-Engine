"""Realize a supplied retained-group mean gain through one actual member AL.

The address and ordinary AL factor are supplied. This exact represented-gain
domain adds no selector, macro-node, factor override or collective grammar.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from fractions import Fraction
from typing import Any

from .._exact_time import finite_represented_real
from ..alias import get_attr
from ..config.parsing import parse_bool
from ..constants import get_param
from ..constants.aliases import (
    ALIAS_DNFR,
    ALIAS_EMISSION_TIMESTAMP,
    ALIAS_EPI_KIND,
    ALIAS_VF,
)
from ..utils._structural_signature import proof_stamps_are_identical
from ._argument_validation import finite_real, validate_common_execution_arguments
from .al_sha_stage_proposals import _LATENCY_KEYS, propose_emission_stage
from .emission import Emission
from .factor_contracts import resolve_runtime_operator_factors
from .grammar_debt import advance_debt, advance_prior_coherence, read_grammar_state
from .network_stage import (
    TWO_PHASE_JACOBI,
    GraphTransactionSnapshot,
    NetworkStageResult,
    execute_pointwise_stage,
)
from .preconditions.emission import validate_emission_strict
from .retained_organization import (
    _OPTIONS,
    RetainedOrganization,
    _member_payload,
    _protected_state,
    _readmit_retained_organization,
    _signature,
    capture_retained_organization,
)
from .retained_synthesis import _boundary, _same

__all__ = ["RetainedEmissionDispatch", "execute_retained_emission_stage"]

_OPERATOR = "Retained Emission"
_CANONICAL_EMISSION = Emission
_METHODS = (
    "__call__",
    "_execute",
    "_validate_application_preconditions",
    "_validate_requested_preconditions",
    "_validate_preconditions",
    "_validate_requested_hard_invariants",
    "_validate_hard_invariants",
)
_BINDINGS = tuple(getattr(Emission, key) for key in _METHODS)
_NAME, _GLYPH = Emission.name, Emission.glyph
_LIFECYCLE_KEYS = (
    *_LATENCY_KEYS,
    *ALIAS_EMISSION_TIMESTAMP,
    "_emission_activated",
    "_emission_origin",
    "_structural_lineage",
)


def _index(value: Any, size: int) -> int:
    if type(value) is not int:
        raise TypeError("retained Emission member_index must be an exact integer")
    if value < 0 or value >= size:
        raise IndexError("retained Emission member_index is out of range")
    return value


def _admit_record(record: Any) -> None:
    if type(record) is not RetainedOrganization:
        raise TypeError("retained Emission requires an exact RetainedOrganization")
    if not record._proof_fields_are_intact():
        raise ValueError("retained Emission record is invalid or unsealed")


def _exact_gain(value: Any) -> Fraction:
    represented, exact = finite_represented_real(value, "retained mean_gain")
    if represented <= 0:
        raise ValueError("retained mean_gain must be positive")
    return exact


def _canonical_emission() -> Emission:
    if Emission is not _CANONICAL_EMISSION:
        raise TypeError("retained Emission requires the exact canonical class")
    operator = Emission()
    if (
        type(operator) is not Emission
        or any(
            getattr(Emission, key) is not method
            for key, method in zip(_METHODS, _BINDINGS)
        )
        or operator.name != _NAME
        or operator.glyph is not _GLYPH
        or any(key in vars(operator) for key in (*_METHODS, "name", "glyph"))
    ):
        raise TypeError("retained Emission requires unmodified canonical bindings")
    return operator


def _materialize_options(raw: Mapping[str, Any] | None) -> dict[str, Any]:
    if raw is not None and not isinstance(raw, Mapping):
        raise TypeError("execution_kwargs must be a mapping or None")
    options = {} if raw is None else dict(raw)
    if any(type(key) is not str or key not in _OPTIONS for key in options):
        raise ValueError("unsupported retained Emission execution option")
    return options


def _admit_options(graph: Any, options: dict[str, Any]) -> dict[str, Any]:
    from . import _validated_execution_window

    options.setdefault("validate_preconditions", True)
    validate_common_execution_arguments(graph.graph, options, operator=_OPERATOR)
    if not options["validate_preconditions"]:
        raise ValueError("retained Emission requires validate_preconditions=True")
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
    if (
        not parse_bool(canon.get("enabled", False))
        or graph.graph.get("GRAMMAR_REJECTION_MODE") != "raise"
    ):
        raise ValueError("retained Emission requires enabled grammar and raise mode")
    return options


def _held_target(graph: Any, node: Any) -> Any:
    data = graph.nodes[node]
    return _signature(
        graph,
        tuple(
            (key, data[key])
            for key in (*ALIAS_VF, *ALIAS_DNFR, *ALIAS_EPI_KIND)
            if key in data
        ),
    )


def _configuration(graph: Any) -> Any:
    return _signature(
        graph,
        tuple(
            (key, graph.graph[key])
            for key in ("GLYPH_FACTORS", "_t")
            if key in graph.graph
        ),
    )


def _lineage_values(lineage: Mapping[str, Any], count: int | None = None) -> Any:
    # Retain opaque constituent identities in lineage values; never deepcopy
    # them to construct an expected lifecycle. The signature freezes values.
    return (
        type(lineage),
        tuple(
            (key, count if key == "activation_count" and count is not None else value)
            for key, value in lineage.items()
        ),
    )


def _lifecycle_values(data: Mapping[str, Any]) -> dict[str, Any]:
    values = {key: data[key] for key in _LIFECYCLE_KEYS if key in data}
    lineage = values.get("_structural_lineage")
    if isinstance(lineage, Mapping):
        values["_structural_lineage"] = _lineage_values(lineage)
    return values


def _prospective_lifecycle(graph: Any, node: Any, proposal: Any) -> Any:
    expected = _lifecycle_values(graph.nodes[node])
    for key in proposal.latency_keys_to_clear:
        expected.pop(key, None)
    if proposal.initialize_emission:
        expected.pop("_emission_activated", None)
    else:
        expected["_structural_lineage"] = _lineage_values(
            graph.nodes[node]["_structural_lineage"],
            proposal.lineage_activation_count_after,
        )
    return _signature(graph, expected)


def _verify_lifecycle(graph: Any, node: Any, expected: Any, proposal: Any) -> None:
    """Check the shared proposal's actual patch after all stage checks/hooks."""
    actual = graph.nodes[node]
    actual_values = _lifecycle_values(actual)
    if proposal.initialize_emission:
        timestamp = get_attr(
            actual,
            ALIAS_EMISSION_TIMESTAMP,
            None,
            strict=True,
            conv=lambda value: value,
        )
        if type(timestamp) is not str or not timestamp:
            raise RuntimeError("retained Emission lost its first activation timestamp")
        # set_attr_str writes the authoritative timestamp alias. No old alias
        # may exist on an admitted first activation with inactive provenance.
        created = {
            ALIAS_EMISSION_TIMESTAMP[0]: timestamp,
            "_emission_activated": True,
            "_emission_origin": timestamp,
            "_structural_lineage": _lineage_values(
                {
                    "origin": timestamp,
                    "activation_count": 1,
                    "derived_nodes": [],
                    "parent_emission": None,
                }
            ),
        }
        received = {key: actual_values.pop(key, None) for key in created}
        if not proof_stamps_are_identical(
            _signature(graph, created), _signature(graph, received)
        ):
            raise RuntimeError(
                "retained Emission changed its first activation lifecycle"
            )
    if not proof_stamps_are_identical(expected, _signature(graph, actual_values)):
        raise RuntimeError("retained Emission changed its counted activation lifecycle")


def _verify_grammar(before: Any, after: Any, window: int) -> None:
    unbounded = (*before.history, _GLYPH.value)
    if (
        after.operator_step != before.operator_step + 1
        or after.history != (unbounded[-window:] if window else ())
        or after.u2_debt != advance_debt(before.u2_debt, _GLYPH)
        or after.prior_coherence
        != advance_prior_coherence(before.prior_coherence, _GLYPH)
    ):
        raise RuntimeError("retained Emission changed its own AL grammar advance")


@dataclass(frozen=True, slots=True)
class RetainedEmissionDispatch:
    """Actual singleton AL realization; a detached summary grants no authority."""

    before: RetainedOrganization
    after: RetainedOrganization
    member_index: int
    mean_gain: Fraction
    local_boost: float
    stage_result: NetworkStageResult

    def __post_init__(self) -> None:
        _admit_record(self.before)
        _admit_record(self.after)
        _index(self.member_index, len(self.before.members))
        if (
            self.before._graph_owner is not self.after._graph_owner
            or not _same(self.before.members, self.after.members)
            or not _same(_boundary(self.before), _boundary(self.after))
        ):
            raise ValueError(
                "retained Emission must preserve member identities and ports"
            )
        if type(self.mean_gain) is not Fraction or self.mean_gain <= 0:
            raise TypeError(
                "retained Emission requires a positive exact represented gain"
            )
        boost = finite_represented_real(self.local_boost, "retained AL boost")[1]
        if boost != len(self.before.members) * self.mean_gain:
            raise ValueError("retained Emission configured boost contradicts mean gain")
        for index, (before, after) in enumerate(
            zip(self.before.member_states, self.after.member_states)
        ):
            if index == self.member_index:
                if not _same(
                    (before.capacity, before.pressure, before.epi_kind),
                    (after.capacity, after.pressure, after.epi_kind),
                ):
                    raise ValueError("retained Emission changed held member channels")
                if after.grammar.operator_step != before.grammar.operator_step + 1:
                    raise ValueError("retained Emission requires one own AL occurrence")
            elif not _same(
                (_member_payload(before), self.before.member_metadata[index]),
                (_member_payload(after), self.after.member_metadata[index]),
            ):
                raise ValueError("retained Emission changed a nonselected member")
        if self.realized_mean_gain != self.mean_gain:
            raise ValueError(
                "retained Emission actual mean gain does not match request"
            )
        stage = self.stage_result
        if type(stage) is not NetworkStageResult:
            raise TypeError("retained Emission requires its actual NetworkStageResult")
        stage.__post_init__()
        if (
            stage.operator != _NAME
            or stage.glyph != _GLYPH.value
            or stage.schedule != TWO_PHASE_JACOBI
            or stage.nodes_processed != 1
            or stage.pointwise_epi_jump_certificate is not None
            or stage.epi_jump_certificate is not None
            or stage.epi_jump_certificate_abstention_reason is not None
        ):
            raise ValueError("retained Emission requires its actual singleton AL stage")

    @property
    def member(self) -> Any:
        return self.before.members[self.member_index]

    @property
    def realized_mean_gain(self) -> Fraction:
        return sum(
            (
                Fraction(after.epi) - Fraction(before.epi)
                for before, after in zip(
                    self.before.member_states, self.after.member_states
                )
            ),
            Fraction(0),
        ) / len(self.before.members)


def execute_retained_emission_stage(
    graph: Any,
    record: RetainedOrganization,
    member_index: int,
    *,
    mean_gain: Any,
    execution_kwargs: Mapping[str, Any] | None = None,
) -> RetainedEmissionDispatch:
    """Execute one strict AL realizing an exact represented group-mean gain.

    The caller supplies the member ordinal and the ordinary live AL_boost.
    This consumer requires boost = member_count * mean_gain and an exact
    unprojected represented translation. Strict AL entry admission is applied
    even if the optional global precondition switch is disabled. No context
    pooling, substitute member, factor override or duration is supplied.

    Other node state, support, selected phase/capacity/pressure/kinds, configured
    factor source and runtime clock are protected. Actual own AL history and
    lifecycle remain distinct across representations. Graph-owned failure
    rolls back the invocation; external callback effects cannot be restored.
    """
    transaction = GraphTransactionSnapshot(graph)
    try:
        options = _materialize_options(execution_kwargs)
        transaction.restore(graph)
        # Protect the entry before potentially active scalar/factor reads.
        entry = _protected_state(graph, ())
        configuration = _configuration(graph)
        factor_source = graph.graph.get("GLYPH_FACTORS")
        options = _admit_options(graph, options)
        _admit_record(record)
        index = _index(member_index, len(record.members))
        gain = _exact_gain(mean_gain)
        operator = _canonical_emission()
        before = _readmit_retained_organization(graph, record)
        if before is None:
            raise ValueError(
                "retained organization is foreign, stale, invalid or unsealed"
            )
        node = before.members[index]
        protected = _protected_state(graph, (node,))
        held = _held_target(graph, node)
        grammar = read_grammar_state(graph.nodes[node])
        validate_emission_strict(graph, node)
        factors = resolve_runtime_operator_factors(factor_source, _GLYPH, graph.graph)
        boost, exact_boost = finite_represented_real(
            factors["AL_boost"], "retained AL boost"
        )
        if exact_boost != len(before.members) * gain:
            raise ValueError("configured AL_boost must equal member_count * mean_gain")
        proposal = propose_emission_stage(
            graph, node, factors, timestamp=datetime.now(timezone.utc).isoformat()
        )
        if Fraction(proposal.epi_after) - Fraction(proposal.epi_before) != exact_boost:
            raise ValueError(
                "retained Emission excludes projected or rounded translations"
            )
        if not proposal.initialize_emission and not proposal.increment_lineage:
            raise ValueError("retained Emission requires counted activation lineage")
        if (
            not proof_stamps_are_identical(entry, _protected_state(graph, ()))
            or not proof_stamps_are_identical(configuration, _configuration(graph))
            or graph.graph.get("GLYPH_FACTORS") is not factor_source
        ):
            raise RuntimeError("retained Emission admission changed source state")
        lifecycle = _prospective_lifecycle(graph, node, proposal)
        stage = execute_pointwise_stage(
            graph,
            operator,
            (node,),
            transaction_snapshot=transaction,
            require_exact_operator=True,
            **options,
        )
        if not proof_stamps_are_identical(protected, _protected_state(graph, (node,))):
            raise RuntimeError("retained Emission changed protected state or ports")
        if not proof_stamps_are_identical(held, _held_target(graph, node)):
            raise RuntimeError("retained Emission changed held target channels")
        if graph.graph.get(
            "GLYPH_FACTORS"
        ) is not factor_source or not proof_stamps_are_identical(
            configuration, _configuration(graph)
        ):
            raise RuntimeError(
                "retained Emission changed configuration or runtime clock"
            )
        _verify_lifecycle(graph, node, lifecycle, proposal)
        _verify_grammar(
            grammar, read_grammar_state(graph.nodes[node]), options["window"]
        )
        after = capture_retained_organization(graph, before.members)
        if after.member_states[index].epi != proposal.epi_after:
            raise RuntimeError("retained Emission changed its prospective AL form")
        return RetainedEmissionDispatch(before, after, index, gain, boost, stage)
    except BaseException as failure:
        transaction.restore_after_failure(graph, failure)
        raise
