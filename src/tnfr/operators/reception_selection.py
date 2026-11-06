"""Configured EN source compatibility and one conditional canonical attempt.

Compatibility selects realized sources at a supplied invocation boundary.
The existing atomic EN stage owns the action, grammar and rollback. Source
absence is an abstention, not an empty EN event or an autonomous event clock.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ..rng import validate_graph_seed
from ..types import Glyph
from ..utils._structural_signature import (
    proof_stamps_are_identical,
    structural_proof_signature,
)
from ._argument_validation import (
    nonnegative_integer,
    strict_bool,
    validate_common_execution_arguments,
)
from ._reception_compatibility import (
    ReceptionCompatibilityDecision,
    ReceptionCompatibilityPolicy,
    _reception_compatibility_payload_stamp,
    observe_reception_compatibility,
)
from .network_stage import (
    TWO_PHASE_JACOBI,
    GraphTransactionSnapshot,
    NetworkStageResult,
    _detached_stage_graph,
    execute_neighbor_stage,
)
from .reception import Reception

__all__ = [
    "ReceptionCompatibilityPolicy",
    "ReceptionCompatibilityDecision",
    "EligibleReceptionDispatch",
    "observe_reception_compatibility",
    "execute_eligible_reception_stage",
]

_OPTIONS = frozenset(
    {
        "window",
        "validate_preconditions",
        "collect_metrics",
        "validate_nodal_equation",
        "dt",
        "track_sources",
        "max_distance",
    }
)


@dataclass(frozen=True, slots=True)
class EligibleReceptionDispatch:
    """One supplied attempt, with source abstention or a real EN stage.

    The source decision does not certify grammar, future stability or autonomous
    occurrence. A failed nonempty attempt raises and returns no successful record.
    """

    compatibility: ReceptionCompatibilityDecision
    stage_result: NetworkStageResult | None

    def __post_init__(self) -> None:
        decision = self.compatibility
        if (
            type(decision) is not ReceptionCompatibilityDecision
            or not decision._proof_fields_are_intact()
        ):
            raise ValueError("Reception dispatch requires an intact source decision")
        stage = self.stage_result
        if not decision.selected_sources:
            if stage is not None:
                raise ValueError("A source-rule abstention cannot claim an EN stage")
            return
        if (
            type(stage) is not NetworkStageResult
            or stage.operator != Reception.name
            or stage.glyph != Glyph.EN.value
            or stage.schedule != TWO_PHASE_JACOBI
            or stage.nodes_processed != 1
            or len(stage.reception_observations) != 1
            or stage.epi_jump_certificate is not None
        ):
            raise ValueError("Selected Reception requires one canonical EN stage")
        observation = stage.reception_observations[0]
        retained = observation.compatibility_decision
        if (
            not observation._proof_fields_are_intact()
            or observation.source_selection_origin != "compatibility_policy"
            or type(retained) is not ReceptionCompatibilityDecision
            or not proof_stamps_are_identical(
                _reception_compatibility_payload_stamp(retained),
                _reception_compatibility_payload_stamp(decision),
            )
            or not proof_stamps_are_identical(
                structural_proof_signature(
                    (observation.node, observation.source_nodes)
                ),
                structural_proof_signature((decision.node, decision.selected_sources)),
            )
        ):
            raise ValueError(
                "Reception stage does not retain its compatibility decision"
            )

    @property
    def executed(self) -> bool:
        """Whether this attempt returned an actual committed EN stage."""
        return self.stage_result is not None


def _materialize_options(options: Mapping[str, Any] | None) -> dict[str, Any]:
    if options is None:
        return {}
    if not isinstance(options, Mapping):
        raise TypeError("execution_kwargs must be a mapping or None")
    result = dict(options)
    if any(type(key) is not str or key not in _OPTIONS for key in result):
        raise ValueError("unsupported Reception compatibility execution option")
    return result


def execute_eligible_reception_stage(
    graph: Any,
    node: Any,
    policy: ReceptionCompatibilityPolicy,
    *,
    execution_kwargs: Mapping[str, Any] | None = None,
    compute_delta_nfr: Any = None,
    include_epi_jump_certificate: bool = False,
) -> EligibleReceptionDispatch:
    """Attempt one EN at supplied time using current compatible sources.

    The source observer reads all incoming phase/capacity associations; the
    atomic stage revalidates that decision before reading selected forms.
    No selected sources means no stage, history increment, schedule write or
    pressure callback. With sources, normal grammar and precondition errors
    reject; an alternative glyph cannot satisfy this request. All graph-owned
    failures roll back, including a late pressure callback. External effects
    outside the graph are not transactional. Neither this function nor the
    pure observer installs a recurring selector, clock or continuous law.
    """
    transaction = GraphTransactionSnapshot(graph)
    try:
        options = _materialize_options(execution_kwargs)
        strict_bool(
            include_epi_jump_certificate,
            operator="Reception",
            label="include_epi_jump_certificate",
        )
        if compute_delta_nfr is not None and not callable(compute_delta_nfr):
            raise TypeError("compute_delta_nfr must be callable or None")
        transaction.restore(graph)
        snapshot = _detached_stage_graph(graph)
        from . import _validated_execution_window

        options["window"] = _validated_execution_window(snapshot, options.get("window"))
        validate_graph_seed(snapshot)
        validate_common_execution_arguments(
            snapshot.graph, options, operator="Reception"
        )
        if "track_sources" in options:
            strict_bool(
                options["track_sources"], operator="Reception", label="track_sources"
            )
        if "max_distance" in options:
            nonnegative_integer(
                options["max_distance"], operator="Reception", label="max_distance"
            )
        decision = observe_reception_compatibility(snapshot, node, policy)
        transaction.restore(graph)
        if not decision.selected_sources:
            return EligibleReceptionDispatch(decision, None)
        stage = execute_neighbor_stage(
            graph,
            Reception(),
            (node,),
            source_nodes=decision.selected_sources,
            _reception_compatibility_decision=decision,
            transaction_snapshot=transaction,
            compute_delta_nfr=compute_delta_nfr,
            include_epi_jump_certificate=include_epi_jump_certificate,
            **options,
        )
        return EligibleReceptionDispatch(decision, stage)
    except BaseException as failure:
        transaction.restore_after_failure(graph, failure)
        raise
