"""Shared execution of canonical operator words across a network.

The executor owns the network-stage schedule used by SDK and physics probes.
When grammar retains the requested glyph, all thirteen canonical operators use
immutable all-target proposals. Recursivity commits only its deduplicated
node-level advisory; explicit delayed EPI mixing remains a separate network
operation. Grammar replacements retain the explicit operator-major
Gauss--Seidel schedule inside a failure-atomic transaction.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from operator import index as integer_index
from typing import TYPE_CHECKING, Any, Callable

import networkx as nx

from ..errors import TNFRValueError

if TYPE_CHECKING:
    from .network_stage import NetworkStageResult


def _validate_cycles(cycles: Any) -> int:
    """Return a nonnegative integer cycle count without boolean coercion."""

    if isinstance(cycles, bool):
        raise TNFRValueError(
            "cycles must be a nonnegative integer",
            context={"cycles": repr(cycles), "reason": "boolean_not_allowed"},
        )
    try:
        count = integer_index(cycles)
    except (OverflowError, TypeError, ValueError) as exc:
        raise TNFRValueError(
            "cycles must be a nonnegative integer",
            context={"cycles": repr(cycles), "reason": "not_an_integer"},
        ) from exc
    if count < 0:
        raise TNFRValueError(
            "cycles must be a nonnegative integer",
            context={"cycles": count, "reason": "negative"},
        )
    return count


def preflight_network_mutation_sequence(
    graph: nx.Graph,
    names: list[str],
    *,
    cycles: int,
) -> None:
    """Reject stale Mutation evidence after validating a nonnegative cycle count."""

    cycles = _validate_cycles(cycles)
    if cycles == 0:
        return

    from ._mutation_gate import validate_mutation_runtime_gate
    from .operator_contracts import StateChannel, contract_for
    from .preconditions import OperatorPreconditionError

    try:
        contracts = [contract_for(name) for name in names]
    except KeyError:
        # Preserve the registry diagnostic for an unknown operator.
        return
    mutation_positions = [
        index for index, contract in enumerate(contracts) if contract.glyph == "ZHIR"
    ]
    if not mutation_positions:
        return

    for node in list(graph.nodes()):
        node_data = graph.nodes[node]
        if node_data.get("epi_time_history") is not None:
            for mutation_index in mutation_positions:
                prior_contracts = contracts[:mutation_index]
                if cycles > 1:
                    prior_contracts = contracts + prior_contracts
                epi_writers = [
                    contract.english_name
                    for contract in prior_contracts
                    if contract.primary_channel is StateChannel.EPI
                ]
                if epi_writers:
                    raise TNFRValueError(
                        "Mutation sequence preflight failed.",
                        context={
                            "node": node,
                            "reason": "physical_mutation_evidence_would_be_stale",
                            "mutation_index": mutation_index,
                            "prior_epi_channel_operators": epi_writers,
                        },
                        suggestion=(
                            "Advance and record physical EPI time evidence after the "
                            "preceding EPI-channel operations, or use explicitly "
                            "legacy unit-step evidence."
                        ),
                    )
        try:
            validate_mutation_runtime_gate(node_data, graph.graph)
        except OperatorPreconditionError as exc:
            raise TNFRValueError(
                "Mutation sequence preflight failed.",
                context={
                    "node": node,
                    "reason": exc.reason,
                    "sequence": names,
                },
                suggestion=(
                    "Provide valid current Mutation evidence that strictly crosses "
                    "the configured threshold and an active structural frequency."
                ),
            ) from exc


def execute_network_operator_stage(
    graph: nx.Graph,
    operator: Any,
    targets: Sequence[Any],
    *,
    sequence_context: Any = None,
    compute_delta_nfr: Any = None,
) -> "NetworkStageResult":
    """Dispatch one fixed-target stage through the canonical network route.

    Callers own whole-word grammar validation and supply its per-step context.
    This helper centralizes stage selection only; every routed implementation
    retains its own preconditions, immutable proposal boundary and rollback.
    Recursivity remains advisory-only here. Delayed EPI mixing is exposed
    separately by apply_network_remesh.
    """

    from .network_stage import (
        POINTWISE_TWO_PHASE_GLYPHS,
        execute_coupling_stage,
        execute_dissonance_stage,
        execute_neighbor_stage,
        execute_operator_major_stage,
        execute_pointwise_stage,
        execute_recursivity_stage,
        execute_self_organization_stage,
    )

    kwargs = {
        "sequence_context": sequence_context,
        "compute_delta_nfr": compute_delta_nfr,
    }
    if operator.name in {"reception", "resonance"}:
        return execute_neighbor_stage(graph, operator, targets, **kwargs)
    if operator.name == "coupling":
        return execute_coupling_stage(graph, operator, targets, **kwargs)
    if operator.name == "dissonance":
        return execute_dissonance_stage(graph, operator, targets, **kwargs)
    if operator.name == "self_organization":
        return execute_self_organization_stage(
            graph,
            operator,
            targets,
            **kwargs,
        )
    if operator.name == "recursivity":
        return execute_recursivity_stage(graph, operator, targets, **kwargs)
    if operator.glyph in POINTWISE_TWO_PHASE_GLYPHS:
        return execute_pointwise_stage(graph, operator, targets, **kwargs)
    return execute_operator_major_stage(graph, operator, targets, **kwargs)


def run_network_sequence(
    graph: nx.Graph,
    operator_names: list[str],
    *,
    cycles: int = 1,
    validate: bool = True,
    suppress_birth_warnings: bool = False,
    context: dict[str, Any] | None = None,
    on_step: Callable[[str], None] | None = None,
) -> None:
    """Evolve all nodes with the declared operator-major stage schedule.

    When every target retains the requested glyph, all thirteen operators read one
    immutable stage snapshot, propose every target, validate every proposal,
    and commit the stage atomically. EN, IL and RA read neighbours; UM merges
    phase/topology proposals; OZ reduces local and propagated pressure; THOL
    merges child support and hierarchy; AL, SHA, VAL, NUL, ZHIR and NAV use
    pointwise proposals; and REMESH deduplicates its graph advisory. Grammar
    replacements update in graph iteration order inside a stage-level rollback
    boundary.
    Complete-word grammar validation is optional for compatibility; live
    operator preconditions remain active in both modes.
    ``cycles`` must be a non-boolean nonnegative integer; zero validates the
    declared word without applying a stage.
    """

    from ..validation import validate_sequence
    from .grammar_execution import ValidatedSequence
    from .registry import get_operator_class

    cycles = _validate_cycles(cycles)
    names = list(operator_names)
    if not names:
        return
    ops = [get_operator_class(name)() for name in names]
    execution_word = ValidatedSequence(ops, context=context) if validate else None
    if validate:
        outcome = validate_sequence(names, context=context)
        if not outcome.passed:
            raise TNFRValueError(
                "Invalid sequence: "
                + outcome.summary.get("message", "validation failed"),
                context={"sequence": names, "outcome": outcome.summary},
            )
    preflight_network_mutation_sequence(graph, names, cycles=cycles)
    compute = graph.graph.get("compute_delta_nfr")
    nodes = list(graph.nodes())
    with warnings.catch_warnings():
        if suppress_birth_warnings:
            warnings.filterwarnings("ignore", message=r".*has no sources.*")
        for _ in range(cycles):
            for index, operator in enumerate(ops):
                sequence_step = (
                    None if execution_word is None else execution_word.step(index)
                )
                execute_network_operator_stage(
                    graph,
                    operator,
                    nodes,
                    sequence_context=sequence_step,
                    compute_delta_nfr=compute,
                )
                if on_step is not None:
                    on_step(operator.name)


__all__ = [
    "execute_network_operator_stage",
    "preflight_network_mutation_sequence",
    "run_network_sequence",
]
