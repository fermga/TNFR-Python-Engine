"""Native-only Mutation occurrence and unavailable-request policies.

This policy does not change Mutation's shared trigger or intervention APIs.
Interval keys use the certificate's represented source/times, never sampled
values, history length, a supplied decision clock or inferred event duration.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .._exact_time import finite_represented_real

if TYPE_CHECKING:
    from ..physics.mutation_trigger import MutationTriggerCertificate

POLICY_KEY = "NATIVE_ZHIR_OCCURRENCE_POLICY"
ABSTENTION_POLICY_KEY = "NATIVE_ZHIR_ABSTENTION_POLICY"
CONSUMED_INTERVALS_KEY = "_native_zhir_consumed_intervals"
LEVEL_POLICY = "level"
CONSUME_POLICY = "consume_timestamped_interval"
COHERENCE_ABSTENTION = "coherence"
SKIP_ABSTENTION = "skip"
MutationInterval = tuple[str, float, float]


def _field(mapping, key, default):
    if type(mapping) in (dict, OrderedDict):
        return dict.get(mapping, key, default)
    from ..operators.network_stage import _runtime_mapping_known_value

    return _runtime_mapping_known_value(mapping, key, default)


def native_mutation_occurrence_policy(graph_data) -> str:
    # Preserve the graph configuration's existing mapping interface in the
    # default path; only opted-in nodal receipts need the strict state reader.
    policy = graph_data.get(POLICY_KEY, LEVEL_POLICY)
    if type(policy) is not str or policy not in (LEVEL_POLICY, CONSUME_POLICY):
        raise ValueError(
            f"{POLICY_KEY} must be 'level' or 'consume_timestamped_interval'"
        )
    return policy


def native_mutation_abstention_policy(graph_data) -> str:
    """Admit the response to a blocked native ZHIR request, without writes.

    Coherence preserves the historical real IL substitution. Skip requests no
    replacement action; neither choice changes the primary selector or gate.
    """
    policy = graph_data.get(ABSTENTION_POLICY_KEY, COHERENCE_ABSTENTION)
    if type(policy) is not str or policy not in (
        COHERENCE_ABSTENTION,
        SKIP_ABSTENTION,
    ):
        raise ValueError(f"{ABSTENTION_POLICY_KEY} must be 'coherence' or 'skip'")
    return policy


def _admit_interval(value: Any) -> MutationInterval:
    if type(value) is not tuple or len(value) != 3:
        raise TypeError("consumed Mutation interval must be an exact three-item tuple")
    source, left, right = value
    if type(source) is not str or source != "epi_time_history":
        raise ValueError("consumed Mutation interval requires epi_time_history source")
    left_value = finite_represented_real(left, "Mutation interval previous time")[0]
    right_value = finite_represented_real(right, "Mutation interval current time")[0]
    if left_value >= right_value:
        raise ValueError("consumed Mutation interval times must increase strictly")
    return source, left_value, right_value


def consumed_mutation_intervals(node_data) -> tuple[MutationInterval, ...]:
    raw = _field(node_data, CONSUMED_INTERVALS_KEY, ())
    if type(raw) is not tuple:
        raise TypeError(f"{CONSUMED_INTERVALS_KEY} must be an exact tuple")
    # Admit all scalars before comparing keys: True must not alias time 1.
    admitted = tuple(_admit_interval(item) for item in raw)
    if len(set(admitted)) != len(admitted):
        raise ValueError("consumed Mutation intervals must not contain duplicates")
    return admitted


def preflight_native_mutation_occurrence(graph, nodes) -> str:
    policy = native_mutation_occurrence_policy(graph.graph)
    if policy == CONSUME_POLICY:
        for node in nodes:
            consumed_mutation_intervals(graph.nodes[node])
    return policy


@dataclass(frozen=True, slots=True)
class MutationOccurrenceAdmission:
    policy: str
    interval: MutationInterval | None
    previously_consumed: bool | None
    consumed_intervals: tuple[MutationInterval, ...]
    reason: str | None


def admit_native_mutation_occurrence(
    node_data, certificate: MutationTriggerCertificate, policy: str
) -> MutationOccurrenceAdmission:
    if policy == LEVEL_POLICY:
        return MutationOccurrenceAdmission(policy, None, None, (), None)
    if policy != CONSUME_POLICY:
        raise ValueError("unknown native Mutation occurrence policy")
    ledger = consumed_mutation_intervals(node_data)
    evidence = certificate.evidence
    if (
        evidence is None
        or not certificate.evidence_valid
        or not certificate.physical_time_resolved
        or certificate.source != "epi_time_history"
        or certificate.time_basis != "physical_time"
        or evidence.previous_time is None
        or evidence.current_time is None
    ):
        return MutationOccurrenceAdmission(
            policy, None, None, ledger, "unidentified_mutation_interval"
        )
    interval = _admit_interval(
        (evidence.source, evidence.previous_time, evidence.current_time)
    )
    consumed = interval in ledger
    return MutationOccurrenceAdmission(
        policy,
        interval,
        consumed,
        ledger,
        "mutation_interval_already_consumed" if consumed else None,
    )


def commit_native_mutation_interval(node_data, admission) -> None:
    """Record one successful direct ZHIR return before later bookkeeping.

    The receipt belongs to this node's retained memory. No automatic history
    truncation, mode switch or glyph-window reset erases earlier consumption.
    External ledger mutation during dispatch rejects this commit after the
    primitive's effects; it is outside the intact-memory at-most-once contract.
    """
    if (
        type(admission) is not MutationOccurrenceAdmission
        or admission.policy != CONSUME_POLICY
        or admission.interval is None
        or admission.reason is not None
        or admission.previously_consumed is not False
    ):
        raise ValueError("Mutation consumption requires admitted unused own interval")
    current = consumed_mutation_intervals(node_data)
    if current != admission.consumed_intervals:
        raise RuntimeError("Mutation consumption memory changed during native dispatch")
    from ..operators.network_stage import _set_runtime_mapping_item

    _set_runtime_mapping_item(
        node_data, CONSUMED_INTERVALS_KEY, (*current, admission.interval)
    )
