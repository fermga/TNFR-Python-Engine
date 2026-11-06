"""Pure, sealed source decisions for an opt-in Reception compatibility family.

The supplied phase and capacity tolerances define a policy, not a derived U3
law. Selection reads every incoming candidate before any Reception form write.
It does not admit EPI, grammar, activity or the ensuing operator event.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from fractions import Fraction
from typing import Any

from .._exact_time import (
    bounded_represented_real,
    finite_represented_real,
    represented_fraction,
)
from ..alias import get_attr
from ..constants.aliases import ALIAS_THETA, ALIAS_VF
from ..utils._structural_signature import (
    proof_stamps_are_identical,
    structural_proof_signature,
)
from ..utils.numeric import angle_diff

_DECISION_VERSION = "reception_compatibility_v1"
_MISSING = object()


@dataclass(frozen=True, slots=True)
class ReceptionCompatibilityPolicy:
    """Required tolerances for circular phase and relative capacity contrast.

    ``select`` keeps a compatible subsequence in incoming support order;
    ``require_all`` abstains unless the complete nonempty set is compatible.
    Capacity is nonnegative structural reorganization capacity, not frequency
    observed against an independent clock. No tolerance defaults are supplied.
    """

    max_phase_gap: float
    max_relative_capacity_gap: float
    mode: str = "select"

    def __post_init__(self) -> None:
        phase, _ = bounded_represented_real(
            self.max_phase_gap, "Reception max_phase_gap", minimum=0, maximum=math.pi
        )
        capacity, _ = bounded_represented_real(
            self.max_relative_capacity_gap,
            "Reception max_relative_capacity_gap",
            minimum=0,
            maximum=1,
        )
        if type(self.mode) is not str or self.mode not in ("select", "require_all"):
            raise ValueError(
                "Reception compatibility mode must be select or require_all"
            )
        object.__setattr__(self, "max_phase_gap", phase)
        object.__setattr__(self, "max_relative_capacity_gap", capacity)


def _policy_payload(policy: ReceptionCompatibilityPolicy) -> tuple[Any, ...]:
    if type(policy) is not ReceptionCompatibilityPolicy:
        raise TypeError("Reception compatibility requires a canonical policy")
    phase = represented_fraction(policy.max_phase_gap, "Reception max_phase_gap")
    capacity = represented_fraction(
        policy.max_relative_capacity_gap, "Reception max_relative_capacity_gap"
    )
    if not 0 <= phase <= Fraction.from_float(math.pi) or not 0 <= capacity <= 1:
        raise ValueError("Reception compatibility policy thresholds are out of bounds")
    if type(policy.mode) is not str or policy.mode not in ("select", "require_all"):
        raise ValueError("Reception compatibility policy mode is invalid")
    return policy.max_phase_gap, policy.max_relative_capacity_gap, policy.mode


def _selected_sources(
    policy: ReceptionCompatibilityPolicy,
    target_phase: float,
    target_capacity: float,
    candidates: tuple[Any, ...],
    phases: tuple[float, ...],
    capacities: tuple[float, ...],
) -> tuple[tuple[Any, ...], str]:
    """Compare represented capacities exactly, without division or tolerance."""

    eta = Fraction.from_float(policy.max_relative_capacity_gap)
    target = Fraction.from_float(target_capacity)
    selected = []
    for candidate, phase, capacity in zip(candidates, phases, capacities):
        # Evaluate every phase gap even when capacity alone excludes this source:
        # invalid/unrepresentable consumed arithmetic is not incompatibility.
        gap = abs(angle_diff(phase, target_phase))
        if not math.isfinite(gap):
            raise ValueError("Reception circular phase gap must be finite")
        source = Fraction.from_float(capacity)
        if gap <= policy.max_phase_gap and abs(source - target) <= eta * max(
            source, target
        ):
            selected.append(candidate)
    if not candidates:
        return (), "no_candidates"
    if policy.mode == "require_all" and len(selected) != len(candidates):
        return (), "not_all_compatible"
    if not selected:
        return (), "no_compatible_sources"
    return tuple(selected), "selected"


def _reception_compatibility_payload_stamp(
    decision: ReceptionCompatibilityDecision,
) -> tuple[Any, ...]:
    """Seal the complete compatibility read without traversing a graph owner."""

    return structural_proof_signature(
        (
            _DECISION_VERSION,
            decision.node,
            _policy_payload(decision.policy),
            decision.target_phase,
            decision.target_capacity,
            decision.candidates,
            decision.candidate_phases,
            decision.candidate_capacities,
            decision.selected_sources,
            decision.reason,
        )
    )


@dataclass(frozen=True, slots=True)
class ReceptionCompatibilityDecision:
    """A sealed phase/capacity decision, not admission of a Reception event."""

    node: Any
    policy: ReceptionCompatibilityPolicy
    target_phase: float
    target_capacity: float
    candidates: tuple[Any, ...]
    candidate_phases: tuple[float, ...]
    candidate_capacities: tuple[float, ...]
    selected_sources: tuple[Any, ...]
    reason: str
    _proof_stamp: tuple[Any, ...] = field(default=(), repr=False, compare=False)

    def __post_init__(self) -> None:
        self._validate_payload()

    def _validate_payload(self) -> None:
        _policy_payload(self.policy)
        hash(self.node)
        if any(
            type(values) is not tuple
            for values in (
                self.candidates,
                self.candidate_phases,
                self.candidate_capacities,
                self.selected_sources,
            )
        ):
            raise TypeError("Reception compatibility sequences must be tuples")
        if len(set(self.candidates)) != len(self.candidates):
            raise ValueError("Reception compatibility candidates must be unique")
        if len(self.candidates) != len(self.candidate_phases) or len(
            self.candidates
        ) != len(self.candidate_capacities):
            raise ValueError("Reception compatibility source coordinates must align")
        for phase in (self.target_phase, *self.candidate_phases):
            represented_fraction(phase, "Reception phase")
        for capacity in (self.target_capacity, *self.candidate_capacities):
            if represented_fraction(capacity, "Reception capacity") < 0:
                raise ValueError("Reception capacity must be nonnegative")
        selected, reason = _selected_sources(
            self.policy,
            self.target_phase,
            self.target_capacity,
            self.candidates,
            self.candidate_phases,
            self.candidate_capacities,
        )
        if type(self.reason) is not str or not proof_stamps_are_identical(
            structural_proof_signature((self.selected_sources, self.reason)),
            structural_proof_signature((selected, reason)),
        ):
            raise ValueError("Reception compatibility result contradicts its inputs")

    @property
    def eligible(self) -> bool:
        return bool(self.selected_sources)

    @property
    def abstention_reason(self) -> str | None:
        return None if self.eligible else self.reason

    def _proof_fields_are_intact(self) -> bool:
        try:
            if type(self) is not ReceptionCompatibilityDecision:
                return False
            self._validate_payload()
            return proof_stamps_are_identical(
                self._proof_stamp, _reception_compatibility_payload_stamp(self)
            )
        except BaseException:
            return False


def _read_phase_capacity(graph: Any, node: Any) -> tuple[float, float]:
    def required(aliases: tuple[str, ...], label: str) -> Any:
        value = get_attr(
            graph.nodes[node], aliases, _MISSING, strict=True, conv=lambda item: item
        )
        if value is _MISSING:
            raise ValueError(f"Reception compatibility requires explicit {label}")
        return value

    phase, _ = finite_represented_real(
        required(ALIAS_THETA, "phase"), "Reception phase"
    )
    capacity, _ = bounded_represented_real(
        required(ALIAS_VF, "capacity"), "Reception capacity", minimum=0
    )
    return phase, capacity


def observe_reception_compatibility(
    graph: Any, node: Any, policy: ReceptionCompatibilityPolicy
) -> ReceptionCompatibilityDecision:
    """Read every realized incoming phase/capacity and select without mutation.

    Missing or invalid consumed state raises. Valid empty or incompatible support
    produces an explicit abstention. EPI, semantic kinds and execution grammar are
    not consumed; a selected decision does not guarantee later event admission.
    """

    # Delayed import retains the single incoming-support owner; the EN snapshot
    # imports this pure decision type while initializing its own module.
    from ._reception_kernel import reception_input_neighbors

    _policy_payload(policy)
    target_phase, target_capacity = _read_phase_capacity(graph, node)
    candidates = reception_input_neighbors(graph, node)
    coordinates = tuple(_read_phase_capacity(graph, source) for source in candidates)
    phases = tuple(value[0] for value in coordinates)
    capacities = tuple(value[1] for value in coordinates)
    selected, reason = _selected_sources(
        policy, target_phase, target_capacity, candidates, phases, capacities
    )
    decision = ReceptionCompatibilityDecision(
        node=node,
        policy=policy,
        target_phase=target_phase,
        target_capacity=target_capacity,
        candidates=candidates,
        candidate_phases=phases,
        candidate_capacities=capacities,
        selected_sources=selected,
        reason=reason,
    )
    return replace(
        decision, _proof_stamp=_reception_compatibility_payload_stamp(decision)
    )


def reception_compatibility_matches_graph(
    graph: Any, decision: ReceptionCompatibilityDecision
) -> bool:
    """Revalidate the complete read, including excluded sources and their order.

    An equivalent detached graph can match. The containing EN snapshot separately
    binds graph ownership, EPI and kind. This helper does not execute an event.
    """

    try:
        if (
            type(decision) is not ReceptionCompatibilityDecision
            or not decision._proof_fields_are_intact()
        ):
            return False
        fresh = observe_reception_compatibility(graph, decision.node, decision.policy)
        return proof_stamps_are_identical(
            decision._proof_stamp, _reception_compatibility_payload_stamp(fresh)
        )
    except BaseException:
        return False


__all__ = [
    "ReceptionCompatibilityPolicy",
    "ReceptionCompatibilityDecision",
    "observe_reception_compatibility",
    "reception_compatibility_matches_graph",
]
