"""Read-only grammar/history observation reports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..config.operator_names import BIFURCATION_WINDOW
from .grammar_debt import node_debt, node_has_prior_coherence
from .grammar_dynamics import validate_sequence_incremental
from .grammar_types import DESTABILIZERS, glyph_function_name
from .grammar_validate import validate_grammar
from .operator_contracts import contract_for

__all__ = ["GrammarObservation", "observe_grammar"]


@dataclass(frozen=True)
class GrammarObservation:
    """Separate grammar, history, contract and telemetry facts."""

    sequence_valid: bool
    sequence_message: str
    accepted_history: tuple[str, ...]
    history_length: int
    u2_debt: int
    prior_coherence: bool
    incremental_allowed: tuple[bool, ...]
    phase_gate_requested: bool
    phase_preconditions_checked: bool
    phase_gate_allowed: bool | None
    max_recursivity_depth: int
    recent_destabilizer: str | None
    recent_destabilizer_distance: int | None
    declared_contracts: tuple[str, ...]
    contract_postconditions_checked: bool
    contract_satisfied: bool | None
    trajectory_telemetry_present: bool
    u6_checked: bool
    scope: str = "read_only_observation"

    def as_dict(self) -> dict[str, Any]:
        """Return a detached report suitable for telemetry serialization."""
        return {
            "sequence_valid": self.sequence_valid,
            "sequence_message": self.sequence_message,
            "accepted_history": list(self.accepted_history),
            "history_length": self.history_length,
            "u2_debt": self.u2_debt,
            "prior_coherence": self.prior_coherence,
            "incremental_allowed": list(self.incremental_allowed),
            "phase_gate_requested": self.phase_gate_requested,
            "phase_preconditions_checked": self.phase_preconditions_checked,
            "phase_gate_allowed": self.phase_gate_allowed,
            "max_recursivity_depth": self.max_recursivity_depth,
            "recent_destabilizer": self.recent_destabilizer,
            "recent_destabilizer_distance": self.recent_destabilizer_distance,
            "declared_contracts": list(self.declared_contracts),
            "contract_postconditions_checked": (
                self.contract_postconditions_checked
            ),
            "contract_satisfied": self.contract_satisfied,
            "trajectory_telemetry_present": self.trajectory_telemetry_present,
            "u6_checked": self.u6_checked,
            "scope": self.scope,
        }


def observe_grammar(
    graph: Any,
    node: Any,
    sequence: list[Any],
    *,
    contract_satisfied: bool | None = None,
    u6_checked: bool = False,
) -> GrammarObservation:
    """Observe grammar and history state without mutating the graph."""
    data = graph.nodes[node]
    history = data.get("glyph_history") or ()
    history_length = len(history)
    try:
        result = validate_grammar(
            sequence, epi_initial=float(data.get("EPI", 0.0))
        )
        sequence_message = "valid" if result else "grammar validation failed"
    except (AttributeError, TypeError, ValueError) as exc:
        result = False
        sequence_message = f"grammar validation unavailable: {exc}"

    incremental = validate_sequence_incremental(graph, node, sequence)
    allowed = tuple(item.allowed for item in incremental)

    def operator_name(operator: Any) -> str:
        name = glyph_function_name(operator)
        if name:
            return name.lower()
        return str(getattr(operator, "name", operator)).lower()

    phase_requested = any(
        operator_name(operator) in {"coupling", "resonance", "um", "ra"}
        for operator in sequence
    )
    phase_allowed = (
        all(item.allowed for item in incremental)
        if phase_requested else None
    )
    depths = [
        int(getattr(operator, "depth", 1))
        for operator in sequence
        if operator_name(operator) in {"recursivity", "remesh"}
    ]
    accepted_history = tuple(operator_name(item) for item in history)
    recent_name = None
    recent_distance = None
    for distance, item in enumerate(reversed(accepted_history), start=1):
        if distance > BIFURCATION_WINDOW:
            break
        if item in DESTABILIZERS:
            recent_name = item
            recent_distance = distance
            break
    contracts = []
    for operator in sequence:
        try:
            contract = contract_for(operator_name(operator))
            contracts.append(contract.english_name)
        except KeyError:
            continue
    telemetry_present = bool(graph.graph.get("operator_metrics"))
    postconditions_checked = contract_satisfied is not None
    return GrammarObservation(
        sequence_valid=bool(result),
        sequence_message=sequence_message,
        accepted_history=accepted_history,
        history_length=history_length,
        u2_debt=node_debt(data),
        prior_coherence=node_has_prior_coherence(data),
        incremental_allowed=allowed,
        phase_gate_requested=phase_requested,
        phase_preconditions_checked=phase_requested,
        phase_gate_allowed=phase_allowed,
        max_recursivity_depth=max(depths, default=0),
        recent_destabilizer=recent_name,
        recent_destabilizer_distance=recent_distance,
        declared_contracts=tuple(contracts),
        contract_postconditions_checked=postconditions_checked,
        contract_satisfied=contract_satisfied,
        trajectory_telemetry_present=telemetry_present,
        u6_checked=bool(u6_checked),
    )
