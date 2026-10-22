"""Read-only grammar/history observation reports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .event_runtime import OperatorEventExecutionResult
    from .grammar_evidence import StructuralGrammarEvidence

from ..alias import get_attr
from ..config.operator_names import BIFURCATION_WINDOW
from ..constants.aliases import ALIAS_EPI
from ._epi_domain import require_real_scalar_epi
from .grammar_debt import node_debt, node_has_prior_coherence
from .grammar_dynamics import validate_sequence_incremental
from .grammar_memoization import validate_sequence_optimized
from .grammar_types import DESTABILIZERS, glyph_function_name
from .operator_contracts import contract_for

__all__ = ["GrammarObservation", "observe_grammar"]


@dataclass(frozen=True)
class GrammarObservation:
    """Separate grammar, history, contract and telemetry facts.

    Phase fields describe only the requested UM/RA checks on the current
    graph. Isolated nodes remain unassessed by incremental selection, with
    ``phase_preconditions_checked=False`` and ``phase_gate_allowed=None``.
    An allowed phase gate does not imply that other grammar rules passed,
    later operator effects preserve compatibility, or a merged stage is valid.
    """

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
    structural_evidence: StructuralGrammarEvidence | None = None

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
            "contract_postconditions_checked": (self.contract_postconditions_checked),
            "contract_satisfied": self.contract_satisfied,
            "trajectory_telemetry_present": self.trajectory_telemetry_present,
            "u6_checked": self.u6_checked,
            "scope": self.scope,
            "word_validation_scope": "canonical_word_policy",
            "contract_evidence_source": (
                "caller_declaration"
                if self.contract_postconditions_checked
                else "unavailable"
            ),
            "u6_evidence_source": (
                "caller_declaration" if self.u6_checked else "unavailable"
            ),
            "verified_contract_postconditions": False,
            "verified_u6": False,
            "structural_evidence": (
                self.structural_evidence.as_dict()
                if self.structural_evidence is not None
                else None
            ),
        }


def observe_grammar(
    graph: Any,
    node: Any,
    sequence: list[Any],
    *,
    contract_satisfied: bool | None = None,
    u6_checked: bool = False,
    execution_evidence: OperatorEventExecutionResult | None = None,
) -> GrammarObservation:
    """Observe word/history policy separately from typed structural evidence.

    Legacy contract/U6 flags are caller declarations, not verified receipts.
    Optional sealed execution evidence concerns its own finite represented EPI
    maps. It neither authenticates the supplied current graph snapshot nor
    authorizes a future operator, and it cannot waive live grammar checks.
    """
    data = graph.nodes[node]
    history = data.get("glyph_history") or ()
    history_length = len(history)
    try:
        epi_value = get_attr(
            data,
            ALIAS_EPI,
            0.0,
            strict=True,
            conv=lambda value: value,
        )
        result, messages = validate_sequence_optimized(
            sequence,
            epi_initial=require_real_scalar_epi(
                epi_value,
                operator="Grammar observation",
                label="initial EPI",
            ),
        )
        sequence_message = "valid" if result else "; ".join(messages)
    except (AttributeError, TypeError, ValueError) as exc:
        result = False
        sequence_message = f"grammar validation unavailable: {exc}"

    def operator_name(operator: Any) -> str:
        name = glyph_function_name(operator)
        if name:
            return name.lower()
        return str(getattr(operator, "name", operator)).lower()

    # The incremental API accepts glyph/name tokens, while this observation
    # also accepts public operator instances for its batch/contract readouts.
    incremental = validate_sequence_incremental(
        graph, node, [operator_name(operator) for operator in sequence]
    )
    allowed = tuple(item.allowed for item in incremental)

    # Every known UM/RA candidate reaches the shared incremental U3 check,
    # including candidates rejected by another grammar rule. That owner leaves
    # isolates unassessed rather than granting concrete execution permission.
    phase_results = tuple(
        item for item in incremental if item.candidate in {"UM", "RA"}
    )
    phase_requested = bool(phase_results)
    phase_checked = phase_requested and bool(tuple(graph.neighbors(node)))
    phase_allowed = (
        all(
            not any(violation.rule == "U3" for violation in item.violations)
            for item in phase_results
        )
        if phase_checked
        else None
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
    structural_evidence = None
    if execution_evidence is not None:
        from .grammar_evidence import assess_structural_grammar_evidence

        structural_evidence = assess_structural_grammar_evidence(execution_evidence)
    return GrammarObservation(
        sequence_valid=bool(result),
        sequence_message=sequence_message,
        accepted_history=accepted_history,
        history_length=history_length,
        u2_debt=node_debt(data),
        prior_coherence=node_has_prior_coherence(data),
        incremental_allowed=allowed,
        phase_gate_requested=phase_requested,
        phase_preconditions_checked=phase_checked,
        phase_gate_allowed=phase_allowed,
        max_recursivity_depth=max(depths, default=0),
        recent_destabilizer=recent_name,
        recent_destabilizer_distance=recent_distance,
        declared_contracts=tuple(contracts),
        contract_postconditions_checked=postconditions_checked,
        contract_satisfied=contract_satisfied,
        trajectory_telemetry_present=telemetry_present,
        u6_checked=bool(u6_checked),
        structural_evidence=structural_evidence,
    )
