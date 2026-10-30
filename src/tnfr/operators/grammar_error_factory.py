"""Structured projection of the shared canonical word validator.

Provides structured, introspection-enriched grammar errors referencing
syntax admission, canonical word rules U1-U5 and TNFR
invariants. Reuses existing :class:`StructuralGrammarError` base from
``grammar_types`` to avoid duplication.

Why a Factory?
--------------
Existing validation returns (bool, message) pairs. Downstream tooling
needs richer payloads tying violations to:
 - Rule identifier (SYNTAX, U1a, U1b, U2, U3, U4a, U4b, U2-REMESH, U5)
 - Related canonical invariants (AGENTS.md § Canonical Invariants)
 - Operator metadata (category, contracts, grammar roles)
 - Sequence context (window slice, involved operators)

The factory assembles this without modifying core validator logic,
preserving backward compatibility.

Public API
----------
collect_grammar_errors(sequence, epi_initial=0.0) -> list[ExtendedGrammarError]
make_grammar_error(rule, candidate, message, sequence, index=None)
    -> ExtendedGrammarError

Invariants Mapping (canonical, derived)
---------------------------------------
Each grammar-rule violation relates to its primary physics invariant plus
Grammar Compliance (#4). The mapping is DERIVED from
``grammar_canon.GRAMMAR_RULES`` (the single source of truth) via
``related_invariants``, reconciled to the 6-invariant canon (AGENTS.md
§Canonical Invariants):

U1a/U1b/U2 -> (1, 4)   # Nodal Equation Integrity + Grammar Compliance
U3         -> (2, 4)   # Phase-Coherent Coupling + Grammar Compliance
U4a/U4b    -> (4,)     # Grammar Compliance (bifurcation dynamics)
U5         -> (3, 4)   # Multi-Scale Fractality + Grammar Compliance
U6         -> (4, 5)   # Grammar Compliance + Structural Metrology

NOTE: Sourced from grammar_canon so the annotation cannot drift from the
canonical rule registry; a consistency test pins the agreement.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Sequence

from .grammar_canon import GRAMMAR_COMPLIANCE_INVARIANT
from .grammar_canon import GRAMMAR_RULES as _GRAMMAR_RULES
from .grammar_canon import related_invariants as _related_invariants
from .grammar_core import GrammarValidator
from .grammar_types import StructuralGrammarError, _operator_name, glyph_function_name

__all__ = [
    "ExtendedGrammarError",
    "collect_grammar_errors",
    "make_grammar_error",
]

# U-rule violation → canonical invariants it relates to (its primary physics
# invariant + Grammar Compliance #4). DERIVED from grammar_canon.GRAMMAR_RULES,
# the single source of truth, reconciled to the 6-invariant canon (AGENTS.md
# §Canonical Invariants). This replaces the stale pre-optimization 10-invariant
# numbering (which referenced invariants 7/9 that no longer exist). The
# "U6_CONFINEMENT" key aliases the canonical "U6" telemetry rule.
_RULE_INVARIANTS: dict[str, tuple[int, ...]] = {
    r.rule_id: _related_invariants(r.rule_id) for r in _GRAMMAR_RULES
}
_RULE_INVARIANTS["U6_CONFINEMENT"] = _related_invariants("U6")
_RULE_INVARIANTS["U2-REMESH"] = _related_invariants("U2")
_RULE_INVARIANTS["SYNTAX"] = (GRAMMAR_COMPLIANCE_INVARIANT,)


@dataclass(slots=True)
class ExtendedGrammarError:
    """Structured grammar error with invariant & operator metadata.

    Attributes
    ----------
    rule : str
        Grammar rule identifier (U1a, U2, ...)
    candidate : str
        Operator mnemonic or 'sequence'
    message : str
        Human-readable description
    invariants : tuple[int, ...]
        Canonical invariant IDs related to violation
    operator_meta : dict[str, Any] | None
        Introspection metadata if candidate resolves to operator
    order : tuple[str, ...]
        Canonical sequence slice (may be full sequence)
    index : int | None
        Index in sequence of offending operator (if applicable)
    """

    rule: str
    candidate: str
    message: str
    invariants: tuple[int, ...]
    operator_meta: dict[str, Any] | None
    order: tuple[str, ...]
    index: int | None = None

    def to_payload(self) -> dict[str, Any]:  # noqa: D401
        return {
            "rule": self.rule,
            "candidate": self.candidate,
            "message": self.message,
            "invariants": self.invariants,
            "operator_meta": self.operator_meta,
            "order": self.order,
            "index": self.index,
        }

    def to_structural_error(self) -> StructuralGrammarError:
        """Convert to existing StructuralGrammarError for compatibility."""
        return StructuralGrammarError(
            rule=self.rule,
            candidate=self.candidate,
            message=self.message,
            order=list(self.order),
            context={
                "invariants": self.invariants,
                "operator_meta": self.operator_meta,
                "index": self.index,
            },
        )


def make_grammar_error(
    *,
    rule: str,
    candidate: str,
    message: str,
    sequence: Sequence[str],
    index: int | None = None,
) -> ExtendedGrammarError:
    """Create an ExtendedGrammarError with invariants + introspection."""
    # Lazy imports avoid building contract/metadata views during facade imports.
    from .introspection import get_operator_meta
    from .operator_contracts import contract_for

    invariants = _RULE_INVARIANTS.get(rule, ())
    op_meta: dict[str, Any] | None = None
    try:
        meta = get_operator_meta(contract_for(candidate).glyph)
    except KeyError:
        meta = None
    if meta is not None:
        op_meta = {
            "name": meta.name,
            "mnemonic": meta.mnemonic,
            "category": meta.category,
            "grammar_roles": meta.grammar_roles,
            "contracts": meta.contracts,
        }
    return ExtendedGrammarError(
        rule=rule,
        candidate=candidate,
        message=message,
        invariants=invariants,
        operator_meta=op_meta,
        order=tuple(sequence),
        index=index,
    )


def collect_grammar_errors(
    sequence: Sequence[Any],
    epi_initial: float = 0.0,
) -> list[ExtendedGrammarError]:
    """Project all blocking outcomes of ``GrammarValidator.validate_checks``.

    Strings use shared name/glyph admission. Actual operator instances remain
    intact, including declared Recursivity depth. Unknown identifiers produce
    SYNTAX errors; this reader neither executes live U3 gates nor observes U6.
    """
    from .operator_contracts import contract_for

    normalized = []
    for operator in sequence:
        if isinstance(operator, str):
            try:
                name = contract_for(operator).name
            except KeyError:
                name = glyph_function_name(operator)
            normalized.append(SimpleNamespace(canonical_name=name))
        else:
            normalized.append(operator)

    checks = GrammarValidator().validate_checks(normalized, epi_initial)
    names = [_operator_name(operator) for operator in normalized]
    canonical = [name if isinstance(name, str) else repr(name) for name in names]
    errors: list[ExtendedGrammarError] = []
    for check in checks:
        if check.passed or not check.blocking:
            continue
        # Position extraction enriches a failed outcome; prose never decides
        # whether the check passed. Sequence-wide messages retain index=None.
        position = re.search(r"at position (\d+)", check.message)
        index = int(position.group(1)) if position else None
        if index is None and canonical:
            if check.rule == "U1a":
                index = 0
            elif check.rule == "U1b":
                index = len(canonical) - 1
            elif check.rule == "U3":
                index = next(
                    (
                        i
                        for i, name in enumerate(canonical)
                        if name in {"coupling", "resonance"}
                    ),
                    None,
                )
        if index is not None and not 0 <= index < len(canonical):
            index = None
        errors.append(
            make_grammar_error(
                rule=check.rule,
                candidate=canonical[index] if index is not None else "sequence",
                message=check.message,
                sequence=canonical,
                index=index,
            )
        )
    return errors
