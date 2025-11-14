"""Grammar Error Factory (Phase 3).

Provides structured, introspection-enriched grammar errors referencing
canonical rules (U1-U4 primary, U6 confinement read-only) and TNFR
invariants. Reuses existing :class:`StructuralGrammarError` base from
``grammar_types`` to avoid duplication.

Why a Factory?
--------------
Existing validation returns (bool, message) pairs. Downstream tooling
needs richer payloads tying violations to:
 - Rule identifier (U1a, U1b, U2, U3, U4a, U4b, U6)
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

Invariants Mapping (Minimal)
----------------------------
U1a -> (1,4)        # EPI initiation & operator closure precondition
U1b -> (4)          # Closure / bounded sequence end
U2  -> (3,4)        # ΔNFR semantics & closure (stabilizer presence)
U3  -> (5)          # Phase verification
U4a -> (3,4,5)      # Trigger handling (ΔNFR pressure + handlers + phase)
U4b -> (3,4,7)      # Transformers need stabilised base & fractality preserved
U6  -> (3,9)        # Potential confinement + metrics integrity

NOTE: Mapping kept intentionally lean; can be extended in future without
breaking existing consumers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Sequence

from .definitions import get_operator_meta
from .grammar_core import GrammarValidator
from .grammar_types import StructuralGrammarError

__all__ = [
    "ExtendedGrammarError",
    "collect_grammar_errors",
    "make_grammar_error",
]


_RULE_INVARIANTS = {
    "U1a": (1, 4),
    "U1b": (4,),
    "U2": (3, 4),
    "U3": (5,),
    "U4a": (3, 4, 5),
    "U4b": (3, 4, 7),
    "U6_CONFINEMENT": (3, 9),
}


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
    invariants = _RULE_INVARIANTS.get(rule, ())
    op_meta: dict[str, Any] | None = None
    try:
        meta = get_operator_meta(candidate)
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
) -> List[ExtendedGrammarError]:
    """Run canonical validations and build structured error list.

    Only U1-U4 are active fail conditions; U6 confinement would attach
    separately when integrated with telemetry (read-only safety check).
    """
    validator = GrammarValidator()
    errors: List[ExtendedGrammarError] = []

    # Accept glyph strings by wrapping them in lightweight stubs
    # expected by GrammarValidator (which accesses .name / .canonical_name).
    GLYPH_TO_NAME = {
        "AL": "emission",
        "EN": "reception",
        "IL": "coherence",
        "OZ": "dissonance",
        "UM": "coupling",
        "RA": "resonance",
        "SHA": "silence",
        "VAL": "expansion",
        "NUL": "contraction",
        "THOL": "self_organization",
        "ZHIR": "mutation",
        "NAV": "transition",
        "REMESH": "recursivity",
    }
    
    class _OpStub:  # local minimal stub
        def __init__(self, glyph: str):
            canonical = GLYPH_TO_NAME.get(glyph.upper(), glyph.lower())
            self.canonical_name = canonical
            self.name = canonical

    normalized: List[Any] = [
        (_OpStub(op) if isinstance(op, str) else op) for op in sequence
    ]

    # Canonical operator names for reporting
    canonical = [
        getattr(op, "canonical_name", getattr(op, "name", "?"))
        for op in normalized
    ]

    # U1a
    ok, msg = validator.validate_initiation(list(normalized), epi_initial)
    if not ok:
        errors.append(
            make_grammar_error(
                rule="U1a",
                candidate=canonical[0] if canonical else "sequence",
                message=msg,
                sequence=canonical,
                index=0 if canonical else None,
            )
        )
    # U1b
    ok, msg = validator.validate_closure(list(normalized))
    if not ok:
        errors.append(
            make_grammar_error(
                rule="U1b",
                candidate=canonical[-1] if canonical else "sequence",
                message=msg,
                sequence=canonical,
                index=(len(canonical) - 1) if canonical else None,
            )
        )
    # U2
    ok, msg = validator.validate_convergence(list(normalized))
    if not ok:
        errors.append(
            make_grammar_error(
                rule="U2",
                candidate="sequence",
                message=msg,
                sequence=canonical,
            )
        )
    # U3
    ok, msg = validator.validate_resonant_coupling(list(normalized))
    if not ok:
        # Find first coupling/resonance candidate if available
        idx = next(
            (
                i
                for i, c in enumerate(canonical)
                if c in {"coupling", "resonance"}
            ),
            None,
        )
        cand = canonical[idx] if idx is not None else "sequence"
        errors.append(
            make_grammar_error(
                rule="U3",
                candidate=cand,
                message=msg,
                sequence=canonical,
                index=idx,
            )
        )
    return errors
