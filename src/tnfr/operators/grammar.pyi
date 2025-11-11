from __future__ import annotations

from ..types import Glyph, NodeId, TNFRGraph
from ..node import NodeProtocol
from ..validation import ValidationOutcome
from _typeshed import Incomplete
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

__all__ = [
    "GrammarContext",
    "GrammarConfigurationError",
    "StructuralGrammarError",
    "RepeatWindowError",
    "MutationPreconditionError",
    "TholClosureError",
    "TransitionCompatibilityError",
    "SequenceSyntaxError",
    "SequenceValidationResult",
    "record_grammar_violation",
    "_gram_state",
    "apply_glyph_with_grammar",
    "enforce_canonical_grammar",
    "on_applied_glyph",
    "parse_sequence",
    "validate_sequence",
    "FUNCTION_TO_GLYPH",
    "GLYPH_TO_FUNCTION",
    "glyph_function_name",
    "function_name_to_glyph",
]

GLYPH_TO_FUNCTION: dict[Glyph, str]
FUNCTION_TO_GLYPH: dict[str, Glyph]

def glyph_function_name(val: Glyph | str | None, *, default: str | None = None) -> str | None: ...
def function_name_to_glyph(
    val: str | Glyph | None, *, default: Glyph | None = None
) -> Glyph | None: ...
@dataclass
class GrammarContext:
    G: TNFRGraph
    cfg_soft: dict[str, Any]
    cfg_canon: dict[str, Any]
    norms: dict[str, Any]
    def __post_init__(self) -> None: ...
    @classmethod
    def from_graph(cls, G: TNFRGraph) -> GrammarContext: ...

def _gram_state(nd: dict[str, Any]) -> dict[str, Any]: ...

class GrammarConfigurationError(ValueError):
    section: Incomplete
    messages: Incomplete
    details: Incomplete
    def __init__(
        self,
        section: str,
        messages: Sequence[str],
        *,
        details: Sequence[tuple[str, str]] | None = None,
    ) -> None: ...

class SequenceSyntaxError(ValueError):
    index: Incomplete
    token: Incomplete
    message: Incomplete
    def __init__(self, index: int, token: object, message: str) -> None: ...

class SequenceValidationResult(ValidationOutcome[tuple[str, ...]]):
    tokens: Incomplete
    canonical_tokens: Incomplete
    message: Incomplete
    metadata: Incomplete
    error: Incomplete
    def __init__(
        self,
        *,
        tokens: Sequence[str],
        canonical_tokens: Sequence[str],
        passed: bool,
        message: str,
        metadata: Mapping[str, object],
        error: SequenceSyntaxError | None = None,
    ) -> None: ...

class _SequenceAutomaton:
    def __init__(self) -> None: ...
    def run(self, names: Sequence[str]) -> None: ...
    @property
    def canonical(self) -> tuple[str, ...]: ...
    def metadata(self) -> Mapping[str, object]: ...

class StructuralGrammarError(RuntimeError):
    rule: Incomplete
    candidate: Incomplete
    message: Incomplete
    window: Incomplete
    threshold: Incomplete
    order: Incomplete
    context: dict[str, object]
    def __init__(
        self,
        *,
        rule: str,
        candidate: str,
        message: str,
        window: int | None = None,
        threshold: float | None = None,
        order: Sequence[str] | None = None,
        context: Mapping[str, object] | None = None,
    ) -> None: ...
    def attach_context(self, **context: object) -> StructuralGrammarError: ...
    def to_payload(self) -> dict[str, object]: ...

class RepeatWindowError(StructuralGrammarError): ...
class MutationPreconditionError(StructuralGrammarError): ...
class TholClosureError(StructuralGrammarError): ...
class TransitionCompatibilityError(StructuralGrammarError): ...

def record_grammar_violation(
    G: TNFRGraph, node: NodeId, error: StructuralGrammarError, *, stage: str
) -> None: ...
def validate_sequence(
    names: Iterable[str] | object = ..., **kwargs: object
) -> ValidationOutcome[tuple[str, ...]]: ...
def parse_sequence(names: Iterable[str]) -> SequenceValidationResult: ...
def enforce_canonical_grammar(
    G: TNFRGraph, n: NodeId, cand: Glyph | str, ctx: GrammarContext | None = None
) -> Glyph | str: ...
def on_applied_glyph(G: TNFRGraph, n: NodeId, applied: Glyph | str) -> None: ...
def apply_glyph_with_grammar(
    G: TNFRGraph,
    nodes: Iterable[NodeId | NodeProtocol] | None,
    glyph: Glyph | str,
    window: int | None = None,
) -> None: ...
