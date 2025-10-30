from typing import Any, Generic, Mapping, Protocol, TypeVar

SubjectT = TypeVar("SubjectT")


class ValidationOutcome(Generic[SubjectT]):
    subject: SubjectT
    passed: bool
    summary: Mapping[str, Any]
    artifacts: Mapping[str, Any] | None


class Validator(Protocol[SubjectT]):
    def validate(self, subject: SubjectT, /, **kwargs: Any) -> ValidationOutcome[SubjectT]: ...

    def report(self, summary: Mapping[str, Any]) -> str: ...


CANON_COMPAT: Any
CANON_FALLBACK: Any
GrammarContext: Any
GraphCanonicalValidator: Any
NFRValidator: Any
apply_canonical_clamps: Any
apply_glyph_with_grammar: Any
coerce_glyph: Any
enforce_canonical_grammar: Any
get_norm: Any
glyph_fallback: Any
normalized_dnfr: Any
on_applied_glyph: Any
run_validators: Any
validate_canon: Any
validate_sequence: Any
validate_window: Any
GRAPH_VALIDATORS: Any

__all__: tuple[str, ...]

def __getattr__(name: str) -> Any: ...

